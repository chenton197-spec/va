#!/usr/bin/env python3
"""Merge LeRobot image-sequence roots and carve a proportional val split.

Hardlinks parquet + per-episode image dirs (same filesystem) so the merge is
fast and space-efficient. Episode indices are preserved (sources must not
overlap). Val episodes are drawn from each source in proportion to episode
counts (Hamilton / largest-remainder), then written to a separate val root;
train root excludes those episodes from meta.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _format_data_path(template: str, episode_index: int, chunks_size: int) -> str:
    return template.format(
        episode_chunk=_episode_chunk(episode_index, chunks_size),
        episode_index=episode_index,
    )


def _camera_feature_keys(info: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for k, v in (info.get("features") or {}).items():
        if k.startswith("observation.images."):
            keys.append(k)
            continue
        if isinstance(v, dict) and v.get("dtype") == "image":
            keys.append(k)
    return sorted(keys)


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=False)
        else:
            shutil.copy2(src, dst)


def _hardlink_tree(src: Path, dst: Path) -> None:
    """Hardlink files under src into dst; create dirs as needed."""
    if not src.exists():
        return
    if src.is_file():
        _hardlink_or_copy(src, dst)
        return
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        out_dir = dst / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            _hardlink_or_copy(Path(root) / name, out_dir / name)


def _link_episode_assets(
    src_root: Path,
    dst_root: Path,
    info: dict[str, Any],
    episode_index: int,
    cam_keys: list[str],
) -> None:
    chunks_size = int(info["chunks_size"])
    data_rel = _format_data_path(info["data_path"], episode_index, chunks_size)
    src_pq = src_root / data_rel
    if not src_pq.is_file():
        raise FileNotFoundError(f"Missing parquet: {src_pq}")
    _hardlink_or_copy(src_pq, dst_root / data_rel)

    chunk = _episode_chunk(episode_index, chunks_size)
    ep_name = f"episode_{episode_index:06d}"
    for cam in cam_keys:
        src_img = src_root / "images" / f"chunk-{chunk:03d}" / cam / ep_name
        if not src_img.is_dir():
            raise FileNotFoundError(f"Missing images: {src_img}")
        dst_img = dst_root / "images" / f"chunk-{chunk:03d}" / cam / ep_name
        _hardlink_tree(src_img, dst_img)

    audit_src = src_root / "meta" / "recording_audit" / f"{ep_name}.jsonl"
    if audit_src.is_file():
        _hardlink_or_copy(
            audit_src,
            dst_root / "meta" / "recording_audit" / f"{ep_name}.jsonl",
        )


def _hamilton_allocate(counts: list[int], total_slots: int) -> list[int]:
    """Largest-remainder (Hamilton) apportionment."""
    if total_slots < 0:
        raise ValueError("total_slots must be >= 0")
    n = sum(counts)
    if n <= 0:
        raise ValueError("empty population")
    if total_slots > n:
        raise ValueError(f"Cannot sample {total_slots} from {n} episodes")
    raw = [c * total_slots / n for c in counts]
    base = [int(x) for x in raw]
    rem = total_slots - sum(base)
    order = sorted(range(len(counts)), key=lambda i: (raw[i] - base[i], counts[i]), reverse=True)
    for i in order[:rem]:
        base[i] += 1
    return base


def _sample_val(
    source_episodes: list[list[int]],
    n_val: int,
    seed: int,
) -> list[list[int]]:
    counts = [len(eps) for eps in source_episodes]
    quotas = _hamilton_allocate(counts, n_val)
    rng = random.Random(seed)
    picked: list[list[int]] = []
    for eps, k in zip(source_episodes, quotas):
        if k == 0:
            picked.append([])
            continue
        chosen = sorted(rng.sample(eps, k))
        picked.append(chosen)
    return picked


def _assert_compatible(infos: list[dict[str, Any]], roots: list[Path]) -> None:
    keys = (
        "codebase_version",
        "image_storage",
        "robot_type",
        "fps",
        "chunks_size",
        "data_path",
        "features",
    )
    base = infos[0]
    for info, root in zip(infos[1:], roots[1:]):
        for k in keys:
            if info.get(k) != base.get(k):
                raise ValueError(
                    f"Schema mismatch on {k!r} for {root}:\n"
                    f"  base={base.get(k)!r}\n  other={info.get(k)!r}"
                )


def _build_root(
    dst: Path,
    base_info: dict[str, Any],
    episodes: list[tuple[Path, dict[str, Any], dict[str, Any] | None]],
    *,
    split_name: str,
    cam_keys: list[str],
    tasks_rows: list[dict[str, Any]],
    extra_meta: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} exists; pass --overwrite")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    ep_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    total_frames = 0
    for src_root, ep_row, stats_row in episodes:
        ep_id = int(ep_row["episode_index"])
        _link_episode_assets(src_root, dst, base_info, ep_id, cam_keys)
        ep_rows.append(ep_row)
        if stats_row is not None:
            stats_rows.append(stats_row)
        total_frames += int(ep_row.get("length") or 0)

    ep_rows.sort(key=lambda r: int(r["episode_index"]))
    stats_rows.sort(key=lambda r: int(r["episode_index"]))
    indices = [int(r["episode_index"]) for r in ep_rows]
    max_ep = max(indices) if indices else -1

    out_info = dict(base_info)
    out_info["total_episodes"] = len(ep_rows)
    out_info["total_frames"] = total_frames
    out_info["total_chunks"] = (
        _episode_chunk(max_ep, int(base_info["chunks_size"])) + 1 if indices else 0
    )
    out_info["splits"] = {split_name: f"0:{max_ep + 1}" if indices else "0:0"}
    if extra_meta:
        out_info.update(extra_meta)

    _write_json(dst / "meta" / "info.json", out_info)
    _write_jsonl(dst / "meta" / "episodes.jsonl", ep_rows)
    _write_jsonl(dst / "meta" / "episodes_stats.jsonl", stats_rows)
    _write_jsonl(dst / "meta" / "tasks.jsonl", tasks_rows)

    try:
        from robotfm.data.stats import compute_lerobot_stats, save_stats

        save_stats(dst, compute_lerobot_stats(dst))
        print(f"  recomputed stats.json via robotfm for {dst}")
    except Exception as exc:  # noqa: BLE001
        print(f"  warning: robotfm stats recompute failed ({exc}); writing placeholder")
        _write_json(
            dst / "stats.json",
            {
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
                "note": f"placeholder; recompute failed: {exc}",
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="Source LeRobot image-sequence roots (episode indices must not overlap)",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        required=True,
        help="Output train dataset root",
    )
    parser.add_argument(
        "--val-out",
        type=Path,
        required=True,
        help="Output val dataset root",
    )
    parser.add_argument("--n-val", type=int, default=10, help="Total val episodes")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for val sampling")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    roots = [p.resolve() for p in args.sources]
    for r in roots:
        if not (r / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Not a LeRobot root: {r}")

    infos = [_load_json(r / "meta" / "info.json") for r in roots]
    _assert_compatible(infos, roots)
    base_info = infos[0]
    cam_keys = _camera_feature_keys(base_info)
    if not cam_keys:
        raise ValueError("No observation.images.* features found")

    source_eps: list[list[int]] = []
    ep_meta: dict[int, tuple[Path, dict[str, Any]]] = {}
    stats_meta: dict[int, dict[str, Any]] = {}
    for root in roots:
        rows = _load_jsonl(root / "meta" / "episodes.jsonl")
        srows = {
            int(r["episode_index"]): r
            for r in _load_jsonl(root / "meta" / "episodes_stats.jsonl")
        }
        eps: list[int] = []
        for row in rows:
            ep = int(row["episode_index"])
            if ep in ep_meta:
                raise ValueError(
                    f"Overlapping episode_index={ep}: {ep_meta[ep][0]} vs {root}"
                )
            pq = root / _format_data_path(
                base_info["data_path"], ep, int(base_info["chunks_size"])
            )
            if not pq.is_file():
                print(f"warning: skip missing parquet ep={ep} under {root}")
                continue
            ep_meta[ep] = (root, row)
            if ep in srows:
                stats_meta[ep] = srows[ep]
            eps.append(ep)
        source_eps.append(sorted(eps))

    quotas = _hamilton_allocate([len(e) for e in source_eps], args.n_val)
    val_by_source = _sample_val(source_eps, args.n_val, args.seed)
    val_set = sorted({ep for group in val_by_source for ep in group})
    train_set = sorted(set(ep_meta) - set(val_set))

    print("Sources:")
    for root, eps, q, picked in zip(roots, source_eps, quotas, val_by_source):
        print(f"  {root}: n={len(eps)} val_quota={q} val={picked}")
    print(f"Total train={len(train_set)} val={len(val_set)}")

    tasks = _load_jsonl(roots[0] / "meta" / "tasks.jsonl")
    if not tasks:
        tasks = [{"task_index": 0, "task": "openarm_hcx_dual_arm_teleoperation"}]

    split_record = {
        "created_at": datetime.now().astimezone().isoformat(),
        "seed": args.seed,
        "n_val": args.n_val,
        "sources": [str(r) for r in roots],
        "source_counts": [len(e) for e in source_eps],
        "val_quotas": quotas,
        "val_episodes_by_source": val_by_source,
        "val_episodes": val_set,
        "train_episodes": train_set,
    }

    train_eps = [
        (ep_meta[ep][0], ep_meta[ep][1], stats_meta.get(ep)) for ep in train_set
    ]
    val_eps = [(ep_meta[ep][0], ep_meta[ep][1], stats_meta.get(ep)) for ep in val_set]

    print(f"Building train -> {args.train_out}")
    _build_root(
        args.train_out.resolve(),
        base_info,
        train_eps,
        split_name="train",
        cam_keys=cam_keys,
        tasks_rows=tasks,
        extra_meta={
            "merge_split": {
                "role": "train",
                "seed": args.seed,
                "val_episodes": val_set,
                "sources": [str(r) for r in roots],
            }
        },
        overwrite=args.overwrite,
    )
    _write_json(args.train_out.resolve() / "meta" / "merge_split.json", split_record)

    print(f"Building val -> {args.val_out}")
    _build_root(
        args.val_out.resolve(),
        base_info,
        val_eps,
        split_name="val",
        cam_keys=cam_keys,
        tasks_rows=tasks,
        extra_meta={
            "merge_split": {
                "role": "val",
                "seed": args.seed,
                "val_episodes": val_set,
                "sources": [str(r) for r in roots],
            }
        },
        overwrite=args.overwrite,
    )
    _write_json(args.val_out.resolve() / "meta" / "merge_split.json", split_record)
    print("Done.")


if __name__ == "__main__":
    main()
