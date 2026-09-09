#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robotfm.data.lerobot_dataset import (
    DEPTH_FEATURE_PREFIX,
    IMAGE_FEATURE_PREFIX,
    _format_data_path,
    is_lerobot_image_sequence_root,
    list_episode_indices,
    load_lerobot_info,
)
from robotfm.data.stats import compute_stats, save_stats

SRC_FPS = 30
DST_FPS = 15
ACTION_FROM_OBS = {
    "action": "observation.state",
    "action.gripper": "observation.gripper",
    "action.left_gripper": "observation.left_gripper",
    "action.right_gripper": "observation.right_gripper",
}


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[1] / path).resolve()


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
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _cell_path(cell: Any) -> str:
    if isinstance(cell, dict):
        return str(cell["path"])
    return str(cell[0])


def _scalar_stats(arr: np.ndarray) -> dict[str, Any]:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        return {
            "min": float(a.min()) if a.size else 0.0,
            "max": float(a.max()) if a.size else 0.0,
            "mean": float(a.mean()) if a.size else 0.0,
            "std": float(a.std()) if a.size else 0.0,
            "count": int(a.size),
        }
    return {
        "min": a.min(axis=0).tolist(),
        "max": a.max(axis=0).tolist(),
        "mean": a.mean(axis=0).tolist(),
        "std": a.std(axis=0).tolist(),
        "count": int(a.shape[0]),
    }


def _feature_keys(info: dict[str, Any], prefix: str) -> list[str]:
    return [k for k in info["features"] if k.startswith(prefix)]


def _media_rel(
    kind: str, feat_key: str, episode_index: int, frame_index: int, chunks_size: int
) -> str:
    chunk = episode_index // max(chunks_size, 1)
    ext = "jpg" if kind == "images" else "png"
    return (
        f"{kind}/chunk-{chunk:03d}/{feat_key}/"
        f"episode_{episode_index:06d}/frame_{frame_index:06d}.{ext}"
    )


def _shift_next(values: list[Any]) -> list[Any]:
    if len(values) <= 1:
        return list(values)
    return list(values[1:]) + [values[-1]]


def _grip_array(col: list[Any]) -> np.ndarray:
    vals: list[float] = []
    for v in col:
        if isinstance(v, (list, tuple)):
            vals.append(float(v[0]))
        else:
            vals.append(float(v))
    return np.asarray(vals, dtype=np.float64)


def write_split_episode(
    src_root: Path,
    dst_root: Path,
    info: dict[str, Any],
    src_ep: int,
    dst_ep: int,
    keep: np.ndarray,
    *,
    fps: float,
    global_index_start: int,
) -> tuple[int, dict[str, Any]]:
    chunks_size = int(info["chunks_size"])
    image_keys = _feature_keys(info, IMAGE_FEATURE_PREFIX)
    depth_keys = _feature_keys(info, DEPTH_FEATURE_PREFIX)
    data_rel_src = _format_data_path(info["data_path"], src_ep, chunks_size)
    table = pq.read_table(src_root / data_rel_src)
    data = table.to_pydict()
    t_src = table.num_rows
    if keep.size == 0:
        raise ValueError(f"ep{src_ep}: empty keep")
    if int(keep.max()) >= t_src:
        raise ValueError(f"ep{src_ep}: keep out of range")

    n = int(keep.shape[0])
    chunk = dst_ep // max(chunks_size, 1)
    for feat in image_keys:
        (
            dst_root
            / f"images/chunk-{chunk:03d}"
            / feat
            / f"episode_{dst_ep:06d}"
        ).mkdir(parents=True, exist_ok=True)
    for feat in depth_keys:
        (
            dst_root
            / f"depth/chunk-{chunk:03d}"
            / feat
            / f"episode_{dst_ep:06d}"
        ).mkdir(parents=True, exist_ok=True)

    out: dict[str, Any] = {
        "index": list(range(global_index_start, global_index_start + n)),
        "episode_index": [dst_ep] * n,
        "frame_index": list(range(n)),
        "timestamp": [i / fps for i in range(n)],
    }
    for field in table.schema:
        name = field.name
        if name in out:
            continue
        col = data[name]
        out[name] = [col[int(i)] for i in keep]

    for act_key, obs_key in ACTION_FROM_OBS.items():
        if act_key in out and obs_key in out:
            out[act_key] = _shift_next(out[obs_key])

    for feat in image_keys:
        col = data[feat]
        new_col = []
        for new_fi, src_fi in enumerate(keep):
            src_path = _cell_path(col[int(src_fi)])
            dst_rel = _media_rel("images", feat, dst_ep, new_fi, chunks_size)
            _link(src_root / src_path, dst_root / dst_rel)
            new_col.append({"path": dst_rel, "timestamp": new_fi / fps})
        out[feat] = new_col

    for feat in depth_keys:
        col = data[feat]
        new_col = []
        for new_fi, src_fi in enumerate(keep):
            src_item = col[int(src_fi)]
            if isinstance(src_item, dict):
                dst_cell = dict(src_item)
            else:
                dst_cell = {"path": src_item[0]}
            src_path = str(dst_cell["path"])
            dst_rel = _media_rel("depth", feat, dst_ep, new_fi, chunks_size)
            _link(src_root / src_path, dst_root / dst_rel)
            dst_cell["path"] = dst_rel
            dst_cell["timestamp"] = new_fi / fps
            new_col.append(dst_cell)
        out[feat] = new_col

    arrays = [pa.array(out[field.name], type=field.type) for field in table.schema]
    out_table = pa.Table.from_arrays(arrays, schema=table.schema)
    data_rel_dst = _format_data_path(info["data_path"], dst_ep, chunks_size)
    dst_pq = dst_root / data_rel_dst
    dst_pq.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, dst_pq)

    stats_entry: dict[str, Any] = {"episode_index": dst_ep, "stats": {}}
    stats_entry["stats"]["observation.state"] = _scalar_stats(
        np.asarray(out["observation.state"], dtype=np.float64)
    )
    stats_entry["stats"]["action"] = _scalar_stats(
        np.asarray(out["action"], dtype=np.float64)
    )
    for key in (
        "observation.gripper",
        "action.gripper",
        "observation.left_gripper",
        "action.left_gripper",
        "observation.right_gripper",
        "action.right_gripper",
    ):
        if key in out:
            stats_entry["stats"][key] = _scalar_stats(_grip_array(out[key]))
    return n, stats_entry


def run(src_root: Path, dst_root: Path, *, overwrite: bool, append: bool) -> None:
    if overwrite and append:
        raise SystemExit("pass only one of --overwrite / --append")
    if not is_lerobot_image_sequence_root(src_root):
        raise SystemExit(f"Not a leobot image_sequence dataset: {src_root}")
    info = load_lerobot_info(src_root)
    src_fps = int(info["fps"])
    if src_fps != SRC_FPS:
        raise SystemExit(f"src fps={src_fps}, expected {SRC_FPS}")
    if dst_root.resolve() == src_root.resolve():
        raise SystemExit("dst must differ from src")
    if append:
        if not dst_root.exists():
            raise SystemExit(f"Output missing: {dst_root} (needed for --append)")
        if not is_lerobot_image_sequence_root(dst_root):
            raise SystemExit(f"Not a leobot image_sequence dataset: {dst_root}")
    elif dst_root.exists():
        if not overwrite:
            raise SystemExit(f"Output exists: {dst_root} (pass --overwrite or --append)")
        shutil.rmtree(dst_root)
        dst_root.mkdir(parents=True)
    else:
        dst_root.mkdir(parents=True)

    episodes = list_episode_indices(src_root, info)
    ep_meta_src = {
        int(r["episode_index"]): r
        for r in _load_jsonl(src_root / "meta" / "episodes.jsonl")
    }
    image_keys = _feature_keys(info, IMAGE_FEATURE_PREFIX)
    depth_keys = _feature_keys(info, DEPTH_FEATURE_PREFIX)
    n_cams = len(image_keys)
    n_depth = len(depth_keys)

    meta_dst = dst_root / "meta"
    meta_dst.mkdir(parents=True, exist_ok=True)
    for name in ("depth_sources.json", "tasks.jsonl"):
        src_meta = src_root / "meta" / name
        if src_meta.is_file():
            shutil.copy2(src_meta, meta_dst / name)

    chunks_size = int(info["chunks_size"])
    existing_dst: set[int] = set()
    ep_jsonl_rows: list[dict[str, Any]] = []
    ep_stats_rows: list[dict[str, Any]] = []
    global_index = 0
    total_frames = 0
    if append:
        dst_info = load_lerobot_info(dst_root)
        existing_dst = set(list_episode_indices(dst_root, dst_info))
        ep_jsonl_rows = _load_jsonl(dst_root / "meta" / "episodes.jsonl")
        ep_stats_rows = _load_jsonl(dst_root / "meta" / "episodes_stats.jsonl")
        existing_dst |= {int(r["episode_index"]) for r in ep_jsonl_rows}
        global_index = int(dst_info["total_frames"])
        total_frames = global_index

    added = 0
    for src_ep in episodes:
        data_rel = _format_data_path(info["data_path"], src_ep, chunks_size)
        t_src = pq.read_table(src_root / data_rel, columns=["frame_index"]).num_rows
        phases = (
            (src_ep * 2, np.arange(0, t_src, 2, dtype=np.int64)),
            (src_ep * 2 + 1, np.arange(1, t_src, 2, dtype=np.int64)),
        )
        src_row = ep_meta_src.get(src_ep, {"episode_index": src_ep})
        for dst_ep, keep in phases:
            if keep.size == 0:
                continue
            if dst_ep in existing_dst:
                continue
            n, stats_entry = write_split_episode(
                src_root,
                dst_root,
                info,
                src_ep,
                dst_ep,
                keep,
                fps=float(DST_FPS),
                global_index_start=global_index,
            )
            global_index += n
            total_frames += n
            added += 1
            existing_dst.add(dst_ep)
            ep_stats_rows.append(stats_entry)
            ep_jsonl_rows.append(
                {
                    "episode_index": dst_ep,
                    "tasks": src_row.get("tasks", []),
                    "length": n,
                }
            )
            print(f"src {src_ep:03d} -> dst {dst_ep:03d}  {t_src}->{n}")

    if not ep_jsonl_rows:
        raise SystemExit("no episodes written")
    max_ep = max(int(r["episode_index"]) for r in ep_jsonl_rows)
    written = len(ep_jsonl_rows)
    out_info = dict(info)
    out_info["fps"] = DST_FPS
    out_info["total_episodes"] = written
    out_info["total_frames"] = total_frames
    out_info["total_images"] = total_frames * n_cams
    out_info["total_depth_images"] = total_frames * n_depth
    out_info["total_chunks"] = max_ep // max(chunks_size, 1) + 1
    out_info["splits"] = {"train": f"0:{max_ep + 1}"}
    out_info["phase_split"] = "even_odd"
    _write_json(meta_dst / "info.json", out_info)
    _write_jsonl(meta_dst / "episodes.jsonl", ep_jsonl_rows)
    _write_jsonl(meta_dst / "episodes_stats.jsonl", ep_stats_rows)

    stats = compute_stats(dst_root)
    src_stats = src_root / "stats.json"
    if src_stats.is_file():
        old = json.loads(src_stats.read_text())
        for k in ("image_mean", "image_std"):
            if k in old:
                stats[k] = np.asarray(old[k], dtype=np.float32)
    save_stats(dst_root, stats)
    print(
        f"wrote {dst_root} added={added} episodes={written} "
        f"frames={total_frames} fps={DST_FPS}"
    )


def main() -> None:
    va_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=va_root / "data" / "openarm_hcx_dual_arm_with_out_room_s",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=va_root / "data" / "openarm_hcx_dual_arm_with_out_room_s_15fps",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    run(
        _resolve(args.src),
        _resolve(args.dst),
        overwrite=args.overwrite,
        append=args.append,
    )


if __name__ == "__main__":
    main()
