#!/usr/bin/env python3
"""Trim leading frames where the left arm has not yet moved.

Default: left-arm (joints 0:7) per-step L2 speed must stay >= ``--speed-thresh-deg``
for ``--enter-frames`` consecutive steps; keep from the first frame of that run.

Optional ``--mode displacement``: keep from the first frame whose L2 displacement
from t0 is >= ``--motion-thresh-deg``.

Optional ``--drop-n-last-frames N``: also drop the last N frames of each episode
(after the idle-prefix trim).

In-place mode rewrites parquet + meta, deletes unused leading/trailing JPEGs, and
recomputes ``stats.json``. Missing on-disk episodes listed in meta are dropped.

Examples (from ``va/``, conda env ``lerobot``)::

    PYTHONPATH=. python scripts/trim_left_arm_idle_prefix.py \\
        data/openarm_hcx_dual_arm --dry-run
    PYTHONPATH=. python scripts/trim_left_arm_idle_prefix.py \\
        data/openarm_hcx_dual_arm --drop-n-last-frames 30 --in-place
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robotfm.data.lerobot_dataset import (
    IMAGE_FEATURE_PREFIX,
    _format_data_path,
    is_lerobot_image_sequence_root,
    list_episode_indices,
    load_lerobot_info,
)
from robotfm.data.stats import compute_stats, save_stats


def _resolve_run_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    va_root = Path(__file__).resolve().parents[1]
    return (va_root / path).resolve()


def _camera_feature_keys(info: dict[str, Any]) -> list[str]:
    return [k for k in info["features"] if k.startswith(IMAGE_FEATURE_PREFIX)]


def _episodes_on_disk(run_dir: Path, info: dict[str, Any]) -> list[int]:
    """Episode indices that have a parquet file (intersection with meta list)."""
    meta_eps = list_episode_indices(run_dir, info)
    chunks_size = int(info["chunks_size"])
    out: list[int] = []
    for ep in meta_eps:
        rel = _format_data_path(info["data_path"], ep, chunks_size)
        if (run_dir / rel).is_file():
            out.append(ep)
    return out


def left_arm_motion_start(
    state: np.ndarray,
    *,
    mode: str = "speed",
    motion_thresh_deg: float = 1.0,
    speed_thresh_deg: float = 0.5,
    enter_frames: int = 5,
) -> int:
    """Return first keep-frame index after left-arm idle prefix.

    ``state`` is (T, >=7) in degrees; left arm = columns 0:7. Returns 0 if the
    arm never reaches the motion criterion (keep full episode).
    """
    if state.ndim != 2 or state.shape[1] < 7:
        raise ValueError(f"expected state (T, >=7), got {state.shape}")
    if state.shape[0] <= 1:
        return 0
    left = state[:, :7].astype(np.float64)
    if mode == "displacement":
        disp = np.linalg.norm(left - left[0], axis=1)
        hits = np.flatnonzero(disp >= float(motion_thresh_deg))
        return int(hits[0]) if hits.size else 0
    if mode != "speed":
        raise ValueError(f"unknown mode {mode!r}; use 'speed' or 'displacement'")
    if enter_frames <= 0:
        raise ValueError(f"enter_frames must be > 0, got {enter_frames}")
    speeds = np.linalg.norm(np.diff(left, axis=0), axis=1)
    run = 0
    for i, moving in enumerate(speeds >= float(speed_thresh_deg)):
        if moving:
            run += 1
            if run >= enter_frames:
                return int(i - enter_frames + 1)
        else:
            run = 0
    return 0


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


def _image_path_from_cell(cell: Any) -> str:
    if isinstance(cell, dict):
        return str(cell["path"])
    if isinstance(cell, (list, tuple)):
        return str(cell[0])
    return str(cell)


def trim_episode_table(
    table: pa.Table,
    drop: int,
    *,
    fps: float,
    global_index_start: int,
    drop_last: int = 0,
) -> tuple[pa.Table, list[str]]:
    """Slice rows [drop:T-drop_last], renumber index/frame_index/timestamp.

    Returns (new_table, relative image paths of dropped leading/trailing frames).
    """
    if drop < 0:
        raise ValueError(f"drop must be >= 0, got {drop}")
    if drop_last < 0:
        raise ValueError(f"drop_last must be >= 0, got {drop_last}")
    t = table.num_rows
    keep_end = t - drop_last
    if drop >= keep_end:
        raise ValueError(
            f"drop={drop} + drop_last={drop_last} leaves no frames (length={t})"
        )
    dropped_paths: list[str] = []
    if drop > 0:
        head = table.slice(0, drop).to_pydict()
        for key, col in head.items():
            if key.startswith(IMAGE_FEATURE_PREFIX):
                for cell in col:
                    dropped_paths.append(_image_path_from_cell(cell))
    if drop_last > 0:
        tail = table.slice(keep_end).to_pydict()
        for key, col in tail.items():
            if key.startswith(IMAGE_FEATURE_PREFIX):
                for cell in col:
                    dropped_paths.append(_image_path_from_cell(cell))

    if drop == 0 and drop_last == 0:
        data = table.to_pydict()
        n = t
    else:
        data = table.slice(drop, keep_end - drop).to_pydict()
        n = keep_end - drop

    data["index"] = list(range(global_index_start, global_index_start + n))
    data["frame_index"] = list(range(n))
    data["timestamp"] = [i / float(fps) for i in range(n)]
    # Refresh embedded timestamps inside image cells; keep path.
    for key, col in list(data.items()):
        if not key.startswith(IMAGE_FEATURE_PREFIX):
            continue
        new_col = []
        for new_fi, cell in enumerate(col):
            path = _image_path_from_cell(cell)
            new_col.append({"path": path, "timestamp": new_fi / float(fps)})
        data[key] = new_col

    arrays = [pa.array(data[field.name], type=field.type) for field in table.schema]
    return pa.Table.from_arrays(arrays, schema=table.schema), dropped_paths


def _episode_stats_entry(table: pa.Table, episode_index: int) -> dict[str, Any]:
    data = table.to_pydict()
    entry: dict[str, Any] = {"episode_index": episode_index, "stats": {}}
    state = np.asarray(data["observation.state"], dtype=np.float64)
    action = np.asarray(data["action"], dtype=np.float64)
    entry["stats"]["observation.state"] = _scalar_stats(state)
    entry["stats"]["action"] = _scalar_stats(action)
    for key in (
        "observation.gripper",
        "action.gripper",
        "observation.left_gripper",
        "action.left_gripper",
        "observation.right_gripper",
        "action.right_gripper",
    ):
        if key not in data:
            continue
        col = data[key]
        # gripper cells may be scalar or length-1 list
        vals = []
        for v in col:
            if isinstance(v, (list, tuple)):
                vals.append(float(v[0]))
            else:
                vals.append(float(v))
        entry["stats"][key] = _scalar_stats(np.asarray(vals, dtype=np.float64))
    return entry


def run(
    run_dir: Path,
    *,
    mode: str,
    motion_thresh_deg: float,
    speed_thresh_deg: float,
    enter_frames: int,
    drop_n_last_frames: int,
    dry_run: bool,
    in_place: bool,
    delete_images: bool,
) -> None:
    if not is_lerobot_image_sequence_root(run_dir):
        raise SystemExit(f"Not a leobot image_sequence dataset: {run_dir}")
    if not dry_run and not in_place:
        raise SystemExit("Pass --in-place to apply, or --dry-run to only report")
    if drop_n_last_frames < 0:
        raise SystemExit(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

    info = load_lerobot_info(run_dir)
    fps = float(info["fps"])
    chunks_size = int(info["chunks_size"])
    meta_eps = list_episode_indices(run_dir, info)
    eps = _episodes_on_disk(run_dir, info)
    missing = sorted(set(meta_eps) - set(eps))
    feat_keys = _camera_feature_keys(info)
    n_cams = len(feat_keys)

    ep_meta_src = {}
    ep_jsonl = run_dir / "meta" / "episodes.jsonl"
    if ep_jsonl.is_file():
        with ep_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ep_meta_src[int(row["episode_index"])] = row

    # ep, length, drop_prefix, drop_suffix
    plans: list[tuple[int, int, int, int]] = []
    for ep in eps:
        rel = _format_data_path(info["data_path"], ep, chunks_size)
        table = pq.read_table(run_dir / rel, columns=["observation.state"])
        state = np.stack(table.column("observation.state").to_pylist()).astype(np.float64)
        length = int(state.shape[0])
        drop = left_arm_motion_start(
            state,
            mode=mode,
            motion_thresh_deg=motion_thresh_deg,
            speed_thresh_deg=speed_thresh_deg,
            enter_frames=enter_frames,
        )
        drop_last = int(drop_n_last_frames)
        # Keep at least a few frames after both trims
        min_keep = 2
        if drop + drop_last >= length - min_keep + 1:
            # Prefer keeping motion start; shrink suffix first, then prefix.
            avail = max(length - min_keep, 0)
            drop_last = min(drop_last, max(avail - drop, 0))
            if drop + drop_last >= length - min_keep + 1:
                drop = max(0, length - min_keep - drop_last)
        if drop >= length - 1:
            drop = 0
        plans.append((ep, length, drop, drop_last))

    total_src = sum(L for _, L, _, _ in plans)
    total_drop_prefix = sum(d for _, _, d, _ in plans)
    total_drop_suffix = sum(s for _, _, _, s in plans)
    total_drop = total_drop_prefix + total_drop_suffix
    crit = (
        f"mode=displacement thresh={motion_thresh_deg}deg"
        if mode == "displacement"
        else f"mode=speed thresh={speed_thresh_deg}deg/step enter={enter_frames}"
    )
    print(
        f"src={run_dir}  eps_on_disk={len(eps)}/{len(meta_eps)}  "
        f"missing_parquet={missing}  {crit}  drop_last={drop_n_last_frames}  "
        f"{'[DRY-RUN]' if dry_run else '[IN-PLACE]'}"
    )
    for ep, length, drop, drop_last in plans:
        if drop == 0 and drop_last == 0:
            continue
        keep = length - drop - drop_last
        print(
            f"  ep{ep:03d}  prefix {drop:3d} + suffix {drop_last:3d} / {length:4d} "
            f"({(drop + drop_last) / fps:.1f}s)  keep={keep}"
        )
    n_with = sum(1 for _, _, d, s in plans if d > 0 or s > 0)
    print(
        f"summary: {n_with}/{len(plans)} eps trimmed  "
        f"prefix=-{total_drop_prefix} suffix=-{total_drop_suffix}  "
        f"frames {total_src}->{total_src - total_drop} (-{total_drop}, "
        f"{100.0 * total_drop / max(total_src, 1):.1f}%)"
    )
    if dry_run:
        return

    # Apply in place
    global_index = 0
    ep_jsonl_rows: list[dict[str, Any]] = []
    ep_stats_rows: list[dict[str, Any]] = []
    total_frames = 0
    deleted_images = 0

    for ep, length, drop, drop_last in plans:
        rel = _format_data_path(info["data_path"], ep, chunks_size)
        pq_path = run_dir / rel
        table = pq.read_table(pq_path)
        new_table, dropped_paths = trim_episode_table(
            table,
            drop,
            fps=fps,
            global_index_start=global_index,
            drop_last=drop_last,
        )
        # Atomic-ish rewrite via tmp
        tmp = pq_path.with_suffix(".parquet.tmp")
        pq.write_table(new_table, tmp)
        os.replace(tmp, pq_path)

        if delete_images and dropped_paths:
            for rel_img in dropped_paths:
                img = run_dir / rel_img
                if img.is_file():
                    img.unlink()
                    deleted_images += 1

        n = new_table.num_rows
        global_index += n
        total_frames += n
        src_row = ep_meta_src.get(ep, {"episode_index": ep, "tasks": []})
        ep_jsonl_rows.append(
            {
                "episode_index": ep,
                "tasks": src_row.get("tasks", []),
                "length": n,
            }
        )
        ep_stats_rows.append(_episode_stats_entry(new_table, ep))
        print(f"  wrote ep{ep:03d}: {length}->{n}")

    out_info = dict(info)
    out_info["total_episodes"] = len(plans)
    out_info["total_frames"] = total_frames
    out_info["total_images"] = total_frames * n_cams
    # Keep original episode index numbering; splits cover available indices.
    if plans:
        out_info["splits"] = {"train": f"0:{plans[-1][0] + 1}"}
    prev = info.get("trim_left_arm_idle_prefix")
    out_info["trim_left_arm_idle_prefix"] = {
        "mode": mode,
        "motion_thresh_deg": motion_thresh_deg,
        "speed_thresh_deg": speed_thresh_deg,
        "enter_frames": enter_frames,
        "drop_n_last_frames": drop_n_last_frames,
        "frames_dropped_this_pass": total_drop,
        "frames_dropped_prefix": total_drop_prefix,
        "frames_dropped_suffix": total_drop_suffix,
        "missing_parquet_removed_from_meta": missing,
        "episodes": [ep for ep, _, _, _ in plans],
        "previous": prev,
    }
    with (run_dir / "meta" / "info.json").open("w") as f:
        json.dump(out_info, f, indent=2)
        f.write("\n")

    with (run_dir / "meta" / "episodes.jsonl").open("w") as f:
        for row in ep_jsonl_rows:
            f.write(json.dumps(row) + "\n")

    with (run_dir / "meta" / "episodes_stats.jsonl").open("w") as f:
        for row in ep_stats_rows:
            f.write(json.dumps(row) + "\n")

    # Drop orphan recording_audit for missing eps if present
    audit_dir = run_dir / "meta" / "recording_audit"
    if audit_dir.is_dir() and missing:
        for ep in missing:
            p = audit_dir / f"episode_{ep:06d}.jsonl"
            if p.is_file():
                p.unlink()

    src_stats = run_dir / "stats.json"
    old_image = {}
    if src_stats.is_file():
        old = json.loads(src_stats.read_text())
        for k in ("image_mean", "image_std"):
            if k in old:
                old_image[k] = np.asarray(old[k], dtype=np.float32)

    stats = compute_stats(run_dir)
    stats.update(old_image)
    save_stats(run_dir, stats)

    print(
        f"done: episodes={len(plans)} frames {total_src}->{total_frames}  "
        f"deleted_jpegs={deleted_images}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=str, help="LeRobot image_sequence dataset root")
    p.add_argument(
        "--mode",
        choices=("speed", "displacement"),
        default="speed",
        help="Idle detection: sustained speed (default) or cumulative displacement",
    )
    p.add_argument(
        "--motion-thresh-deg",
        type=float,
        default=1.0,
        help="Displacement mode: L2 displacement from t0 (default 1.0)",
    )
    p.add_argument(
        "--speed-thresh-deg",
        type=float,
        default=0.5,
        help="Speed mode: per-step L2 deg threshold (default 0.5)",
    )
    p.add_argument(
        "--enter-frames",
        type=int,
        default=5,
        help="Speed mode: consecutive steps above threshold (default 5)",
    )
    p.add_argument(
        "--drop-n-last-frames",
        type=int,
        default=0,
        help="Also drop this many trailing frames per episode (default 0)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--in-place", action="store_true", help="Rewrite dataset in place")
    p.add_argument(
        "--keep-images",
        action="store_true",
        help="Do not delete trimmed leading/trailing JPEG files",
    )
    args = p.parse_args()
    run(
        _resolve_run_dir(Path(args.run_dir)),
        mode=args.mode,
        motion_thresh_deg=args.motion_thresh_deg,
        speed_thresh_deg=args.speed_thresh_deg,
        enter_frames=args.enter_frames,
        drop_n_last_frames=args.drop_n_last_frames,
        dry_run=args.dry_run,
        in_place=args.in_place,
        delete_images=not args.keep_images,
    )


if __name__ == "__main__":
    main()
