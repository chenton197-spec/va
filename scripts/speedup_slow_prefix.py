#!/usr/bin/env python3
"""Speed up slow episode prefixes via path-length frame subsampling.

Per episode, take mid-segment L2 joint speed median as ``v_target`` (state in
degrees). Detect the slow prefix, keep frames so retained prefix motion matches
``v_target`` at the dataset fps, write a new LeRobot image-sequence root.

Examples (from ``va/``, conda env ``lerobot``)::

    PYTHONPATH=. python scripts/speedup_slow_prefix.py shine_shoes_fr3_s256 --dry-run
    PYTHONPATH=. python scripts/speedup_slow_prefix.py shine_shoes_fr3_s256 \\
        --episodes 11 --output outputs/ep11_head_speedup --compare-video
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robotfm.data.lerobot_dataset import (
    IMAGE_FEATURE_PREFIX,
    _format_data_path,
    _load_image_rgb,
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


def _short_cam(feat_key: str) -> str:
    return feat_key[len(IMAGE_FEATURE_PREFIX) :]


def joint_l2_speeds(state: np.ndarray, fps: float) -> np.ndarray:
    """Per-step L2 joint speed (°/s). ``state`` is (T, D) in degrees."""
    if state.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    dq = np.diff(state.astype(np.float64), axis=0)
    return np.linalg.norm(dq, axis=1) * float(fps)


def mid_segment_speeds(
    speeds: np.ndarray,
    *,
    head_drop_frac: float,
    tail_drop_frac: float,
) -> np.ndarray:
    """Drop head/tail fractions of the speed series; return the middle slice."""
    n = int(speeds.shape[0])
    if n <= 0:
        return speeds
    lo = int(np.floor(head_drop_frac * n))
    hi = int(np.ceil((1.0 - tail_drop_frac) * n))
    hi = max(hi, lo + 1)
    hi = min(hi, n)
    mid = speeds[lo:hi]
    if mid.size == 0:
        return speeds
    return mid


def find_slow_prefix_end(
    speeds: np.ndarray,
    v_target: float,
    *,
    enter_ratio: float,
    enter_frames: int,
) -> int:
    """Return frame index ``t*`` where the slow prefix ends (suffix starts).

    Speeds have length T-1; speed[i] is the step from frame i -> i+1. Prefix end
    is the first index where ``enter_frames`` consecutive steps are all
    >= enter_ratio * v_target. Returns that start index as ``t*``. If never
    reached, return 0 (no prefix to speed up).
    """
    if speeds.size == 0 or v_target <= 0 or enter_frames <= 0:
        return 0
    thresh = float(enter_ratio) * float(v_target)
    run = 0
    for i, v in enumerate(speeds):
        if v >= thresh:
            run += 1
            if run >= enter_frames:
                return i - enter_frames + 1
        else:
            run = 0
    return 0


def path_length_keep_indices(
    state: np.ndarray,
    prefix_end: int,
    v_target: float,
    fps: float,
) -> np.ndarray:
    """Keep indices for frames [0, T): subsample [0, prefix_end], keep rest.

    ``prefix_end`` is ``t*`` (first full-speed frame). Always keep ``prefix_end``
    so the join is continuous.
    """
    t = int(state.shape[0])
    if t <= 0:
        return np.zeros(0, dtype=np.int64)
    if prefix_end <= 0:
        return np.arange(t, dtype=np.int64)

    prefix_end = min(prefix_end, t - 1)
    step = float(v_target) / max(float(fps), 1e-6)
    if step <= 0:
        return np.arange(t, dtype=np.int64)

    q = state.astype(np.float64)
    keep: list[int] = [0]
    acc = 0.0
    for i in range(1, prefix_end + 1):
        acc += float(np.linalg.norm(q[i] - q[i - 1]))
        if acc >= step or i == prefix_end:
            if keep[-1] != i:
                keep.append(i)
            acc = 0.0

    for i in range(prefix_end + 1, t):
        keep.append(i)
    return np.asarray(keep, dtype=np.int64)


@dataclass
class EpisodePlan:
    episode_index: int
    length_src: int
    length_dst: int
    v_target: float
    prefix_end: int
    prefix_src: int
    prefix_dst: int
    keep: np.ndarray
    prefix_l2_before: float
    prefix_l2_after: float


def plan_episode(
    state: np.ndarray,
    fps: float,
    episode_index: int,
    *,
    head_drop_frac: float,
    tail_drop_frac: float,
    enter_ratio: float,
    enter_frames: int,
) -> EpisodePlan:
    speeds = joint_l2_speeds(state, fps)
    mid = mid_segment_speeds(
        speeds, head_drop_frac=head_drop_frac, tail_drop_frac=tail_drop_frac
    )
    v_target = float(np.median(mid)) if mid.size else 0.0
    prefix_end = find_slow_prefix_end(
        speeds, v_target, enter_ratio=enter_ratio, enter_frames=enter_frames
    )
    keep = path_length_keep_indices(state, prefix_end, v_target, fps)

    if prefix_end > 0 and keep.size >= 2:
        pref_keep = keep[keep <= prefix_end]
        before = float(speeds[:prefix_end].mean()) if prefix_end > 0 else 0.0
        if pref_keep.size >= 2:
            dq = np.diff(state[pref_keep].astype(np.float64), axis=0)
            after = float(np.linalg.norm(dq, axis=1).mean() * fps)
        else:
            after = before
    else:
        before = 0.0
        after = 0.0

    return EpisodePlan(
        episode_index=episode_index,
        length_src=int(state.shape[0]),
        length_dst=int(keep.shape[0]),
        v_target=v_target,
        prefix_end=int(prefix_end),
        prefix_src=int(prefix_end + 1) if prefix_end > 0 else 0,
        prefix_dst=int(np.sum(keep <= prefix_end)) if prefix_end > 0 else 0,
        keep=keep,
        prefix_l2_before=before,
        prefix_l2_after=after,
    )


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if not src.is_file():
        raise FileNotFoundError(f"Missing source image: {src}")
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _frame_images_exist(
    src_root: Path,
    data: dict[str, Any],
    feat_keys: list[str],
    frame_index: int,
) -> bool:
    for feat in feat_keys:
        row = data[feat][frame_index]
        rel = row["path"] if isinstance(row, dict) else row[0]
        if not (src_root / rel).is_file():
            return False
    return True


def filter_keep_existing_images(
    src_root: Path,
    data: dict[str, Any],
    feat_keys: list[str],
    keep: np.ndarray,
) -> np.ndarray:
    """Drop keep indices whose camera JPEGs are missing on disk."""
    ok = [
        int(i)
        for i in keep
        if _frame_images_exist(src_root, data, feat_keys, int(i))
    ]
    if not ok:
        raise ValueError("No keep frames have all camera images on disk")
    # Always keep at least first and last surviving frames already in ok
    return np.asarray(ok, dtype=np.int64)


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


def _image_rel_path(
    feat_key: str, episode_index: int, frame_index: int, chunks_size: int
) -> str:
    chunk = episode_index // max(chunks_size, 1)
    return (
        f"images/chunk-{chunk:03d}/{feat_key}/"
        f"episode_{episode_index:06d}/frame_{frame_index:06d}.jpg"
    )


def write_episode(
    src_root: Path,
    dst_root: Path,
    info: dict[str, Any],
    episode_index: int,
    keep: np.ndarray,
    *,
    global_index_start: int,
) -> tuple[int, dict[str, Any]]:
    chunks_size = int(info["chunks_size"])
    fps = float(info["fps"])
    feat_keys = _camera_feature_keys(info)
    data_rel = _format_data_path(info["data_path"], episode_index, chunks_size)
    table = pq.read_table(src_root / data_rel)
    data = table.to_pydict()
    t_src = table.num_rows
    if keep.size == 0:
        raise ValueError(f"ep{episode_index}: empty keep set")
    if int(keep.max()) >= t_src:
        raise ValueError(f"ep{episode_index}: keep index out of range")

    n = int(keep.shape[0])
    out: dict[str, Any] = {}
    out["index"] = list(range(global_index_start, global_index_start + n))
    out["episode_index"] = [episode_index] * n
    out["frame_index"] = list(range(n))
    out["timestamp"] = [i / fps for i in range(n)]
    out["task_index"] = [data["task_index"][int(i)] for i in keep]

    for key in ("action", "observation.state", "action.gripper", "observation.gripper"):
        if key not in data:
            continue
        col = data[key]
        out[key] = [col[int(i)] for i in keep]

    for feat in feat_keys:
        col = data[feat]
        new_col = []
        for new_fi, src_fi in enumerate(keep):
            src_item = col[int(src_fi)]
            src_path = src_item["path"] if isinstance(src_item, dict) else src_item[0]
            dst_rel = _image_rel_path(feat, episode_index, new_fi, chunks_size)
            _hardlink_or_copy(src_root / src_path, dst_root / dst_rel)
            new_col.append({"path": dst_rel, "timestamp": new_fi / fps})
        out[feat] = new_col

    arrays = []
    for field in table.schema:
        arrays.append(pa.array(out[field.name], type=field.type))
    out_table = pa.Table.from_arrays(arrays, schema=table.schema)
    dst_pq = dst_root / data_rel
    dst_pq.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, dst_pq)

    stats_entry: dict[str, Any] = {"episode_index": episode_index, "stats": {}}
    state = np.asarray(out["observation.state"], dtype=np.float64)
    action = np.asarray(out["action"], dtype=np.float64)
    stats_entry["stats"]["observation.state"] = _scalar_stats(state)
    stats_entry["stats"]["action"] = _scalar_stats(action)
    if "observation.gripper" in out:
        stats_entry["stats"]["observation.gripper"] = _scalar_stats(
            np.asarray(out["observation.gripper"], dtype=np.float64)
        )
    if "action.gripper" in out:
        stats_entry["stats"]["action.gripper"] = _scalar_stats(
            np.asarray(out["action.gripper"], dtype=np.float64)
        )
    return n, stats_entry


def _load_episodes_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tile_bgr(frames: list[np.ndarray], labels: list[str]) -> np.ndarray:
    h = max(f.shape[0] for f in frames)
    parts = []
    for frame, label in zip(frames, labels):
        if frame.shape[0] != h:
            scale = h / frame.shape[0]
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), h))
        canvas = frame.copy()
        cv2.putText(
            canvas,
            label,
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        parts.append(canvas)
    return np.concatenate(parts, axis=1)


def write_compare_video(
    src_root: Path,
    dst_root: Path,
    info: dict[str, Any],
    episode_index: int,
    plan: EpisodePlan,
    out_path: Path,
) -> Path:
    """Side-by-side before|after video at dataset fps (after may end earlier)."""
    fps = float(info["fps"])
    feat_keys = _camera_feature_keys(info)
    chunks_size = int(info["chunks_size"])
    data_rel = _format_data_path(info["data_path"], episode_index, chunks_size)

    src_data = pq.read_table(src_root / data_rel).to_pydict()
    dst_data = pq.read_table(dst_root / data_rel).to_pydict()

    def paths_for(data: dict, feat: str) -> list[str]:
        return [
            row["path"] if isinstance(row, dict) else row[0] for row in data[feat]
        ]

    src_paths = {f: paths_for(src_data, f) for f in feat_keys}
    dst_paths = {f: paths_for(dst_data, f) for f in feat_keys}
    n_src = len(next(iter(src_paths.values())))
    n_dst = len(next(iter(dst_paths.values())))
    n = max(n_src, n_dst)

    sample = _load_image_rgb(src_root / src_paths[feat_keys[0]][0])
    probe = tile_bgr(
        [cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)] * len(feat_keys),
        [_short_cam(f) for f in feat_keys],
    )
    panel_w, panel_h = probe.shape[1], probe.shape[0]
    gap = 8
    header = 48
    out_w = panel_w * 2 + gap
    out_h = panel_h + header

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_path}")

    last_dst_panel = None
    for i in range(n):
        if i < n_src:
            frames = [
                cv2.cvtColor(
                    _load_image_rgb(src_root / src_paths[f][i]), cv2.COLOR_RGB2BGR
                )
                for f in feat_keys
            ]
            before = tile_bgr(frames, [_short_cam(f) for f in feat_keys])
        else:
            before = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

        if i < n_dst:
            frames = [
                cv2.cvtColor(
                    _load_image_rgb(dst_root / dst_paths[f][i]), cv2.COLOR_RGB2BGR
                )
                for f in feat_keys
            ]
            after = tile_bgr(frames, [_short_cam(f) for f in feat_keys])
            last_dst_panel = after
        else:
            after = (
                last_dst_panel
                if last_dst_panel is not None
                else np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            )

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[header : header + panel_h, 0:panel_w] = before
        canvas[header : header + panel_h, panel_w + gap :] = after
        title = (
            f"ep{episode_index}  before {min(i + 1, n_src)}/{n_src}  |  "
            f"after {min(i + 1, n_dst)}/{n_dst}  "
            f"prefix {plan.prefix_src}->{plan.prefix_dst}  "
            f"v_tgt={plan.v_target:.2f}deg/s  "
            f"L2 {plan.prefix_l2_before:.2f}->{plan.prefix_l2_after:.2f}"
        )
        cv2.putText(
            canvas, title, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, "BEFORE", (8, header - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            "AFTER",
            (panel_w + gap + 8, header - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 255, 180),
            1,
            cv2.LINE_AA,
        )
        writer.write(canvas)

    writer.release()
    return out_path


def parse_episodes(spec: str | None, available: list[int]) -> list[int]:
    if spec is None or spec.strip().lower() == "all":
        return list(available)
    avail = set(available)
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [e for e in out if e not in avail]
    if bad:
        raise ValueError(f"Episode(s) not in dataset: {bad}")
    return list(dict.fromkeys(out))


def run(
    src_root: Path,
    dst_root: Path | None,
    *,
    episodes: list[int],
    head_drop_frac: float,
    tail_drop_frac: float,
    enter_ratio: float,
    enter_frames: int,
    dry_run: bool,
    overwrite: bool,
    compare_video: bool,
) -> list[EpisodePlan]:
    if not is_lerobot_image_sequence_root(src_root):
        raise SystemExit(f"Not a leobot image_sequence dataset: {src_root}")
    info = load_lerobot_info(src_root)
    fps = float(info["fps"])
    available = list_episode_indices(src_root, info)
    episodes = [e for e in episodes if e in set(available)]
    if not episodes:
        raise SystemExit("No episodes to process")

    if not dry_run:
        if dst_root is None:
            raise SystemExit("--output is required unless --dry-run")
        if dst_root.resolve() == src_root.resolve():
            raise SystemExit("Refusing to write in-place; choose a different --output")
        if dst_root.exists():
            if not overwrite:
                raise SystemExit(f"Output exists: {dst_root} (pass --overwrite)")
            shutil.rmtree(dst_root)
        dst_root.mkdir(parents=True, exist_ok=True)

    plans: list[EpisodePlan] = []
    ep_meta_src = {
        int(r["episode_index"]): r
        for r in _load_episodes_jsonl(src_root / "meta" / "episodes.jsonl")
    }

    print(
        f"src={src_root}  eps={len(episodes)}  fps={fps}  "
        f"mid=drop_head{head_drop_frac:.0%}/tail{tail_drop_frac:.0%}  "
        f"enter_ratio={enter_ratio} enter_frames={enter_frames}"
        + ("  [DRY-RUN]" if dry_run else f"  dst={dst_root}")
    )

    for ep in episodes:
        data_rel = _format_data_path(info["data_path"], ep, int(info["chunks_size"]))
        table = pq.read_table(src_root / data_rel)
        state = np.stack(table.column("observation.state").to_pylist()).astype(np.float64)
        plan = plan_episode(
            state,
            fps,
            ep,
            head_drop_frac=head_drop_frac,
            tail_drop_frac=tail_drop_frac,
            enter_ratio=enter_ratio,
            enter_frames=enter_frames,
        )
        plans.append(plan)
        dropped = plan.length_src - plan.length_dst
        print(
            f"  ep{ep:3d}  T {plan.length_src}->{plan.length_dst} "
            f"(-{dropped}, {100.0 * dropped / max(plan.length_src, 1):.1f}%)  "
            f"prefix {plan.prefix_src}->{plan.prefix_dst}  "
            f"v_tgt={plan.v_target:.2f}  "
            f"prefix_L2 {plan.prefix_l2_before:.2f}->{plan.prefix_l2_after:.2f}"
        )

    if dry_run:
        total_src = sum(p.length_src for p in plans)
        total_dst = sum(p.length_dst for p in plans)
        print(f"total frames {total_src}->{total_dst} (-{total_src - total_dst})")
        return plans

    assert dst_root is not None

    meta_src = src_root / "meta"
    meta_dst = dst_root / "meta"
    meta_dst.mkdir(parents=True, exist_ok=True)
    skip_meta = {"info.json", "episodes.jsonl", "episodes_stats.jsonl"}
    for child in meta_src.iterdir():
        if child.name in skip_meta:
            continue
        dst = meta_dst / child.name
        if child.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(child, dst)
        else:
            shutil.copy2(child, dst)

    global_index = 0
    ep_stats_rows: list[dict[str, Any]] = []
    ep_jsonl_rows: list[dict[str, Any]] = []
    total_frames = 0
    feat_keys = _camera_feature_keys(info)
    n_cams = len(feat_keys)

    for plan in plans:
        data_rel = _format_data_path(
            info["data_path"], plan.episode_index, int(info["chunks_size"])
        )
        data = pq.read_table(src_root / data_rel).to_pydict()
        keep = filter_keep_existing_images(src_root, data, feat_keys, plan.keep)
        dropped_missing = int(plan.keep.shape[0] - keep.shape[0])
        if dropped_missing:
            print(
                f"  ep{plan.episode_index:3d}  drop {dropped_missing} keep-index "
                f"frames with missing camera JPEGs"
            )
            plan.keep = keep
            plan.length_dst = int(keep.shape[0])
        n, stats_entry = write_episode(
            src_root,
            dst_root,
            info,
            plan.episode_index,
            keep,
            global_index_start=global_index,
        )
        global_index += n
        total_frames += n
        ep_stats_rows.append(stats_entry)
        src_row = ep_meta_src.get(plan.episode_index, {"episode_index": plan.episode_index})
        ep_jsonl_rows.append(
            {
                "episode_index": plan.episode_index,
                "tasks": src_row.get("tasks", []),
                "length": n,
            }
        )

    out_info = dict(info)
    out_info["total_episodes"] = len(plans)
    out_info["total_frames"] = total_frames
    out_info["total_images"] = total_frames * n_cams
    out_info["splits"] = {"train": f"0:{len(plans)}"}
    out_info["speedup_slow_prefix"] = {
        "source": str(src_root),
        "head_drop_frac": head_drop_frac,
        "tail_drop_frac": tail_drop_frac,
        "enter_ratio": enter_ratio,
        "enter_frames": enter_frames,
        "episodes": [p.episode_index for p in plans],
    }
    with (meta_dst / "info.json").open("w") as f:
        json.dump(out_info, f, indent=2)
        f.write("\n")

    with (meta_dst / "episodes.jsonl").open("w") as f:
        for row in ep_jsonl_rows:
            f.write(json.dumps(row) + "\n")

    with (meta_dst / "episodes_stats.jsonl").open("w") as f:
        for row in ep_stats_rows:
            f.write(json.dumps(row) + "\n")

    src_stats = src_root / "stats.json"
    if src_stats.is_file():
        shutil.copy2(src_stats, dst_root / "stats.json")
    stats = compute_stats(dst_root)
    if src_stats.is_file():
        old = json.loads(src_stats.read_text())
        for k in ("image_mean", "image_std"):
            if k in old:
                stats[k] = np.asarray(old[k], dtype=np.float32)
    save_stats(dst_root, stats)

    if compare_video:
        for plan in plans:
            vid = (
                dst_root
                / "compare_videos"
                / f"ep{plan.episode_index:06d}_before_after.mp4"
            )
            path = write_compare_video(
                src_root, dst_root, info, plan.episode_index, plan, vid
            )
            print(f"compare video: {path}")

    print(
        f"Wrote {dst_root}: episodes={len(plans)} frames "
        f"{sum(p.length_src for p in plans)}->{total_frames}"
    )
    return plans


def main() -> None:
    p = argparse.ArgumentParser(
        description="Speed up slow LeRobot episode prefixes (mid-median target)."
    )
    p.add_argument("run_dir", type=str, help="Source image_sequence dataset root")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output dataset root (required unless --dry-run)",
    )
    p.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="Episode list/ranges, e.g. 11 or 0-5,11 (default: all)",
    )
    p.add_argument(
        "--head-drop-frac",
        type=float,
        default=0.10,
        help="Fraction dropped at start for mid median (default 0.10)",
    )
    p.add_argument(
        "--tail-drop-frac",
        type=float,
        default=0.05,
        help="Fraction dropped at end for mid median (default 0.05)",
    )
    p.add_argument("--enter-ratio", type=float, default=0.7)
    p.add_argument("--enter-frames", type=int, default=15)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--compare-video",
        action="store_true",
        help="Write before|after mp4 under <output>/compare_videos/",
    )
    args = p.parse_args()

    src = _resolve_run_dir(Path(args.run_dir))
    if not is_lerobot_image_sequence_root(src):
        raise SystemExit(f"Not a leobot image_sequence dataset: {src}")
    info = load_lerobot_info(src)
    available = list_episode_indices(src, info)
    episodes = parse_episodes(args.episodes, available)

    dst = _resolve_run_dir(Path(args.output)) if args.output else None
    if args.compare_video and (args.dry_run or dst is None):
        raise SystemExit("--compare-video requires a real --output write (not --dry-run)")

    run(
        src,
        dst,
        episodes=episodes,
        head_drop_frac=args.head_drop_frac,
        tail_drop_frac=args.tail_drop_frac,
        enter_ratio=args.enter_ratio,
        enter_frames=args.enter_frames,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        compare_video=args.compare_video,
    )


if __name__ == "__main__":
    main()
