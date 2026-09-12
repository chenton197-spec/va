#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_fps_even_odd import (
    _feature_keys,
    _load_jsonl,
    _resolve,
    _write_json,
    _write_jsonl,
    write_split_episode,
)

FPS = 30
KEEP_PAUSE = 6
G_CLOSED = 0.5
LEFT_STILL = 0.4
RIGHT_STILL = 0.18
HOLE = 2
LEFT_HOLE = 3
MIN_HOLD = 40
MIN_LIFT = 20
MIN_LIFT_R1 = 4.0
MIN_LIFT_R4 = 15.0
MIN_PAUSE = 6
MAX_PAUSE_DRIFT = 2.5
TRIM_EPS = frozenset(range(70)) | frozenset(
    (
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        113,
        114,
        115,
        117,
        118,
        119,
        120,
        122,
        123,
        124,
        126,
        127,
        185,
    )
)


def _fill_holes(mask: np.ndarray, hole: int) -> np.ndarray:
    n = len(mask)
    out = mask.copy()
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= hole:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _segments(mask: np.ndarray) -> list[tuple[int, int, bool, int]]:
    out: list[tuple[int, int, bool, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        j = i
        v = bool(mask[i])
        while j < n and bool(mask[j]) == v:
            j += 1
        out.append((i, j, v, j - i))
        i = j
    return out


def _grip_col(col: list[Any]) -> np.ndarray:
    vals: list[float] = []
    for v in col:
        if isinstance(v, (list, tuple)):
            vals.append(float(v[0]))
        else:
            vals.append(float(v))
    return np.asarray(vals, dtype=np.float32)


def detect_pauses(state: np.ndarray, lg: np.ndarray, rg: np.ndarray) -> list[tuple[int, int]]:
    n = int(state.shape[0])
    L = state[:, :7]
    sR = state[:, 7:14]
    dL = np.max(np.abs(np.diff(L, axis=0, prepend=L[:1])), axis=1)
    dR = np.max(np.abs(np.diff(sR, axis=0, prepend=sR[:1])), axis=1)
    rgc = [i for i in range(1, n) if rg[i] < G_CLOSED and rg[i - 1] >= G_CLOSED]
    pauses: list[tuple[int, int]] = []
    for rc in rgc:
        if lg[rc] >= G_CLOSED:
            continue
        t0 = rc
        while t0 > 0 and lg[t0 - 1] < G_CLOSED:
            t0 -= 1
        if rc - t0 < MIN_HOLD:
            continue
        left_still = _fill_holes(dL <= LEFT_STILL, LEFT_HOLE)
        hold_idx = np.where(left_still[t0 : rc + 1] & (lg[t0 : rc + 1] < G_CLOSED))[0]
        if len(hold_idx) < MIN_HOLD:
            continue
        hs = t0 + int(hold_idx[0])
        he = rc
        if he - hs < MIN_HOLD:
            continue
        win0 = max(0, hs - 25)
        still_r = _fill_holes(dR <= RIGHT_STILL, HOLE)
        segs = [
            (win0 + a, win0 + b, v, ln)
            for a, b, v, ln in _segments(still_r[win0 : he + 1])
        ]
        lift = None
        for a, b, v, ln in segs:
            if v or b <= hs - 5 or ln < MIN_LIFT:
                continue
            dR1 = float(sR[min(b, n) - 1, 1] - sR[a, 1])
            dR4 = float(sR[min(b, n) - 1, 4] - sR[a, 4])
            if dR1 >= MIN_LIFT_R1 or abs(dR4) >= MIN_LIFT_R4:
                lift = (a, b)
                break
        if lift is None:
            continue
        lb = lift[1]
        for a, b, v, ln in segs:
            if not v or a < lb - 1 or ln < MIN_PAUSE:
                continue
            drift = float(np.max(np.abs(sR[min(b, n) - 1] - sR[a])))
            if drift > MAX_PAUSE_DRIFT:
                continue
            later = [x for x in segs if (not x[2]) and x[0] >= b - 2 and x[3] >= 8]
            if later:
                pauses.append((a, b - 1))
                break
    return pauses


def keep_indices(state: np.ndarray, lg: np.ndarray, rg: np.ndarray, trim: bool) -> np.ndarray:
    t = int(state.shape[0])
    keep = np.ones(t, dtype=bool)
    if trim:
        for a, b in detect_pauses(state, lg, rg):
            extra = (b - a + 1) - KEEP_PAUSE
            if extra > 0:
                keep[a + KEEP_PAUSE : b + 1] = False
    return np.flatnonzero(keep).astype(np.int64)


def run(src_root: Path, dst_root: Path, *, overwrite: bool) -> None:
    if not is_lerobot_image_sequence_root(src_root):
        raise SystemExit(f"Not a leobot image_sequence dataset: {src_root}")
    info = load_lerobot_info(src_root)
    if int(info["fps"]) != FPS:
        raise SystemExit(f"src fps={info['fps']}, expected {FPS}")
    if dst_root.resolve() == src_root.resolve():
        raise SystemExit("dst must differ from src")
    if dst_root.exists():
        if not overwrite:
            raise SystemExit(f"Output exists: {dst_root} (pass --overwrite)")
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)

    episodes = list_episode_indices(src_root, info)
    ep_meta_src = {
        int(r["episode_index"]): r
        for r in _load_jsonl(src_root / "meta" / "episodes.jsonl")
    }
    image_keys = _feature_keys(info, IMAGE_FEATURE_PREFIX)
    depth_keys = _feature_keys(info, DEPTH_FEATURE_PREFIX)
    chunks_size = int(info["chunks_size"])

    meta_dst = dst_root / "meta"
    meta_dst.mkdir(parents=True, exist_ok=True)
    for name in ("depth_sources.json", "tasks.jsonl"):
        src_meta = src_root / "meta" / name
        if src_meta.is_file():
            shutil.copy2(src_meta, meta_dst / name)

    ep_jsonl_rows: list[dict[str, Any]] = []
    ep_stats_rows: list[dict[str, Any]] = []
    global_index = 0
    total_frames = 0
    dropped_total = 0
    for src_ep in episodes:
        data_rel = _format_data_path(info["data_path"], src_ep, chunks_size)
        table = pq.read_table(
            src_root / data_rel,
            columns=[
                "observation.state",
                "observation.left_gripper",
                "observation.right_gripper",
            ],
        )
        d = table.to_pydict()
        state = np.stack(d["observation.state"]).astype(np.float32)
        lg = _grip_col(d["observation.left_gripper"])
        rg = _grip_col(d["observation.right_gripper"])
        keep = keep_indices(state, lg, rg, src_ep in TRIM_EPS)
        t_src = int(state.shape[0])
        n, stats_entry = write_split_episode(
            src_root,
            dst_root,
            info,
            src_ep,
            src_ep,
            keep,
            fps=float(FPS),
            global_index_start=global_index,
        )
        dropped = t_src - n
        dropped_total += dropped
        global_index += n
        total_frames += n
        ep_stats_rows.append(stats_entry)
        src_row = ep_meta_src.get(src_ep, {"episode_index": src_ep})
        ep_jsonl_rows.append(
            {
                "episode_index": src_ep,
                "tasks": src_row.get("tasks", []),
                "length": n,
            }
        )
        print(f"ep {src_ep:03d}  {t_src}->{n}  drop={dropped}")

    max_ep = max(int(r["episode_index"]) for r in ep_jsonl_rows)
    written = len(ep_jsonl_rows)
    out_info = dict(info)
    out_info["fps"] = FPS
    out_info["total_episodes"] = written
    out_info["total_frames"] = total_frames
    out_info["total_images"] = total_frames * len(image_keys)
    out_info["total_depth_images"] = total_frames * len(depth_keys)
    out_info["total_chunks"] = max_ep // max(chunks_size, 1) + 1
    out_info["splits"] = {"train": f"0:{max_ep + 1}"}
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
        f"wrote {dst_root} episodes={written} frames={total_frames} "
        f"dropped={dropped_total} fps={FPS}"
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
        default=va_root / "data" / "30fps_clean",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(_resolve(args.src), _resolve(args.dst), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
