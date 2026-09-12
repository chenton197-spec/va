#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robotfm.data.lerobot_dataset import (
    DEPTH_FEATURE_PREFIX,
    IMAGE_FEATURE_PREFIX,
    _format_data_path,
    _load_image_rgb,
    is_lerobot_image_sequence_root,
    list_episode_indices,
    load_episode_arrays_from_parquet,
    load_lerobot_info,
)
from robotfm.data.stats import compute_stats, save_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preview_openarm_hcx_dual_arm import (
    camera_feature_keys,
    draw_signal_panel,
    load_episode_extras,
    mosaic_cameras,
    parse_episode_spec,
)
from trim_left_arm_idle_prefix import _episode_stats_entry, _image_path_from_cell

DEFAULT_RUN_DIR = "data/openarm_hcx_dual_arm_with_out_room_s"


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_media_key(key: str) -> bool:
    return key.startswith(IMAGE_FEATURE_PREFIX) or key.startswith(DEPTH_FEATURE_PREFIX)


def drop_frame_range(
    table: pa.Table, lo: int, hi: int, fps: float
) -> tuple[pa.Table, list[str]]:
    t = table.num_rows
    if not (0 <= lo <= hi < t):
        raise ValueError(f"invalid range [{lo}, {hi}] for T={t}")
    n_drop = hi - lo + 1
    if n_drop >= t:
        raise ValueError("would drop entire episode")
    data = table.to_pydict()
    dropped_paths: list[str] = []
    for key, col in data.items():
        if not _is_media_key(key):
            continue
        for i in range(lo, hi + 1):
            dropped_paths.append(_image_path_from_cell(col[i]))
    keep = list(range(0, lo)) + list(range(hi + 1, t))
    out: dict[str, Any] = {}
    for field in table.schema:
        col = data[field.name]
        out[field.name] = [col[i] for i in keep]
    n = len(keep)
    out["frame_index"] = list(range(n))
    out["timestamp"] = [i / float(fps) for i in range(n)]
    for key, col in list(out.items()):
        if not _is_media_key(key):
            continue
        new_col = []
        for new_fi, cell in enumerate(col):
            path = _image_path_from_cell(cell)
            ts = new_fi / float(fps)
            if isinstance(cell, dict):
                new_cell = dict(cell)
                new_cell["path"] = path
                new_cell["timestamp"] = ts
                new_col.append(new_cell)
            else:
                new_col.append({"path": path, "timestamp": ts})
        out[key] = new_col
    arrays = [pa.array(out[field.name], type=field.type) for field in table.schema]
    return pa.Table.from_arrays(arrays, schema=table.schema), dropped_paths


def apply_cut(
    run_dir: Path,
    episode_index: int,
    info: dict[str, Any],
    lo: int,
    hi: int,
) -> int:
    fps = float(info.get("fps", 30))
    chunks_size = int(info["chunks_size"])
    rel = _format_data_path(info["data_path"], episode_index, chunks_size)
    pq_path = run_dir / rel
    table = pq.read_table(pq_path)
    old_n = table.num_rows
    new_table, dropped_paths = drop_frame_range(table, lo, hi, fps)
    n = new_table.num_rows
    n_drop = old_n - n
    n_cams = sum(1 for k in info["features"] if k.startswith(IMAGE_FEATURE_PREFIX))
    n_depth = sum(1 for k in info["features"] if k.startswith(DEPTH_FEATURE_PREFIX))

    tmp = pq_path.with_suffix(".parquet.tmp")
    pq.write_table(new_table, tmp)
    os.replace(tmp, pq_path)

    ep_path = run_dir / "meta" / "episodes.jsonl"
    ep_rows = _load_jsonl(ep_path)
    for row in ep_rows:
        if int(row["episode_index"]) == episode_index:
            row["length"] = n
    _write_jsonl(ep_path, ep_rows)

    stats_path = run_dir / "meta" / "episodes_stats.jsonl"
    stats_rows = _load_jsonl(stats_path)
    new_entry = _episode_stats_entry(new_table, episode_index)
    found = False
    for i, row in enumerate(stats_rows):
        if int(row["episode_index"]) == episode_index:
            stats_rows[i] = new_entry
            found = True
            break
    if not found:
        stats_rows.append(new_entry)
        stats_rows.sort(key=lambda r: int(r["episode_index"]))
    _write_jsonl(stats_path, stats_rows)

    info_path = run_dir / "meta" / "info.json"
    out_info = json.loads(info_path.read_text())
    out_info["total_frames"] = int(out_info.get("total_frames", 0)) - n_drop
    out_info["total_images"] = int(out_info.get("total_images", 0)) - n_drop * n_cams
    if n_depth:
        out_info["total_depth_images"] = (
            int(out_info.get("total_depth_images", 0)) - n_drop * n_depth
        )
    with info_path.open("w") as f:
        json.dump(out_info, f, indent=2)
        f.write("\n")
    info.update(out_info)

    src_stats = run_dir / "stats.json"
    old_image: dict[str, np.ndarray] = {}
    if src_stats.is_file():
        old = json.loads(src_stats.read_text())
        for k in ("image_mean", "image_std"):
            if k in old:
                old_image[k] = np.asarray(old[k], dtype=np.float32)
    stats = compute_stats(run_dir)
    stats.update(old_image)
    save_stats(run_dir, stats)

    deleted = 0
    for rel_img in dropped_paths:
        img = run_dir / rel_img
        if img.is_file():
            img.unlink()
            deleted += 1
    print(
        f"cut ep{episode_index:03d} [{lo},{hi}] T={old_n}->{n} "
        f"deleted_files={deleted}"
    )
    return n


def play_episode(
    run_dir: Path,
    episode_index: int,
    info: dict[str, Any],
    feat_keys: list[str],
    *,
    speed: float,
    window: str,
    max_width: int,
) -> str:
    payload = load_episode_arrays_from_parquet(run_dir, episode_index, info, feat_keys)
    extras = load_episode_extras(run_dir, episode_index, info)
    short_names = [k[len(IMAGE_FEATURE_PREFIX) :] for k in feat_keys]
    image_paths = payload["image_paths"]
    state = payload["state"]
    action = payload["action"]
    t_len = int(payload["length"])
    fps = float(info.get("fps", 15))
    base_dt = 1.0 / max(fps, 1e-6)
    print(
        f"episode {episode_index}: T={t_len} fps={fps} cameras={short_names}"
    )

    paused = False
    dragging = False
    mark: int | None = None
    frame_i = 0
    view = {"mosaic_h": 0, "canvas_w": 1, "canvas_h": 1}

    def _scale_xy(x: int, y: int) -> tuple[int, int]:
        try:
            _, _, ww, wh = cv2.getWindowImageRect(window)
        except cv2.error:
            ww, wh = view["canvas_w"], view["canvas_h"]
        if ww <= 0 or wh <= 0:
            return x, y
        return int(x * view["canvas_w"] / ww), int(y * view["canvas_h"] / wh)

    def _x_to_frame(x: int) -> int:
        if t_len <= 1:
            return 0
        w = max(view["canvas_w"] - 1, 1)
        return min(max(int(round(x * (t_len - 1) / w)), 0), t_len - 1)

    def on_mouse(event, x, y, _flags, _param) -> None:
        nonlocal paused, dragging, frame_i
        x, y = _scale_xy(x, y)
        if event == cv2.EVENT_LBUTTONDOWN and y >= view["mosaic_h"]:
            dragging = True
            paused = True
            frame_i = _x_to_frame(x)
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            paused = True
            frame_i = _x_to_frame(x)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False

    cv2.setMouseCallback(window, on_mouse)
    while 0 <= frame_i < t_len:
        t0 = time.perf_counter()
        frames_bgr = []
        for name in short_names:
            rel = image_paths[name][frame_i]
            rgb = _load_image_rgb(run_dir / rel)
            frames_bgr.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        mosaic = mosaic_cameras(frames_bgr, short_names, max_width)
        panel = draw_signal_panel(state, action, extras, frame_i, mosaic.shape[1])
        if mark is not None and t_len > 1:
            mx = int(round(mark * (panel.shape[1] - 1) / (t_len - 1)))
            cv2.line(panel, (mx, 0), (mx, panel.shape[0] - 1), (0, 255, 0), 1)
        canvas = np.concatenate([mosaic, panel], axis=0)
        view["mosaic_h"] = mosaic.shape[0]
        view["canvas_w"] = canvas.shape[1]
        view["canvas_h"] = canvas.shape[0]

        status = f"ep {episode_index}  {frame_i + 1}/{t_len}  x{speed:.2f}"
        if paused:
            status += "  [PAUSED]"
        status += "  space=pause"
        if mark is None:
            status += " m=mark"
        else:
            lo, hi = (mark, frame_i) if mark <= frame_i else (frame_i, mark)
            status += f"  MARK {mark}  CUT[{lo},{hi}] m=cut c=cancel"
        cv2.putText(
            canvas,
            status,
            (8, mosaic.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, canvas)

        wait_ms = 1 if paused else max(1, int(1000 * base_dt / max(speed, 1e-3)))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord("q"), 27):
            return "quit"
        if key == ord(" "):
            paused = not paused
        elif key == ord("m"):
            if mark is None:
                mark = frame_i
                paused = True
            else:
                lo, hi = (mark, frame_i) if mark <= frame_i else (frame_i, mark)
                try:
                    new_n = apply_cut(run_dir, episode_index, info, lo, hi)
                except ValueError as e:
                    print(e)
                    continue
                payload = load_episode_arrays_from_parquet(
                    run_dir, episode_index, info, feat_keys
                )
                extras = load_episode_extras(run_dir, episode_index, info)
                image_paths = payload["image_paths"]
                state = payload["state"]
                action = payload["action"]
                t_len = int(payload["length"])
                if t_len != new_n:
                    t_len = new_n
                frame_i = min(lo, t_len - 1)
                mark = None
                paused = True
                continue
        elif key == ord("c"):
            mark = None
        elif key == ord("n"):
            return "next"
        elif key == ord("p"):
            return "prev"
        elif key == ord("["):
            speed = max(0.1, speed / 1.25)
            print(f"speed={speed:.2f}")
        elif key == ord("]"):
            speed = min(16.0, speed * 1.25)
            print(f"speed={speed:.2f}")
        elif key == ord(",") and paused:
            frame_i = max(0, frame_i - 1)
            continue
        elif key == ord(".") and paused:
            frame_i = min(t_len - 1, frame_i + 1)
            continue

        if not paused:
            elapsed = time.perf_counter() - t0
            target = base_dt / max(speed, 1e-3)
            if elapsed < target:
                time.sleep(target - elapsed)
            frame_i += 1

    return "done"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=DEFAULT_RUN_DIR)
    parser.add_argument("--episode", type=str, default="0")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--max-width", type=int, default=1280)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (base_dir / run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Dataset root missing: {run_dir}")
    if not is_lerobot_image_sequence_root(run_dir):
        raise ValueError(f"Not a leobot image_sequence dataset: {run_dir}")

    info = load_lerobot_info(run_dir)
    available = list_episode_indices(run_dir, info)
    episodes = parse_episode_spec(args.episode, available)
    feat_keys = camera_feature_keys(info)
    if not feat_keys:
        raise ValueError("No observation.images.* features found")

    window = f"cut: {run_dir.name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        import tkinter as _tk
        _r = _tk.Tk()
        _r.withdraw()
        _sw, _sh = int(_r.winfo_screenwidth()), int(_r.winfo_screenheight())
        _r.destroy()
    except Exception:
        _sw, _sh = 1920, 1080
    cv2.resizeWindow(window, _sw * 3 // 4, _sh * 3 // 4)

    speed = float(args.speed)
    idx = 0
    while True:
        action = play_episode(
            run_dir,
            episodes[idx],
            info,
            feat_keys,
            speed=speed,
            window=window,
            max_width=int(args.max_width),
        )
        if action == "quit":
            break
        if action == "prev":
            idx = (idx - 1) % len(episodes)
            continue
        if action == "next":
            idx = (idx + 1) % len(episodes)
            continue
        if idx + 1 < len(episodes):
            idx += 1
            continue
        break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        raise SystemExit(130)
