#!/usr/bin/env python3
"""Play a leobot / LeRobot image-sequence dataset in an OpenCV window (no save).

Examples (from va/, conda env lerobot)::

    PYTHONPATH=. python scripts/play_dataset.py --run-dir shine_shoes_fr3_s256
    PYTHONPATH=. python scripts/play_dataset.py --run-dir shine_shoes_fr3_s256 --episode 3
    PYTHONPATH=. python scripts/play_dataset.py --run-dir shine_shoes_fr3_s256 --episode 0-5 --speed 2

Keys: q/ESC quit | space pause | n next episode | p prev | [/] slower/faster
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from robotfm.data.lerobot_dataset import (
    IMAGE_FEATURE_PREFIX,
    _load_image_rgb,
    is_lerobot_image_sequence_root,
    list_episode_indices,
    load_episode_arrays_from_parquet,
    load_lerobot_info,
)


def parse_episode_spec(spec: str, available: list[int]) -> list[int]:
    available_set = set(available)
    total = max(available) + 1 if available else 0
    spec = spec.strip().lower()
    if spec == "all":
        return list(available)

    episodes: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                raise ValueError(f"Invalid episode range: {part}")
            episodes.extend(range(start, end + 1))
        else:
            episodes.append(int(part))

    if not episodes:
        raise ValueError(f"No episodes parsed from: {spec!r}")
    invalid = [ep for ep in episodes if ep not in available_set]
    if invalid:
        raise ValueError(f"Episode index not in dataset [0, {total - 1}]: {invalid}")
    return list(dict.fromkeys(episodes))


def camera_feature_keys(info: dict) -> list[str]:
    return [k for k in info["features"] if k.startswith(IMAGE_FEATURE_PREFIX)]


def tile_cameras(frames_bgr: list[np.ndarray], labels: list[str]) -> np.ndarray:
    """Side-by-side BGR frames with camera labels."""
    if not frames_bgr:
        raise ValueError("No frames to display")
    h = max(f.shape[0] for f in frames_bgr)
    resized = []
    for frame, label in zip(frames_bgr, labels):
        if frame.shape[0] != h:
            scale = h / frame.shape[0]
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), h))
        canvas = frame.copy()
        cv2.putText(
            canvas,
            label,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        resized.append(canvas)
    return np.concatenate(resized, axis=1)


def play_episode(
    run_dir: Path,
    episode_index: int,
    info: dict,
    feat_keys: list[str],
    *,
    speed: float,
    window: str,
) -> str:
    """Play one episode. Returns action: quit | next | prev | done."""
    payload = load_episode_arrays_from_parquet(run_dir, episode_index, info, feat_keys)
    short_names = [k[len(IMAGE_FEATURE_PREFIX) :] for k in feat_keys]
    image_paths = payload["image_paths"]
    t_len = int(payload["length"])
    fps = float(info.get("fps", 30))
    base_dt = 1.0 / max(fps, 1e-6)

    print(
        f"episode {episode_index}: T={t_len} fps={fps} cameras={short_names} "
        f"state={payload['state'].shape} action={payload['action'].shape}"
    )

    paused = False
    frame_i = 0
    while 0 <= frame_i < t_len:
        t0 = time.perf_counter()
        frames_bgr = []
        for name in short_names:
            rel = image_paths[name][frame_i]
            path = run_dir / rel
            rgb = _load_image_rgb(path)
            frames_bgr.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        canvas = tile_cameras(frames_bgr, short_names)
        status = f"ep {episode_index}  {frame_i + 1}/{t_len}  x{speed:.2f}"
        if paused:
            status += "  [PAUSED]"
        cv2.putText(
            canvas,
            status,
            (8, canvas.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
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
            # Keep roughly on-time if decode is fast.
            elapsed = time.perf_counter() - t0
            target = base_dt / max(speed, 1e-3)
            if elapsed < target:
                time.sleep(target - elapsed)
            frame_i += 1

    return "done"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Play leobot image-sequence dataset (OpenCV, no save)"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Dataset root (e.g. shine_shoes_fr3_s256)",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default="0",
        help="Episode selector: index, range 0-5, list 0,3,10, or all",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop through selected episodes until quit",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (base_dir / run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Dataset root missing: {run_dir}")
    if not is_lerobot_image_sequence_root(run_dir):
        raise ValueError(
            f"Not a leobot image_sequence dataset: {run_dir}\n"
            "Expected meta/info.json with image_storage=image_sequence"
        )

    info = load_lerobot_info(run_dir)
    available = list_episode_indices(run_dir, info)
    episodes = parse_episode_spec(args.episode, available)
    feat_keys = camera_feature_keys(info)
    if not feat_keys:
        raise ValueError("No observation.images.* features found")

    window = f"play_dataset: {run_dir.name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    speed = float(args.speed)
    idx = 0
    while True:
        ep = episodes[idx]
        action = play_episode(
            run_dir, ep, info, feat_keys, speed=speed, window=window
        )
        if action == "quit":
            break
        if action == "prev":
            idx = (idx - 1) % len(episodes)
            continue
        if action == "next":
            idx = (idx + 1) % len(episodes)
            continue
        # done
        if idx + 1 < len(episodes):
            idx += 1
            continue
        if args.loop:
            idx = 0
            continue
        break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
