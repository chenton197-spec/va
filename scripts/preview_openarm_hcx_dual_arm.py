#!/usr/bin/env python3
"""Preview openarm_hcx_dual_arm LeRobot image-sequence data.

Shows head + left/right hand cameras in a mosaic, with a live joint/gripper strip.
Defaults to ``data/openarm_hcx_dual_arm``.

Examples (from va/, conda env lerobot)::

    PYTHONPATH=. python scripts/preview_openarm_hcx_dual_arm.py
    PYTHONPATH=. python scripts/preview_openarm_hcx_dual_arm.py --episode 3
    PYTHONPATH=. python scripts/preview_openarm_hcx_dual_arm.py --episode 0-5 --speed 2
    PYTHONPATH=. python scripts/preview_openarm_hcx_dual_arm.py --info
    PYTHONPATH=. python scripts/preview_openarm_hcx_dual_arm.py --episode 0 --save-dir outputs/preview_openarm
    PYTHONPATH=. python scripts/preview_openarm_hcx_dual_arm.py --episode 0 --export-only \\
        --save-dir outputs/preview_openarm --stride 2

Keys: q/ESC quit | space pause | n next | p prev | [/] slower/faster | ,/. step when paused
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from robotfm.data.lerobot_dataset import (
    IMAGE_FEATURE_PREFIX,
    _load_image_rgb,
    is_lerobot_image_sequence_root,
    list_episode_indices,
    load_episode_arrays_from_parquet,
    load_lerobot_info,
)

DEFAULT_RUN_DIR = "data/openarm_hcx_dual_arm"
PREFERRED_CAM_ORDER = ("head", "left_hand", "right_hand")
JOINT_NAMES = [f"L{i}" for i in range(7)] + [f"R{i}" for i in range(7)]


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
        raise ValueError(f"Episode index not in dataset: {invalid}")
    return list(dict.fromkeys(episodes))


def camera_feature_keys(info: dict) -> list[str]:
    keys = [k for k in info["features"] if k.startswith(IMAGE_FEATURE_PREFIX)]
    short_to_key = {k[len(IMAGE_FEATURE_PREFIX) :]: k for k in keys}
    ordered: list[str] = []
    for name in PREFERRED_CAM_ORDER:
        if name in short_to_key:
            ordered.append(short_to_key.pop(name))
    ordered.extend(short_to_key[n] for n in sorted(short_to_key))
    return ordered


def _scalar_col(data: dict, key: str, t: int) -> np.ndarray | None:
    if key not in data:
        return None
    arr = np.asarray(data[key], dtype=np.float32).reshape(-1)
    if arr.shape[0] != t:
        raise ValueError(f"{key} length {arr.shape[0]} != T={t}")
    return arr


def load_episode_extras(run_dir: Path, episode_index: int, info: dict) -> dict[str, np.ndarray]:
    """Load dual-arm gripper columns (not concatenated by the shared loader)."""
    template = info["data_path"]
    chunks_size = int(info["chunks_size"])
    chunk = episode_index // chunks_size
    rel = template.format(episode_chunk=chunk, episode_index=episode_index)
    table = pq.read_table(run_dir / rel)
    data = table.to_pydict()
    t = table.num_rows
    extras: dict[str, np.ndarray] = {}
    for key in (
        "observation.left_gripper",
        "observation.right_gripper",
        "action.left_gripper",
        "action.right_gripper",
    ):
        col = _scalar_col(data, key, t)
        if col is not None:
            extras[key] = col
    return extras


def resize_max_width(img: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or img.shape[1] <= max_width:
        return img
    scale = max_width / img.shape[1]
    return cv2.resize(img, (max_width, int(round(img.shape[0] * scale))))


def mosaic_cameras(frames_bgr: list[np.ndarray], labels: list[str], max_width: int) -> np.ndarray:
    """head on top; left_hand | right_hand below when all three present."""
    labeled = []
    for frame, label in zip(frames_bgr, labels):
        canvas = frame.copy()
        cv2.putText(
            canvas,
            label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        labeled.append(canvas)

    if len(labeled) == 3 and set(labels) >= {"head", "left_hand", "right_hand"}:
        by_name = dict(zip(labels, labeled))
        head = by_name["head"]
        left = by_name["left_hand"]
        right = by_name["right_hand"]
        hand_h = min(left.shape[0], right.shape[0])
        left = cv2.resize(left, (int(left.shape[1] * hand_h / left.shape[0]), hand_h))
        right = cv2.resize(right, (int(right.shape[1] * hand_h / right.shape[0]), hand_h))
        hands = np.concatenate([left, right], axis=1)
        target_w = hands.shape[1]
        head = cv2.resize(head, (target_w, int(head.shape[0] * target_w / head.shape[1])))
        mosaic = np.concatenate([head, hands], axis=0)
    else:
        h = max(f.shape[0] for f in labeled)
        resized = []
        for frame in labeled:
            if frame.shape[0] != h:
                frame = cv2.resize(frame, (int(frame.shape[1] * h / frame.shape[0]), h))
            resized.append(frame)
        mosaic = np.concatenate(resized, axis=1)

    return resize_max_width(mosaic, max_width)


def draw_signal_panel(
    state: np.ndarray,
    action: np.ndarray,
    extras: dict[str, np.ndarray],
    frame_i: int,
    width: int,
    height: int = 220,
) -> np.ndarray:
    """Compact joint + gripper strip with a playhead."""
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    t_len = state.shape[0]
    if t_len < 2 or width < 40:
        return panel

    # Left / right arm joints (14) + optional grippers.
    series: list[tuple[str, np.ndarray, tuple[int, int, int]]] = []
    for i, name in enumerate(JOINT_NAMES[: state.shape[1]]):
        color = (80, 180, 255) if i < 7 else (255, 160, 80)
        series.append((name, state[:, i], color))
    for key, color in (
        ("observation.left_gripper", (80, 255, 120)),
        ("observation.right_gripper", (120, 255, 80)),
        ("action.left_gripper", (180, 255, 180)),
        ("action.right_gripper", (200, 255, 160)),
    ):
        if key in extras:
            short = key.split(".", 1)[1]
            series.append((short, extras[key], color))

    n = len(series)
    row_h = max(1, (height - 28) // max(n, 1))
    x = np.linspace(0, width - 1, t_len).astype(np.int32)
    play_x = int(round(frame_i * (width - 1) / max(t_len - 1, 1)))

    for row, (name, values, color) in enumerate(series):
        y0 = 20 + row * row_h
        y1 = y0 + row_h - 2
        mid = (y0 + y1) // 2
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        span = max(vmax - vmin, 1e-6)
        ys = (y1 - (values - vmin) / span * (y1 - y0 - 2)).astype(np.int32)
        pts = np.stack([x, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(panel, [pts], False, color, 1, cv2.LINE_AA)
        cv2.putText(
            panel,
            f"{name}:{values[frame_i]:.2f}",
            (4, mid + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    cv2.line(panel, (play_x, 0), (play_x, height - 1), (0, 0, 255), 1)
    err = float(np.linalg.norm(action[frame_i] - state[frame_i]))
    cv2.putText(
        panel,
        f"state joints + grippers  t={frame_i}  |a-s|={err:.3f}",
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return panel


def print_dataset_info(run_dir: Path, info: dict, available: list[int]) -> None:
    features = info["features"]
    cams = [k[len(IMAGE_FEATURE_PREFIX) :] for k in camera_feature_keys(info)]
    lengths = []
    ep_path = run_dir / "meta" / "episodes.jsonl"
    if ep_path.is_file():
        with ep_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    lengths.append(int(json.loads(line)["length"]))
    print(f"root: {run_dir}")
    print(f"robot_type: {info.get('robot_type')}")
    print(f"fps: {info.get('fps')}  episodes: {len(available)}  frames: {info.get('total_frames')}")
    print(f"cameras: {cams}")
    print(
        f"state_dim: {features['observation.state']['shape'][0]}  "
        f"action_dim: {features['action']['shape'][0]}"
    )
    grip = [k for k in features if "gripper" in k]
    if grip:
        print(f"gripper_features: {grip}")
    if lengths:
        arr = np.asarray(lengths)
        print(
            f"episode_length: min={arr.min()} median={int(np.median(arr))} "
            f"max={arr.max()} mean={arr.mean():.1f}"
        )
    print(f"episode_indices: {available[0]}..{available[-1]} (n={len(available)})")


def save_episode_artifacts(
    out_dir: Path,
    episode_index: int,
    mosaic_frames: list[np.ndarray],
    state: np.ndarray,
    action: np.ndarray,
    extras: dict[str, np.ndarray],
    fps: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if mosaic_frames:
        h, w = mosaic_frames[0].shape[:2]
        video_path = out_dir / f"ep_{episode_index:06d}.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(fps, 1.0),
            (w, h),
        )
        for frame in mosaic_frames:
            writer.write(frame)
        writer.release()
        print(f"saved video: {video_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skip trajectory plot")
        return

    t = np.arange(state.shape[0])
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for i in range(min(7, state.shape[1])):
        axes[0].plot(t, state[:, i], label=JOINT_NAMES[i], linewidth=1)
    axes[0].set_ylabel("left joints")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[0].set_title(f"episode {episode_index} state")

    for i in range(7, min(14, state.shape[1])):
        axes[1].plot(t, state[:, i], label=JOINT_NAMES[i], linewidth=1)
    axes[1].set_ylabel("right joints")
    axes[1].legend(loc="upper right", ncol=4, fontsize=8)

    grip_plotted = False
    for key, label in (
        ("observation.left_gripper", "obs L grip"),
        ("observation.right_gripper", "obs R grip"),
        ("action.left_gripper", "act L grip"),
        ("action.right_gripper", "act R grip"),
    ):
        if key in extras:
            axes[2].plot(t, extras[key], label=label, linewidth=1)
            grip_plotted = True
    if not grip_plotted:
        axes[2].plot(t, np.linalg.norm(action - state, axis=1), label="|action-state|")
    axes[2].set_ylabel("grippers")
    axes[2].set_xlabel("frame")
    axes[2].legend(loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout()
    plot_path = out_dir / f"ep_{episode_index:06d}_traj.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"saved plot: {plot_path}")


def play_episode(
    run_dir: Path,
    episode_index: int,
    info: dict,
    feat_keys: list[str],
    *,
    speed: float,
    window: str,
    max_width: int,
    save_dir: Path | None,
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
        f"episode {episode_index}: T={t_len} fps={fps} cameras={short_names} "
        f"state={state.shape} action={action.shape} grippers={list(extras)}"
    )

    saved_frames: list[np.ndarray] = []
    paused = False
    frame_i = 0
    while 0 <= frame_i < t_len:
        t0 = time.perf_counter()
        frames_bgr = []
        for name in short_names:
            rel = image_paths[name][frame_i]
            rgb = _load_image_rgb(run_dir / rel)
            frames_bgr.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        mosaic = mosaic_cameras(frames_bgr, short_names, max_width)
        panel = draw_signal_panel(state, action, extras, frame_i, mosaic.shape[1])
        canvas = np.concatenate([mosaic, panel], axis=0)

        status = f"ep {episode_index}  {frame_i + 1}/{t_len}  x{speed:.2f}"
        if paused:
            status += "  [PAUSED]"
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
        if save_dir is not None:
            saved_frames.append(canvas.copy())

        wait_ms = 1 if paused else max(1, int(1000 * base_dt / max(speed, 1e-3)))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord("q"), 27):
            if save_dir is not None and saved_frames:
                save_episode_artifacts(
                    save_dir, episode_index, saved_frames, state, action, extras, fps
                )
            return "quit"
        if key == ord(" "):
            paused = not paused
        elif key == ord("n"):
            if save_dir is not None and saved_frames:
                save_episode_artifacts(
                    save_dir, episode_index, saved_frames, state, action, extras, fps
                )
            return "next"
        elif key == ord("p"):
            if save_dir is not None and saved_frames:
                save_episode_artifacts(
                    save_dir, episode_index, saved_frames, state, action, extras, fps
                )
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

    if save_dir is not None and saved_frames:
        save_episode_artifacts(
            save_dir, episode_index, saved_frames, state, action, extras, fps
        )
    return "done"


def export_episode(
    run_dir: Path,
    episode_index: int,
    info: dict,
    feat_keys: list[str],
    *,
    max_width: int,
    save_dir: Path,
    stride: int = 1,
) -> None:
    """Decode episode offline and write mp4 + trajectory plot (no GUI)."""
    payload = load_episode_arrays_from_parquet(run_dir, episode_index, info, feat_keys)
    extras = load_episode_extras(run_dir, episode_index, info)
    short_names = [k[len(IMAGE_FEATURE_PREFIX) :] for k in feat_keys]
    image_paths = payload["image_paths"]
    state = payload["state"]
    action = payload["action"]
    t_len = int(payload["length"])
    fps = float(info.get("fps", 15))
    print(
        f"export episode {episode_index}: T={t_len} fps={fps} cameras={short_names} "
        f"stride={stride}"
    )

    frames: list[np.ndarray] = []
    for frame_i in range(0, t_len, max(stride, 1)):
        frames_bgr = []
        for name in short_names:
            rgb = _load_image_rgb(run_dir / image_paths[name][frame_i])
            frames_bgr.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        mosaic = mosaic_cameras(frames_bgr, short_names, max_width)
        panel = draw_signal_panel(state, action, extras, frame_i, mosaic.shape[1])
        frames.append(np.concatenate([mosaic, panel], axis=0))

    export_fps = fps / max(stride, 1)
    save_episode_artifacts(save_dir, episode_index, frames, state, action, extras, export_fps)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preview openarm_hcx_dual_arm dataset (cameras + joints)"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=DEFAULT_RUN_DIR,
        help=f"Dataset root (default: {DEFAULT_RUN_DIR})",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default="0",
        help="Episode selector: index, range 0-5, list 0,3,10, or all",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Downscale mosaic width for display (0 = native)",
    )
    parser.add_argument("--loop", action="store_true", help="Loop selected episodes")
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print dataset summary and exit (no GUI)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="If set, also save mp4 + trajectory plot under this directory",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="No GUI: write mp4/plots to --save-dir (required)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride for --export-only (e.g. 2 = half fps)",
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
    if args.info:
        print_dataset_info(run_dir, info, available)
        return 0

    episodes = parse_episode_spec(args.episode, available)
    feat_keys = camera_feature_keys(info)
    if not feat_keys:
        raise ValueError("No observation.images.* features found")

    save_dir = Path(args.save_dir).resolve() if args.save_dir else None
    if args.export_only:
        if save_dir is None:
            raise SystemExit("--export-only requires --save-dir")
        for ep in episodes:
            export_episode(
                run_dir,
                ep,
                info,
                feat_keys,
                max_width=int(args.max_width),
                save_dir=save_dir,
                stride=int(args.stride),
            )
        return 0

    window = f"preview: {run_dir.name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

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
            save_dir=save_dir,
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
