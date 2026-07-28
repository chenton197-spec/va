#!/usr/bin/env python3
"""Visualize the local LeRobot PushT dataset at datasets/pusht.

Examples:
    # Interactive Rerun viewer for episode 0
    python va/scripts/visualize_pusht.py --episode 0 --mode rerun

    # Visualize all episodes (videos/plots for each; one Rerun app for all)
    python va/scripts/visualize_pusht.py --episode all --mode video --output-dir va/outputs/pusht
    python va/scripts/visualize_pusht.py --episode all --mode rerun

    # Episode range or list
    python va/scripts/visualize_pusht.py --episode 0-5 --mode plot
    python va/scripts/visualize_pusht.py --episode 0,3,10 --mode video

    # Print dataset overview
    python va/scripts/visualize_pusht.py --mode overview
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import tqdm

# Use the local lerobot checkout when available.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = PROJECT_ROOT / "lerobot" / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets import LeRobotDataset  # noqa: E402


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "pusht"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "va" / "outputs" / "pusht"


def resolve_dataset_root(path: str | Path | None) -> Path:
    root = Path(path) if path is not None else DEFAULT_DATASET_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    return root


def load_dataset(root: Path, episode: int | None) -> LeRobotDataset:
    episodes = [episode] if episode is not None else None
    return LeRobotDataset("pusht", root=root, episodes=episodes)


def load_info(root: Path) -> dict:
    with (root / "meta" / "info.json").open() as f:
        return json.load(f)


def parse_episode_spec(spec: str, total_episodes: int) -> list[int]:
    """Parse episode selector: 'all', '0-5', '0,3,10', or a single index."""
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(total_episodes))

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

    invalid = [ep for ep in episodes if ep < 0 or ep >= total_episodes]
    if invalid:
        raise ValueError(
            f"Episode index out of range [0, {total_episodes - 1}]: {invalid}"
        )
    # Preserve order but drop duplicates.
    return list(dict.fromkeys(episodes))


def to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    if image.dtype != torch.float32:
        image = image.float()
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    return (image.clamp(0, 1) * 255).byte().cpu().numpy()


def collect_episode_arrays(dataset: LeRobotDataset) -> dict[str, np.ndarray]:
    states, actions, rewards, dones, timestamps = [], [], [], [], []
    images = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        states.append(item["observation.state"].numpy())
        actions.append(item["action"].numpy())
        rewards.append(float(item["next.reward"]))
        dones.append(bool(item["next.done"]))
        timestamps.append(float(item["timestamp"]))
        images.append(to_hwc_uint8(item["observation.image"]))

    return {
        "states": np.asarray(states),
        "actions": np.asarray(actions),
        "rewards": np.asarray(rewards),
        "dones": np.asarray(dones),
        "timestamps": np.asarray(timestamps),
        "images": np.asarray(images),
    }


def print_overview(root: Path) -> None:
    info = load_info(root)
    print(f"Dataset root: {root}")
    print(f"Codebase version: {info['codebase_version']}")
    print(f"Total episodes: {info['total_episodes']}")
    print(f"Total frames: {info['total_frames']}")
    print(f"FPS: {info['fps']}")
    print(f"Features:")
    for key, feature in info["features"].items():
        print(f"  - {key}: dtype={feature['dtype']}, shape={feature['shape']}")


def save_plots(episode: int, arrays: dict[str, np.ndarray], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    states = arrays["states"]
    actions = arrays["actions"]
    rewards = arrays["rewards"]
    timestamps = arrays["timestamps"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"PushT Episode {episode}", fontsize=14)

    ax = axes[0, 0]
    ax.plot(states[:, 0], states[:, 1], color="#2563eb", linewidth=2, label="state")
    ax.scatter(states[0, 0], states[0, 1], color="#16a34a", s=40, label="start")
    ax.scatter(states[-1, 0], states[-1, 1], color="#dc2626", s=40, label="end")
    ax.set_title("Agent trajectory (observation.state)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(actions[:, 0], actions[:, 1], color="#9333ea", linewidth=2, label="action")
    ax.scatter(actions[0, 0], actions[0, 1], color="#16a34a", s=40, label="start")
    ax.scatter(actions[-1, 0], actions[-1, 1], color="#dc2626", s=40, label="end")
    ax.set_title("Action targets")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(timestamps, states[:, 0], label="state x")
    ax.plot(timestamps, states[:, 1], label="state y")
    ax.plot(timestamps, actions[:, 0], "--", label="action x")
    ax.plot(timestamps, actions[:, 1], "--", label="action y")
    ax.set_title("State / action over time")
    ax.set_xlabel("timestamp (s)")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(timestamps, rewards, color="#ea580c", linewidth=2)
    ax.set_title("Reward over time")
    ax.set_xlabel("timestamp (s)")
    ax.set_ylabel("next.reward")
    ax.grid(True, alpha=0.3)

    plot_path = output_dir / f"episode_{episode:03d}_summary.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def save_video(episode: int, arrays: dict[str, np.ndarray], output_dir: Path, fps: int) -> None:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    images = arrays["images"]
    states = arrays["states"]
    actions = arrays["actions"]
    rewards = arrays["rewards"]

    height, width = images.shape[1:3]
    panel_h = 220
    out_h = height + panel_h
    out_w = width

    video_path = output_dir / f"episode_{episode:03d}.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )

    for frame_idx, image in enumerate(tqdm.tqdm(images, desc="Writing video")):
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:height, :width] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        panel = canvas[height:, :].copy()
        panel[:] = (245, 245, 245)

        state = states[frame_idx]
        action = actions[frame_idx]
        reward = rewards[frame_idx]

        lines = [
            f"frame={frame_idx}",
            f"state=({state[0]:.1f}, {state[1]:.1f})",
            f"action=({action[0]:.1f}, {action[1]:.1f})",
            f"reward={reward:.4f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                panel,
                line,
                (12, 28 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

        # Mini trajectory map in the panel.
        map_size = 180
        map_x0 = out_w - map_size - 12
        map_y0 = 12
        cv2.rectangle(panel, (map_x0, map_y0), (map_x0 + map_size, map_y0 + map_size), (200, 200, 200), 1)

        def to_map_xy(xy: np.ndarray) -> tuple[int, int]:
            xs = states[: frame_idx + 1, 0]
            ys = states[: frame_idx + 1, 1]
            xmin, xmax = states[:, 0].min(), states[:, 0].max()
            ymin, ymax = states[:, 1].min(), states[:, 1].max()
            x = int(map_x0 + (xy[0] - xmin) / max(xmax - xmin, 1e-6) * (map_size - 8) + 4)
            y = int(map_y0 + (xy[1] - ymin) / max(ymax - ymin, 1e-6) * (map_size - 8) + 4)
            return x, y

        for prev_idx in range(1, frame_idx + 1):
            p0 = to_map_xy(states[prev_idx - 1])
            p1 = to_map_xy(states[prev_idx])
            cv2.line(panel, p0, p1, (37, 99, 235), 1, cv2.LINE_AA)

        cv2.circle(panel, to_map_xy(state), 4, (220, 38, 38), -1, cv2.LINE_AA)
        cv2.circle(panel, to_map_xy(action), 4, (147, 51, 234), -1, cv2.LINE_AA)

        canvas[height:, :] = panel
        writer.write(canvas)

    writer.release()
    print(f"Saved video: {video_path}")


def visualize_rerun(
    root: Path,
    episodes: list[int],
    save_rrd: bool,
    output_dir: Path | None,
) -> None:
    import rerun as rr

    spawn = not save_rrd
    if len(episodes) == 1:
        app_id = f"pusht/episode_{episodes[0]}"
    else:
        app_id = f"pusht/episodes_{episodes[0]}-{episodes[-1]}"
    rr.init(app_id, spawn=spawn)

    for episode in episodes:
        dataset = load_dataset(root, episode)
        prefix = f"episode_{episode:03d}" if len(episodes) > 1 else ""
        first_index = None

        for idx in tqdm.tqdm(range(len(dataset)), desc=f"Rerun ep {episode}"):
            item = dataset[idx]
            if first_index is None:
                first_index = int(item["index"])

            rr.set_time("frame_index", sequence=int(item["index"]) - first_index)
            rr.set_time("timestamp", timestamp=float(item["timestamp"]))
            if len(episodes) > 1:
                rr.set_time("episode", sequence=episode)

            path = f"{prefix}/" if prefix else ""
            image = to_hwc_uint8(item["observation.image"])
            rr.log(f"{path}observation.image", rr.Image(image))

            state = item["observation.state"]
            action = item["action"]
            for dim_idx, val in enumerate(state):
                rr.log(f"{path}observation.state/{dim_idx}", rr.Scalars(float(val)))
            for dim_idx, val in enumerate(action):
                rr.log(f"{path}action/{dim_idx}", rr.Scalars(float(val)))

            rr.log(f"{path}next.reward", rr.Scalars(float(item["next.reward"])))
            rr.log(f"{path}next.done", rr.Scalars(float(item["next.done"])))
            rr.log(f"{path}next.success", rr.Scalars(float(item["next.success"])))

    if save_rrd:
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        if len(episodes) == 1:
            rrd_path = output_dir / f"pusht_episode_{episodes[0]:03d}.rrd"
        else:
            rrd_path = output_dir / f"pusht_episodes_{episodes[0]:03d}-{episodes[-1]:03d}.rrd"
        rr.save(rrd_path)
        print(f"Saved Rerun recording: {rrd_path}")
        print(f"View later with: rerun {rrd_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the local PushT LeRobot dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Path to the dataset root (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default="0",
        help="Episode index, list (0,3,10), range (0-5), or 'all'.",
    )
    parser.add_argument(
        "--mode",
        choices=["overview", "rerun", "plot", "video", "all"],
        default="rerun",
        help="Visualization mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for saved artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="FPS for exported video.",
    )
    parser.add_argument(
        "--save-rrd",
        action="store_true",
        help="When using rerun mode, save a .rrd file instead of opening a viewer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_dataset_root(args.dataset_root)

    if args.mode == "overview":
        print_overview(root)
        return

    info = load_info(root)
    episodes = parse_episode_spec(args.episode, info["total_episodes"])
    print(f"Visualizing {len(episodes)} episode(s): {episodes[0]}..{episodes[-1]}")

    if args.mode in {"plot", "video", "all"}:
        for episode in episodes:
            dataset = load_dataset(root, episode)
            print(f"Loaded episode {episode}: {len(dataset)} frames")
            arrays = collect_episode_arrays(dataset)
            if args.mode in {"plot", "all"}:
                save_plots(episode, arrays, args.output_dir)
            if args.mode in {"video", "all"}:
                save_video(episode, arrays, args.output_dir, fps=args.fps)

    if args.mode in {"rerun", "all"}:
        # For many episodes, prefer saving .rrd over spawning one huge interactive session.
        save_rrd = args.save_rrd or args.mode == "all" or len(episodes) > 20
        if save_rrd and not args.save_rrd and args.mode != "all":
            print(
                f"Note: {len(episodes)} episodes selected; saving .rrd instead of spawning viewer. "
                "Pass --save-rrd explicitly to silence this, or use a smaller range for live viewing."
            )
        visualize_rerun(
            root,
            episodes=episodes,
            save_rrd=save_rrd,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()

