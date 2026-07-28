#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from robotfm.data.schema import image_key, load_episode, load_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize robotfm NPZ dataset")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs/viz")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    meta = load_meta(run_dir)
    ep_path = run_dir / "episodes" / f"ep_{args.episode:06d}.npz"
    payload = load_episode(ep_path)
    arrays = payload["arrays"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = arrays["state"]
    action = arrays["action"]
    t = np.arange(state.shape[0])

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(t, state[:, 0], label="state_x")
    ax[0].plot(t, state[:, 1], label="state_y")
    ax[0].legend()
    ax[0].set_title("State")
    ax[1].plot(t, action[:, 0], label="action_x")
    ax[1].plot(t, action[:, 1], label="action_y")
    ax[1].legend()
    ax[1].set_title("Action")
    fig.tight_layout()
    fig.savefig(out_dir / f"ep_{args.episode:06d}_traj.png")
    plt.close(fig)

    cam = meta.camera_names[0]
    frames = arrays[image_key(cam)]
    import cv2

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_dir / f"ep_{args.episode:06d}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        meta.fps,
        (w, h),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved plots/video to {out_dir}")


if __name__ == "__main__":
    main()
