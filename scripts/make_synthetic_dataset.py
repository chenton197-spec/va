#!/usr/bin/env python3
"""Create a tiny synthetic PushT-format dataset for smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from robotfm.data.stats import compute_stats, save_stats
from robotfm.data.writer import EpisodeWriter
from robotfm.types import EpisodeMeta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default="data/demos/synthetic_pusht")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--length", type=int, default=50)
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parents[1] / args.run_dir
    meta = EpisodeMeta(
        backend="pusht",
        embodiment="pusht_sim",
        fps=10,
        cameras={"top": {"height": 96, "width": 96, "channels": 3}},
        state_dim=2,
        action_dim=2,
        state_names=["x", "y"],
        action_names=["x", "y"],
        task="synthetic",
    )
    writer = EpisodeWriter(run_dir, meta)

    for ep in range(args.episodes):
        t = args.length
        images = {"top": np.random.randint(0, 255, (t, 96, 96, 3), dtype=np.uint8)}
        state = np.random.uniform(0, 512, (t, 2)).astype(np.float32)
        action = np.random.uniform(0, 512, (t, 2)).astype(np.float32)
        reward = np.zeros(t, dtype=np.float32)
        done = np.zeros(t, dtype=bool)
        done[-1] = True
        writer.write_episode(ep, images, state, action, reward, done, success=True, task="synthetic")

    save_stats(run_dir, compute_stats(run_dir))
    print(f"Wrote synthetic dataset to {run_dir}")


if __name__ == "__main__":
    main()
