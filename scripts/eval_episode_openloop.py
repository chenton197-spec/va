#!/usr/bin/env python3
"""Open-loop offline eval: predicted vs GT actions on one dataset episode.

Matches training/eval conditions from the checkpoint config:
- stats + norm_mode from checkpoint
- resize → fixed center crop (no random crop / no color jitter)
- action chunking with policy.n_action_steps / horizon / num_inference_steps

Usage (from va/ with conda env lerobot)::

    PYTHONPATH=. python scripts/eval_episode_openloop.py \\
      --checkpoint outputs/fm_shine_shoes_limits_260730163921/checkpoint_030000.pt \\
      --episode 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from robotfm.collect.loop import get_run_dir
from robotfm.config import resolve_path
from robotfm.data.dataset import crop_images, resize_images
from robotfm.data.lerobot_dataset import (
    _load_image_rgb,
    _short_camera_name,
    load_episode_arrays_from_parquet,
    load_lerobot_info,
)
from robotfm.data.stats import denormalize, normalize
from robotfm.train import build_policy


def _build_obs_batch(
    *,
    run_dir: Path,
    image_paths: dict[str, list[str]],
    states: np.ndarray,
    cameras: list[str],
    t: int,
    n_obs_steps: int,
    resize_size: int | None,
    crop_size: int | None,
    stats: dict,
    norm_mode: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    obs_start = max(0, t - n_obs_steps + 1)
    obs_indices = list(range(obs_start, t + 1))
    while len(obs_indices) < n_obs_steps:
        obs_indices.insert(0, obs_indices[0])

    camera_histories = []
    for cam in cameras:
        frames = []
        for fi in obs_indices:
            img = _load_image_rgb(run_dir / image_paths[cam][fi])
            img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(img))
        camera_histories.append(torch.stack(frames, dim=0))

    obs_images = torch.stack(camera_histories, dim=0)
    obs_images = resize_images(obs_images, resize_size)
    if crop_size is not None:
        obs_images = crop_images(obs_images, crop_size, random=False)

    state = normalize(states[obs_indices].astype(np.float32), stats, prefix="state", mode=norm_mode)
    return {
        "obs_images": obs_images.unsqueeze(0).to(device),
        "obs_state": torch.from_numpy(state).unsqueeze(0).to(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Open-loop episode action comparison")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episode", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: <ckpt_dir>/eval_epXXXX)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = base_dir / ckpt_path

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    stats = ckpt["stats"]
    norm_mode = cfg.dataset.norm_mode
    n_obs = cfg.dataset.n_obs_steps
    horizon = cfg.dataset.horizon
    n_action_steps = int(cfg.policy.n_action_steps)
    action_names = list(cfg.action_names)
    cameras = list(cfg.cameras)

    run_dir = get_run_dir(cfg, base_dir)
    info = load_lerobot_info(run_dir)
    features = info["features"]
    cam_feat_keys = [
        next(k for k in features if _short_camera_name(k) == cam) for cam in cameras
    ]
    payload = load_episode_arrays_from_parquet(
        run_dir, args.episode, info, cam_feat_keys
    )
    states = payload["state"]
    actions_gt = payload["action"]
    image_paths = payload["image_paths"]
    length = int(payload["length"])

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else ckpt_path.parent / f"eval_ep{args.episode:04d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    policy = build_policy(cfg, stats)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()

    print(f"checkpoint: {ckpt_path}")
    print(f"run_dir: {run_dir}")
    print(f"episode: {args.episode} length={length}")
    print(
        f"norm_mode={norm_mode} n_obs={n_obs} horizon={horizon} "
        f"n_action_steps={n_action_steps} num_inference_steps={cfg.policy.num_inference_steps}"
    )
    print(
        f"resize={cfg.dataset.resize_size} crop={cfg.dataset.crop_size} "
        f"eval_fixed_crop={cfg.dataset.eval_fixed_crop} device={device}"
    )

    pred_actions = np.full_like(actions_gt, np.nan, dtype=np.float32)
    replan_ts: list[int] = []

    t = 0
    while t < length:
        batch = _build_obs_batch(
            run_dir=run_dir,
            image_paths=image_paths,
            states=states,
            cameras=cameras,
            t=t,
            n_obs_steps=n_obs,
            resize_size=cfg.dataset.resize_size,
            crop_size=cfg.dataset.crop_size,
            stats=stats,
            norm_mode=norm_mode,
            device=device,
        )
        with torch.no_grad():
            pred_norm = policy.sample_actions(batch)[0].cpu()
        pred_phys = denormalize(pred_norm, stats, prefix="action", mode=norm_mode).numpy()

        take = min(n_action_steps, length - t, pred_phys.shape[0])
        pred_actions[t : t + take] = pred_phys[:take]
        replan_ts.append(t)
        t += take

    valid = np.isfinite(pred_actions).all(axis=1)
    gt = actions_gt[valid]
    pred = pred_actions[valid]
    err = pred - gt
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))

    metrics = {
        "checkpoint": str(ckpt_path),
        "episode": args.episode,
        "length": length,
        "n_valid_steps": int(valid.sum()),
        "n_replans": len(replan_ts),
        "norm_mode": norm_mode,
        "n_obs_steps": n_obs,
        "horizon": horizon,
        "n_action_steps": n_action_steps,
        "num_inference_steps": int(cfg.policy.num_inference_steps),
        "resize_size": cfg.dataset.resize_size,
        "crop_size": cfg.dataset.crop_size,
        "seed": args.seed,
        "mae_per_dim": {n: float(v) for n, v in zip(action_names, mae)},
        "rmse_per_dim": {n: float(v) for n, v in zip(action_names, rmse)},
        "mae_joints": float(np.mean(mae[:6])),
        "mae_gripper": float(mae[6]),
        "mae_all": float(np.mean(mae)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        out_dir / "pred_vs_gt.npz",
        pred=pred_actions,
        gt=actions_gt,
        state=states,
        replan_ts=np.asarray(replan_ts, dtype=np.int32),
        action_names=np.asarray(action_names),
    )

    steps = np.arange(length)
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()
    for i, name in enumerate(action_names):
        ax = axes[i]
        ax.plot(steps, actions_gt[:, i], label="GT", color="#1f77b4", linewidth=1.2)
        ax.plot(steps, pred_actions[:, i], label="Pred", color="#d62728", linewidth=1.0, alpha=0.85)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{name}  MAE={mae[i]:.4f}  RMSE={rmse[i]:.4f}")
        if i == 0:
            ax.legend(loc="upper right")
    axes[-1].set_visible(False)
    for ax in axes[-3:-1]:
        ax.set_xlabel("frame")
    fig.suptitle(
        f"Episode {args.episode} open-loop  |  {ckpt_path.name}  |  "
        f"MAE joints={metrics['mae_joints']:.4f} gripper={metrics['mae_gripper']:.4f}",
        fontsize=12,
    )
    fig.tight_layout()
    plot_path = out_dir / "pred_vs_gt.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    print("\n===== MAE / RMSE (physical units) =====")
    for i, name in enumerate(action_names):
        print(f"  {name:8s}  MAE={mae[i]:.6f}  RMSE={rmse[i]:.6f}")
    print(f"  joints   MAE={metrics['mae_joints']:.6f}")
    print(f"  gripper  MAE={metrics['mae_gripper']:.6f}")
    print(f"  all      MAE={metrics['mae_all']:.6f}")
    print(f"\nsaved: {plot_path}")
    print(f"saved: {out_dir / 'metrics.json'}")
    print(f"saved: {out_dir / 'pred_vs_gt.npz'}")


if __name__ == "__main__":
    main()
