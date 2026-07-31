#!/usr/bin/env python3
"""Open-loop pred vs GT on one shine_shoes episode (joints + gripper).

Matches training/eval preprocessing:
  - run ``config.yaml`` (n_obs/horizon/resize/crop/norm_mode/A2A hyperparams)
  - checkpoint ``stats`` + ``limits`` denorm
  - center crop (``eval_fixed_crop``), no color jitter
  - ``policy.sample_actions`` with the same ``num_inference_steps``

Usage (from va/, conda env lerobot)::

    PYTHONPATH=. python scripts/eval_shine_shoes_openloop.py \\
      --checkpoint outputs/a2a_noise_shine_shoes_limits_260730175409/checkpoint_070000.pt \\
      --episode 5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from robotfm.collect.loop import get_run_dir
from robotfm.config import load_config
from robotfm.data.dataset import crop_images
from robotfm.data.lerobot_dataset import (
    _load_image_rgb,
    load_episode_arrays_from_parquet,
    load_lerobot_info,
)
from robotfm.data.stats import denormalize, normalize
from robotfm.train import build_policy

JOINT_NAMES = [f"j{i}" for i in range(6)] + ["gripper"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open-loop shine_shoes checkpoint check")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "outputs/a2a_noise_shine_shoes_limits_260730175409/checkpoint_070000.pt"
        ),
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Defaults to <checkpoint_dir>/config.yaml",
    )
    p.add_argument("--episode", type=int, default=5)
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Evaluate every N frames (pred uses first action of the chunk).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on number of evaluated frames (after stride).",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-history-noise",
        action="store_true",
        help="Zero history_noise_std for deterministic decode (not training-default).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Defaults to <checkpoint_dir>/openloop_ep{episode}",
    )
    return p.parse_args()


def _obs_indices(t: int, n_obs: int) -> list[int]:
    start = max(0, t - n_obs + 1)
    idxs = list(range(start, t + 1))
    while len(idxs) < n_obs:
        idxs.insert(0, idxs[0])
    return idxs


def _build_batch(
    *,
    payload: dict,
    run_dir: Path,
    t: int,
    n_obs: int,
    horizon: int,
    resize_size: int | None,
    crop_size: int | None,
    stats: dict,
    norm_mode: str,
    camera_names: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Mirror LeRobotImageSequenceDataset.__getitem__ with eval crop (no jitter)."""
    length = int(payload["length"])
    state_all = payload["state"]
    action_all = payload["action"]
    image_paths = payload["image_paths"]

    obs_idxs = _obs_indices(t, n_obs)
    images = []
    for cam in camera_names:
        frames = []
        for fi in obs_idxs:
            img = _load_image_rgb(run_dir / image_paths[cam][fi], resize_size=resize_size)
            frames.append(np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0)
        images.append(torch.from_numpy(np.stack(frames, axis=0)))
    obs_images = torch.stack(images, dim=0)
    obs_images = crop_images(obs_images, crop_size, random=False)

    state = normalize(
        state_all[obs_idxs].astype(np.float32), stats, prefix="state", mode=norm_mode
    )

    action_end = min(t + horizon, length)
    valid_len = action_end - t
    actions = action_all[t:action_end].astype(np.float32)
    mask = np.zeros((horizon, 1), dtype=np.float32)
    mask[:valid_len] = 1.0
    if valid_len < horizon:
        pad = np.zeros((horizon - valid_len, actions.shape[1]), dtype=np.float32)
        actions = np.concatenate([actions, pad], axis=0)
    actions = normalize(actions, stats, prefix="action", mode=norm_mode)

    return {
        "obs_images": obs_images.unsqueeze(0).to(device),
        "obs_state": torch.from_numpy(state).unsqueeze(0).to(device),
        "action": torch.from_numpy(actions).unsqueeze(0).to(device),
        "action_mask": torch.from_numpy(mask).unsqueeze(0).to(device),
    }


def _mae(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(a - b), axis=0)


def main() -> int:
    args = _parse_args()
    base_dir = Path(__file__).resolve().parents[1]

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = base_dir / ckpt_path
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    cfg_path = Path(args.config) if args.config else ckpt_path.parent / "config.yaml"
    if not cfg_path.is_absolute():
        cfg_path = base_dir / cfg_path
    cfg = load_config(cfg_path)

    out_dir = Path(args.out_dir) if args.out_dir else (
        ckpt_path.parent / f"openloop_ep{args.episode:03d}"
    )
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device_str = args.device or cfg.train.device
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

    print(f"checkpoint: {ckpt_path}")
    print(f"config:     {cfg_path}")
    print(f"device:     {device}")
    print(
        f"policy={cfg.policy.type} n_obs={cfg.dataset.n_obs_steps} "
        f"horizon={cfg.dataset.horizon} n_action_steps={cfg.policy.n_action_steps} "
        f"resize={cfg.dataset.resize_size} crop={cfg.dataset.crop_size} "
        f"norm={cfg.dataset.norm_mode} infer_steps={cfg.policy.num_inference_steps} "
        f"history_noise_std={cfg.policy.history_noise_std}"
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stats = ckpt["stats"]
    if args.no_history_noise:
        cfg.policy.history_noise_std = 0.0
        print("history_noise_std forced to 0")

    policy = build_policy(cfg, stats)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()

    run_dir = get_run_dir(cfg, base_dir)
    info = load_lerobot_info(run_dir)
    camera_keys = [f"observation.images.{c}" for c in cfg.cameras]
    payload = load_episode_arrays_from_parquet(
        run_dir, args.episode, info, camera_keys
    )
    length = int(payload["length"])
    print(f"episode={args.episode} length={length} run_dir={run_dir}")

    n_obs = cfg.dataset.n_obs_steps
    horizon = cfg.dataset.horizon
    n_act = cfg.policy.n_action_steps
    norm_mode = cfg.dataset.norm_mode
    action_dim = cfg.action_dim

    # Open-loop: at each t, take first predicted action of the chunk.
    ts: list[int] = []
    pred_step0: list[np.ndarray] = []
    gt_step0: list[np.ndarray] = []
    chunk_maes: list[np.ndarray] = []

    frame_ids = list(range(0, length, max(1, args.stride)))
    if args.max_frames is not None:
        frame_ids = frame_ids[: args.max_frames]

    for t in tqdm(frame_ids, desc=f"ep{args.episode} open-loop"):
        batch = _build_batch(
            payload=payload,
            run_dir=run_dir,
            t=t,
            n_obs=n_obs,
            horizon=horizon,
            resize_size=cfg.dataset.resize_size,
            crop_size=cfg.dataset.crop_size,
            stats=stats,
            norm_mode=norm_mode,
            camera_names=list(cfg.cameras),
            device=device,
        )
        with torch.no_grad():
            pred_norm = policy.sample_actions(batch)[0].detach().cpu().numpy()  # (H, A)

        gt_norm = batch["action"][0].detach().cpu().numpy()
        mask = batch["action_mask"][0].detach().cpu().numpy().reshape(-1)
        valid = int(mask[:n_act].sum())
        if valid <= 0:
            continue

        pred = denormalize(pred_norm, stats, prefix="action", mode=norm_mode)
        gt = denormalize(gt_norm, stats, prefix="action", mode=norm_mode)

        ts.append(t)
        pred_step0.append(pred[0].astype(np.float64))
        gt_step0.append(gt[0].astype(np.float64))
        chunk_maes.append(np.mean(np.abs(pred[:valid] - gt[:valid]), axis=0))

    pred_arr = np.stack(pred_step0, axis=0)
    gt_arr = np.stack(gt_step0, axis=0)
    t_arr = np.asarray(ts, dtype=np.int64)
    step0_mae = _mae(pred_arr, gt_arr)
    chunk_mae = np.mean(np.stack(chunk_maes, axis=0), axis=0)

    print("\n=== Open-loop MAE (physical units; joints deg, gripper [0,1]) ===")
    print(f"{'dim':>10}  {'step0_mae':>12}  {'chunk_mae':>12}")
    for i, name in enumerate(JOINT_NAMES[:action_dim]):
        print(f"{name:>10}  {step0_mae[i]:12.4f}  {chunk_mae[i]:12.4f}")
    print(f"{'joints':>10}  {step0_mae[:6].mean():12.4f}  {chunk_mae[:6].mean():12.4f}")
    print(f"{'all':>10}  {step0_mae.mean():12.4f}  {chunk_mae.mean():12.4f}")

    # CSV
    csv_path = out_dir / "pred_vs_gt_step0.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["t"]
        for name in JOINT_NAMES[:action_dim]:
            header += [f"gt_{name}", f"pred_{name}", f"err_{name}"]
        w.writerow(header)
        for i, t in enumerate(t_arr):
            row = [int(t)]
            for d in range(action_dim):
                g, p = gt_arr[i, d], pred_arr[i, d]
                row += [f"{g:.6f}", f"{p:.6f}", f"{(p - g):.6f}"]
            w.writerow(row)
    print(f"wrote {csv_path}")

    # Plots: 6 joints + gripper
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()
    for d in range(action_dim):
        ax = axes[d]
        ax.plot(t_arr, gt_arr[:, d], label="GT", linewidth=1.5)
        ax.plot(t_arr, pred_arr[:, d], label="Pred", linewidth=1.2, alpha=0.85)
        ax.set_ylabel(JOINT_NAMES[d])
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(loc="upper right")
    for ax in axes[action_dim:]:
        ax.axis("off")
    axes[min(action_dim, len(axes)) - 1].set_xlabel("frame")
    if action_dim >= 2:
        axes[action_dim - 2].set_xlabel("frame")
    fig.suptitle(
        f"Open-loop ep{args.episode}  ckpt={ckpt_path.name}  "
        f"step0 joint MAE={step0_mae[:6].mean():.3f}  grip MAE={step0_mae[6]:.4f}",
        fontsize=12,
    )
    fig.tight_layout()
    plot_path = out_dir / "pred_vs_gt_step0.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    print(f"wrote {plot_path}")

    # Error plot
    err = pred_arr - gt_arr
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for d in range(6):
        axes[0].plot(t_arr, err[:, d], label=JOINT_NAMES[d], linewidth=1.0)
    axes[0].set_ylabel("joint err (deg)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t_arr, err[:, 6], color="C3", label="gripper")
    axes[1].set_ylabel("gripper err")
    axes[1].set_xlabel("frame")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Open-loop error  ep{args.episode}  {ckpt_path.name}")
    fig.tight_layout()
    err_path = out_dir / "pred_vs_gt_error.png"
    fig.savefig(err_path, dpi=140)
    plt.close(fig)
    print(f"wrote {err_path}")

    metrics_path = out_dir / "metrics.txt"
    with metrics_path.open("w") as f:
        f.write(f"checkpoint: {ckpt_path}\n")
        f.write(f"config: {cfg_path}\n")
        f.write(f"episode: {args.episode}\n")
        f.write(f"n_frames: {len(t_arr)} stride={args.stride}\n")
        f.write(f"history_noise_std: {cfg.policy.history_noise_std}\n")
        f.write(f"seed: {args.seed}\n")
        for i, name in enumerate(JOINT_NAMES[:action_dim]):
            f.write(f"step0_mae_{name}={step0_mae[i]:.6f}\n")
            f.write(f"chunk_mae_{name}={chunk_mae[i]:.6f}\n")
        f.write(f"step0_mae_joints={step0_mae[:6].mean():.6f}\n")
        f.write(f"step0_mae_all={step0_mae.mean():.6f}\n")
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
