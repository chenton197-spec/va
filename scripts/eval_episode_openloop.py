#!/usr/bin/env python3
"""Open-loop offline eval: predicted vs GT actions on one dataset episode.

训练相关配置优先来自训练 YAML（``--config``，或 checkpoint 同目录的
``config_source.yaml`` / ``config.yaml``），否则回退到 checkpoint 内嵌
config。权重与 stats 仍从 checkpoint 加载。

评估预处理与部署一致：
- resize → fixed center crop（无 random crop / 无 color jitter）
- action chunking 用 policy.n_action_steps / horizon / num_inference_steps

Usage (from va/ with conda env lerobot)::

    PYTHONPATH=. python scripts/eval_episode_openloop.py \\
      --checkpoint outputs/a2a_noise_shine_shoes_limits_s256_xxx/checkpoint_320000.pt \\
      --config configs/shine_shoes_a2a_noise_limits.yaml \\
      --episode 120 --device cpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from robotfm.collect.loop import get_run_dir
from robotfm.config import load_config, resolve_path
from robotfm.data.dataset import spatial_preprocess_images
from robotfm.data.action_delta import (
    denormalize_predicted_action,
    flow_history_from_phys,
    joint_mask_from_names,
)
from robotfm.data.lerobot_dataset import (
    _load_image_rgb,
    _short_camera_name,
    load_episode_arrays_from_parquet,
    load_lerobot_info,
)
from robotfm.data.uint8_cache import Uint8ImageCache, resolve_cache_dir
from robotfm.data.stats import normalize
from robotfm.train import build_policy


def _resolve_train_config(ckpt_path: Path, config_arg: str | None, base_dir: Path) -> Path | None:
    """Resolve training YAML: CLI > config_source.yaml > config.yaml beside ckpt."""
    if config_arg:
        p = Path(config_arg)
        if not p.is_absolute():
            p = base_dir / p
        return p.resolve()
    for name in ("config_source.yaml", "config.yaml"):
        cand = ckpt_path.parent / name
        if cand.is_file():
            return cand
    return None


def _action_names_from_cfg(cfg) -> list[str]:
    names = list(cfg.action_names)
    if len(names) == int(cfg.action_dim):
        return names
    # YAML often omits names; defaults stay PushT ["x","y"] while action_dim=7.
    return [f"j{i + 1}" for i in range(int(cfg.action_dim) - 1)] + ["gripper"]


def _build_obs_batch(
    *,
    run_dir: Path,
    image_paths: dict[str, list[str]],
    states: np.ndarray,
    cameras: list[str],
    t: int,
    n_obs_steps: int,
    pre_crop_size: int | None,
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
    obs_images = spatial_preprocess_images(
        obs_images,
        pre_crop_size=pre_crop_size,
        resize_size=resize_size,
        crop_size=crop_size,
        random_crop=False,
    )

    state = normalize(states[obs_indices].astype(np.float32), stats, prefix="state", mode=norm_mode)
    flow_hist = flow_history_from_phys(
        states[obs_indices].astype(np.float32),
        stats,
        norm_mode,
        action_names=None,
    )
    return {
        "obs_images": obs_images.unsqueeze(0).to(device),
        "obs_state": torch.from_numpy(state).unsqueeze(0).to(device),
        "obs_history": torch.from_numpy(flow_hist).unsqueeze(0).to(device),
    }


def _obs_indices(t: int, n_obs: int) -> list[int]:
    obs_start = max(0, t - n_obs + 1)
    obs_indices = list(range(obs_start, t + 1))
    while len(obs_indices) < n_obs:
        obs_indices.insert(0, obs_indices[0])
    return obs_indices


def _preload_episode_images(
    *,
    run_dir: Path,
    image_paths: dict[str, list[str]],
    cameras: list[str],
    length: int,
    pre_crop_size: int | None,
    resize_size: int | None,
    crop_size: int | None,
    episode: int,
) -> torch.Tensor:
    """Load all episode images once. Returns ``(C, T, 3, H, W)`` uint8 RGB.

    Prefers uint8 cache (already pre_crop + resize); otherwise decodes JPEGs.
    """
    cache_path = None
    if resize_size is not None:
        cache_path = resolve_cache_dir(
            run_dir, resize_size=resize_size, pre_crop_size=pre_crop_size
        )
    if cache_path is not None and (cache_path / "meta.json").is_file():
        cache = Uint8ImageCache(cache_path)
        if episode in cache.episode_ids and all(c in cache.cameras for c in cameras):
            ep_local = cache.episode_ids.index(episode)
            if cache.episode_lengths[ep_local] == length:
                cams = []
                frame_idx = list(range(length))
                for cam in cameras:
                    hwc = cache.load_cam_frames(ep_local, cam, frame_idx)
                    chw = np.ascontiguousarray(np.transpose(hwc, (0, 3, 1, 2)))
                    cams.append(torch.from_numpy(chw))
                images = torch.stack(cams, dim=0)
                if crop_size is not None:
                    images = images.float().div_(255.0)
                    images = spatial_preprocess_images(
                        images,
                        pre_crop_size=None,
                        resize_size=None,
                        crop_size=crop_size,
                        random_crop=False,
                    )
                    images = (images.clamp(0, 1) * 255.0).to(torch.uint8)
                print(f"images: uint8_cache {cache_path} shape={tuple(images.shape)}")
                return images

    cams = []
    for cam in cameras:
        frames = []
        for fi in range(length):
            img = _load_image_rgb(run_dir / image_paths[cam][fi])
            frames.append(torch.from_numpy(np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0))
        stacked = torch.stack(frames, dim=0)
        stacked = spatial_preprocess_images(
            stacked,
            pre_crop_size=pre_crop_size,
            resize_size=resize_size,
            crop_size=crop_size,
            random_crop=False,
        )
        cams.append(stacked.to(torch.uint8) if stacked.dtype == torch.uint8 else (stacked.clamp(0, 1) * 255.0).to(torch.uint8))
    images = torch.stack(cams, dim=0)
    print(f"images: decoded JPEG shape={tuple(images.shape)}")
    return images


def _infer_episode_batched(
    *,
    policy,
    cam_frames: torch.Tensor,
    states: np.ndarray,
    stats: dict,
    norm_mode: str,
    n_obs: int,
    n_action_steps: int,
    exec_steps: int,
    delta_joint_mask: np.ndarray,
    predict_joint_delta: bool,
    device: torch.device,
    batch_size: int,
    reanchor: bool,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """GPU batched open-loop. ``exec_steps=1`` infers every frame (uses pred[0])."""
    length = int(states.shape[0])
    action_dim = int(states.shape[1])
    infer_ts = list(range(length)) if exec_steps == 1 else list(range(0, length, exec_steps))
    n_infer = len(infer_ts)
    obs_idx = np.array([_obs_indices(t, n_obs) for t in infer_ts], dtype=np.int64)
    pred_raw = np.full((length, action_dim), np.nan, dtype=np.float32)
    pred = np.full((length, action_dim), np.nan, dtype=np.float32)
    cam_frames = cam_frames.contiguous()
    bs = max(int(batch_size), 1)
    for start in range(0, n_infer, bs):
        sl = slice(start, min(start + bs, n_infer))
        ts = infer_ts[sl]
        idx = obs_idx[sl]
        # cam_frames: (C, T, 3, H, W); idx: (B, n_obs) → (B, C, n_obs, 3, H, W)
        obs_images = cam_frames[:, torch.as_tensor(idx)].permute(1, 0, 2, 3, 4, 5)
        obs_images = obs_images.to(device, non_blocking=True)
        if obs_images.dtype == torch.uint8:
            obs_images = obs_images.float().div_(255.0)
        state_win = states[idx].astype(np.float32)
        state_norm = normalize(state_win, stats, prefix="state", mode=norm_mode)
        if predict_joint_delta:
            hist = np.stack(
                [
                    flow_history_from_phys(
                        states[row].astype(np.float32),
                        stats,
                        norm_mode,
                        predict_joint_delta=True,
                        joint_mask=delta_joint_mask,
                    )
                    for row in idx
                ],
                axis=0,
            )
        else:
            hist = state_norm
        batch = {
            "obs_images": obs_images,
            "obs_state": torch.from_numpy(state_norm).to(device, non_blocking=True),
            "obs_history": torch.from_numpy(np.ascontiguousarray(hist)).to(
                device, non_blocking=True
            ),
        }
        with torch.no_grad():
            pred_norm = policy.sample_actions(batch)[:, :n_action_steps].float().cpu()
        q_now = states[np.asarray(ts, dtype=np.int64)].astype(np.float32)
        pred_phys = np.asarray(
            denormalize_predicted_action(
                pred_norm,
                stats,
                norm_mode,
                q_now_phys=q_now,
                predict_joint_delta=predict_joint_delta,
                joint_mask=delta_joint_mask,
            )
        )
        for i, t in enumerate(ts):
            take = min(exec_steps, length - t, pred_phys.shape[1])
            chunk_raw = np.asarray(pred_phys[i, :take], dtype=np.float32)
            pred_raw[t : t + take] = chunk_raw
            if reanchor:
                chunk = chunk_raw.copy()
                offset = q_now[i, delta_joint_mask] - chunk[0, delta_joint_mask]
                chunk[:, delta_joint_mask] = chunk[:, delta_joint_mask] + offset
                pred[t : t + take] = chunk
            else:
                pred[t : t + take] = chunk_raw
    return pred, pred_raw, infer_ts


def main() -> None:
    parser = argparse.ArgumentParser(description="Open-loop episode action comparison")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="训练配置 YAML（与训练一致）。默认用 checkpoint 同目录 config_source.yaml/config.yaml，再回退 checkpoint 内嵌 config",
    )
    parser.add_argument("--episode", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. cpu / cuda). Default: config train.device if CUDA available else cpu",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: <ckpt_dir>/eval_epXXXX)",
    )
    parser.add_argument(
        "--exec-steps",
        type=int,
        default=None,
        help="Only execute the first N steps of each predicted chunk before replanning "
        "(default: policy.n_action_steps). Model still predicts full n_action_steps. "
        "Use 1 to infer every frame.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="GPU batched inference size (uint8 cache / preloaded frames). 1 = sequential.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the policy on CUDA (first batch slower, then faster).",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override flow-matching Euler steps at inference "
        "(default: policy.num_inference_steps).",
    )
    parser.add_argument(
        "--history-noise-std",
        type=float,
        default=None,
        help="Gaussian noise std added to obs_state (flow start) at inference. "
        "Default: training policy.history_noise_std.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Override LeRobot dataset root (default: data_root / dataset.run_name).",
    )
    parser.add_argument(
        "--reanchor-pred0-to-qnow",
        action="store_true",
        help="Inference-time pseudo-delta: per chunk, joints become "
        "pred - pred[0] + q_now. Grippers stay absolute. Does not require "
        "predict_joint_delta training.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = base_dir / ckpt_path

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_cfg_path = _resolve_train_config(ckpt_path, args.config, base_dir)
    if train_cfg_path is not None:
        if not train_cfg_path.is_file():
            raise FileNotFoundError(f"找不到训练配置: {train_cfg_path}")
        cfg = load_config(train_cfg_path)
        print(f"train_config: {train_cfg_path}")
    else:
        cfg = ckpt["config"]
        print("train_config: checkpoint embedded config")
    stats = ckpt["stats"]
    norm_mode = cfg.dataset.norm_mode
    n_obs = cfg.dataset.n_obs_steps
    horizon = cfg.dataset.horizon
    n_action_steps = int(cfg.policy.n_action_steps)
    exec_steps = int(args.exec_steps) if args.exec_steps is not None else n_action_steps
    if exec_steps <= 0:
        raise ValueError(f"--exec-steps must be > 0, got {exec_steps}")
    if exec_steps > n_action_steps:
        raise ValueError(
            f"--exec-steps ({exec_steps}) cannot exceed policy.n_action_steps ({n_action_steps})"
        )
    if args.num_inference_steps is not None:
        if int(args.num_inference_steps) <= 0:
            raise ValueError(
                f"--num-inference-steps must be > 0, got {args.num_inference_steps}"
            )
        cfg.policy.num_inference_steps = int(args.num_inference_steps)
    if args.history_noise_std is not None:
        if float(args.history_noise_std) < 0:
            raise ValueError(
                f"--history-noise-std must be >= 0, got {args.history_noise_std}"
            )
        cfg.policy.history_noise_std = float(args.history_noise_std)
    num_inference_steps = int(cfg.policy.num_inference_steps)
    history_noise_std = float(cfg.policy.history_noise_std)
    action_names = _action_names_from_cfg(cfg)
    cameras = list(cfg.cameras)
    delta_joint_mask = joint_mask_from_names(action_names, int(cfg.action_dim))
    reanchor = bool(args.reanchor_pred0_to_qnow)

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = (base_dir / run_dir).resolve()
    else:
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

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    policy = build_policy(cfg, stats)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if args.compile:
        if device.type != "cuda":
            print("warning: --compile ignored (not CUDA)")
        else:
            policy = torch.compile(policy, mode="default")
            print("train.compile: True (mode=default)")

    print(f"checkpoint: {ckpt_path}")
    print(f"run_dir: {run_dir}")
    print(f"episode: {args.episode} length={length}")
    print(
        f"norm_mode={norm_mode} n_obs={n_obs} horizon={horizon} "
        f"n_action_steps={n_action_steps} exec_steps={exec_steps} "
        f"batch_size={int(args.batch_size)} "
        f"num_inference_steps={num_inference_steps} "
        f"history_noise_std={history_noise_std} "
        f"predict_joint_delta={bool(cfg.policy.predict_joint_delta)} "
        f"reanchor_pred0_to_qnow={reanchor}"
    )
    print(
        f"pre_crop={cfg.dataset.pre_crop_size} resize={cfg.dataset.resize_size} "
        f"crop={cfg.dataset.crop_size} eval_fixed_crop={cfg.dataset.eval_fixed_crop} "
        f"device={device}"
    )

    t_img = time.perf_counter()
    cam_frames = _preload_episode_images(
        run_dir=run_dir,
        image_paths=image_paths,
        cameras=cameras,
        length=length,
        pre_crop_size=cfg.dataset.pre_crop_size,
        resize_size=cfg.dataset.resize_size,
        crop_size=cfg.dataset.crop_size if cfg.dataset.eval_fixed_crop else None,
        episode=args.episode,
    )
    print(f"preload_images: {time.perf_counter() - t_img:.2f}s")

    t_inf = time.perf_counter()
    pred_actions, pred_actions_raw, replan_ts = _infer_episode_batched(
        policy=policy,
        cam_frames=cam_frames,
        states=states,
        stats=stats,
        norm_mode=norm_mode,
        n_obs=n_obs,
        n_action_steps=n_action_steps,
        exec_steps=exec_steps,
        delta_joint_mask=delta_joint_mask,
        predict_joint_delta=bool(cfg.policy.predict_joint_delta),
        device=device,
        batch_size=int(args.batch_size),
        reanchor=reanchor,
    )
    infer_s = time.perf_counter() - t_inf
    print(
        f"infer: {infer_s:.2f}s  {len(replan_ts) / max(infer_s, 1e-6):.1f} frames/s  "
        f"n_replans={len(replan_ts)}"
    )

    def _mae_rmse(pred_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid_m = np.isfinite(pred_arr).all(axis=1)
        err_m = pred_arr[valid_m] - actions_gt[valid_m]
        return np.mean(np.abs(err_m), axis=0), np.sqrt(np.mean(err_m**2, axis=0))

    valid = np.isfinite(pred_actions).all(axis=1)
    gt = actions_gt[valid]
    pred = pred_actions[valid]
    err = pred - gt
    mae, rmse = _mae_rmse(pred_actions)
    mae_raw, rmse_raw = _mae_rmse(pred_actions_raw) if reanchor else (None, None)

    # Dual-arm 16-D: [L0..L6, R0..R6, left_gripper, right_gripper]
    # Single-arm 7-D: [j1..j6, gripper]
    gripper_mask = np.array(["gripper" in n.lower() for n in action_names], dtype=bool)
    if not gripper_mask.any() and len(mae) >= 7:
        gripper_mask = np.zeros(len(mae), dtype=bool)
        gripper_mask[-1] = True
    joint_mask = ~gripper_mask
    mae_joints = float(np.mean(mae[joint_mask])) if joint_mask.any() else float("nan")
    mae_gripper = float(np.mean(mae[gripper_mask])) if gripper_mask.any() else float("nan")

    metrics = {
        "checkpoint": str(ckpt_path),
        "train_config": str(train_cfg_path) if train_cfg_path is not None else None,
        "episode": args.episode,
        "length": length,
        "n_valid_steps": int(valid.sum()),
        "n_replans": len(replan_ts),
        "policy_type": cfg.policy.type,
        "norm_mode": norm_mode,
        "n_obs_steps": n_obs,
        "horizon": horizon,
        "n_action_steps": n_action_steps,
        "exec_steps": exec_steps,
        "num_inference_steps": num_inference_steps,
        "history_noise_std": history_noise_std,
        "predict_joint_delta": bool(cfg.policy.predict_joint_delta),
        "reanchor_pred0_to_qnow": reanchor,
        "run_dir": str(run_dir),
        "pre_crop_size": cfg.dataset.pre_crop_size,
        "resize_size": cfg.dataset.resize_size,
        "crop_size": cfg.dataset.crop_size,
        "run_name": cfg.dataset.run_name,
        "seed": args.seed,
        "mae_per_dim": {n: float(v) for n, v in zip(action_names, mae)},
        "rmse_per_dim": {n: float(v) for n, v in zip(action_names, rmse)},
        "mae_joints": mae_joints,
        "mae_gripper": mae_gripper,
        "mae_all": float(np.mean(mae)),
    }
    if reanchor and mae_raw is not None:
        metrics["mae_joints_raw"] = float(np.mean(mae_raw[joint_mask])) if joint_mask.any() else float("nan")
        metrics["mae_gripper_raw"] = (
            float(np.mean(mae_raw[gripper_mask])) if gripper_mask.any() else float("nan")
        )
        metrics["mae_all_raw"] = float(np.mean(mae_raw))
        metrics["mae_per_dim_raw"] = {n: float(v) for n, v in zip(action_names, mae_raw)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    npz_kw = dict(
        pred=pred_actions,
        gt=actions_gt,
        state=states,
        replan_ts=np.asarray(replan_ts, dtype=np.int32),
        action_names=np.asarray(action_names),
    )
    if reanchor:
        npz_kw["pred_raw"] = pred_actions_raw
    np.savez_compressed(out_dir / "pred_vs_gt.npz", **npz_kw)

    steps = np.arange(length)
    n_dim = len(action_names)
    ncols = 2
    nrows = int(np.ceil(n_dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(3.0 * nrows, 6.0)), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for i, name in enumerate(action_names):
        ax = axes[i]
        ax.plot(steps, actions_gt[:, i], label="GT", color="#1f77b4", linewidth=1.2)
        if reanchor:
            ax.plot(
                steps,
                pred_actions_raw[:, i],
                label="Pred raw",
                color="#7f7f7f",
                linewidth=0.9,
                alpha=0.7,
                linestyle="--",
            )
            ax.plot(
                steps,
                pred_actions[:, i],
                label="Pred Δq+q_now",
                color="#d62728",
                linewidth=1.0,
                alpha=0.9,
            )
        else:
            ax.plot(steps, pred_actions[:, i], label="Pred", color="#d62728", linewidth=1.0, alpha=0.85)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{name}  MAE={mae[i]:.4f}  RMSE={rmse[i]:.4f}")
        if i == 0:
            ax.legend(loc="upper right")
    for j in range(n_dim, len(axes)):
        axes[j].set_visible(False)
    for ax in axes[max(0, n_dim - ncols) : n_dim]:
        ax.set_xlabel("frame")
    fig.suptitle(
        f"Episode {args.episode} open-loop  |  {ckpt_path.name}  |  "
        f"fm={num_inference_steps} exec={exec_steps}/{n_action_steps}"
        f"{'  |  reanchor pred-pred[0]+q_now' if reanchor else ''}  |  "
        f"MAE joints={metrics['mae_joints']:.4f} gripper={metrics['mae_gripper']:.4f}",
        fontsize=12,
    )
    fig.tight_layout()
    plot_path = out_dir / "pred_vs_gt.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    # Left / right arm joint deviation: x=frame, y=pred-GT (physical units)
    err_full = pred_actions - actions_gt
    left_idx = [i for i, n in enumerate(action_names) if n.startswith("L") and n[1:].isdigit()]
    right_idx = [i for i, n in enumerate(action_names) if n.startswith("R") and n[1:].isdigit()]
    if left_idx or right_idx:
        fig_lr, axes_lr = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
        arm_specs = [
            (left_idx, "Left arm", axes_lr[0]),
            (right_idx, "Right arm", axes_lr[1]),
        ]
        cmap = plt.cm.tab10
        for joint_indices, title, ax in arm_specs:
            if not joint_indices:
                ax.set_visible(False)
                continue
            for j, dim_i in enumerate(joint_indices):
                ax.plot(
                    steps,
                    err_full[:, dim_i],
                    label=action_names[dim_i],
                    color=cmap(j % 10),
                    linewidth=1.0,
                    alpha=0.9,
                )
            ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_title(title)
            ax.set_xlabel("frame")
            ax.set_ylabel("joint angle deviation (pred - GT)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8, ncol=2)
        fig_lr.suptitle(
            f"Episode {args.episode} joint deviation"
            f"{'  (reanchor pred-pred[0]+q_now)' if reanchor else ''}  |  {ckpt_path.name}",
            fontsize=12,
        )
        fig_lr.tight_layout()
        dev_plot_path = out_dir / "joint_deviation_lr.png"
        fig_lr.savefig(dev_plot_path, dpi=140)
        plt.close(fig_lr)

    print("\n===== MAE / RMSE (physical units) =====")
    if reanchor and mae_raw is not None:
        print("  (Pred = reanchor pred-pred[0]+q_now; raw = absolute decode)")
    for i, name in enumerate(action_names):
        extra = ""
        if reanchor and mae_raw is not None:
            extra = f"  raw_MAE={mae_raw[i]:.6f}"
        print(f"  {name:8s}  MAE={mae[i]:.6f}  RMSE={rmse[i]:.6f}{extra}")
    print(f"  joints   MAE={metrics['mae_joints']:.6f}")
    if reanchor and "mae_joints_raw" in metrics:
        print(f"  joints   MAE_raw={metrics['mae_joints_raw']:.6f}")
    print(f"  gripper  MAE={metrics['mae_gripper']:.6f}")
    print(f"  all      MAE={metrics['mae_all']:.6f}")
    if reanchor and "mae_all_raw" in metrics:
        print(f"  all      MAE_raw={metrics['mae_all_raw']:.6f}")
    print(f"\nsaved: {plot_path}")
    if left_idx or right_idx:
        print(f"saved: {dev_plot_path}")
    print(f"saved: {out_dir / 'metrics.json'}")
    print(f"saved: {out_dir / 'pred_vs_gt.npz'}")


if __name__ == "__main__":
    main()
