#!/usr/bin/env python3
"""Simulate inference-loop FPS mismatch vs training FPS on one episode.

Training samples consecutive frames at ``cfg.fps`` (here 15 Hz):
  obs history = 8 frames spanning (n_obs-1)/fps s
  action chunk = 32 steps spanning n_action_steps/fps s

Action execution stays at train fps (15 Hz): pred[k] vs GT[t+k], replan every
n_action_steps dataset frames. Only the 8-frame observation stack is resampled
to simulate a different camera / inference tick rate.

30 Hz is simulated on 15 Hz data by repeating adjacent frames (no new pixels).

Usage (from va/ with conda env lerobot)::

    PYTHONPATH=. python scripts/eval_episode_fps_mismatch.py \\
      --checkpoint model/.../checkpoint_final.pt \\
      --run-dir data/openarm_hcx_dual_arm_train \\
      --episode 206
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from robotfm.collect.loop import get_run_dir
from robotfm.config import load_config
from robotfm.data.action_delta import (
    denormalize_predicted_action,
    flow_history_from_phys,
    joint_mask_from_names,
)
from robotfm.data.dataset import spatial_preprocess_images
from robotfm.data.lerobot_dataset import (
    _load_image_rgb,
    _short_camera_name,
    load_episode_arrays_from_parquet,
    load_lerobot_info,
)
from robotfm.data.stats import normalize
from robotfm.train import build_policy


def _resolve_train_config(ckpt_path: Path, config_arg: str | None, base_dir: Path) -> Path | None:
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
    return [f"j{i + 1}" for i in range(int(cfg.action_dim) - 1)] + ["gripper"]


def _history_indices(t: int, n_obs: int, stride_num: int, stride_den: int) -> list[int]:
    """Oldest → newest dataset indices for an n_obs stack at given stride."""
    idxs: list[int] = []
    for k in range(n_obs):
        offset = ((n_obs - 1 - k) * stride_num) // stride_den
        idxs.append(max(0, t - offset))
    return idxs


def _align_index(t: int, k: int, stride_num: int, stride_den: int) -> int:
    return t + (k * stride_num) // stride_den


def _chunk_dataset_span(n_action: int, stride_num: int, stride_den: int) -> int:
    return max(1, (n_action * stride_num) // stride_den)


def _preload_episode_images(
    *,
    run_dir: Path,
    image_paths: dict[str, list[str]],
    cameras: list[str],
    length: int,
    pre_crop_size: int | None,
    resize_size: int | None,
    crop_size: int | None,
) -> dict[str, torch.Tensor]:
    """Decode + spatial preprocess every frame once. Returns cam -> (T, 3, H, W)."""
    cached: dict[str, torch.Tensor] = {}
    for cam in cameras:
        frames = []
        for fi in range(length):
            img = _load_image_rgb(run_dir / image_paths[cam][fi])
            img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(img))
        stacked = torch.stack(frames, dim=0)
        stacked = spatial_preprocess_images(
            stacked,
            pre_crop_size=pre_crop_size,
            resize_size=resize_size,
            crop_size=crop_size,
            random_crop=False,
        )
        cached[cam] = stacked.contiguous()
        print(f"  cached {cam}: {tuple(cached[cam].shape)}")
    return cached


def _build_obs_batch_from_cache(
    *,
    image_cache: dict[str, torch.Tensor],
    states: np.ndarray,
    cameras: list[str],
    obs_indices: list[int],
    stats: dict,
    norm_mode: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    camera_histories = [image_cache[cam][obs_indices] for cam in cameras]
    obs_images = torch.stack(camera_histories, dim=0)
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


def _mae_rmse(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    err = pred - gt
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    return mae, rmse


def _run_condition(
    *,
    policy,
    image_cache: dict[str, torch.Tensor],
    states: np.ndarray,
    actions_gt: np.ndarray,
    cameras: list[str],
    stats: dict,
    norm_mode: str,
    n_obs: int,
    n_action_steps: int,
    obs_num: int,
    obs_den: int,
    act_num: int,
    act_den: int,
    delta_joint_mask: np.ndarray,
    predict_joint_delta: bool,
    device: torch.device,
) -> dict:
    length = int(actions_gt.shape[0])
    pred_15hz = np.full_like(actions_gt, np.nan, dtype=np.float32)
    tick_pred: list[np.ndarray] = []
    tick_gt: list[np.ndarray] = []
    tick_k: list[int] = []
    replan_ts: list[int] = []

    t = 0
    while t < length:
        obs_indices = _history_indices(t, n_obs, obs_num, obs_den)
        batch = _build_obs_batch_from_cache(
            image_cache=image_cache,
            states=states,
            cameras=cameras,
            obs_indices=obs_indices,
            stats=stats,
            norm_mode=norm_mode,
            device=device,
        )
        with torch.no_grad():
            pred_norm = policy.sample_actions(batch)[0].cpu()
        q_now = states[t].astype(np.float32)
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

        take = 0
        for k in range(min(n_action_steps, pred_phys.shape[0])):
            gt_i = _align_index(t, k, act_num, act_den)
            if gt_i >= length:
                break
            tick_pred.append(pred_phys[k].astype(np.float32))
            tick_gt.append(actions_gt[gt_i])
            tick_k.append(k)
            t0 = gt_i
            t1 = _align_index(t, k + 1, act_num, act_den)
            if t1 <= t0:
                t1 = t0 + 1
            pred_15hz[t0 : min(t1, length)] = pred_phys[k]
            take = k + 1
        replan_ts.append(t)
        if take == 0:
            break
        t += _chunk_dataset_span(take, act_num, act_den)

    tick_pred_a = np.stack(tick_pred, axis=0) if tick_pred else np.zeros((0, actions_gt.shape[1]), np.float32)
    tick_gt_a = np.stack(tick_gt, axis=0) if tick_gt else np.zeros((0, actions_gt.shape[1]), np.float32)
    tick_k_a = np.asarray(tick_k, dtype=np.int32)
    return {
        "pred_15hz": pred_15hz,
        "tick_pred": tick_pred_a,
        "tick_gt": tick_gt_a,
        "tick_k": tick_k_a,
        "replan_ts": np.asarray(replan_ts, dtype=np.int32),
        "obs_span_s": (n_obs - 1) * (obs_num / obs_den) / 15.0,
        "act_span_s": n_action_steps * (act_num / act_den) / 15.0,
    }


def _summarize(
    result: dict,
    action_names: list[str],
    joint_mask: np.ndarray,
    grip_mask: np.ndarray,
) -> dict:
    pred = result["tick_pred"]
    gt = result["tick_gt"]
    ks = result["tick_k"]
    if pred.shape[0] == 0:
        nan = float("nan")
        return {
            "n_ticks": 0,
            "mae_joints": nan,
            "mae_gripper": nan,
            "mae_all": nan,
            "mae_step0_joints": nan,
            "mae_early_joints": nan,
            "mae_late_joints": nan,
        }
    mae, rmse = _mae_rmse(pred, gt)
    step0 = ks == 0
    early = ks < 8
    late = ks >= 24

    def _joint_mae(mask_rows: np.ndarray) -> float:
        if not mask_rows.any():
            return float("nan")
        err = np.abs(pred[mask_rows] - gt[mask_rows])[:, joint_mask]
        return float(err.mean()) if err.size else float("nan")

    return {
        "n_ticks": int(pred.shape[0]),
        "n_replans": int(result["replan_ts"].shape[0]),
        "obs_span_s": float(result["obs_span_s"]),
        "act_span_s": float(result["act_span_s"]),
        "mae_per_dim": {n: float(v) for n, v in zip(action_names, mae)},
        "rmse_per_dim": {n: float(v) for n, v in zip(action_names, rmse)},
        "mae_joints": float(mae[joint_mask].mean()) if joint_mask.any() else float("nan"),
        "mae_gripper": float(mae[grip_mask].mean()) if grip_mask.any() else float("nan"),
        "mae_all": float(mae.mean()),
        "mae_step0_joints": _joint_mae(step0),
        "mae_early_joints": _joint_mae(early),
        "mae_late_joints": _joint_mae(late),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FPS mismatch open-loop eval")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--episode", type=int, default=206)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = base_dir / ckpt_path

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_cfg_path = _resolve_train_config(ckpt_path, args.config, base_dir)
    if train_cfg_path is None:
        raise FileNotFoundError("need config_source.yaml / config.yaml beside checkpoint")
    cfg = load_config(train_cfg_path)
    # Deterministic open-loop: drop training history noise.
    cfg.policy.history_noise_std = 0.0
    stats = ckpt["stats"]
    norm_mode = cfg.dataset.norm_mode
    n_obs = int(cfg.dataset.n_obs_steps)
    n_action_steps = int(cfg.policy.n_action_steps)
    train_fps = float(cfg.fps)
    action_names = _action_names_from_cfg(cfg)
    cameras = list(cfg.cameras)
    delta_joint_mask = joint_mask_from_names(action_names, int(cfg.action_dim))
    grip_mask = np.array(["gripper" in n.lower() for n in action_names], dtype=bool)
    if not grip_mask.any():
        grip_mask = np.zeros(len(action_names), dtype=bool)
        grip_mask[-1] = True
    joint_mask = ~grip_mask

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = (base_dir / run_dir).resolve()
    else:
        run_dir = get_run_dir(cfg, base_dir)

    info = load_lerobot_info(run_dir)
    features = info["features"]
    cam_feat_keys = [next(k for k in features if _short_camera_name(k) == cam) for cam in cameras]
    payload = load_episode_arrays_from_parquet(run_dir, args.episode, info, cam_feat_keys)
    states = payload["state"]
    actions_gt = payload["action"]
    image_paths = payload["image_paths"]
    length = int(payload["length"])

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else ckpt_path.parent / f"eval_ep{args.episode:04d}_fps_mismatch"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else (cfg.train.device if torch.cuda.is_available() else "cpu"))
    policy = build_policy(cfg, stats)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()

    print(f"checkpoint: {ckpt_path}")
    print(f"train_config: {train_cfg_path}")
    print(f"run_dir: {run_dir}")
    print(f"episode: {args.episode} length={length} train_fps={train_fps} n_obs={n_obs} n_action={n_action_steps}")
    print(f"device={device} history_noise_std=0")
    print("preloading images...")
    image_cache = _preload_episode_images(
        run_dir=run_dir,
        image_paths=image_paths,
        cameras=cameras,
        length=length,
        pre_crop_size=cfg.dataset.pre_crop_size,
        resize_size=cfg.dataset.resize_size,
        crop_size=cfg.dataset.crop_size,
    )

    # Action stride is always 1/1 (15 Hz). Only observation temporal spacing changes.
    conditions = [
        {"name": "15hz_matched", "infer_fps": 15.0, "obs": (1, 1), "act": (1, 1), "mode": "obs"},
        {"name": "7p5hz_obs", "infer_fps": 7.5, "obs": (2, 1), "act": (1, 1), "mode": "obs"},
        {"name": "5hz_obs", "infer_fps": 5.0, "obs": (3, 1), "act": (1, 1), "mode": "obs"},
        {"name": "30hz_obs", "infer_fps": 30.0, "obs": (1, 2), "act": (1, 1), "mode": "obs"},
    ]

    summaries: dict[str, dict] = {}
    pred_15hz_by_name: dict[str, np.ndarray] = {}

    for cond in conditions:
        print(f"\n=== {cond['name']}  infer={cond['infer_fps']} Hz  mode={cond['mode']} ===")
        result = _run_condition(
            policy=policy,
            image_cache=image_cache,
            states=states,
            actions_gt=actions_gt,
            cameras=cameras,
            stats=stats,
            norm_mode=norm_mode,
            n_obs=n_obs,
            n_action_steps=n_action_steps,
            obs_num=cond["obs"][0],
            obs_den=cond["obs"][1],
            act_num=cond["act"][0],
            act_den=cond["act"][1],
            delta_joint_mask=delta_joint_mask,
            predict_joint_delta=bool(cfg.policy.predict_joint_delta),
            device=device,
        )
        summary = _summarize(result, action_names, joint_mask, grip_mask)
        summary.update(
            {
                "name": cond["name"],
                "infer_fps": cond["infer_fps"],
                "mode": cond["mode"],
                "obs_stride": f"{cond['obs'][0]}/{cond['obs'][1]}",
                "act_stride": f"{cond['act'][0]}/{cond['act'][1]}",
            }
        )
        summaries[cond["name"]] = summary
        pred_15hz_by_name[cond["name"]] = result["pred_15hz"]
        print(
            f"  ticks={summary['n_ticks']} replans={summary['n_replans']} "
            f"obs_span={summary['obs_span_s']:.3f}s act_span={summary['act_span_s']:.3f}s"
        )
        print(
            f"  MAE joints={summary['mae_joints']:.5f}  "
            f"step0={summary['mae_step0_joints']:.5f}  "
            f"early={summary['mae_early_joints']:.5f}  "
            f"late={summary['mae_late_joints']:.5f}  "
            f"gripper={summary['mae_gripper']:.5f}"
        )

    metrics = {
        "checkpoint": str(ckpt_path),
        "train_config": str(train_cfg_path),
        "run_dir": str(run_dir),
        "episode": args.episode,
        "length": length,
        "train_fps": train_fps,
        "n_obs_steps": n_obs,
        "n_action_steps": n_action_steps,
        "history_noise_std": 0.0,
        "conditions": summaries,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        out_dir / "pred_vs_gt.npz",
        gt=actions_gt,
        action_names=np.asarray(action_names),
        **{f"pred_{k}": v for k, v in pred_15hz_by_name.items()},
    )

    # MAE bar chart
    order = [c["name"] for c in conditions]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    xs = np.arange(len(order))
    joints = [summaries[n]["mae_joints"] for n in order]
    step0 = [summaries[n]["mae_step0_joints"] for n in order]
    width = 0.38
    ax.bar(xs - width / 2, joints, width, label="chunk MAE (joints)", color="#d62728")
    ax.bar(xs + width / 2, step0, width, label="step-0 MAE (joints)", color="#1f77b4")
    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("MAE (physical joint units)")
    ax.set_title(
        f"Episode {args.episode}: obs-rate mismatch, actions at {train_fps:.0f} Hz"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mae_by_condition.png", dpi=140)
    plt.close(fig)

    plot_dims = [i for i, n in enumerate(action_names) if n in {"L0", "L3", "R0", "R3", "left_gripper", "right_gripper"}]
    if not plot_dims:
        plot_dims = list(range(min(4, len(action_names))))
    steps = np.arange(length)
    fig, axes = plt.subplots(len(plot_dims), 1, figsize=(14, 2.4 * len(plot_dims)), sharex=True)
    axes = np.atleast_1d(axes)
    colors = {
        "15hz_matched": "#1f77b4",
        "7p5hz_obs": "#ff7f0e",
        "5hz_obs": "#d62728",
        "30hz_obs": "#2ca02c",
    }
    for ax, dim_i in zip(axes, plot_dims):
        ax.plot(steps, actions_gt[:, dim_i], color="#444444", linewidth=1.4, label="GT")
        for name in order:
            ax.plot(
                steps,
                pred_15hz_by_name[name][:, dim_i],
                color=colors.get(name, None),
                linewidth=1.0,
                alpha=0.9,
                label=name,
            )
        ax.set_ylabel(action_names[dim_i])
        ax.grid(True, alpha=0.3)
        if dim_i == plot_dims[0]:
            ax.legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("dataset frame (15 Hz execution)")
    fig.suptitle(
        f"Episode {args.episode} obs-rate mismatch, actions executed at {train_fps:.0f} Hz",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "traj_overlay.png", dpi=140)
    plt.close(fig)

    print(f"\nsaved: {out_dir / 'metrics.json'}")
    print(f"saved: {out_dir / 'mae_by_condition.png'}")
    print(f"saved: {out_dir / 'traj_overlay.png'}")
    print(f"saved: {out_dir / 'pred_vs_gt.npz'}")


if __name__ == "__main__":
    main()
