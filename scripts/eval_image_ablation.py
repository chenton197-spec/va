#!/usr/bin/env python3
"""Ablate vision at inference: real images vs zeros vs shuffled images.

Reports physical-unit action MAE vs GT, and how much predictions change
when images are dropped (same proprio history, no history noise).

Usage (from va/ with conda env lerobot)::

    PYTHONPATH=. python scripts/eval_image_ablation.py \\
      --checkpoint model/dadi/outputs/.../checkpoint_latest.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from robotfm.collect.loop import get_run_dir
from robotfm.config import load_config
from robotfm.data.action_delta import (
    add_joint_pose,
    denormalize_predicted_action,
    joint_mask_from_names,
)
from robotfm.data.dataset import build_episode_dataset, images_to_float01
from robotfm.data.stats import denormalize
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


def _action_names(cfg) -> list[str]:
    names = list(cfg.action_names)
    if len(names) == int(cfg.action_dim):
        return names
    return [f"j{i + 1}" for i in range(int(cfg.action_dim) - 1)] + ["gripper"]


def _masks(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    grip = np.array(["gripper" in n.lower() for n in names], dtype=bool)
    if not grip.any() and len(names) >= 7:
        grip = np.zeros(len(names), dtype=bool)
        grip[-1] = True
    return ~grip, grip


def _mae_report(err: np.ndarray, names: list[str], joint_mask: np.ndarray, grip_mask: np.ndarray) -> dict:
    """err: (N, T, A) physical units."""
    abs_e = np.abs(err)
    mae_dim = abs_e.mean(axis=(0, 1))
    out = {
        "mae_per_dim": {n: float(v) for n, v in zip(names, mae_dim)},
        "mae_joints": float(mae_dim[joint_mask].mean()) if joint_mask.any() else float("nan"),
        "mae_gripper": float(mae_dim[grip_mask].mean()) if grip_mask.any() else float("nan"),
        "mae_all": float(mae_dim.mean()),
        "rmse_all": float(np.sqrt((err**2).mean())),
    }
    return out


def _print_mae(title: str, metrics: dict) -> None:
    print(f"\n===== {title} =====")
    for name, val in metrics["mae_per_dim"].items():
        print(f"  {name:16s}  MAE={val:.6f}")
    print(f"  {'joints':16s}  MAE={metrics['mae_joints']:.6f}")
    print(f"  {'gripper':16s}  MAE={metrics['mae_gripper']:.6f}")
    print(f"  {'all':16s}  MAE={metrics['mae_all']:.6f}  RMSE={metrics['rmse_all']:.6f}")


@torch.no_grad()
def _img_feat(policy, images: torch.Tensor) -> torch.Tensor:
    return policy.encoder._encode_images(images)


@torch.no_grad()
def _cond_from_img_feat(policy, img_feat: torch.Tensor, obs_state: torch.Tensor) -> torch.Tensor:
    enc = policy.encoder
    b, t, _ = obs_state.shape
    state_feat = enc.state_encoder(obs_state.reshape(b * t, -1)).reshape(b, -1)
    if img_feat.shape[0] == 1 and b > 1:
        img_feat = img_feat.expand(b, -1)
    fused = torch.cat([img_feat, state_feat], dim=-1)
    return policy.obs_projector(enc.proj(fused))


@torch.no_grad()
def _decode(policy, history: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    z = policy.flow_matcher.sample(
        policy.flow_net,
        shape=history.shape,
        device=history.device,
        num_steps=policy.cfg.num_inference_steps,
        start=history,
        global_cond=cond,
        return_traces=False,
    )
    return policy.action_decoder(z)


def _fusion_diagnostics(policy) -> dict:
    enc = policy.encoder
    w0 = enc.proj[0].weight.detach().cpu()  # (cond_dim, fused_dim)
    img_dim = policy.cfg.num_cameras * 128
    img_rms = float(w0[:, :img_dim].pow(2).mean().sqrt())
    state_rms = float(w0[:, img_dim:].pow(2).mean().sqrt())
    adaln = []
    for i, layer in enumerate(policy.flow_net.layers):
        last = layer.time_modulator[-1]
        adaln.append(
            {
                "layer": i,
                "weight_rms": float(last.weight.detach().pow(2).mean().sqrt()),
                "bias_abs_mean": float(last.bias.detach().abs().mean()),
            }
        )
    cond_w = policy.flow_net.cond_embed.weight.detach()
    return {
        "fusion_linear0_img_weight_rms": img_rms,
        "fusion_linear0_state_weight_rms": state_rms,
        "fusion_img_over_state_rms_ratio": img_rms / max(state_rms, 1e-12),
        "flow_cond_embed_weight_rms": float(cond_w.pow(2).mean().sqrt()),
        "adaln_time_modulator": adaln,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference ablation: drop / shuffle images")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-state-samples", type=int, default=256)
    parser.add_argument("--n-image-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Default: <ckpt_dir>/eval_image_ablation.json",
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
        cfg = load_config(train_cfg_path)
        print(f"train_config: {train_cfg_path}")
    else:
        cfg = ckpt["config"]
        print("train_config: checkpoint embedded config")

    stats = ckpt["stats"]
    names = _action_names(cfg)
    joint_mask, grip_mask = _masks(names)
    n_obs = int(cfg.dataset.n_obs_steps)
    n_act = int(cfg.policy.n_action_steps)
    h = int(cfg.dataset.resize_size or 512)
    n_cam = len(cfg.cameras)
    predict_joint_delta = bool(cfg.policy.predict_joint_delta)
    delta_joint_mask = joint_mask_from_names(names, int(cfg.action_dim))

    def _abs_from_norm(pred_norm: torch.Tensor, obs_state: torch.Tensor) -> torch.Tensor:
        q_now = denormalize(
            obs_state[:, -1], stats, prefix="state", mode=cfg.dataset.norm_mode
        )
        return denormalize_predicted_action(
            pred_norm,
            stats,
            cfg.dataset.norm_mode,
            q_now_phys=q_now,
            predict_joint_delta=predict_joint_delta,
            joint_mask=delta_joint_mask,
        )

    def _gt_abs_from_norm(gt_norm: torch.Tensor, obs_state: torch.Tensor) -> torch.Tensor:
        gt_phys = denormalize(gt_norm, stats, prefix="action", mode=cfg.dataset.norm_mode)
        if not predict_joint_delta:
            return gt_phys
        q_now = denormalize(
            obs_state[:, -1], stats, prefix="state", mode=cfg.dataset.norm_mode
        )
        return add_joint_pose(gt_phys, q_now, delta_joint_mask)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] / (1024**2)
        device = torch.device("cuda" if free >= 2500 else "cpu")
        print(f"cuda_free_mib={free:.0f} -> device={device}")
    else:
        device = torch.device("cpu")

    policy = build_policy(cfg, stats)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.cfg.history_noise_std = 0.0
    policy.to(device)
    policy.eval()

    diag = _fusion_diagnostics(policy)
    print("\n===== encoder / flow diagnostics =====")
    print(f"  fusion img   weight RMS = {diag['fusion_linear0_img_weight_rms']:.6f}")
    print(f"  fusion state weight RMS = {diag['fusion_linear0_state_weight_rms']:.6f}")
    print(f"  img/state RMS ratio     = {diag['fusion_img_over_state_rms_ratio']:.4f}")
    print(f"  flow cond_embed RMS     = {diag['flow_cond_embed_weight_rms']:.6f}")
    for row in diag["adaln_time_modulator"]:
        print(
            f"  AdaLN L{row['layer']}  w_rms={row['weight_rms']:.6f}  "
            f"|bias|={row['bias_abs_mean']:.6f}"
        )

    run_dir = get_run_dir(cfg, base_dir)
    print(f"\ncheckpoint: {ckpt_path}")
    print(f"run_dir: {run_dir}")
    print(f"step={ckpt.get('step')} device={device} n_obs={n_obs} n_act={n_act} resize={h} predict_joint_delta={predict_joint_delta}")

    t0 = time.time()
    dataset = build_episode_dataset(
        run_dir=run_dir,
        n_obs_steps=n_obs,
        horizon=cfg.dataset.horizon,
        n_action_steps=n_act,
        drop_n_last_frames=n_act,
        stats=stats,
        normalize=True,
        norm_mode=cfg.dataset.norm_mode,
        resize_size=cfg.dataset.resize_size,
        pre_crop_size=cfg.dataset.pre_crop_size,
        crop_size=None,
        random_crop=False,
        color_jitter_brightness=0.0,
        color_jitter_contrast=0.0,
        color_jitter_saturation=0.0,
        color_jitter_hue=0.0,
        defer_augment=True,
        uint8_cache=True,
        uint8_cache_dir=cfg.dataset.uint8_cache_dir,
        predict_joint_delta=bool(cfg.policy.predict_joint_delta),
    )
    print(f"dataset: n={len(dataset)}  load={time.time() - t0:.1f}s")

    rng = np.random.default_rng(args.seed)
    n_state = min(int(args.n_state_samples), len(dataset))
    n_image = min(int(args.n_image_samples), len(dataset))
    state_idx = rng.choice(len(dataset), size=n_state, replace=False).tolist()
    image_idx = rng.choice(len(dataset), size=n_image, replace=False).tolist()

    zeros = torch.zeros(1, n_cam, n_obs, 3, h, h, device=device)
    t1 = time.time()
    img_feat_zero = _img_feat(policy, zeros)
    print(f"encoded zero images: shape={tuple(img_feat_zero.shape)}  {time.time() - t1:.1f}s")

    def _windows_from_index(idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ep_local, t = dataset.index[idx]
        length = dataset._episode_lengths[ep_local]
        state_all = dataset._states[ep_local]
        action_all = dataset._actions[ep_local]
        obs_start = max(0, t - n_obs + 1)
        obs_indices = list(range(obs_start, t + 1))
        while len(obs_indices) < n_obs:
            obs_indices.insert(0, obs_indices[0])
        obs_state = dataset._normalize_state(state_all[obs_indices].astype(np.float32))
        action_end = min(t + n_act, length)
        gt = action_all[t:action_end].astype(np.float32)
        if gt.shape[0] < n_act:
            pad = np.zeros((n_act - gt.shape[0], gt.shape[1]), dtype=np.float32)
            gt = np.concatenate([gt, pad], axis=0)
        return torch.from_numpy(obs_state), torch.from_numpy(gt)

    pred_zero_chunks = []
    gt_chunks = []
    t2 = time.time()
    bs = max(int(args.batch_size), 1)
    for start in range(0, n_state, bs):
        sl = state_idx[start : start + bs]
        states = []
        gts = []
        for i in sl:
            st, gt = _windows_from_index(i)
            states.append(st)
            gts.append(gt)
        obs_state = torch.stack(states, dim=0).to(device)
        history = policy.history_action_encoder(obs_state)
        cond = _cond_from_img_feat(policy, img_feat_zero, obs_state)
        pred_norm = _decode(policy, history, cond)
        pred_phys = _abs_from_norm(pred_norm, obs_state)
        pred_zero_chunks.append(pred_phys.cpu().numpy())
        gt_chunks.append(torch.stack(gts, dim=0).numpy())
    pred_zero = np.concatenate(pred_zero_chunks, axis=0)
    gt_state = np.concatenate(gt_chunks, axis=0)
    drop_vs_gt = _mae_report(pred_zero - gt_state, names, joint_mask, grip_mask)
    print(f"drop-image vs GT: n={n_state}  {time.time() - t2:.1f}s")
    _print_mae(f"drop images (zeros) vs GT  n={n_state}", drop_vs_gt)

    image_loader = DataLoader(
        Subset(dataset, image_idx),
        batch_size=min(bs, n_image),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    pred_img_chunks = []
    pred_zero_img_chunks = []
    pred_shuf_chunks = []
    gt_img_chunks = []
    cond_l2 = []
    t3 = time.time()
    for batch in image_loader:
        images = images_to_float01(batch["obs_images"]).to(device)
        obs_state = batch["obs_state"].to(device)
        gt_norm = batch["action"][:, :n_act].to(device)
        history = policy.history_action_encoder(obs_state)
        feat = _img_feat(policy, images)
        feat_shuf = torch.roll(feat, shifts=1, dims=0) if feat.shape[0] > 1 else feat
        cond_img = _cond_from_img_feat(policy, feat, obs_state)
        cond_zero = _cond_from_img_feat(policy, img_feat_zero, obs_state)
        cond_shuf = _cond_from_img_feat(policy, feat_shuf, obs_state)
        cond_l2.append((cond_img - cond_zero).pow(2).sum(dim=-1).sqrt().cpu().numpy())
        pred_img = _abs_from_norm(_decode(policy, history, cond_img), obs_state)
        pred_z = _abs_from_norm(_decode(policy, history, cond_zero), obs_state)
        pred_s = _abs_from_norm(_decode(policy, history, cond_shuf), obs_state)
        gt_phys = _gt_abs_from_norm(gt_norm, obs_state)
        pred_img_chunks.append(pred_img.cpu().numpy())
        pred_zero_img_chunks.append(pred_z.cpu().numpy())
        pred_shuf_chunks.append(pred_s.cpu().numpy())
        gt_img_chunks.append(gt_phys.cpu().numpy())
        print(f"  image batch done  n={sum(x.shape[0] for x in pred_img_chunks)}/{n_image}", flush=True)

    pred_img = np.concatenate(pred_img_chunks, axis=0)
    pred_z = np.concatenate(pred_zero_img_chunks, axis=0)
    pred_s = np.concatenate(pred_shuf_chunks, axis=0)
    gt_img = np.concatenate(gt_img_chunks, axis=0)
    cond_delta = np.concatenate(cond_l2, axis=0)

    img_vs_gt = _mae_report(pred_img - gt_img, names, joint_mask, grip_mask)
    zero_vs_gt = _mae_report(pred_z - gt_img, names, joint_mask, grip_mask)
    shuf_vs_gt = _mae_report(pred_s - gt_img, names, joint_mask, grip_mask)
    zero_vs_img = _mae_report(pred_z - pred_img, names, joint_mask, grip_mask)
    shuf_vs_img = _mae_report(pred_s - pred_img, names, joint_mask, grip_mask)
    print(f"paired image ablation: n={n_image}  {time.time() - t3:.1f}s")
    _print_mae(f"with images vs GT  n={n_image}", img_vs_gt)
    _print_mae(f"drop images vs GT  n={n_image} (same windows)", zero_vs_gt)
    _print_mae(f"shuffled images vs GT  n={n_image}", shuf_vs_gt)
    _print_mae("prediction change: drop(zeros) vs with-images", zero_vs_img)
    _print_mae("prediction change: shuffled vs with-images", shuf_vs_img)
    print(
        f"\ncond latent L2 |img - zero|: mean={float(cond_delta.mean()):.4f}  "
        f"std={float(cond_delta.std()):.4f}"
    )
    joints_rel = zero_vs_img["mae_joints"] / max(img_vs_gt["mae_joints"], 1e-8)
    print(
        f"drop-vs-with MAE_joints / with-vs-GT MAE_joints = {joints_rel:.4f}  "
        f"(<<1 means dropping images barely changes the action)"
    )

    out = {
        "checkpoint": str(ckpt_path),
        "train_config": str(train_cfg_path) if train_cfg_path is not None else None,
        "step": ckpt.get("step"),
        "device": str(device),
        "seed": args.seed,
        "n_state_samples": n_state,
        "n_image_samples": n_image,
        "history_noise_std_eval": 0.0,
        "diagnostics": diag,
        "drop_images_vs_gt_state_only": drop_vs_gt,
        "with_images_vs_gt": img_vs_gt,
        "drop_images_vs_gt_paired": zero_vs_gt,
        "shuffled_images_vs_gt": shuf_vs_gt,
        "drop_vs_with_pred_change": zero_vs_img,
        "shuffle_vs_with_pred_change": shuf_vs_img,
        "cond_l2_img_minus_zero_mean": float(cond_delta.mean()),
        "drop_vs_with_over_with_gt_joints": float(joints_rel),
    }
    out_path = Path(args.out_json) if args.out_json else ckpt_path.parent / "eval_image_ablation.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
