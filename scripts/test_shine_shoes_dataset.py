#!/usr/bin/env python3
"""Smoke-test shine_shoes_fr3 loading through robotfm.

Usage (from va/ with conda env lerobot)::

    PYTHONPATH=. python scripts/test_shine_shoes_dataset.py --config configs/shine_shoes_fm_limits.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from robotfm.collect.loop import get_run_dir
from robotfm.config import _normalize_rtc_config, load_config
from robotfm.data.dataset import build_episode_dataset
from robotfm.data.lerobot_dataset import (
    LeRobotImageSequenceDataset,
    is_lerobot_image_sequence_root,
)
from robotfm.data.stats import ensure_stats
from robotfm.policies.flow_matching import FlowMatchingConfig, FlowMatchingPolicy


def _build_fm_policy(cfg) -> FlowMatchingPolicy:
    rtc = _normalize_rtc_config(cfg.policy.rtc)
    fm_cfg = FlowMatchingConfig(
        num_cameras=len(cfg.cameras),
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        horizon=cfg.dataset.horizon,
        n_obs_steps=cfg.dataset.n_obs_steps,
        hidden_dim=cfg.policy.hidden_dim,
        num_inference_steps=cfg.policy.num_inference_steps,
        beta_alpha=cfg.policy.beta_alpha,
        beta_beta=cfg.policy.beta_beta,
        noise_s=cfg.policy.noise_s,
        down_dims=tuple(cfg.policy.down_dims),
        diffusion_step_embed_dim=cfg.policy.diffusion_step_embed_dim,
        kernel_size=cfg.policy.kernel_size,
        n_groups=cfg.policy.n_groups,
        pretrained_encoder=cfg.policy.pretrained_encoder,
        use_frame_diff=cfg.policy.use_frame_diff,
        use_coord_conv=cfg.policy.use_coord_conv,
        vision_backbone=cfg.policy.vision_backbone,
        rtc=rtc if rtc.enabled else None,
    )
    return FlowMatchingPolicy(fm_cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test shine_shoes_fr3 dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/shine_shoes_fm_limits.yaml",
        help="Path to RobotFM yaml (relative to va/ or absolute)",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--skip-train-step", action="store_true")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = base_dir / cfg_path
    cfg = load_config(cfg_path)
    run_dir = get_run_dir(cfg, base_dir)

    print(f"run_dir: {run_dir}")
    assert run_dir.is_dir(), f"Dataset root missing: {run_dir}"
    assert is_lerobot_image_sequence_root(run_dir), "Expected leobot image_sequence root"

    stats = ensure_stats(run_dir, cfg.dataset.norm_mode)
    for key in ("state_mean", "state_std", "action_mean", "action_std"):
        assert key in stats, f"missing {key} in stats"
        assert stats[key].shape == (7,), f"{key} shape {stats[key].shape} != (7,)"
    print(
        f"stats: state_dim={stats['state_mean'].shape[0]} "
        f"action_dim={stats['action_mean'].shape[0]} "
        f"written={run_dir / 'stats.json'}"
    )

    dataset = build_episode_dataset(
        run_dir=run_dir,
        n_obs_steps=cfg.dataset.n_obs_steps,
        horizon=cfg.dataset.horizon,
        n_action_steps=cfg.policy.n_action_steps,
        drop_n_last_frames=0,
        stats=stats,
        normalize=True,
        norm_mode=cfg.dataset.norm_mode,
        resize_size=cfg.dataset.resize_size,
        crop_size=cfg.dataset.crop_size,
        random_crop=True,
    )
    assert isinstance(dataset, LeRobotImageSequenceDataset)
    assert len(dataset) > 0, "empty dataset"
    print(
        f"dataset: type={type(dataset).__name__} num_samples={len(dataset)} "
        f"cameras={dataset.meta.camera_names} "
        f"state_dim={dataset.meta.state_dim} action_dim={dataset.meta.action_dim}"
    )

    sample = dataset[0]
    n_cams = len(cfg.cameras)
    n_obs = cfg.dataset.n_obs_steps
    crop = cfg.dataset.crop_size
    assert sample["obs_images"].shape == (n_cams, n_obs, 3, crop, crop), sample[
        "obs_images"
    ].shape
    assert sample["obs_state"].shape == (n_obs, 7), sample["obs_state"].shape
    assert sample["action"].shape == (cfg.dataset.horizon, 7), sample["action"].shape
    assert sample["action_mask"].shape == (cfg.dataset.horizon, 1), sample[
        "action_mask"
    ].shape
    print(f"sample[0] ok: {[(k, tuple(v.shape)) for k, v in sample.items()]}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    batch = next(iter(loader))
    b = args.batch_size
    assert batch["obs_images"].shape == (b, n_cams, n_obs, 3, crop, crop), batch[
        "obs_images"
    ].shape
    assert batch["obs_state"].shape == (b, n_obs, 7), batch["obs_state"].shape
    assert batch["action"].shape == (b, cfg.dataset.horizon, 7), batch["action"].shape
    print(f"batch ok: {[(k, tuple(v.shape)) for k, v in batch.items()]}")

    if not args.skip_train_step:
        device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
        # Avoid downloading ImageNet weights for a smoke check if offline
        cfg.policy.pretrained_encoder = False
        policy = _build_fm_policy(cfg).to(device)
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        out = policy.compute_loss(batch_dev)
        loss = out[0] if isinstance(out, tuple) else out
        loss.backward()
        print(f"train step ok: device={device} loss={float(loss.detach().cpu()):.6f}")

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
