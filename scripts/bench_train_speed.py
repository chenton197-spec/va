#!/usr/bin/env python3
"""短测训练吞吐：暖机后计量 s/it 与 samp/s（对齐 train.py 数据/AMP 路径）。

用法::

    PYTHONPATH=. python scripts/bench_train_speed.py \\
        --config configs/openarm_hcx_dual_arm_a2a_noise_sepenc.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from robotfm.collect.loop import get_run_dir
from robotfm.config import load_config
from robotfm.data.dataset import apply_image_augments_batch, build_episode_dataset
from robotfm.data.stats import ensure_stats, is_limits_mode
from robotfm.train import _build_optimizer, build_policy


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark train step throughput")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=None, help="覆盖 train.batch_size")
    p.add_argument("--num-workers", type=int, default=None, help="覆盖 train.num_workers")
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 train.compile",
    )
    p.add_argument("--horizon", type=int, default=None, help="覆盖 dataset.horizon")
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="覆盖 policy.n_action_steps（默认与 --horizon 相同）",
    )
    args = p.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    cfg = load_config(config_path)
    if args.batch_size is not None:
        cfg.train.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        cfg.train.num_workers = int(args.num_workers)
    if args.compile is not None:
        cfg.train.compile = bool(args.compile)
    if args.horizon is not None:
        cfg.dataset.horizon = int(args.horizon)
        if args.n_action_steps is None:
            cfg.policy.n_action_steps = int(args.horizon)
    if args.n_action_steps is not None:
        cfg.policy.n_action_steps = int(args.n_action_steps)

    run_dir = get_run_dir(cfg, base_dir)
    require_image = (
        cfg.policy.type == "act"
        and getattr(cfg.dataset, "image_norm_mode", "imagenet") == "dataset"
    )
    stats = ensure_stats(
        run_dir, cfg.dataset.norm_mode, require_image_stats=require_image
    )
    if is_limits_mode(cfg.dataset.norm_mode):
        stats = ensure_stats(run_dir, cfg.dataset.norm_mode)

    gpu_augment = bool(cfg.dataset.gpu_augment)
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
        pre_crop_size=cfg.dataset.pre_crop_size,
        crop_size=cfg.dataset.crop_size,
        random_crop=True,
        color_jitter_brightness=cfg.dataset.color_jitter_brightness,
        color_jitter_contrast=cfg.dataset.color_jitter_contrast,
        color_jitter_saturation=cfg.dataset.color_jitter_saturation,
        color_jitter_hue=cfg.dataset.color_jitter_hue,
        defer_augment=gpu_augment,
        uint8_cache=bool(cfg.dataset.uint8_cache),
            uint8_cache_dir=cfg.dataset.uint8_cache_dir,
            predict_joint_delta=bool(cfg.policy.predict_joint_delta),
        )

    loader_kwargs: dict = {
        "batch_size": cfg.train.batch_size,
        "shuffle": True,
        "num_workers": cfg.train.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }
    if cfg.train.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; cannot bench GPU train speed")
    device = torch.device("cuda")
    use_amp = bool(getattr(cfg.train, "amp", False))
    use_compile = bool(getattr(cfg.train, "compile", False))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    policy = build_policy(cfg, stats).to(device)
    optim = _build_optimizer(policy, cfg)
    if use_compile:
        # 与 train.py 一致：mode=default（勿用 reduce-overhead）
        policy = torch.compile(policy, mode="default")

    print(
        f"bench: bs={cfg.train.batch_size} workers={cfg.train.num_workers} "
        f"amp={use_amp} compile={use_compile} resize={cfg.dataset.resize_size} "
        f"horizon={cfg.dataset.horizon} n_action_steps={cfg.policy.n_action_steps} "
        f"N={len(dataset)} warmup={args.warmup} steps={args.steps}"
    )
    cache = getattr(dataset, "_image_cache", None)
    if cache is not None:
        print(f"uint8_cache: {cache.cache_dir}")

    it = iter(loader)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    def one_step():
        batch = next_batch()
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if gpu_augment:
            batch["obs_images"] = apply_image_augments_batch(
                batch["obs_images"],
                crop_size=cfg.dataset.crop_size,
                random_crop=True,
                brightness=cfg.dataset.color_jitter_brightness,
                contrast=cfg.dataset.color_jitter_contrast,
                saturation=cfg.dataset.color_jitter_saturation,
                hue=cfg.dataset.color_jitter_hue,
            )
        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = policy.compute_loss(batch)
            loss = out[0] if isinstance(out, tuple) else out
        if use_amp:
            scaler.scale(loss).backward()
            if cfg.train.max_grad_norm is not None:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), cfg.train.max_grad_norm
                )
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            if cfg.train.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), cfg.train.max_grad_norm
                )
            optim.step()
        return float(loss.detach().item())

    for i in range(args.warmup):
        loss = one_step()
        print(f"warmup {i + 1}/{args.warmup} loss={loss:.4f}")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    last_loss = 0.0
    for i in range(args.steps):
        last_loss = one_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    s_per_it = elapsed / args.steps
    samp_s = cfg.train.batch_size / s_per_it
    steps_ep = len(dataset) // cfg.train.batch_size
    h_ep = steps_ep * s_per_it / 3600.0
    mem = torch.cuda.max_memory_allocated() / (1024**3)
    try:
        import resource

        maxrss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    except Exception:
        maxrss_gb = float("nan")

    print("---")
    print(f"steady: {args.steps} steps in {elapsed:.1f}s")
    print(f"s/it: {s_per_it:.3f}")
    print(f"samp/s: {samp_s:.2f}")
    print(f"est h/epoch: {h_ep:.2f} (steps/epoch={steps_ep})")
    print(f"peak VRAM: {mem:.2f} GB")
    print(f"peak RSS: {maxrss_gb:.2f} GB")
    print(f"last_loss: {last_loss:.4f}")


if __name__ == "__main__":
    main()
