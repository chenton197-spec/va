#!/usr/bin/env python3
"""Offline sample_actions latency + encoder batched-vs-serial check (no robot).

Usage (from ct/va, lerobot env)::

    PYTHONPATH=. python scripts/bench_infer_latency.py \\
        --config configs/shine_shoes_a2a_noise_limits.yaml \\
        --checkpoint model/a2a_noise_shine_shoes_limits_260730175409/checkpoint_090000.pt

    PYTHONPATH=. python scripts/bench_infer_latency.py ... --compile
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from robotfm.config import load_config
from robotfm.train import build_policy


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bench policy sample_actions latency")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _dummy_batch(cfg, device: torch.device) -> dict[str, torch.Tensor]:
    n_cams = len(cfg.cameras)
    n_obs = int(cfg.dataset.n_obs_steps)
    h = w = int(cfg.dataset.crop_size or cfg.dataset.resize_size or 224)
    return {
        "obs_images": torch.rand(
            1, n_cams, n_obs, 3, h, w, device=device, dtype=torch.float32
        ),
        "obs_state": torch.randn(
            1, n_obs, int(cfg.state_dim), device=device, dtype=torch.float32
        ),
    }


def _compile_submodules(policy: torch.nn.Module) -> list[str]:
    names: list[str] = []
    for name in ("encoder", "unet", "flow_net"):
        mod = getattr(policy, name, None)
        if mod is None:
            continue
        # default (not reduce-overhead): avoid CUDA-graph overwrite with SpatialSoftmax
        setattr(policy, name, torch.compile(mod, mode="default"))
        names.append(name)
    return names


def _time_ms(fn, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _check_encoder_equiv(policy: torch.nn.Module, batch: dict[str, torch.Tensor]) -> float:
    """Return max abs diff between eval batched forward and serial reference."""
    enc = policy.encoder
    if not hasattr(enc, "image_encoder"):
        return float("nan")
    obs_images = batch["obs_images"]
    obs_state = batch["obs_state"]
    with torch.no_grad():
        out_batched = enc(obs_images, obs_state)
        b, cams, t, _, _, _ = obs_images.shape
        cam_feats = [enc.image_encoder(obs_images[:, c]) for c in range(cams)]
        img_feat = torch.cat(cam_feats, dim=-1)
        state = obs_state.reshape(b * t, -1)
        state_feat = enc.state_encoder(state).reshape(b, -1)
        out_serial = enc.proj(torch.cat([img_feat, state_feat], dim=-1))
    return (out_batched - out_serial).abs().max().item()


def main() -> None:
    args = _parse_args()
    va_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    if not cfg_path.is_absolute():
        cfg_path = va_root / cfg_path
    if not ckpt_path.is_absolute():
        ckpt_path = va_root / ckpt_path

    cfg = load_config(cfg_path)
    # Bench unguided path; RTC leftover cost is separate.
    if getattr(cfg.policy, "rtc", None) is not None:
        cfg.policy.rtc.enabled = False

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    policy = build_policy(cfg, ckpt["stats"])
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()

    if args.compile:
        compiled = _compile_submodules(policy)
        print(f"[INFO] torch.compile: {compiled or '(none)'}")

    torch.manual_seed(args.seed)
    batch = _dummy_batch(cfg, device)

    enc_diff = _check_encoder_equiv(policy, batch)
    print(f"[INFO] encoder batched vs serial max_abs_diff={enc_diff:.3e}")

    def _infer():
        with torch.no_grad():
            return policy.sample_actions(batch)

    print(f"[INFO] warmup={args.warmup} iters={args.iters} device={device}")
    for _ in range(args.warmup):
        _ = _infer()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = [_time_ms(_infer, device) for _ in range(args.iters)]
    times_sorted = sorted(times)
    mid = times_sorted[len(times_sorted) // 2]
    mean = sum(times) / len(times)
    print(
        f"[RESULT] sample_actions ms: mean={mean:.2f} median={mid:.2f} "
        f"min={times_sorted[0]:.2f} max={times_sorted[-1]:.2f} "
        f"compile={args.compile}"
    )


if __name__ == "__main__":
    main()
