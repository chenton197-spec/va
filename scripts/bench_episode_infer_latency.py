#!/usr/bin/env python3
"""Compare serial vs batched-camera sample_actions latency on one episode.

Mirrors deploy replan cadence (every n_action_steps). Usage (from ct/va)::

    PYTHONPATH=. python scripts/bench_episode_infer_latency.py \\
      --checkpoint outputs/fm_shine_shoes_limits_260731181324/checkpoint_final.pt \\
      --config configs/shine_shoes_fm_limits.yaml \\
      --episode 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from robotfm.collect.loop import get_run_dir
from robotfm.config import load_config
from robotfm.data.dataset import crop_images, resize_images
from robotfm.data.lerobot_dataset import (
    _load_image_rgb,
    _short_camera_name,
    load_episode_arrays_from_parquet,
    load_lerobot_info,
)
from robotfm.data.stats import normalize
from robotfm.policies.encoders import MultiCameraEncoder
from robotfm.train import build_policy


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Episode replan latency: serial vs batched cams")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--episode", type=int, default=100)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=3, help="Repeats per replan step for stable ms")
    p.add_argument("--compile", action="store_true", help="Also time batched+compile")
    p.add_argument("--max-replans", type=int, default=None, help="Cap replans (debug)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override cfg.data_root (e.g. . when YAML points at another machine)",
    )
    p.add_argument(
        "--rtc",
        action="store_true",
        help="Enable RTC with leftover (matches deploy replan path; delay/horizon below)",
    )
    p.add_argument("--inference-delay", type=int, default=2)
    p.add_argument("--execution-horizon", type=int, default=4)
    return p.parse_args()


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


def _serial_forward(self: MultiCameraEncoder, obs_images, obs_state):
    b, cams, t, _, _, _ = obs_images.shape
    cam_feats = [self.image_encoder(obs_images[:, c]) for c in range(cams)]
    img_feat = torch.cat(cam_feats, dim=-1)
    state = obs_state.reshape(b * t, -1)
    state_feat = self.state_encoder(state).reshape(b, -1)
    return self.proj(torch.cat([img_feat, state_feat], dim=-1))


def _set_encoder_mode(policy: torch.nn.Module, mode: str) -> None:
    enc = policy.encoder
    if not isinstance(enc, MultiCameraEncoder):
        raise TypeError(f"Expected MultiCameraEncoder, got {type(enc)}")
    if mode == "serial":
        enc.forward = MethodType(_serial_forward, enc)  # type: ignore[method-assign]
    elif mode == "batched":
        # Restore class forward (eval batched path).
        enc.forward = MethodType(MultiCameraEncoder.forward, enc)  # type: ignore[method-assign]
    else:
        raise ValueError(mode)
    enc.eval()


def _compile_submodules(policy: torch.nn.Module) -> list[str]:
    names: list[str] = []
    for name in ("encoder", "unet", "flow_net", "action_decoder"):
        mod = getattr(policy, name, None)
        if mod is None:
            continue
        setattr(policy, name, torch.compile(mod, mode="default"))
        names.append(name)
    return names


def _time_infer_ms(
    policy,
    batch,
    device: torch.device,
    *,
    leftover: torch.Tensor | None = None,
    inference_delay: int | None = None,
    execution_horizon: int | None = None,
) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = policy.sample_actions(
            batch,
            prev_chunk_left_over=leftover,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _make_leftover(
    policy,
    batch: dict[str, torch.Tensor],
    *,
    inference_delay: int,
    device: torch.device,
) -> torch.Tensor:
    """One cold-start chunk → leftover = unexecuted suffix (deploy-like)."""
    with torch.no_grad():
        pred = policy.sample_actions(
            batch,
            prev_chunk_left_over=None,
            inference_delay=inference_delay,
            execution_horizon=None,
        )
    # (B, H, A) → (H - delay, A); RTC pads internally if needed.
    delay = min(int(inference_delay), int(pred.shape[1]) - 1)
    leftover = pred[0, delay:].detach().contiguous()
    return leftover.to(device)


def _summarize(xs: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    return float(arr.mean()), float(np.median(arr)), float(arr.std())


def _bench_mode(
    policy,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    mode_name: str,
    warmup: int,
    repeats: int,
    leftover: torch.Tensor | None,
    inference_delay: int | None,
    execution_horizon: int | None,
) -> list[float]:
    for b in batches[: max(1, warmup)]:
        _ = _time_infer_ms(
            policy,
            b,
            device,
            leftover=leftover,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )

    times: list[float] = []
    for b in batches:
        step_times = [
            _time_infer_ms(
                policy,
                b,
                device,
                leftover=leftover,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
            )
            for _ in range(repeats)
        ]
        times.append(float(np.median(step_times)))
    mean, med, std = _summarize(times)
    print(
        f"[RESULT] {mode_name:16s}  per-replan ms: mean={mean:.2f} median={med:.2f} "
        f"std={std:.2f} (n={len(times)})"
    )
    return times


def main() -> None:
    args = _parse_args()
    va_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    if not cfg_path.is_absolute():
        cfg_path = va_root / cfg_path
    if not ckpt_path.is_absolute():
        ckpt_path = va_root / ckpt_path

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(cfg_path)
    if args.data_root is not None:
        cfg.data_root = args.data_root

    # RTC must be enabled before build_policy so rtc_processor is constructed.
    if args.rtc:
        cfg.policy.rtc.enabled = True
        cfg.policy.rtc.inference_delay = int(args.inference_delay)
        cfg.policy.rtc.execution_horizon = int(args.execution_horizon)
    elif getattr(cfg.policy, "rtc", None) is not None:
        cfg.policy.rtc.enabled = False

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    stats = ckpt["stats"]
    cameras = list(cfg.cameras)
    n_obs = int(cfg.dataset.n_obs_steps)
    n_action_steps = int(cfg.policy.n_action_steps)
    norm_mode = cfg.dataset.norm_mode

    run_dir = get_run_dir(cfg, va_root)
    info = load_lerobot_info(run_dir)
    features = info["features"]
    cam_feat_keys = [
        next(k for k in features if _short_camera_name(k) == cam) for cam in cameras
    ]
    payload = load_episode_arrays_from_parquet(
        run_dir, args.episode, info, cam_feat_keys
    )
    states = payload["state"]
    image_paths = payload["image_paths"]
    length = int(payload["length"])

    replan_ts = list(range(0, length, n_action_steps))
    if args.max_replans is not None:
        replan_ts = replan_ts[: args.max_replans]

    print(f"[INFO] ckpt={ckpt_path}")
    print(f"[INFO] episode={args.episode} length={length} replans={len(replan_ts)}")
    print(
        f"[INFO] n_obs={n_obs} n_action_steps={n_action_steps} "
        f"num_inference_steps={cfg.policy.num_inference_steps} device={device}"
    )
    if args.rtc:
        print(
            f"[INFO] RTC ON delay={args.inference_delay} exec_h={args.execution_horizon} "
            "(timing uses leftover; cold-start excluded from mean)"
        )
    else:
        print("[INFO] RTC OFF")

    print("[INFO] building observation batches (disk I/O, not timed)...")
    batches: list[dict[str, torch.Tensor]] = []
    for t in replan_ts:
        batches.append(
            _build_obs_batch(
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
        )

    policy = build_policy(cfg, stats)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()

    rtc_kw: dict = {}
    leftover: torch.Tensor | None = None
    if args.rtc:
        rtc_kw = {
            "inference_delay": int(args.inference_delay),
            "execution_horizon": int(args.execution_horizon),
        }
        print("[INFO] building RTC leftover from cold-start on first batch...")
        leftover = _make_leftover(
            policy,
            batches[0],
            inference_delay=int(args.inference_delay),
            device=device,
        )
        print(f"[INFO] leftover shape={tuple(leftover.shape)}")

        # Also report one cold-start timing (no leftover) for reference.
        cold = [
            _time_infer_ms(policy, batches[0], device, leftover=None, **rtc_kw)
            for _ in range(max(3, args.repeats))
        ]
        print(
            f"[RESULT] {'cold-start':16s}  per-call ms: mean={np.mean(cold):.2f} "
            f"median={np.median(cold):.2f}"
        )

    results: dict[str, list[float]] = {}
    for mode in ("serial", "batched"):
        _set_encoder_mode(policy, mode)
        results[mode] = _bench_mode(
            policy,
            batches,
            device,
            mode_name=mode,
            warmup=args.warmup,
            repeats=args.repeats,
            leftover=leftover,
            inference_delay=rtc_kw.get("inference_delay"),
            execution_horizon=rtc_kw.get("execution_horizon"),
        )

    serial = np.asarray(results["serial"])
    batched = np.asarray(results["batched"])
    saved = serial - batched
    pct = 100.0 * saved / serial
    print(
        f"[SAVE]   batched vs serial: mean_save={saved.mean():.2f}ms/replan "
        f"({pct.mean():.1f}%)  median_save={np.median(saved):.2f}ms "
        f"min={saved.min():.2f} max={saved.max():.2f}"
    )

    if args.compile:
        # Fresh policy so compile starts from batched class forward
        policy2 = build_policy(cfg, stats)
        policy2.load_state_dict(ckpt["policy_state_dict"])
        policy2.to(device)
        policy2.eval()
        compiled = _compile_submodules(policy2)
        print(f"[INFO] torch.compile: {compiled}")

        leftover2 = leftover
        if args.rtc:
            leftover2 = _make_leftover(
                policy2,
                batches[0],
                inference_delay=int(args.inference_delay),
                device=device,
            )

        times_c = _bench_mode(
            policy2,
            batches,
            device,
            mode_name="batched+compile",
            warmup=max(args.warmup, 5),
            repeats=args.repeats,
            leftover=leftover2,
            inference_delay=rtc_kw.get("inference_delay"),
            execution_horizon=rtc_kw.get("execution_horizon"),
        )
        vs_serial = serial - np.asarray(times_c)
        vs_batched = batched - np.asarray(times_c)
        print(
            f"[SAVE]   compile vs serial: mean_save={vs_serial.mean():.2f}ms/replan "
            f"({100.0 * vs_serial.mean() / serial.mean():.1f}%)"
        )
        print(
            f"[SAVE]   compile vs batched: mean_save={vs_batched.mean():.2f}ms/replan "
            f"({100.0 * vs_batched.mean() / batched.mean():.1f}%)"
        )


if __name__ == "__main__":
    main()
