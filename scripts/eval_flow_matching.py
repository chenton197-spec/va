#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from robotfm.config import load_config, resolve_path
from robotfm.eval import evaluate_flow_matching
from robotfm.policies.rtc import RTCConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate flow matching / ACT policy")
    parser.add_argument("--config", type=str, default="configs/pusht_fm.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--save-video", action="store_true", help="将 rollout 存成 mp4")
    parser.add_argument(
        "--render",
        action="store_true",
        help="打开 pygame 窗口实时观看策略推块（玩/演示用）",
    )
    parser.add_argument(
        "--rtc",
        action="store_true",
        help="启用 RTC 推理引导（不改 yaml；训练 checkpoint 可直接用）",
    )
    parser.add_argument(
        "--no-guidance",
        action="store_true",
        help="RTC 开时关闭前缀引导（仅 ahead+discard）",
    )
    parser.add_argument("--inference-delay", type=int, default=None, help="RTC 模拟推理延迟（步）")
    parser.add_argument("--execution-horizon", type=int, default=None, help="RTC execution_horizon")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="覆盖 policy.n_action_steps（缩短可更频繁 replan，无需重训）",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help="启用 ACT temporal ensembling（原版常用 0.01）；会强制 n_action_steps=1",
    )
    parser.add_argument(
        "--no-pace",
        action="store_true",
        help="渲染时不按 fps 限速（默认 --render 会按 cfg.fps 实时播放）",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    cfg = load_config(base_dir / args.config)
    if args.num_episodes is not None:
        cfg.eval.num_episodes = args.num_episodes
    if args.save_video:
        cfg.eval.save_video = True
    if args.n_action_steps is not None:
        cfg.policy.n_action_steps = args.n_action_steps
    if args.temporal_ensemble_coeff is not None:
        cfg.policy.temporal_ensemble_coeff = args.temporal_ensemble_coeff
        # ACTConfig 要求 temporal ensemble 时 n_action_steps=1
        cfg.policy.n_action_steps = 1

    if (
        args.rtc
        or getattr(args, "no_guidance", False)
        or args.inference_delay is not None
        or args.execution_horizon is not None
    ):
        rtc = cfg.policy.rtc
        cfg.policy.rtc = RTCConfig(
            enabled=True if args.rtc else rtc.enabled,
            guidance_enabled=False if getattr(args, "no_guidance", False) else rtc.guidance_enabled,
            prefix_attention_schedule=rtc.prefix_attention_schedule,
            max_guidance_weight=rtc.max_guidance_weight,
            execution_horizon=args.execution_horizon if args.execution_horizon is not None else rtc.execution_horizon,
            inference_delay=args.inference_delay if args.inference_delay is not None else rtc.inference_delay,
            debug=rtc.debug,
            debug_maxlen=rtc.debug_maxlen,
        )

    ckpt = Path(args.checkpoint) if args.checkpoint else resolve_path(base_dir, cfg.output_dir) / "checkpoint_final.pt"
    print(f"checkpoint: {ckpt}")
    print(
        f"rtc.enabled={cfg.policy.rtc.enabled} guidance={cfg.policy.rtc.guidance_enabled} "
        f"delay={cfg.policy.rtc.inference_delay} "
        f"exec_h={cfg.policy.rtc.execution_horizon} render={args.render}"
    )
    print(
        f"n_action_steps={cfg.policy.n_action_steps} "
        f"temporal_ensemble_coeff={cfg.policy.temporal_ensemble_coeff}"
    )
    print(f"success_coverage>={cfg.eval.success_coverage}")
    metrics = evaluate_flow_matching(
        cfg,
        ckpt,
        base_dir,
        render=args.render,
        pace_realtime=args.render and not args.no_pace,
    )
    print(f"Success rate: {metrics['success_rate']:.3f}")
    print(f"Avg reward: {metrics['avg_reward']:.3f}")
    print(f"Mean max coverage: {metrics['mean_max_coverage']:.3f}")


if __name__ == "__main__":
    main()
