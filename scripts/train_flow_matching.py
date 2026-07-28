#!/usr/bin/env python3
"""Flow Matching 策略训练的命令行入口。

本脚本本身不做训练循环，只负责：
1. 解析命令行参数（配置文件路径）；
2. 根据脚本位置定位项目根目录 ``va/``；
3. 加载 YAML 配置为 ``RobotFMConfig``；
4. 调用 ``robotfm.train.train_flow_matching`` 执行真正的 BC / Flow Matching 训练；
5. 打印最终 checkpoint 路径。

典型用法（在 ``va/`` 目录下）::

    python scripts/train_flow_matching.py
    python scripts/train_flow_matching.py --config configs/pusht_n_a2a.yaml

续训（``train.steps`` 为全局总步数）::

    # 已训到 30k，再训 30k → 总步数改为 60000
    python scripts/train_flow_matching.py \\
        --config configs/pusht_n_a2a.yaml \\
        --resume outputs/n_a2a_pusht_xxx/checkpoint_final.pt \\
        --steps 60000

训练数据、优化步数、策略超参等均由配置文件中的
``dataset`` / ``train`` / ``policy`` 等字段决定，详见 ``configs/pusht_fm.yaml``。

默认会把 ``output_dir`` 追加 ``YYMMDDHHMMSS`` 后缀，避免覆盖旧实验；
可用 ``--no-timestamp`` 关闭。续训到同一目录时请加 ``--no-timestamp``，
并把 ``output_dir`` 指到原 run（或在 yaml 里写死该路径）。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from robotfm.config import load_config
from robotfm.train import train_flow_matching


def _timestamped_output_dir(output_dir: str, when: datetime | None = None) -> str:
    """为 output_dir 追加 YYMMDDHHMMSS 后缀，避免多次训练互相覆盖。"""
    stamp = (when or datetime.now()).strftime("%y%m%d%H%M%S")
    base = output_dir.rstrip("/")
    return f"{base}_{stamp}"


def main() -> None:
    """解析参数、加载配置并启动 Flow Matching 训练。"""
    parser = argparse.ArgumentParser(description="Train flow matching policy")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pusht_fm.yaml",
        help="相对项目根目录（va/）的配置文件路径，或绝对路径",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="不给 output_dir 追加 YYMMDDHHMMSS 后缀（默认会追加）",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从 checkpoint 续训（恢复 policy / optimizer / step）",
    )
    parser.add_argument(
        "--reset-step",
        action="store_true",
        help="与 --resume 联用：只加载权重，从 step=0 再训 train.steps（换数据 finetune）",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="覆盖 train.steps（全局总步数；续训时需大于 checkpoint 的 step）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="覆盖 output_dir（续训写回原目录时常用）",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="覆盖 dataset.run_name",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    cfg = load_config(config_path)
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.steps is not None:
        cfg.train.steps = args.steps
    if args.run_name is not None:
        cfg.dataset.run_name = args.run_name
    if not args.no_timestamp:
        cfg.output_dir = _timestamped_output_dir(cfg.output_dir)
    print(f"output_dir: {cfg.output_dir}")
    if args.resume:
        print(f"resume: {args.resume}")
        if args.reset_step:
            print("reset_step: True (train another full train.steps from step 0)")
    print(f"dataset.run_name: {cfg.dataset.run_name}")
    print(f"dataset.norm_mode: {cfg.dataset.norm_mode}")
    print(f"train.steps (total): {cfg.train.steps}")

    resume_path = Path(args.resume) if args.resume else None
    ckpt = train_flow_matching(
        cfg,
        base_dir,
        resume_path=resume_path,
        reset_step=args.reset_step,
        source_config_path=config_path,
    )
    print(f"Training finished: {ckpt}")


if __name__ == "__main__":
    main()
