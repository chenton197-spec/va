#!/usr/bin/env python3
"""遥操作数据采集入口。

CLI 用法
--------
在 conda 环境 ``lerobot`` 下，从 ``va/`` 目录运行::

    conda activate lerobot
    cd /path/to/va
    python scripts/collect.py --config configs/pusht_fm.yaml --target-episodes 10

命令行参数
~~~~~~~~~~
``--config PATH``
    YAML 配置文件路径（相对 ``va/`` 根目录）。默认 ``configs/pusht_fm.yaml``。
    采集相关字段见配置中的 ``collect:`` 块，以及全局的 ``backend``、``cameras``、
    ``state_dim``、``action_dim``、``fps``、``data_root``、``dataset.run_name`` 等。

``--target-episodes N``
    目标保存条数（成功 episode 计数）。若指定，会覆盖配置文件中的
    ``collect.target_episodes``；未指定则使用 yaml 里的默认值。

交互控制（PushT 鼠标遥操作）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **鼠标**：控制末端目标位置 (x, y)
- **R**：丢弃当前 episode 并重新 reset
- **S**：提前保存当前 episode（不要求任务成功）
- **Q / Esc**：退出采集

采集循环按 ``cfg.fps`` 限速；episode 自然结束（成功/超时）或按 **S** 时尝试写入磁盘。
默认只保存任务成功的轨迹（``collect.save_all: false``）；设为 ``true`` 时保存全部 episode。

数据存储格式
------------
一次采集 run 的目录::

    {data_root}/{dataset.run_name}_YYMMDDHHMM/
        meta.json              # 全局元信息（相机、维度、任务描述等）
        stats.json             # 全部 episode 的 state/action 均值与标准差（采集结束后生成）
        episodes/
            ep_000000.npz      # 第 0 条轨迹
            ep_000001.npz
            ...

默认路径示例：``data/demos/pusht_demos_2507211745/``（由 ``pusht_fm.yaml`` 的
``run_name`` 加启动时刻 ``YYMMDDHHMM`` 后缀决定）。

meta.json
~~~~~~~~~
对应 ``EpisodeMeta``，描述整个 run 的固定 schema，训练/可视化时读取::

    {
      "backend": "pusht",
      "embodiment": "pusht_sim",
      "fps": 10,
      "cameras": { "top": { "height": 96, "width": 96, "channels": 3 } },
      "state_dim": 2,
      "action_dim": 2,
      "state_names": ["x", "y"],
      "action_names": ["x", "y"],
      "num_episodes": 10,
      "task": "push the T block to the target",
      "created_at": "2026-07-21T08:00:00+00:00"
    }

同一 run 内所有 episode 的相机集合与 state/action 维度必须一致。

stats.json
~~~~~~~~~~
采集结束后，对所有已保存 episode 计算全局统计量，供训练归一化使用::

    {
      "state_mean": [...],
      "state_std": [...],
      "action_mean": [...],
      "action_std": [...]
    }

各字段为与 ``state_dim`` / ``action_dim`` 等长的 float 列表；``std`` 含 ``1e-6`` 防止除零。

ep_XXXXXX.npz（单条轨迹）
~~~~~~~~~~~~~~~~~~~~~~~~~
压缩 NPZ，时间维 ``T`` 为 episode 步数。数组 key 约定：

+----------------------+---------------------------+--------+
| Key                  | Shape                     | dtype  |
+======================+===========================+========+
| ``images/<camera>``  | ``(T, H, W, 3)``          | uint8  |
| ``state``            | ``(T, state_dim)``        | float32|
| ``action``           | ``(T, action_dim)``       | float32|
| ``reward``           | ``(T,)``                  | float32|
| ``done``             | ``(T,)``                  | bool   |
| ``success``          | 标量                      | bool   |
| ``task``             | 标量                      | str    |
+----------------------+---------------------------+--------+

例如 PushT 单相机 ``top`` 时，图像 key 为 ``images/top``。

行为克隆约定：每帧存 ``(o_t, a_t)``，即在 ``env.step(a_t)`` **之前**的观测与即将执行的动作；
``reward`` / ``done`` / ``success`` 来自该步 ``step`` 的返回结果。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from robotfm.collect.drivers.pusht_mouse import PushTMouseDriver
from robotfm.collect.loop import collect_demos
from robotfm.config import load_config
from robotfm.envs.registry import make_env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="遥操作采集机器人演示数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python scripts/collect.py --config configs/pusht_fm.yaml --target-episodes 10\n\n"
            "数据默认写入 data/demos/<run_name>/，详见脚本顶部文档字符串。"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pusht_fm.yaml",
        help="YAML 配置文件路径（相对 va/ 根目录，默认: configs/pusht_fm.yaml）",
    )
    parser.add_argument(
        "--target-episodes",
        type=int,
        default=None,
        metavar="N",
        help="目标保存 episode 数，覆盖配置文件 collect.target_episodes",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    cfg = load_config(base_dir / args.config)
    if args.target_episodes is not None:
        cfg.collect.target_episodes = args.target_episodes

    env = make_env(
        cfg,
        render_mode="human",
        window_title="PushT Teleop Collect — mouse move | R reset | S save | Q quit",
    )
    if cfg.backend == "pusht":
        driver = PushTMouseDriver()
    else:
        raise ValueError(f"No teleop driver for backend={cfg.backend}")

    run_dir = collect_demos(cfg, env, driver, base_dir)
    print(f"Collection finished: {run_dir}")


if __name__ == "__main__":
    main()
