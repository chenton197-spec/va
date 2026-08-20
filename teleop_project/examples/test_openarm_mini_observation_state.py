#!/usr/bin/env python3
"""持续只读 OpenArm Mini 左右主臂的关节和夹爪状态。

串口和组合标定 JSON 路径由 ``teleop.yaml`` 的 ``openarm_mini`` 段提供。
本程序只握手并读取当前位置，不重新标定、不写入任何电机寄存器，也不发送运动
命令。
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from teleop_sdk.adapters import OpenArmMiniLeaderArm
from teleop_sdk.config import OpenArmMiniLeaderConfig, load_runtime_config


READ_RATE_HZ = 20.0
READ_TIMEOUT_S = 0.1
PRINT_EVERY_N_SAMPLES = 1


def _validate_config(config: OpenArmMiniLeaderConfig) -> None:
    """在打开串口前检查状态读取所需的配置和标定文件。"""

    if not config.port_left.strip() or not config.port_right.strip():
        raise ValueError("openarm_mini.port_left 和 port_right 均不能为空")
    if config.port_left == config.port_right:
        raise ValueError("openarm_mini.port_left 和 port_right 必须是两条不同串口")
    if not config.calibration_path.strip():
        raise ValueError("请在 teleop.yaml 设置 openarm_mini.calibration_path")
    if not isinstance(config.baudrate, int) or config.baudrate <= 0:
        raise ValueError("openarm_mini.baudrate 必须为正整数")
    if not Path(config.calibration_path).expanduser().is_file():
        raise ValueError(
            f"找不到 OpenArm Mini 标定文件: {config.calibration_path}；请先运行标定示例"
        )


def _create_leaders(config: OpenArmMiniLeaderConfig) -> list[OpenArmMiniLeaderArm]:
    """创建两个严格只读的 OpenArm Mini 状态读取器。"""

    return [
        OpenArmMiniLeaderArm(
            port=config.port_left,
            calibration_path=config.calibration_path,
            side="left",
            baudrate=config.baudrate,
            read_only=True,
        ),
        OpenArmMiniLeaderArm(
            port=config.port_right,
            calibration_path=config.calibration_path,
            side="right",
            baudrate=config.baudrate,
            read_only=True,
        ),
    ]


def _format_state(frame: tuple[np.ndarray, float] | None) -> str:
    """将一帧状态格式化为适合终端持续查看的文本。"""

    if frame is None:
        return "unavailable"
    joint_angles, gripper_opening = frame
    return f"joints_deg={np.round(joint_angles, 2).tolist()} gripper={gripper_opening:.3f}"


def main() -> int:
    """以固定频率持续打印左右主臂状态，直到收到 Ctrl+C。"""

    if READ_RATE_HZ <= 0.0:
        raise ValueError("READ_RATE_HZ 必须为正数")
    if PRINT_EVERY_N_SAMPLES <= 0:
        raise ValueError("PRINT_EVERY_N_SAMPLES 必须为正整数")

    runtime = load_runtime_config()
    config = runtime.openarm_mini
    leaders: list[OpenArmMiniLeaderArm] = []

    try:
        _validate_config(config)
    except ValueError as exc:
        print(f"[ERROR] OpenArm Mini 状态读取配置无效: {exc}")
        return 2

    period_s = 1.0 / READ_RATE_HZ
    next_deadline = time.perf_counter()
    sample_index = 0

    try:
        leaders = _create_leaders(config)
        left, right = leaders
        left.connect()
        right.connect()
        print("=" * 60)
        print("    OpenArm Mini 双主臂实时状态")
        print("=" * 60)
        print("  本程序只读取电机位置；不会重标定、写寄存器或发送运动命令。")
        print(f"  读取频率: {READ_RATE_HZ:.1f} Hz；按 Ctrl+C 退出")
        print("-" * 60)

        while True:
            capture_monotonic_ns = time.perf_counter_ns()
            left_frame = left.read_joint_angles_and_gripper_opening(READ_TIMEOUT_S)
            right_frame = right.read_joint_angles_and_gripper_opening(READ_TIMEOUT_S)
            if sample_index % PRINT_EVERY_N_SAMPLES == 0:
                print(
                    f"state[{sample_index}] capture_monotonic_ns={capture_monotonic_ns}\n"
                    f"  left:  {_format_state(left_frame)}\n"
                    f"  right: {_format_state(right_frame)}",
                    flush=True,
                )

            sample_index += 1
            next_deadline += period_s
            delay_s = next_deadline - time.perf_counter()
            if delay_s > 0.0:
                time.sleep(delay_s)
            else:
                next_deadline = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[STOP] 停止读取 OpenArm Mini 双主臂状态")
        return 130
    except Exception as exc:
        print(f"[ERROR] OpenArm Mini 状态读取异常: {exc}")
        return 1
    finally:
        for leader in reversed(leaders):
            leader.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
