#!/usr/bin/env python3
"""只读校验双 OpenArm Mini 主臂的零位和夹爪标定。

串口和组合标定 JSON 路径由 ``teleop.yaml`` 的 ``openarm_mini`` 段提供。
本程序只握手并读取当前位置，不重新标定、不写入任何电机寄存器，也不发送运动
命令。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from teleop_sdk.adapters import OpenArmMiniLeaderArm
from teleop_sdk.config import OpenArmMiniLeaderConfig, load_runtime_config


ZERO_TOLERANCE_DEG = 5.0
GRIPPER_TOLERANCE = 0.05
READ_TIMEOUT_S = 0.2


def _validate_config(config: OpenArmMiniLeaderConfig) -> None:
    """在打开串口前检查校验所需的部署配置和标定文件。"""

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


def _verify_zero_and_closed(side: str, leader: OpenArmMiniLeaderArm) -> bool:
    """检查零位关节角度和完全闭合的归一化夹爪值。"""

    input(f"\n[INPUT] 将 {side} 主臂摆到标定零位并完全闭合夹爪后按 Enter: ")
    frame = leader.read_joint_angles_and_gripper_opening(READ_TIMEOUT_S)
    if frame is None:
        raise RuntimeError(f"未读取到 {side} 主臂状态")
    joint_angles, gripper_opening = frame

    max_abs_angle = float(np.max(np.abs(joint_angles)))
    zero_passed = max_abs_angle <= ZERO_TOLERANCE_DEG
    closed_passed = gripper_opening <= GRIPPER_TOLERANCE
    print(f"[INFO] {side} 零位关节角度: {np.round(joint_angles, 2).tolist()} deg")
    print(
        f"[{'PASS' if zero_passed else 'FAIL'}] {side} 零位最大误差: "
        f"{max_abs_angle:.2f} deg（阈值 {ZERO_TOLERANCE_DEG:.2f} deg）"
    )
    print(
        f"[{'PASS' if closed_passed else 'FAIL'}] {side} 闭合夹爪: "
        f"{gripper_opening:.3f}（要求 <= {GRIPPER_TOLERANCE:.2f}）"
    )
    return zero_passed and closed_passed


def _verify_open(side: str, leader: OpenArmMiniLeaderArm) -> bool:
    """检查完全张开时的归一化夹爪值。"""

    input(f"\n[INPUT] 将 {side} 主臂夹爪完全张开后按 Enter: ")
    gripper_opening = leader.read_gripper_opening(READ_TIMEOUT_S)
    if gripper_opening is None:
        raise RuntimeError(f"未读取到 {side} 主臂夹爪状态")

    minimum_opening = 1.0 - GRIPPER_TOLERANCE
    open_passed = gripper_opening >= minimum_opening
    print(
        f"[{'PASS' if open_passed else 'FAIL'}] {side} 张开夹爪: "
        f"{gripper_opening:.3f}（要求 >= {minimum_opening:.2f}）"
    )
    return open_passed


def main() -> int:
    """依次校验左右 OpenArm Mini；任一指标失败时返回非零状态。"""

    runtime = load_runtime_config()
    config = runtime.openarm_mini
    leaders: list[OpenArmMiniLeaderArm] = []

    try:
        _validate_config(config)
    except ValueError as exc:
        print(f"[ERROR] OpenArm Mini 校验配置无效: {exc}")
        return 2

    print("=" * 60)
    print("    OpenArm Mini 标定校验")
    print("=" * 60)
    print("  本程序只读取电机位置；不会重标定、写寄存器或发送运动命令。")
    print(f"  标定文件: {config.calibration_path}")
    print(
        f"  关节零位阈值: +/-{ZERO_TOLERANCE_DEG:.2f} deg  "
        f"夹爪阈值: 闭合 <= {GRIPPER_TOLERANCE:.2f}，"
        f"张开 >= {1.0 - GRIPPER_TOLERANCE:.2f}"
    )
    print("-" * 60)

    try:
        leaders = _create_leaders(config)
        all_passed = True
        for side, leader in zip(("left", "right"), leaders, strict=True):
            print(f"\n[INFO] 开始校验 {side} OpenArm Mini")
            leader.connect()
            all_passed = _verify_zero_and_closed(side, leader) and all_passed
            all_passed = _verify_open(side, leader) and all_passed

        if all_passed:
            print("\n[PASS] 左右 OpenArm Mini 标定校验通过")
            return 0
        print("\n[FAIL] 存在未通过的标定项；请重新执行标定后再校验")
        return 1
    except KeyboardInterrupt:
        print("\n[STOP] OpenArm Mini 标定校验已取消")
        return 130
    except Exception as exc:
        print(f"[ERROR] OpenArm Mini 标定校验异常: {exc}")
        return 1
    finally:
        for leader in reversed(leaders):
            leader.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
