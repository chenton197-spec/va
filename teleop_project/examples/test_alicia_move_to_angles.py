#!/usr/bin/env python3
"""将 Alicia-D 移动到此文件中指定的六轴关节角度。"""

from __future__ import annotations

import math
from typing import Any

from teleop_sdk.config import load_runtime_config

JOINT_COUNT = 6
# 运行前直接在此修改六轴目标角度，单位为度；不通过 CLI 传参。
TARGET_JOINTS_DEG: tuple[float, ...] = (-37, -59, -118, -88, 90, 47)
SPEED_DEG_S = 1.0


def _create_robot(port: str, gripper_type: str) -> Any:
    """延迟导入厂商 SDK，避免无硬件测试依赖它。"""

    from alicia_d_sdk import create_robot

    return create_robot(port=port, gripper_type=gripper_type)


def _validate_motion_settings() -> None:
    """在连接硬件前拒绝无效或会被 SDK 静默裁剪的目标。"""

    if len(TARGET_JOINTS_DEG) != JOINT_COUNT:
        raise ValueError(f"TARGET_JOINTS_DEG 必须包含 {JOINT_COUNT} 个关节角度")
    if not all(math.isfinite(angle) for angle in TARGET_JOINTS_DEG):
        raise ValueError("TARGET_JOINTS_DEG 必须全部为有限数值")
    if any(abs(angle) > 180.0 for angle in TARGET_JOINTS_DEG):
        raise ValueError("TARGET_JOINTS_DEG 的每个角度必须在 -180 到 180 度之间")
    if not math.isfinite(SPEED_DEG_S) or SPEED_DEG_S <= 0.0:
        raise ValueError("SPEED_DEG_S 必须是正的有限数值")


def main() -> int:
    """连接 Alicia-D，并将其移动到 ``TARGET_JOINTS_DEG``。"""

    robot: Any | None = None

    try:
        _validate_motion_settings()
    except ValueError as exc:
        print(f"[ERROR] Alicia-D 运动参数无效: {exc}")
        return 2

    print("=" * 60)
    print("    Alicia-D 指定关节角度运动")
    print("=" * 60)
    print(f"  目标角度（deg）: {list(TARGET_JOINTS_DEG)}")
    print(f"  关节速度（deg/s）: {SPEED_DEG_S:g}")
    print("  请确认目标姿态、工作空间和机械臂支撑均安全。")
    print("-" * 60)

    try:
        runtime = load_runtime_config()
        robot = _create_robot(
            port=runtime.alicia.port,
            gripper_type=runtime.alicia.gripper_type,
        )
        if not robot.is_connected():
            print("[ERROR] Alicia-D 连接失败，未下发运动目标")
            return 1

        input("[INPUT] 确认后按 Enter 下发关节目标，按 Ctrl+C 取消: ")
        if not robot.set_robot_state(
            target_joints=list(TARGET_JOINTS_DEG),
            joint_format="deg",
            speed_deg_s=SPEED_DEG_S,
            wait_for_completion=True,
        ):
            print("[ERROR] Alicia-D 未能到达指定关节角度")
            return 1

        print("[INFO] Alicia-D 已到达指定关节角度")
        return 0
    except KeyboardInterrupt:
        print("\n[STOP] Alicia-D 指定角度运动已中断")
        print("[WARN] 若目标已下发，机械臂可能仍在运动；请按设备规程处理。")
        return 130
    except Exception as exc:
        print(f"[ERROR] Alicia-D 指定角度运动异常: {exc}")
        return 1
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
