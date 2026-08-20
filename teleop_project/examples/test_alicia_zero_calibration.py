#!/usr/bin/env python3
"""使用 Alicia-D SDK 的交互流程校准设备零位。

该操作会永久修改设备的零位标定。SDK 会先要求确认，随后关闭扭矩；
操作者手动将机械臂摆到目标零位后，再按提示完成第二次确认。
"""

from __future__ import annotations

from typing import Any

from teleop_sdk.config import load_runtime_config


def _create_robot(port: str, gripper_type: str) -> Any:
    """延迟导入厂商 SDK，避免无硬件测试依赖它。"""

    from alicia_d_sdk import create_robot

    return create_robot(port=port, gripper_type=gripper_type)


def main() -> int:
    """连接 Alicia-D 并执行 SDK 提供的永久零位校准流程。"""

    runtime = load_runtime_config()
    robot: Any | None = None

    print("=" * 60)
    print("    Alicia-D 设备零位校准")
    print("=" * 60)
    print("  [WARN] 此操作会永久修改设备零位，不是遥操相对起点设置。")
    print("  [WARN] SDK 将要求两次 Enter，并在中间关闭机械臂扭矩。")
    print("  请确认周围无人、机械臂有支撑，并按设备规程操作。")
    print("-" * 60)

    try:
        robot = _create_robot(
            port=runtime.alicia.port,
            gripper_type=runtime.alicia.gripper_type,
        )
        if not robot.is_connected():
            print("[ERROR] Alicia-D 连接失败，未开始零位校准")
            return 1

        print("[INFO] Alicia-D 已连接；即将进入 SDK 的零位校准交互流程")
        if not robot.zero_calibration():
            print("[ERROR] Alicia-D 零位校准失败")
            return 1

        print("[INFO] Alicia-D 零位校准完成")
        return 0
    except KeyboardInterrupt:
        print("\n[STOP] Alicia-D 零位校准已中断")
        print("[WARN] 若扭矩已经关闭，请按设备规程安全恢复后再断电或继续操作。")
        return 130
    except Exception as exc:
        print(f"[ERROR] Alicia-D 零位校准异常: {exc}")
        return 1
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
