#!/usr/bin/env python3
"""使用 teleop_sdk 控制真实 FAIRINO FR3 从臂。"""

from dataclasses import replace

from teleop_sdk import TeleopController
from teleop_sdk.adapters import AliciaLeaderArm, FairinoFR3Follower, GloriaMGripperFollower
from teleop_sdk.config import load_runtime_config


def main() -> None:
    """使用集中配置启动真实 Alicia-D 到 FR3 遥操作。"""
    runtime = load_runtime_config()
    leader = AliciaLeaderArm(
        port=runtime.alicia.port,
        gripper_type=runtime.alicia.gripper_type,
        connect_retries=runtime.alicia.connect_retries,
        connect_retry_delay_s=runtime.alicia.connect_retry_delay_s,
    )
    follower = FairinoFR3Follower(robot_ip=runtime.fr3.robot_ip)
    gripper = GloriaMGripperFollower(runtime.gloria_m) if runtime.gloria_m.enabled else None
    controller = TeleopController(
        leader,
        follower,
        replace(runtime.teleop, axis_sign=runtime.fr3.axis_sign),
        gripper=gripper,
    )

    print("=" * 60)
    print("    Alicia-D 到真实 FAIRINO FR3 遥操作（teleop_sdk）")
    print("=" * 60)
    print("  请确认工作空间安全，示教器为自动模式，FR3 已上使能")
    print("  参数请在 teleop.yaml 中修改；未填写字段使用 teleop_sdk/config.py 默认值")
    print("-" * 60)

    controller.connect()
    controller.run()


if __name__ == "__main__":
    main()
