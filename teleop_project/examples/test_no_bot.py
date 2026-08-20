#!/usr/bin/env python3
"""使用 teleop_sdk 运行 Alicia-D 空跑模式。"""

from teleop_sdk import TeleopController
from teleop_sdk.adapters import AliciaLeaderArm, DryRunFollower
from teleop_sdk.config import load_runtime_config


def main() -> None:
    """使用集中配置创建空跑遥操作。"""
    runtime = load_runtime_config()
    leader = AliciaLeaderArm(
        port=runtime.alicia.port,
        gripper_type=runtime.alicia.gripper_type,
        connect_retries=runtime.alicia.connect_retries,
        connect_retry_delay_s=runtime.alicia.connect_retry_delay_s,
    )
    follower = DryRunFollower(n_joints=leader.joint_count)
    controller = TeleopController(leader, follower, runtime.teleop)

    print("=" * 60)
    print("    Alicia-D 遥操作空跑模式（teleop_sdk）")
    print("=" * 60)
    print("  不连接从臂硬件；移动示教臂后将打印目标关节角度")
    print("  参数请在 teleop.yaml 中修改；未填写字段使用 teleop_sdk/config.py 默认值")
    print("-" * 60)

    controller.connect()
    controller.run()


if __name__ == "__main__":
    main()
