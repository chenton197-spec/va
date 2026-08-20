#!/usr/bin/env python3
"""使用 teleop_sdk 运行 Alicia-D 到模拟从臂的实时图形模式。"""

import threading

from teleop_sdk import TeleopController, start_visualization
from teleop_sdk.adapters import AliciaLeaderArm, MockFollower
from teleop_sdk.config import load_runtime_config


def main() -> None:
    """使用集中配置启动模拟从臂和实时图形窗口。"""
    runtime = load_runtime_config()
    leader = AliciaLeaderArm(
        port=runtime.alicia.port,
        gripper_type=runtime.alicia.gripper_type,
        connect_retries=runtime.alicia.connect_retries,
        connect_retry_delay_s=runtime.alicia.connect_retry_delay_s,
    )
    follower = MockFollower(n_joints=leader.joint_count)
    controller = TeleopController(leader, follower, runtime.teleop)

    print("[INFO] 模拟模式参数请在 teleop.yaml 中修改")
    controller.connect()
    worker = threading.Thread(target=controller.run, daemon=True)
    worker.start()
    start_visualization(follower, threading.Event())


if __name__ == "__main__":
    main()
