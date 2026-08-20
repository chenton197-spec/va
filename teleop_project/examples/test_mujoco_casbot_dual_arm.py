"""在 MuJoCo 中同时加载并展示 CASBOT 左右手臂。"""

from __future__ import annotations

import time
from pathlib import Path

from teleop_sdk.adapters import MujocoFollower, MujocoSimulation

# 修改此变量即可选择 simulation/urdfs/ 中要加载的机器人模型，无需命令行参数。
URDF_FILENAME = "CASBOTWL12_WL12P1.urdf"

# 仅将这七个关节暴露给左右虚拟从臂；腰、头、腿和二指夹爪不参与本展示示例的控制。
LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
)
RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    simulation_root = project_root / "simulation"
    urdf_path = simulation_root / "urdfs" / URDF_FILENAME

    # URDF 声明项目内 meshes 的相对路径；本展示示例不向二指夹爪发送命令。
    simulation = MujocoSimulation(urdf_path)
    left_follower = MujocoFollower(simulation, LEFT_ARM_JOINT_NAMES)
    right_follower = MujocoFollower(simulation, RIGHT_ARM_JOINT_NAMES)

    try:
        left_follower.connect()
        right_follower.connect()
        if not left_follower.start_servo() or not right_follower.start_servo():
            raise RuntimeError("MuJoCo 虚拟从臂无法进入伺服状态")

        print(
            "[SIM] 已加载左右七轴手臂和二指夹爪；本示例只展示初始姿态，不发送运动命令。"
        )
        print("[SIM] 关闭 MuJoCo 窗口或按 Ctrl+C 退出。")
        # 被动查看器应由主线程创建，避免在不同平台上出现 GUI 线程问题。
        simulation.open_viewer()
        while simulation.viewer_is_running:
            simulation.sync_viewer()
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        print("\n[SIM] 收到退出请求。")
    finally:
        left_follower.stop_servo()
        right_follower.stop_servo()
        right_follower.disconnect()
        left_follower.disconnect()


if __name__ == "__main__":
    main()
