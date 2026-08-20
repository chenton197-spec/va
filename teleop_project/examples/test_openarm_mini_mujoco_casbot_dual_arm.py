#!/usr/bin/env python3
"""用双 OpenArm Mini 主臂遥操 MuJoCo 中的 CASBOT 双臂。

该示例控制左右七个手臂关节，并将 OpenArm 夹爪的归一化开合量同步到 CASBOT 的
Yunqin 二指夹爪。CASBOT 腰、头和腿不参与控制。MuJoCo 从臂采用直接写入 ``qpos``
的运动学模式，用于观察主从映射、滤波和弹簧阻尼的跟随效果，不模拟电机力矩或真实动力学。
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path

from teleop_sdk import TeleopController
from teleop_sdk.adapters import (
    MujocoFollower,
    MujocoGripper,
    MujocoSimulation,
    OpenArmMiniLeaderArm,
)
from teleop_sdk.config import OpenArmMiniLeaderConfig, TeleopConfig, load_runtime_config

# 修改此变量即可选择 simulation/urdfs/ 中要加载的机器人模型，无需命令行参数。
URDF_FILENAME = "CASBOTWL12_WL12P1.urdf"
# 此开关只控制弹簧阻尼和可选前瞻预测；两级滤波始终启用。
ENABLE_FILTER_AND_SPRING = False
# "absolute" 让主臂标定零位直接对应 CASBOT URDF 零位；改为 "relative" 可恢复相对跟随。
ARM_CONTROL_MODE = "absolute"
# 是否将棋盘格地板、渐变天空和定向光编译进 MuJoCo 场景；地板不参与碰撞或动力学。
ENABLE_CHECKERBOARD_FLOOR = True
# 地板平面高度；CASBOT 零位最低点会自动整体对齐到此高度。
CHECKERBOARD_FLOOR_Z_M = -0.5

# OpenArm Mini 的 joint_1 至 joint_7 与 CASBOT 手臂关节按相同顺序对应。
ARM_AXIS_ORDER = (0, 1, 2, 3, 4, 5, 6)
# 首次使用先保持同向。若 viewer 中某一关节方向相反，将对应元素改为 -1.0。
LEFT_AXIS_SIGN = (-1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0)
RIGHT_AXIS_SIGN = (1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0)

# 仅将这七个关节暴露给左右虚拟从臂；腰、头和腿不参与控制。
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

# 每侧二指夹爪使用两条平移关节。OpenArm 的 0 为闭合、1 为张开。
LEFT_GRIPPER_JOINT_NAMES = (
    "left_yunqin_gripper_finger_a_joint",
    "left_yunqin_gripper_finger_b_joint",
)
RIGHT_GRIPPER_JOINT_NAMES = (
    "right_yunqin_gripper_finger_a_joint",
    "right_yunqin_gripper_finger_b_joint",
)
# q=(0, 0) 时两指仍有约 99 mm 间隙；各向内移动 49.47 mm 时网格表面刚好接触。
GRIPPER_CLOSED_POSITIONS_M = (0.04947, -0.04947)
GRIPPER_OPEN_POSITIONS_M = (0.0, 0.0)


def _validate_openarm_config(config: OpenArmMiniLeaderConfig) -> None:
    """在打开串口前检查双 OpenArm Mini 所需的部署配置。"""

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


def _control_config_for_side(
    base_config: TeleopConfig, axis_sign: tuple[float, ...]
) -> TeleopConfig:
    """生成本示例专用的七轴映射、控制模式和弹簧路径开关。"""

    if len(axis_sign) != len(ARM_AXIS_ORDER):
        raise ValueError("OpenArm Mini 到 CASBOT 的轴方向数量必须为七个")
    if ARM_CONTROL_MODE not in {"absolute", "relative"}:
        raise ValueError("ARM_CONTROL_MODE 只能是 absolute 或 relative")
    return replace(
        base_config,
        axis_order=ARM_AXIS_ORDER,
        axis_sign=axis_sign,
        relative_mode=ARM_CONTROL_MODE == "relative",
        filter_enabled=base_config.filter_enabled,
        spring_enabled=base_config.spring_enabled and ENABLE_FILTER_AND_SPRING,
    )


def _create_controller(
    *,
    port: str,
    side: str,
    openarm_config: OpenArmMiniLeaderConfig,
    follower: MujocoFollower,
    gripper: MujocoGripper,
    teleop_config: TeleopConfig,
    axis_sign: tuple[float, ...],
) -> TeleopController:
    """创建一侧只读主臂和对应的虚拟从臂控制器。"""

    leader = OpenArmMiniLeaderArm(
        port=port,
        calibration_path=openarm_config.calibration_path,
        side=side,
        baudrate=openarm_config.baudrate,
        read_only=True,
    )
    return TeleopController(
        leader,
        follower,
        _control_config_for_side(teleop_config, axis_sign),
        gripper=gripper,
    )


def main() -> int:
    """在一个固定频率循环中同步运行左右主从控制链路和 MuJoCo viewer。"""

    try:
        runtime = load_runtime_config()
        _validate_openarm_config(runtime.openarm_mini)
        rate_hz = float(runtime.teleop.rate_hz)
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("teleop.rate_hz 必须为正数")
    except Exception as exc:
        print(f"[ERROR] OpenArm Mini -> MuJoCo 配置无效: {exc}")
        return 2

    project_root = Path(__file__).resolve().parents[1]
    urdf_path = project_root / "simulation" / "urdfs" / URDF_FILENAME
    controllers: list[TeleopController] = []

    try:
        # 一个双臂 URDF 只加载一次，左右虚拟从臂共享同一 MuJoCo 状态和 viewer。
        simulation = MujocoSimulation(urdf_path)
        if ENABLE_CHECKERBOARD_FLOOR:
            simulation.set_mjcf_environment(
                floor_z_m=CHECKERBOARD_FLOOR_Z_M,
                align_model_lowest_point_to_floor=True,
            )
        left_follower = MujocoFollower(simulation, LEFT_ARM_JOINT_NAMES)
        right_follower = MujocoFollower(simulation, RIGHT_ARM_JOINT_NAMES)
        left_gripper = MujocoGripper(
            simulation,
            LEFT_GRIPPER_JOINT_NAMES,
            GRIPPER_CLOSED_POSITIONS_M,
            GRIPPER_OPEN_POSITIONS_M,
        )
        right_gripper = MujocoGripper(
            simulation,
            RIGHT_GRIPPER_JOINT_NAMES,
            GRIPPER_CLOSED_POSITIONS_M,
            GRIPPER_OPEN_POSITIONS_M,
        )
        left_controller = _create_controller(
            port=runtime.openarm_mini.port_left,
            side="left",
            openarm_config=runtime.openarm_mini,
            follower=left_follower,
            gripper=left_gripper,
            teleop_config=runtime.teleop,
            axis_sign=LEFT_AXIS_SIGN,
        )
        right_controller = _create_controller(
            port=runtime.openarm_mini.port_right,
            side="right",
            openarm_config=runtime.openarm_mini,
            follower=right_follower,
            gripper=right_gripper,
            teleop_config=runtime.teleop,
            axis_sign=RIGHT_AXIS_SIGN,
        )
        controllers = [left_controller, right_controller]

        for controller in controllers:
            controller.connect()

        # 被动查看器必须在主线程创建；窗口会持续显示两侧 qpos 的实时更新。
        simulation.open_viewer()
        servo_started = [controller.start_servo() for controller in controllers]
        if not all(servo_started):
            raise RuntimeError("MuJoCo 虚拟从臂无法进入伺服状态")

        print("=" * 60)
        print("    OpenArm Mini -> MuJoCo CASBOT 双臂遥操")
        print("=" * 60)
        if ENABLE_FILTER_AND_SPRING:
            print("  控制链路: 连续化 -> One Euro -> 固定低通 -> 弹簧阻尼/前瞻预测")
        else:
            print(
                "  控制链路: 连续化 -> One Euro -> 固定低通 -> "
                "轴映射 -> URDF 限位 -> 直接位置命令"
            )
        if ARM_CONTROL_MODE == "relative":
            print("  映射模式: 相对同序七轴；启动时保持主臂静止，随后按变化量跟随。")
        else:
            print("  映射模式: 绝对同序七轴；主臂标定零位对应 CASBOT URDF 零位。")
        print(f"  控制频率: {rate_hz:.1f} Hz")
        print(f"  左臂轴方向: {list(LEFT_AXIS_SIGN)}")
        print(f"  右臂轴方向: {list(RIGHT_AXIS_SIGN)}")
        print("  主臂夹爪: 0=二指夹爪闭合，1=二指夹爪张开")
        print(
            f"  天空与棋盘格地板: {'已启用' if ENABLE_CHECKERBOARD_FLOOR else '已关闭'}"
        )
        print("  关闭 MuJoCo 窗口或按 Ctrl+C 安全退出。")
        print("-" * 60)

        period_s = 1.0 / rate_hz
        next_deadline = time.perf_counter()
        while simulation.viewer_is_running:
            timestamp = time.perf_counter()
            # 两侧使用同一时间戳并在同一调度周期执行，避免双臂独立线程产生额外抖动。
            left_controller.step(timestamp)
            right_controller.step(timestamp)
            simulation.sync_viewer()

            next_deadline += period_s
            delay_s = next_deadline - time.perf_counter()
            if delay_s > 0.0:
                time.sleep(delay_s)
            else:
                next_deadline = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[STOP] 收到退出请求。")
    except Exception as exc:
        print(f"[ERROR] OpenArm Mini -> MuJoCo 遥操异常: {exc}")
        return 1
    finally:
        for controller in reversed(controllers):
            controller.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
