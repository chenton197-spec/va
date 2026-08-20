#!/usr/bin/env python3
"""最小 OpenArm Mini -> HCX 单臂 move_joints 遥操作对照。

本示例不启动直伺服，也不做插值。OpenArm Mini 以 100 Hz 读取七轴角度，
将相对变化映射为从臂目标；随后按较低频率非阻塞调用 HCX
``Arm.move_joints(..., interrupt=True)``。每个新规划目标会由控制器替换当前
规划路径，因此本程序用于比较“控制器规划运动”和直接伺服的实际表现。

运行前必须手动完成示教器脱离、报警处理、EtherCAT 就绪、全局使能和对应
单臂使能。本程序会产生真实运动；只有 ``CONFIRM_MOTION`` 为 ``True`` 时
才允许提交规划运动。

运行：

    python -m examples.test_openarm_hcx_single_arm_move_joints_teleop
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from teleop_sdk.adapters import (
    HcxConnection,
    HcxConnectionConfig,
    OpenArmMiniLeaderArm,
)

JOINT_COUNT = 7
ArmSide = Literal["left", "right"]

# 要测试的同侧组合。修改本文件顶部常量即可切换，不接受命令行参数。
TEST_SIDE: ArmSide = "right"

# OpenArm Mini 的只读串口与双侧组合标定文件。
OPENARM_PORT = "/dev/ttyACM0"
OPENARM_CALIBRATION_PATH = "./my_openarm_mini.json"
OPENARM_BAUDRATE = 1_000_000

# HCX 控制器与本次测试的单臂机器人 ID。
HCX_LOCAL_IP = "172.16.0.110"
HCX_REMOTE_IP = "172.16.0.89"
HCX_PORT = 12345
HCX_CONNECT_TIMEOUT_S = 10.0
HCX_ROBOT_ID = 2

# OpenArm -> HCX 七轴方向。+1.0 为同向，-1.0 为反向。
AXIS_SIGN = (-1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0)

# 主臂只读采样频率，单位为 Hz。
LEADER_SAMPLE_RATE_HZ = 100.0
# 每次 OpenArm 读取允许的最长等待时间，单位为秒。超时后保持上一次目标。
LEADER_READ_TIMEOUT_S = 0.008

# 控制器规划运动的新目标提交频率，单位为 Hz。主臂只有 100 Hz 新样本；设为
# 高于 100 Hz 不会产生更多有效目标，重复提交相同目标只会让控制器重复规划。
MOVE_JOINT_COMMAND_RATE_HZ = 25.0
# 控制器规划运动的加速时间，单位为秒。
MOVE_JOINT_ACCELERATION_SECONDS = 0.1
# 控制器规划运动的减速时间，单位为秒。
MOVE_JOINT_DECELERATION_SECONDS = 0.1
# 控制器规划运动的速度比例，范围为 (0, 1]。
MOVE_JOINT_SPEED_RATIO = 0.2
# 控制器规划运动的平滑等级，整数范围为 0 到 9。
MOVE_JOINT_SMOOTH = 1

# 真实运动的显式确认。仅在独立急停和硬件防护已确认后设为 True。
CONFIRM_MOTION = True
# 0.0 表示持续运行直到 Ctrl+C；正数表示自动结束的测试时长，单位为秒。
TEST_DURATION_S = 0.0


@dataclass(frozen=True)
class DemoConfig:
    """单臂 move_joints 遥操作所需配置，所有关节角度均为度。"""

    side: ArmSide
    openarm_port: str
    openarm_calibration_path: str
    openarm_baudrate: int
    hcx_local_ip: str
    hcx_remote_ip: str
    hcx_port: int
    hcx_connect_timeout_s: float
    hcx_robot_id: int
    axis_sign: tuple[float, ...]
    leader_sample_rate_hz: float
    leader_read_timeout_s: float
    move_joint_command_rate_hz: float
    move_joint_acceleration_seconds: float
    move_joint_deceleration_seconds: float
    move_joint_speed_ratio: float
    move_joint_smooth: int
    confirm_motion: bool
    test_duration_s: float

    def validate(self) -> None:
        if self.side not in ("left", "right"):
            raise ValueError("side 必须是 left 或 right")
        if not isinstance(self.openarm_port, str) or not self.openarm_port.strip():
            raise ValueError("openarm_port 不能为空")
        if (
            not isinstance(self.openarm_calibration_path, str)
            or not self.openarm_calibration_path.strip()
        ):
            raise ValueError("openarm_calibration_path 不能为空")
        if (
            not isinstance(self.openarm_baudrate, int)
            or isinstance(self.openarm_baudrate, bool)
            or self.openarm_baudrate <= 0
        ):
            raise ValueError("openarm_baudrate 必须是正整数")
        for name, value in (
            ("hcx_local_ip", self.hcx_local_ip),
            ("hcx_remote_ip", self.hcx_remote_ip),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} 不能为空")
        if (
            not isinstance(self.hcx_port, int)
            or isinstance(self.hcx_port, bool)
            or not 1 <= self.hcx_port <= 65535
        ):
            raise ValueError("hcx_port 必须是 1 到 65535 的整数")
        if (
            not isinstance(self.hcx_robot_id, int)
            or isinstance(self.hcx_robot_id, bool)
            or self.hcx_robot_id < 0
        ):
            raise ValueError("hcx_robot_id 必须是非负整数")
        if len(self.axis_sign) != JOINT_COUNT or any(
            sign not in (-1.0, 1.0) for sign in self.axis_sign
        ):
            raise ValueError(f"axis_sign 必须是 {JOINT_COUNT} 个 +1.0 或 -1.0")
        _positive_finite("leader_sample_rate_hz", self.leader_sample_rate_hz)
        _positive_finite("leader_read_timeout_s", self.leader_read_timeout_s)
        _positive_finite("hcx_connect_timeout_s", self.hcx_connect_timeout_s)
        _positive_finite(
            "move_joint_command_rate_hz", self.move_joint_command_rate_hz
        )
        _positive_finite(
            "move_joint_acceleration_seconds", self.move_joint_acceleration_seconds
        )
        _positive_finite(
            "move_joint_deceleration_seconds", self.move_joint_deceleration_seconds
        )
        speed_ratio = _positive_finite(
            "move_joint_speed_ratio", self.move_joint_speed_ratio
        )
        if speed_ratio > 1.0:
            raise ValueError("move_joint_speed_ratio 必须不大于 1")
        if (
            not isinstance(self.move_joint_smooth, int)
            or isinstance(self.move_joint_smooth, bool)
            or not 0 <= self.move_joint_smooth <= 9
        ):
            raise ValueError("move_joint_smooth 必须是 0 到 9 的整数")
        if not isinstance(self.confirm_motion, bool):
            raise ValueError("confirm_motion 必须是布尔值")
        if not self.confirm_motion:
            raise ValueError("请在文件顶部明确设置 CONFIRM_MOTION = True")
        _nonnegative_finite("test_duration_s", self.test_duration_s)


DEMO_CONFIG = DemoConfig(
    side=TEST_SIDE,
    openarm_port=OPENARM_PORT,
    openarm_calibration_path=OPENARM_CALIBRATION_PATH,
    openarm_baudrate=OPENARM_BAUDRATE,
    hcx_local_ip=HCX_LOCAL_IP,
    hcx_remote_ip=HCX_REMOTE_IP,
    hcx_port=HCX_PORT,
    hcx_connect_timeout_s=HCX_CONNECT_TIMEOUT_S,
    hcx_robot_id=HCX_ROBOT_ID,
    axis_sign=AXIS_SIGN,
    leader_sample_rate_hz=LEADER_SAMPLE_RATE_HZ,
    leader_read_timeout_s=LEADER_READ_TIMEOUT_S,
    move_joint_command_rate_hz=MOVE_JOINT_COMMAND_RATE_HZ,
    move_joint_acceleration_seconds=MOVE_JOINT_ACCELERATION_SECONDS,
    move_joint_deceleration_seconds=MOVE_JOINT_DECELERATION_SECONDS,
    move_joint_speed_ratio=MOVE_JOINT_SPEED_RATIO,
    move_joint_smooth=MOVE_JOINT_SMOOTH,
    confirm_motion=CONFIRM_MOTION,
    test_duration_s=TEST_DURATION_S,
)


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是正的有限数")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是正的有限数") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} 必须是正的有限数")
    return numeric


def _nonnegative_finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是非负有限数")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是非负有限数") from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} 必须是非负有限数")
    return numeric


def _as_joint_vector(name: str, values: object) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (JOINT_COUNT,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} 必须是 {JOINT_COUNT} 个有限关节角度")
    return vector


def map_relative_target(
    leader_angles_deg: np.ndarray,
    leader_origin_deg: np.ndarray,
    follower_origin_deg: np.ndarray,
    axis_sign: tuple[float, ...],
    min_angles_deg: np.ndarray,
    max_angles_deg: np.ndarray,
) -> np.ndarray:
    """将当前主臂相对零点的七轴变化映射为安全的从臂度制目标。"""

    leader = _as_joint_vector("leader_angles_deg", leader_angles_deg)
    leader_origin = _as_joint_vector("leader_origin_deg", leader_origin_deg)
    follower_origin = _as_joint_vector("follower_origin_deg", follower_origin_deg)
    lower = _as_joint_vector("min_angles_deg", min_angles_deg)
    upper = _as_joint_vector("max_angles_deg", max_angles_deg)
    if np.any(lower >= upper):
        raise ValueError("每个从臂关节必须满足 min_angles_deg < max_angles_deg")
    if len(axis_sign) != JOINT_COUNT or any(
        sign not in (-1.0, 1.0) for sign in axis_sign
    ):
        raise ValueError(f"axis_sign 必须是 {JOINT_COUNT} 个 +1.0 或 -1.0")
    target = follower_origin + np.asarray(axis_sign, dtype=float) * (
        leader - leader_origin
    )
    return np.clip(target, lower, upper)


def _read_follower_pose_and_limits(
    arm: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """启动前读取一次从臂姿态和关节限位。"""

    follower_origin = _as_joint_vector("HCX 当前关节角度", arm.joint_angles())
    limits = np.asarray(arm.joint_limits_deg, dtype=float)
    if limits.shape != (JOINT_COUNT, 2) or not np.isfinite(limits).all():
        raise RuntimeError("HCX 返回的七轴关节限位无效")
    lower = limits[:, 0].copy()
    upper = limits[:, 1].copy()
    if np.any(lower >= upper):
        raise RuntimeError("HCX 返回的关节限位必须满足 min < max")
    return follower_origin, lower, upper


def _submit_move_joints(arm: Any, target_deg: np.ndarray, config: DemoConfig) -> Any:
    """非阻塞提交一条可中断的控制器规划关节运动。"""

    target = _as_joint_vector("target_deg", target_deg)
    return arm.move_joints(
        target.tolist(),
        interrupt=True,
        acceleration_seconds=config.move_joint_acceleration_seconds,
        deceleration_seconds=config.move_joint_deceleration_seconds,
        speed_ratio=config.move_joint_speed_ratio,
        smooth=config.move_joint_smooth,
        wait=False,
    )


def run_demo(config: DemoConfig) -> None:
    """运行 100 Hz 主臂采样和低频 move_joints 规划目标替换。"""

    config.validate()
    leader = OpenArmMiniLeaderArm(
        config.openarm_port,
        Path(config.openarm_calibration_path),
        config.side,
        baudrate=config.openarm_baudrate,
        read_only=True,
    )
    connection = HcxConnection(
        HcxConnectionConfig(
            local_ip=config.hcx_local_ip,
            remote_ip=config.hcx_remote_ip,
            port=config.hcx_port,
            connect_timeout_s=config.hcx_connect_timeout_s,
        )
    )
    arm: Any | None = None
    motion_submitted = False

    try:
        leader.connect()
        arm = connection.acquire(config.hcx_robot_id)
        # 本示例不自动处理示教器、报警或使能；这里仅确认现场已经可运动。
        connection.prepare_for_motion(config.hcx_robot_id)
        follower_origin, lower, upper = _read_follower_pose_and_limits(arm)

        print("=" * 72)
        print("    OpenArm Mini -> HCX 单臂 move_joints 遥操作")
        print("=" * 72)
        print(f"  侧别: {config.side}；HCX robot_id: {config.hcx_robot_id}")
        print(f"  主臂采样: {config.leader_sample_rate_hz:.0f} Hz")
        print(
            f"  HCX move_joints: {config.move_joint_command_rate_hz:g} Hz "
            "（控制器规划，最新目标中断并替换当前路径）"
        )
        print(f"  轴方向: {list(config.axis_sign)}")
        print(
            "  规划参数: "
            f"accel={config.move_joint_acceleration_seconds:g} s, "
            f"decel={config.move_joint_deceleration_seconds:g} s, "
            f"speed_ratio={config.move_joint_speed_ratio:g}, "
            f"smooth={config.move_joint_smooth}"
        )
        print("  不启动直伺服、不使用插值或伺服保活。")
        print("  Ctrl+C 或测试结束时会清除本臂规划路径。")
        print("-" * 72)

        sample_period_s = 1.0 / config.leader_sample_rate_hz
        command_period_s = 1.0 / config.move_joint_command_rate_hz
        started_at_s = time.monotonic()
        next_sample_s = started_at_s
        next_command_s = started_at_s
        leader_origin: np.ndarray | None = None
        pending_target = follower_origin.copy()
        last_submitted_target = follower_origin.copy()

        while True:
            now_s = time.monotonic()
            if (
                config.test_duration_s > 0.0
                and now_s - started_at_s >= config.test_duration_s
            ):
                break

            leader_angles = leader.read_joint_angles_deg(config.leader_read_timeout_s)
            if leader_angles is not None:
                current = _as_joint_vector("OpenArm 当前关节角度", leader_angles)
                if leader_origin is None:
                    leader_origin = current.copy()
                    print(
                        "[INFO] 主臂起始位置已记录: "
                        f"{np.round(leader_origin, 1).tolist()}"
                    )
                    print("[INFO] 现在移动主臂，从臂将跟随相对变化量")
                pending_target = map_relative_target(
                    current,
                    leader_origin,
                    follower_origin,
                    config.axis_sign,
                    lower,
                    upper,
                )

            now_s = time.monotonic()
            if (
                now_s >= next_command_s
                and not np.array_equal(pending_target, last_submitted_target)
            ):
                _submit_move_joints(arm, pending_target, config)
                last_submitted_target = pending_target.copy()
                motion_submitted = True
                next_command_s = now_s + command_period_s

            next_sample_s += sample_period_s
            remaining_s = next_sample_s - time.monotonic()
            if remaining_s > 0.0:
                time.sleep(remaining_s)
            else:
                next_sample_s = time.monotonic()
    finally:
        if arm is not None and motion_submitted:
            try:
                arm.clear_route(emergency_stop=True)
            except (RuntimeError, TypeError, ValueError) as exc:
                print(f"[WARN] 清除 HCX 规划路径失败: {exc}")
        if arm is not None:
            try:
                connection.release(config.hcx_robot_id)
            except (RuntimeError, TypeError, ValueError) as exc:
                print(f"[WARN] 释放 HCX 连接失败: {exc}")
        leader.disconnect()


def main() -> int:
    """执行文件顶部定义的单臂规划运动对照；不加载 YAML，也不解析 CLI。"""

    try:
        run_demo(DEMO_CONFIG)
    except KeyboardInterrupt:
        print("\n[STOP] 收到退出请求，正在清除 HCX 单臂规划路径。")
        return 130
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f"[ERROR] 单臂 move_joints 遥操作失败: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
