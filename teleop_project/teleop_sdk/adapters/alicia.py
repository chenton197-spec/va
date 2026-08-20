"""Alicia-D 示教臂适配器。"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from ..interfaces import LeaderArmWithGripper


class AliciaLeaderArm(LeaderArmWithGripper):
    """将 Alicia SDK 的关节弧度读数转换为 SDK 标准角度读数。"""

    def __init__(
        self,
        port: str = "",
        gripper_type: str = "50mm",
        connect_retries: int = 5,
        connect_retry_delay_s: float = 3.0,
    ):
        self.port = port
        self.gripper_type = gripper_type
        self.connect_retries = connect_retries
        self.connect_retry_delay_s = connect_retry_delay_s
        self._robot: Any | None = None

    @property
    def joint_count(self) -> int:
        """Alicia-D 的固定六轴关节数量。"""
        return 6

    def connect(self) -> None:
        """按原脚本的重试策略连接 Alicia-D。"""
        from alicia_d_sdk import create_robot

        for attempt in range(1, self.connect_retries + 1):
            try:
                print(f"[INFO] 连接 Alicia-D 示教臂... (尝试 {attempt}/{self.connect_retries})")
                self._robot = create_robot(port=self.port, gripper_type=self.gripper_type)
                if self._robot.is_connected():
                    print("[INFO] Alicia-D 示教臂已连接")
                    return
                print("[WARN] 连接后 is_connected() 返回 False")
            except Exception as exc:
                print(f"[WARN] 连接失败: {exc}")

            if attempt < self.connect_retries:
                print(f"[INFO] {self.connect_retry_delay_s:.0f} 秒后重试，请确认示教臂已上电...")
                time.sleep(self.connect_retry_delay_s)

        raise RuntimeError(
            "Alicia-D 示教臂连接失败，已重试 "
            + str(self.connect_retries)
            + " 次。\n"
            "  - 请确认示教臂电源已打开\n"
            "  - 请确认 USB 线缆连接正常\n"
            "  - 可用 connect_retries / connect_retry_delay_s 调整重试参数"
        )

    def read_joint_angles_deg(self, timeout_s: float) -> np.ndarray | None:
        """读取 Alicia 原始弧度，并转换为公共接口规定的角度。"""
        if self._robot is None:
            return None
        try:
            state = self._robot.get_robot_state("joint_gripper", timeout=timeout_s)
        except Exception:
            return None
        if state is None:
            return None
        return np.array(state.angles, dtype=float) * (180.0 / math.pi)

    def read_gripper_opening(self, timeout_s: float) -> float | None:
        """读取 Alicia 夹爪值并转换为通用的 0-1 开合量。"""
        state = self._read_joint_gripper_state(timeout_s)
        if state is None:
            return None
        # 保留厂商异常值到滤波阶段；从端下发前才由控制器统一裁剪。
        return float(state.gripper) / 1000.0

    def read_joint_angles_and_gripper_opening(
        self, timeout_s: float = 0.1
    ) -> tuple[np.ndarray, float] | None:
        """用一次 Alicia 状态读取返回关节角度和归一化夹爪开合量。"""
        state = self._read_joint_gripper_state(timeout_s)
        if state is None:
            return None
        return (
            np.array(state.angles, dtype=float) * (180.0 / math.pi),
            float(state.gripper) / 1000.0,
        )

    def get_sync_lock_gripper(
        self, timeout_s: float = 1.0
    ) -> tuple[bool, bool, float] | None:
        """一次读取末端同步状态、锁定状态和夹爪值。

        返回顺序固定为 ``(is_synced, is_locked, gripper_value)``。其中
        ``gripper_value`` 的范围为 0-1000，0 表示完全关闭，1000 表示
        完全张开。该方法是 Alicia-D 专用扩展，不属于 ``LeaderArm`` 接口。
        """
        state = self._read_joint_gripper_state(timeout_s)
        if state is None:
            return None
        status = state.run_status_text
        return (
            status in {"sync", "sync_locked"},
            status in {"locked", "sync_locked"},
            float(state.gripper),
        )

    def read_joint_angles_and_sync_lock_gripper(
        self, timeout_s: float = 0.1
    ) -> tuple[np.ndarray, bool, bool, float] | None:
        """用一次 Alicia 状态读取同时取得关节、同步、锁定和夹爪值。"""
        state = self._read_joint_gripper_state(timeout_s)
        if state is None:
            return None
        status = state.run_status_text
        return (
            np.array(state.angles, dtype=float) * (180.0 / math.pi),
            status in {"sync", "sync_locked"},
            status in {"locked", "sync_locked"},
            float(state.gripper),
        )

    def _read_joint_gripper_state(self, timeout_s: float) -> Any | None:
        """读取 Alicia 的 joint_gripper 状态包。"""
        if self._robot is None:
            return None
        try:
            return self._robot.get_robot_state("joint_gripper", timeout=timeout_s)
        except Exception:
            return None

    def disconnect(self) -> None:
        """断开 Alicia-D 连接。"""
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None
