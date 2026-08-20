"""用于测试和演示的内存从臂适配器。"""

from __future__ import annotations

import threading
import time

import numpy as np

from ..interfaces import FollowerArm


class MockFollower(FollowerArm):
    """立即接受关节目标的无硬件从臂实现。"""

    def __init__(self, n_joints: int = 6):
        self._lock = threading.Lock()
        self._angles = np.zeros(n_joints, dtype=float)
        self._target = np.zeros(n_joints, dtype=float)
        self._servo_started = False
        self.command_count = 0
        self._last_command_time = 0.0

    @property
    def joint_count(self) -> int:
        return len(self._angles)

    @property
    def joint_limits_deg(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.full(self.joint_count, -360.0, dtype=float),
            np.full(self.joint_count, 360.0, dtype=float),
        )

    def connect(self) -> None:
        print("[SIM] 模拟从臂已连接")

    def read_joint_angles_deg(self) -> np.ndarray:
        with self._lock:
            return self._angles.copy()

    def start_servo(self) -> bool:
        self._servo_started = True
        return True

    def send_joint_angles_deg(self, angles_deg: np.ndarray, command_time_s: float) -> bool:
        if not self._servo_started:
            return False
        with self._lock:
            self._target = np.asarray(angles_deg, dtype=float).copy()
            self._angles = self._target.copy()
            self.command_count += 1
            self._last_command_time = time.perf_counter()
        return True

    def recover(self) -> bool:
        self._servo_started = True
        return True

    def stop_servo(self) -> None:
        self._servo_started = False

    def disconnect(self) -> None:
        # ``TeleopController.shutdown()`` already stops ServoMove before it
        # disconnects the adapter. Keep the mock's lifecycle equivalent so a
        # decorated test follower does not observe a duplicate stop request.
        self._servo_started = False

    def get_state(self) -> tuple[np.ndarray, np.ndarray, int]:
        """返回当前角度、最近目标和命令数量，供测试或可视化读取。"""
        with self._lock:
            return self._angles.copy(), self._target.copy(), self.command_count

    def get_visualization_state(self) -> tuple[np.ndarray, np.ndarray, int, float]:
        """返回供图形界面刷新使用的原子状态快照。"""
        with self._lock:
            return (
                self._angles.copy(),
                self._target.copy(),
                self.command_count,
                self._last_command_time,
            )
