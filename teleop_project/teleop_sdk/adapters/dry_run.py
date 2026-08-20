"""空跑从臂适配器。"""

from __future__ import annotations

import numpy as np

from ..interfaces import FollowerArm


class DryRunFollower(FollowerArm):
    """不连接任何硬件，只保存最近一帧目标角度。"""

    def __init__(self, n_joints: int = 6):
        self._angles = np.zeros(n_joints, dtype=float)
        self.command_history: list[np.ndarray] = []

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
        print("[INFO] 空跑模式：不连接从臂，只打印目标角度")

    def read_joint_angles_deg(self) -> np.ndarray:
        return self._angles.copy()

    def start_servo(self) -> bool:
        return True

    def send_joint_angles_deg(self, angles_deg: np.ndarray, command_time_s: float) -> bool:
        self._angles = np.asarray(angles_deg, dtype=float).copy()
        self.command_history.append(self._angles.copy())
        print(f"[ACTION] 从臂目标: {np.round(self._angles, 2).tolist()}")
        return True

    def recover(self) -> bool:
        return True

    def stop_servo(self) -> None:
        return None

    def disconnect(self) -> None:
        return None
