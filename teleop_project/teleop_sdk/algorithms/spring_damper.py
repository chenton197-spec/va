"""临界阻尼弹簧跟踪算法。"""

from __future__ import annotations

import numpy as np


class SpringDamper:
    """原样迁移的临界阻尼弹簧关节跟踪器。

    算法只处理关节数组，不读取设备状态、不发送命令，也不拥有控制循环。
    首次 ``step()`` 调用或 ``reset()`` 后，使用 ``initial_angles`` 建立内部
    状态并原样返回该初始值；这一行为与原控制器保持一致。
    """

    def __init__(
        self,
        rate_hz: float,
        omega: float,
        jump_threshold_deg: float,
        max_accel_deg_s2: float,
        max_vel_deg_s: float,
        min_angles_deg: np.ndarray,
        max_angles_deg: np.ndarray,
    ):
        self.rate_hz = rate_hz
        self.omega = omega
        self.jump_threshold_deg = jump_threshold_deg
        self.max_accel_deg_s2 = max_accel_deg_s2
        self.max_vel_deg_s = max_vel_deg_s
        self.min_angles_deg = np.asarray(min_angles_deg, dtype=float).copy()
        self.max_angles_deg = np.asarray(max_angles_deg, dtype=float).copy()
        self._position: np.ndarray | None = None
        self._velocity = np.zeros_like(self.min_angles_deg, dtype=float)
        self._last_target: np.ndarray | None = None

    def reset(self) -> None:
        """清除位置和速度；下一帧将使用调用者提供的初始角度重新初始化。"""
        self._position = None
        self._velocity = np.zeros_like(self.min_angles_deg, dtype=float)
        self._last_target = None

    def step(self, target_angles: np.ndarray, initial_angles: np.ndarray) -> np.ndarray:
        """推进一个控制周期并返回平滑后的绝对关节目标。"""
        target = np.asarray(target_angles, dtype=float)
        if self._position is None:
            self._position = np.asarray(initial_angles, dtype=float).copy()
            self._velocity = np.zeros_like(self._position, dtype=float)
            self._last_target = target.copy()
            return self._position.copy()

        assert self._last_target is not None
        frame_delta = np.abs(target - self._last_target)
        glitch = frame_delta > self.jump_threshold_deg
        if np.any(glitch):
            print(
                "[WARN] 关节目标跳变，仅冻结该轴本帧: "
                + ", ".join(
                    f"J{index + 1}({frame_delta[index]:.1f}°)"
                    for index in np.where(glitch)[0]
                )
            )
            target = np.where(glitch, self._last_target, target)
        self._last_target = target.copy()

        dt = 1.0 / self.rate_hz
        error = target - self._position
        acceleration = self.omega**2 * error - 2.0 * self.omega * self._velocity
        acceleration = np.clip(
            acceleration,
            -self.max_accel_deg_s2,
            self.max_accel_deg_s2,
        )
        self._velocity = self._velocity + acceleration * dt
        self._velocity = np.clip(
            self._velocity,
            -self.max_vel_deg_s,
            self.max_vel_deg_s,
        )
        self._position = self._position + self._velocity * dt

        clipped = np.clip(self._position, self.min_angles_deg, self.max_angles_deg)
        at_limit = clipped != self._position
        self._velocity[at_limit] = 0.0
        self._position = clipped
        return self._position.copy()

    def predict(self, lookahead_s: float) -> np.ndarray:
        """按当前已限幅速度线性外推，并保持在关节限位内。"""
        if self._position is None or lookahead_s <= 0.0:
            return self._position.copy() if self._position is not None else np.zeros_like(self._velocity)
        return np.clip(
            self._position + self._velocity * lookahead_s,
            self.min_angles_deg,
            self.max_angles_deg,
        )
