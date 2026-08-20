"""从原遥操作脚本迁移的关节角度滤波器。"""

from __future__ import annotations

import numpy as np


class OneEuroFilter:
    """OneEuro 自适应低通滤波器。

    此实现保持原脚本的计算公式和状态更新顺序不变。
    """

    def __init__(
        self,
        n_joints: int = 6,
        mincutoff: float = 3.0,
        beta: float = 0.07,
        dcutoff: float = 1.0,
    ):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x: np.ndarray | None = None
        self._dx = np.zeros(n_joints)
        self._last_t: float | None = None

    def reset(self) -> None:
        """清除滤波器历史状态。"""
        self._x = None
        self._dx[:] = 0.0
        self._last_t = None

    def step(self, z: np.ndarray, t: float) -> np.ndarray:
        """输入一帧关节角度并返回滤波结果。"""
        z = np.asarray(z, dtype=float)
        if self._x is None or self._last_t is None:
            self._x = z.copy()
            self._last_t = t
            return self._x.copy()
        dt = max(1e-6, t - self._last_t)
        self._last_t = t
        # 先低通滤波关节速度，再依据速度调整截止频率。
        raw_dx = (z - self._x) / dt
        alpha_d = self._alpha(dt, self.dcutoff)
        self._dx = alpha_d * raw_dx + (1.0 - alpha_d) * self._dx
        cutoff = self.mincutoff + self.beta * np.abs(self._dx)
        alpha = self._alpha(dt, cutoff)
        self._x = alpha * z + (1.0 - alpha) * self._x
        return self._x.copy()

    @staticmethod
    def _alpha(dt: float, cutoff: float | np.ndarray) -> np.ndarray:
        tau = 1.0 / (2.0 * np.pi * np.asarray(cutoff, dtype=float))
        return dt / (dt + tau)


class LowPassFilter:
    """固定截止频率一阶 IIR 低通滤波器。"""

    def __init__(self, n_joints: int = 6, cutoff_hz: float = 3.0):
        self.cutoff_hz = cutoff_hz
        self._y: np.ndarray | None = None
        self._last_t: float | None = None

    def reset(self) -> None:
        """清除滤波器历史状态。"""
        self._y = None
        self._last_t = None

    def step(self, x: np.ndarray, t: float) -> np.ndarray:
        """输入一帧关节角度并返回滤波结果。"""
        x = np.asarray(x, dtype=float)
        if self._y is None or self._last_t is None:
            self._y = x.copy()
            self._last_t = t
            return self._y.copy()
        dt = max(1e-6, t - self._last_t)
        self._last_t = t
        tau = 1.0 / (2.0 * np.pi * self.cutoff_hz)
        alpha = dt / (dt + tau)
        self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y.copy()
