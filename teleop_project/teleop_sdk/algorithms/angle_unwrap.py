"""关节角度连续化算法。"""

from __future__ import annotations

import numpy as np


class AngleUnwrapper:
    """消除旋转关节跨越 +/-180 度时产生的 +/-360 度读数回绕。"""

    def __init__(self, n_joints: int):
        self._previous: np.ndarray | None = None
        self._offset = np.zeros(n_joints, dtype=float)

    def reset(self) -> None:
        """清除历史读数和累计补偿。"""
        self._previous = None
        self._offset.fill(0.0)

    def step(self, raw_angles_deg: np.ndarray) -> np.ndarray:
        """返回与历史读数连续的关节角度。"""
        raw = np.asarray(raw_angles_deg, dtype=float)
        if self._previous is not None:
            difference = raw - self._previous
            self._offset[difference > 180.0] -= 360.0
            self._offset[difference < -180.0] += 360.0
        self._previous = raw.copy()
        return raw + self._offset
