"""固定频率关节目标的线性重采样。"""

from __future__ import annotations

import math

import numpy as np


class LinearInterpolator:
    """将低频角度制关节目标线性重采样到整数倍输出频率。

    ``interpolate()`` 不包含起点、包含终点，因此相邻输入段不会重复发送
    边界点。例如 100 Hz 到 1000 Hz 时，每段返回 10 个目标。
    """

    def __init__(self, source_rate_hz: int | float, output_rate_hz: int | float):
        self.source_rate_hz = self._validate_rate("source_rate_hz", source_rate_hz)
        self.output_rate_hz = self._validate_rate("output_rate_hz", output_rate_hz)
        if self.output_rate_hz < self.source_rate_hz:
            raise ValueError("output_rate_hz must be greater than or equal to source_rate_hz")
        if self.output_rate_hz % self.source_rate_hz != 0:
            raise ValueError("output_rate_hz must be an integer multiple of source_rate_hz")
        self.samples_per_interval = self.output_rate_hz // self.source_rate_hz

    @staticmethod
    def _validate_rate(name: str, value: int | float) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if not math.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
            raise ValueError(f"{name} must be a positive integer")
        return int(numeric)

    def interpolate(
        self, start_angles_deg: np.ndarray, end_angles_deg: np.ndarray
    ) -> np.ndarray:
        """返回从 ``start`` 到 ``end`` 的一个输出周期目标段，单位为度。"""

        start = np.asarray(start_angles_deg, dtype=float)
        end = np.asarray(end_angles_deg, dtype=float)
        if start.ndim != 1 or end.ndim != 1 or start.shape != end.shape or start.size == 0:
            raise ValueError("start_angles_deg and end_angles_deg must be matching non-empty vectors")
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            raise ValueError("interpolation angles must be finite")

        fractions = np.arange(1, self.samples_per_interval + 1, dtype=float)
        fractions /= self.samples_per_interval
        return start[np.newaxis, :] + fractions[:, np.newaxis] * (end - start)
