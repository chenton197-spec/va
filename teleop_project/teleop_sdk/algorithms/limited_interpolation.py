"""以固定输出频率生成限速、限加速度的关节目标序列。"""

from __future__ import annotations

import math

import numpy as np


class LimitedInterpolator:
    """按输出周期执行低通、速度限制和加速度限制的关节轨迹生成器。

    输入目标通常以较低频率更新；每次 :meth:`interpolate` 保持该目标一个
    输入周期，并返回该周期内的所有高频输出点。算法状态连续保留，因此当
    目标变化过快时不会强行在一个输入周期内到达终点。
    """

    def __init__(
        self,
        source_rate_hz: int | float,
        output_rate_hz: int | float,
        *,
        max_velocity_deg_s: float,
        max_acceleration_deg_s2: float,
        lowpass_alpha: float,
        min_angles_deg: np.ndarray | None = None,
        max_angles_deg: np.ndarray | None = None,
    ) -> None:
        self.source_rate_hz = self._validate_rate("source_rate_hz", source_rate_hz)
        self.output_rate_hz = self._validate_rate("output_rate_hz", output_rate_hz)
        if self.output_rate_hz < self.source_rate_hz:
            raise ValueError(
                "output_rate_hz must be greater than or equal to source_rate_hz"
            )
        if self.output_rate_hz % self.source_rate_hz != 0:
            raise ValueError(
                "output_rate_hz must be an integer multiple of source_rate_hz"
            )
        self.samples_per_interval = self.output_rate_hz // self.source_rate_hz
        self.dt_s = 1.0 / self.output_rate_hz
        self.max_velocity_deg_s = self._validate_positive_finite(
            "max_velocity_deg_s", max_velocity_deg_s
        )
        self.max_acceleration_deg_s2 = self._validate_positive_finite(
            "max_acceleration_deg_s2", max_acceleration_deg_s2
        )
        self.lowpass_alpha = self._validate_alpha(lowpass_alpha)
        self._min_angles_deg, self._max_angles_deg = self._validate_limits(
            min_angles_deg, max_angles_deg
        )
        self._command: np.ndarray | None = None
        self._filtered_target: np.ndarray | None = None
        self._velocity_deg_s: np.ndarray | None = None

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

    @staticmethod
    def _validate_positive_finite(name: str, value: float) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive finite value")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive finite value") from exc
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{name} must be a positive finite value")
        return numeric

    @staticmethod
    def _validate_alpha(value: float) -> float:
        if isinstance(value, bool):
            raise ValueError("lowpass_alpha must be a finite value from 0 through 1")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "lowpass_alpha must be a finite value from 0 through 1"
            ) from exc
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError("lowpass_alpha must be a finite value from 0 through 1")
        return numeric

    @staticmethod
    def _validate_angles(name: str, angles_deg: np.ndarray) -> np.ndarray:
        values = np.asarray(angles_deg, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional vector")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
        return values

    @classmethod
    def _validate_limits(
        cls, min_angles_deg: np.ndarray | None, max_angles_deg: np.ndarray | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if min_angles_deg is None and max_angles_deg is None:
            return None, None
        if min_angles_deg is None or max_angles_deg is None:
            raise ValueError(
                "min_angles_deg and max_angles_deg must be provided together"
            )
        minimum = cls._validate_angles("min_angles_deg", min_angles_deg)
        maximum = cls._validate_angles("max_angles_deg", max_angles_deg)
        if minimum.shape != maximum.shape or np.any(minimum >= maximum):
            raise ValueError("joint limits must have matching shapes with min < max")
        return minimum.copy(), maximum.copy()

    def reset(self, initial_angles_deg: np.ndarray) -> None:
        """以当前安全保持姿态重置位置、低通目标和速度状态。"""

        initial = self._validate_angles("initial_angles_deg", initial_angles_deg)
        if (
            self._min_angles_deg is not None
            and initial.shape != self._min_angles_deg.shape
        ):
            raise ValueError(
                "initial_angles_deg must match the joint-limit vector shape"
            )
        if self._min_angles_deg is not None and (
            np.any(initial < self._min_angles_deg)
            or np.any(initial > self._max_angles_deg)
        ):
            raise ValueError(
                "initial_angles_deg must be within the configured joint limits"
            )
        self._command = initial.copy()
        self._filtered_target = initial.copy()
        self._velocity_deg_s = np.zeros_like(initial)

    @property
    def initialized(self) -> bool:
        """是否已通过 :meth:`reset` 建立连续轨迹状态。"""

        return self._command is not None

    def step(
        self, target_angles_deg: np.ndarray, *, elapsed_s: float | None = None
    ) -> np.ndarray:
        """积分一个限速、限加速度输出点，单位为度。

        ``elapsed_s`` 省略时使用标称输出周期。实时发送线程可传入实际经过
        时间，保持速度和加速度单位为每秒；调用方应自行限制异常长的调度停顿，
        以避免在一次恢复调用中形成过大的位置步长。
        """

        target = self._validate_angles("target_angles_deg", target_angles_deg)
        if (
            self._command is None
            or self._filtered_target is None
            or self._velocity_deg_s is None
        ):
            raise RuntimeError("LimitedInterpolator must be reset before step")
        if target.shape != self._command.shape:
            raise ValueError("target_angles_deg must match the reset vector shape")
        if elapsed_s is None:
            dt_s = self.dt_s
        else:
            if isinstance(elapsed_s, bool):
                raise ValueError("elapsed_s must be a positive finite value")
            try:
                dt_s = float(elapsed_s)
            except (TypeError, ValueError) as exc:
                raise ValueError("elapsed_s must be a positive finite value") from exc
            if not math.isfinite(dt_s) or dt_s <= 0.0:
                raise ValueError("elapsed_s must be a positive finite value")

        if self._min_angles_deg is not None:
            target = np.clip(target, self._min_angles_deg, self._max_angles_deg)

        # alpha 表示一个标称输出周期的低通权重。时间不完全等于标称周期时，
        # 用等价连续衰减保持相同的时间常数。
        alpha = 1.0 - (1.0 - self.lowpass_alpha) ** (dt_s / self.dt_s)
        self._filtered_target = (
            alpha * target + (1.0 - alpha) * self._filtered_target
        )
        error = self._filtered_target - self._command
        # 用离散积分的剩余距离计算可安全刹停的下一周期最大速度。相比
        # ``error / dt``，该约束会在最大制动距离之前开始减速；``a * dt``
        # 项还避免最后一个周期硬置零而突破加速度上限。
        acceleration_step = self.max_acceleration_deg_s2 * dt_s
        stopping_speed = (
            np.sqrt(
                acceleration_step**2
                + 8.0 * self.max_acceleration_deg_s2 * np.abs(error)
            )
            - acceleration_step
        ) / 2.0
        desired_speed = np.minimum(self.max_velocity_deg_s, stopping_speed)
        desired_velocity = np.copysign(desired_speed, error)
        acceleration = (desired_velocity - self._velocity_deg_s) / dt_s
        acceleration = np.clip(
            acceleration,
            -self.max_acceleration_deg_s2,
            self.max_acceleration_deg_s2,
        )
        self._velocity_deg_s = self._velocity_deg_s + acceleration * dt_s
        self._velocity_deg_s = np.clip(
            self._velocity_deg_s,
            -self.max_velocity_deg_s,
            self.max_velocity_deg_s,
        )
        self._command = self._command + self._velocity_deg_s * dt_s
        if self._min_angles_deg is not None:
            clipped_command = np.clip(
                self._command, self._min_angles_deg, self._max_angles_deg
            )
            at_limit = clipped_command != self._command
            self._velocity_deg_s[at_limit] = 0.0
            self._command = clipped_command
        return self._command.copy()

    def interpolate(self, target_angles_deg: np.ndarray) -> np.ndarray:
        """返回一个输入周期内的高频限幅目标序列，单位为度。"""

        target = self._validate_angles("target_angles_deg", target_angles_deg)
        points = np.empty((self.samples_per_interval, target.size), dtype=float)
        for index in range(self.samples_per_interval):
            points[index] = self.step(target)
        return points
