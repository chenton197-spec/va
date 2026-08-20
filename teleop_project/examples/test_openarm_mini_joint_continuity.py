#!/usr/bin/env python3
"""验证单侧 OpenArm Mini 关节采样是否在单向移动时发生折返。

本程序只读取一个 OpenArm Mini 的七轴角度，不连接 HCX，不写 OpenArm
寄存器，不使用滤波、映射、插值或任何运动命令。它检查的是
``OpenArmMiniLeaderArm.read_joint_angles_deg()`` 输出，也就是遥操作控制链
实际接收到的度制主臂样本。

测试方法：选择一个关节后，只沿一个方向稳定移动该关节，不要人为反向。程序会
用第一段有效位移自动判定运动趋势；从最近峰值或谷值反向超过
``REVERSAL_CONFIRM_DEG`` 时，才记为一次确认折返。所有原始七轴数据都会写入
CSV，方便核查短暂但未达到确认阈值的反向样本。

运行：

    python -m examples.test_openarm_mini_joint_continuity
"""

from __future__ import annotations

import csv
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Protocol

import numpy as np

from teleop_sdk.adapters import OpenArmMiniLeaderArm


JOINT_COUNT = 7
ArmSide = Literal["left", "right"]

# 要测试的 OpenArm Mini 侧别；该 demo 一次只读取一侧，避免另一条串口影响结果。
TEST_SIDE: ArmSide = "right"
# 对应 TEST_SIDE 的 OpenArm Mini 只读串口。当前右侧通常为 /dev/ttyACM0，
# 左侧通常为 /dev/ttyACM1；请按现场实际设备路径修改。
OPENARM_PORT = "/dev/ttyACM0"
# 左右组合标定 JSON；本 demo 仅读取，不会写入或重标定。
OPENARM_CALIBRATION_PATH = "./my_openarm_mini.json"
# OpenArm Mini 串口波特率。
OPENARM_BAUDRATE = 1_000_000

# 要检查的关节编号，按 J1-J7 填写 1-7。测试时只单向移动这一轴。
TEST_JOINT_NUMBER = 1
# 读取频率，单位为 Hz；与当前遥操作主臂采样频率保持一致。
SAMPLE_RATE_HZ = 100.0
# 单次读取最多等待的时间，单位为秒。
READ_TIMEOUT_S = 0.008
# 测试时长，单位为秒；0.0 表示持续到 Ctrl+C。
TEST_DURATION_S = 30.0
# 终端汇总输出间隔，单位为秒；不逐帧打印，避免打印本身干扰采样。
SUMMARY_INTERVAL_S = 1.0

# 相邻样本变化不超过该值时，计为量化平台或微小噪声，单位为度。
SAMPLE_DEADBAND_DEG = 0.10
# 累积位移达到该值后，才建立“当前应持续单向移动”的趋势，单位为度。
TREND_START_DEG = 0.30
# 从最近峰值/谷值朝反方向回退达到该值时，确认记录一次折返，单位为度。
REVERSAL_CONFIRM_DEG = 0.50
# 单个 100 Hz 样本步长达到该值时，额外标记为大跳变，单位为度。
LARGE_SAMPLE_STEP_DEG = 3.0
# 有效样本间隔超过标称周期的该倍数时，标记为采样间隙异常。
GAP_WARNING_MULTIPLIER = 2.0

# 原始记录 CSV 路径。空字符串表示不导出；文件包含每帧七轴角度和选中轴的
# 增量、趋势、回退量及折返标记。
TRACE_CSV_PATH = "./csv/openarm_mini_joint_continuity.csv"


class _LeaderReader(Protocol):
    """本诊断需要的 OpenArm 只读接口。"""

    @property
    def joint_count(self) -> int: ...

    def connect(self) -> None: ...

    def read_joint_angles_deg(self, timeout_s: float) -> np.ndarray | None: ...

    def disconnect(self) -> None: ...


@dataclass(frozen=True)
class ContinuityConfig:
    """单关节采样连续性判定参数。"""

    joint_number: int
    sample_rate_hz: float
    read_timeout_s: float
    test_duration_s: float
    summary_interval_s: float
    sample_deadband_deg: float
    trend_start_deg: float
    reversal_confirm_deg: float
    large_sample_step_deg: float
    gap_warning_multiplier: float

    def validate(self) -> None:
        if (
            not isinstance(self.joint_number, int)
            or isinstance(self.joint_number, bool)
            or not 1 <= self.joint_number <= JOINT_COUNT
        ):
            raise ValueError(f"joint_number 必须是 1 到 {JOINT_COUNT} 的整数")
        for name, value in (
            ("sample_rate_hz", self.sample_rate_hz),
            ("read_timeout_s", self.read_timeout_s),
            ("summary_interval_s", self.summary_interval_s),
            ("trend_start_deg", self.trend_start_deg),
            ("reversal_confirm_deg", self.reversal_confirm_deg),
            ("large_sample_step_deg", self.large_sample_step_deg),
        ):
            _positive_finite(name, value)
        _nonnegative_finite("test_duration_s", self.test_duration_s)
        _nonnegative_finite("sample_deadband_deg", self.sample_deadband_deg)
        if self.trend_start_deg < self.sample_deadband_deg:
            raise ValueError("trend_start_deg 必须不小于 sample_deadband_deg")
        if self.reversal_confirm_deg < self.sample_deadband_deg:
            raise ValueError("reversal_confirm_deg 必须不小于 sample_deadband_deg")
        if (
            not math.isfinite(self.gap_warning_multiplier)
            or self.gap_warning_multiplier < 1.0
        ):
            raise ValueError("gap_warning_multiplier 必须是不小于 1 的有限数")

    @property
    def joint_index(self) -> int:
        return self.joint_number - 1

    @property
    def period_s(self) -> float:
        return 1.0 / self.sample_rate_hz

    @property
    def gap_warning_s(self) -> float:
        return self.period_s * self.gap_warning_multiplier


CONTINUITY_CONFIG = ContinuityConfig(
    joint_number=TEST_JOINT_NUMBER,
    sample_rate_hz=SAMPLE_RATE_HZ,
    read_timeout_s=READ_TIMEOUT_S,
    test_duration_s=TEST_DURATION_S,
    summary_interval_s=SUMMARY_INTERVAL_S,
    sample_deadband_deg=SAMPLE_DEADBAND_DEG,
    trend_start_deg=TREND_START_DEG,
    reversal_confirm_deg=REVERSAL_CONFIRM_DEG,
    large_sample_step_deg=LARGE_SAMPLE_STEP_DEG,
    gap_warning_multiplier=GAP_WARNING_MULTIPLIER,
)


@dataclass(frozen=True)
class ReversalEvent:
    """确认到一次持续反向位移时的证据。"""

    sample_index: int
    timestamp_s: float
    old_direction: int
    new_direction: int
    extreme_angle_deg: float
    observed_angle_deg: float
    backtrack_deg: float


@dataclass(frozen=True)
class TraceRecord:
    """一次采样尝试的原始角度与连续性分析结果。"""

    sample_index: int
    completed_at_s: float
    read_duration_s: float
    status: str
    angles_deg: np.ndarray | None
    selected_angle_deg: float | None
    sample_gap_s: float | None
    delta_deg: float | None
    trend_direction: int
    opposite_excursion_deg: float | None
    large_step: bool
    reversal_event: ReversalEvent | None
    detail: str


@dataclass
class JointContinuityStats:
    """选定关节的原始样本、趋势和折返统计。"""

    joint_index: int
    attempted_count: int = 0
    valid_count: int = 0
    none_frame_count: int = 0
    invalid_frame_count: int = 0
    exception_count: int = 0
    same_step_count: int = 0
    changed_step_count: int = 0
    large_step_count: int = 0
    gap_warning_count: int = 0
    total_read_duration_s: float = 0.0
    max_read_duration_s: float = 0.0
    max_sample_gap_s: float = 0.0
    max_abs_step_deg: float = 0.0
    max_opposite_excursion_deg: float = 0.0
    min_angle_deg: float | None = None
    max_angle_deg: float | None = None
    events: list[ReversalEvent] = field(default_factory=list)
    records: list[TraceRecord] = field(default_factory=list)
    _valid_timestamps_s: list[float] = field(default_factory=list)
    _last_angles_deg: np.ndarray | None = None
    _last_selected_angle_deg: float | None = None
    _trend_anchor_deg: float | None = None
    _trend_direction: int = 0
    _trend_extreme_deg: float | None = None

    def record(
        self,
        *,
        completed_at_s: float,
        read_duration_s: float,
        status: str,
        angles_deg: np.ndarray | None,
        detail: str,
        config: ContinuityConfig,
    ) -> ReversalEvent | None:
        """记录一帧，并在出现持续反向位移时返回事件。"""

        self.attempted_count += 1
        sample_index = self.attempted_count
        duration_s = max(0.0, float(read_duration_s))
        self.total_read_duration_s += duration_s
        self.max_read_duration_s = max(self.max_read_duration_s, duration_s)

        if status != "ok" or angles_deg is None:
            if status == "none":
                self.none_frame_count += 1
            elif status == "invalid":
                self.invalid_frame_count += 1
            else:
                self.exception_count += 1
            self.records.append(
                TraceRecord(
                    sample_index=sample_index,
                    completed_at_s=completed_at_s,
                    read_duration_s=duration_s,
                    status=status,
                    angles_deg=None,
                    selected_angle_deg=None,
                    sample_gap_s=None,
                    delta_deg=None,
                    trend_direction=self._trend_direction,
                    opposite_excursion_deg=None,
                    large_step=False,
                    reversal_event=None,
                    detail=detail,
                )
            )
            return None

        values = _as_joint_vector("angles_deg", angles_deg).copy()
        selected = float(values[self.joint_index])
        self.valid_count += 1
        self.min_angle_deg = (
            selected
            if self.min_angle_deg is None
            else min(self.min_angle_deg, selected)
        )
        self.max_angle_deg = (
            selected
            if self.max_angle_deg is None
            else max(self.max_angle_deg, selected)
        )

        sample_gap_s: float | None = None
        delta_deg: float | None = None
        large_step = False
        if self._valid_timestamps_s:
            sample_gap_s = completed_at_s - self._valid_timestamps_s[-1]
            self.max_sample_gap_s = max(self.max_sample_gap_s, sample_gap_s)
            if sample_gap_s > config.gap_warning_s:
                self.gap_warning_count += 1
            assert self._last_selected_angle_deg is not None
            delta_deg = selected - self._last_selected_angle_deg
            abs_delta = abs(delta_deg)
            self.max_abs_step_deg = max(self.max_abs_step_deg, abs_delta)
            large_step = abs_delta >= config.large_sample_step_deg
            if abs_delta <= config.sample_deadband_deg:
                self.same_step_count += 1
            else:
                self.changed_step_count += 1
            if large_step:
                self.large_step_count += 1

        event, opposite_excursion = self._update_trend(
            selected,
            completed_at_s,
            sample_index,
            config,
        )
        if opposite_excursion is not None:
            self.max_opposite_excursion_deg = max(
                self.max_opposite_excursion_deg,
                opposite_excursion,
            )
        if event is not None:
            self.events.append(event)

        self._valid_timestamps_s.append(completed_at_s)
        self._last_angles_deg = values
        self._last_selected_angle_deg = selected
        self.records.append(
            TraceRecord(
                sample_index=sample_index,
                completed_at_s=completed_at_s,
                read_duration_s=duration_s,
                status="ok",
                angles_deg=values,
                selected_angle_deg=selected,
                sample_gap_s=sample_gap_s,
                delta_deg=delta_deg,
                trend_direction=self._trend_direction,
                opposite_excursion_deg=opposite_excursion,
                large_step=large_step,
                reversal_event=event,
                detail=detail,
            )
        )
        return event

    def _update_trend(
        self,
        angle_deg: float,
        timestamp_s: float,
        sample_index: int,
        config: ContinuityConfig,
    ) -> tuple[ReversalEvent | None, float | None]:
        """使用峰值/谷值滞回识别持续反向，而非量化噪声。"""

        if self._trend_anchor_deg is None:
            self._trend_anchor_deg = angle_deg
            self._trend_extreme_deg = angle_deg
            return None, None

        assert self._trend_extreme_deg is not None
        if self._trend_direction == 0:
            offset_deg = angle_deg - self._trend_anchor_deg
            if abs(offset_deg) >= config.trend_start_deg:
                self._trend_direction = 1 if offset_deg > 0.0 else -1
                self._trend_extreme_deg = angle_deg
            return None, None

        if self._trend_direction > 0:
            if angle_deg >= self._trend_extreme_deg:
                self._trend_extreme_deg = angle_deg
                return None, None
            backtrack_deg = self._trend_extreme_deg - angle_deg
            if backtrack_deg < config.reversal_confirm_deg:
                return None, backtrack_deg
            event = ReversalEvent(
                sample_index=sample_index,
                timestamp_s=timestamp_s,
                old_direction=1,
                new_direction=-1,
                extreme_angle_deg=self._trend_extreme_deg,
                observed_angle_deg=angle_deg,
                backtrack_deg=backtrack_deg,
            )
            self._trend_direction = -1
            self._trend_extreme_deg = angle_deg
            return event, backtrack_deg

        if angle_deg <= self._trend_extreme_deg:
            self._trend_extreme_deg = angle_deg
            return None, None
        backtrack_deg = angle_deg - self._trend_extreme_deg
        if backtrack_deg < config.reversal_confirm_deg:
            return None, backtrack_deg
        event = ReversalEvent(
            sample_index=sample_index,
            timestamp_s=timestamp_s,
            old_direction=-1,
            new_direction=1,
            extreme_angle_deg=self._trend_extreme_deg,
            observed_angle_deg=angle_deg,
            backtrack_deg=backtrack_deg,
        )
        self._trend_direction = 1
        self._trend_extreme_deg = angle_deg
        return event, backtrack_deg

    @property
    def observed_rate_hz(self) -> float | None:
        if len(self._valid_timestamps_s) < 2:
            return None
        elapsed_s = self._valid_timestamps_s[-1] - self._valid_timestamps_s[0]
        if elapsed_s <= 0.0:
            return None
        return (len(self._valid_timestamps_s) - 1) / elapsed_s

    @property
    def mean_read_duration_s(self) -> float | None:
        if self.attempted_count == 0:
            return None
        return self.total_read_duration_s / self.attempted_count

    @property
    def trend_direction(self) -> int:
        return self._trend_direction


@dataclass(frozen=True)
class ContinuityReport:
    """一次单关节采样连续性测试的结果。"""

    config: ContinuityConfig
    elapsed_s: float
    cycle_count: int
    missed_tick_count: int
    maximum_lateness_s: float
    interrupted: bool
    stats: JointContinuityStats

    @property
    def observed_loop_rate_hz(self) -> float | None:
        if self.elapsed_s <= 0.0:
            return None
        return self.cycle_count / self.elapsed_s


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


def _validate_connection_config() -> None:
    if TEST_SIDE not in ("left", "right"):
        raise ValueError("TEST_SIDE 必须是 left 或 right")
    if not isinstance(OPENARM_PORT, str) or not OPENARM_PORT.strip():
        raise ValueError("OPENARM_PORT 不能为空")
    if (
        not isinstance(OPENARM_CALIBRATION_PATH, str)
        or not OPENARM_CALIBRATION_PATH.strip()
    ):
        raise ValueError("OPENARM_CALIBRATION_PATH 不能为空")
    if not Path(OPENARM_CALIBRATION_PATH).expanduser().is_file():
        raise ValueError(
            f"找不到 OpenArm Mini 标定文件: {OPENARM_CALIBRATION_PATH}"
        )
    if (
        not isinstance(OPENARM_BAUDRATE, int)
        or isinstance(OPENARM_BAUDRATE, bool)
        or OPENARM_BAUDRATE <= 0
    ):
        raise ValueError("OPENARM_BAUDRATE 必须是正整数")


def _configured_csv_path() -> Path | None:
    if not isinstance(TRACE_CSV_PATH, str):
        raise ValueError("TRACE_CSV_PATH 必须是字符串")
    normalized = TRACE_CSV_PATH.strip()
    return Path(normalized) if normalized else None


def _read_once(
    leader: _LeaderReader,
    timeout_s: float,
    *,
    clock: Callable[[], float],
) -> tuple[float, float, str, np.ndarray | None, str]:
    """读取一帧并将异常转换为可统计的状态，不中止诊断循环。"""

    started_at_s = clock()
    try:
        angles = leader.read_joint_angles_deg(timeout_s)
    except Exception as exc:
        completed_at_s = clock()
        return completed_at_s, completed_at_s - started_at_s, "error", None, str(exc)
    completed_at_s = clock()
    if angles is None:
        return completed_at_s, completed_at_s - started_at_s, "none", None, ""
    try:
        values = _as_joint_vector("OpenArm 当前关节角度", angles)
    except (TypeError, ValueError) as exc:
        return completed_at_s, completed_at_s - started_at_s, "invalid", None, str(exc)
    return completed_at_s, completed_at_s - started_at_s, "ok", values, ""


def _build_report(
    config: ContinuityConfig,
    *,
    started_at_s: float,
    completed_at_s: float,
    cycle_count: int,
    missed_tick_count: int,
    maximum_lateness_s: float,
    interrupted: bool,
    stats: JointContinuityStats,
) -> ContinuityReport:
    return ContinuityReport(
        config=config,
        elapsed_s=max(0.0, completed_at_s - started_at_s),
        cycle_count=cycle_count,
        missed_tick_count=missed_tick_count,
        maximum_lateness_s=maximum_lateness_s,
        interrupted=interrupted,
        stats=stats,
    )


def run_continuity_test(
    leader: _LeaderReader,
    config: ContinuityConfig,
    *,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    on_summary: Callable[[ContinuityReport], None] | None = None,
    on_reversal: Callable[[ReversalEvent], None] | None = None,
) -> ContinuityReport:
    """固定频率读取一侧主臂，并识别选定关节的持续反向位移。"""

    config.validate()
    if leader.joint_count != JOINT_COUNT:
        raise ValueError(f"OpenArm Mini 必须是 {JOINT_COUNT} 轴")

    stats = JointContinuityStats(config.joint_index)
    started_at_s = clock()
    next_deadline_s = started_at_s
    next_summary_s = started_at_s + config.summary_interval_s
    end_at_s = (
        math.inf
        if config.test_duration_s == 0.0
        else started_at_s + config.test_duration_s
    )
    cycle_count = 0
    missed_tick_count = 0
    maximum_lateness_s = 0.0
    interrupted = False

    try:
        while clock() < end_at_s:
            completed_at_s, duration_s, status, angles, detail = _read_once(
                leader,
                config.read_timeout_s,
                clock=clock,
            )
            event = stats.record(
                completed_at_s=completed_at_s,
                read_duration_s=duration_s,
                status=status,
                angles_deg=angles,
                detail=detail,
                config=config,
            )
            cycle_count += 1
            if event is not None and on_reversal is not None:
                on_reversal(event)

            finished_at_s = clock()
            if finished_at_s >= next_summary_s:
                if on_summary is not None:
                    on_summary(
                        _build_report(
                            config,
                            started_at_s=started_at_s,
                            completed_at_s=finished_at_s,
                            cycle_count=cycle_count,
                            missed_tick_count=missed_tick_count,
                            maximum_lateness_s=maximum_lateness_s,
                            interrupted=False,
                            stats=stats,
                        )
                    )
                next_summary_s = finished_at_s + config.summary_interval_s

            next_deadline_s += config.period_s
            delay_s = next_deadline_s - finished_at_s
            if delay_s > 0.0:
                sleep(delay_s)
            else:
                missed_ticks = int((-delay_s) / config.period_s)
                if missed_ticks:
                    missed_tick_count += missed_ticks
                    maximum_lateness_s = max(maximum_lateness_s, -delay_s)
                next_deadline_s = finished_at_s
    except KeyboardInterrupt:
        interrupted = True

    return _build_report(
        config,
        started_at_s=started_at_s,
        completed_at_s=clock(),
        cycle_count=cycle_count,
        missed_tick_count=missed_tick_count,
        maximum_lateness_s=maximum_lateness_s,
        interrupted=interrupted,
        stats=stats,
    )


def _format_rate(rate_hz: float | None) -> str:
    return "--" if rate_hz is None else f"{rate_hz:.1f}"


def _format_duration(duration_s: float | None) -> str:
    return "--" if duration_s is None else f"{duration_s * 1_000.0:.2f} ms"


def _direction_text(direction: int) -> str:
    if direction > 0:
        return "+"
    if direction < 0:
        return "-"
    return "未建立"


def print_reversal_event(event: ReversalEvent) -> None:
    """打印确认折返，不逐帧打印普通样本。"""

    print(
        f"[REVERSAL] sample={event.sample_index} "
        f"{_direction_text(event.old_direction)} -> {_direction_text(event.new_direction)} "
        f"peak/trough={event.extreme_angle_deg:.3f} deg "
        f"now={event.observed_angle_deg:.3f} deg "
        f"backtrack={event.backtrack_deg:.3f} deg"
    )


def print_live_report(report: ContinuityReport) -> None:
    """每秒输出一次选定关节的采样与折返统计。"""

    stats = report.stats
    latest = (
        "--"
        if stats._last_selected_angle_deg is None
        else f"{stats._last_selected_angle_deg:.3f} deg"
    )
    print(
        f"[OpenArm Trace] J{report.config.joint_number} "
        f"loop={_format_rate(report.observed_loop_rate_hz)}/"
        f"{report.config.sample_rate_hz:.0f}Hz "
        f"valid={stats.valid_count}/{stats.attempted_count} "
        f"latest={latest} trend={_direction_text(stats.trend_direction)}"
    )
    print(
        "  "
        f"read avg/max={_format_duration(stats.mean_read_duration_s)}/"
        f"{_format_duration(stats.max_read_duration_s)} "
        f"gap-max={_format_duration(stats.max_sample_gap_s)} "
        f"gap-warning={stats.gap_warning_count} miss={report.missed_tick_count}"
    )
    print(
        "  "
        f"range={_range_text(stats)} max-step={stats.max_abs_step_deg:.3f} deg "
        f"max-backtrack={stats.max_opposite_excursion_deg:.3f} deg "
        f"confirmed-reversals={len(stats.events)} "
        f"large-step={stats.large_step_count} same={stats.same_step_count}"
    )


def _range_text(stats: JointContinuityStats) -> str:
    if stats.min_angle_deg is None or stats.max_angle_deg is None:
        return "--"
    return f"{stats.max_angle_deg - stats.min_angle_deg:.3f} deg"


def print_final_report(report: ContinuityReport) -> None:
    """输出判定结果，并说明折返结论成立的前提。"""

    stats = report.stats
    print("=" * 72)
    print("    OpenArm Mini 单关节采样连续性结果")
    print("=" * 72)
    print(
        f"  测试 J{report.config.joint_number}，时长={report.elapsed_s:.3f} s，"
        f"样本={stats.valid_count}/{stats.attempted_count}"
    )
    print_live_report(report)
    if stats.none_frame_count or stats.invalid_frame_count or stats.exception_count:
        print(
            "  [WARN] "
            f"none={stats.none_frame_count} invalid={stats.invalid_frame_count} "
            f"error={stats.exception_count}"
        )
    if stats.events:
        print(
            "  [WARN] 在单向测试假设下检测到 "
            f"{len(stats.events)} 次确认折返；请核对 CSV 中对应样本。"
        )
    else:
        print(
            "  [RESULT] 未检测到超过确认阈值的折返。该结论仅在测试期间该关节"
            "确实保持单向移动时成立。"
        )
    print(
        "  原始数据未经滤波、轴映射或插值；它反映遥操作读取接口输出的关节角度。"
    )


def write_csv(path: Path, report: ContinuityReport) -> None:
    """导出逐帧原始七轴样本和选中关节的连续性标记。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "sample_index",
        "completion_time_s",
        "read_duration_ms",
        "status",
        "selected_joint",
        "selected_angle_deg",
        "sample_gap_ms",
        "delta_deg",
        "trend_direction",
        "opposite_excursion_deg",
        "large_step",
        "reversal",
        "reversal_backtrack_deg",
        "detail",
    ]
    header.extend(f"j{axis + 1}_deg" for axis in range(JOINT_COUNT))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        for record in report.stats.records:
            event = record.reversal_event
            angles = (
                [""] * JOINT_COUNT
                if record.angles_deg is None
                else record.angles_deg.tolist()
            )
            writer.writerow(
                (
                    record.sample_index,
                    record.completed_at_s,
                    record.read_duration_s * 1_000.0,
                    record.status,
                    report.config.joint_number,
                    ""
                    if record.selected_angle_deg is None
                    else record.selected_angle_deg,
                    "" if record.sample_gap_s is None else record.sample_gap_s * 1_000.0,
                    "" if record.delta_deg is None else record.delta_deg,
                    record.trend_direction,
                    ""
                    if record.opposite_excursion_deg is None
                    else record.opposite_excursion_deg,
                    record.large_step,
                    event is not None,
                    "" if event is None else event.backtrack_deg,
                    record.detail,
                    *angles,
                )
            )


def main() -> int:
    """连接单侧 OpenArm Mini，运行只读采样连续性诊断。"""

    leader: OpenArmMiniLeaderArm | None = None
    try:
        CONTINUITY_CONFIG.validate()
        _validate_connection_config()
        csv_path = _configured_csv_path()
        leader = OpenArmMiniLeaderArm(
            OPENARM_PORT,
            Path(OPENARM_CALIBRATION_PATH),
            TEST_SIDE,
            baudrate=OPENARM_BAUDRATE,
            read_only=True,
        )
        leader.connect()

        print("=" * 72)
        print("    OpenArm Mini 单关节采样连续性诊断（无 HCX、无运动命令）")
        print("=" * 72)
        print(
            f"  侧别={TEST_SIDE}，串口={OPENARM_PORT}，检查关节=J{TEST_JOINT_NUMBER}"
        )
        print(
            f"  采样={SAMPLE_RATE_HZ:.0f} Hz，时长="
            f"{'直到 Ctrl+C' if TEST_DURATION_S == 0.0 else f'{TEST_DURATION_S:g} s'}"
        )
        print(
            f"  趋势起始={TREND_START_DEG:g} deg，"
            f"确认折返={REVERSAL_CONFIRM_DEG:g} deg"
        )
        print("  请只沿一个方向稳定移动选定关节；不要人为反向。")
        print("  不会写 OpenArm 寄存器，不会连接或控制 HCX。")
        print("-" * 72)

        report = run_continuity_test(
            leader,
            CONTINUITY_CONFIG,
            on_summary=print_live_report,
            on_reversal=print_reversal_event,
        )
        print_final_report(report)
        if csv_path is not None:
            write_csv(csv_path, report)
            print(f"[INFO] 已写入原始采样 CSV: {csv_path}")
        return 130 if report.interrupted else 0
    except KeyboardInterrupt:
        print("\n[STOP] 在连接或初始化阶段收到退出请求。")
        return 130
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[ERROR] OpenArm Mini 连续性诊断失败: {exc}", file=sys.stderr)
        return 1
    finally:
        if leader is not None:
            try:
                leader.disconnect()
            except Exception as exc:
                print(f"[WARN] 关闭 OpenArm Mini 连接失败: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
