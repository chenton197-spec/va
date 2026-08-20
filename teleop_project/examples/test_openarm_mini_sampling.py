#!/usr/bin/env python3
"""诊断双 OpenArm Mini 主臂的真实采样节奏。

本程序只连接并读取左右 OpenArm Mini，不加载 ``teleop.yaml``，不连接 HCX，
不执行滤波、映射、插值或任何运动命令。采样路径与双臂遥操作一致：每侧都通过
同步 ``read_joint_angles_deg()`` 读取七个关节，左右两侧并行读取。

运行时请分别缓慢移动左右主臂的一个关节，再以正常速度移动一次。终端会报告：

* 主循环实际频率、错过的 100 Hz 周期与左右读取完成时间差；
* 每侧有效采样率、读取耗时、样本最大间隔和空帧数；
* 相邻有效帧相同的次数与异常大的关节跳变次数。

``same`` 仅表示相邻关节数组没有变化；主臂静止时这是正常现象。只有在持续移动
某个关节时 ``same`` 持续增长，才可能表示编码器读数或串口采样停滞。

例如：

    python -m examples.test_openarm_mini_sampling
"""

from __future__ import annotations

import csv
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from teleop_sdk.adapters import OpenArmMiniLeaderArm


JOINT_COUNT = 7

# 左右 OpenArm Mini 的只读串口；必须是不同的设备路径。
LEFT_PORT = "/dev/ttyACM1"
RIGHT_PORT = "/dev/ttyACM0"
# 左右组合标定 JSON 的路径；该 demo 不会修改此文件。
CALIBRATION_PATH = "./my_openarm_mini.json"
# OpenArm Mini 串口波特率。
BAUDRATE = 1_000_000

# 与当前 OpenArm -> HCX 遥操作一致的主臂读取频率，单位为 Hz。
SAMPLE_RATE_HZ = 100.0
# 单次同步读取允许的最长阻塞时间，单位为秒；保持为遥操作当前使用的 0.1 秒。
READ_TIMEOUT_S = 0.1
# 单次采样测试时长，单位为秒；可随时按 Ctrl+C 提前结束并输出已收集的结果。
TEST_DURATION_S = 30.0
# 终端统计输出间隔，单位为秒；不要逐帧打印，以免反过来干扰采样。
SUMMARY_INTERVAL_S = 1.0

# 相邻有效帧七轴最大差值不超过该值时记为 same，单位为度。
SAME_FRAME_TOLERANCE_DEG = 0.01
# 相邻有效帧单轴变化达到该值时记为 jump，单位为度；它是诊断标记，不会丢弃样本。
LARGE_JOINT_STEP_DEG = 5.0
# 有效样本间隔超过标称周期的该倍数时记为 gap-warning。
SAMPLE_GAP_WARNING_MULTIPLIER = 2.0
# 有效采样率低于目标频率该比例时，在最终结果中提示采样调度异常。
MINIMUM_ACCEPTABLE_RATE_RATIO = 0.95

# 留空时不写文件；填写 CSV 路径后导出每次读取的时间、耗时、状态和七轴角度。
SAMPLING_CSV_PATH = ""


class _LeaderReader(Protocol):
    """本诊断循环所需的 OpenArm 只读接口。"""

    @property
    def joint_count(self) -> int: ...

    def connect(self) -> None: ...

    def read_joint_angles_deg(self, timeout_s: float) -> np.ndarray | None: ...

    def disconnect(self) -> None: ...


@dataclass(frozen=True)
class SamplingConfig:
    """采样诊断的时间和帧判定参数。"""

    rate_hz: float
    read_timeout_s: float
    test_duration_s: float
    summary_interval_s: float
    same_frame_tolerance_deg: float
    large_joint_step_deg: float
    gap_warning_multiplier: float
    minimum_acceptable_rate_ratio: float

    def validate(self) -> None:
        for name, value in (
            ("rate_hz", self.rate_hz),
            ("read_timeout_s", self.read_timeout_s),
            ("test_duration_s", self.test_duration_s),
            ("summary_interval_s", self.summary_interval_s),
            ("large_joint_step_deg", self.large_joint_step_deg),
        ):
            _positive_finite(name, value)
        _nonnegative_finite(
            "same_frame_tolerance_deg", self.same_frame_tolerance_deg
        )
        if self.gap_warning_multiplier < 1.0 or not math.isfinite(
            self.gap_warning_multiplier
        ):
            raise ValueError("gap_warning_multiplier 必须是不小于 1 的有限数")
        if (
            not math.isfinite(self.minimum_acceptable_rate_ratio)
            or not 0.0 < self.minimum_acceptable_rate_ratio <= 1.0
        ):
            raise ValueError(
                "minimum_acceptable_rate_ratio 必须是 (0, 1] 内的有限数"
            )

    @property
    def period_s(self) -> float:
        return 1.0 / self.rate_hz

    @property
    def gap_warning_s(self) -> float:
        return self.period_s * self.gap_warning_multiplier


SAMPLING_CONFIG = SamplingConfig(
    rate_hz=SAMPLE_RATE_HZ,
    read_timeout_s=READ_TIMEOUT_S,
    test_duration_s=TEST_DURATION_S,
    summary_interval_s=SUMMARY_INTERVAL_S,
    same_frame_tolerance_deg=SAME_FRAME_TOLERANCE_DEG,
    large_joint_step_deg=LARGE_JOINT_STEP_DEG,
    gap_warning_multiplier=SAMPLE_GAP_WARNING_MULTIPLIER,
    minimum_acceptable_rate_ratio=MINIMUM_ACCEPTABLE_RATE_RATIO,
)


@dataclass(frozen=True)
class ReadOutcome:
    """一次主臂读取的完成时刻、耗时和有效性。"""

    completed_at_s: float
    duration_s: float
    status: str
    angles_deg: np.ndarray | None
    detail: str = ""


@dataclass(frozen=True)
class SampleRecord:
    """用于最终统计和可选 CSV 导出的单侧读取记录。"""

    completed_at_s: float
    duration_s: float
    status: str
    angles_deg: np.ndarray | None
    sample_gap_s: float | None
    max_joint_delta_deg: float | None
    same_as_previous: bool
    large_joint_step: bool
    detail: str


@dataclass
class ArmSamplingStats:
    """单侧主臂的采样质量计数器和原始记录。"""

    side: str
    attempted_count: int = 0
    valid_count: int = 0
    none_frame_count: int = 0
    invalid_frame_count: int = 0
    exception_count: int = 0
    same_frame_count: int = 0
    changed_frame_count: int = 0
    large_joint_step_count: int = 0
    gap_warning_count: int = 0
    total_read_duration_s: float = 0.0
    max_read_duration_s: float = 0.0
    max_sample_gap_s: float = 0.0
    _valid_timestamps_s: list[float] = field(default_factory=list)
    _last_angles_deg: np.ndarray | None = None
    records: list[SampleRecord] = field(default_factory=list)

    def record(self, outcome: ReadOutcome, config: SamplingConfig) -> None:
        """记录一帧结果，并以相邻有效帧计算时间与角度连续性。"""

        self.attempted_count += 1
        duration_s = max(0.0, float(outcome.duration_s))
        self.total_read_duration_s += duration_s
        self.max_read_duration_s = max(self.max_read_duration_s, duration_s)

        if outcome.status != "ok" or outcome.angles_deg is None:
            if outcome.status == "none":
                self.none_frame_count += 1
            elif outcome.status == "invalid":
                self.invalid_frame_count += 1
            else:
                self.exception_count += 1
            self.records.append(
                SampleRecord(
                    completed_at_s=outcome.completed_at_s,
                    duration_s=duration_s,
                    status=outcome.status,
                    angles_deg=None,
                    sample_gap_s=None,
                    max_joint_delta_deg=None,
                    same_as_previous=False,
                    large_joint_step=False,
                    detail=outcome.detail,
                )
            )
            return

        angles_deg = outcome.angles_deg.copy()
        self.valid_count += 1
        sample_gap_s: float | None = None
        max_joint_delta_deg: float | None = None
        same_as_previous = False
        large_joint_step = False
        if self._valid_timestamps_s:
            sample_gap_s = outcome.completed_at_s - self._valid_timestamps_s[-1]
            self.max_sample_gap_s = max(self.max_sample_gap_s, sample_gap_s)
            if sample_gap_s > config.gap_warning_s:
                self.gap_warning_count += 1
            assert self._last_angles_deg is not None
            max_joint_delta_deg = float(
                np.max(np.abs(angles_deg - self._last_angles_deg))
            )
            same_as_previous = max_joint_delta_deg <= config.same_frame_tolerance_deg
            large_joint_step = max_joint_delta_deg >= config.large_joint_step_deg
            if same_as_previous:
                self.same_frame_count += 1
            else:
                self.changed_frame_count += 1
            if large_joint_step:
                self.large_joint_step_count += 1

        self._valid_timestamps_s.append(outcome.completed_at_s)
        self._last_angles_deg = angles_deg.copy()
        self.records.append(
            SampleRecord(
                completed_at_s=outcome.completed_at_s,
                duration_s=duration_s,
                status="ok",
                angles_deg=angles_deg,
                sample_gap_s=sample_gap_s,
                max_joint_delta_deg=max_joint_delta_deg,
                same_as_previous=same_as_previous,
                large_joint_step=large_joint_step,
                detail=outcome.detail,
            )
        )

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


@dataclass(frozen=True)
class SamplingReport:
    """一次双臂采样测试的汇总。"""

    config: SamplingConfig
    elapsed_s: float
    cycle_count: int
    missed_tick_count: int
    maximum_lateness_s: float
    mean_pair_completion_skew_s: float | None
    maximum_pair_completion_skew_s: float | None
    interrupted: bool
    left: ArmSamplingStats
    right: ArmSamplingStats

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


def _validate_connection_config() -> None:
    """在打开串口前验证顶部定义的只读连接参数。"""

    if not isinstance(LEFT_PORT, str) or not LEFT_PORT.strip():
        raise ValueError("LEFT_PORT 不能为空")
    if not isinstance(RIGHT_PORT, str) or not RIGHT_PORT.strip():
        raise ValueError("RIGHT_PORT 不能为空")
    if LEFT_PORT == RIGHT_PORT:
        raise ValueError("LEFT_PORT 和 RIGHT_PORT 必须是不同串口")
    if not isinstance(CALIBRATION_PATH, str) or not CALIBRATION_PATH.strip():
        raise ValueError("CALIBRATION_PATH 不能为空")
    if not Path(CALIBRATION_PATH).expanduser().is_file():
        raise ValueError(f"找不到 OpenArm Mini 标定文件: {CALIBRATION_PATH}")
    if not isinstance(BAUDRATE, int) or isinstance(BAUDRATE, bool) or BAUDRATE <= 0:
        raise ValueError("BAUDRATE 必须是正整数")


def _configured_csv_path() -> Path | None:
    """返回顶部配置的 CSV 路径；空字符串表示不导出。"""

    if not isinstance(SAMPLING_CSV_PATH, str):
        raise ValueError("SAMPLING_CSV_PATH 必须是字符串")
    normalized = SAMPLING_CSV_PATH.strip()
    return Path(normalized) if normalized else None


def _create_leaders() -> tuple[OpenArmMiniLeaderArm, OpenArmMiniLeaderArm]:
    """创建两条严格只读的 OpenArm Mini 连接。"""

    return (
        OpenArmMiniLeaderArm(
            port=LEFT_PORT,
            calibration_path=CALIBRATION_PATH,
            side="left",
            baudrate=BAUDRATE,
            read_only=True,
        ),
        OpenArmMiniLeaderArm(
            port=RIGHT_PORT,
            calibration_path=CALIBRATION_PATH,
            side="right",
            baudrate=BAUDRATE,
            read_only=True,
        ),
    )


def _read_once(
    leader: _LeaderReader,
    timeout_s: float,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> ReadOutcome:
    """完成一次同步读取，并将异常和无效帧转成可统计的结果。"""

    started_at_s = clock()
    try:
        raw_angles_deg = leader.read_joint_angles_deg(timeout_s)
    except Exception as exc:
        completed_at_s = clock()
        return ReadOutcome(
            completed_at_s=completed_at_s,
            duration_s=completed_at_s - started_at_s,
            status="exception",
            angles_deg=None,
            detail=f"{type(exc).__name__}: {exc}",
        )
    completed_at_s = clock()
    if raw_angles_deg is None:
        return ReadOutcome(
            completed_at_s=completed_at_s,
            duration_s=completed_at_s - started_at_s,
            status="none",
            angles_deg=None,
        )
    try:
        angles_deg = np.asarray(raw_angles_deg, dtype=float)
    except (TypeError, ValueError) as exc:
        return ReadOutcome(
            completed_at_s=completed_at_s,
            duration_s=completed_at_s - started_at_s,
            status="invalid",
            angles_deg=None,
            detail=f"{type(exc).__name__}: {exc}",
        )
    if angles_deg.shape != (leader.joint_count,) or not np.isfinite(angles_deg).all():
        return ReadOutcome(
            completed_at_s=completed_at_s,
            duration_s=completed_at_s - started_at_s,
            status="invalid",
            angles_deg=None,
            detail="主臂返回的关节数组形状或数值无效",
        )
    return ReadOutcome(
        completed_at_s=completed_at_s,
        duration_s=completed_at_s - started_at_s,
        status="ok",
        angles_deg=angles_deg.copy(),
    )


def _build_report(
    config: SamplingConfig,
    *,
    started_at_s: float,
    completed_at_s: float,
    cycle_count: int,
    missed_tick_count: int,
    maximum_lateness_s: float,
    pair_completion_skews_s: list[float],
    interrupted: bool,
    left: ArmSamplingStats,
    right: ArmSamplingStats,
) -> SamplingReport:
    """从当前计数器构造实时或最终报告。"""

    return SamplingReport(
        config=config,
        elapsed_s=max(0.0, completed_at_s - started_at_s),
        cycle_count=cycle_count,
        missed_tick_count=missed_tick_count,
        maximum_lateness_s=maximum_lateness_s,
        mean_pair_completion_skew_s=(
            sum(pair_completion_skews_s) / len(pair_completion_skews_s)
            if pair_completion_skews_s
            else None
        ),
        maximum_pair_completion_skew_s=(
            max(pair_completion_skews_s) if pair_completion_skews_s else None
        ),
        interrupted=interrupted,
        left=left,
        right=right,
    )


def run_sampling_test(
    left_leader: _LeaderReader,
    right_leader: _LeaderReader,
    config: SamplingConfig,
    *,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    on_summary: Callable[[SamplingReport], None] | None = None,
) -> SamplingReport:
    """以固定节奏并行读取两侧主臂，直到时长结束或收到 Ctrl+C。"""

    config.validate()
    if left_leader.joint_count != JOINT_COUNT or right_leader.joint_count != JOINT_COUNT:
        raise ValueError(f"左右 OpenArm Mini 都必须是 {JOINT_COUNT} 轴")

    left_stats = ArmSamplingStats("left")
    right_stats = ArmSamplingStats("right")
    pair_completion_skews_s: list[float] = []
    started_at_s = clock()
    next_deadline_s = started_at_s
    next_summary_s = started_at_s + config.summary_interval_s
    end_at_s = started_at_s + config.test_duration_s
    cycle_count = 0
    missed_tick_count = 0
    maximum_lateness_s = 0.0
    interrupted = False

    try:
        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="openarm-sampling"
        ) as executor:
            while clock() < end_at_s:
                left_future = executor.submit(
                    _read_once, left_leader, config.read_timeout_s, clock=clock
                )
                right_future = executor.submit(
                    _read_once, right_leader, config.read_timeout_s, clock=clock
                )
                left_outcome = left_future.result()
                right_outcome = right_future.result()
                left_stats.record(left_outcome, config)
                right_stats.record(right_outcome, config)
                pair_completion_skews_s.append(
                    abs(left_outcome.completed_at_s - right_outcome.completed_at_s)
                )
                cycle_count += 1

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
                                pair_completion_skews_s=pair_completion_skews_s,
                                interrupted=False,
                                left=left_stats,
                                right=right_stats,
                            )
                        )
                    next_summary_s = finished_at_s + config.summary_interval_s

                next_deadline_s += config.period_s
                delay_s = next_deadline_s - finished_at_s
                if delay_s > 0.0:
                    sleep(delay_s)
                    continue

                missed_ticks = 1 + int((-delay_s) / config.period_s)
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
        pair_completion_skews_s=pair_completion_skews_s,
        interrupted=interrupted,
        left=left_stats,
        right=right_stats,
    )


def _format_rate(rate_hz: float | None) -> str:
    return "--" if rate_hz is None else f"{rate_hz:.1f}"


def _format_duration(duration_s: float | None) -> str:
    return "--" if duration_s is None else f"{duration_s * 1_000.0:.2f}ms"


def _format_arm_stats(stats: ArmSamplingStats, target_rate_hz: float) -> str:
    """格式化单侧终端统计；same 并不自动代表采样异常。"""

    return (
        f"{stats.side[0].upper()} ok {stats.valid_count}/{stats.attempted_count} "
        f"{_format_rate(stats.observed_rate_hz)}/{target_rate_hz:.0f}Hz "
        f"read avg/max {_format_duration(stats.mean_read_duration_s)}/"
        f"{_format_duration(stats.max_read_duration_s)} "
        f"gap-max {_format_duration(stats.max_sample_gap_s)} "
        f"none {stats.none_frame_count} invalid {stats.invalid_frame_count} "
        f"error {stats.exception_count} same {stats.same_frame_count} "
        f"jump {stats.large_joint_step_count}"
    )


def print_live_report(report: SamplingReport) -> None:
    """每秒输出一次采样状态，避免逐帧输出干扰测量。"""

    loop_rate = _format_rate(report.observed_loop_rate_hz)
    pair_skew = _format_duration(report.maximum_pair_completion_skew_s)
    print(
        "[OpenArm RX] "
        f"loop {loop_rate}/{report.config.rate_hz:.0f}Hz "
        f"miss {report.missed_tick_count} "
        f"late-max {_format_duration(report.maximum_lateness_s)} "
        f"pair-skew-max {pair_skew}"
    )
    print(f"  {_format_arm_stats(report.left, report.config.rate_hz)}")
    print(f"  {_format_arm_stats(report.right, report.config.rate_hz)}")


def _sampling_warnings(report: SamplingReport) -> list[str]:
    """生成保守的节奏诊断结论，不把静止时的 same 当作错误。"""

    warnings: list[str] = []
    loop_rate = report.observed_loop_rate_hz
    minimum_rate = report.config.rate_hz * report.config.minimum_acceptable_rate_ratio
    if loop_rate is None or loop_rate < minimum_rate:
        warnings.append(
            f"主循环 {(_format_rate(loop_rate))} Hz，低于 {minimum_rate:.1f} Hz"
        )
    if report.missed_tick_count:
        warnings.append(f"主循环错过 {report.missed_tick_count} 个计划周期")
    for stats in (report.left, report.right):
        rate = stats.observed_rate_hz
        if rate is None or rate < minimum_rate:
            warnings.append(
                f"{stats.side} 有效采样率 {_format_rate(rate)} Hz，低于 "
                f"{minimum_rate:.1f} Hz"
            )
        if stats.none_frame_count or stats.invalid_frame_count or stats.exception_count:
            warnings.append(
                f"{stats.side} 存在 none={stats.none_frame_count}、"
                f"invalid={stats.invalid_frame_count}、error={stats.exception_count}"
            )
        if stats.gap_warning_count:
            warnings.append(
                f"{stats.side} 有 {stats.gap_warning_count} 个样本间隔超过 "
                f"{report.config.gap_warning_s * 1_000.0:.1f} ms"
            )
    return warnings


def print_final_report(report: SamplingReport) -> None:
    """输出完整测量结果和保守的采样质量结论。"""

    print("=" * 72)
    print("    OpenArm Mini 双主臂采样诊断结果")
    print("=" * 72)
    print(f"  测试时长: {report.elapsed_s:.3f} s")
    print_live_report(report)
    warnings = _sampling_warnings(report)
    for warning in warnings:
        print(f"  [WARN] {warning}")
    if not warnings:
        print("  [RESULT] 本次未发现明显的主臂采样频率、空帧或样本间隔异常。")
    print(
        "  same 仅表示相邻读数未变化；请在持续移动某个关节时观察其是否持续增长。"
    )
    print(
        "  本程序未读取 HCX 反馈；若这里稳定而遥操作仍顿挫，应继续检查 "
        "100 Hz 映射目标到 500 Hz 直伺服输出及驱动跟踪。"
    )


def write_csv(path: Path, report: SamplingReport) -> None:
    """导出原始采样记录，便于离线检查间隔、重复帧和角度曲线。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "side",
        "completion_time_s",
        "read_duration_ms",
        "status",
        "sample_gap_ms",
        "max_joint_delta_deg",
        "same_as_previous",
        "large_joint_step",
        "detail",
    ]
    header.extend(f"j{index + 1}_deg" for index in range(JOINT_COUNT))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        for stats in (report.left, report.right):
            for record in stats.records:
                angles = (
                    [""] * JOINT_COUNT
                    if record.angles_deg is None
                    else record.angles_deg.tolist()
                )
                writer.writerow(
                    (
                        stats.side,
                        record.completed_at_s,
                        record.duration_s * 1_000.0,
                        record.status,
                        ""
                        if record.sample_gap_s is None
                        else record.sample_gap_s * 1_000.0,
                        ""
                        if record.max_joint_delta_deg is None
                        else record.max_joint_delta_deg,
                        record.same_as_previous,
                        record.large_joint_step,
                        record.detail,
                        *angles,
                    )
                )


def main() -> int:
    """连接两侧 OpenArm Mini，执行不影响从臂的采样诊断。"""

    leaders: tuple[OpenArmMiniLeaderArm, OpenArmMiniLeaderArm] | None = None
    try:
        SAMPLING_CONFIG.validate()
        _validate_connection_config()
        csv_path = _configured_csv_path()
        leaders = _create_leaders()
        left, right = leaders
        left.connect()
        right.connect()

        print("=" * 72)
        print("    OpenArm Mini 双主臂采样诊断（无 HCX、无运动命令）")
        print("=" * 72)
        print(
            f"  采样: 左右并行同步读取，目标 {SAMPLING_CONFIG.rate_hz:.0f} Hz；"
            f"单次超时 {SAMPLING_CONFIG.read_timeout_s * 1_000.0:.0f} ms"
        )
        print(
            f"  时长: {SAMPLING_CONFIG.test_duration_s:.1f} s；"
            "测试时请分别缓慢和正常速度移动左右主臂的一个关节。"
        )
        print("  不会写 OpenArm 寄存器，不会连接或控制 HCX；按 Ctrl+C 可提前结束。")
        print("-" * 72)

        report = run_sampling_test(
            left,
            right,
            SAMPLING_CONFIG,
            on_summary=print_live_report,
        )
        print_final_report(report)
        if csv_path is not None:
            write_csv(csv_path, report)
            print(f"[INFO] 已写入采样 CSV: {csv_path}")
        return 130 if report.interrupted else 0
    except KeyboardInterrupt:
        print("\n[STOP] 在连接或初始化阶段收到退出请求。")
        return 130
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[ERROR] OpenArm Mini 采样诊断失败: {exc}", file=sys.stderr)
        return 1
    finally:
        if leaders is not None:
            for leader in reversed(leaders):
                try:
                    leader.disconnect()
                except Exception as exc:
                    print(f"[WARN] 关闭 OpenArm Mini 连接失败: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
