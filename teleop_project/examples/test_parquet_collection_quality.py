#!/usr/bin/env python3
"""分析一个采集 episode Parquet 的帧和点位连续性。

本脚本只读取数据，不连接 OpenArm、HCX、Gloria-M 或相机。修改文件顶部的
``PARQUET_PATH`` 后直接运行：

    python -m examples.test_parquet_collection_quality

``action`` 与 ``observation.state`` 按项目采集约定均以角度表示。关节跳变阈值
只是数据质量告警阈值，不是机械臂的安全限位或速度限制。
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# 要分析的单个 episode Parquet 文件。请改成实际采集文件路径。
PARQUET_PATH = Path("datasets/episode_000000.parquet")
# 期望采集帧率；None 表示用有效时间间隔的中位数估算基准频率。
EXPECTED_FRAME_RATE_HZ: float | None = None
# 时间间隔超过基准周期的多少倍时标记为采集帧间隙。
TIMESTAMP_GAP_WARNING_MULTIPLIER = 2.0
# 相邻记录中任一关节变化超过该角度时标记为点位跳变。
JOINT_STEP_DISCONTINUITY_DEG = 5.0
# 小于该变化量的关节增量不计入方向反转统计，用来滤除量化噪声。
DIRECTION_DEAD_ZONE_DEG = 0.01
# 终端最多列出多少个最大的关节跳变事件。
MAX_REPORTED_DISCONTINUITIES = 12


@dataclass(frozen=True)
class AnalysisConfig:
    """本地数据质量阈值，不影响采集和控制。"""

    expected_frame_rate_hz: float | None
    timestamp_gap_warning_multiplier: float
    joint_step_discontinuity_deg: float
    direction_dead_zone_deg: float
    max_reported_discontinuities: int


ANALYSIS_CONFIG = AnalysisConfig(
    expected_frame_rate_hz=EXPECTED_FRAME_RATE_HZ,
    timestamp_gap_warning_multiplier=TIMESTAMP_GAP_WARNING_MULTIPLIER,
    joint_step_discontinuity_deg=JOINT_STEP_DISCONTINUITY_DEG,
    direction_dead_zone_deg=DIRECTION_DEAD_ZONE_DEG,
    max_reported_discontinuities=MAX_REPORTED_DISCONTINUITIES,
)


@dataclass(frozen=True)
class SequenceMetrics:
    """一个整数索引列的完整性指标。"""

    name: str
    count: int
    invalid_count: int
    mismatch_count: int
    missing_count: int
    duplicate_count: int
    backward_count: int
    first_value: int | None
    last_value: int | None


@dataclass(frozen=True)
class TimelineMetrics:
    """Parquet 行时间轴的完整性和帧间隔指标。"""

    count: int
    invalid_count: int
    duplicate_timestamp_count: int
    backward_timestamp_count: int
    duration_s: float | None
    observed_rate_hz: float | None
    reference_rate_hz: float | None
    median_gap_ms: float | None
    p95_gap_ms: float | None
    max_gap_ms: float | None
    gap_warning_threshold_ms: float | None
    gap_warning_count: int


@dataclass(frozen=True)
class Discontinuity:
    """一个相邻帧间的单关节点位跳变。"""

    previous_frame_row: int
    frame_row: int
    joint_index: int
    step_deg: float
    elapsed_ms: float | None


@dataclass(frozen=True)
class VectorMetrics:
    """一个角度向量列的连续性统计。"""

    name: str
    joint_count: int
    invalid_row_count: int
    range_deg: np.ndarray
    p95_step_deg: np.ndarray
    maximum_step_deg: np.ndarray
    p95_velocity_deg_s: np.ndarray
    maximum_velocity_deg_s: np.ndarray
    reversal_count: np.ndarray
    discontinuity_count: np.ndarray
    discontinuities: tuple[Discontinuity, ...]


@dataclass(frozen=True)
class AlignmentMetrics:
    """同一行 action 与 observation.state 的绝对误差。"""

    comparable_row_count: int
    p95_absolute_error_deg: np.ndarray
    maximum_absolute_error_deg: np.ndarray


@dataclass(frozen=True)
class ScalarMetrics:
    """夹爪等标量列的连续性统计。"""

    name: str
    valid_count: int
    invalid_count: int
    out_of_normalized_range_count: int
    minimum: float | None
    maximum: float | None
    maximum_step: float | None


@dataclass(frozen=True)
class CameraMetrics:
    """一个图像列的路径与相对数据行时间指标。"""

    name: str
    valid_entry_count: int
    missing_entry_count: int
    duplicate_path_count: int
    timestamp_count: int
    future_timestamp_count: int
    median_offset_ms: float | None
    minimum_offset_ms: float | None
    maximum_offset_ms: float | None


@dataclass(frozen=True)
class AuditMetrics:
    """可选录制审计 JSONL 的行计数。"""

    path: Path
    record_count: int
    emitted_count: int
    skipped_count: int
    malformed_count: int
    skip_reasons: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ParquetQualityReport:
    """一个 Parquet episode 的完整分析结果。"""

    path: Path
    row_count: int
    column_names: tuple[str, ...]
    frame_index: SequenceMetrics | None
    global_index: SequenceMetrics | None
    episode_values: tuple[int, ...]
    timeline: TimelineMetrics | None
    vectors: tuple[VectorMetrics, ...]
    alignment: AlignmentMetrics | None
    scalars: tuple[ScalarMetrics, ...]
    cameras: tuple[CameraMetrics, ...]
    audit: AuditMetrics | None


def _validate_config(config: AnalysisConfig) -> None:
    if config.expected_frame_rate_hz is not None:
        _positive_finite(config.expected_frame_rate_hz, "expected_frame_rate_hz")
    if config.timestamp_gap_warning_multiplier <= 1.0:
        raise ValueError("timestamp_gap_warning_multiplier 必须大于 1")
    _positive_finite(
        config.joint_step_discontinuity_deg, "joint_step_discontinuity_deg"
    )
    if config.direction_dead_zone_deg < 0.0 or not math.isfinite(
        config.direction_dead_zone_deg
    ):
        raise ValueError("direction_dead_zone_deg 必须是非负有限数")
    if (
        isinstance(config.max_reported_discontinuities, bool)
        or not isinstance(config.max_reported_discontinuities, int)
        or config.max_reported_discontinuities <= 0
    ):
        raise ValueError("max_reported_discontinuities 必须是正整数")


def _positive_finite(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} 必须是正的有限数")


def _column_values(table: Any, name: str) -> list[Any] | None:
    if name not in tuple(table.column_names):
        return None
    return table.column(name).to_pylist()


def _numeric_values(values: Sequence[Any]) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    for index, value in enumerate(values):
        if isinstance(value, bool):
            continue
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate):
            result[index] = candidate
    return result


def _integer_values(values: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    numeric = _numeric_values(values)
    rounded = np.rint(numeric)
    valid = np.isfinite(numeric) & np.isclose(numeric, rounded, atol=0.0, rtol=0.0)
    result = np.zeros(len(values), dtype=np.int64)
    result[valid] = rounded[valid].astype(np.int64)
    return result, valid


def analyze_sequence(
    name: str, values: Sequence[Any], *, expected_start: int | None
) -> SequenceMetrics:
    """检查一列整数索引是否逐行连续。"""

    integers, valid = _integer_values(values)
    invalid_count = int(np.count_nonzero(~valid))
    valid_indices = np.flatnonzero(valid)
    first = int(integers[valid_indices[0]]) if valid_indices.size else None
    last = int(integers[valid_indices[-1]]) if valid_indices.size else None
    baseline = expected_start if expected_start is not None else first
    if baseline is None:
        mismatch_count = 0
    else:
        expected = np.arange(len(values), dtype=np.int64) + baseline
        mismatch_count = int(np.count_nonzero(~valid | (integers != expected)))

    missing_count = 0
    duplicate_count = 0
    backward_count = 0
    if valid_indices.size >= 2:
        previous = integers[valid_indices[:-1]]
        current = integers[valid_indices[1:]]
        deltas = current - previous
        missing_count = int(np.sum(np.maximum(deltas - 1, 0)))
        duplicate_count = int(np.count_nonzero(deltas == 0))
        backward_count = int(np.count_nonzero(deltas < 0))
    return SequenceMetrics(
        name=name,
        count=len(values),
        invalid_count=invalid_count,
        mismatch_count=mismatch_count,
        missing_count=missing_count,
        duplicate_count=duplicate_count,
        backward_count=backward_count,
        first_value=first,
        last_value=last,
    )


def analyze_timestamps(
    values: Sequence[Any], config: AnalysisConfig
) -> tuple[np.ndarray, TimelineMetrics]:
    """统计数据行时间轴并返回便于后续轨迹计算的数值数组。"""

    timestamps = _numeric_values(values)
    valid = np.isfinite(timestamps)
    invalid_count = int(np.count_nonzero(~valid))
    valid_indices = np.flatnonzero(valid)
    duplicate_count = 0
    backward_count = 0
    positive_gaps = np.empty(0, dtype=float)
    if valid_indices.size >= 2:
        gaps = np.diff(timestamps[valid_indices])
        duplicate_count = int(np.count_nonzero(gaps == 0.0))
        backward_count = int(np.count_nonzero(gaps < 0.0))
        positive_gaps = gaps[gaps > 0.0]

    duration_s: float | None = None
    observed_rate_hz: float | None = None
    if valid_indices.size >= 2:
        elapsed = timestamps[valid_indices[-1]] - timestamps[valid_indices[0]]
        if elapsed > 0.0:
            duration_s = float(elapsed)
            observed_rate_hz = float((valid_indices.size - 1) / elapsed)

    reference_rate_hz = config.expected_frame_rate_hz
    if reference_rate_hz is None and positive_gaps.size:
        reference_rate_hz = float(1.0 / np.median(positive_gaps))
    warning_threshold_s: float | None = None
    warning_count = 0
    if reference_rate_hz is not None:
        warning_threshold_s = (
            config.timestamp_gap_warning_multiplier / reference_rate_hz
        )
        warning_count = int(np.count_nonzero(positive_gaps > warning_threshold_s))

    def milliseconds(percentile: float) -> float | None:
        if not positive_gaps.size:
            return None
        return float(np.percentile(positive_gaps, percentile) * 1_000.0)

    return timestamps, TimelineMetrics(
        count=len(values),
        invalid_count=invalid_count,
        duplicate_timestamp_count=duplicate_count,
        backward_timestamp_count=backward_count,
        duration_s=duration_s,
        observed_rate_hz=observed_rate_hz,
        reference_rate_hz=reference_rate_hz,
        median_gap_ms=milliseconds(50.0),
        p95_gap_ms=milliseconds(95.0),
        max_gap_ms=milliseconds(100.0),
        gap_warning_threshold_ms=(
            None if warning_threshold_s is None else warning_threshold_s * 1_000.0
        ),
        gap_warning_count=warning_count,
    )


def _vector_matrix(values: Sequence[Any]) -> tuple[np.ndarray | None, int]:
    """将 Parquet list 列转换为固定宽度矩阵，坏行保持 NaN。"""

    width: int | None = None
    parsed: list[np.ndarray | None] = []
    for value in values:
        try:
            vector = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            parsed.append(None)
            continue
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            parsed.append(None)
            continue
        if width is None:
            width = int(vector.size)
        if vector.size != width:
            parsed.append(None)
            continue
        parsed.append(vector.copy())
    if width is None:
        return None, len(values)
    matrix = np.full((len(values), width), np.nan, dtype=float)
    invalid_count = 0
    for row_index, vector in enumerate(parsed):
        if vector is None:
            invalid_count += 1
            continue
        matrix[row_index] = vector
    return matrix, invalid_count


def analyze_vector_column(
    name: str,
    values: Sequence[Any],
    timestamps: np.ndarray,
    config: AnalysisConfig,
) -> tuple[VectorMetrics | None, np.ndarray | None]:
    """计算角度向量的步长、速度、方向反转和大跳变。"""

    matrix, invalid_count = _vector_matrix(values)
    if matrix is None:
        return None, None
    joint_count = matrix.shape[1]
    valid_rows = np.isfinite(matrix).all(axis=1)
    valid_values = matrix[valid_rows]
    zero = np.zeros(joint_count, dtype=float)
    if not valid_values.size:
        return (
            VectorMetrics(
                name=name,
                joint_count=joint_count,
                invalid_row_count=invalid_count,
                range_deg=zero,
                p95_step_deg=zero,
                maximum_step_deg=zero,
                p95_velocity_deg_s=zero,
                maximum_velocity_deg_s=zero,
                reversal_count=np.zeros(joint_count, dtype=int),
                discontinuity_count=np.zeros(joint_count, dtype=int),
                discontinuities=(),
            ),
            matrix,
        )

    range_deg = np.ptp(valid_values, axis=0)
    if len(matrix) < 2:
        return (
            VectorMetrics(
                name=name,
                joint_count=joint_count,
                invalid_row_count=invalid_count,
                range_deg=range_deg,
                p95_step_deg=zero,
                maximum_step_deg=zero,
                p95_velocity_deg_s=zero,
                maximum_velocity_deg_s=zero,
                reversal_count=np.zeros(joint_count, dtype=int),
                discontinuity_count=np.zeros(joint_count, dtype=int),
                discontinuities=(),
            ),
            matrix,
        )

    row_pairs = np.arange(len(matrix) - 1)
    usable_pairs = valid_rows[:-1] & valid_rows[1:]
    used_pair_rows = row_pairs[usable_pairs]
    if not used_pair_rows.size:
        return (
            VectorMetrics(
                name=name,
                joint_count=joint_count,
                invalid_row_count=invalid_count,
                range_deg=range_deg,
                p95_step_deg=zero,
                maximum_step_deg=zero,
                p95_velocity_deg_s=zero,
                maximum_velocity_deg_s=zero,
                reversal_count=np.zeros(joint_count, dtype=int),
                discontinuity_count=np.zeros(joint_count, dtype=int),
                discontinuities=(),
            ),
            matrix,
        )

    deltas = matrix[used_pair_rows + 1] - matrix[used_pair_rows]
    absolute_deltas = np.abs(deltas)
    p95_step = np.percentile(absolute_deltas, 95.0, axis=0)
    maximum_step = np.max(absolute_deltas, axis=0)
    discontinuity_mask = absolute_deltas > config.joint_step_discontinuity_deg
    discontinuity_count = np.count_nonzero(discontinuity_mask, axis=0).astype(int)

    events: list[Discontinuity] = []
    for pair_index, joint_index in np.argwhere(discontinuity_mask):
        previous_row = int(used_pair_rows[pair_index])
        elapsed_s = timestamps[previous_row + 1] - timestamps[previous_row]
        events.append(
            Discontinuity(
                previous_frame_row=previous_row,
                frame_row=previous_row + 1,
                joint_index=int(joint_index),
                step_deg=float(deltas[pair_index, joint_index]),
                elapsed_ms=(
                    None if not math.isfinite(elapsed_s) else float(elapsed_s * 1_000.0)
                ),
            )
        )
    events.sort(key=lambda event: abs(event.step_deg), reverse=True)

    positive_time_pairs = np.isfinite(timestamps[:-1]) & np.isfinite(timestamps[1:])
    positive_time_pairs &= np.diff(timestamps) > 0.0
    velocity_pair_rows = row_pairs[usable_pairs & positive_time_pairs]
    p95_velocity = zero
    maximum_velocity = zero
    if velocity_pair_rows.size:
        velocity_deltas = matrix[velocity_pair_rows + 1] - matrix[velocity_pair_rows]
        elapsed_s = timestamps[velocity_pair_rows + 1] - timestamps[velocity_pair_rows]
        velocities = np.abs(velocity_deltas / elapsed_s[:, np.newaxis])
        p95_velocity = np.percentile(velocities, 95.0, axis=0)
        maximum_velocity = np.max(velocities, axis=0)

    reversal_count = np.zeros(joint_count, dtype=int)
    previous_direction = np.zeros(joint_count, dtype=int)
    for delta in deltas:
        direction = np.sign(delta).astype(int)
        direction[np.abs(delta) <= config.direction_dead_zone_deg] = 0
        reversal_count += (
            (direction != 0)
            & (previous_direction != 0)
            & (direction != previous_direction)
        ).astype(int)
        previous_direction[direction != 0] = direction[direction != 0]

    return (
        VectorMetrics(
            name=name,
            joint_count=joint_count,
            invalid_row_count=invalid_count,
            range_deg=range_deg,
            p95_step_deg=p95_step,
            maximum_step_deg=maximum_step,
            p95_velocity_deg_s=p95_velocity,
            maximum_velocity_deg_s=maximum_velocity,
            reversal_count=reversal_count,
            discontinuity_count=discontinuity_count,
            discontinuities=tuple(events[: config.max_reported_discontinuities]),
        ),
        matrix,
    )


def analyze_alignment(
    action: np.ndarray | None, observation: np.ndarray | None
) -> AlignmentMetrics | None:
    """计算同一采集行的指令和反馈差值；不是控制器内部时间同步误差。"""

    if action is None or observation is None or action.shape != observation.shape:
        return None
    valid_rows = np.isfinite(action).all(axis=1) & np.isfinite(observation).all(axis=1)
    if not np.any(valid_rows):
        return None
    errors = np.abs(action[valid_rows] - observation[valid_rows])
    return AlignmentMetrics(
        comparable_row_count=int(np.count_nonzero(valid_rows)),
        p95_absolute_error_deg=np.percentile(errors, 95.0, axis=0),
        maximum_absolute_error_deg=np.max(errors, axis=0),
    )


def analyze_scalar_column(name: str, values: Sequence[Any]) -> ScalarMetrics:
    """统计夹爪归一化开合量等标量列。"""

    numeric = _numeric_values(values)
    valid = np.isfinite(numeric)
    samples = numeric[valid]
    maximum_step: float | None = None
    if samples.size >= 2:
        maximum_step = float(np.max(np.abs(np.diff(samples))))
    normalized_range_errors = 0
    if "gripper" in name:
        normalized_range_errors = int(
            np.count_nonzero(samples < 0.0) + np.count_nonzero(samples > 1.0)
        )
    return ScalarMetrics(
        name=name,
        valid_count=int(samples.size),
        invalid_count=int(np.count_nonzero(~valid)),
        out_of_normalized_range_count=normalized_range_errors,
        minimum=None if not samples.size else float(np.min(samples)),
        maximum=None if not samples.size else float(np.max(samples)),
        maximum_step=maximum_step,
    )


def analyze_camera_column(
    name: str, values: Sequence[Any], row_timestamps: np.ndarray
) -> CameraMetrics:
    """检查每行图像路径，以及相机时间是否晚于对应数据行。"""

    paths: list[str] = []
    offsets_s: list[float] = []
    valid_entries = 0
    missing_entries = 0
    future_count = 0
    timestamp_count = 0
    for row_index, value in enumerate(values):
        if not isinstance(value, dict):
            missing_entries += 1
            continue
        path = value.get("path")
        if not isinstance(path, str) or not path:
            missing_entries += 1
            continue
        valid_entries += 1
        paths.append(path)
        timestamp = _numeric_values([value.get("timestamp")])[0]
        row_timestamp = row_timestamps[row_index]
        if not math.isfinite(timestamp) or not math.isfinite(row_timestamp):
            continue
        offset_s = float(timestamp - row_timestamp)
        offsets_s.append(offset_s)
        timestamp_count += 1
        if offset_s > 1e-6:
            future_count += 1
    duplicate_path_count = len(paths) - len(set(paths))
    if not offsets_s:
        return CameraMetrics(
            name=name,
            valid_entry_count=valid_entries,
            missing_entry_count=missing_entries,
            duplicate_path_count=duplicate_path_count,
            timestamp_count=timestamp_count,
            future_timestamp_count=future_count,
            median_offset_ms=None,
            minimum_offset_ms=None,
            maximum_offset_ms=None,
        )
    offsets_ms = np.asarray(offsets_s, dtype=float) * 1_000.0
    return CameraMetrics(
        name=name,
        valid_entry_count=valid_entries,
        missing_entry_count=missing_entries,
        duplicate_path_count=duplicate_path_count,
        timestamp_count=timestamp_count,
        future_timestamp_count=future_count,
        median_offset_ms=float(np.median(offsets_ms)),
        minimum_offset_ms=float(np.min(offsets_ms)),
        maximum_offset_ms=float(np.max(offsets_ms)),
    )


def _audit_path_for_parquet(path: Path) -> Path | None:
    if path.parent.name.startswith("chunk-") and path.parent.parent.name == "data":
        return (
            path.parent.parent.parent
            / "meta"
            / "recording_audit"
            / f"{path.stem}.jsonl"
        )
    return None


def analyze_audit(path: Path) -> AuditMetrics | None:
    """读取标准采集布局下的可选审计 JSONL。"""

    audit_path = _audit_path_for_parquet(path)
    if audit_path is None or not audit_path.is_file():
        return None
    total = 0
    emitted = 0
    malformed = 0
    reasons: Counter[str] = Counter()
    with audit_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if item.get("emitted") is True:
                emitted += 1
                continue
            reason = item.get("skip_reason")
            reasons[str(reason) if reason is not None else "unknown"] += 1
    return AuditMetrics(
        path=audit_path,
        record_count=total,
        emitted_count=emitted,
        skipped_count=total - emitted - malformed,
        malformed_count=malformed,
        skip_reasons=tuple(reasons.most_common()),
    )


def analyze_table(
    table: Any, path: Path, config: AnalysisConfig = ANALYSIS_CONFIG
) -> ParquetQualityReport:
    """分析已读入的 PyArrow Table，便于无文件单元测试。"""

    _validate_config(config)
    column_names = tuple(table.column_names)
    frame_values = _column_values(table, "frame_index")
    global_index_values = _column_values(table, "index")
    timestamp_values = _column_values(table, "timestamp")
    episode_values = _column_values(table, "episode_index")
    timeline: TimelineMetrics | None = None
    timestamps = np.full(table.num_rows, np.nan, dtype=float)
    if timestamp_values is not None:
        timestamps, timeline = analyze_timestamps(timestamp_values, config)

    vectors: list[VectorMetrics] = []
    matrices: dict[str, np.ndarray] = {}
    for name in ("action", "observation.state"):
        values = _column_values(table, name)
        if values is None:
            continue
        metrics, matrix = analyze_vector_column(name, values, timestamps, config)
        if metrics is not None:
            vectors.append(metrics)
        if matrix is not None:
            matrices[name] = matrix

    scalar_names = sorted(
        name
        for name in column_names
        if (name.startswith("action.") or name.startswith("observation."))
        and "gripper" in name
    )
    scalars = tuple(
        analyze_scalar_column(name, values)
        for name in scalar_names
        if (values := _column_values(table, name)) is not None
    )
    camera_names = sorted(
        name for name in column_names if name.startswith("observation.images.")
    )
    cameras = tuple(
        analyze_camera_column(name, values, timestamps)
        for name in camera_names
        if (values := _column_values(table, name)) is not None
    )

    episodes: tuple[int, ...] = ()
    if episode_values is not None:
        integer_episodes, valid_episode_rows = _integer_values(episode_values)
        episodes = tuple(
            sorted({int(value) for value in integer_episodes[valid_episode_rows]})
        )
    return ParquetQualityReport(
        path=path,
        row_count=int(table.num_rows),
        column_names=column_names,
        frame_index=(
            None
            if frame_values is None
            else analyze_sequence("frame_index", frame_values, expected_start=0)
        ),
        global_index=(
            None
            if global_index_values is None
            else analyze_sequence("index", global_index_values, expected_start=None)
        ),
        episode_values=episodes,
        timeline=timeline,
        vectors=tuple(vectors),
        alignment=analyze_alignment(
            matrices.get("action"), matrices.get("observation.state")
        ),
        scalars=scalars,
        cameras=cameras,
        audit=analyze_audit(path),
    )


def _joint_labels(count: int) -> tuple[str, ...]:
    if count == 14:
        return tuple(f"L-J{index}" for index in range(1, 8)) + tuple(
            f"R-J{index}" for index in range(1, 8)
        )
    return tuple(f"J{index}" for index in range(1, count + 1))


def _format_optional(value: float | None, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def _print_rule() -> None:
    print("-" * 72)


def _sequence_is_continuous(metrics: SequenceMetrics | None) -> bool:
    return metrics is not None and all(
        value == 0
        for value in (
            metrics.invalid_count,
            metrics.mismatch_count,
            metrics.missing_count,
            metrics.duplicate_count,
            metrics.backward_count,
        )
    )


def _vector_is_continuous(metrics: VectorMetrics) -> bool:
    return metrics.invalid_row_count == 0 and not np.any(metrics.discontinuity_count)


def _camera_is_consistent(metrics: CameraMetrics) -> bool:
    return (
        metrics.missing_entry_count == 0
        and metrics.duplicate_path_count == 0
        and metrics.future_timestamp_count == 0
    )


def _print_vector_table(metrics: VectorMetrics, config: AnalysisConfig) -> None:
    """按一行一个关节打印连续性指标，避免十四轴结果挤在同一行。"""

    print(f"[POINT] {metrics.name}（{metrics.joint_count} 轴）")
    print(
        "  关节      范围°    P95步长°   最大步长°   P95速度°/s   最大速度°/s   "
        "反转  跳变"
    )
    for row in zip(
        _joint_labels(metrics.joint_count),
        metrics.range_deg,
        metrics.p95_step_deg,
        metrics.maximum_step_deg,
        metrics.p95_velocity_deg_s,
        metrics.maximum_velocity_deg_s,
        metrics.reversal_count,
        metrics.discontinuity_count,
    ):
        (
            label,
            span,
            p95_step,
            max_step,
            p95_velocity,
            max_velocity,
            reversals,
            jumps,
        ) = row
        print(
            f"  {label:<5} {span:9.3f} {p95_step:11.3f} {max_step:11.3f} "
            f"{p95_velocity:13.3f} {max_velocity:13.3f} {int(reversals):5d} {int(jumps):5d}"
        )
    print(
        f"  说明: 跳变表示相邻帧步长 > {config.joint_step_discontinuity_deg:.2f}°；"
        f"无效点行={metrics.invalid_row_count}。"
    )
    if metrics.discontinuities:
        print(f"  最大 {len(metrics.discontinuities)} 个跳变:")
        for event in metrics.discontinuities:
            elapsed = "--" if event.elapsed_ms is None else f"{event.elapsed_ms:.2f} ms"
            label = _joint_labels(metrics.joint_count)[event.joint_index]
            print(
                f"    row {event.previous_frame_row:>5} -> {event.frame_row:<5} "
                f"{label:<5} {event.step_deg:+.3f}°，间隔 {elapsed}"
            )


def _print_alignment_table(metrics: AlignmentMetrics) -> None:
    print("[ALIGN] 同一采集行的 action 与 observation.state 差值")
    print("  关节      P95绝对误差°   最大绝对误差°")
    for label, p95_error, max_error in zip(
        _joint_labels(len(metrics.p95_absolute_error_deg)),
        metrics.p95_absolute_error_deg,
        metrics.maximum_absolute_error_deg,
    ):
        print(f"  {label:<5} {p95_error:14.3f} {max_error:16.3f}")
    print(
        f"  可比较行数={metrics.comparable_row_count}；该差值不是控制器内部时间同步延迟。"
    )


def print_report(
    report: ParquetQualityReport, config: AnalysisConfig = ANALYSIS_CONFIG
) -> None:
    """以适合终端排查的形式打印分析结果。"""

    print("=" * 72)
    print("    采集 Parquet 帧与点位连续性分析")
    print("=" * 72)
    print(f"[FILE] {report.path}")
    print(
        f"[FRAME] 帧数={report.row_count}，列数={len(report.column_names)}，"
        f"episode={list(report.episode_values) or '--'}"
    )
    _print_rule()
    print("[SUMMARY]")
    if _sequence_is_continuous(report.frame_index):
        assert report.frame_index is not None
        print(
            f"  [PASS] frame_index 连续: {report.frame_index.first_value}.."
            f"{report.frame_index.last_value}"
        )
    elif report.frame_index is None:
        print("  [WARN] 没有 frame_index，无法验证帧号连续性")
    else:
        metrics = report.frame_index
        print(
            f"  [WARN] frame_index 不连续: 缺失={metrics.missing_count}，"
            f"重复={metrics.duplicate_count}，倒退={metrics.backward_count}，"
            f"无效={metrics.invalid_count}"
        )
    if _sequence_is_continuous(report.global_index):
        print("  [PASS] 全局 index 连续")
    elif report.global_index is None:
        print("  [WARN] 没有全局 index，无法验证跨 episode 索引")
    else:
        metrics = report.global_index
        print(
            f"  [WARN] 全局 index 不连续: 缺失={metrics.missing_count}，"
            f"重复={metrics.duplicate_count}，倒退={metrics.backward_count}"
        )
    if report.timeline is None:
        print("  [WARN] 没有 timestamp，无法验证采集节奏和速度")
    elif report.timeline.gap_warning_count == 0 and all(
        value == 0
        for value in (
            report.timeline.invalid_count,
            report.timeline.duplicate_timestamp_count,
            report.timeline.backward_timestamp_count,
        )
    ):
        print(
            f"  [PASS] 时间轴连续: {_format_optional(report.timeline.observed_rate_hz)} Hz，"
            "无长间隙"
        )
    else:
        timeline = report.timeline
        print(
            f"  [NOTICE] 时间轴: {timeline.observed_rate_hz or 0.0:.2f} Hz，"
            f"长间隙={timeline.gap_warning_count}，最大={_format_optional(timeline.max_gap_ms)} ms"
        )
    for metrics in report.vectors:
        result = "PASS" if _vector_is_continuous(metrics) else "NOTICE"
        print(
            f"  [{result}] {metrics.name}: 无效行={metrics.invalid_row_count}，"
            f"关节跳变数={int(np.sum(metrics.discontinuity_count))}"
        )
    for metrics in report.cameras:
        result = "PASS" if _camera_is_consistent(metrics) else "NOTICE"
        print(
            f"  [{result}] {metrics.name}: 缺失={metrics.missing_entry_count}，"
            f"重复路径={metrics.duplicate_path_count}，未来时间戳={metrics.future_timestamp_count}"
        )
    _print_rule()
    print("[DETAIL] 帧编号")
    for metrics in (report.frame_index, report.global_index):
        if metrics is None:
            continue
        print(
            f"  {metrics.name}: 首/末={metrics.first_value}/{metrics.last_value}，"
            f"无效={metrics.invalid_count}，错位={metrics.mismatch_count}，"
            f"缺失={metrics.missing_count}，重复={metrics.duplicate_count}，"
            f"倒退={metrics.backward_count}"
        )

    timeline = report.timeline
    _print_rule()
    print("[DETAIL] 采集时间轴")
    if timeline is None:
        print("  没有 timestamp 列，无法分析帧间隔和关节速度")
    else:
        print(
            f"  时长: {_format_optional(timeline.duration_s, 3)} s；"
            f"实际频率: {_format_optional(timeline.observed_rate_hz)} Hz；"
            f"参考频率: {_format_optional(timeline.reference_rate_hz)} Hz"
        )
        print(
            f"  帧间隔: 中位={_format_optional(timeline.median_gap_ms)} ms，"
            f"P95={_format_optional(timeline.p95_gap_ms)} ms，"
            f"最大={_format_optional(timeline.max_gap_ms)} ms"
        )
        print(
            f"  长间隙: {timeline.gap_warning_count} 个 "
            f"(阈值 > {_format_optional(timeline.gap_warning_threshold_ms)} ms)；"
            f"无效/重复/倒退时间戳: {timeline.invalid_count}/"
            f"{timeline.duplicate_timestamp_count}/{timeline.backward_timestamp_count}"
        )

    for metrics in report.vectors:
        _print_rule()
        _print_vector_table(metrics, config)

    if report.alignment is not None:
        _print_rule()
        _print_alignment_table(report.alignment)

    if report.scalars:
        _print_rule()
        print("[GRIPPER] 归一化夹爪开合量")
        print(
            "  列名                              有效  无效     最小       最大       最大步长    超出[0,1]"
        )
        for metrics in report.scalars:
            print(
                f"  {metrics.name:<32} {metrics.valid_count:4d} {metrics.invalid_count:4d} "
                f"{_format_optional(metrics.minimum, 4):>10} "
                f"{_format_optional(metrics.maximum, 4):>10} "
                f"{_format_optional(metrics.maximum_step, 4):>12} "
                f"{metrics.out_of_normalized_range_count:10d}"
            )

    if report.cameras:
        _print_rule()
        print("[CAMERA] 图像路径与相对行时间")
        print(
            "  相机列                              有效  缺失  重复路径  未来时间  中位偏移ms  最小/最大偏移ms"
        )
        for metrics in report.cameras:
            minimum_maximum = (
                f"{_format_optional(metrics.minimum_offset_ms)}/"
                f"{_format_optional(metrics.maximum_offset_ms)}"
            )
            print(
                f"  {metrics.name:<32} {metrics.valid_entry_count:4d} "
                f"{metrics.missing_entry_count:4d} {metrics.duplicate_path_count:8d} "
                f"{metrics.future_timestamp_count:8d} "
                f"{_format_optional(metrics.median_offset_ms):>11} {minimum_maximum:>18}"
            )

    if report.audit is not None:
        _print_rule()
        audit = report.audit
        print(
            f"[AUDIT] records={audit.record_count} emitted={audit.emitted_count} "
            f"skipped={audit.skipped_count} malformed={audit.malformed_count}"
        )
        if audit.skip_reasons:
            print(
                "  skip-reasons: "
                + ", ".join(f"{reason}={count}" for reason, count in audit.skip_reasons)
            )
    _print_rule()


def main() -> int:
    """读取顶部指定的 Parquet 并打印采集质量报告。"""

    if not PARQUET_PATH.is_file():
        raise FileNotFoundError(f"找不到 Parquet 文件: {PARQUET_PATH}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("分析 Parquet 需要安装 pyarrow") from exc
    report = analyze_table(pq.read_table(PARQUET_PATH), PARQUET_PATH)
    print_report(report, ANALYSIS_CONFIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
