#!/usr/bin/env python3
"""诊断 OpenArm Mini -> HCX 双臂遥操作的三段链路。

本程序运行真实的双 OpenArm Mini 到 HCX 双臂直接伺服链路，不加载
``teleop.yaml``，也不接受命令行参数。所有现场连接、映射与测试参数都在本
文件顶部。它同时记录以下三条独立的数据流：

1. 控制器在 100 Hz 主循环中生成的映射/限位/可选平滑后目标；
2. HCX Python 输出线程成功返回 ``DirectServoSession.set_target()`` 的目标；
3. 可选的低频 HCX 实际关节反馈。

三条流分别用于回答：

* 主臂映射目标是否已出现量化平台、缺帧或异常反向；
* Python 到 HCX 薄原生绑定的实际输出是否达到设定频率；
* 已成功提交的平滑目标与实际关节反馈是否存在明显跟踪误差。

反馈时间戳、``set_target()`` 成功返回时间戳均是本机 Python 时间，不能代替
控制器内部时间同步或驱动器周期时间。本 demo 的反馈误差只用于缩小问题范围。

本程序会产生真实运动。必须先确认独立急停、工作空间、轴方向、示教器状态和
硬件保护；只有 ``CONFIRM_DIRECT_SERVO`` 明确为 ``True`` 才会启动直伺服。

运行：

    python -m examples.test_openarm_hcx_teleop_diagnostics
"""

from __future__ import annotations

import csv
import math
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Callable, Literal, Sequence

import numpy as np

from teleop_sdk import TeleopController
from teleop_sdk.adapters import (
    HcxConnection,
    HcxConnectionConfig,
    HcxDirectServoConfig,
    HcxDirectServoOutputStats,
    HcxFollower,
    OpenArmMiniLeaderArm,
)
from teleop_sdk.config import HcxConfig, TeleopConfig

JOINT_COUNT = 7
ARM_AXIS_ORDER = tuple(range(JOINT_COUNT))
ArmSide = Literal["left", "right"]
ARM_SIDES: tuple[ArmSide, ArmSide] = ("left", "right")

# 左右 OpenArm Mini 的只读串口。必须分别连接到两个不同的设备节点。
LEFT_OPENARM_PORT = "/dev/ttyACM1"
RIGHT_OPENARM_PORT = "/dev/ttyACM0"
# 双主臂组合标定文件；本 demo 只读取，绝不修改它。
OPENARM_CALIBRATION_PATH = "./my_openarm_mini.json"
# OpenArm Mini 串口波特率。
OPENARM_BAUDRATE = 1_000_000

# HCX 控制器连接参数。
HCX_LOCAL_IP = "172.16.0.110"
HCX_REMOTE_IP = "172.16.0.89"
HCX_PORT = 12345
HCX_CONNECT_TIMEOUT_S = 10.0
# 控制器项目中左右机械臂的机器人 ID。
LEFT_HCX_ROBOT_ID = 2
RIGHT_HCX_ROBOT_ID = 1

# OpenArm -> HCX 每侧七轴方向；+1.0 为同向，-1.0 为反向。
LEFT_AXIS_SIGN = (-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0)
RIGHT_AXIS_SIGN = (-1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0)

# 主臂读取、映射和通用控制器计算频率，单位为 Hz。
CONTROL_RATE_HZ = 100.0

# HCX Python 输出线程调用一次 set_target() 的目标频率，单位为 Hz。
DIRECT_SERVO_RATE_HZ = 500
# 原生直伺服看门狗时间，单位为秒。它约束“是否持续收到目标”，不限制机械臂静止。
DIRECT_SERVO_WATCHDOG_S = 2.0
# 可选 "direct"、"linear" 或 "limited"；默认与当前遥操作一致。
DIRECT_SERVO_INTERPOLATION: Literal["direct", "linear", "limited"] = "limited"
# limited 模式的最大速度，单位为度/秒。
LIMITED_MAX_VELOCITY_DEG_S = 120.0
# limited 模式的最大加速度，单位为度/秒平方。
LIMITED_MAX_ACCELERATION_DEG_S2 = 80.0
# limited 模式每个高频周期的固定低通权重，范围 [0, 1]。
LIMITED_LOWPASS_ALPHA = 0.2
# 真实直接伺服的显式安全确认；只有为 True 时才会连接后启动运动。
CONFIRM_DIRECT_SERVO = True

# 两级主臂滤波组开关；诊断时应与待比较的正式遥操设置相同。
FILTER_ENABLED = False
# One Euro 最低截止频率，单位为 Hz；FILTER_ENABLED 为 True 时才生效。
FILTER_MINCUTOFF_HZ = 3.0
# One Euro 速度自适应系数；FILTER_ENABLED 为 True 时才生效。
FILTER_BETA = 0.05
# 二级固定低通截止频率，单位为 Hz；FILTER_ENABLED 为 True 时才生效。
TREMOR_CUTOFF_HZ = 5.0
# 弹簧阻尼/前瞻组开关；诊断时应与待比较的正式遥操设置相同。
SPRING_ENABLED = False
# 临界阻尼弹簧固有频率，单位为 rad/s；SPRING_ENABLED 为 True 时才生效。
SPRING_OMEGA = 2.0
# 弹簧阻尼路径最大目标加速度，单位为度/秒平方。
SPRING_MAX_ACCELERATION_DEG_S2 = 50.0
# 弹簧阻尼路径最大目标速度，单位为度/秒。
SPRING_MAX_VELOCITY_DEG_S = 25.0
# 弹簧前瞻时间，单位为毫秒；0.0 表示关闭前瞻预测。
PREDICT_LOOKAHEAD_MS = 0.0
# 控制器死区，单位为度。生成目标流仍会记录死区内目标，便于判断平台来自哪里。
DEAD_ZONE_DEG = 0.005

# True 时后台以低频读取实际 HCX 关节反馈；该读取会与直伺服并发访问控制器。
# 先以 False 跑一次 #1/#2 基线，再设 True 比较 TX miss 是否增加，可判断反馈轮询影响。
FEEDBACK_ENABLED = False
# 每侧 HCX 实际关节反馈轮询频率，单位为 Hz。保持低频，避免干扰直伺服输出线程。
FEEDBACK_RATE_HZ = 10.0

# 本次诊断时长，单位为秒；设为 0.0 时持续运行，直到 Ctrl+C。
TEST_DURATION_S = 30.0
# 终端实时统计输出间隔，单位为秒；不会逐帧打印。
SUMMARY_INTERVAL_S = 1.0
# 保留的诊断历史时长，单位为秒；超过后只保留最新样本，防止长时间运行占用内存。
TRACE_HISTORY_S = 120.0
# 0 表示自动展示当前变化范围最大的关节；填写 1-7 时固定展示指定关节。
REPORT_JOINT_NUMBER = 0
# 留空时不写文件；填写 CSV 路径后导出最近 TRACE_HISTORY_S 秒三条原始流。
DIAGNOSTIC_CSV_PATH = "./csv/false_500.csv"

# 目标相邻帧单轴变化不超过该值时记为平台，单位为度。
SOURCE_SAME_TOLERANCE_DEG = 0.01
# 方向变化小于该值时忽略，避免编码器量化噪声被记作反向，单位为度。
SOURCE_REVERSAL_TOLERANCE_DEG = 0.02
# 两个相邻样本间隔超过标称周期的该倍数时记为输出间隙。
GAP_WARNING_MULTIPLIER = 1.5
# 只有关节总活动范围至少达到该值，才将平台比例或频繁反向作为输入问题提示。
MEANINGFUL_MOTION_RANGE_DEG = 1.0
# 单向移动时每秒方向翻转超过该值，才提示可能存在输入抖动或映射问题。
SOURCE_REVERSAL_WARNING_HZ = 8.0
# 实际反馈相对最近成功提交目标的 P95 绝对误差超过该值时给出跟踪提示，单位为度。
TRACKING_ERROR_WARNING_DEG = 5.0

# True 时在连接后请求控制器脱离示教器；仅限示教器已物理拔除的现场。
AUTO_DETACH_HMI_IF_NO_TEACH_PENDANT = True
# True 时请求清除控制器报警；只在确认报警原因已排除后开启。
AUTO_CLEAR_ALARMS = False
# True 时请求全局和左右单臂使能。
AUTO_ENABLE = True
# 连接建立后、执行启动前置流程前等待控制器稳定的时间，单位为秒。
CONTROLLER_INITIALIZATION_WAIT_S = 2.0
# 等待进入 OP 状态的 EtherCAT 主站索引；非 EtherCAT 部署应填写空元组。
ETHERCAT_MASTER_INDICES: tuple[int, ...] = (0, 1)
# 每个 EtherCAT 主站等待 OP 状态的最长时间，单位为秒。
ETHERCAT_OP_TIMEOUT_S = 15.0
# 自动清报警最大重试次数。
ALARM_CLEAR_RETRY_COUNT = 5
# 自动清报警重试间隔，单位为秒。
ALARM_CLEAR_RETRY_INTERVAL_S = 1.0
# 全局使能最大重试次数。
GLOBAL_ENABLE_RETRY_COUNT = 5
# 全局使能重试间隔，单位为秒。
GLOBAL_ENABLE_RETRY_INTERVAL_S = 1.0
# 单臂使能后等待反馈确认的最长时间，单位为秒。
SINGLE_ARM_ENABLE_TIMEOUT_S = 5.0
# 单臂使能反馈轮询间隔，单位为秒。
ENABLE_STATUS_POLL_INTERVAL_S = 0.1


@dataclass(frozen=True)
class DiagnosticRunConfig:
    """本次三段链路诊断的时间和判定阈值。"""

    control_rate_hz: float
    feedback_enabled: bool
    feedback_rate_hz: float
    test_duration_s: float
    summary_interval_s: float
    trace_history_s: float
    report_joint_number: int
    same_tolerance_deg: float
    reversal_tolerance_deg: float
    gap_warning_multiplier: float
    meaningful_motion_range_deg: float
    source_reversal_warning_hz: float
    tracking_error_warning_deg: float

    def validate(self) -> None:
        for name, value in (
            ("control_rate_hz", self.control_rate_hz),
            ("feedback_rate_hz", self.feedback_rate_hz),
            ("summary_interval_s", self.summary_interval_s),
            ("trace_history_s", self.trace_history_s),
            ("meaningful_motion_range_deg", self.meaningful_motion_range_deg),
            ("source_reversal_warning_hz", self.source_reversal_warning_hz),
            ("tracking_error_warning_deg", self.tracking_error_warning_deg),
        ):
            _positive_finite(name, value)
        _nonnegative_finite("test_duration_s", self.test_duration_s)
        _nonnegative_finite("same_tolerance_deg", self.same_tolerance_deg)
        _nonnegative_finite("reversal_tolerance_deg", self.reversal_tolerance_deg)
        if (
            not math.isfinite(self.gap_warning_multiplier)
            or self.gap_warning_multiplier <= 1.0
        ):
            raise ValueError("gap_warning_multiplier 必须是大于 1 的有限数")
        if not isinstance(self.feedback_enabled, bool):
            raise ValueError("feedback_enabled 必须是布尔值")
        if (
            not isinstance(self.report_joint_number, int)
            or isinstance(self.report_joint_number, bool)
            or not 0 <= self.report_joint_number <= JOINT_COUNT
        ):
            raise ValueError(
                f"report_joint_number 必须是 0 或 1 到 {JOINT_COUNT} 的整数"
            )


DIAGNOSTIC_RUN_CONFIG = DiagnosticRunConfig(
    control_rate_hz=CONTROL_RATE_HZ,
    feedback_enabled=FEEDBACK_ENABLED,
    feedback_rate_hz=FEEDBACK_RATE_HZ,
    test_duration_s=TEST_DURATION_S,
    summary_interval_s=SUMMARY_INTERVAL_S,
    trace_history_s=TRACE_HISTORY_S,
    report_joint_number=REPORT_JOINT_NUMBER,
    same_tolerance_deg=SOURCE_SAME_TOLERANCE_DEG,
    reversal_tolerance_deg=SOURCE_REVERSAL_TOLERANCE_DEG,
    gap_warning_multiplier=GAP_WARNING_MULTIPLIER,
    meaningful_motion_range_deg=MEANINGFUL_MOTION_RANGE_DEG,
    source_reversal_warning_hz=SOURCE_REVERSAL_WARNING_HZ,
    tracking_error_warning_deg=TRACKING_ERROR_WARNING_DEG,
)


DIAGNOSTIC_HCX_CONFIG = HcxConfig(
    local_ip=HCX_LOCAL_IP,
    remote_ip=HCX_REMOTE_IP,
    port=HCX_PORT,
    connect_timeout_s=HCX_CONNECT_TIMEOUT_S,
    left_robot_id=LEFT_HCX_ROBOT_ID,
    right_robot_id=RIGHT_HCX_ROBOT_ID,
    left_axis_sign=LEFT_AXIS_SIGN,
    right_axis_sign=RIGHT_AXIS_SIGN,
    direct_servo_rate_hz=DIRECT_SERVO_RATE_HZ,
    direct_servo_watchdog_s=DIRECT_SERVO_WATCHDOG_S,
    direct_servo_confirm_unsafe=CONFIRM_DIRECT_SERVO,
    direct_servo_interpolation=DIRECT_SERVO_INTERPOLATION,
    direct_servo_limited_max_vel_deg_s=LIMITED_MAX_VELOCITY_DEG_S,
    direct_servo_limited_max_accel_deg_s2=LIMITED_MAX_ACCELERATION_DEG_S2,
    direct_servo_limited_lowpass_alpha=LIMITED_LOWPASS_ALPHA,
    auto_detach_hmi=AUTO_DETACH_HMI_IF_NO_TEACH_PENDANT,
    auto_clear_alarms=AUTO_CLEAR_ALARMS,
    auto_enable=AUTO_ENABLE,
    controller_initialization_wait_s=CONTROLLER_INITIALIZATION_WAIT_S,
    ethercat_master_indices=ETHERCAT_MASTER_INDICES,
    ethercat_op_timeout_s=ETHERCAT_OP_TIMEOUT_S,
    alarm_clear_retry_count=ALARM_CLEAR_RETRY_COUNT,
    alarm_clear_retry_interval_s=ALARM_CLEAR_RETRY_INTERVAL_S,
    global_enable_retry_count=GLOBAL_ENABLE_RETRY_COUNT,
    global_enable_retry_interval_s=GLOBAL_ENABLE_RETRY_INTERVAL_S,
    single_arm_enable_timeout_s=SINGLE_ARM_ENABLE_TIMEOUT_S,
    enable_status_poll_interval_s=ENABLE_STATUS_POLL_INTERVAL_S,
)


DIAGNOSTIC_TELEOP_CONFIG = TeleopConfig(
    rate_hz=CONTROL_RATE_HZ,
    dead_zone_deg=DEAD_ZONE_DEG,
    filter_enabled=FILTER_ENABLED,
    filter_mincutoff_hz=FILTER_MINCUTOFF_HZ,
    filter_beta=FILTER_BETA,
    tremor_cutoff_hz=TREMOR_CUTOFF_HZ,
    spring_enabled=SPRING_ENABLED,
    spring_omega=SPRING_OMEGA,
    max_accel_deg_s2=SPRING_MAX_ACCELERATION_DEG_S2,
    max_vel_deg_s=SPRING_MAX_VELOCITY_DEG_S,
    predict_lookahead_ms=PREDICT_LOOKAHEAD_MS,
    relative_mode=True,
    latency_probe_enabled=False,
)


@dataclass(frozen=True)
class TargetSample:
    """生成或成功提交的一帧七轴目标，所有角度均为度。"""

    timestamp_s: float
    angles_deg: np.ndarray


@dataclass(frozen=True)
class FeedbackSample:
    """一次 HCX 反馈读取结果；失败记录保留错误，不伪造角度。"""

    timestamp_s: float
    started_at_s: float
    duration_s: float
    angles_deg: np.ndarray | None
    error: str | None


@dataclass(frozen=True)
class FeedbackWorkerStats:
    """低频反馈工作线程的调度统计，不代表 HCX 控制器内部周期。"""

    attempt_count: int
    missed_tick_count: int
    maximum_lateness_s: float
    running: bool


@dataclass(frozen=True)
class StreamMetrics:
    """一个七轴目标流的时间连续性与位置连续性指标。"""

    sample_count: int
    observed_rate_hz: float | None
    duration_s: float | None
    mean_gap_s: float | None
    maximum_gap_s: float | None
    gap_warning_count: int
    joint_range_deg: np.ndarray
    same_ratio: np.ndarray
    reversal_count: np.ndarray
    maximum_step_deg: np.ndarray
    maximum_velocity_deg_s: np.ndarray
    maximum_acceleration_deg_s2: np.ndarray


@dataclass(frozen=True)
class FeedbackMetrics:
    """实际反馈读取质量与相对最近成功提交目标的近似误差。"""

    attempt_count: int
    valid_count: int
    error_count: int
    observed_rate_hz: float | None
    mean_read_duration_s: float | None
    maximum_read_duration_s: float | None
    matched_target_count: int
    mean_command_age_s: float | None
    mean_absolute_error_deg: np.ndarray | None
    p95_absolute_error_deg: np.ndarray | None
    maximum_absolute_error_deg: np.ndarray | None


@dataclass(frozen=True)
class SideDiagnosticReport:
    """单侧主臂目标、直伺服输出和实际反馈的汇总结论。"""

    side: ArmSide
    generated: StreamMetrics
    transmitted: StreamMetrics
    feedback: FeedbackMetrics | None
    feedback_worker: FeedbackWorkerStats | None
    direct_servo_stats: HcxDirectServoOutputStats | None
    conclusions: tuple[str, ...]


@dataclass(frozen=True)
class ControlLoopStats:
    """双 OpenArm 主控调度的本机统计。"""

    cycle_count: int
    missed_tick_count: int
    maximum_lateness_s: float


@dataclass(frozen=True)
class TraceSnapshot:
    """当前保留的三段原始诊断流。"""

    generated: dict[ArmSide, tuple[TargetSample, ...]]
    transmitted: dict[ArmSide, tuple[TargetSample, ...]]
    feedback: dict[ArmSide, tuple[FeedbackSample, ...]]


@dataclass(frozen=True)
class DiagnosticReport:
    """一次真实三段链路诊断的最终结果。"""

    elapsed_s: float
    interrupted: bool
    control_loop: ControlLoopStats
    left: SideDiagnosticReport
    right: SideDiagnosticReport
    traces: TraceSnapshot


class TraceRecorder:
    """让实时线程无阻塞地向主线程交付诊断样本。"""

    def __init__(
        self,
        control_rate_hz: float,
        direct_servo_rate_hz: int,
        feedback_rate_hz: float,
        history_s: float,
    ) -> None:
        generated_maxlen = max(512, math.ceil(control_rate_hz * history_s * 1.2))
        transmitted_maxlen = max(
            2_048, math.ceil(direct_servo_rate_hz * history_s * 1.2)
        )
        feedback_maxlen = max(256, math.ceil(feedback_rate_hz * history_s * 1.2))
        self._generated_pending: dict[ArmSide, SimpleQueue[TargetSample]] = {
            side: SimpleQueue() for side in ARM_SIDES
        }
        self._transmitted_pending: dict[ArmSide, SimpleQueue[TargetSample]] = {
            side: SimpleQueue() for side in ARM_SIDES
        }
        self._feedback_pending: dict[ArmSide, SimpleQueue[FeedbackSample]] = {
            side: SimpleQueue() for side in ARM_SIDES
        }
        self._generated: dict[ArmSide, deque[TargetSample]] = {
            side: deque(maxlen=generated_maxlen) for side in ARM_SIDES
        }
        self._transmitted: dict[ArmSide, deque[TargetSample]] = {
            side: deque(maxlen=transmitted_maxlen) for side in ARM_SIDES
        }
        self._feedback: dict[ArmSide, deque[FeedbackSample]] = {
            side: deque(maxlen=feedback_maxlen) for side in ARM_SIDES
        }

    def generated_callback(self, side: ArmSide) -> Callable[[np.ndarray, float], None]:
        """返回控制器生成目标的轻量回调。"""

        def record(angles_deg: np.ndarray, timestamp_s: float) -> None:
            self._generated_pending[side].put(
                # 控制器已传入独立副本；不要再复制，避免扰动 100 Hz 主控循环。
                TargetSample(float(timestamp_s), np.asarray(angles_deg, dtype=float))
            )

        return record

    def transmitted_callback(
        self, side: ArmSide
    ) -> Callable[[np.ndarray, float], None]:
        """返回直伺服成功提交目标的轻量回调。"""

        def record(angles_deg: np.ndarray, timestamp_s: float) -> None:
            self._transmitted_pending[side].put(
                # HCX 适配器已传入独立副本；输出线程这里只做一次非阻塞入队。
                TargetSample(float(timestamp_s), np.asarray(angles_deg, dtype=float))
            )

        return record

    def record_feedback(
        self,
        side: ArmSide,
        *,
        timestamp_s: float,
        started_at_s: float,
        duration_s: float,
        angles_deg: np.ndarray | None,
        error: str | None,
    ) -> None:
        """从低频反馈工作线程非阻塞地提交一次读取结果。"""

        values = (
            None if angles_deg is None else np.asarray(angles_deg, dtype=float).copy()
        )
        self._feedback_pending[side].put(
            FeedbackSample(
                timestamp_s=float(timestamp_s),
                started_at_s=float(started_at_s),
                duration_s=max(0.0, float(duration_s)),
                angles_deg=values,
                error=error,
            )
        )

    def drain(self) -> None:
        """仅由主线程将待处理样本转入有界历史。"""

        for side in ARM_SIDES:
            self._drain_queue(self._generated_pending[side], self._generated[side])
            self._drain_queue(self._transmitted_pending[side], self._transmitted[side])
            self._drain_queue(self._feedback_pending[side], self._feedback[side])

    @staticmethod
    def _drain_queue(queue: SimpleQueue, output: deque) -> None:
        while True:
            try:
                output.append(queue.get_nowait())
            except Empty:
                return

    def snapshot(self) -> TraceSnapshot:
        """返回仅含 Python 内存历史的只读快照。"""

        self.drain()
        return TraceSnapshot(
            generated={side: tuple(self._generated[side]) for side in ARM_SIDES},
            transmitted={side: tuple(self._transmitted[side]) for side in ARM_SIDES},
            feedback={side: tuple(self._feedback[side]) for side in ARM_SIDES},
        )


class FeedbackPoller:
    """在独立低频线程中读取一侧 HCX 实际关节反馈。"""

    def __init__(
        self,
        side: ArmSide,
        follower: HcxFollower,
        rate_hz: float,
        recorder: TraceRecorder,
    ) -> None:
        self._side = side
        self._follower = follower
        self._period_s = 1.0 / rate_hz
        self._recorder = recorder
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._attempt_count = 0
        self._missed_tick_count = 0
        self._maximum_lateness_s = 0.0
        self._thread = threading.Thread(
            target=self._run,
            name=f"hcx-{side}-feedback-diagnostic",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> bool:
        """请求退出；读取卡住时不无限等待，调用方仍可继续停止直伺服。"""

        self._stop_event.set()
        self._thread.join(timeout=max(1.0, 2.0 * self._period_s))
        return not self._thread.is_alive()

    def snapshot(self) -> FeedbackWorkerStats:
        with self._lock:
            return FeedbackWorkerStats(
                attempt_count=self._attempt_count,
                missed_tick_count=self._missed_tick_count,
                maximum_lateness_s=self._maximum_lateness_s,
                running=self._thread.is_alive() and not self._stop_event.is_set(),
            )

    def _run(self) -> None:
        next_tick_s = time.perf_counter()
        while not self._stop_event.is_set():
            remaining_s = next_tick_s - time.perf_counter()
            if remaining_s > 0.0 and self._stop_event.wait(remaining_s):
                return

            started_at_s = time.perf_counter()
            lateness_s = max(0.0, started_at_s - next_tick_s)
            with self._lock:
                self._attempt_count += 1
                self._maximum_lateness_s = max(self._maximum_lateness_s, lateness_s)
            try:
                angles_deg = self._follower.read_joint_angles_deg()
                error = None
            except Exception as exc:
                angles_deg = None
                error = f"{type(exc).__name__}: {exc}"
            completed_at_s = time.perf_counter()
            self._recorder.record_feedback(
                self._side,
                timestamp_s=completed_at_s,
                started_at_s=started_at_s,
                duration_s=completed_at_s - started_at_s,
                angles_deg=angles_deg,
                error=error,
            )

            next_tick_s += self._period_s
            if completed_at_s >= next_tick_s:
                skipped = 1 + int((completed_at_s - next_tick_s) / self._period_s)
                with self._lock:
                    self._missed_tick_count += skipped
                next_tick_s = completed_at_s + self._period_s


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


def _build_direct_servo_config(
    hcx_config: HcxConfig, control_rate_hz: float
) -> HcxDirectServoConfig:
    """从顶部 HCX 常量构造直伺服配置，不读取部署 YAML。"""

    source_rate_hz: int | None = None
    if hcx_config.direct_servo_interpolation in ("linear", "limited"):
        if not float(control_rate_hz).is_integer():
            raise ValueError("linear/limited 模式要求 control_rate_hz 为整数")
        source_rate_hz = int(control_rate_hz)
    direct_config = HcxDirectServoConfig.from_runtime_config(
        hcx_config, source_rate_hz=source_rate_hz
    )
    if not direct_config.confirm_unsafe:
        raise ValueError(
            "本 demo 会产生真实运动；请在顶部明确设置 CONFIRM_DIRECT_SERVO = True"
        )
    if direct_config.watchdog_s <= 1.0 / direct_config.rate_hz:
        raise ValueError("DIRECT_SERVO_WATCHDOG_S 必须大于一个直伺服输出周期")
    return direct_config


def _control_config_for_side(
    base_config: TeleopConfig, axis_sign: tuple[float, ...]
) -> TeleopConfig:
    """创建一侧 HCX 七轴相对映射控制器配置。"""

    if len(axis_sign) != JOINT_COUNT or any(
        value not in (-1.0, 1.0) for value in axis_sign
    ):
        raise ValueError("每侧 axis_sign 必须是七个 +1.0 或 -1.0")
    return replace(
        base_config,
        axis_order=ARM_AXIS_ORDER,
        axis_sign=axis_sign,
        relative_mode=True,
    )


def _create_controller(
    *,
    side: ArmSide,
    port: str,
    axis_sign: tuple[float, ...],
    follower: HcxFollower,
    base_config: TeleopConfig,
    recorder: TraceRecorder,
) -> TeleopController:
    """创建一侧只读 OpenArm 与 HCX 从臂控制链，不接入夹爪。"""

    leader = OpenArmMiniLeaderArm(
        port=port,
        calibration_path=OPENARM_CALIBRATION_PATH,
        side=side,
        baudrate=OPENARM_BAUDRATE,
        read_only=True,
    )
    return TeleopController(
        leader,
        follower,
        _control_config_for_side(base_config, axis_sign),
        on_joint_target_generated=recorder.generated_callback(side),
    )


def _target_arrays(
    samples: Sequence[TargetSample],
) -> tuple[np.ndarray, np.ndarray]:
    """过滤无效样本并按时间升序返回时间和七轴目标矩阵。"""

    timestamps: list[float] = []
    targets: list[np.ndarray] = []
    for sample in samples:
        try:
            timestamp_s = float(sample.timestamp_s)
            target = np.asarray(sample.angles_deg, dtype=float)
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(timestamp_s)
            or target.shape != (JOINT_COUNT,)
            or not np.isfinite(target).all()
        ):
            continue
        timestamps.append(timestamp_s)
        targets.append(target.copy())
    if not timestamps:
        return np.empty(0, dtype=float), np.empty((0, JOINT_COUNT), dtype=float)
    order = np.argsort(np.asarray(timestamps, dtype=float), kind="stable")
    return (
        np.asarray(timestamps, dtype=float)[order],
        np.asarray(targets, dtype=float)[order],
    )


def analyze_stream(
    samples: Sequence[TargetSample],
    *,
    expected_rate_hz: float,
    same_tolerance_deg: float,
    reversal_tolerance_deg: float,
    gap_warning_multiplier: float,
) -> StreamMetrics:
    """计算单个生成/发送目标流的节奏、平台和运动连续性。"""

    expected_rate_hz = _positive_finite("expected_rate_hz", expected_rate_hz)
    same_tolerance_deg = _nonnegative_finite("same_tolerance_deg", same_tolerance_deg)
    reversal_tolerance_deg = _nonnegative_finite(
        "reversal_tolerance_deg", reversal_tolerance_deg
    )
    if not math.isfinite(gap_warning_multiplier) or gap_warning_multiplier <= 1.0:
        raise ValueError("gap_warning_multiplier 必须是大于 1 的有限数")

    timestamps, targets = _target_arrays(samples)
    zero = np.zeros(JOINT_COUNT, dtype=float)
    if timestamps.size == 0:
        return StreamMetrics(
            sample_count=0,
            observed_rate_hz=None,
            duration_s=None,
            mean_gap_s=None,
            maximum_gap_s=None,
            gap_warning_count=0,
            joint_range_deg=zero,
            same_ratio=zero,
            reversal_count=np.zeros(JOINT_COUNT, dtype=int),
            maximum_step_deg=zero,
            maximum_velocity_deg_s=zero,
            maximum_acceleration_deg_s2=zero,
        )

    duration_s = float(timestamps[-1] - timestamps[0])
    observed_rate_hz = (
        (timestamps.size - 1) / duration_s
        if timestamps.size >= 2 and duration_s > 0.0
        else None
    )
    joint_range_deg = np.ptp(targets, axis=0)
    if timestamps.size < 2:
        return StreamMetrics(
            sample_count=int(timestamps.size),
            observed_rate_hz=observed_rate_hz,
            duration_s=duration_s,
            mean_gap_s=None,
            maximum_gap_s=None,
            gap_warning_count=0,
            joint_range_deg=joint_range_deg,
            same_ratio=zero,
            reversal_count=np.zeros(JOINT_COUNT, dtype=int),
            maximum_step_deg=zero,
            maximum_velocity_deg_s=zero,
            maximum_acceleration_deg_s2=zero,
        )

    gaps_s = np.diff(timestamps)
    deltas_deg = np.diff(targets, axis=0)
    valid = gaps_s > 0.0
    if not np.any(valid):
        return StreamMetrics(
            sample_count=int(timestamps.size),
            observed_rate_hz=observed_rate_hz,
            duration_s=duration_s,
            mean_gap_s=None,
            maximum_gap_s=None,
            gap_warning_count=0,
            joint_range_deg=joint_range_deg,
            same_ratio=zero,
            reversal_count=np.zeros(JOINT_COUNT, dtype=int),
            maximum_step_deg=np.max(np.abs(deltas_deg), axis=0),
            maximum_velocity_deg_s=zero,
            maximum_acceleration_deg_s2=zero,
        )

    valid_gaps_s = gaps_s[valid]
    valid_deltas_deg = deltas_deg[valid]
    # 静止前后的长时间保持不应被误报为“主臂量化平台”。每个关节仅从首次
    # 有效变化到末次有效变化之间计算平台比例；慢速运动中的实际阶梯仍会保留。
    same_ratio = np.ones(JOINT_COUNT, dtype=float)
    for axis in range(JOINT_COUNT):
        moved = np.flatnonzero(np.abs(valid_deltas_deg[:, axis]) > same_tolerance_deg)
        if moved.size:
            movement_deltas = valid_deltas_deg[moved[0] : moved[-1] + 1, axis]
            same_ratio[axis] = float(
                np.mean(np.abs(movement_deltas) <= same_tolerance_deg)
            )
    maximum_step_deg = np.max(np.abs(valid_deltas_deg), axis=0)
    velocities_deg_s = valid_deltas_deg / valid_gaps_s[:, np.newaxis]
    maximum_velocity_deg_s = np.max(np.abs(velocities_deg_s), axis=0)
    if velocities_deg_s.shape[0] >= 2:
        velocity_times_s = timestamps[:-1][valid] + valid_gaps_s / 2.0
        acceleration_gaps_s = np.diff(velocity_times_s)
        valid_acceleration = acceleration_gaps_s > 0.0
        if np.any(valid_acceleration):
            accelerations_deg_s2 = (
                np.diff(velocities_deg_s, axis=0)[valid_acceleration]
                / acceleration_gaps_s[valid_acceleration, np.newaxis]
            )
            maximum_acceleration_deg_s2 = np.max(np.abs(accelerations_deg_s2), axis=0)
        else:
            maximum_acceleration_deg_s2 = zero
    else:
        maximum_acceleration_deg_s2 = zero

    reversals = np.zeros(JOINT_COUNT, dtype=int)
    previous_direction = np.zeros(JOINT_COUNT, dtype=int)
    for delta_deg in valid_deltas_deg:
        direction = np.sign(delta_deg).astype(int)
        direction[np.abs(delta_deg) <= reversal_tolerance_deg] = 0
        changed = (
            (direction != 0)
            & (previous_direction != 0)
            & (direction != previous_direction)
        )
        reversals += changed.astype(int)
        previous_direction[direction != 0] = direction[direction != 0]

    expected_period_s = 1.0 / expected_rate_hz
    return StreamMetrics(
        sample_count=int(timestamps.size),
        observed_rate_hz=observed_rate_hz,
        duration_s=duration_s,
        mean_gap_s=float(np.mean(valid_gaps_s)),
        maximum_gap_s=float(np.max(valid_gaps_s)),
        gap_warning_count=int(
            np.count_nonzero(valid_gaps_s > gap_warning_multiplier * expected_period_s)
        ),
        joint_range_deg=joint_range_deg,
        same_ratio=same_ratio,
        reversal_count=reversals,
        maximum_step_deg=maximum_step_deg,
        maximum_velocity_deg_s=maximum_velocity_deg_s,
        maximum_acceleration_deg_s2=maximum_acceleration_deg_s2,
    )


def analyze_feedback(
    samples: Sequence[FeedbackSample],
    transmitted: Sequence[TargetSample],
    *,
    expected_rate_hz: float,
) -> FeedbackMetrics:
    """将每个反馈与其完成前最近成功提交的目标进行近似匹配。"""

    _positive_finite("expected_rate_hz", expected_rate_hz)
    attempt_count = len(samples)
    valid_samples: list[FeedbackSample] = []
    read_durations_s: list[float] = []
    for sample in samples:
        read_durations_s.append(max(0.0, float(sample.duration_s)))
        if sample.angles_deg is None:
            continue
        values = np.asarray(sample.angles_deg, dtype=float)
        if values.shape == (JOINT_COUNT,) and np.isfinite(values).all():
            valid_samples.append(sample)

    valid_samples.sort(key=lambda sample: sample.timestamp_s)
    if len(valid_samples) >= 2:
        feedback_timestamps_s = np.asarray(
            [sample.timestamp_s for sample in valid_samples], dtype=float
        )
        feedback_duration_s = feedback_timestamps_s[-1] - feedback_timestamps_s[0]
        observed_rate_hz = (
            (len(valid_samples) - 1) / feedback_duration_s
            if feedback_duration_s > 0.0
            else None
        )
    else:
        observed_rate_hz = None

    transmitted_timestamps_s, transmitted_targets_deg = _target_arrays(transmitted)
    errors_deg: list[np.ndarray] = []
    command_ages_s: list[float] = []
    for sample in valid_samples:
        target_index = int(
            np.searchsorted(transmitted_timestamps_s, sample.timestamp_s, side="right")
            - 1
        )
        if target_index < 0:
            continue
        feedback_deg = np.asarray(sample.angles_deg, dtype=float)
        errors_deg.append(feedback_deg - transmitted_targets_deg[target_index])
        command_ages_s.append(
            max(0.0, sample.timestamp_s - transmitted_timestamps_s[target_index])
        )

    if errors_deg:
        absolute_errors_deg = np.abs(np.asarray(errors_deg, dtype=float))
        mean_error = np.mean(absolute_errors_deg, axis=0)
        p95_error = np.quantile(absolute_errors_deg, 0.95, axis=0)
        maximum_error = np.max(absolute_errors_deg, axis=0)
        mean_command_age_s: float | None = float(np.mean(command_ages_s))
    else:
        mean_error = None
        p95_error = None
        maximum_error = None
        mean_command_age_s = None

    return FeedbackMetrics(
        attempt_count=attempt_count,
        valid_count=len(valid_samples),
        error_count=attempt_count - len(valid_samples),
        observed_rate_hz=observed_rate_hz,
        mean_read_duration_s=(
            float(np.mean(read_durations_s)) if read_durations_s else None
        ),
        maximum_read_duration_s=(
            float(np.max(read_durations_s)) if read_durations_s else None
        ),
        matched_target_count=len(errors_deg),
        mean_command_age_s=mean_command_age_s,
        mean_absolute_error_deg=mean_error,
        p95_absolute_error_deg=p95_error,
        maximum_absolute_error_deg=maximum_error,
    )


def diagnose_side(
    *,
    generated: StreamMetrics,
    transmitted: StreamMetrics,
    feedback: FeedbackMetrics | None,
    direct_servo_stats: HcxDirectServoOutputStats | None,
    config: DiagnosticRunConfig,
) -> tuple[str, ...]:
    """从三个独立指标产生保守的排查提示，不将近似反馈误差当作故障码。"""

    conclusions: list[str] = []
    minimum_control_rate = config.control_rate_hz * 0.95
    source_duration_s = generated.duration_s or 0.0
    active_axes = generated.joint_range_deg >= config.meaningful_motion_range_deg
    plateau_axes = np.flatnonzero(active_axes & (generated.same_ratio >= 0.75))
    reversal_rates = (
        generated.reversal_count / source_duration_s
        if source_duration_s > 0.0
        else np.zeros(JOINT_COUNT)
    )
    noisy_axes = np.flatnonzero(
        active_axes & (reversal_rates >= config.source_reversal_warning_hz)
    )

    if generated.sample_count < 2:
        conclusions.append("#1 控制器生成目标样本不足，不能判断主臂映射连续性。")
    elif (
        generated.observed_rate_hz is not None
        and generated.observed_rate_hz < minimum_control_rate
    ) or generated.gap_warning_count:
        conclusions.append(
            "#1 控制器生成目标存在低于设定频率或较长间隙；先检查主控循环和主臂读取。"
        )
    elif plateau_axes.size:
        conclusions.append(
            "#1 活动关节的生成目标存在较多平台；这可能是主臂编码器量化/死区，"
            "也可能在慢速运动时形成阶梯输入。"
        )
    elif noisy_axes.size:
        conclusions.append(
            "#1 活动关节出现频繁方向反转；若测试时为单向移动，应检查主臂噪声、"
            "轴方向或映射。"
        )
    else:
        conclusions.append("#1 本次生成目标未见明显的频率缺口或异常反向。")

    expected_direct_rate = (
        direct_servo_stats.configured_rate_hz
        if direct_servo_stats is not None
        else DIRECT_SERVO_RATE_HZ
    )
    output_rate_low = (
        transmitted.observed_rate_hz is not None
        and transmitted.observed_rate_hz < expected_direct_rate * 0.95
    )
    output_missed = (
        direct_servo_stats is not None and direct_servo_stats.missed_tick_count > 0
    )
    if transmitted.sample_count < 2:
        conclusions.append("#2 成功 set_target 样本不足，不能判断 Python 直伺服输出。")
    elif output_rate_low or transmitted.gap_warning_count or output_missed:
        conclusions.append(
            "#2 Python 直伺服输出存在降频、间隙或 miss；重点查看 set_target 耗时、"
            "线程调度与控制器通信。"
        )
    else:
        conclusions.append("#2 本次成功 set_target 流接近设定频率，未见明显输出缺口。")

    if feedback is None:
        conclusions.append("#3 当前 FEEDBACK_ENABLED=False，未读取实际关节反馈。")
    elif feedback.valid_count == 0:
        conclusions.append("#3 未获得有效 HCX 反馈；先检查关节状态读取和控制器通信。")
    elif feedback.matched_target_count == 0:
        conclusions.append("#3 反馈未能匹配到成功提交目标，不能评估实际跟踪。")
    elif (
        feedback.p95_absolute_error_deg is not None
        and float(np.max(feedback.p95_absolute_error_deg))
        > config.tracking_error_warning_deg
    ):
        conclusions.append(
            "#3 实际反馈相对最近成功提交目标的 P95 误差较大；在慢速单轴、"
            "保持一段时间的测试中复核驱动器跟踪、限位、报警和机械振动。"
        )
    else:
        conclusions.append(
            "#3 本次低频反馈未见超过阈值的近似跟踪误差；时间戳仍非控制器时钟。"
        )
    return tuple(conclusions)


def _safe_direct_servo_stats(
    follower: HcxFollower,
) -> HcxDirectServoOutputStats | None:
    """读取纯 Python 输出计数器；失败时诊断继续运行。"""

    try:
        return follower.direct_servo_output_stats()
    except Exception:
        return None


def _recent_targets(
    samples: Sequence[TargetSample], now_s: float, window_s: float
) -> tuple[TargetSample, ...]:
    return tuple(sample for sample in samples if sample.timestamp_s >= now_s - window_s)


def _recent_feedback(
    samples: Sequence[FeedbackSample], now_s: float, window_s: float
) -> tuple[FeedbackSample, ...]:
    return tuple(sample for sample in samples if sample.timestamp_s >= now_s - window_s)


def _report_axis(metrics: StreamMetrics, configured_axis: int) -> int:
    if configured_axis:
        return configured_axis - 1
    return int(np.argmax(metrics.joint_range_deg))


def _rate_text(rate_hz: float | None) -> str:
    return "--" if rate_hz is None else f"{rate_hz:.1f}"


def _live_side_text(
    side: ArmSide,
    traces: TraceSnapshot,
    follower: HcxFollower,
    config: DiagnosticRunConfig,
    now_s: float,
) -> str:
    """格式化最近一秒的单侧简洁状态，避免每帧终端输出。"""

    generated = analyze_stream(
        _recent_targets(traces.generated[side], now_s, config.summary_interval_s),
        expected_rate_hz=config.control_rate_hz,
        same_tolerance_deg=config.same_tolerance_deg,
        reversal_tolerance_deg=config.reversal_tolerance_deg,
        gap_warning_multiplier=config.gap_warning_multiplier,
    )
    direct_stats = _safe_direct_servo_stats(follower)
    expected_direct_rate = (
        direct_stats.configured_rate_hz
        if direct_stats is not None
        else DIRECT_SERVO_RATE_HZ
    )
    transmitted = analyze_stream(
        _recent_targets(traces.transmitted[side], now_s, config.summary_interval_s),
        expected_rate_hz=expected_direct_rate,
        same_tolerance_deg=config.same_tolerance_deg,
        reversal_tolerance_deg=config.reversal_tolerance_deg,
        gap_warning_multiplier=config.gap_warning_multiplier,
    )
    axis = _report_axis(generated, config.report_joint_number)
    source_gap_ms = (
        "--"
        if generated.maximum_gap_s is None
        else f"{generated.maximum_gap_s * 1_000.0:.1f}"
    )
    tx_gap_ms = (
        "--"
        if transmitted.maximum_gap_s is None
        else f"{transmitted.maximum_gap_s * 1_000.0:.1f}"
    )
    tx_miss = (
        "--" if direct_stats is None else str(direct_stats.recent_missed_tick_count)
    )
    call_max_ms = (
        "--"
        if direct_stats is None or direct_stats.max_set_target_duration_s is None
        else f"{direct_stats.max_set_target_duration_s * 1_000.0:.1f}"
    )
    text = (
        f"{side[0].upper()} src {_rate_text(generated.observed_rate_hz)}/{config.control_rate_hz:.0f}Hz "
        f"gap {source_gap_ms}ms J{axis + 1} plateau {generated.same_ratio[axis] * 100.0:.0f}% "
        f"| tx {_rate_text(transmitted.observed_rate_hz)}/{expected_direct_rate}Hz "
        f"gap {tx_gap_ms}ms miss {tx_miss} call-max {call_max_ms}ms"
    )
    if config.feedback_enabled:
        feedback = analyze_feedback(
            _recent_feedback(traces.feedback[side], now_s, config.summary_interval_s),
            _recent_targets(traces.transmitted[side], now_s, config.summary_interval_s),
            expected_rate_hz=config.feedback_rate_hz,
        )
        tracking_text = (
            "--"
            if feedback.p95_absolute_error_deg is None
            else f"{feedback.p95_absolute_error_deg[axis]:.2f}deg"
        )
        read_max_ms = (
            "--"
            if feedback.maximum_read_duration_s is None
            else f"{feedback.maximum_read_duration_s * 1_000.0:.1f}ms"
        )
        text += (
            f" | fb {_rate_text(feedback.observed_rate_hz)}/{config.feedback_rate_hz:.0f}Hz "
            f"read-max {read_max_ms} p95 {tracking_text}"
        )
    return text


def _print_live_summary(
    recorder: TraceRecorder,
    left_follower: HcxFollower,
    right_follower: HcxFollower,
    config: DiagnosticRunConfig,
    now_s: float,
) -> None:
    traces = recorder.snapshot()
    print(
        "[DIAG] "
        + _live_side_text("left", traces, left_follower, config, now_s)
        + "\n       "
        + _live_side_text("right", traces, right_follower, config, now_s)
    )


def _run_control_loop(
    left_controller: TeleopController,
    right_controller: TeleopController,
    left_follower: HcxFollower,
    right_follower: HcxFollower,
    recorder: TraceRecorder,
    config: DiagnosticRunConfig,
) -> tuple[ControlLoopStats, bool]:
    """并行驱动左右 100 Hz 控制器，并以固定频率显示旁路统计。"""

    period_s = 1.0 / config.control_rate_hz
    started_at_s = time.perf_counter()
    next_tick_s = started_at_s
    next_summary_s = started_at_s + config.summary_interval_s
    cycle_count = 0
    missed_tick_count = 0
    maximum_lateness_s = 0.0
    interrupted = False

    try:
        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="openarm-hcx-diagnostic"
        ) as executor:
            while True:
                now_s = time.perf_counter()
                if (
                    config.test_duration_s > 0.0
                    and now_s - started_at_s >= config.test_duration_s
                ):
                    break
                left_step = executor.submit(left_controller.step, now_s)
                right_step = executor.submit(right_controller.step, now_s)
                left_step.result()
                right_step.result()
                cycle_count += 1
                recorder.drain()

                summary_now_s = time.perf_counter()
                if summary_now_s >= next_summary_s:
                    _print_live_summary(
                        recorder,
                        left_follower,
                        right_follower,
                        config,
                        summary_now_s,
                    )
                    next_summary_s = summary_now_s + config.summary_interval_s

                next_tick_s += period_s
                completed_at_s = time.perf_counter()
                if completed_at_s < next_tick_s:
                    time.sleep(next_tick_s - completed_at_s)
                    continue

                lateness_s = completed_at_s - next_tick_s
                maximum_lateness_s = max(maximum_lateness_s, lateness_s)
                skipped = 1 + int(lateness_s / period_s)
                missed_tick_count += skipped
                next_tick_s = completed_at_s
    except KeyboardInterrupt:
        interrupted = True
        print("\n[STOP] 收到退出请求，正在停止诊断直伺服。")

    return (
        ControlLoopStats(
            cycle_count=cycle_count,
            missed_tick_count=missed_tick_count,
            maximum_lateness_s=maximum_lateness_s,
        ),
        interrupted,
    )


def _side_report(
    *,
    side: ArmSide,
    traces: TraceSnapshot,
    follower: HcxFollower,
    feedback_worker: FeedbackPoller | None,
    config: DiagnosticRunConfig,
) -> SideDiagnosticReport:
    direct_stats = _safe_direct_servo_stats(follower)
    expected_direct_rate = (
        direct_stats.configured_rate_hz
        if direct_stats is not None
        else DIRECT_SERVO_RATE_HZ
    )
    generated = analyze_stream(
        traces.generated[side],
        expected_rate_hz=config.control_rate_hz,
        same_tolerance_deg=config.same_tolerance_deg,
        reversal_tolerance_deg=config.reversal_tolerance_deg,
        gap_warning_multiplier=config.gap_warning_multiplier,
    )
    transmitted = analyze_stream(
        traces.transmitted[side],
        expected_rate_hz=expected_direct_rate,
        same_tolerance_deg=config.same_tolerance_deg,
        reversal_tolerance_deg=config.reversal_tolerance_deg,
        gap_warning_multiplier=config.gap_warning_multiplier,
    )
    feedback = (
        analyze_feedback(
            traces.feedback[side],
            traces.transmitted[side],
            expected_rate_hz=config.feedback_rate_hz,
        )
        if config.feedback_enabled
        else None
    )
    worker_stats = feedback_worker.snapshot() if feedback_worker is not None else None
    return SideDiagnosticReport(
        side=side,
        generated=generated,
        transmitted=transmitted,
        feedback=feedback,
        feedback_worker=worker_stats,
        direct_servo_stats=direct_stats,
        conclusions=diagnose_side(
            generated=generated,
            transmitted=transmitted,
            feedback=feedback,
            direct_servo_stats=direct_stats,
            config=config,
        ),
    )


def run_diagnostic(
    hcx_config: HcxConfig,
    teleop_config: TeleopConfig,
    run_config: DiagnosticRunConfig,
) -> DiagnosticReport:
    """执行真实双臂三段链路诊断并在返回前停止所有设备。"""

    run_config.validate()
    if not math.isclose(
        float(teleop_config.rate_hz),
        run_config.control_rate_hz,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("teleop_config.rate_hz 必须与 control_rate_hz 一致")
    direct_config = _build_direct_servo_config(hcx_config, run_config.control_rate_hz)
    connection = HcxConnection(HcxConnectionConfig.from_runtime_config(hcx_config))
    recorder = TraceRecorder(
        run_config.control_rate_hz,
        direct_config.rate_hz,
        run_config.feedback_rate_hz,
        run_config.trace_history_s,
    )
    controllers: list[TeleopController] = []
    feedback_pollers: dict[ArmSide, FeedbackPoller] = {}
    left_follower: HcxFollower | None = None
    right_follower: HcxFollower | None = None
    control_loop = ControlLoopStats(0, 0, 0.0)
    interrupted = False
    started_at_s = time.perf_counter()
    report: DiagnosticReport | None = None

    try:
        left_follower = HcxFollower(
            connection,
            robot_id=hcx_config.left_robot_id,
            side="left",
            direct_servo_config=direct_config,
            on_direct_servo_target_submitted=recorder.transmitted_callback("left"),
        )
        right_follower = HcxFollower(
            connection,
            robot_id=hcx_config.right_robot_id,
            side="right",
            direct_servo_config=direct_config,
            on_direct_servo_target_submitted=recorder.transmitted_callback("right"),
        )
        left_controller = _create_controller(
            side="left",
            port=LEFT_OPENARM_PORT,
            axis_sign=hcx_config.left_axis_sign,
            follower=left_follower,
            base_config=teleop_config,
            recorder=recorder,
        )
        right_controller = _create_controller(
            side="right",
            port=RIGHT_OPENARM_PORT,
            axis_sign=hcx_config.right_axis_sign,
            follower=right_follower,
            base_config=teleop_config,
            recorder=recorder,
        )
        controllers = [left_controller, right_controller]

        for controller in controllers:
            controller.connect()
        if not left_controller.start_servo():
            raise RuntimeError("HCX 左臂未通过直伺服启动前置检查")
        if not right_controller.start_servo():
            raise RuntimeError("HCX 右臂未通过直伺服启动前置检查")

        _print_start_message(direct_config, run_config)
        if run_config.feedback_enabled:
            feedback_pollers = {
                "left": FeedbackPoller(
                    "left", left_follower, run_config.feedback_rate_hz, recorder
                ),
                "right": FeedbackPoller(
                    "right", right_follower, run_config.feedback_rate_hz, recorder
                ),
            }
            for poller in feedback_pollers.values():
                poller.start()

        control_loop, interrupted = _run_control_loop(
            left_controller,
            right_controller,
            left_follower,
            right_follower,
            recorder,
            run_config,
        )
        for side, poller in feedback_pollers.items():
            if not poller.stop():
                print(f"[WARN] HCX {side} 反馈诊断线程未在超时内退出")
        recorder.drain()
        traces = recorder.snapshot()
        report = DiagnosticReport(
            elapsed_s=time.perf_counter() - started_at_s,
            interrupted=interrupted,
            control_loop=control_loop,
            left=_side_report(
                side="left",
                traces=traces,
                follower=left_follower,
                feedback_worker=feedback_pollers.get("left"),
                config=run_config,
            ),
            right=_side_report(
                side="right",
                traces=traces,
                follower=right_follower,
                feedback_worker=feedback_pollers.get("right"),
                config=run_config,
            ),
            traces=traces,
        )
    finally:
        for side, poller in feedback_pollers.items():
            if not poller.stop():
                print(f"[WARN] HCX {side} 反馈诊断线程未在超时内退出")
        recorder.drain()
        for controller in reversed(controllers):
            try:
                controller.shutdown()
            except Exception as exc:
                print(f"[WARN] 关闭一侧诊断控制链失败: {exc}")
    if report is None:
        raise RuntimeError("诊断在生成结果前结束")
    return report


def _axis_values_text(values: np.ndarray, *, precision: int = 2) -> str:
    return " ".join(
        f"J{axis + 1}={float(values[axis]):.{precision}f}"
        for axis in range(JOINT_COUNT)
    )


def _print_stream_report(name: str, metrics: StreamMetrics) -> None:
    rate_text = _rate_text(metrics.observed_rate_hz)
    gap_text = (
        "--"
        if metrics.maximum_gap_s is None
        else f"{metrics.maximum_gap_s * 1_000.0:.2f} ms"
    )
    print(
        f"  {name}: samples={metrics.sample_count} rate={rate_text} Hz "
        f"max-gap={gap_text} gap-warning={metrics.gap_warning_count}"
    )
    print(f"    range(deg): {_axis_values_text(metrics.joint_range_deg)}")
    print(
        "    plateau(%): "
        + _axis_values_text(metrics.same_ratio * 100.0, precision=0)
        + " | reversals: "
        + " ".join(
            f"J{axis + 1}={int(metrics.reversal_count[axis])}"
            for axis in range(JOINT_COUNT)
        )
    )
    print(f"    max-step(deg): {_axis_values_text(metrics.maximum_step_deg)}")
    print(
        f"    max-velocity(deg/s): {_axis_values_text(metrics.maximum_velocity_deg_s)}"
    )


def _print_feedback_report(
    metrics: FeedbackMetrics, worker: FeedbackWorkerStats | None
) -> None:
    read_max = (
        "--"
        if metrics.maximum_read_duration_s is None
        else f"{metrics.maximum_read_duration_s * 1_000.0:.2f} ms"
    )
    print(
        "  #3 实际反馈: "
        f"valid={metrics.valid_count}/{metrics.attempt_count} rate={_rate_text(metrics.observed_rate_hz)} Hz "
        f"read-max={read_max} matched={metrics.matched_target_count}"
    )
    if worker is not None:
        print(
            "    feedback-worker: "
            f"miss={worker.missed_tick_count} late-max={worker.maximum_lateness_s * 1_000.0:.2f} ms"
        )
    if metrics.p95_absolute_error_deg is not None:
        command_age = (
            "--"
            if metrics.mean_command_age_s is None
            else f"{metrics.mean_command_age_s * 1_000.0:.1f} ms"
        )
        print(
            "    approximate p95 abs error(deg): "
            + _axis_values_text(metrics.p95_absolute_error_deg)
            + f" | mean submitted-command age={command_age}"
        )


def _print_side_report(report: SideDiagnosticReport) -> None:
    print(f"\n[{report.side.upper()}]")
    _print_stream_report("#1 控制器生成目标", report.generated)
    _print_stream_report("#2 成功 set_target", report.transmitted)
    if report.direct_servo_stats is not None:
        stats = report.direct_servo_stats
        call_max = (
            "--"
            if stats.max_set_target_duration_s is None
            else f"{stats.max_set_target_duration_s * 1_000.0:.2f} ms"
        )
        late_max = (
            "--"
            if stats.max_start_lateness_s is None
            else f"{stats.max_start_lateness_s * 1_000.0:.2f} ms"
        )
        print(
            "    adapter TX session: "
            f"sent={stats.successful_command_count} miss={stats.missed_tick_count} "
            f"call-max={call_max} late-max={late_max} running={stats.running}"
        )
    if report.feedback is not None:
        _print_feedback_report(report.feedback, report.feedback_worker)
    for conclusion in report.conclusions:
        print(f"  [CONCLUSION] {conclusion}")


def print_report(report: DiagnosticReport) -> None:
    """输出最终诊断报告，并明确反馈比较的时间边界。"""

    print("\n" + "=" * 72)
    print("    OpenArm Mini -> HCX 双臂三段链路诊断结果")
    print("=" * 72)
    print(
        f"  时长: {report.elapsed_s:.2f} s；主控周期: {report.control_loop.cycle_count}；"
        f"主控 miss: {report.control_loop.missed_tick_count}；"
        f"late-max: {report.control_loop.maximum_lateness_s * 1_000.0:.2f} ms"
    )
    print(
        "  #3 误差使用本机完成时间与最近成功 set_target 近似配对；"
        "它不是控制器内部时间同步或驱动器报警诊断。"
    )
    _print_side_report(report.left)
    _print_side_report(report.right)


def write_csv(path: str | Path, report: DiagnosticReport) -> None:
    """导出当前保留的三段原始流，便于离线画图与逐样本分析。"""

    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "stream",
                "side",
                "timestamp_s",
                "started_at_s",
                "duration_s",
                "status",
                "error",
                *(f"J{axis + 1}_deg" for axis in range(JOINT_COUNT)),
            )
        )
        for side in ARM_SIDES:
            for name, samples in (
                ("generated", report.traces.generated[side]),
                ("set_target_success", report.traces.transmitted[side]),
            ):
                for sample in samples:
                    writer.writerow(
                        (
                            name,
                            side,
                            f"{sample.timestamp_s:.9f}",
                            "",
                            "",
                            "ok",
                            "",
                            *(f"{value:.9f}" for value in sample.angles_deg),
                        )
                    )
            for sample in report.traces.feedback[side]:
                values = (
                    ("",) * JOINT_COUNT
                    if sample.angles_deg is None
                    else tuple(f"{value:.9f}" for value in sample.angles_deg)
                )
                writer.writerow(
                    (
                        "joint_feedback",
                        side,
                        f"{sample.timestamp_s:.9f}",
                        f"{sample.started_at_s:.9f}",
                        f"{sample.duration_s:.9f}",
                        "ok" if sample.angles_deg is not None else "error",
                        sample.error or "",
                        *values,
                    )
                )


def _print_start_message(
    direct_config: HcxDirectServoConfig, config: DiagnosticRunConfig
) -> None:
    print("=" * 72)
    print("    OpenArm Mini -> HCX 双臂三段链路诊断（无夹爪）")
    print("=" * 72)
    print(
        f"  #1 控制器目标: {config.control_rate_hz:.0f} Hz；"
        f"滤波={'启用' if FILTER_ENABLED else '关闭'}；"
        f"弹簧/前瞻={'启用' if SPRING_ENABLED else '关闭'}"
    )
    print(
        f"  #2 HCX 直伺服: {direct_config.rate_hz} Hz；"
        f"mode={direct_config.interpolation}；watchdog={direct_config.watchdog_s:.3f} s"
    )
    if config.feedback_enabled:
        print(
            f"  #3 实际反馈: 每侧 {config.feedback_rate_hz:.0f} Hz 独立线程；"
            "先将 FEEDBACK_ENABLED=False 跑一次基线，再开启它比较 TX miss。"
        )
    else:
        print("  #3 实际反馈: 已关闭，仅诊断 #1 控制器目标和 #2 set_target 输出。")
    print(
        "  建议先静止 2 秒，再仅慢速单向移动一根关节并保持 2 秒；"
        "不要把快速人为往复的反向数当作输入故障。"
    )
    print("  Ctrl+C 会停止 Python 直伺服软件下发；独立急停和硬件保护必须可用。")
    print("-" * 72)


def main() -> int:
    """使用文件顶部常量运行诊断；不加载 YAML，也不解析 CLI 参数。"""

    try:
        report = run_diagnostic(
            DIAGNOSTIC_HCX_CONFIG,
            DIAGNOSTIC_TELEOP_CONFIG,
            DIAGNOSTIC_RUN_CONFIG,
        )
    except KeyboardInterrupt:
        print("\n[STOP] 诊断已中断，已请求停止直伺服并断开设备。")
        return 130
    except Exception as exc:
        print(f"[ERROR] OpenArm -> HCX 诊断启动或运行失败: {exc}")
        return 1
    print_report(report)
    if DIAGNOSTIC_CSV_PATH:
        try:
            write_csv(DIAGNOSTIC_CSV_PATH, report)
        except OSError as exc:
            print(f"[WARN] 无法写入诊断 CSV: {exc}")
        else:
            print(f"[INFO] 已导出诊断 CSV: {DIAGNOSTIC_CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
