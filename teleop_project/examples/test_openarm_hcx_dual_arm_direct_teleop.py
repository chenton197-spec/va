#!/usr/bin/env python3
"""最小 OpenArm Mini -> HCX 双臂 limited 遥操作基线。

本示例是 ``test_openarm_hcx_single_arm_direct_teleop.py`` 的双臂版本。左右
OpenArm Mini 各自以 100 Hz 读取七轴角度；同一个 HCX ``RobotClient`` 中的两个
``Arm`` 各自以 500 Hz 调用 ``DirectServoSession.set_target()``。每侧发送线程
使用自身的 ``LimitedInterpolator``，执行低通、限速和限加速度。

除单臂基线已有的相对映射、LimitedInterpolator 和 30 Hz 实际角度观测外，不使用
``TeleopController``、主臂滤波、弹簧阻尼、图窗、自动恢复或 ``teleop.yaml``。

运行前必须手动完成示教器脱离、报警处理、EtherCAT 就绪、全局使能和两侧单臂
使能。本程序会产生真实运动，只有 ``CONFIRM_DIRECT_SERVO`` 为 ``True`` 时才
启动 HCX 直接伺服。

运行：

    python -m examples.test_openarm_hcx_dual_arm_direct_teleop
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Any, Literal

import numpy as np

from teleop_sdk.adapters import HcxConnection, HcxConnectionConfig, OpenArmMiniLeaderArm
from teleop_sdk.algorithms import LimitedInterpolator

JOINT_COUNT = 7
ArmSide = Literal["left", "right"]

# HCX 七轴软件安全范围，单位为度。它在控制器原始关节限位两端各向内收 1°，
# 让 direct-servo 目标不会贴着控制器边界发送。
HCX_SAFE_MIN_ANGLES_DEG = np.array(
    (-169.0, -109.0, -169.0, -139.0, -169.0, -54.0, -59.0),
    dtype=float,
)
HCX_SAFE_MAX_ANGLES_DEG = np.array(
    (169.0, 109.0, 169.0, 54.0, 169.0, 54.0, 59.0),
    dtype=float,
)

# 左右 OpenArm Mini 的只读串口与双侧组合标定文件。
LEFT_OPENARM_PORT = "/dev/ttyACM1"
RIGHT_OPENARM_PORT = "/dev/ttyACM0"
OPENARM_CALIBRATION_PATH = "./my_openarm_mini.json"
OPENARM_BAUDRATE = 1_000_000

# HCX 控制器连接参数。左右臂共享同一个 RobotClient。
HCX_LOCAL_IP = "172.16.0.110"
HCX_REMOTE_IP = "172.16.0.89"
HCX_PORT = 12345
HCX_CONNECT_TIMEOUT_S = 10.0

# 左右 HCX 机器人 ID。
LEFT_HCX_ROBOT_ID = 1
RIGHT_HCX_ROBOT_ID = 2

# 左右 OpenArm -> HCX 七轴方向。+1.0 为同向，-1.0 为反向。
LEFT_AXIS_SIGN = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
RIGHT_AXIS_SIGN = (-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0)

# 每侧主臂只读采样频率，单位为 Hz。
LEADER_SAMPLE_RATE_HZ = 100.0
# 每次 OpenArm 读取允许的最长等待时间，单位为秒。超时后保持上一次从臂目标。
LEADER_READ_TIMEOUT_S = 0.008

# 每侧 HCX 直接伺服目标发送频率，单位为 Hz。
DIRECT_SERVO_RATE_HZ = 500
# 直伺服软件侧看门狗，单位为秒；它允许主臂静止，但要求发送线程持续下发目标。
DIRECT_SERVO_WATCHDOG_S = 2.0
# 真实运动的显式确认。仅在独立急停和硬件防护已确认后设为 True。
CONFIRM_DIRECT_SERVO = True

# 每侧 HCX 实际关节角度读取频率，单位为 Hz；读取结果不参与伺服目标生成。
FEEDBACK_RATE_HZ = 30.0
# 终端展示最近一帧实际关节角度的频率，单位为 Hz；不要逐帧打印影响实时性。
FEEDBACK_REPORT_RATE_HZ = 1.0

# 每侧 LimitedInterpolator 的最大关节速度，单位为度/秒。
LIMITED_MAX_VELOCITY_DEG_S = 120.0
# 每侧 LimitedInterpolator 的最大关节加速度，单位为度/秒平方。
LIMITED_MAX_ACCELERATION_DEG_S2 = 80.0
# 每侧 LimitedInterpolator 的固定低通权重，范围 [0, 1]；越小低通越强。
LIMITED_LOWPASS_ALPHA = 0.2

# 0.0 表示持续运行直到 Ctrl+C；正数表示自动结束的测试时长，单位为秒。
TEST_DURATION_S = 0.0


@dataclass(frozen=True)
class ArmConfig:
    """一侧 OpenArm 与 HCX 的固定对应关系。"""

    side: ArmSide
    openarm_port: str
    hcx_robot_id: int
    axis_sign: tuple[float, ...]

    def validate(self) -> None:
        if self.side not in ("left", "right"):
            raise ValueError("side 必须是 left 或 right")
        if not isinstance(self.openarm_port, str) or not self.openarm_port.strip():
            raise ValueError(f"{self.side}.openarm_port 不能为空")
        if (
            not isinstance(self.hcx_robot_id, int)
            or isinstance(self.hcx_robot_id, bool)
            or self.hcx_robot_id < 0
        ):
            raise ValueError(f"{self.side}.hcx_robot_id 必须是非负整数")
        if len(self.axis_sign) != JOINT_COUNT or any(
            sign not in (-1.0, 1.0) for sign in self.axis_sign
        ):
            raise ValueError(
                f"{self.side}.axis_sign 必须是 {JOINT_COUNT} 个 +1.0 或 -1.0"
            )


@dataclass(frozen=True)
class DemoConfig:
    """双臂 limited 遥操作所需的最小配置，所有关节角度均为度。"""

    left: ArmConfig
    right: ArmConfig
    openarm_calibration_path: str
    openarm_baudrate: int
    hcx_local_ip: str
    hcx_remote_ip: str
    hcx_port: int
    hcx_connect_timeout_s: float
    leader_sample_rate_hz: float
    leader_read_timeout_s: float
    direct_servo_rate_hz: int
    direct_servo_watchdog_s: float
    confirm_direct_servo: bool
    feedback_rate_hz: float
    feedback_report_rate_hz: float
    limited_max_velocity_deg_s: float
    limited_max_acceleration_deg_s2: float
    limited_lowpass_alpha: float
    test_duration_s: float

    def validate(self) -> None:
        self.left.validate()
        self.right.validate()
        if (
            not isinstance(self.openarm_calibration_path, str)
            or not self.openarm_calibration_path.strip()
        ):
            raise ValueError("openarm_calibration_path 不能为空")
        if (
            not isinstance(self.openarm_baudrate, int)
            or isinstance(self.openarm_baudrate, bool)
            or self.openarm_baudrate <= 0
        ):
            raise ValueError("openarm_baudrate 必须是正整数")
        for name, value in (
            ("hcx_local_ip", self.hcx_local_ip),
            ("hcx_remote_ip", self.hcx_remote_ip),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} 不能为空")
        if (
            not isinstance(self.hcx_port, int)
            or isinstance(self.hcx_port, bool)
            or not 1 <= self.hcx_port <= 65535
        ):
            raise ValueError("hcx_port 必须是 1 到 65535 的整数")
        _positive_finite("hcx_connect_timeout_s", self.hcx_connect_timeout_s)
        _positive_finite("leader_sample_rate_hz", self.leader_sample_rate_hz)
        _positive_finite("leader_read_timeout_s", self.leader_read_timeout_s)
        _positive_finite("direct_servo_watchdog_s", self.direct_servo_watchdog_s)
        _positive_finite("feedback_rate_hz", self.feedback_rate_hz)
        _positive_finite("feedback_report_rate_hz", self.feedback_report_rate_hz)
        _nonnegative_finite("test_duration_s", self.test_duration_s)
        if (
            not isinstance(self.direct_servo_rate_hz, int)
            or isinstance(self.direct_servo_rate_hz, bool)
            or not 100 <= self.direct_servo_rate_hz <= 1000
        ):
            raise ValueError("direct_servo_rate_hz 必须是 100 到 1000 的整数")
        if self.direct_servo_watchdog_s <= 1.0 / self.direct_servo_rate_hz:
            raise ValueError("direct_servo_watchdog_s 必须大于一个伺服周期")
        if not isinstance(self.confirm_direct_servo, bool):
            raise ValueError("confirm_direct_servo 必须是布尔值")
        if not self.confirm_direct_servo:
            raise ValueError("请在文件顶部明确设置 CONFIRM_DIRECT_SERVO = True")
        _positive_finite("limited_max_velocity_deg_s", self.limited_max_velocity_deg_s)
        _positive_finite(
            "limited_max_acceleration_deg_s2",
            self.limited_max_acceleration_deg_s2,
        )
        _unit_interval("limited_lowpass_alpha", self.limited_lowpass_alpha)


DEMO_CONFIG = DemoConfig(
    left=ArmConfig(
        side="left",
        openarm_port=LEFT_OPENARM_PORT,
        hcx_robot_id=LEFT_HCX_ROBOT_ID,
        axis_sign=LEFT_AXIS_SIGN,
    ),
    right=ArmConfig(
        side="right",
        openarm_port=RIGHT_OPENARM_PORT,
        hcx_robot_id=RIGHT_HCX_ROBOT_ID,
        axis_sign=RIGHT_AXIS_SIGN,
    ),
    openarm_calibration_path=OPENARM_CALIBRATION_PATH,
    openarm_baudrate=OPENARM_BAUDRATE,
    hcx_local_ip=HCX_LOCAL_IP,
    hcx_remote_ip=HCX_REMOTE_IP,
    hcx_port=HCX_PORT,
    hcx_connect_timeout_s=HCX_CONNECT_TIMEOUT_S,
    leader_sample_rate_hz=LEADER_SAMPLE_RATE_HZ,
    leader_read_timeout_s=LEADER_READ_TIMEOUT_S,
    direct_servo_rate_hz=DIRECT_SERVO_RATE_HZ,
    direct_servo_watchdog_s=DIRECT_SERVO_WATCHDOG_S,
    confirm_direct_servo=CONFIRM_DIRECT_SERVO,
    feedback_rate_hz=FEEDBACK_RATE_HZ,
    feedback_report_rate_hz=FEEDBACK_REPORT_RATE_HZ,
    limited_max_velocity_deg_s=LIMITED_MAX_VELOCITY_DEG_S,
    limited_max_acceleration_deg_s2=LIMITED_MAX_ACCELERATION_DEG_S2,
    limited_lowpass_alpha=LIMITED_LOWPASS_ALPHA,
    test_duration_s=TEST_DURATION_S,
)


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


def _unit_interval(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须在 0 到 1 之间")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须在 0 到 1 之间") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} 必须在 0 到 1 之间")
    return numeric


def _as_joint_vector(name: str, values: object) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (JOINT_COUNT,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} 必须是 {JOINT_COUNT} 个有限关节角度")
    return vector


def map_relative_target(
    leader_angles_deg: np.ndarray,
    leader_origin_deg: np.ndarray,
    follower_origin_deg: np.ndarray,
    axis_sign: tuple[float, ...],
    min_angles_deg: np.ndarray,
    max_angles_deg: np.ndarray,
) -> np.ndarray:
    """将当前主臂相对零点的七轴变化映射为从臂度制目标。"""

    leader = _as_joint_vector("leader_angles_deg", leader_angles_deg)
    leader_origin = _as_joint_vector("leader_origin_deg", leader_origin_deg)
    follower_origin = _as_joint_vector("follower_origin_deg", follower_origin_deg)
    lower = _as_joint_vector("min_angles_deg", min_angles_deg)
    upper = _as_joint_vector("max_angles_deg", max_angles_deg)
    if np.any(lower >= upper):
        raise ValueError("每个从臂关节必须满足 min_angles_deg < max_angles_deg")
    if len(axis_sign) != JOINT_COUNT or any(
        sign not in (-1.0, 1.0) for sign in axis_sign
    ):
        raise ValueError(f"axis_sign 必须是 {JOINT_COUNT} 个 +1.0 或 -1.0")
    target = follower_origin + np.asarray(axis_sign, dtype=float) * (
        leader - leader_origin
    )
    return np.clip(target, lower, upper)


class LatestTarget:
    """单写入者/单读取者的最近主臂目标槽，不积压旧样本。"""

    def __init__(self, initial_angles_deg: np.ndarray) -> None:
        self._target = _as_joint_vector("initial_angles_deg", initial_angles_deg).copy()

    def publish(self, angles_deg: np.ndarray) -> None:
        self._target = _as_joint_vector("angles_deg", angles_deg).copy()

    def snapshot(self) -> np.ndarray:
        return self._target


@dataclass(frozen=True)
class FeedbackSnapshot:
    """一帧 HCX 实际关节角度及其本机读取信息。"""

    angles_deg: np.ndarray | None
    received_at_s: float | None
    read_duration_s: float | None
    sample_count: int
    last_error: str | None


class LatestFeedback:
    """单写入者/单读取者的最新 HCX 反馈槽，不积压旧反馈。"""

    def __init__(self) -> None:
        self._snapshot = FeedbackSnapshot(
            angles_deg=None,
            received_at_s=None,
            read_duration_s=None,
            sample_count=0,
            last_error=None,
        )

    def publish(
        self,
        angles_deg: np.ndarray,
        received_at_s: float,
        read_duration_s: float,
    ) -> None:
        received_at = _nonnegative_finite("received_at_s", received_at_s)
        read_duration = _nonnegative_finite("read_duration_s", read_duration_s)
        previous = self._snapshot
        self._snapshot = FeedbackSnapshot(
            angles_deg=_as_joint_vector("HCX 实际关节角度", angles_deg).copy(),
            received_at_s=received_at,
            read_duration_s=read_duration,
            sample_count=previous.sample_count + 1,
            last_error=None,
        )

    def record_error(self, error: BaseException, received_at_s: float) -> None:
        _nonnegative_finite("received_at_s", received_at_s)
        previous = self._snapshot
        self._snapshot = FeedbackSnapshot(
            angles_deg=previous.angles_deg,
            received_at_s=previous.received_at_s,
            read_duration_s=previous.read_duration_s,
            sample_count=previous.sample_count,
            last_error=f"{type(error).__name__}: {error}",
        )

    def snapshot(self) -> FeedbackSnapshot:
        current = self._snapshot
        return FeedbackSnapshot(
            angles_deg=(
                None if current.angles_deg is None else current.angles_deg.copy()
            ),
            received_at_s=current.received_at_s,
            read_duration_s=current.read_duration_s,
            sample_count=current.sample_count,
            last_error=current.last_error,
        )


@dataclass
class _ArmRuntime:
    """一个侧别的对象和线程状态。"""

    label: str
    config: ArmConfig
    leader: Any
    arm: Any | None = None
    session: Any | None = None
    follower_origin_deg: np.ndarray | None = None
    min_angles_deg: np.ndarray | None = None
    max_angles_deg: np.ndarray | None = None
    latest_target: LatestTarget | None = None
    latest_feedback: LatestFeedback | None = None
    sender_thread: threading.Thread | None = None
    feedback_thread: threading.Thread | None = None
    sampler_thread: threading.Thread | None = None


def _read_follower_pose_and_limits(
    arm: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取当前姿态，并将控制器限位收紧到本 demo 的固定安全范围。"""

    follower_origin = _as_joint_vector("HCX 当前关节角度", arm.joint_angles())
    limits = np.asarray(arm.joint_limits_deg, dtype=float)
    if limits.shape != (JOINT_COUNT, 2) or not np.isfinite(limits).all():
        raise RuntimeError("HCX 返回的七轴关节限位无效")
    lower = np.maximum(limits[:, 0], HCX_SAFE_MIN_ANGLES_DEG)
    upper = np.minimum(limits[:, 1], HCX_SAFE_MAX_ANGLES_DEG)
    if np.any(lower >= upper):
        raise RuntimeError("HCX 控制器限位与 demo 固定安全范围没有可用交集")
    if np.any(follower_origin < lower) or np.any(follower_origin > upper):
        raise RuntimeError(
            "HCX 当前关节角度已在 demo 固定安全范围外；"
            "请先手动移回安全范围后再启动直伺服"
        )
    return follower_origin, lower, upper


def _record_worker_failure(
    failures: SimpleQueue[BaseException],
    stop_event: threading.Event,
    label: str,
    exc: BaseException,
) -> None:
    failures.put(RuntimeError(f"{label} 工作线程失败: {exc}"))
    stop_event.set()


def _run_direct_servo_sender(
    label: str,
    session: Any,
    latest_target: LatestTarget,
    rate_hz: int,
    max_velocity_deg_s: float,
    max_acceleration_deg_s2: float,
    lowpass_alpha: float,
    min_angles_deg: np.ndarray,
    max_angles_deg: np.ndarray,
    stop_event: threading.Event,
    failures: SimpleQueue[BaseException],
) -> None:
    """一侧独立高频发送线程；每周期生成一个 limited 关节目标并下发。"""

    period_s = 1.0 / rate_hz
    interpolator = LimitedInterpolator(
        rate_hz,
        rate_hz,
        max_velocity_deg_s=max_velocity_deg_s,
        max_acceleration_deg_s2=max_acceleration_deg_s2,
        lowpass_alpha=lowpass_alpha,
        min_angles_deg=min_angles_deg,
        max_angles_deg=max_angles_deg,
    )
    interpolator.reset(latest_target.snapshot())
    next_tick_s = time.monotonic() + period_s
    last_generation_s = time.monotonic()
    try:
        while not stop_event.is_set():
            remaining_s = next_tick_s - time.monotonic()
            if remaining_s > 0.0 and stop_event.wait(remaining_s):
                return

            dispatch_started_s = time.monotonic()
            elapsed_s = min(
                max(dispatch_started_s - last_generation_s, 0.0),
                2.0 * period_s,
            )
            point = interpolator.step(latest_target.snapshot(), elapsed_s=elapsed_s)
            last_generation_s = dispatch_started_s
            session.set_target(point.tolist())

            next_tick_s += period_s
            loop_finished_s = time.monotonic()
            if loop_finished_s >= next_tick_s:
                next_tick_s = loop_finished_s + period_s
    except (RuntimeError, TypeError, ValueError) as exc:
        _record_worker_failure(failures, stop_event, label, exc)


def _run_feedback_reader(
    arm: Any,
    latest_feedback: LatestFeedback,
    rate_hz: float,
    stop_event: threading.Event,
) -> None:
    """独立低频线程读取一侧 HCX 实际角度，不参与目标生成。"""

    period_s = 1.0 / rate_hz
    next_tick_s = time.monotonic()
    while not stop_event.is_set():
        remaining_s = next_tick_s - time.monotonic()
        if remaining_s > 0.0 and stop_event.wait(remaining_s):
            return

        started_at_s = time.monotonic()
        try:
            angles_deg = _as_joint_vector("HCX 实际关节角度", arm.joint_angles())
        except Exception as exc:
            latest_feedback.record_error(exc, time.monotonic())
        else:
            completed_at_s = time.monotonic()
            latest_feedback.publish(
                angles_deg,
                completed_at_s,
                completed_at_s - started_at_s,
            )

        next_tick_s += period_s
        completed_at_s = time.monotonic()
        if completed_at_s >= next_tick_s:
            next_tick_s = completed_at_s + period_s


def _run_leader_sampler(
    runtime: _ArmRuntime,
    leader_sample_rate_hz: float,
    leader_read_timeout_s: float,
    stop_event: threading.Event,
    failures: SimpleQueue[BaseException],
) -> None:
    """独立 100 Hz 读取一侧主臂，并发布最新的相对从臂目标。"""

    assert runtime.follower_origin_deg is not None
    assert runtime.min_angles_deg is not None
    assert runtime.max_angles_deg is not None
    assert runtime.latest_target is not None

    period_s = 1.0 / leader_sample_rate_hz
    next_tick_s = time.monotonic()
    leader_origin_deg: np.ndarray | None = None
    try:
        while not stop_event.is_set():
            remaining_s = next_tick_s - time.monotonic()
            if remaining_s > 0.0 and stop_event.wait(remaining_s):
                return

            leader_angles = runtime.leader.read_joint_angles_deg(leader_read_timeout_s)
            if leader_angles is not None:
                current = _as_joint_vector("OpenArm 当前关节角度", leader_angles)
                if leader_origin_deg is None:
                    leader_origin_deg = current.copy()
                    print(
                        f"[INFO] {runtime.label} 主臂起始位置已记录: "
                        f"{np.round(leader_origin_deg, 1).tolist()}"
                    )
                    print(
                        f"[INFO] 现在移动 {runtime.label} 主臂，"
                        f"{runtime.label} 从臂将跟随相对变化量"
                    )
                runtime.latest_target.publish(
                    map_relative_target(
                        current,
                        leader_origin_deg,
                        runtime.follower_origin_deg,
                        runtime.config.axis_sign,
                        runtime.min_angles_deg,
                        runtime.max_angles_deg,
                    )
                )

            next_tick_s += period_s
            completed_at_s = time.monotonic()
            if completed_at_s >= next_tick_s:
                next_tick_s = completed_at_s
    except Exception as exc:
        _record_worker_failure(failures, stop_event, runtime.label, exc)


def _print_feedback_snapshot(label: str, snapshot: FeedbackSnapshot, now_s: float) -> None:
    """低频输出一侧最近实际角度，避免反馈线程逐帧打印扰动伺服发送。"""

    if snapshot.angles_deg is None or snapshot.received_at_s is None:
        message = f"[HCX RX] {label} 尚未收到有效实际关节反馈"
    else:
        age_ms = max(0.0, now_s - snapshot.received_at_s) * 1_000.0
        read_ms = (
            "--"
            if snapshot.read_duration_s is None
            else f"{snapshot.read_duration_s * 1_000.0:.2f}"
        )
        message = (
            f"[HCX RX] {label} samples={snapshot.sample_count} "
            f"age={age_ms:.1f} ms read={read_ms} ms "
            f"angles(deg)={np.round(snapshot.angles_deg, 2).tolist()}"
        )
    if snapshot.last_error is not None:
        message += f" | last-error={snapshot.last_error}"
    print(message)


def _raise_worker_failure(failures: SimpleQueue[BaseException]) -> None:
    try:
        failure = failures.get_nowait()
    except Empty:
        return
    raise RuntimeError(f"双臂直伺服运行失败: {failure}") from failure


def _start_runtime_workers(
    runtime: _ArmRuntime,
    config: DemoConfig,
    stop_event: threading.Event,
    failures: SimpleQueue[BaseException],
) -> None:
    """启动一侧的 500 Hz 发送、30 Hz 反馈和 100 Hz 主臂采样线程。"""

    assert runtime.session is not None
    assert runtime.latest_target is not None
    assert runtime.latest_feedback is not None
    assert runtime.min_angles_deg is not None
    assert runtime.max_angles_deg is not None
    assert runtime.arm is not None

    runtime.sender_thread = threading.Thread(
        target=_run_direct_servo_sender,
        args=(
            runtime.label,
            runtime.session,
            runtime.latest_target,
            config.direct_servo_rate_hz,
            config.limited_max_velocity_deg_s,
            config.limited_max_acceleration_deg_s2,
            config.limited_lowpass_alpha,
            runtime.min_angles_deg,
            runtime.max_angles_deg,
            stop_event,
            failures,
        ),
        name=f"openarm-hcx-{runtime.config.side}-direct-servo",
        daemon=True,
    )
    runtime.sender_thread.start()
    runtime.feedback_thread = threading.Thread(
        target=_run_feedback_reader,
        args=(
            runtime.arm,
            runtime.latest_feedback,
            config.feedback_rate_hz,
            stop_event,
        ),
        name=f"hcx-{runtime.config.side}-joint-feedback",
        daemon=True,
    )
    runtime.feedback_thread.start()
    runtime.sampler_thread = threading.Thread(
        target=_run_leader_sampler,
        args=(
            runtime,
            config.leader_sample_rate_hz,
            config.leader_read_timeout_s,
            stop_event,
            failures,
        ),
        name=f"openarm-{runtime.config.side}-joint-sampler",
        daemon=True,
    )
    runtime.sampler_thread.start()


def _stop_runtime_workers(runtime: _ArmRuntime, config: DemoConfig) -> None:
    """等待一侧的工作线程退出。"""

    thread_timeouts = (
        (runtime.sampler_thread, max(1.0, 2.0 / config.leader_sample_rate_hz)),
        (runtime.feedback_thread, max(1.0, 2.0 / config.feedback_rate_hz)),
        (runtime.sender_thread, max(1.0, 2.0 / config.direct_servo_rate_hz)),
    )
    for thread, timeout_s in thread_timeouts:
        if thread is None:
            continue
        thread.join(timeout=timeout_s)
        if thread.is_alive():
            print(f"[WARN] {runtime.label} {thread.name} 未在预期时间内退出")


def run_demo(config: DemoConfig) -> None:
    """运行左右独立 100 Hz 主臂采样与 500 Hz limited 直伺服下发。"""

    config.validate()
    runtimes = [
        _ArmRuntime(
            label="LEFT",
            config=config.left,
            leader=OpenArmMiniLeaderArm(
                config.left.openarm_port,
                Path(config.openarm_calibration_path),
                config.left.side,
                baudrate=config.openarm_baudrate,
                read_only=True,
            ),
        ),
        _ArmRuntime(
            label="RIGHT",
            config=config.right,
            leader=OpenArmMiniLeaderArm(
                config.right.openarm_port,
                Path(config.openarm_calibration_path),
                config.right.side,
                baudrate=config.openarm_baudrate,
                read_only=True,
            ),
        ),
    ]
    connection = HcxConnection(
        HcxConnectionConfig(
            local_ip=config.hcx_local_ip,
            remote_ip=config.hcx_remote_ip,
            port=config.hcx_port,
            connect_timeout_s=config.hcx_connect_timeout_s,
        )
    )
    stop_event = threading.Event()
    failures: SimpleQueue[BaseException] = SimpleQueue()

    try:
        for runtime in runtimes:
            runtime.leader.connect()
        for runtime in runtimes:
            runtime.arm = connection.acquire(runtime.config.hcx_robot_id)
        for runtime in runtimes:
            connection.prepare_for_motion(runtime.config.hcx_robot_id)
        for runtime in runtimes:
            assert runtime.arm is not None
            (
                runtime.follower_origin_deg,
                runtime.min_angles_deg,
                runtime.max_angles_deg,
            ) = _read_follower_pose_and_limits(runtime.arm)
            runtime.latest_target = LatestTarget(runtime.follower_origin_deg)
            runtime.latest_feedback = LatestFeedback()
        for runtime in runtimes:
            assert runtime.arm is not None
            assert runtime.follower_origin_deg is not None
            runtime.session = runtime.arm.start_direct_servo(
                rate_hz=config.direct_servo_rate_hz,
                watchdog_s=config.direct_servo_watchdog_s,
                confirm_unsafe=config.confirm_direct_servo,
            )
            runtime.session.set_target(runtime.follower_origin_deg.tolist())
        for runtime in runtimes:
            _start_runtime_workers(runtime, config, stop_event, failures)

        print("=" * 72)
        print("    OpenArm Mini -> HCX 双臂最小 limited 遥操作")
        print("=" * 72)
        for runtime in runtimes:
            print(
                f"  {runtime.label}: OpenArm={runtime.config.openarm_port}；"
                f"HCX robot_id={runtime.config.hcx_robot_id}；"
                f"axis_sign={list(runtime.config.axis_sign)}"
            )
        print(f"  每侧主臂采样: {config.leader_sample_rate_hz:.0f} Hz")
        print(
            f"  每侧 HCX set_target: {config.direct_servo_rate_hz} Hz"
            "（最新目标 limited 输出）"
        )
        print(
            f"  每侧 HCX 实际角度反馈: {config.feedback_rate_hz:g} Hz"
            f"（终端显示 {config.feedback_report_rate_hz:g} Hz）"
        )
        print(
            "  映射: 每侧相对同序七轴 + 100 Hz 目标 -> 500 Hz "
            "低通/限速/限加速度；HCX 实际角度仅用于观测，不参与控制。"
        )
        print(
            "  limited: "
            f"max_vel={config.limited_max_velocity_deg_s:g} deg/s, "
            f"max_accel={config.limited_max_acceleration_deg_s2:g} deg/s^2, "
            f"alpha={config.limited_lowpass_alpha:g}"
        )
        print("  已先保持两侧从臂当前姿态；各侧收到第一帧主臂角度后开始相对跟随。")
        print("  Ctrl+C 停止软件侧目标下发；独立急停和硬件保护必须可用。")
        print("-" * 72)

        feedback_report_period_s = 1.0 / config.feedback_report_rate_hz
        started_at_s = time.monotonic()
        next_feedback_report_s = started_at_s
        while not stop_event.is_set():
            now_s = time.monotonic()
            if (
                config.test_duration_s > 0.0
                and now_s - started_at_s >= config.test_duration_s
            ):
                break
            _raise_worker_failure(failures)

            if now_s >= next_feedback_report_s:
                for runtime in runtimes:
                    assert runtime.latest_feedback is not None
                    _print_feedback_snapshot(
                        runtime.label,
                        runtime.latest_feedback.snapshot(),
                        now_s,
                    )
                next_feedback_report_s = now_s + feedback_report_period_s
            stop_event.wait(0.01)

        _raise_worker_failure(failures)
    finally:
        stop_event.set()
        for runtime in runtimes:
            _stop_runtime_workers(runtime, config)
        for runtime in reversed(runtimes):
            if runtime.session is not None:
                try:
                    runtime.session.stop()
                except (RuntimeError, TypeError, ValueError) as exc:
                    print(f"[WARN] 停止 {runtime.label} HCX 直伺服会话失败: {exc}")
        for runtime in reversed(runtimes):
            if runtime.arm is not None:
                try:
                    connection.release(runtime.config.hcx_robot_id)
                except (RuntimeError, TypeError, ValueError) as exc:
                    print(f"[WARN] 释放 {runtime.label} HCX 连接失败: {exc}")
        for runtime in reversed(runtimes):
            runtime.leader.disconnect()


def main() -> int:
    """执行文件顶部定义的双臂硬件基线；不加载 YAML，也不解析 CLI。"""

    try:
        run_demo(DEMO_CONFIG)
    except KeyboardInterrupt:
        print("\n[STOP] 收到退出请求，正在停止 HCX 双臂直伺服下发。")
        return 130
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f"[ERROR] 双臂最小 limited 遥操作失败: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
