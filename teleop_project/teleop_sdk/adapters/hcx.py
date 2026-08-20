"""HCX 双七轴机械臂的规划运动和直接伺服从臂适配器。"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..algorithms import LimitedInterpolator, LinearInterpolator
from ..config import HcxConfig
from ..interfaces import FollowerArm

HcxArmSide = Literal["left", "right"]
DirectServoTargetObserver = Callable[[np.ndarray, float], None]

_AXIS_COUNT = 7
_LIMIT_TOLERANCE_DEG = 0.01
# 与控制器物理软限位保留 1 度裕量，避免直伺服重复发送临界值。
_JOINT_LIMITS_DEG: tuple[tuple[float, ...], tuple[float, ...]] = (
    (-169.0, -100.0, -169.0, -139.0, -169.0, -54.0, -59.0),
    (169.0, 100.0, 169.0, 54.0, 169.0, 54.0, 59.0),
)


@dataclass(frozen=True)
class _DirectServoTrajectoryBatch:
    """由低频输入侧预生成的一段高频关节目标。

    ``points`` 在发布后不再修改。每个 ``HcxFollower`` 只有一个 100 Hz
    生产者和一个高频消费者；在 CPython 中 ``deque.append`` 与 ``popleft``
    可安全地在这两个线程间交接完整批次，无需让发送线程取得从臂状态锁。
    """

    points_deg: np.ndarray


@dataclass(frozen=True)
class HcxDirectServoOutputStats:
    """Python 直伺服输出线程最近一秒的下发统计。

    这些数据的边界是 ``DirectServoSession.set_target()`` 成功返回，也就是
    Python 已把目标交给薄原生绑定；它们不表示控制器接收时刻或机械臂反馈。
    ``missed_tick_count`` 只统计输出线程主动丢弃的计划发送时隙。当前发送
    策略不会主动丢弃时隙，因此该值应为 ``0``；若某次调用迟到，下一轮会立即
    补发到期时隙，迟到程度由 ``max_start_lateness_s`` 表示。
    """

    configured_rate_hz: int
    observed_rate_hz: float | None
    observation_window_s: float
    successful_command_count: int
    recent_successful_command_count: int
    missed_tick_count: int
    recent_missed_tick_count: int
    mean_set_target_duration_s: float | None
    max_set_target_duration_s: float | None
    max_start_lateness_s: float | None
    running: bool


class _DirectServoOutputTelemetry:
    """输出线程专用计数器；统计锁不参与 HCX 厂商调用或目标状态锁。"""

    _WINDOW_S = 1.0

    def __init__(self, configured_rate_hz: int) -> None:
        self._configured_rate_hz = configured_rate_hz
        self._lock = threading.Lock()
        self._successful_command_count = 0
        self._missed_tick_count = 0
        # (完成时刻, set_target 耗时, 本次主动丢弃时隙数, 本次起始迟到时间)
        self._recent_samples: deque[tuple[float, float, int, float]] = deque()

    def record_success(
        self,
        *,
        completed_at_s: float,
        set_target_duration_s: float,
        missed_ticks: int,
        start_lateness_s: float,
    ) -> None:
        """记录一次已成功完成的 Python 到原生层提交。"""

        with self._lock:
            self._successful_command_count += 1
            self._missed_tick_count += missed_ticks
            self._recent_samples.append(
                (
                    completed_at_s,
                    set_target_duration_s,
                    missed_ticks,
                    start_lateness_s,
                )
            )
            self._discard_expired_locked(completed_at_s)

    def snapshot(self, *, running: bool) -> HcxDirectServoOutputStats:
        """返回最近一秒窗口的只读统计快照。"""

        now_s = time.monotonic()
        with self._lock:
            self._discard_expired_locked(now_s)
            samples = tuple(self._recent_samples)
            successful_command_count = self._successful_command_count
            missed_tick_count = self._missed_tick_count

        if len(samples) >= 2:
            elapsed_s = samples[-1][0] - samples[0][0]
            observed_rate_hz = (
                (len(samples) - 1) / elapsed_s if elapsed_s > 0.0 else None
            )
        else:
            observed_rate_hz = None
        if samples:
            durations_s = tuple(sample[1] for sample in samples)
            mean_duration_s: float | None = sum(durations_s) / len(durations_s)
            max_duration_s: float | None = max(durations_s)
            max_start_lateness_s: float | None = max(
                sample[3] for sample in samples
            )
        else:
            mean_duration_s = None
            max_duration_s = None
            max_start_lateness_s = None

        return HcxDirectServoOutputStats(
            configured_rate_hz=self._configured_rate_hz,
            observed_rate_hz=observed_rate_hz,
            observation_window_s=self._WINDOW_S,
            successful_command_count=successful_command_count,
            recent_successful_command_count=len(samples),
            missed_tick_count=missed_tick_count,
            recent_missed_tick_count=sum(sample[2] for sample in samples),
            mean_set_target_duration_s=mean_duration_s,
            max_set_target_duration_s=max_duration_s,
            max_start_lateness_s=max_start_lateness_s,
            running=running,
        )

    def _discard_expired_locked(self, reference_s: float) -> None:
        cutoff_s = reference_s - self._WINDOW_S
        while self._recent_samples and self._recent_samples[0][0] < cutoff_s:
            self._recent_samples.popleft()


@dataclass(frozen=True)
class HcxStartupConfig:
    """HCX 双臂启动前置流程的显式安全授权和时序参数。

    ``robot_ids`` 为空时保留旧版适配器的只检查行为；需要控制器级自动
    操作时，必须提供一对左右机器人 ID，并在对应开关中明确授权。
    """

    robot_ids: tuple[int, ...] = ()
    auto_detach_hmi: bool = False
    auto_clear_alarms: bool = False
    auto_enable: bool = False
    controller_initialization_wait_s: float = 2.0
    ethercat_master_indices: tuple[int, ...] = ()
    ethercat_op_timeout_s: float = 15.0
    alarm_clear_retry_count: int = 5
    alarm_clear_retry_interval_s: float = 1.0
    global_enable_retry_count: int = 5
    global_enable_retry_interval_s: float = 1.0
    single_arm_enable_timeout_s: float = 5.0
    enable_status_poll_interval_s: float = 0.1


@dataclass(frozen=True)
class HcxConnectionConfig:
    """HCX 控制器连接参数。"""

    local_ip: str
    remote_ip: str
    port: int
    connect_timeout_s: float | None = 10.0
    startup: HcxStartupConfig = HcxStartupConfig()

    @classmethod
    def from_runtime_config(cls, config: HcxConfig) -> "HcxConnectionConfig":
        """从根目录 ``teleop.yaml`` 对应的 HCX 配置构造连接参数。"""

        return cls(
            local_ip=config.local_ip,
            remote_ip=config.remote_ip,
            port=config.port,
            connect_timeout_s=config.connect_timeout_s,
            startup=HcxStartupConfig(
                robot_ids=(config.left_robot_id, config.right_robot_id),
                auto_detach_hmi=config.auto_detach_hmi,
                auto_clear_alarms=config.auto_clear_alarms,
                auto_enable=config.auto_enable,
                controller_initialization_wait_s=(
                    config.controller_initialization_wait_s
                ),
                ethercat_master_indices=config.ethercat_master_indices,
                ethercat_op_timeout_s=config.ethercat_op_timeout_s,
                alarm_clear_retry_count=config.alarm_clear_retry_count,
                alarm_clear_retry_interval_s=config.alarm_clear_retry_interval_s,
                global_enable_retry_count=config.global_enable_retry_count,
                global_enable_retry_interval_s=config.global_enable_retry_interval_s,
                single_arm_enable_timeout_s=config.single_arm_enable_timeout_s,
                enable_status_poll_interval_s=config.enable_status_poll_interval_s,
            ),
        )


@dataclass(frozen=True)
class HcxMoveJointsConfig:
    """每帧非阻塞 ``move_joints`` 命令的厂商规划参数。"""

    acceleration_seconds: float | None = None
    deceleration_seconds: float | None = None
    speed_ratio: float | None = None
    smooth: int = 1


@dataclass(frozen=True)
class HcxDirectServoConfig:
    """HCX ``PluseToServo`` 的 Python 侧直接关节伺服配置。

    原生层只执行一次 ``PluseToServo`` 调用。``linear`` 和 ``limited``
    在低频上游目标更新时预生成一整批高频点；Python 输出线程只消费点并
    调用 ``set_target``，不会执行插值、低通、限速或限加速度计算。
    """

    rate_hz: int = 125
    watchdog_s: float = 0.2
    confirm_unsafe: bool = False
    interpolation: Literal["direct", "linear", "limited"] = "direct"
    # linear / limited 的上游目标频率。它由遥操作入口从 teleop.rate_hz
    # 自动传入，不是独立 YAML 配置项。
    source_rate_hz: int | None = None
    limited_max_velocity_deg_s: float | None = None
    limited_max_acceleration_deg_s2: float | None = None
    limited_lowpass_alpha: float | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.rate_hz, int)
            or isinstance(self.rate_hz, bool)
            or not 100 <= self.rate_hz <= 1000
        ):
            raise ValueError(
                "HCX direct-servo rate_hz must be an integer from 100 through 1000"
            )
        if isinstance(self.watchdog_s, bool):
            raise ValueError(
                "HCX direct-servo watchdog_s must be a finite value in (0, 60]"
            )
        try:
            watchdog_s = float(self.watchdog_s)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "HCX direct-servo watchdog_s must be a finite value in (0, 60]"
            ) from exc
        if not math.isfinite(watchdog_s) or not 0.0 < watchdog_s <= 60.0:
            raise ValueError(
                "HCX direct-servo watchdog_s must be a finite value in (0, 60]"
            )
        if not isinstance(self.confirm_unsafe, bool):
            raise ValueError("HCX direct-servo confirm_unsafe must be a boolean")
        if self.interpolation not in ("direct", "linear", "limited"):
            raise ValueError(
                "HCX direct-servo interpolation must be 'direct', 'linear', or 'limited'"
            )
        if self.interpolation in ("linear", "limited"):
            if (
                not isinstance(self.source_rate_hz, int)
                or isinstance(self.source_rate_hz, bool)
                or self.source_rate_hz <= 0
            ):
                raise ValueError(
                    "HCX interpolated direct-servo source_rate_hz must be a positive integer"
                )
            if self.source_rate_hz >= self.rate_hz:
                raise ValueError(
                    "HCX interpolated direct-servo rate_hz must exceed source_rate_hz"
                )
            if self.rate_hz % self.source_rate_hz != 0:
                raise ValueError(
                    "HCX interpolated direct-servo rate_hz must be an integer multiple of source_rate_hz"
                )
        elif self.source_rate_hz is not None:
            raise ValueError(
                "HCX only linear or limited direct-servo accepts source_rate_hz"
            )
        limited_fields = (
            self.limited_max_velocity_deg_s,
            self.limited_max_acceleration_deg_s2,
            self.limited_lowpass_alpha,
        )
        if self.interpolation == "limited":
            max_velocity = self._validate_positive_finite(
                "limited_max_velocity_deg_s", self.limited_max_velocity_deg_s
            )
            max_acceleration = self._validate_positive_finite(
                "limited_max_acceleration_deg_s2",
                self.limited_max_acceleration_deg_s2,
            )
            lowpass_alpha = self._validate_lowpass_alpha(self.limited_lowpass_alpha)
            object.__setattr__(self, "limited_max_velocity_deg_s", max_velocity)
            object.__setattr__(
                self, "limited_max_acceleration_deg_s2", max_acceleration
            )
            object.__setattr__(self, "limited_lowpass_alpha", lowpass_alpha)
        elif any(value is not None for value in limited_fields):
            raise ValueError(
                "HCX limited direct-servo parameters require interpolation='limited'"
            )
        object.__setattr__(self, "watchdog_s", watchdog_s)

    @staticmethod
    def _validate_positive_finite(name: str, value: object) -> float:
        if isinstance(value, bool):
            raise ValueError(f"HCX {name} must be a positive finite value")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"HCX {name} must be a positive finite value") from exc
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"HCX {name} must be a positive finite value")
        return numeric

    @staticmethod
    def _validate_lowpass_alpha(value: object) -> float:
        if isinstance(value, bool):
            raise ValueError(
                "HCX limited_lowpass_alpha must be a finite value from 0 through 1"
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "HCX limited_lowpass_alpha must be a finite value from 0 through 1"
            ) from exc
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(
                "HCX limited_lowpass_alpha must be a finite value from 0 through 1"
            )
        return numeric

    @classmethod
    def from_runtime_config(
        cls, config: HcxConfig, *, source_rate_hz: int | None = None
    ) -> "HcxDirectServoConfig":
        """从根目录 ``teleop.yaml`` 的 HCX 配置构造直接伺服参数。"""

        if not isinstance(config, HcxConfig):
            raise TypeError("config must be an HcxConfig")
        limited_kwargs: dict[str, float] = {}
        if config.direct_servo_interpolation == "limited":
            limited_kwargs = {
                "limited_max_velocity_deg_s": config.direct_servo_limited_max_vel_deg_s,
                "limited_max_acceleration_deg_s2": (
                    config.direct_servo_limited_max_accel_deg_s2
                ),
                "limited_lowpass_alpha": config.direct_servo_limited_lowpass_alpha,
            }
        return cls(
            rate_hz=config.direct_servo_rate_hz,
            watchdog_s=config.direct_servo_watchdog_s,
            confirm_unsafe=config.direct_servo_confirm_unsafe,
            interpolation=config.direct_servo_interpolation,
            source_rate_hz=source_rate_hz,
            **limited_kwargs,
        )


class HcxConnection:
    """让多个 HCX 从臂共享连接并协调双臂的安全启动流程。"""

    def __init__(self, config: HcxConnectionConfig):
        if not isinstance(config, HcxConnectionConfig):
            raise TypeError("config must be an HcxConnectionConfig")
        self._validate_startup_config(config.startup)
        self.config = config
        self._lock = threading.RLock()
        self._client: Any | None = None
        self._active_robot_ids: set[int] = set()
        self._startup_prepared = False
        self._prepared_robot_ids: tuple[int, ...] = ()

    @staticmethod
    def _validate_startup_config(startup: HcxStartupConfig) -> None:
        if not isinstance(startup, HcxStartupConfig):
            raise TypeError("startup must be an HcxStartupConfig")
        if not isinstance(startup.robot_ids, tuple):
            raise ValueError("HCX startup robot_ids must be a tuple")
        if startup.robot_ids:
            if len(startup.robot_ids) != 2:
                raise ValueError("HCX paired startup requires exactly two robot IDs")
            if len(set(startup.robot_ids)) != len(startup.robot_ids):
                raise ValueError("HCX startup robot IDs must be unique")
            for robot_id in startup.robot_ids:
                if (
                    not isinstance(robot_id, int)
                    or isinstance(robot_id, bool)
                    or robot_id < 0
                ):
                    raise ValueError("HCX startup robot IDs must be non-negative integers")

        for field_name in (
            "auto_detach_hmi",
            "auto_clear_alarms",
            "auto_enable",
        ):
            if not isinstance(getattr(startup, field_name), bool):
                raise ValueError(f"HCX startup {field_name} must be a boolean")
        if not startup.robot_ids and (
            startup.auto_detach_hmi
            or startup.auto_clear_alarms
            or startup.auto_enable
        ):
            raise ValueError(
                "HCX automatic startup requires a configured left/right robot pair"
            )

        HcxConnection._validate_duration(
            startup.controller_initialization_wait_s,
            "controller_initialization_wait_s",
            allow_zero=True,
        )
        for field_name in (
            "ethercat_op_timeout_s",
            "alarm_clear_retry_interval_s",
            "global_enable_retry_interval_s",
            "single_arm_enable_timeout_s",
            "enable_status_poll_interval_s",
        ):
            HcxConnection._validate_duration(getattr(startup, field_name), field_name)
        for field_name in (
            "alarm_clear_retry_count",
            "global_enable_retry_count",
        ):
            value = getattr(startup, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"HCX startup {field_name} must be a non-negative integer")

        if not isinstance(startup.ethercat_master_indices, tuple):
            raise ValueError("HCX EtherCAT master indices must be a tuple")
        if len(set(startup.ethercat_master_indices)) != len(
            startup.ethercat_master_indices
        ):
            raise ValueError("HCX EtherCAT master indices must be unique")
        for master_index in startup.ethercat_master_indices:
            if (
                not isinstance(master_index, int)
                or isinstance(master_index, bool)
                or not 0 <= master_index <= 1
            ):
                raise ValueError("HCX EtherCAT master indices must be 0 or 1")

    @staticmethod
    def _validate_duration(
        value: object, field_name: str, *, allow_zero: bool = False
    ) -> None:
        if isinstance(value, bool):
            raise ValueError(f"HCX startup {field_name} must be a finite duration")
        try:
            seconds = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"HCX startup {field_name} must be a finite duration"
            ) from exc
        if not math.isfinite(seconds) or seconds < 0.0 or (not allow_zero and seconds == 0.0):
            raise ValueError(f"HCX startup {field_name} must be a positive duration")

    @staticmethod
    def _load_robot_client() -> Any:
        """仅在连接真实硬件时导入 HCX 原生 SDK。"""

        from hcx_sdk import RobotClient

        return RobotClient

    @property
    def client(self) -> Any | None:
        """返回已连接的共享客户端；未连接时返回 ``None``。"""

        with self._lock:
            return self._client

    def acquire(self, robot_id: int) -> Any:
        """取得一个机械臂句柄，并保留共享客户端直到所有从臂释放。"""

        with self._lock:
            configured_robot_ids = self.config.startup.robot_ids
            if configured_robot_ids and robot_id not in configured_robot_ids:
                raise ValueError(
                    f"HCX robot_id {robot_id} is not in the configured startup pair"
                )
            if robot_id in self._active_robot_ids:
                raise RuntimeError(f"HCX robot_id {robot_id} is already in use")

            created_client = False
            if self._client is None:
                robot_client_class = self._load_robot_client()
                client = robot_client_class(
                    self.config.local_ip, self.config.remote_ip, self.config.port
                )
                try:
                    client.connect(timeout_s=self.config.connect_timeout_s)
                except Exception:
                    try:
                        client.close()
                    except Exception:
                        pass
                    raise
                self._client = client
                created_client = True

            assert self._client is not None
            try:
                arm = self._client.arm(robot_id)
            except Exception:
                if created_client:
                    try:
                        self._client.close()
                    finally:
                        self._client = None
                raise

            self._active_robot_ids.add(robot_id)
            return arm

    def release(self, robot_id: int) -> None:
        """释放一个机械臂句柄；最后一个释放者关闭共享 SDK 客户端。"""

        with self._lock:
            self._active_robot_ids.discard(robot_id)
            if self._active_robot_ids or self._client is None:
                return
            client = self._client
            self._client = None
            self._reset_startup_locked()
            client.close()

    def prepare_for_motion(self, robot_id: int) -> bool:
        """完成配置的双臂前置流程，并确认当前可安全发送规划运动。"""

        with self._lock:
            client = self._require_client_locked()
            robot_ids = self._motion_robot_ids_locked(robot_id)
            arms = self._arms_locked(client, robot_ids)
            if not self.config.startup.robot_ids:
                self._require_motion_ready_locked(client, arms, require_hmi_detached=False)
                return True

            if self._startup_prepared:
                if self._prepared_robot_ids != robot_ids:
                    raise RuntimeError("HCX startup pair changed while the connection is active")
                self._require_motion_ready_locked(client, arms, require_hmi_detached=True)
                return True

            self._prepare_pair_for_motion_locked(client, arms)
            self._startup_prepared = True
            self._prepared_robot_ids = robot_ids
            return True

    def motion_ready(self, robot_id: int) -> bool:
        """只读复核当前机械臂及其成对启动条件是否仍然满足。"""

        try:
            with self._lock:
                client = self._require_client_locked()
                robot_ids = self._motion_robot_ids_locked(robot_id)
                if self.config.startup.robot_ids and (
                    not self._startup_prepared
                    or self._prepared_robot_ids != robot_ids
                ):
                    return False
                arms = self._arms_locked(client, robot_ids)
                self._require_motion_ready_locked(
                    client,
                    arms,
                    require_hmi_detached=bool(self.config.startup.robot_ids),
                )
                return True
        except (RuntimeError, TypeError, ValueError):
            return False

    def _prepare_pair_for_motion_locked(
        self, client: Any, arms: tuple[tuple[int, Any], ...]
    ) -> None:
        startup = self.config.startup
        initialization_wait_s = float(startup.controller_initialization_wait_s)
        if initialization_wait_s > 0.0:
            time.sleep(initialization_wait_s)

        if not bool(client.hmi_detached) and startup.auto_detach_hmi:
            client.detach_hmi()
        if not bool(client.hmi_detached):
            raise RuntimeError(
                "HCX teach pendant is not detached; confirm it is physically absent and "
                "set hcx.auto_detach_hmi: true if a detach request is appropriate"
            )

        self._clear_active_alarms_locked(client)
        if not bool(client.soft_emergency_stop_normal):
            raise RuntimeError("HCX controller soft emergency stop is not normal")
        for master_index in startup.ethercat_master_indices:
            self._wait_for_state_locked(
                lambda master_index=master_index: bool(
                    client.ethercat_master_operational(master_index)
                ),
                f"EtherCAT master {master_index} OP state",
                float(startup.ethercat_op_timeout_s),
                float(startup.enable_status_poll_interval_s),
            )

        self._ensure_global_enabled_locked(client)
        self._ensure_arms_enabled_locked(arms)
        self._require_motion_ready_locked(client, arms, require_hmi_detached=True)

    def _clear_active_alarms_locked(self, client: Any) -> None:
        startup = self.config.startup
        alarms = tuple(client.active_alarms)
        if not alarms:
            return
        if not startup.auto_clear_alarms:
            raise RuntimeError(
                "HCX controller has active alarms; resolve them on site or explicitly set "
                "hcx.auto_clear_alarms: true after confirming a reset is safe: "
                + "; ".join(str(alarm) for alarm in alarms)
            )

        for attempt in range(startup.alarm_clear_retry_count + 1):
            client.clear_alarms()
            alarms = tuple(client.active_alarms)
            if not alarms:
                return
            if attempt < startup.alarm_clear_retry_count:
                time.sleep(float(startup.alarm_clear_retry_interval_s))
        raise RuntimeError(
            "HCX alarms remain after configured clear attempts: "
            + "; ".join(str(alarm) for alarm in alarms)
        )

    def _ensure_global_enabled_locked(self, client: Any) -> None:
        startup = self.config.startup
        if bool(client.global_enabled):
            return
        if not startup.auto_enable:
            raise RuntimeError(
                "HCX global enable is false; enable it on site or explicitly set "
                "hcx.auto_enable: true"
            )

        last_error: Exception | None = None
        for attempt in range(startup.global_enable_retry_count + 1):
            try:
                client.set_global_enable(True)
            except (RuntimeError, TypeError, ValueError) as exc:
                last_error = exc
            try:
                if bool(client.global_enabled):
                    return
            except (RuntimeError, TypeError, ValueError) as exc:
                last_error = exc
            if attempt < startup.global_enable_retry_count:
                time.sleep(float(startup.global_enable_retry_interval_s))

        message = "HCX global enable did not become true after configured retries"
        if last_error is not None:
            raise RuntimeError(message) from last_error
        raise RuntimeError(message)

    def _ensure_arms_enabled_locked(self, arms: tuple[tuple[int, Any], ...]) -> None:
        startup = self.config.startup
        for robot_id, arm in arms:
            if bool(arm.enabled):
                continue
            if not startup.auto_enable:
                raise RuntimeError(
                    f"HCX robot {robot_id} is not enabled; enable it on site or explicitly "
                    "set hcx.auto_enable: true"
                )
            arm.set_enabled(True)
            self._wait_for_state_locked(
                lambda arm=arm: bool(arm.enabled),
                f"HCX robot {robot_id} enable state",
                float(startup.single_arm_enable_timeout_s),
                float(startup.enable_status_poll_interval_s),
            )

    @staticmethod
    def _wait_for_state_locked(
        state: Callable[[], bool],
        description: str,
        timeout_s: float,
        poll_interval_s: float,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while not state():
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                raise RuntimeError(f"HCX {description} did not become ready before timeout")
            time.sleep(min(poll_interval_s, remaining_s))

    def _require_motion_ready_locked(
        self,
        client: Any,
        arms: tuple[tuple[int, Any], ...],
        *,
        require_hmi_detached: bool,
    ) -> None:
        if not bool(client.connected):
            raise RuntimeError("HCX client is not connected")
        if require_hmi_detached and not bool(client.hmi_detached):
            raise RuntimeError("HCX teach pendant is not detached")
        alarms = tuple(client.active_alarms)
        if alarms:
            raise RuntimeError(
                "HCX controller has active alarms: "
                + "; ".join(str(alarm) for alarm in alarms)
            )
        if not bool(client.soft_emergency_stop_normal):
            raise RuntimeError("HCX controller soft emergency stop is not normal")
        for master_index in self.config.startup.ethercat_master_indices:
            if not bool(client.ethercat_master_operational(master_index)):
                raise RuntimeError(f"HCX EtherCAT master {master_index} is not OP")
        if not bool(client.global_enabled):
            raise RuntimeError("HCX global enable is false")
        for robot_id, arm in arms:
            if not bool(arm.enabled):
                raise RuntimeError(f"HCX robot {robot_id} enable is false")

    def _require_client_locked(self) -> Any:
        if self._client is None:
            raise RuntimeError("HCX connection is not active")
        return self._client

    def _motion_robot_ids_locked(self, requested_robot_id: int) -> tuple[int, ...]:
        configured_robot_ids = self.config.startup.robot_ids
        if not configured_robot_ids:
            return (requested_robot_id,)
        if requested_robot_id not in configured_robot_ids:
            raise ValueError(
                f"HCX robot_id {requested_robot_id} is not in the configured startup pair"
            )
        return configured_robot_ids

    @staticmethod
    def _arms_locked(client: Any, robot_ids: tuple[int, ...]) -> tuple[tuple[int, Any], ...]:
        return tuple((robot_id, client.arm(robot_id)) for robot_id in robot_ids)

    def _reset_startup_locked(self) -> None:
        self._startup_prepared = False
        self._prepared_robot_ids = ()


class HcxFollower(FollowerArm):
    """绑定 HCX 单侧七轴机械臂的 ``FollowerArm`` 实现。

    默认使用非阻塞 ``move_joints`` 规划运动；提供 ``direct_servo_config``
    时，改用高频 ``PluseToServo`` 直伺服会话。两种模式均保持度制七轴目标
    和 ``FollowerArm`` 的通用发送契约。
    """

    def __init__(
        self,
        connection: HcxConnection,
        robot_id: int,
        side: HcxArmSide,
        motion_config: HcxMoveJointsConfig = HcxMoveJointsConfig(),
        *,
        direct_servo_config: HcxDirectServoConfig | None = None,
        on_direct_servo_target_submitted: DirectServoTargetObserver | None = None,
    ):
        if not isinstance(robot_id, int) or isinstance(robot_id, bool) or robot_id < 0:
            raise ValueError("robot_id must be a non-negative integer")
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if not isinstance(motion_config, HcxMoveJointsConfig):
            raise TypeError("motion_config must be an HcxMoveJointsConfig")
        if (
            direct_servo_config is not None
            and not isinstance(direct_servo_config, HcxDirectServoConfig)
        ):
            raise TypeError("direct_servo_config must be an HcxDirectServoConfig or None")
        if on_direct_servo_target_submitted is not None and not callable(
            on_direct_servo_target_submitted
        ):
            raise TypeError(
                "on_direct_servo_target_submitted must be callable or None"
            )
        if (
            on_direct_servo_target_submitted is not None
            and direct_servo_config is None
        ):
            raise ValueError(
                "on_direct_servo_target_submitted requires direct_servo_config"
            )

        self.connection = connection
        self.robot_id = robot_id
        self.side = side
        self.motion_config = motion_config
        self.direct_servo_config = direct_servo_config
        lower_deg, upper_deg = _JOINT_LIMITS_DEG
        self._min_angles_deg = np.asarray(lower_deg, dtype=float)
        self._max_angles_deg = np.asarray(upper_deg, dtype=float)
        self._lock = threading.RLock()
        # 生命周期操作需要在等待 Python 输出线程退出时释放 _lock；单独的锁
        # 防止 start/recover/stop 之间交错替换同一个原生会话。
        self._direct_servo_lifecycle_lock = threading.RLock()
        self._arm: Any | None = None
        self._direct_servo_session: Any | None = None
        self._direct_servo_target: np.ndarray | None = None
        self._direct_servo_source_target: np.ndarray | None = None
        # 轨迹批次只由 100 Hz 目标更新路径追加，只由高频发送线程弹出。
        # 高速线程绝不在此队列上执行插值或限幅计算。
        self._direct_servo_trajectory_batches: deque[_DirectServoTrajectoryBatch] = (
            deque()
        )
        # 此观察者记录 Python 高频输出线程已成功提交给 C++ 薄桥接层的目标；它
        # 不读取机械臂反馈。
        self._on_direct_servo_target_submitted = on_direct_servo_target_submitted
        self._direct_servo_interpolator: LinearInterpolator | None = None
        self._direct_servo_limited_interpolator: LimitedInterpolator | None = None
        self._direct_servo_output_thread: threading.Thread | None = None
        self._direct_servo_output_stop_event: threading.Event | None = None
        self._direct_servo_output_error: BaseException | None = None
        self._direct_servo_output_telemetry: _DirectServoOutputTelemetry | None = None
        if (
            direct_servo_config is not None
            and direct_servo_config.interpolation == "linear"
        ):
            assert direct_servo_config.source_rate_hz is not None
            self._direct_servo_interpolator = LinearInterpolator(
                direct_servo_config.source_rate_hz, direct_servo_config.rate_hz
            )
        elif (
            direct_servo_config is not None
            and direct_servo_config.interpolation == "limited"
        ):
            assert direct_servo_config.source_rate_hz is not None
            assert direct_servo_config.limited_max_velocity_deg_s is not None
            assert direct_servo_config.limited_max_acceleration_deg_s2 is not None
            assert direct_servo_config.limited_lowpass_alpha is not None
            self._direct_servo_limited_interpolator = LimitedInterpolator(
                direct_servo_config.source_rate_hz,
                direct_servo_config.rate_hz,
                max_velocity_deg_s=direct_servo_config.limited_max_velocity_deg_s,
                max_acceleration_deg_s2=(
                    direct_servo_config.limited_max_acceleration_deg_s2
                ),
                lowpass_alpha=direct_servo_config.limited_lowpass_alpha,
                min_angles_deg=self._min_angles_deg,
                max_angles_deg=self._max_angles_deg,
            )
        self._connected = False
        self._servo_started = False

    @property
    def joint_count(self) -> int:
        """HCX 每侧机械臂固定公开七个关节。"""

        return _AXIS_COUNT

    @property
    def joint_limits_deg(self) -> tuple[np.ndarray, np.ndarray]:
        """返回该侧部署定义的七轴安全限位，单位为度。"""

        return self._min_angles_deg.copy(), self._max_angles_deg.copy()

    @property
    def requires_per_cycle_target_updates(self) -> bool:
        """插值模式必须收到每个上游周期，以生成完整的高频点批次。"""

        return bool(
            self.direct_servo_config is not None
            and self.direct_servo_config.interpolation in ("linear", "limited")
        )

    def direct_servo_output_stats(self) -> HcxDirectServoOutputStats | None:
        """返回本侧 Python 输出线程的实际下发统计。

        调用此方法不会读取 HCX 关节反馈，也不会调用厂商 SDK。它用于区分
        “配置请求 500 Hz” 与 “Python 到薄原生绑定实际成功返回的频率”。
        """

        with self._lock:
            telemetry = self._direct_servo_output_telemetry
            session = self._direct_servo_session
            running = self._direct_servo_output_running_locked(session)
        if telemetry is None:
            return None
        return telemetry.snapshot(running=running)

    def connect(self) -> None:
        """连接共享控制器，并验证轴数和部署限位。"""

        with self._lock:
            if self._connected:
                return
            arm = self.connection.acquire(self.robot_id)
            try:
                if arm.axis_count != self.joint_count:
                    raise RuntimeError(
                        f"HCX {self.side} arm must expose {self.joint_count} axes, "
                        f"got {arm.axis_count}"
                    )
                self._validate_controller_limits(arm)
            except Exception:
                self.connection.release(self.robot_id)
                raise
            self._arm = arm
            self._connected = True

    def _validate_controller_limits(self, arm: Any) -> None:
        try:
            controller_limits = np.asarray(arm.joint_limits_deg, dtype=float)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "HCX controller did not provide valid joint limits"
            ) from exc
        if controller_limits.shape != (self.joint_count, 2):
            raise RuntimeError(
                f"HCX {self.side} arm returned invalid joint-limit dimensions "
                f"{controller_limits.shape}"
            )
        if not np.isfinite(controller_limits).all() or np.any(
            controller_limits[:, 0] >= controller_limits[:, 1]
        ):
            raise RuntimeError(f"HCX {self.side} arm returned invalid joint limits")
        deployment_limits_exceed_controller = np.any(
            self._min_angles_deg < controller_limits[:, 0] - _LIMIT_TOLERANCE_DEG
        ) or np.any(
            self._max_angles_deg > controller_limits[:, 1] + _LIMIT_TOLERANCE_DEG
        )
        if deployment_limits_exceed_controller:
            raise RuntimeError(
                f"HCX {self.side} deployment limits exceed controller joint limits"
            )

    def read_joint_angles_deg(self) -> np.ndarray:
        """读取当前七轴关节角度，单位为度。"""

        with self._lock:
            arm = self._require_connected()
            try:
                angles_deg = np.asarray(arm.joint_angles(), dtype=float)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"failed to read HCX {self.side} joint angles"
                ) from exc
            if (
                angles_deg.shape != (self.joint_count,)
                or not np.isfinite(angles_deg).all()
            ):
                raise RuntimeError(
                    f"HCX {self.side} arm returned invalid joint feedback"
                )
            if (
                self.direct_servo_config is not None
                and self._direct_servo_target is None
            ):
                # 首次读取的实际姿态是启动直伺服时最安全的保持目标。
                self._direct_servo_target = angles_deg.copy()
            return angles_deg.copy()

    def start_servo(self) -> bool:
        """执行启动前置流程，并启动所选的规划或直接伺服模式。"""

        with self._direct_servo_lifecycle_lock:
            with self._lock:
                if not self._connected or self._arm is None:
                    return False
                direct_config = self.direct_servo_config

            if direct_config is not None and not direct_config.confirm_unsafe:
                try:
                    self._stop_direct_servo_session()
                except (RuntimeError, TypeError, ValueError) as exc:
                    print(f"[WARN] 停止 HCX {self.side} 未确认的直伺服失败: {exc}")
                print(
                    f"[WARN] HCX {self.side} 直伺服要求显式设置 "
                    "direct_servo_confirm_unsafe: true"
                )
                with self._lock:
                    self._servo_started = False
                return False

            with self._lock:
                session = self._direct_servo_session
            if direct_config is not None and session is not None:
                try:
                    ready = self.connection.motion_ready(self.robot_id)
                    with self._lock:
                        session = self._direct_servo_session
                        session_running = bool(
                            session is not None
                            and getattr(session.state, "running", False)
                        )
                        output_running = self._direct_servo_output_running_locked(
                            session
                        )
                        if session_running and ready and output_running:
                            self._servo_started = True
                            return True
                        if session is not None and not output_running:
                            output_error = self._direct_servo_output_error
                            self._report_direct_servo_failure_locked(
                                "高频输出状态检查",
                                output_error
                                or RuntimeError(
                                    "HCX Python direct-servo output worker is not running"
                                ),
                                session,
                            )
                except (RuntimeError, TypeError, ValueError) as exc:
                    try:
                        self._stop_direct_servo_session()
                    except (RuntimeError, TypeError, ValueError) as stop_exc:
                        print(
                            f"[WARN] 停止 HCX {self.side} 状态异常后的直伺服失败: "
                            f"{stop_exc}"
                        )
                    print(f"[WARN] HCX {self.side} 直伺服状态检查失败: {exc}")
                    with self._lock:
                        self._servo_started = False
                    return False
                try:
                    self._stop_direct_servo_session()
                except (RuntimeError, TypeError, ValueError) as exc:
                    print(f"[WARN] 停止 HCX {self.side} 失效的直伺服失败: {exc}")
                    with self._lock:
                        self._servo_started = False
                    return False

            try:
                self.connection.prepare_for_motion(self.robot_id)
                ready = self.connection.motion_ready(self.robot_id)
                if ready and direct_config is not None:
                    with self._lock:
                        self._start_direct_servo_session_locked(direct_config)
            except (RuntimeError, TypeError, ValueError) as exc:
                print(f"[WARN] HCX {self.side} 启动前置检查失败: {exc}")
                ready = False
            with self._lock:
                self._servo_started = ready
                return ready

    def send_joint_angles_deg(
        self, angles_deg: np.ndarray, command_time_s: float
    ) -> bool:
        """提交一帧七轴度制目标。

        规划模式使用 ``motion_config`` 的非阻塞、可中断 ``move_joints``。
        直接模式在此低频上游路径预生成一个 ``linear`` 或 ``limited``
        高频点批次；本侧独立输出线程只按 ``direct_servo_config.rate_hz``
        消费点并调用薄原生绑定。``command_time_s`` 仅校验通用调用契约。
        """

        try:
            duration_s = float(command_time_s)
            target = np.asarray(angles_deg, dtype=float)
        except (TypeError, ValueError):
            return False
        if (
            not math.isfinite(duration_s)
            or duration_s <= 0.0
            or target.shape != (self.joint_count,)
            or not np.isfinite(target).all()
            or np.any(target < self._min_angles_deg)
            or np.any(target > self._max_angles_deg)
        ):
            return False

        with self._lock:
            if not self._servo_started:
                return False
            if self.direct_servo_config is not None:
                session = self._direct_servo_session
                output_error = self._direct_servo_output_error
                if session is None:
                    self._report_direct_servo_failure_locked(
                        "更新目标",
                        RuntimeError("HCX direct-servo session is not active"),
                        session,
                    )
                    return False
                if output_error is not None:
                    self._report_direct_servo_failure_locked(
                        "更新目标", output_error, session
                    )
                    return False
                if not self._direct_servo_output_running_locked(session):
                    self._report_direct_servo_failure_locked(
                        "更新目标",
                        RuntimeError(
                            "HCX Python direct-servo output worker is not running"
                        ),
                        session,
                    )
                    return False
                # 每个上游周期在这里完成全部轨迹计算，并发布一批完整的
                # 高频目标。高频线程只会 popleft 一个已生成点并 set_target。
                # 不能在这里读取 session.state：该属性与 set_target() 共享
                # DirectServoSession 的会话锁；健康状态仍由高频线程写入。
                self._publish_direct_servo_trajectory_locked(target)
                self._direct_servo_target = target.copy()
                return True

            arm = self._require_connected()
            try:
                with self.connection._lock:
                    arm.move_joints(
                        target.tolist(),
                        interrupt=True,
                        acceleration_seconds=self.motion_config.acceleration_seconds,
                        deceleration_seconds=self.motion_config.deceleration_seconds,
                        speed_ratio=self.motion_config.speed_ratio,
                        smooth=self.motion_config.smooth,
                        wait=False,
                    )
            except (RuntimeError, TypeError, ValueError) as exc:
                return False
        return True

    def _publish_direct_servo_trajectory_locked(self, target: np.ndarray) -> None:
        """在上游控制周期预生成一整段高频轨迹并发布。

        调用方持有 ``self._lock``。此处是 ``linear``/``limited`` 唯一允许
        调用插值器的位置；高频发送线程只消费已经不可变的批次。
        """

        direct_config = self.direct_servo_config
        if direct_config is None:
            raise RuntimeError("HCX direct-servo config is not initialized")

        if direct_config.interpolation == "direct":
            points = target[np.newaxis, :].copy()
        elif direct_config.interpolation == "linear":
            interpolator = self._direct_servo_interpolator
            source_target = self._direct_servo_source_target
            if interpolator is None or source_target is None:
                raise RuntimeError(
                    "HCX linear direct-servo trajectory producer is not initialized"
                )
            points = interpolator.interpolate(source_target, target)
            self._direct_servo_source_target = target.copy()
        else:
            interpolator = self._direct_servo_limited_interpolator
            if interpolator is None:
                raise RuntimeError(
                    "HCX limited direct-servo trajectory producer is not initialized"
                )
            points = interpolator.interpolate(target)

        if direct_config.interpolation in ("linear", "limited"):
            assert direct_config.source_rate_hz is not None
            expected_point_count = direct_config.rate_hz // direct_config.source_rate_hz
        else:
            expected_point_count = 1
        trajectory = np.asarray(points, dtype=float)
        if (
            trajectory.shape != (expected_point_count, self.joint_count)
            or not np.isfinite(trajectory).all()
        ):
            raise RuntimeError("HCX direct-servo trajectory producer returned invalid points")

        # 发布后轨迹只由发送侧读取，避免生产者复用/修改数组造成半批次目标。
        published = trajectory.copy()
        published.setflags(write=False)
        self._direct_servo_trajectory_batches.append(
            _DirectServoTrajectoryBatch(points_deg=published)
        )

    def refresh_servo_target(self) -> bool:
        """检查 Python 直伺服输出线程，不重复下发或轮询原生会话。"""

        with self._lock:
            if self.direct_servo_config is None:
                return True
            if not self._servo_started or not self._connected or self._arm is None:
                return False
            session = self._direct_servo_session
            target = self._direct_servo_target
            if session is None or target is None:
                self._report_direct_servo_failure_locked(
                    "刷新目标",
                    RuntimeError("HCX direct-servo session or target is not initialized"),
                    session,
                )
                return False
            if self._direct_servo_output_error is not None:
                self._report_direct_servo_failure_locked(
                    "刷新目标", self._direct_servo_output_error, session
                )
                return False
            if not self._direct_servo_output_running_locked(session):
                self._report_direct_servo_failure_locked(
                    "刷新目标",
                    RuntimeError("HCX Python direct-servo output worker is not running"),
                    session,
                )
                return False
        return True

    def recover(self) -> bool:
        """重新检查状态；直伺服故障时重建本侧会话但不重复启动前置操作。"""

        with self._direct_servo_lifecycle_lock:
            with self._lock:
                if not self._connected or self._arm is None:
                    self._servo_started = False
                    return False
                direct_config = self.direct_servo_config
                if direct_config is None:
                    self._servo_started = self.connection.motion_ready(self.robot_id)
                    return self._servo_started
                if not direct_config.confirm_unsafe:
                    self._servo_started = False
                    return False
            try:
                self._stop_direct_servo_session()
                if not self.connection.motion_ready(self.robot_id):
                    with self._lock:
                        self._servo_started = False
                    return False
                with self._lock:
                    self._start_direct_servo_session_locked(direct_config)
            except (RuntimeError, TypeError, ValueError) as exc:
                print(f"[WARN] 恢复 HCX {self.side} 直伺服失败: {exc}")
                with self._lock:
                    self._servo_started = False
                return False
            with self._lock:
                self._servo_started = True
                return True

    def stop_servo(self) -> None:
        """停止本侧规划路径或直接伺服软件下发，不影响另一侧机械臂。"""

        with self._direct_servo_lifecycle_lock:
            with self._lock:
                was_started = self._servo_started
                self._servo_started = False
                direct_servo = self.direct_servo_config is not None
            if direct_servo:
                try:
                    self._stop_direct_servo_session()
                except (RuntimeError, TypeError, ValueError) as exc:
                    print(f"[WARN] 停止 HCX {self.side} 直伺服失败: {exc}")
                with self._lock:
                    self._direct_servo_target = None
                    self._direct_servo_source_target = None
                    self._direct_servo_trajectory_batches = deque()
                return
            with self._lock:
                if not was_started or not self._connected or self._arm is None:
                    return
                arm = self._arm
            try:
                with self.connection._lock:
                    arm.clear_route(emergency_stop=True)
            except (RuntimeError, TypeError, ValueError) as exc:
                print(f"[WARN] 停止 HCX {self.side} 机械臂路径失败: {exc}")

    def disconnect(self) -> None:
        """释放本侧机械臂；最后一个从臂断开时关闭共享连接。"""

        with self._direct_servo_lifecycle_lock:
            with self._lock:
                if not self._connected:
                    return
                should_stop = (
                    self._servo_started or self._direct_servo_session is not None
                )
            if should_stop:
                self.stop_servo()
            with self._lock:
                if not self._connected:
                    return
                self._servo_started = False
                self._connected = False
                self._arm = None
                self._direct_servo_target = None
                self._direct_servo_source_target = None
                self._direct_servo_trajectory_batches = deque()
                self.connection.release(self.robot_id)

    def _start_direct_servo_session_locked(
        self, direct_config: HcxDirectServoConfig
    ) -> None:
        """创建 Python 会话、发送安全保持点并启动本侧输出线程。

        调用方必须持有 ``self._lock``。华成原生扩展在此路径只会收到一次
        ``PluseToServo`` 调用；后续目标由低频上游路径预生成，高频线程只
        执行固定频率调度和单次 ``PluseToServo`` 调用。
        """

        arm = self._require_connected()
        # 每次重建会话都从实际反馈开始，避免故障恢复时将未到达的旧目标直接
        # 重新下发而产生姿态跳变。
        target = self._read_direct_servo_seed_target_locked(arm)
        session: Any | None = None
        try:
            session = arm.start_direct_servo(
                rate_hz=direct_config.rate_hz,
                watchdog_s=direct_config.watchdog_s,
                confirm_unsafe=direct_config.confirm_unsafe,
            )
            # 首次下发是安全保持点；随后 Python 发送线程以 rate_hz 重复下发。
            session.set_target(target.tolist())
            if self._direct_servo_limited_interpolator is not None:
                self._direct_servo_limited_interpolator.reset(target)
        except (RuntimeError, TypeError, ValueError) as exc:
            self._report_direct_servo_failure_locked("启动或初始化", exc, session)
            if session is not None:
                try:
                    session.stop()
                except (RuntimeError, TypeError, ValueError) as stop_exc:
                    print(f"[WARN] 停止 HCX {self.side} 初始化失败的直伺服: {stop_exc}")
            raise

        self._direct_servo_target = target.copy()
        self._direct_servo_source_target = target.copy()
        # 新会话不能消费恢复前遗留的旧姿态轨迹。
        self._direct_servo_trajectory_batches = deque()
        self._direct_servo_session = session
        self._direct_servo_output_error = None
        telemetry = _DirectServoOutputTelemetry(direct_config.rate_hz)
        self._direct_servo_output_telemetry = telemetry
        stop_event = threading.Event()
        output_thread = threading.Thread(
            target=self._run_direct_servo_output,
            args=(session, stop_event, telemetry),
            name=f"hcx-{self.side}-direct-servo",
            daemon=True,
        )
        self._direct_servo_output_stop_event = stop_event
        self._direct_servo_output_thread = output_thread
        try:
            output_thread.start()
        except RuntimeError:
            stop_event.set()
            if output_thread.is_alive():
                output_thread.join()
            self._direct_servo_output_stop_event = None
            self._direct_servo_output_thread = None
            self._direct_servo_session = None
            session.stop()
            raise

    def _direct_servo_output_running_locked(self, session: Any | None) -> bool:
        """返回当前会话的 Python 高频输出线程是否仍在运行。"""

        output_thread = self._direct_servo_output_thread
        stop_event = self._direct_servo_output_stop_event
        return (
            session is not None
            and self._direct_servo_session is session
            and self._direct_servo_output_error is None
            and output_thread is not None
            and output_thread.is_alive()
            and stop_event is not None
            and not stop_event.is_set()
        )

    def _run_direct_servo_output(
        self,
        session: Any,
        stop_event: threading.Event,
        telemetry: _DirectServoOutputTelemetry,
    ) -> None:
        """按配置频率消费已生成目标，并单次下发给 HCX SDK。

        此循环刻意不访问 ``LinearInterpolator`` 或 ``LimitedInterpolator``。
        所有插值、低通、限速和限加速度均在 ``send_joint_angles_deg()`` 的
        上游控制周期完成；队列暂时没有新点时，只重发最后一个已发送点。
        """

        direct_config = self.direct_servo_config
        if direct_config is None:
            return

        period_s = 1.0 / direct_config.rate_hz
        next_tick_s = time.monotonic() + period_s
        with self._lock:
            if self._direct_servo_session is not session:
                return
            initial_target = self._direct_servo_target
            if initial_target is None:
                self._record_direct_servo_output_failure(
                    session,
                    stop_event,
                    RuntimeError("HCX direct-servo target is not initialized"),
                )
                return
            last_output = initial_target.copy()
            trajectory_batches = self._direct_servo_trajectory_batches
        active_batch: _DirectServoTrajectoryBatch | None = None
        active_point_index = 0

        while not stop_event.is_set():
            remaining_s = next_tick_s - time.monotonic()
            if remaining_s > 0.0 and stop_event.wait(remaining_s):
                return

            dispatch_started_s = time.monotonic()
            start_lateness_s = max(dispatch_started_s - next_tick_s, 0.0)
            try:
                if (
                    active_batch is None
                    or active_point_index >= active_batch.points_deg.shape[0]
                ):
                    try:
                        active_batch = trajectory_batches.popleft()
                    except IndexError:
                        active_batch = None
                    active_point_index = 0

                if active_batch is None:
                    # 主臂没有新样本或低频生产端暂时迟到时，仍严格维持
                    # direct_servo_rate_hz；不做任何轨迹计算。
                    point = last_output
                else:
                    point = active_batch.points_deg[active_point_index]
                    active_point_index += 1

                if stop_event.is_set():
                    return
                # 不持有 HcxConnection._lock：每侧 Python 线程可并行进入
                # pybind 的单次 PluseToServo 调用，扩展在该调用期间释放 GIL。
                # 厂商 bool 返回值按直伺服透传约定忽略。
                set_target_started_s = time.monotonic()
                session.set_target(point.tolist())
                set_target_finished_s = time.monotonic()
                last_output = point
                self._notify_direct_servo_target_submitted(
                    ((point, time.perf_counter()),)
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                self._record_direct_servo_output_failure(session, stop_event, exc)
                return

            loop_finished_s = time.monotonic()
            next_tick_s += period_s
            # 固定时间轴绝不因一次慢调用而向前跳跃。若下一计划时隙已经到期，
            # 下次循环会立即发送一个目标（新目标优先，否则重发上一目标），
            # 直到追上节拍；不会为了避免补发而主动丢掉任何 500 Hz 时隙。
            telemetry.record_success(
                completed_at_s=loop_finished_s,
                set_target_duration_s=set_target_finished_s - set_target_started_s,
                missed_ticks=0,
                start_lateness_s=start_lateness_s,
            )

    def _record_direct_servo_output_failure(
        self, session: Any, stop_event: threading.Event, exc: BaseException
    ) -> None:
        """记录 Python 输出线程故障，让控制器沿既有恢复路径处理。"""

        with self._lock:
            if stop_event.is_set() or self._direct_servo_session is not session:
                return
            self._direct_servo_output_error = exc
            self._report_direct_servo_failure_locked("高频输出", exc, session)

    def _notify_direct_servo_target_submitted(
        self, samples: tuple[tuple[np.ndarray, float], ...]
    ) -> None:
        """通知已由 Python 输出线程成功提交给薄原生绑定的目标。"""

        observer = self._on_direct_servo_target_submitted
        if observer is None:
            return
        for target, timestamp_s in samples:
            try:
                observer(target.copy(), timestamp_s)
            except Exception as exc:
                with self._lock:
                    if self._on_direct_servo_target_submitted is observer:
                        self._on_direct_servo_target_submitted = None
                print(f"[WARN] HCX {self.side} 直伺服目标观察回调已禁用: {exc}")
                return

    def _read_direct_servo_seed_target_locked(self, arm: Any) -> np.ndarray:
        """读取当前姿态作为首次直伺服目标，避免启动时产生位置跳变。"""

        try:
            target = np.asarray(arm.joint_angles(), dtype=float)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"failed to read HCX {self.side} direct-servo seed target"
            ) from exc
        if target.shape != (self.joint_count,) or not np.isfinite(target).all():
            raise RuntimeError(
                f"HCX {self.side} returned an invalid direct-servo seed target"
            )
        if np.any(target < self._min_angles_deg) or np.any(target > self._max_angles_deg):
            raise RuntimeError(
                f"HCX {self.side} current pose is outside the configured direct-servo limits"
            )
        return target.copy()

    def _report_direct_servo_failure_locked(
        self, operation: str, exc: BaseException, session: Any | None
    ) -> None:
        """输出原生会话故障快照，避免把厂商失败压缩成泛化告警。"""

        state: Any | None = None
        if session is not None:
            try:
                state = session.state
            except (RuntimeError, TypeError, ValueError):
                state = None
        if state is None:
            print(f"[WARN] HCX {self.side} 直伺服{operation}失败: {exc}; 无会话状态快照")
            return

        running = getattr(state, "running", "unknown")
        faulted = getattr(state, "faulted", "unknown")
        sent_count = getattr(state, "sent_count", "unknown")
        error = getattr(state, "error", None)
        print(
            f"[WARN] HCX {self.side} 直伺服{operation}失败: {exc}; "
            f"running={running}, faulted={faulted}, sent_count={sent_count}, error={error!r}"
        )

    def _stop_direct_servo_session(self) -> None:
        """先停止本侧 Python 输出线程，再关闭其 Python 会话。"""

        with self._lock:
            session = self._direct_servo_session
            output_thread = self._direct_servo_output_thread
            output_stop_event = self._direct_servo_output_stop_event
            self._direct_servo_session = None
            self._direct_servo_output_thread = None
            self._direct_servo_output_stop_event = None
            self._direct_servo_output_error = None
            if output_stop_event is not None:
                output_stop_event.set()

        if output_thread is not None:
            if output_thread is threading.current_thread():
                raise RuntimeError(
                    "HCX direct-servo output thread cannot stop itself"
                )
            output_thread.join()
        if session is not None:
            session.stop()

    def _require_connected(self) -> Any:
        if not self._connected or self._arm is None:
            raise RuntimeError(f"HCX {self.side} follower is not connected")
        return self._arm
