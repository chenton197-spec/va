"""基于 HCX 原生桥接层的高层关节控制接口，角度单位为度。"""

from __future__ import annotations

import logging
import math
import threading
import time
import weakref
from collections.abc import Callable, Iterable
from typing import Optional

from ._native_loader import native
from .errors import (
    AlarmActiveError,
    ConnectionStateError,
    DirectServoFault,
    HcxSdkError,
    JointLimitError,
    MotionRejectedError,
    MotionTimeoutError,
)
from .types import DirectServoState, JointAngles, MotionResult

LOGGER = logging.getLogger(__name__)


def _timeout_ms(timeout_s: Optional[float]) -> int:
    if timeout_s is None:
        return -1
    if not math.isfinite(timeout_s) or timeout_s < 0:
        raise ValueError("timeout_s must be a finite, non-negative number")
    return math.ceil(timeout_s * 1000)


def _native_error(
    operation: str, error_type: type[HcxSdkError] = HcxSdkError
) -> HcxSdkError:
    return error_type(f"{operation} failed in the native HCX SDK")


def _connection_timeout_seconds(timeout_s: Optional[float]) -> Optional[float]:
    """验证连接等待超时；``None`` 表示无限等待。"""

    if timeout_s is None:
        return None
    if isinstance(timeout_s, bool):
        raise ValueError("timeout_s must be None or a positive finite number")
    try:
        value = float(timeout_s)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "timeout_s must be None or a positive finite number"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout_s must be None or a positive finite number")
    return value


class MotionHandle:
    """跟踪单次 `moveJoints2` 请求，且不阻塞 SDK 回调线程。"""

    def __init__(self, client: RobotClient, robot_id: int, sequence: int) -> None:
        self._client = client
        self.robot_id = robot_id
        self.sequence = sequence
        self._callbacks: list[Callable[[MotionResult], None]] = []
        self._callback_lock = threading.Lock()
        self._callback_thread: Optional[threading.Thread] = None

    def _status(self):
        try:
            status = self._client._native.motion_status(self.sequence)
        except RuntimeError as exc:
            raise _native_error("motion status") from exc
        if not status.known:
            raise MotionRejectedError(f"motion sequence {self.sequence} is unknown")
        return status

    @property
    def done(self) -> bool:
        status = self._status()
        return bool(status.done or status.cancelled)

    @property
    def succeeded(self) -> Optional[bool]:
        status = self._status()
        if status.cancelled:
            return False
        if not status.done:
            return None
        return bool(status.succeeded)

    def wait(self, timeout_s: Optional[float] = None) -> MotionResult:
        """等待运动完成回调并返回结果。"""

        try:
            status = self._client._native.wait_motion(
                self.sequence, _timeout_ms(timeout_s)
            )
        except RuntimeError as exc:
            raise _native_error("motion wait") from exc
        if not status.known:
            raise MotionRejectedError(f"motion sequence {self.sequence} is unknown")
        if status.timed_out:
            raise MotionTimeoutError(
                f"motion sequence {self.sequence} did not finish before timeout"
            )
        return MotionResult(
            robot_id=self.robot_id,
            sequence=self.sequence,
            succeeded=bool(status.succeeded),
            cancelled=bool(status.cancelled),
        )

    def add_done_callback(self, callback: Callable[[MotionResult], None]) -> None:
        """运动完成后，在 Python 守护线程中执行 `callback`。"""

        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._callback_lock:
            self._callbacks.append(callback)
            if self._callback_thread is None:
                self._callback_thread = threading.Thread(
                    target=self._dispatch_callbacks,
                    name=f"hcx-motion-{self.sequence}",
                    daemon=True,
                )
                self._callback_thread.start()

    def _dispatch_callbacks(self) -> None:
        try:
            result = self.wait()
        except HcxSdkError:
            LOGGER.exception(
                "motion callback dispatcher failed for sequence %s", self.sequence
            )
            with self._callback_lock:
                self._callback_thread = None
            return
        while True:
            with self._callback_lock:
                callbacks = tuple(self._callbacks)
                self._callbacks.clear()
                if not callbacks:
                    self._callback_thread = None
                    return
            for callback in callbacks:
                try:
                    callback(result)
                except Exception:
                    LOGGER.exception(
                        "motion completion callback raised for sequence %s",
                        self.sequence,
                    )


class DirectServoSession:
    """兼容层：每次 ``set_target`` 一对一调用 ``PluseToServo``。

    该类不管理轨迹、频率、看门狗、限位或故障恢复。它只保留旧调用方式，
    使现有程序可逐步迁移到 :meth:`Arm.pluse_to_servo`。发送调度和任何控制
    算法必须由调用方在 Python 中实现。
    """

    def __init__(
        self,
        arm: Arm,
        *,
        rate_hz: int | None = None,
        watchdog_s: float | None = None,
    ) -> None:
        self._arm = arm
        # These are retained as caller-owned metadata for source compatibility.
        # They do not affect vendor calls.
        self.rate_hz = rate_hz
        self.watchdog_s = watchdog_s
        self._state_lock = threading.Lock()
        self._stopped = False
        self._sent_count = 0
        self._error: Optional[str] = None

    def __enter__(self) -> DirectServoSession:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    def set_target(self, angles_deg: Iterable[float]) -> bool:
        """原样调用厂商 ``PluseToServo`` 并返回其 ``bool`` 结果。

        不读取反馈、不检查轴数或关节限位，也不会因一次 ``false`` 停止后续
        调用。角度数组以度为单位，长度和可接受范围均由厂商 SDK 决定。
        """

        with self._state_lock:
            if self._stopped:
                raise DirectServoFault("direct-servo session is stopped")

        accepted = self._arm.pluse_to_servo(angles_deg)
        with self._state_lock:
            if accepted:
                self._sent_count += 1
                self._error = None
            else:
                self._error = "PluseToServo returned false"
        return accepted

    @property
    def state(self) -> DirectServoState:
        with self._state_lock:
            return DirectServoState(
                running=not self._stopped,
                faulted=False,
                sent_count=self._sent_count,
                error=self._error,
                axis_count=0,
            )

    def stop(self) -> None:
        """仅停止软件侧下发，不发送物理停止命令。"""

        with self._state_lock:
            if self._stopped:
                return
            self._stopped = True


class Arm:
    """共享控制器上单个机器人 ID 的控制接口，角度单位为度。"""

    def __init__(self, client: RobotClient, robot_id: int) -> None:
        self._client = client
        self.robot_id = robot_id
        self._axis_count: Optional[int] = None
        self._limits: Optional[tuple[tuple[float, float], ...]] = None

    def joint_angles(self) -> JointAngles:
        """读取当前关节角度，并将每个角度四舍五入到小数点后三位。"""

        self._client._require_connected()
        try:
            values = tuple(
                round(float(value), 3)
                for value in self._client._native.joints(self.robot_id)
            )
        except RuntimeError as exc:
            raise _native_error("get joint angles", ConnectionStateError) from exc
        if not values:
            raise ConnectionStateError(
                f"robot {self.robot_id} returned no joint feedback"
            )
        self._axis_count = len(values)
        return values

    def joint_torque_feedback(self) -> tuple[int, ...]:
        """读取各轴原始力矩反馈值，不切换运行模式或下发控制命令。"""

        self._client._require_connected()
        try:
            values = tuple(
                int(value) for value in self._client._native.torque_feedback(self.robot_id)
            )
        except RuntimeError as exc:
            raise _native_error("get joint torque feedback", ConnectionStateError) from exc
        if not values:
            raise ConnectionStateError(
                f"robot {self.robot_id} returned no joint torque feedback"
            )
        if self._axis_count is None:
            self._axis_count = len(values)
        elif len(values) != self._axis_count:
            raise ConnectionStateError(
                f"robot {self.robot_id} returned {len(values)} torque values for "
                f"{self._axis_count} axes"
            )
        return values

    @property
    def axis_count(self) -> int:
        if self._axis_count is None:
            self.joint_angles()
        assert self._axis_count is not None
        return self._axis_count

    @property
    def joint_limits_deg(self) -> tuple[tuple[float, float], ...]:
        """返回控制器配置的各轴角度限位，单位为度。

        每个元素按 ``(min_angle, max_angle)`` 排列。首次读取会从控制器缓存
        限位；后续读取返回同一份不可变快照。
        """

        self._client._require_connected()
        return self._cache_limits()

    def set_enabled(self, enabled: bool) -> None:
        self._client._require_connected()
        try:
            accepted = self._client._native.set_single_enable(
                self.robot_id, bool(enabled)
            )
        except RuntimeError as exc:
            raise _native_error("set robot enable") from exc
        if not accepted:
            raise MotionRejectedError(
                f"controller rejected enable change for robot {self.robot_id}"
            )

    @property
    def enabled(self) -> bool:
        self._client._require_connected()
        try:
            return bool(self._client._native.single_enabled(self.robot_id))
        except RuntimeError as exc:
            raise _native_error("get robot enable") from exc

    @property
    def protection_enabled(self) -> bool:
        self._client._require_connected()
        try:
            return bool(self._client._native.protection_enabled(self.robot_id))
        except RuntimeError as exc:
            raise _native_error("get robot protection") from exc

    def set_protection(self, enabled: bool, *, confirm_unsafe: bool = False) -> None:
        """一对一调用厂商 ``setRobotProtectStatus``。"""

        self._client._require_connected()
        del confirm_unsafe
        try:
            accepted = self._client._native.set_protection(self.robot_id, bool(enabled))
        except RuntimeError as exc:
            raise _native_error("set robot protection") from exc
        if not accepted:
            raise MotionRejectedError(
                f"controller rejected protection change for robot {self.robot_id}"
            )

    def pause(self) -> None:
        self._set_paused(True)

    def resume(self) -> None:
        self._set_paused(False)

    def _set_paused(self, paused: bool) -> None:
        self._client._require_connected()
        try:
            accepted = self._client._native.pause(self.robot_id, paused)
        except RuntimeError as exc:
            raise _native_error("pause robot") from exc
        if not accepted:
            raise MotionRejectedError(
                f"controller rejected pause change for robot {self.robot_id}"
            )

    def clear_route(self, *, emergency_stop: bool = True) -> None:
        self._client._require_connected()
        try:
            accepted = self._client._native.clear_route(
                self.robot_id, bool(emergency_stop)
            )
        except RuntimeError as exc:
            raise _native_error("clear robot route") from exc
        if not accepted:
            raise MotionRejectedError(
                f"controller rejected route clear for robot {self.robot_id}"
            )

    def move_joints(
        self,
        angles_deg: Iterable[float],
        *,
        interrupt: bool = False,
        acceleration_seconds: Optional[float] = None,
        deceleration_seconds: Optional[float] = None,
        speed_ratio: Optional[float] = None,
        smooth: int = 1,
        wait: bool = False,
        timeout_s: Optional[float] = None,
    ) -> MotionHandle:
        """通过厂商控制器规划关节空间运动，并返回对应句柄。"""

        self._client._require_motion_ready(self.robot_id)
        values = self._validate_angles(angles_deg, cached_limits=False)
        acceleration = self._validate_time("acceleration_seconds", acceleration_seconds)
        deceleration = self._validate_time("deceleration_seconds", deceleration_seconds)
        ratio = self._validate_speed_ratio(speed_ratio)
        if not isinstance(smooth, int) or not 0 <= smooth <= 9:
            raise ValueError("smooth must be an integer from 0 through 9")

        try:
            sequence = int(
                self._client._native.move_joints(
                    self.robot_id,
                    values,
                    bool(interrupt),
                    acceleration,
                    deceleration,
                    ratio,
                    smooth,
                )
            )
        except RuntimeError as exc:
            raise MotionRejectedError(str(exc)) from exc

        handle = MotionHandle(self._client, self.robot_id, sequence)
        if wait:
            handle.wait(timeout_s)
        return handle

    def start_direct_servo(
        self,
        *,
        rate_hz: int | None = None,
        watchdog_s: float | None = None,
        confirm_unsafe: bool | None = None,
    ) -> DirectServoSession:
        """返回不带控制策略的 ``PluseToServo`` 兼容调用器。

        ``rate_hz``、``watchdog_s`` 和 ``confirm_unsafe`` 仅为兼容旧调用方而
        保留，不传给厂商 SDK，也不触发任何检查或状态变更。
        """

        del confirm_unsafe
        self._client._require_connected()
        return DirectServoSession(self, rate_hz=rate_hz, watchdog_s=watchdog_s)

    def pluse_to_servo(self, angles_deg: Iterable[float]) -> bool:
        """一对一调用 ``RobotManager::PluseToServo``。

        此方法不增加任何轨迹、插值、限位、看门狗、使能或报警判断。Python 仅
        将输入转换为 ``float`` 序列；厂商 SDK 的 ``bool`` 返回值原样交给调用
        方。关节单位为度。
        """

        self._client._require_connected()
        try:
            values = [float(value) for value in angles_deg]
        except TypeError as exc:
            raise TypeError("angles_deg must be an iterable of degree values") from exc
        return bool(
            self._client._native.pluse_to_servo(self.robot_id, values)
        )

    def pulse_to_servo(self, angles_deg: Iterable[float]) -> bool:
        """``pluse_to_servo`` 的兼容别名。"""

        return self.pluse_to_servo(angles_deg)

    def _validate_angles(
        self, angles_deg: Iterable[float], *, cached_limits: bool
    ) -> tuple[float, ...]:
        try:
            values = tuple(float(value) for value in angles_deg)
        except TypeError as exc:
            raise TypeError(
                "angles_deg must be an iterable of numeric degree values"
            ) from exc
        if len(values) != self.axis_count:
            raise ValueError(
                f"robot {self.robot_id} expects {self.axis_count} joint angles, got {len(values)}"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("joint angles must be finite degree values")

        if cached_limits:
            limits = self._cache_limits()
            for axis_id, (value, (negative, positive)) in enumerate(
                zip(values, limits)
            ):
                if value < negative - 1e-2 or value > positive + 1e-2:
                    raise JointLimitError(
                        f"joint {axis_id} target {value} deg is outside [{negative}, {positive}] deg"
                    )
        else:
            for axis_id, value in enumerate(values):
                try:
                    positive, negative = self._client._native.joint_out_limit(
                        self.robot_id, axis_id, value, 1e-2
                    )
                except RuntimeError as exc:
                    raise _native_error("joint limit check") from exc
                if positive or negative:
                    raise JointLimitError(
                        f"joint {axis_id} target {value} deg is outside the configured limits"
                    )
        return values

    def _cache_limits(self) -> tuple[tuple[float, float], ...]:
        if self._limits is not None:
            return self._limits
        limits: list[tuple[float, float]] = []
        for axis_id in range(self.axis_count):
            try:
                negative, positive = self._client._native.axis_limits(
                    self.robot_id, axis_id
                )
            except RuntimeError as exc:
                raise _native_error("read joint limits") from exc
            negative = float(negative)
            positive = float(positive)
            if (
                not math.isfinite(negative)
                or not math.isfinite(positive)
                or negative >= positive
            ):
                raise JointLimitError(f"joint {axis_id} has invalid configured limits")
            limits.append((negative, positive))
        self._limits = tuple(limits)
        return self._limits

    @staticmethod
    def _validate_time(name: str, value: Optional[float]) -> float:
        if value is None:
            return 0.0
        if not math.isfinite(value) or value == 0 or not 0.1 <= value <= 1.0:
            raise ValueError(
                f"{name} must be None or a value from 0.1 through 1.0 seconds"
            )
        return float(value)

    @staticmethod
    def _validate_speed_ratio(value: Optional[float]) -> float:
        if value is None:
            return 0.0
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("speed_ratio must be None or a value in (0, 1]")
        return float(value)


class _ConnectionAttempt:
    """记录一条可能被厂商库内部重试阻塞的连接请求。"""

    def __init__(self, client: RobotClient) -> None:
        self.client = client
        self.done = threading.Event()
        self.cancel_requested = False
        self.error: Optional[ConnectionStateError] = None


class RobotClient:
    """持有进程级 HCX SDK 连接，并提供各机器人的机械臂接口。"""

    _process_lock = threading.RLock()
    _process_owner: Optional[weakref.ReferenceType[RobotClient]] = None
    _connection_attempt: Optional[_ConnectionAttempt] = None

    def __init__(self, local_ip: str, remote_ip: str, port: int) -> None:
        if not isinstance(port, int) or not 0 < port <= 65535:
            raise ValueError("port must be an integer from 1 through 65535")
        self.local_ip = local_ip
        self.remote_ip = remote_ip
        self.port = port
        self._native = native.NativeSdk()
        self._connected = False
        self._arms: dict[int, Arm] = {}

    def __enter__(self) -> RobotClient:
        return self.connect()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def connected(self) -> bool:
        # 初始化尚未返回时，原生层持有全局锁；此处先检查 Python 状态，
        # 避免状态查询再次进入厂商库而被内部重试一同阻塞。
        return self._connected and bool(self._native.connected)

    @property
    def link_status(self) -> bool:
        """读取厂商控制器当前通信链路状态。"""

        self._require_connected()
        try:
            return bool(self._native.link_status())
        except RuntimeError as exc:
            raise _native_error(
                "get controller link status", ConnectionStateError
            ) from exc

    def connect(self, *, timeout_s: Optional[float] = None) -> RobotClient:
        """连接控制器，并在主线程中等待可被 ``Ctrl+C`` 打断的结果。

        华成底层库在连接失败时可能自行重试且不返回。连接调用因此放在守护
        线程中执行；超时或中断只取消等待，不能安全地强行终止厂商调用。
        """

        timeout_s = _connection_timeout_seconds(timeout_s)
        client_class = self.__class__
        with client_class._process_lock:
            attempt = client_class._connection_attempt
            if attempt is not None:
                if attempt.client is not self or attempt.cancel_requested:
                    raise ConnectionStateError(
                        "HCX SDK 正在后台初始化或清理；请等待其返回，"
                        "或按 Ctrl+C 退出当前 Python 进程"
                    )
            else:
                owner = (
                    client_class._process_owner()
                    if client_class._process_owner is not None
                    else None
                )
                if owner is not None and owner is not self and owner.connected:
                    raise ConnectionStateError(
                        "only one RobotClient may own the static HCX SDK in this process"
                    )
                if self.connected:
                    return self
                attempt = _ConnectionAttempt(self)
                client_class._connection_attempt = attempt
                worker = threading.Thread(
                    target=self._run_connection_attempt,
                    args=(attempt,),
                    name="hcx-sdk-connect",
                    daemon=True,
                )
                try:
                    worker.start()
                except RuntimeError as exc:
                    client_class._connection_attempt = None
                    raise ConnectionStateError("无法启动控制器连接线程") from exc

        return self._wait_for_connection_attempt(attempt, timeout_s)

    def _wait_for_connection_attempt(
        self, attempt: _ConnectionAttempt, timeout_s: Optional[float]
    ) -> RobotClient:
        """在主线程等待连接结果，使信号处理始终有机会运行。"""

        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        try:
            while not attempt.done.is_set():
                wait_s = 0.1
                if deadline is not None:
                    remaining_s = deadline - time.monotonic()
                    if remaining_s <= 0:
                        if self._request_connection_cancellation(attempt):
                            raise ConnectionStateError(
                                f"控制器连接在 {timeout_s:g} 秒内未返回；"
                                "厂商初始化仍在后台重试。请按 Ctrl+C 退出，"
                                "或等待其返回后再重新连接"
                            )
                        continue
                    wait_s = min(wait_s, remaining_s)
                attempt.done.wait(wait_s)
        except KeyboardInterrupt:
            self._request_connection_cancellation(attempt)
            raise

        if attempt.error is not None:
            raise attempt.error
        if self.connected:
            return self
        raise ConnectionStateError("HCX SDK 连接线程未返回有效连接状态")

    def _request_connection_cancellation(self, attempt: _ConnectionAttempt) -> bool:
        """标记连接请求取消；返回值表示请求是否仍在执行。"""

        client_class = self.__class__
        with client_class._process_lock:
            if attempt.done.is_set() or client_class._connection_attempt is not attempt:
                return False
            attempt.cancel_requested = True
            return True

    def _run_connection_attempt(self, attempt: _ConnectionAttempt) -> None:
        """在守护线程执行厂商初始化，并处理延迟返回后的清理。"""

        client_class = self.__class__
        initialized = False
        error: Optional[ConnectionStateError] = None
        try:
            self._native.connect(self.local_ip, self.remote_ip, self.port)
            initialized = True
            with client_class._process_lock:
                cancel_requested = attempt.cancel_requested
            if not cancel_requested and not self._native.link_status():
                error = ConnectionStateError("SDK 已初始化，但控制器链路状态为 false")
        except RuntimeError as exc:
            error = ConnectionStateError(str(exc))
        except Exception as exc:
            error = ConnectionStateError(f"底层 HCX SDK 初始化异常: {exc}")

        with client_class._process_lock:
            cancel_requested = attempt.cancel_requested
            if error is None and initialized and not cancel_requested:
                self._connected = True
                client_class._process_owner = weakref.ref(self)
                self._finish_connection_attempt_locked(attempt, None)
                return

        if initialized:
            try:
                self._native.close()
            except RuntimeError as exc:
                cleanup_error = f"；关闭延迟返回的底层连接失败: {exc}"
                error = (
                    ConnectionStateError(f"{error}{cleanup_error}")
                    if error is not None
                    else ConnectionStateError(
                        "连接已取消，但关闭延迟返回的底层连接失败: "
                        f"{exc}"
                    )
                )
            except Exception as exc:
                cleanup_error = f"；关闭延迟返回的底层连接异常: {exc}"
                error = (
                    ConnectionStateError(f"{error}{cleanup_error}")
                    if error is not None
                    else ConnectionStateError(
                        "连接已取消，但关闭延迟返回的底层连接异常: "
                        f"{exc}"
                    )
                )

        if error is None:
            error = ConnectionStateError(
                "连接等待已取消；底层 SDK 初始化返回后已关闭连接"
            )
        with client_class._process_lock:
            self._connected = False
            self._arms.clear()
            if (
                client_class._process_owner is not None
                and client_class._process_owner() is self
            ):
                client_class._process_owner = None
            self._finish_connection_attempt_locked(attempt, error)

    def _finish_connection_attempt_locked(
        self, attempt: _ConnectionAttempt, error: Optional[ConnectionStateError]
    ) -> None:
        """在进程锁已持有时发布一条连接尝试的最终结果。"""

        client_class = self.__class__
        attempt.error = error
        if client_class._connection_attempt is attempt:
            client_class._connection_attempt = None
        attempt.done.set()

    def close(self) -> None:
        client_class = self.__class__
        with client_class._process_lock:
            attempt = client_class._connection_attempt
            if (
                attempt is not None
                and attempt.client is self
                and not attempt.done.is_set()
            ):
                # init_data() 仍可能持有厂商内部锁，不能在此并发调用 stop_data()。
                attempt.cancel_requested = True
                return
            if not self._connected:
                return
            try:
                self._native.close()
            except RuntimeError as exc:
                raise _native_error("close SDK") from exc
            finally:
                self._connected = False
                self._arms.clear()
                if (
                    client_class._process_owner is not None
                    and client_class._process_owner() is self
                ):
                    client_class._process_owner = None

    def arm(self, robot_id: int) -> Arm:
        self._require_connected()
        if not isinstance(robot_id, int) or robot_id < 0:
            raise ValueError("robot_id must be a non-negative integer")
        return self._arms.setdefault(robot_id, Arm(self, robot_id))

    @property
    def active_alarms(self) -> tuple[str, ...]:
        """读取当前活动报警文本，不清除报警也不修改控制器状态。"""

        self._require_connected()
        try:
            return tuple(str(value) for value in self._native.alarms())
        except RuntimeError as exc:
            raise _native_error("get active alarms", ConnectionStateError) from exc

    def clear_alarms(self) -> None:
        """请求控制器清除当前报警；必须先由现场人员排除报警原因。"""

        self._require_connected()
        try:
            accepted = self._native.clear_alarms()
        except RuntimeError as exc:
            raise _native_error("clear alarms") from exc
        if not accepted:
            raise MotionRejectedError("controller rejected alarm clear request")

    def detach_hmi(self) -> None:
        """请求脱离示教器控制；仅限现场确认未接示教器且已物理拔除时调用。"""

        self._require_connected()
        try:
            accepted = self._native.detach_hmi()
        except RuntimeError as exc:
            raise _native_error("detach HMI") from exc
        if not accepted:
            raise MotionRejectedError("controller rejected HMI detach request")

    @property
    def hmi_detached(self) -> bool:
        """当前控制器是否已处于脱离示教器状态。"""

        self._require_connected()
        try:
            return bool(self._native.hmi_detached())
        except RuntimeError as exc:
            raise _native_error("get HMI detach status", ConnectionStateError) from exc

    @property
    def soft_emergency_stop_normal(self) -> bool:
        """读取软急停状态；为 ``False`` 时表示控制器不在正常可使能状态。"""

        self._require_connected()
        try:
            return bool(self._native.soft_emergency_stop_normal())
        except RuntimeError as exc:
            raise _native_error("get soft emergency-stop status", ConnectionStateError) from exc

    def ethercat_master_operational(self, master_index: int) -> bool:
        """读取 EtherCAT 主站 OP 状态；仅在控制器配置 EtherCAT 伺服时调用。"""

        self._require_connected()
        if (
            not isinstance(master_index, int)
            or isinstance(master_index, bool)
            or not 0 <= master_index <= 1
        ):
            raise ValueError("master_index must be 0 or 1")
        try:
            return bool(self._native.ethercat_master_operational(master_index))
        except RuntimeError as exc:
            raise _native_error("get EtherCAT master OP status", ConnectionStateError) from exc

    def set_global_enable(self, enabled: bool) -> None:
        self._require_connected()
        try:
            accepted = self._native.set_global_enable(bool(enabled))
        except RuntimeError as exc:
            raise _native_error("set global enable") from exc
        if not accepted:
            raise MotionRejectedError("controller rejected global enable change")

    @property
    def global_enabled(self) -> bool:
        self._require_connected()
        try:
            return bool(self._native.global_enabled())
        except RuntimeError as exc:
            raise _native_error("get global enable") from exc

    def _require_connected(self) -> None:
        if not self.connected:
            raise ConnectionStateError("RobotClient is not connected")

    def _require_motion_ready(self, robot_id: int) -> None:
        self._require_connected()
        try:
            if not self._native.link_status():
                raise ConnectionStateError("controller link status is false")
            alarms = tuple(self._native.alarms())
            if alarms:
                raise AlarmActiveError(
                    "controller alarms are active: " + "; ".join(alarms)
                )
            if not self._native.global_enabled():
                raise MotionRejectedError("global robot enable is false")
            if not self._native.single_enabled(robot_id):
                raise MotionRejectedError(f"robot {robot_id} enable is false")
        except HcxSdkError:
            raise
        except RuntimeError as exc:
            raise _native_error("motion readiness", ConnectionStateError) from exc
