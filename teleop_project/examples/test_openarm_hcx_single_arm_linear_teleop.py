#!/usr/bin/env python3
"""最小 OpenArm Mini -> HCX 单臂线性插值直伺服 demo。

本示例只保留一条控制链路：OpenArm Mini 以 100 Hz 读取七轴角度，HCX
机械臂以 500 Hz 调用 ``DirectServoSession.set_target()``。每个相邻的
100 Hz 主臂目标被拆成 5 个等距的线性角度点；主臂没有新样本时，发送线程
持续重复最后一个点，以满足直伺服心跳要求。

线性插值不做低通、限速或限加速度。它用于直接观察“100 Hz 输入点经过线性
重采样后”的效果，不应替代需要机械速度/加速度约束的 limited 遥操作。

为避免启动姿态跳变，映射使用相对七轴模式：第一次有效主臂样本与从臂当前
姿态分别记为零点，后续仅跟随主臂相对变化。除此以外本示例不使用
``TeleopController``、主臂滤波、弹簧阻尼、图窗、自动恢复或
``teleop.yaml``。

运行前必须手动完成示教器脱离、报警处理、EtherCAT 就绪、全局使能和对应
单臂使能。本程序会产生真实运动，只有 ``CONFIRM_DIRECT_SERVO`` 为 ``True``
时才启动 HCX 直接伺服。

运行：

    python -m examples.test_openarm_hcx_single_arm_linear_teleop
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

from teleop_sdk.adapters import (
    HcxConnection,
    HcxConnectionConfig,
    OpenArmMiniLeaderArm,
)
from teleop_sdk.algorithms import LinearInterpolator

JOINT_COUNT = 7
ArmSide = Literal["left", "right"]

# 要测试的同侧组合。修改本文件顶部常量即可切换，不接受命令行参数。
TEST_SIDE: ArmSide = "right"

# OpenArm Mini 的只读串口与双侧组合标定文件。
OPENARM_PORT = "/dev/ttyACM1"
OPENARM_CALIBRATION_PATH = "./my_openarm_mini.json"
OPENARM_BAUDRATE = 1_000_000

# HCX 控制器与本次测试的单臂机器人 ID。根据当前现场探测结果，右臂为
# robot_id=1 / port=12346；切到左臂时改为 robot_id=2 / port=12345。
HCX_LOCAL_IP = "172.16.0.110"
HCX_REMOTE_IP = "172.16.0.89"
HCX_PORT = 12346
HCX_CONNECT_TIMEOUT_S = 10.0
HCX_ROBOT_ID = 1

# OpenArm -> HCX 七轴方向。+1.0 为同向，-1.0 为反向。
AXIS_SIGN = (-1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0)

# 主臂只读采样频率，单位为 Hz。
LEADER_SAMPLE_RATE_HZ = 100.0
# 每次 OpenArm 读取允许的最长等待时间，单位为秒。超时后保持上一次从臂目标。
LEADER_READ_TIMEOUT_S = 0.008

# HCX 直接伺服目标发送频率，单位为 Hz。必须是主臂采样频率的整数倍。
DIRECT_SERVO_RATE_HZ = 500
# 直伺服软件侧看门狗，单位为秒；它允许主臂静止，但要求发送线程持续下发目标。
DIRECT_SERVO_WATCHDOG_S = 2.0
# 真实运动的显式确认。仅在独立急停和硬件防护已确认后设为 True。
CONFIRM_DIRECT_SERVO = True

# 0.0 表示持续运行直到 Ctrl+C；正数表示自动结束的测试时长，单位为秒。
TEST_DURATION_S = 0.0


@dataclass(frozen=True)
class DemoConfig:
    """单臂线性插值遥操作所需的最小配置，所有关节角度均为度。"""

    side: ArmSide
    openarm_port: str
    openarm_calibration_path: str
    openarm_baudrate: int
    hcx_local_ip: str
    hcx_remote_ip: str
    hcx_port: int
    hcx_connect_timeout_s: float
    hcx_robot_id: int
    axis_sign: tuple[float, ...]
    leader_sample_rate_hz: float
    leader_read_timeout_s: float
    direct_servo_rate_hz: int
    direct_servo_watchdog_s: float
    confirm_direct_servo: bool
    test_duration_s: float

    def validate(self) -> None:
        if self.side not in ("left", "right"):
            raise ValueError("side 必须是 left 或 right")
        if not isinstance(self.openarm_port, str) or not self.openarm_port.strip():
            raise ValueError("openarm_port 不能为空")
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
        if (
            not isinstance(self.hcx_robot_id, int)
            or isinstance(self.hcx_robot_id, bool)
            or self.hcx_robot_id < 0
        ):
            raise ValueError("hcx_robot_id 必须是非负整数")
        if len(self.axis_sign) != JOINT_COUNT or any(
            sign not in (-1.0, 1.0) for sign in self.axis_sign
        ):
            raise ValueError(f"axis_sign 必须是 {JOINT_COUNT} 个 +1.0 或 -1.0")
        _positive_finite("leader_sample_rate_hz", self.leader_sample_rate_hz)
        _positive_finite("leader_read_timeout_s", self.leader_read_timeout_s)
        _positive_finite("hcx_connect_timeout_s", self.hcx_connect_timeout_s)
        _positive_finite("direct_servo_watchdog_s", self.direct_servo_watchdog_s)
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
        try:
            LinearInterpolator(
                self.leader_sample_rate_hz,
                self.direct_servo_rate_hz,
            )
        except ValueError as exc:
            raise ValueError(
                "线性插值要求 direct_servo_rate_hz 是 "
                "leader_sample_rate_hz 的整数倍"
            ) from exc


DEMO_CONFIG = DemoConfig(
    side=TEST_SIDE,
    openarm_port=OPENARM_PORT,
    openarm_calibration_path=OPENARM_CALIBRATION_PATH,
    openarm_baudrate=OPENARM_BAUDRATE,
    hcx_local_ip=HCX_LOCAL_IP,
    hcx_remote_ip=HCX_REMOTE_IP,
    hcx_port=HCX_PORT,
    hcx_connect_timeout_s=HCX_CONNECT_TIMEOUT_S,
    hcx_robot_id=HCX_ROBOT_ID,
    axis_sign=AXIS_SIGN,
    leader_sample_rate_hz=LEADER_SAMPLE_RATE_HZ,
    leader_read_timeout_s=LEADER_READ_TIMEOUT_S,
    direct_servo_rate_hz=DIRECT_SERVO_RATE_HZ,
    direct_servo_watchdog_s=DIRECT_SERVO_WATCHDOG_S,
    confirm_direct_servo=CONFIRM_DIRECT_SERVO,
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
    """将当前主臂相对零点的七轴变化映射为安全的从臂度制目标。"""

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
        """返回当前最新目标；写入方不会原地修改已发布数组。"""

        return self._target


class LinearTargetStream:
    """将最新低频目标持续重采样为固定频率的线性发送点。"""

    def __init__(
        self,
        source_rate_hz: int | float,
        output_rate_hz: int | float,
        initial_angles_deg: np.ndarray,
    ) -> None:
        initial = _as_joint_vector("initial_angles_deg", initial_angles_deg)
        self._interpolator = LinearInterpolator(source_rate_hz, output_rate_hz)
        self._last_sent = initial.copy()
        self._segment = np.empty((0, JOINT_COUNT), dtype=float)
        self._segment_index = 0

    @property
    def samples_per_interval(self) -> int:
        return self._interpolator.samples_per_interval

    def step(self, latest_target_deg: np.ndarray) -> np.ndarray:
        """返回下一条高频目标；无新目标时保持最后发送点。"""

        latest = _as_joint_vector("latest_target_deg", latest_target_deg)
        if self._segment_index >= len(self._segment):
            self._segment = np.empty((0, JOINT_COUNT), dtype=float)
            self._segment_index = 0
            if not np.array_equal(latest, self._last_sent):
                self._segment = self._interpolator.interpolate(
                    self._last_sent,
                    latest,
                )

        if self._segment_index < len(self._segment):
            self._last_sent = self._segment[self._segment_index].copy()
            self._segment_index += 1
        return self._last_sent.copy()


def _read_follower_pose_and_limits(
    arm: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """启动前读取一次从臂姿态和关节限位。"""

    follower_origin = _as_joint_vector("HCX 当前关节角度", arm.joint_angles())
    limits = np.asarray(arm.joint_limits_deg, dtype=float)
    if limits.shape != (JOINT_COUNT, 2) or not np.isfinite(limits).all():
        raise RuntimeError("HCX 返回的七轴关节限位无效")
    lower = limits[:, 0].copy()
    upper = limits[:, 1].copy()
    if np.any(lower >= upper):
        raise RuntimeError("HCX 返回的关节限位必须满足 min < max")
    return follower_origin, lower, upper


def _record_worker_failure(
    failures: SimpleQueue[BaseException],
    stop_event: threading.Event,
    exc: BaseException,
) -> None:
    failures.put(exc)
    stop_event.set()


def _run_direct_servo_sender(
    session: Any,
    latest_target: LatestTarget,
    source_rate_hz: float,
    output_rate_hz: int,
    stop_event: threading.Event,
    failures: SimpleQueue[BaseException],
) -> None:
    """独立固定频率发送线程；每周期下发一个线性重采样目标。"""

    period_s = 1.0 / output_rate_hz
    stream = LinearTargetStream(
        source_rate_hz,
        output_rate_hz,
        latest_target.snapshot(),
    )
    next_tick_s = time.monotonic() + period_s
    try:
        while not stop_event.is_set():
            remaining_s = next_tick_s - time.monotonic()
            if remaining_s > 0.0 and stop_event.wait(remaining_s):
                return

            point = stream.step(latest_target.snapshot())
            session.set_target(point.tolist())

            next_tick_s += period_s
            loop_finished_s = time.monotonic()
            if loop_finished_s >= next_tick_s:
                # 不突发补发陈旧点；从当前完成时间起等待一个完整的新周期。
                next_tick_s = loop_finished_s + period_s
    except (RuntimeError, TypeError, ValueError) as exc:
        _record_worker_failure(failures, stop_event, exc)


def _raise_worker_failure(failures: SimpleQueue[BaseException]) -> None:
    try:
        failure = failures.get_nowait()
    except Empty:
        return
    raise RuntimeError(f"HCX 直伺服发送失败: {failure}") from failure


def run_demo(config: DemoConfig) -> None:
    """连接一对同侧主从臂并运行 100 Hz 采样、500 Hz 线性下发。"""

    config.validate()
    leader = OpenArmMiniLeaderArm(
        config.openarm_port,
        Path(config.openarm_calibration_path),
        config.side,
        baudrate=config.openarm_baudrate,
        read_only=True,
    )
    connection = HcxConnection(
        HcxConnectionConfig(
            local_ip=config.hcx_local_ip,
            remote_ip=config.hcx_remote_ip,
            port=config.hcx_port,
            connect_timeout_s=config.hcx_connect_timeout_s,
        )
    )
    arm: Any | None = None
    session: Any | None = None
    sender_thread: threading.Thread | None = None
    stop_event = threading.Event()
    failures: SimpleQueue[BaseException] = SimpleQueue()

    try:
        leader.connect()
        arm = connection.acquire(config.hcx_robot_id)
        # 本示例不自动处理示教器、报警或使能；这里仅确认现场已经可运动。
        connection.prepare_for_motion(config.hcx_robot_id)
        follower_origin, lower, upper = _read_follower_pose_and_limits(arm)
        latest_target = LatestTarget(follower_origin)

        session = arm.start_direct_servo(
            rate_hz=config.direct_servo_rate_hz,
            watchdog_s=config.direct_servo_watchdog_s,
            confirm_unsafe=config.confirm_direct_servo,
        )
        # 首个有效主臂样本到来前，先持续保持从臂当前姿态。
        session.set_target(follower_origin.tolist())
        sender_thread = threading.Thread(
            target=_run_direct_servo_sender,
            args=(
                session,
                latest_target,
                config.leader_sample_rate_hz,
                config.direct_servo_rate_hz,
                stop_event,
                failures,
            ),
            name=f"openarm-hcx-{config.side}-linear-direct-servo",
            daemon=True,
        )
        sender_thread.start()

        samples_per_interval = config.direct_servo_rate_hz // int(
            config.leader_sample_rate_hz
        )
        print("=" * 72)
        print("    OpenArm Mini -> HCX 单臂线性插值直伺服")
        print("=" * 72)
        print(f"  侧别: {config.side}；HCX robot_id: {config.hcx_robot_id}")
        print(f"  主臂采样: {config.leader_sample_rate_hz:.0f} Hz")
        print(f"  HCX set_target: {config.direct_servo_rate_hz} Hz")
        print(
            "  线性插值: "
            f"{config.leader_sample_rate_hz:g} Hz 主臂目标 -> "
            f"{samples_per_interval} 个 "
            f"{config.direct_servo_rate_hz} Hz 等距目标点"
        )
        print(f"  轴方向: {list(config.axis_sign)}")
        print("  不使用低通、限速、限加速度或从臂反馈读取。")
        print("  已先保持从臂当前姿态；收到第一帧主臂角度后开始相对跟随。")
        print("  Ctrl+C 停止软件侧目标下发；独立急停和硬件保护必须可用。")
        print("-" * 72)

        sample_period_s = 1.0 / config.leader_sample_rate_hz
        started_at_s = time.monotonic()
        next_tick_s = started_at_s
        leader_origin: np.ndarray | None = None
        while not stop_event.is_set():
            now_s = time.monotonic()
            if (
                config.test_duration_s > 0.0
                and now_s - started_at_s >= config.test_duration_s
            ):
                break
            _raise_worker_failure(failures)

            leader_angles = leader.read_joint_angles_deg(config.leader_read_timeout_s)
            if leader_angles is not None:
                current = _as_joint_vector("OpenArm 当前关节角度", leader_angles)
                if leader_origin is None:
                    leader_origin = current.copy()
                    print(
                        "[INFO] 主臂起始位置已记录: "
                        f"{np.round(leader_origin, 1).tolist()}"
                    )
                    print("[INFO] 现在移动主臂，从臂将跟随相对变化量")
                latest_target.publish(
                    map_relative_target(
                        current,
                        leader_origin,
                        follower_origin,
                        config.axis_sign,
                        lower,
                        upper,
                    )
                )

            next_tick_s += sample_period_s
            remaining_s = next_tick_s - time.monotonic()
            if remaining_s > 0.0:
                stop_event.wait(remaining_s)
            else:
                next_tick_s = time.monotonic()

        _raise_worker_failure(failures)
    finally:
        stop_event.set()
        if sender_thread is not None:
            sender_thread.join(timeout=max(1.0, 2.0 / config.direct_servo_rate_hz))
            if sender_thread.is_alive():
                print("[WARN] 直伺服发送线程未在预期时间内退出")
        if session is not None:
            try:
                session.stop()
            except (RuntimeError, TypeError, ValueError) as exc:
                print(f"[WARN] 停止 HCX 直伺服会话失败: {exc}")
        if arm is not None:
            try:
                connection.release(config.hcx_robot_id)
            except (RuntimeError, TypeError, ValueError) as exc:
                print(f"[WARN] 释放 HCX 连接失败: {exc}")
        leader.disconnect()


def main() -> int:
    """执行文件顶部定义的单臂线性插值硬件 demo。"""

    try:
        run_demo(DEMO_CONFIG)
    except KeyboardInterrupt:
        print("\n[STOP] 收到退出请求，正在停止 HCX 单臂直伺服下发。")
        return 130
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f"[ERROR] 单臂线性插值直伺服失败: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
