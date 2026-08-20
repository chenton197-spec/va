#!/usr/bin/env python3
"""使用双 OpenArm Mini 主臂遥操作 HCX 双七轴机械臂和可选 Gloria-M 夹爪。

HCX 七轴关节由独立直伺服输出线程控制。启用 Gloria-M 时，每侧夹爪仅从已由
关节控制链读取的 OpenArm 缓存中获取开合量，并在自己的低频线程中发送，绝不
进入 HCX 的 500 Hz 直伺服发送路径。OpenArm Mini 以只读方式连接，HCX 的
示教器、报警和使能前置流程由根目录 ``teleop.yaml`` 中的 ``hcx.auto_*``
显式授权项决定。HCX 使用 ``PluseToServo`` 直接关节伺服，必须另行显式确认
危险操作。
"""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from teleop_sdk import TeleopController
from teleop_sdk.adapters import (
    GloriaMGripperFollower,
    HcxConnection,
    HcxConnectionConfig,
    HcxDirectServoConfig,
    HcxFollower,
    OpenArmMiniLeaderArm,
)
from teleop_sdk.config import (
    GloriaMDualGripperConfig,
    GloriaMGripperConfig,
    HcxConfig,
    OpenArmMiniLeaderConfig,
    TeleopConfig,
    load_runtime_config,
)

_AXIS_COUNT = 7
ARM_AXIS_ORDER = tuple(range(_AXIS_COUNT))
HCX_FEEDBACK_RATE_HZ = 30.0
ArmSide = Literal["left", "right"]


class HcxFeedbackPoller:
    """在独立线程中读取一侧 HCX 实际关节反馈。"""

    _STOP_JOIN_TIMEOUT_S = 1.0

    def __init__(
        self,
        side: str,
        follower: HcxFollower,
        rate_hz: float,
    ) -> None:
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("HCX feedback rate must be positive and finite")
        self._side = side
        self._follower = follower
        self._period_s = 1.0 / rate_hz
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_angles_deg: np.ndarray | None = None
        self._latest_error: str | None = None
        self._started = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"hcx-{side}-feedback",
            daemon=True,
        )

    def start(self) -> None:
        """启动本侧低频反馈采样。"""

        if self._started:
            raise RuntimeError(f"HCX {self._side} feedback poller is already started")
        self._started = True
        self._thread.start()

    def stop(self) -> bool:
        """请求停止；反馈读取卡住时不阻塞直伺服关闭。"""

        self._stop_event.set()
        if not self._started:
            return True
        self._thread.join(timeout=self._STOP_JOIN_TIMEOUT_S)
        return not self._thread.is_alive()

    def latest(self) -> tuple[np.ndarray | None, str | None]:
        """返回最近一次反馈快照；不调用 HCX SDK。"""

        with self._lock:
            angles_deg = (
                None
                if self._latest_angles_deg is None
                else self._latest_angles_deg.copy()
            )
            return angles_deg, self._latest_error

    def _run(self) -> None:
        next_tick_s = time.monotonic()
        while not self._stop_event.is_set():
            wait_s = next_tick_s - time.monotonic()
            if wait_s > 0.0 and self._stop_event.wait(wait_s):
                return

            try:
                angles_deg = np.asarray(
                    self._follower.read_joint_angles_deg(), dtype=float
                )
                if (
                    angles_deg.shape != (_AXIS_COUNT,)
                    or not np.isfinite(angles_deg).all()
                ):
                    raise RuntimeError("HCX returned invalid joint feedback")
                error = None
            except Exception as exc:
                angles_deg = None
                error = f"{type(exc).__name__}: {exc}"

            with self._lock:
                self._latest_angles_deg = (
                    None if angles_deg is None else angles_deg.copy()
                )
                self._latest_error = error

            # 反馈过慢时跳到下一周期，而不是补读历史样本，避免形成读取突发。
            next_tick_s += self._period_s
            if time.monotonic() >= next_tick_s:
                next_tick_s = time.monotonic() + self._period_s


class HcxFeedbackConsole:
    """显示两侧最新反馈快照，不在采样线程中打印。"""

    _REPORT_PERIOD_S = 1.0

    def __init__(
        self,
        left: HcxFeedbackPoller,
        right: HcxFeedbackPoller,
    ) -> None:
        self._pollers = {"left": left, "right": right}
        self._next_report_s = time.monotonic() + self._REPORT_PERIOD_S

    def refresh(self) -> None:
        """每秒打印一次本地缓存的双臂角度，不读取 HCX SDK。"""

        now_s = time.monotonic()
        if now_s < self._next_report_s:
            return
        self._next_report_s = now_s + self._REPORT_PERIOD_S
        left = self._format_side("left", "L")
        right = self._format_side("right", "R")
        print(f"[HCX Feedback] {left} | {right}")

    def _format_side(self, side: str, label: str) -> str:
        angles_deg, error = self._pollers[side].latest()
        if angles_deg is None:
            return f"{label} current(deg)=unavailable ({error or 'waiting'})"
        values = [round(float(value), 1) for value in angles_deg]
        return f"{label} current(deg)={values}"


@dataclass
class _GloriaGripperWorkerStats:
    """夹爪线程的低频状态；不参与任何 HCX 控制决策。"""

    samples: int = 0
    unavailable: int = 0
    send_failures: int = 0
    last_target: float | None = None


class GloriaGripperWorker:
    """在独立低频线程中将一侧 OpenArm 缓存映射到一只 Gloria-M。"""

    _STOP_JOIN_TIMEOUT_S = 1.0

    def __init__(
        self,
        side: ArmSide,
        leader: OpenArmMiniLeaderArm,
        gripper: GloriaMGripperFollower,
        *,
        rate_hz: float,
        status_print_interval_s: float,
    ) -> None:
        self.side = side
        self._leader = leader
        self._gripper = gripper
        self._period_s = 1.0 / _positive_finite("gloria_m_dual.rate_hz", rate_hz)
        self._status_print_interval_s = _positive_finite(
            "gloria_m_dual.status_print_interval_s", status_print_interval_s
        )
        self._stop_event = threading.Event()
        self._closed_lock = threading.Lock()
        self._closed = False
        self._started = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"openarm-gloria-{side}",
            daemon=True,
        )

    def start(self) -> None:
        """启动夹爪线程；不会读取 OpenArm 串口或调用 HCX SDK。"""

        if self._started:
            raise RuntimeError(f"Gloria-M {self.side} worker is already started")
        self._started = True
        self._thread.start()

    def request_stop(self) -> None:
        """请求退出，供主线程先停止夹爪发送再关闭 HCX。"""

        self._stop_event.set()

    def close(self) -> bool:
        """停止本侧线程并关闭本侧夹爪，不阻塞 HCX 直伺服关闭。"""

        self.request_stop()
        if self._started:
            self._thread.join(timeout=self._STOP_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                return False
        self._disable_and_disconnect()
        return True

    def _run(self) -> None:
        next_deadline_s = time.perf_counter()
        next_status_s = next_deadline_s + self._status_print_interval_s
        stats = _GloriaGripperWorkerStats()

        try:
            while not self._stop_event.is_set():
                opening = self._leader.read_cached_gripper_opening()
                target = _clamp_opening(opening)
                stats.samples += 1
                if target is None:
                    stats.unavailable += 1
                else:
                    stats.last_target = target
                    if not self._gripper.send_normalized(target):
                        stats.send_failures += 1

                now_s = time.perf_counter()
                if now_s >= next_status_s:
                    target_text = (
                        "unavailable"
                        if stats.last_target is None
                        else f"{stats.last_target:.3f}"
                    )
                    # print(
                    #     f"[GLORIA {self.side.upper()}] cached-target={target_text} "
                    #     f"samples={stats.samples} unavailable={stats.unavailable} "
                    #     f"send-failed={stats.send_failures}",
                    #     flush=True,
                    # )
                    next_status_s = now_s + self._status_print_interval_s

                next_deadline_s += self._period_s
                wait_s = next_deadline_s - time.perf_counter()
                if wait_s > 0.0:
                    self._stop_event.wait(wait_s)
                else:
                    # 夹爪发送变慢时丢弃历史周期，不追赶而形成串口突发。
                    next_deadline_s = time.perf_counter()
        except Exception as exc:
            # 夹爪链路失败只停止本侧夹爪。HCX 直伺服和另一侧夹爪继续运行。
            print(
                f"[WARN] Gloria-M {self.side} 夹爪线程已停止，不影响 HCX 双臂遥操: "
                f"{type(exc).__name__}: {exc}"
            )
            self._disable_and_disconnect()

    def _disable_and_disconnect(self) -> None:
        """幂等地关闭本侧夹爪，避免失败线程持续保持最后力矩。"""

        with self._closed_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._gripper.disable()
        except Exception as exc:
            print(f"[WARN] Gloria-M {self.side} 夹爪失能失败: {exc}")
        try:
            self._gripper.disconnect()
        except Exception as exc:
            print(f"[WARN] Gloria-M {self.side} 夹爪断开失败: {exc}")


def _validate_openarm_config(config: OpenArmMiniLeaderConfig) -> None:
    """在打开串口前检查双 OpenArm Mini 的只读连接配置。"""

    if (
        not isinstance(config.port_left, str)
        or not isinstance(config.port_right, str)
        or not config.port_left.strip()
        or not config.port_right.strip()
    ):
        raise ValueError("openarm_mini.port_left 和 port_right 均不能为空")
    if config.port_left == config.port_right:
        raise ValueError("openarm_mini.port_left 和 port_right 必须是两条不同串口")
    if (
        not isinstance(config.calibration_path, str)
        or not config.calibration_path.strip()
    ):
        raise ValueError("请在 teleop.yaml 设置 openarm_mini.calibration_path")
    if not isinstance(config.baudrate, int) or config.baudrate <= 0:
        raise ValueError("openarm_mini.baudrate 必须为正整数")
    if not Path(config.calibration_path).expanduser().is_file():
        raise ValueError(
            f"找不到 OpenArm Mini 标定文件: {config.calibration_path}；请先运行标定示例"
        )


def _validate_hcx_config(config: HcxConfig) -> None:
    """在创建 HCX 网络连接前检查部署地址与双臂标识。"""

    if not isinstance(config.local_ip, str) or not config.local_ip.strip():
        raise ValueError("hcx.local_ip 不能为空")
    if not isinstance(config.remote_ip, str) or not config.remote_ip.strip():
        raise ValueError("hcx.remote_ip 不能为空")
    if (
        not isinstance(config.port, int)
        or isinstance(config.port, bool)
        or not 1 <= config.port <= 65535
    ):
        raise ValueError("hcx.port 必须是 1 到 65535 的整数")
    if config.connect_timeout_s is not None:
        if isinstance(config.connect_timeout_s, bool):
            raise ValueError("hcx.connect_timeout_s 必须是正的有限秒数或 null")
        try:
            timeout_s = float(config.connect_timeout_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("hcx.connect_timeout_s 必须是正的有限秒数或 null") from exc
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("hcx.connect_timeout_s 必须是正的有限秒数或 null")

    for name, robot_id in (
        ("hcx.left_robot_id", config.left_robot_id),
        ("hcx.right_robot_id", config.right_robot_id),
    ):
        if not isinstance(robot_id, int) or isinstance(robot_id, bool) or robot_id < 0:
            raise ValueError(f"{name} 必须是非负整数")
    if config.left_robot_id == config.right_robot_id:
        raise ValueError("hcx.left_robot_id 与 hcx.right_robot_id 必须不同")
    _validate_axis_sign("hcx.left_axis_sign", config.left_axis_sign)
    _validate_axis_sign("hcx.right_axis_sign", config.right_axis_sign)


def _validate_axis_sign(name: str, axis_sign: object) -> None:
    """确认 YAML 中的单侧七轴方向只包含有限的 +1 或 -1。"""

    if not isinstance(axis_sign, tuple) or len(axis_sign) != _AXIS_COUNT:
        raise ValueError(f"{name} 必须是包含 {_AXIS_COUNT} 个元素的方向数组")
    for sign in axis_sign:
        if isinstance(sign, bool):
            raise ValueError(f"{name} 只能包含 +1.0 或 -1.0")
        try:
            parsed_sign = float(sign)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 只能包含 +1.0 或 -1.0") from exc
        if not math.isfinite(parsed_sign) or parsed_sign not in (-1.0, 1.0):
            raise ValueError(f"{name} 只能包含 +1.0 或 -1.0")


def _positive_finite(name: str, value: object) -> float:
    """验证一个供独立工作线程使用的正有限频率或时间。"""

    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是正的有限数")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是正的有限数") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} 必须是正的有限数")
    return numeric


def _clamp_opening(value: object) -> float | None:
    """将 OpenArm 的缓存夹爪值归一化限制在 0 到 1。"""

    try:
        opening = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(opening):
        return None
    return max(0.0, min(1.0, opening))


def _enabled_gloria_sides(config: GloriaMDualGripperConfig) -> tuple[ArmSide, ...]:
    """返回明确启用的 Gloria-M 侧别；空集合表示仅运行 HCX 手臂。"""

    sides: list[ArmSide] = []
    if config.left.enabled:
        sides.append("left")
    if config.right.enabled:
        sides.append("right")
    return tuple(sides)


def _validate_gloria_config(
    openarm_config: OpenArmMiniLeaderConfig,
    config: GloriaMDualGripperConfig,
) -> tuple[ArmSide, ...]:
    """验证已启用夹爪的独立串口和低频工作线程参数。"""

    enabled_sides = _enabled_gloria_sides(config)
    if not enabled_sides:
        return enabled_sides

    _positive_finite("gloria_m_dual.rate_hz", config.rate_hz)
    _positive_finite(
        "gloria_m_dual.status_print_interval_s", config.status_print_interval_s
    )
    used_ports = {
        openarm_config.port_left: "openarm_mini.port_left",
        openarm_config.port_right: "openarm_mini.port_right",
    }
    for side in enabled_sides:
        side_config: GloriaMGripperConfig = config.side_config(side)
        port = side_config.port
        if not isinstance(port, str) or not port.strip():
            raise ValueError(f"gloria_m_dual.{side}.port 不能为空")
        if port.lower() == "auto":
            raise ValueError(
                f"gloria_m_dual.{side}.port 不能为 auto；双臂遥操必须使用明确串口"
            )
        if (
            not isinstance(side_config.baudrate, int)
            or isinstance(side_config.baudrate, bool)
            or side_config.baudrate <= 0
        ):
            raise ValueError(f"gloria_m_dual.{side}.baudrate 必须为正整数")
        duplicate = used_ports.get(port)
        if duplicate is not None:
            raise ValueError(
                f"gloria_m_dual.{side}.port 与 {duplicate} 不能使用同一串口"
            )
        used_ports[port] = f"gloria_m_dual.{side}.port"
    return enabled_sides


def _validated_rate_hz(config: TeleopConfig) -> float:
    """返回正的有限控制频率，避免在设备连接后才暴露配置错误。"""

    if isinstance(config.rate_hz, bool):
        raise ValueError("teleop.rate_hz 必须是正的有限数")
    try:
        rate_hz = float(config.rate_hz)
    except (TypeError, ValueError) as exc:
        raise ValueError("teleop.rate_hz 必须是正的有限数") from exc
    if not math.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("teleop.rate_hz 必须是正的有限数")
    return rate_hz


def _direct_servo_config(
    config: HcxConfig, control_rate_hz: float
) -> HcxDirectServoConfig:
    """构造本 demo 的直伺服配置，并在连接硬件前验证安全授权。"""

    source_rate_hz: int | None = None
    if config.direct_servo_interpolation in ("linear", "limited"):
        if not control_rate_hz.is_integer():
            raise ValueError(
                "hcx.direct_servo_interpolation: linear/limited 要求 "
                "teleop.rate_hz 为整数"
            )
        source_rate_hz = int(control_rate_hz)
    try:
        direct_config = HcxDirectServoConfig.from_runtime_config(
            config, source_rate_hz=source_rate_hz
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"HCX 直伺服配置无效: {exc}") from exc
    if not direct_config.confirm_unsafe:
        raise ValueError(
            "OpenArm -> HCX 直伺服要求在 teleop.yaml 明确设置 "
            "hcx.direct_servo_confirm_unsafe: true"
        )
    if direct_config.watchdog_s <= 1.0 / direct_config.rate_hz:
        raise ValueError("hcx.direct_servo_watchdog_s 必须大于一个 HCX 直伺服输出周期")
    return direct_config


def _control_config_for_side(
    base_config: TeleopConfig, axis_sign: tuple[float, ...]
) -> TeleopConfig:
    """保留通用滤波参数，仅覆盖本 demo 的七轴相对映射。"""

    _validate_axis_sign("HCX 轴方向", axis_sign)
    return replace(
        base_config,
        axis_order=ARM_AXIS_ORDER,
        axis_sign=axis_sign,
        relative_mode=True,
    )


def _create_controller(
    *,
    port: str,
    side: str,
    openarm_config: OpenArmMiniLeaderConfig,
    follower: HcxFollower,
    teleop_config: TeleopConfig,
    axis_sign: tuple[float, ...],
) -> TeleopController:
    """创建一侧只读主臂和 HCX 七轴从臂，不将夹爪接入控制器。"""

    leader = OpenArmMiniLeaderArm(
        port=port,
        calibration_path=openarm_config.calibration_path,
        side=side,
        baudrate=openarm_config.baudrate,
        read_only=True,
    )
    return TeleopController(
        leader,
        follower,
        _control_config_for_side(teleop_config, axis_sign),
    )


def _create_gloria_workers(
    config: GloriaMDualGripperConfig,
    enabled_sides: tuple[ArmSide, ...],
    leaders: dict[ArmSide, OpenArmMiniLeaderArm],
) -> list[GloriaGripperWorker]:
    """连接可选夹爪，但不让夹爪故障阻断 HCX 双臂遥操。"""

    workers: list[GloriaGripperWorker] = []
    for side in enabled_sides:
        gripper: GloriaMGripperFollower | None = None
        try:
            leader = leaders[side]
            cached_read = getattr(leader, "read_cached_gripper_opening", None)
            if not callable(cached_read):
                raise RuntimeError("OpenArm Mini 适配器缺少夹爪缓存读取接口")
            gripper = GloriaMGripperFollower(config.side_config(side))
            gripper.connect()
            workers.append(
                GloriaGripperWorker(
                    side,
                    leader,
                    gripper,
                    rate_hz=float(config.rate_hz),
                    status_print_interval_s=float(config.status_print_interval_s),
                )
            )
        except Exception as exc:
            print(
                f"[WARN] Gloria-M {side} 夹爪不可用，本次仅继续 HCX 双臂遥操: "
                f"{type(exc).__name__}: {exc}"
            )
            if gripper is not None:
                try:
                    gripper.disable()
                except Exception:
                    pass
                try:
                    gripper.disconnect()
                except Exception:
                    pass
    return workers


def _run_control_loop(
    left_controller: TeleopController,
    right_controller: TeleopController,
    rate_hz: float,
    feedback_console: HcxFeedbackConsole | None = None,
) -> None:
    """并行读取双侧主臂并在同一调度周期驱动两条控制链。"""

    period_s = 1.0 / rate_hz
    next_deadline = time.perf_counter()
    next_overrun_report_at = 0.0
    # 两侧 OpenArm Mini 使用独立串口。并行执行避免一侧同步读取的偶发等待
    # 阻塞另一侧主臂样本；HCX 两侧 Python 输出线程也独立调用薄原生绑定。
    with ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="openarm-hcx-control"
    ) as executor:
        while True:
            timestamp = time.perf_counter()
            left_step = executor.submit(left_controller.step, timestamp)
            right_step = executor.submit(right_controller.step, timestamp)
            left_step.result()
            right_step.result()
            if feedback_console is not None:
                feedback_console.refresh()

            next_deadline += period_s
            finished_at = time.perf_counter()
            delay_s = next_deadline - finished_at
            if delay_s > 0.0:
                time.sleep(delay_s)
                continue

            # 只限频输出，避免诊断输出本身再次拖慢 100 Hz 控制循环。
            if timestamp >= next_overrun_report_at:
                print(
                    "[WARN] 双臂主控周期超期: "
                    f"耗时 {-delay_s * 1000.0:.1f} ms，目标周期 "
                    f"{period_s * 1000.0:.1f} ms"
                )
                next_overrun_report_at = timestamp + 1.0
            next_deadline = finished_at


def _print_started_message(
    rate_hz: float,
    direct_config: HcxDirectServoConfig,
    left_axis_sign: tuple[float, ...],
    right_axis_sign: tuple[float, ...],
    gloria_sides: tuple[ArmSide, ...],
) -> None:
    """输出双臂遥操和可选独立夹爪链路的关键约束。"""

    print("=" * 60)
    print("    OpenArm Mini -> HCX 双臂遥操作")
    print("=" * 60)
    print("  映射模式: 相对同序七轴，左右轴方向由独立符号数组决定")
    print(f"  左臂轴方向: {list(left_axis_sign)}")
    print(f"  右臂轴方向: {list(right_axis_sign)}")
    print(f"  控制频率: {rate_hz:.1f} Hz")
    print(
        "  HCX 直伺服: "
        f"{direct_config.rate_hz} Hz，watchdog={direct_config.watchdog_s:.3f} s"
    )
    if direct_config.interpolation == "linear":
        print(
            "  插值: "
            f"{direct_config.source_rate_hz} Hz 主臂目标 -> "
            f"{direct_config.rate_hz} Hz Python 线性队列"
        )
    elif direct_config.interpolation == "limited":
        print(
            "  限幅批次: "
            f"{direct_config.source_rate_hz} Hz 主臂目标 -> "
            f"{direct_config.rate_hz} Hz 预生成低通/限速/限加速度点"
        )
    else:
        print("  插值: direct（不生成插值队列，Python 线程重发当前目标）")
    if gloria_sides:
        print(
            "  Gloria-M 夹爪: "
            f"{', '.join(side.upper() for side in gloria_sides)} 独立低频线程；"
            "只读取 OpenArm 缓存，不进入 HCX 直伺服路径。"
        )
    else:
        print("  Gloria-M 夹爪: 未启用。")
    print("  OpenArm Mini 只读连接。")
    print("  HCX 状态变更仅由 teleop.yaml 中显式启用的 hcx.auto_* 执行。")
    print("  Ctrl+C 仅停止 SDK 软件侧目标重发；独立急停和硬件保护必须可用。")
    print("  请先确认双臂工作空间、关节方向和安全回路；按 Ctrl+C 安全退出。")
    print("-" * 60)


def main() -> int:
    """装配双主臂、共享 HCX 双臂从端并运行同步遥操作循环。"""

    try:
        runtime = load_runtime_config()
        _validate_openarm_config(runtime.openarm_mini)
        _validate_hcx_config(runtime.hcx)
        gloria_config = getattr(runtime, "gloria_m_dual", GloriaMDualGripperConfig())
        if not isinstance(gloria_config, GloriaMDualGripperConfig):
            raise ValueError("gloria_m_dual 配置必须是 GloriaMDualGripperConfig")
        gloria_enabled_sides = _validate_gloria_config(
            runtime.openarm_mini, gloria_config
        )
        rate_hz = _validated_rate_hz(runtime.teleop)
        direct_config = _direct_servo_config(runtime.hcx, rate_hz)
        connection = HcxConnection(HcxConnectionConfig.from_runtime_config(runtime.hcx))
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f"[ERROR] OpenArm Mini -> HCX 配置无效: {exc}")
        return 2

    controllers: list[TeleopController] = []
    feedback_pollers: list[HcxFeedbackPoller] = []
    gripper_workers: list[GloriaGripperWorker] = []
    try:
        left_follower = HcxFollower(
            connection,
            robot_id=runtime.hcx.left_robot_id,
            side="left",
            direct_servo_config=direct_config,
        )
        right_follower = HcxFollower(
            connection,
            robot_id=runtime.hcx.right_robot_id,
            side="right",
            direct_servo_config=direct_config,
        )
        left_controller = _create_controller(
            port=runtime.openarm_mini.port_left,
            side="left",
            openarm_config=runtime.openarm_mini,
            follower=left_follower,
            teleop_config=runtime.teleop,
            axis_sign=runtime.hcx.left_axis_sign,
        )
        right_controller = _create_controller(
            port=runtime.openarm_mini.port_right,
            side="right",
            openarm_config=runtime.openarm_mini,
            follower=right_follower,
            teleop_config=runtime.teleop,
            axis_sign=runtime.hcx.right_axis_sign,
        )
        controllers = [left_controller, right_controller]

        for controller in controllers:
            controller.connect()
        # Gloria-M 的连接在 HCX 直伺服启动前完成。后续 worker 只读取已缓存
        # 的 OpenArm 开合量，不会在 500 Hz HCX 发送路径上访问任何夹爪资源。
        gripper_workers = _create_gloria_workers(
            gloria_config,
            gloria_enabled_sides,
            {
                "left": left_controller.leader,
                "right": right_controller.leader,
            },
        )
        if not left_controller.start_servo():
            raise RuntimeError("HCX 左臂未通过启动前置检查")
        if not right_controller.start_servo():
            raise RuntimeError("HCX 右臂未通过启动前置检查")
        for worker in gripper_workers:
            worker.start()

        feedback_pollers = [
            HcxFeedbackPoller("left", left_follower, HCX_FEEDBACK_RATE_HZ),
            HcxFeedbackPoller("right", right_follower, HCX_FEEDBACK_RATE_HZ),
        ]
        for poller in feedback_pollers:
            poller.start()
        feedback_console = HcxFeedbackConsole(feedback_pollers[0], feedback_pollers[1])

        _print_started_message(
            rate_hz,
            direct_config,
            runtime.hcx.left_axis_sign,
            runtime.hcx.right_axis_sign,
            tuple(worker.side for worker in gripper_workers),
        )
        print(
            "  平滑组: "
            f"滤波={'启用' if runtime.teleop.filter_enabled else '关闭'}，"
            f"弹簧/前瞻={'启用' if runtime.teleop.spring_enabled else '关闭'}"
        )
        print("  HCX 反馈: 左右臂各自 30 Hz 独立采样；终端每秒显示一次最新关节角度。")
        _run_control_loop(
            left_controller,
            right_controller,
            rate_hz,
            feedback_console=feedback_console,
        )
    except KeyboardInterrupt:
        print("\n[STOP] 收到退出请求，正在停止 HCX 双臂直伺服软件下发。")
    except Exception as exc:
        print(f"[ERROR] OpenArm Mini -> HCX 遥操异常: {exc}")
        return 1
    finally:
        # 先通知夹爪线程停止，但不等待它完成；若夹爪串口卡住，HCX 直伺服仍可
        # 立即按原顺序关闭，避免低频外设拖延双臂停止。
        for worker in gripper_workers:
            worker.request_stop()
        for side, poller in zip(("left", "right"), feedback_pollers):
            if not poller.stop():
                print(f"[WARN] HCX {side} 反馈线程未在超时内退出")
        for controller in reversed(controllers):
            try:
                controller.shutdown()
            except Exception as exc:
                print(f"[WARN] 关闭一侧 OpenArm Mini -> HCX 控制链失败: {exc}")
        for worker in reversed(gripper_workers):
            if not worker.close():
                print(
                    f"[WARN] Gloria-M {worker.side} 夹爪线程未在超时内退出；"
                    "未在主线程强制断开该夹爪"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
