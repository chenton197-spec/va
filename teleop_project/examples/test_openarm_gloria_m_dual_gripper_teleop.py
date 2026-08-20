#!/usr/bin/env python3
"""OpenArm Mini 双侧示教夹爪控制 Gloria-M 双侧从端夹爪。

左侧 OpenArm Mini 只控制左侧 Gloria-M，右侧同理。可按配置单独启用任意一侧，
开合量统一为 0.0（闭合）到 1.0（张开）；本示例不会连接 HCX，也不会发送任何
机械臂关节命令。

运行前在 ``teleop.yaml`` 的 ``gloria_m_dual`` 段分别填写两只 Gloria-M 的串口，
确认接线后将需要运行的 ``left.enabled`` 和/或 ``right.enabled`` 设为 ``true``：

    python -m examples.test_openarm_gloria_m_dual_gripper_teleop
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Literal

from teleop_sdk.adapters import GloriaMGripperFollower, OpenArmMiniLeaderArm
from teleop_sdk.config import (
    GloriaMDualGripperConfig,
    GloriaMGripperConfig,
    OpenArmMiniLeaderConfig,
    load_runtime_config,
)
from teleop_sdk.interfaces import GripperActuator, LeaderGripperInput

ArmSide = Literal["left", "right"]


@dataclass(frozen=True)
class _SideDevices:
    """一侧独立的示教夹爪输入与从端夹爪输出。"""

    side: ArmSide
    leader: LeaderGripperInput
    gripper: GripperActuator


@dataclass
class _SideStats:
    """仅用于低频终端状态，不参与夹爪控制。"""

    samples: int = 0
    unavailable_reads: int = 0
    send_failures: int = 0
    last_target: float | None = None


def _positive_finite(name: str, value: object) -> float:
    """验证配置中的正有限浮点数。"""

    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是正的有限数")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是正的有限数") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} 必须是正的有限数")
    return numeric


def _enabled_sides(config: GloriaMDualGripperConfig) -> tuple[ArmSide, ...]:
    """返回本次运行中明确启用的夹爪侧别。"""

    sides: list[ArmSide] = []
    if config.left.enabled:
        sides.append("left")
    if config.right.enabled:
        sides.append("right")
    if not sides:
        raise ValueError(
            "请至少将 gloria_m_dual.left.enabled 或 right.enabled 设为 true"
        )
    return tuple(sides)


def _openarm_port(config: OpenArmMiniLeaderConfig, side: ArmSide) -> str:
    """返回一侧 OpenArm Mini 对应的串口。"""

    return config.port_left if side == "left" else config.port_right


def _gloria_side_config(
    config: GloriaMDualGripperConfig, side: ArmSide
) -> GloriaMGripperConfig:
    """返回一侧 Gloria-M 的部署配置。"""

    return config.left if side == "left" else config.right


def _validate_openarm_config(
    config: OpenArmMiniLeaderConfig, enabled_sides: tuple[ArmSide, ...]
) -> None:
    """在打开已启用一侧的 OpenArm Mini 串口前验证只读配置。"""

    for side in enabled_sides:
        port = _openarm_port(config, side)
        if not isinstance(port, str) or not port.strip():
            raise ValueError(f"openarm_mini.port_{side} 不能为空")
    if len(enabled_sides) == 2 and config.port_left == config.port_right:
        raise ValueError("启用双侧时 openarm_mini.port_left 和 port_right 必须不同")
    if (
        not isinstance(config.calibration_path, str)
        or not config.calibration_path.strip()
    ):
        raise ValueError("请在 teleop.yaml 设置 openarm_mini.calibration_path")
    if not Path(config.calibration_path).expanduser().is_file():
        raise ValueError(
            f"找不到 OpenArm Mini 标定文件: {config.calibration_path}；请先运行标定示例"
        )
    if not isinstance(config.baudrate, int) or isinstance(config.baudrate, bool):
        raise ValueError("openarm_mini.baudrate 必须为正整数")
    if config.baudrate <= 0:
        raise ValueError("openarm_mini.baudrate 必须为正整数")


def _validate_gloria_side(side: ArmSide, config: GloriaMGripperConfig) -> None:
    """验证一只 Gloria-M 的明确部署参数。"""

    if not isinstance(config.port, str) or not config.port.strip():
        raise ValueError(f"gloria_m_dual.{side}.port 不能为空")
    if config.port.lower() == "auto":
        raise ValueError(
            f"gloria_m_dual.{side}.port 不能为 auto；双夹爪必须使用明确且不同的串口"
        )
    if not isinstance(config.baudrate, int) or isinstance(config.baudrate, bool):
        raise ValueError(f"gloria_m_dual.{side}.baudrate 必须为正整数")
    if config.baudrate <= 0:
        raise ValueError(f"gloria_m_dual.{side}.baudrate 必须为正整数")


def _validate_config(
    openarm: OpenArmMiniLeaderConfig,
    gloria: GloriaMDualGripperConfig,
) -> tuple[ArmSide, ...]:
    """验证已启用的 OpenArm 与 Gloria-M 侧别，并返回其顺序。"""

    enabled_sides = _enabled_sides(gloria)
    _validate_openarm_config(openarm, enabled_sides)
    for side in enabled_sides:
        _validate_gloria_side(side, _gloria_side_config(gloria, side))
    _positive_finite("gloria_m_dual.rate_hz", gloria.rate_hz)
    _positive_finite(
        "gloria_m_dual.leader_read_timeout_s", gloria.leader_read_timeout_s
    )
    _positive_finite(
        "gloria_m_dual.status_print_interval_s", gloria.status_print_interval_s
    )

    ports = tuple(
        port
        for side in enabled_sides
        for port in (
            _openarm_port(openarm, side),
            _gloria_side_config(gloria, side).port,
        )
    )
    if len(set(ports)) != len(ports):
        raise ValueError("已启用的 OpenArm 和 Gloria-M 必须使用不同串口")
    return enabled_sides


def _create_leader(
    config: OpenArmMiniLeaderConfig, side: ArmSide
) -> OpenArmMiniLeaderArm:
    """创建一侧严格只读的 OpenArm Mini 示教夹爪输入。"""

    return OpenArmMiniLeaderArm(
        port=_openarm_port(config, side),
        calibration_path=config.calibration_path,
        side=side,
        baudrate=config.baudrate,
        read_only=True,
    )


def _clamp_opening(value: object) -> float | None:
    """将有效读数限制到统一的 0-1 开合范围。"""

    try:
        opening = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(opening):
        return None
    return max(0.0, min(1.0, opening))


def _send_gripper_target(
    leader: LeaderGripperInput,
    gripper: GripperActuator,
    timeout_s: float,
) -> tuple[float | None, bool | None]:
    """读取一侧示教夹爪并只向对应从端夹爪下发目标。"""

    opening = leader.read_gripper_opening(timeout_s)
    target = _clamp_opening(opening)
    if target is None:
        return None, None
    return target, gripper.send_normalized(target)


def _print_status(side: ArmSide, stats: _SideStats) -> None:
    """低频输出一侧夹爪的最近控制状态。"""

    target_text = (
        "unavailable" if stats.last_target is None else f"{stats.last_target:.3f}"
    )
    print(
        f"[GLORIA {side.upper()}] target={target_text} "
        f"samples={stats.samples} unavailable={stats.unavailable_reads} "
        f"send-failed={stats.send_failures}",
        flush=True,
    )


def _run_side(
    devices: _SideDevices,
    *,
    rate_hz: float,
    leader_read_timeout_s: float,
    status_print_interval_s: float,
    stop_event: threading.Event,
    failures: SimpleQueue[tuple[ArmSide, BaseException]],
    print_lock: threading.Lock,
) -> None:
    """在独立线程中完成本侧 OpenArm -> Gloria-M 的直接映射。"""

    period_s = 1.0 / rate_hz
    next_deadline = time.perf_counter()
    next_status_s = next_deadline
    stats = _SideStats()

    try:
        while not stop_event.is_set():
            target, sent = _send_gripper_target(
                devices.leader, devices.gripper, leader_read_timeout_s
            )
            stats.samples += 1
            if target is None:
                stats.unavailable_reads += 1
            else:
                stats.last_target = target
                if sent is False:
                    stats.send_failures += 1

            now = time.perf_counter()
            if now >= next_status_s:
                with print_lock:
                    _print_status(devices.side, stats)
                next_status_s = now + status_print_interval_s

            next_deadline += period_s
            wait_s = next_deadline - time.perf_counter()
            if wait_s > 0.0:
                stop_event.wait(wait_s)
            else:
                next_deadline = time.perf_counter()
    except BaseException as exc:
        failures.put((devices.side, exc))
        stop_event.set()


def _shutdown(
    workers: list[threading.Thread],
    stop_event: threading.Event,
    grippers: list[GloriaMGripperFollower],
    leaders: list[OpenArmMiniLeaderArm],
) -> None:
    """先停止控制线程，再安全失能并关闭从端与示教端串口。"""

    stop_event.set()
    for worker in workers:
        worker.join()
    for gripper in reversed(grippers):
        gripper.disable()
        gripper.disconnect()
    for leader in reversed(leaders):
        leader.disconnect()


def main() -> int:
    """连接双侧夹爪并运行，直到 Ctrl+C 或一侧工作线程异常。"""

    runtime = load_runtime_config()
    try:
        enabled_sides = _validate_config(runtime.openarm_mini, runtime.gloria_m_dual)
    except ValueError as exc:
        print(f"[ERROR] 双侧夹爪遥操作配置无效: {exc}")
        return 2

    openarm = runtime.openarm_mini
    gloria = runtime.gloria_m_dual
    leaders: list[OpenArmMiniLeaderArm] = []
    grippers: list[GloriaMGripperFollower] = []
    workers: list[threading.Thread] = []
    stop_event = threading.Event()
    failures: SimpleQueue[tuple[ArmSide, BaseException]] = SimpleQueue()
    print_lock = threading.Lock()

    try:
        sides: list[_SideDevices] = []
        for side in enabled_sides:
            leader = _create_leader(openarm, side)
            gripper = GloriaMGripperFollower(gloria.side_config(side))
            leader.connect()
            leaders.append(leader)
            gripper.connect()
            grippers.append(gripper)
            sides.append(_SideDevices(side, leader, gripper))

        for devices in sides:
            worker = threading.Thread(
                target=_run_side,
                kwargs={
                    "devices": devices,
                    "rate_hz": float(gloria.rate_hz),
                    "leader_read_timeout_s": float(gloria.leader_read_timeout_s),
                    "status_print_interval_s": float(gloria.status_print_interval_s),
                    "stop_event": stop_event,
                    "failures": failures,
                    "print_lock": print_lock,
                },
                name=f"openarm-gloria-{devices.side}",
            )
            worker.start()
            workers.append(worker)

        print("=" * 72)
        print("    OpenArm Mini 示教夹爪 -> Gloria-M 从端夹爪")
        print("=" * 72)
        print(
            f"  已启用: {', '.join(side.upper() for side in enabled_sides)}；"
            f"控制频率: {gloria.rate_hz:g} Hz"
        )
        print(
            "  仅控制夹爪开合；不会连接 HCX 或发送机械臂关节命令。按 Ctrl+C 安全退出。"
        )
        print("-" * 72)

        while not stop_event.wait(0.1):
            try:
                side, exc = failures.get_nowait()
            except Empty:
                continue
            raise RuntimeError(f"{side} 侧夹爪控制线程异常: {exc}") from exc

        try:
            side, exc = failures.get_nowait()
        except Empty:
            return 0
        raise RuntimeError(f"{side} 侧夹爪控制线程异常: {exc}") from exc
    except KeyboardInterrupt:
        print("\n[STOP] 停止 OpenArm Mini -> Gloria-M 双侧夹爪遥操作")
        return 130
    except Exception as exc:
        print(f"[ERROR] 双侧夹爪遥操作异常: {exc}")
        return 1
    finally:
        _shutdown(workers, stop_event, grippers, leaders)


if __name__ == "__main__":
    raise SystemExit(main())
