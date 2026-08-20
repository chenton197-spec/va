#!/usr/bin/env python3
"""探测同一进程能否由两个 HCX RobotClient 分别读取双臂反馈。

本程序只调用 ``RobotClient.connect()``、``Arm.joint_angles()`` 和 ``close()``。
不会读取 ``teleop.yaml``，也不会调用使能、报警清除、示教器脱离、规划运动或
直接伺服接口。

当前 HCX SDK 的设计预期同一 Python 进程只能有一个已连接的 ``RobotClient``。
因此，第二个连接在第一个保持连接时被拒绝是一次成功的探测结果，而不是程序
错误。该情况下，本程序关闭左侧 client 后再连接右侧 client，分别确认两侧的
关节反馈读取能力。

若第二个 client 意外连接成功，程序只以 30 Hz 并行读取两侧实际关节角度，不会
发送任何控制命令。此路径仅用于记录现场 SDK 的实际连接行为。

运行：

    python -m examples.test_hcx_two_clients_feedback
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

JOINT_COUNT = 7

# HCX 控制器连接参数。此 demo 不读取 teleop.yaml；请直接修改这里。
HCX_LOCAL_IP = "172.16.0.110"
HCX_REMOTE_IP = "172.16.0.89"
# 两个 RobotClient 使用不同的 HCX 端口。请按控制器实际监听端口修改；本示例
# 会拒绝相同端口，以确保确实在测试两个端口。
LEFT_HCX_PORT = 12345
RIGHT_HCX_PORT = 12346
HCX_CONNECT_TIMEOUT_S = 10.0

# 左右 HCX 机器人 ID。它们必须不同。
LEFT_ROBOT_ID = 2
RIGHT_ROBOT_ID = 1

# 每侧实际关节角度读取频率，单位为 Hz。
FEEDBACK_RATE_HZ = 30.0
# 每侧读取持续时长，单位为秒。
FEEDBACK_DURATION_S = 3.0
# 终端输出实时读取状态的最小间隔，单位为秒；不要逐帧打印。
FEEDBACK_REPORT_INTERVAL_S = 1.0


@dataclass(frozen=True)
class ProbeConfig:
    """双 client 只读探测所需的全部配置。"""

    local_ip: str
    remote_ip: str
    left_port: int
    right_port: int
    connect_timeout_s: float
    left_robot_id: int
    right_robot_id: int
    feedback_rate_hz: float
    feedback_duration_s: float
    feedback_report_interval_s: float

    def validate(self) -> None:
        for name, value in (
            ("local_ip", self.local_ip),
            ("remote_ip", self.remote_ip),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} 不能为空")
        for name, value in (
            ("left_port", self.left_port),
            ("right_port", self.right_port),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= 65535
            ):
                raise ValueError(f"{name} 必须是 1 到 65535 的整数")
        if self.left_port == self.right_port:
            raise ValueError("left_port 与 right_port 必须不同")
        _positive_finite("connect_timeout_s", self.connect_timeout_s)
        for name, value in (
            ("left_robot_id", self.left_robot_id),
            ("right_robot_id", self.right_robot_id),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} 必须是非负整数")
        if self.left_robot_id == self.right_robot_id:
            raise ValueError("left_robot_id 与 right_robot_id 必须不同")
        _positive_finite("feedback_rate_hz", self.feedback_rate_hz)
        _positive_finite("feedback_duration_s", self.feedback_duration_s)
        _positive_finite("feedback_report_interval_s", self.feedback_report_interval_s)


PROBE_CONFIG = ProbeConfig(
    local_ip=HCX_LOCAL_IP,
    remote_ip=HCX_REMOTE_IP,
    left_port=LEFT_HCX_PORT,
    right_port=RIGHT_HCX_PORT,
    connect_timeout_s=HCX_CONNECT_TIMEOUT_S,
    left_robot_id=LEFT_ROBOT_ID,
    right_robot_id=RIGHT_ROBOT_ID,
    feedback_rate_hz=FEEDBACK_RATE_HZ,
    feedback_duration_s=FEEDBACK_DURATION_S,
    feedback_report_interval_s=FEEDBACK_REPORT_INTERVAL_S,
)


@dataclass(frozen=True)
class FeedbackSummary:
    """一侧只读关节反馈的本机采样统计。"""

    label: str
    robot_id: int
    sample_count: int
    observed_rate_hz: float | None
    maximum_gap_s: float
    mean_read_duration_s: float
    maximum_read_duration_s: float
    latest_angles_deg: tuple[float, ...]


@dataclass(frozen=True)
class ProbeResult:
    """两个 RobotClient 连接探测的结果。"""

    second_client_rejection: str | None
    left_feedback: FeedbackSummary
    right_feedback: FeedbackSummary

    @property
    def concurrent_clients_connected(self) -> bool:
        return self.second_client_rejection is None


FeedbackReader = Callable[[Any, int, str, float, float, float], FeedbackSummary]


def _load_robot_client() -> Any:
    """仅在真实运行时导入包含原生扩展的 HCX SDK。"""

    from hcx_sdk import RobotClient

    return RobotClient


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


def _as_joint_angles(values: object) -> tuple[float, ...]:
    try:
        angles = tuple(float(value) for value in values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("HCX 关节反馈必须是数值数组") from exc
    if len(angles) != JOINT_COUNT or not all(math.isfinite(value) for value in angles):
        raise ValueError(f"HCX 关节反馈必须包含 {JOINT_COUNT} 个有限角度")
    return angles


def _read_feedback(
    client: Any,
    robot_id: int,
    label: str,
    rate_hz: float,
    duration_s: float,
    report_interval_s: float,
) -> FeedbackSummary:
    """以固定低频读取一侧实际关节角度，并返回本机定时统计。"""

    period_s = 1.0 / rate_hz
    arm = client.arm(robot_id)
    started_at_s = time.monotonic()
    deadline_s = started_at_s + duration_s
    next_tick_s = started_at_s
    next_report_s = started_at_s + report_interval_s
    timestamps_s: list[float] = []
    read_durations_s: list[float] = []
    latest_angles_deg: tuple[float, ...] | None = None

    while True:
        now_s = time.monotonic()
        if now_s >= deadline_s:
            break
        remaining_s = next_tick_s - now_s
        if remaining_s > 0.0:
            time.sleep(remaining_s)

        read_started_s = time.monotonic()
        latest_angles_deg = _as_joint_angles(arm.joint_angles())
        completed_at_s = time.monotonic()
        timestamps_s.append(completed_at_s)
        read_durations_s.append(completed_at_s - read_started_s)

        if completed_at_s >= next_report_s:
            print(
                f"[HCX RX] {label} robot_id={robot_id} "
                f"samples={len(timestamps_s)} "
                f"angles(deg)={[round(value, 2) for value in latest_angles_deg]}"
            )
            next_report_s = completed_at_s + report_interval_s

        next_tick_s += period_s
        if completed_at_s >= next_tick_s:
            # 读取慢时不突发补读，避免诊断程序本身持续占用控制器通信。
            next_tick_s = completed_at_s + period_s

    if latest_angles_deg is None:
        raise RuntimeError(f"{label} 在 {duration_s:g} 秒内没有获得关节反馈")

    maximum_gap_s = max(
        (
            current - previous
            for previous, current in zip(timestamps_s, timestamps_s[1:])
        ),
        default=0.0,
    )
    elapsed_s = timestamps_s[-1] - timestamps_s[0] if len(timestamps_s) > 1 else 0.0
    observed_rate_hz = (
        (len(timestamps_s) - 1) / elapsed_s
        if len(timestamps_s) > 1 and elapsed_s > 0.0
        else None
    )
    return FeedbackSummary(
        label=label,
        robot_id=robot_id,
        sample_count=len(timestamps_s),
        observed_rate_hz=observed_rate_hz,
        maximum_gap_s=maximum_gap_s,
        mean_read_duration_s=sum(read_durations_s) / len(read_durations_s),
        maximum_read_duration_s=max(read_durations_s),
        latest_angles_deg=latest_angles_deg,
    )


def _is_expected_single_client_rejection(error: BaseException) -> bool:
    """识别 SDK 的进程级单连接拒绝，不把普通网络故障误判为成功探测。"""

    message = str(error)
    return (
        "only one RobotClient may own the static HCX SDK in this process" in message
        or "the process already owns an HCX SDK connection" in message
    )


def _close_client(client: Any | None, label: str) -> None:
    """关闭一个已构造的 client，并保留有助于定位的问题上下文。"""

    if client is None:
        return
    try:
        client.close()
    except Exception as exc:
        raise RuntimeError(f"关闭 {label} RobotClient 失败: {exc}") from exc


def _read_two_clients_concurrently(
    left_client: Any,
    right_client: Any,
    config: ProbeConfig,
    feedback_reader: FeedbackReader,
) -> tuple[FeedbackSummary, FeedbackSummary]:
    """仅在两个连接确实同时成功时并行读取两侧只读反馈。"""

    with ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="hcx-feedback"
    ) as executor:
        left_future = executor.submit(
            feedback_reader,
            left_client,
            config.left_robot_id,
            "LEFT",
            config.feedback_rate_hz,
            config.feedback_duration_s,
            config.feedback_report_interval_s,
        )
        right_future = executor.submit(
            feedback_reader,
            right_client,
            config.right_robot_id,
            "RIGHT",
            config.feedback_rate_hz,
            config.feedback_duration_s,
            config.feedback_report_interval_s,
        )
        return left_future.result(), right_future.result()


def run_probe(
    robot_client_class: Any,
    config: ProbeConfig,
    *,
    feedback_reader: FeedbackReader = _read_feedback,
) -> ProbeResult:
    """执行双 client 只读探测；所有已创建 client 均会在退出时关闭。"""

    config.validate()
    left_client: Any | None = None
    right_client: Any | None = None
    left_closed = False
    right_closed = False

    try:
        left_client = robot_client_class(
            config.local_ip, config.remote_ip, config.left_port
        )
        right_client = robot_client_class(
            config.local_ip, config.remote_ip, config.right_port
        )
        left_client.connect(timeout_s=config.connect_timeout_s)
        try:
            right_client.connect(timeout_s=config.connect_timeout_s)
        except Exception as exc:
            if not _is_expected_single_client_rejection(exc):
                raise RuntimeError(f"第二个 RobotClient 连接异常: {exc}") from exc

            second_client_rejection = f"{type(exc).__name__}: {exc}"
            left_feedback = feedback_reader(
                left_client,
                config.left_robot_id,
                "LEFT",
                config.feedback_rate_hz,
                config.feedback_duration_s,
                config.feedback_report_interval_s,
            )
            _close_client(left_client, "LEFT")
            left_closed = True

            right_client.connect(timeout_s=config.connect_timeout_s)
            right_feedback = feedback_reader(
                right_client,
                config.right_robot_id,
                "RIGHT",
                config.feedback_rate_hz,
                config.feedback_duration_s,
                config.feedback_report_interval_s,
            )
        else:
            second_client_rejection = None
            left_feedback, right_feedback = _read_two_clients_concurrently(
                left_client,
                right_client,
                config,
                feedback_reader,
            )

        return ProbeResult(
            second_client_rejection=second_client_rejection,
            left_feedback=left_feedback,
            right_feedback=right_feedback,
        )
    finally:
        close_error: BaseException | None = None
        if not right_closed:
            try:
                _close_client(right_client, "RIGHT")
                right_closed = True
            except BaseException as exc:
                close_error = exc
        if not left_closed:
            try:
                _close_client(left_client, "LEFT")
                left_closed = True
            except BaseException as exc:
                if close_error is None:
                    close_error = exc
        if close_error is not None:
            raise close_error


def _rate_text(rate_hz: float | None) -> str:
    return "--" if rate_hz is None else f"{rate_hz:.1f}"


def _print_feedback_summary(summary: FeedbackSummary) -> None:
    print(
        f"  {summary.label} robot_id={summary.robot_id}: "
        f"samples={summary.sample_count} "
        f"rate={_rate_text(summary.observed_rate_hz)}/{FEEDBACK_RATE_HZ:g} Hz "
        f"read-mean/max={summary.mean_read_duration_s * 1_000.0:.2f}/"
        f"{summary.maximum_read_duration_s * 1_000.0:.2f} ms "
        f"gap-max={summary.maximum_gap_s * 1_000.0:.2f} ms\n"
        f"    latest(deg)={[round(value, 2) for value in summary.latest_angles_deg]}"
    )


def main() -> int:
    """执行顶部常量定义的只读双 RobotClient 探测。"""

    try:
        PROBE_CONFIG.validate()
        robot_client_class = _load_robot_client()
        print("=" * 72)
        print("    HCX 双 RobotClient 只读反馈探测")
        print("=" * 72)
        print("  仅连接和读取关节角度；不会调用使能、报警处理、运动或直伺服。")
        print(
            f"  左侧 robot_id={PROBE_CONFIG.left_robot_id}，"
            f"port={PROBE_CONFIG.left_port}；"
            f"右侧 robot_id={PROBE_CONFIG.right_robot_id}，"
            f"port={PROBE_CONFIG.right_port}"
        )
        print(
            f"  每侧反馈: {PROBE_CONFIG.feedback_rate_hz:g} Hz，"
            f"持续 {PROBE_CONFIG.feedback_duration_s:g} s"
        )
        print("-" * 72)
        result = run_probe(robot_client_class, PROBE_CONFIG)
    except KeyboardInterrupt:
        print("\n[STOP] 收到退出请求，正在关闭已创建的 HCX client。")
        return 130
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[ERROR] HCX 双 RobotClient 只读探测失败: {exc}")
        return 1

    print("=" * 72)
    print("    探测结果")
    print("=" * 72)
    if result.concurrent_clients_connected:
        print(
            "[WARN] 第二个 RobotClient 已同时连接；本次只并行读取反馈，未发送控制命令。"
        )
    else:
        assert result.second_client_rejection is not None
        print("[PASS] 同进程第二个 RobotClient 被 SDK 按预期拒绝。")
        print(f"  rejection: {result.second_client_rejection}")
        print("  已关闭左侧 client 后再连接右侧 client，分别完成反馈读取。")
    _print_feedback_summary(result.left_feedback)
    _print_feedback_summary(result.right_feedback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
