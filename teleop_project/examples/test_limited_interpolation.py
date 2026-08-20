#!/usr/bin/env python3
"""使用 HCX SDK 测试 LimitedInterpolator 的分段轨迹和固定频率下发。

该示例将轨迹生成和直伺服通信明确拆开：每个
``LIMITED_UPDATE_RATE_HZ`` 周期执行一次
``selected joint step target -> LimitedInterpolator.interpolate``，生成一批
以 ``DIRECT_SERVO_RATE_HZ`` 为输出频率的轨迹点；独立的 500 Hz 发送线程在
每一个直伺服时隙只取出一个已经生成的点并调用
``DirectServoSession.set_target``。因此 500 Hz 是命令通信频率，不会在每个
发送时隙重新运行限速、限加速度算法。若暂时没有下一批轨迹点，发送端会重发
最后一个命令，仍保持固定的伺服心跳。

其中限幅器使用项目当前的提前制动实现，而不是 ``test.c`` 中旧的
``error / dt`` 速度目标公式。

连接由 :class:`teleop_sdk.adapters.HcxConnection` 管理，因此在实际
``acquire()`` 时会加载 ``hcx_sdk``，返回的 ``arm`` 是 HCX SDK 的 ``Arm``，
并由其 ``start_direct_servo()`` 创建 Python 直伺服会话。

本示例不接受命令行参数，也不读取 ``teleop.yaml``；所有连接、启动、直伺服和
测试参数都定义在本文件顶部。默认只测试 ``RIGHT_ARM_ID``。运行会产生真实运动，
必须同时满足：

* ``DIRECT_SERVO_INTERPOLATION == "limited"``；
* ``CONFIRM_DIRECT_SERVO == True``；
* 控制器状态通过本文件的 ``AUTO_*`` 常量或已在现场准备完成；
* 单臂防护已关闭，或明确设置 ``TEST_DISABLE_PROTECTION = True``
  让本程序暂时关闭并在退出时恢复。

例如：

    python -m examples.test_limited_interpolation
"""

from __future__ import annotations

import csv
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Any, Literal

import numpy as np

from teleop_sdk.adapters import HcxConnection, HcxConnectionConfig
from teleop_sdk.algorithms import LimitedInterpolator
from teleop_sdk.config import HcxConfig

# HCX 单臂固定关节数；本 demo 只支持七轴机械臂。
JOINT_COUNT = 7
# 可测试的单臂名称类型。
HcxSide = Literal["left", "right"]

# 本机用于接收 HCX 控制器通信数据的网卡地址。
LOCAL_IP = "172.16.0.110"
# HCX 控制器的网络地址。
REMOTE_IP = "172.16.0.89"
# HCX SDK 的控制器通信端口。
PORT = 12345
# 建立 SDK 通信连接的最长等待时间，单位为秒。
CONNECT_TIMEOUT_S = 10.0

# 控制器项目中左臂的机器人 ID；不是固定的左右臂编号。
LEFT_ARM_ID = 1
# 控制器项目中右臂的机器人 ID；不是固定的左右臂编号。
RIGHT_ARM_ID = 2

# LimitedInterpolator 每秒生成多少批轨迹。每批生成到 500 Hz 的多个命令点；
# 这不是 HCX 通信频率，也不是主臂采样频率。
LIMITED_UPDATE_RATE_HZ = 100
# 直伺服命令发送频率，单位为 Hz。本 demo 固定为 500 Hz；每个时隙只发送一个
# 已生成的目标点或上一目标，不在该时隙执行 LimitedInterpolator。
DIRECT_SERVO_RATE_HZ = 500
# Python 会话看门狗阈值，单位为秒；每个控制周期调用一次 set_target()。
DIRECT_SERVO_WATCHDOG_S = 2.0
# 本 demo 固定使用 LimitedInterpolator；不能设置为 direct 或 linear。
DIRECT_SERVO_INTERPOLATION: Literal["limited"] = "limited"
# 仅在隔离单元、硬件保护与独立急停均已现场确认后设为 True。
CONFIRM_DIRECT_SERVO = True

# LimitedInterpolator 的最大关节速度，单位为度/秒。
LIMITED_MAX_VELOCITY_DEG_S = 120.0
# LimitedInterpolator 的最大关节加速度，单位为度/秒平方。
LIMITED_MAX_ACCELERATION_DEG_S2 = 80.0
# LimitedInterpolator 的固定低通系数，范围 [0, 1]；越小滤波越强。
LIMITED_LOWPASS_ALPHA = 0.25

# True 时在启动前请求控制器脱离示教器；仅在示教器已物理拔除时开启。
AUTO_DETACH_HMI_IF_NO_TEACH_PENDANT = True
# True 时在存在报警时请求控制器清除报警；必须先确认报警原因已消除。
AUTO_CLEAR_ALARMS = False
# True 时在启动前请求全局使能和左右单臂使能。
AUTO_ENABLE = True
# SDK 连接建立后、执行启动前置流程前的稳定等待时间，单位为秒。
CONTROLLER_INITIALIZATION_WAIT_S = 2.0
# 需要检查 OP 状态的 EtherCAT 主站索引；非 EtherCAT 机械臂应设为空元组。
ETHERCAT_MASTER_INDICES: tuple[int, ...] = (0, 1)
# 每个 EtherCAT 主站等待进入 OP 状态的最长时间，单位为秒。
ETHERCAT_OP_TIMEOUT_S = 15.0
# 自动清报警的最大重试次数；总请求次数为初始请求加该次数。
ALARM_CLEAR_RETRY_COUNT = 5
# 两次自动清报警请求之间的间隔，单位为秒。
ALARM_CLEAR_RETRY_INTERVAL_S = 1.0
# 全局使能的最大重试次数；总请求次数为初始请求加该次数。
GLOBAL_ENABLE_RETRY_COUNT = 5
# 两次全局使能请求之间的间隔，单位为秒。
GLOBAL_ENABLE_RETRY_INTERVAL_S = 1.0
# 单臂使能请求后等待使能反馈变为真的最长时间，单位为秒。
SINGLE_ARM_ENABLE_TIMEOUT_S = 5.0
# 轮询单臂使能反馈的时间间隔，单位为秒。
ENABLE_STATUS_POLL_INTERVAL_S = 0.1

# 执行测试的单臂；可改为 "left" 或 "right"。
TEST_SIDE: HcxSide = "right"
# 要测试的关节编号；按 J1-J7 填写 1-7，例如 3 表示测试 J3。
TEST_JOINT_NUMBER = 3
# 测试关节相对初始角度的一次阶跃量，单位为度；正数沿正方向移动，负数沿负方向移动。
TEST_STEP_DEG = 40.0
# 原始测试关节阶跃目标保持时间，单位为秒；这段时间用于观察限速、限加速度爬升过程。
TEST_STEP_HOLD_SECONDS = 2.0
# 原始目标切回起始角后的等待时间，单位为秒；让限幅输出有机会平稳回到起始角。
TEST_RETURN_SETTLE_SECONDS = 2.0
# true 时本测试会暂时关闭本侧防护，并在退出时请求恢复；通常应保持 false。
TEST_DISABLE_PROTECTION = True
# 留空时不导出；填写路径后导出 Python 提交给薄原生绑定的原始和限幅七轴命令。
TEST_CSV_PATH = ""

# 将顶部连接、启动和直伺服常量组装为 HcxConnection 使用的配置对象。
TEST_HCX_CONFIG = HcxConfig(
    local_ip=LOCAL_IP,
    remote_ip=REMOTE_IP,
    port=PORT,
    connect_timeout_s=CONNECT_TIMEOUT_S,
    left_robot_id=LEFT_ARM_ID,
    right_robot_id=RIGHT_ARM_ID,
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


@dataclass(frozen=True)
class DemoConfig:
    """真实 HCX 指定关节阶跃限幅测试参数，所有关节量均为度。"""

    side: HcxSide
    joint_number: int
    step_deg: float
    step_hold_seconds: float
    return_settle_seconds: float
    disable_protection: bool

    def validate(self) -> None:
        """验证阶跃和显式防护授权以外的本地测试参数。"""

        if self.side not in ("left", "right"):
            raise ValueError("side 必须是 left 或 right")
        if (
            not isinstance(self.joint_number, int)
            or isinstance(self.joint_number, bool)
            or not 1 <= self.joint_number <= JOINT_COUNT
        ):
            raise ValueError(f"joint_number 必须是 1 到 {JOINT_COUNT} 的整数")
        for name, value, allow_zero in (
            ("step_hold_seconds", self.step_hold_seconds, False),
            ("return_settle_seconds", self.return_settle_seconds, True),
        ):
            if isinstance(value, bool):
                raise ValueError(f"{name} 必须是有限数")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} 必须是有限数") from exc
            if (
                not math.isfinite(numeric)
                or numeric < 0.0
                or (not allow_zero and numeric == 0.0)
            ):
                qualifier = "非负" if allow_zero else "正"
                raise ValueError(f"{name} 必须是有限{qualifier}数")
        if isinstance(self.step_deg, bool):
            raise ValueError("step_deg 必须是非零有限数")
        try:
            step_deg = float(self.step_deg)
        except (TypeError, ValueError) as exc:
            raise ValueError("step_deg 必须是非零有限数") from exc
        if not math.isfinite(step_deg) or step_deg == 0.0:
            raise ValueError("step_deg 必须是非零有限数")
        if not isinstance(self.disable_protection, bool):
            raise ValueError("disable_protection 必须是布尔值")

    @property
    def joint_index(self) -> int:
        """将用户填写的 J1-J7 编号转换为七轴数组索引。"""

        return self.joint_number - 1

    @property
    def joint_label(self) -> str:
        """返回终端输出使用的关节名称，例如 J3。"""

        return f"J{self.joint_number}"


# 将顶部测试关节、阶跃和防护常量组装为本次测试的不可变配置对象。
TEST_CONFIG = DemoConfig(
    side=TEST_SIDE,
    joint_number=TEST_JOINT_NUMBER,
    step_deg=TEST_STEP_DEG,
    step_hold_seconds=TEST_STEP_HOLD_SECONDS,
    return_settle_seconds=TEST_RETURN_SETTLE_SECONDS,
    disable_protection=TEST_DISABLE_PROTECTION,
)


@dataclass(frozen=True)
class DemoTrace:
    """实际提交给 HCX SDK 的原始目标和 500 Hz 命令序列。"""

    timestamps_s: np.ndarray
    raw_targets_deg: np.ndarray
    command_targets_deg: np.ndarray
    limited_batch_ids: np.ndarray
    precomputed_target_mask: np.ndarray
    initial_angles_deg: np.ndarray

    def joint_velocity_deg_s(self, joint_index: int, rate_hz: int) -> np.ndarray:
        """根据指定关节已提交命令计算离散速度。"""

        positions = np.concatenate(
            (
                self.initial_angles_deg[joint_index : joint_index + 1],
                self.command_targets_deg[:, joint_index],
            )
        )
        return np.diff(positions) * rate_hz

    def joint_acceleration_deg_s2(
        self, joint_index: int, rate_hz: int
    ) -> np.ndarray:
        """根据指定关节已提交命令计算离散加速度。"""

        velocity = self.joint_velocity_deg_s(joint_index, rate_hz)
        return np.diff(np.concatenate((np.zeros(1), velocity))) * rate_hz


@dataclass(frozen=True)
class DemoReport:
    """一次真实透传测试的周期和 Python 会话诊断。"""

    robot_id: int
    side: HcxSide
    joint_number: int
    rate_hz: int
    limited_update_rate_hz: int
    watchdog_s: float
    max_velocity_deg_s: float
    max_acceleration_deg_s2: float
    planned_command_count: int
    submitted_count: int
    limited_update_count: int
    held_target_count: int
    deadline_miss_count: int
    late_send_count: int
    maximum_lateness_s: float
    observed_rate_hz: float | None
    native_sent_count: int
    native_running: bool
    native_faulted: bool
    native_error: str | None
    trace: DemoTrace


@dataclass(frozen=True)
class _TrajectoryBatch:
    """一次低频限幅计算生成的完整高频命令批次。"""

    sequence: int
    raw_target_deg: np.ndarray
    command_targets_deg: np.ndarray


class _LatestTrajectoryBatch:
    """单生产者/单消费者的最新轨迹批次槽。

    生产端以新批次整体替换引用，500 Hz 发送线程每个时隙只读取该引用并取一个
    已生成点。CPython 中的引用替换和读取受 GIL 保护；这里没有互斥锁，也不会
    在发送线程中执行 LimitedInterpolator。
    """

    def __init__(self) -> None:
        self._next_sequence = 1
        self._batch: _TrajectoryBatch | None = None

    def publish(
        self, raw_target_deg: np.ndarray, command_targets_deg: np.ndarray
    ) -> int:
        raw_target = np.asarray(raw_target_deg, dtype=float)
        commands = np.asarray(command_targets_deg, dtype=float)
        if raw_target.shape != (JOINT_COUNT,) or not np.isfinite(raw_target).all():
            raise ValueError("raw_target_deg 必须是有效的七轴度制数组")
        if (
            commands.ndim != 2
            or commands.shape[0] == 0
            or commands.shape[1] != JOINT_COUNT
            or not np.isfinite(commands).all()
        ):
            raise ValueError("command_targets_deg 必须是非空的有效七轴轨迹")
        sequence = self._next_sequence
        self._next_sequence += 1
        self._batch = _TrajectoryBatch(
            sequence=sequence,
            raw_target_deg=raw_target.copy(),
            command_targets_deg=commands.copy(),
        )
        return sequence

    def snapshot(self) -> _TrajectoryBatch | None:
        """返回当前完整批次；调用者只读该批次中的数组。"""

        return self._batch


@dataclass
class _SenderRecorder:
    """仅由 500 Hz 发送线程写入的命令 trace 与调度统计。"""

    initial_angles_deg: np.ndarray
    timestamps_s: list[float]
    raw_targets_deg: list[np.ndarray]
    command_targets_deg: list[np.ndarray]
    limited_batch_ids: list[int]
    precomputed_target_mask: list[bool]
    deadline_miss_count: int = 0
    late_send_count: int = 0
    maximum_lateness_s: float = 0.0

    @classmethod
    def create(cls, initial_angles_deg: np.ndarray) -> "_SenderRecorder":
        return cls(
            initial_angles_deg=initial_angles_deg.copy(),
            timestamps_s=[],
            raw_targets_deg=[],
            command_targets_deg=[],
            limited_batch_ids=[],
            precomputed_target_mask=[],
        )

    def record(
        self,
        *,
        timestamp_s: float,
        raw_target_deg: np.ndarray,
        command_target_deg: np.ndarray,
        limited_batch_id: int,
        used_precomputed_target: bool,
    ) -> None:
        self.timestamps_s.append(timestamp_s)
        self.raw_targets_deg.append(raw_target_deg.copy())
        self.command_targets_deg.append(command_target_deg.copy())
        self.limited_batch_ids.append(limited_batch_id)
        self.precomputed_target_mask.append(used_precomputed_target)

    def trace(self) -> DemoTrace:
        return DemoTrace(
            timestamps_s=np.asarray(self.timestamps_s, dtype=float),
            raw_targets_deg=np.asarray(self.raw_targets_deg, dtype=float),
            command_targets_deg=np.asarray(self.command_targets_deg, dtype=float),
            limited_batch_ids=np.asarray(self.limited_batch_ids, dtype=int),
            precomputed_target_mask=np.asarray(
                self.precomputed_target_mask, dtype=bool
            ),
            initial_angles_deg=self.initial_angles_deg.copy(),
        )

    def observed_rate_hz(self) -> float | None:
        if len(self.timestamps_s) < 2:
            return None
        elapsed_s = self.timestamps_s[-1] - self.timestamps_s[0]
        return (len(self.timestamps_s) - 1) / elapsed_s if elapsed_s > 0.0 else None


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


def _planned_command_count(config: DemoConfig, rate_hz: int) -> int:
    """返回测试时段内必须完成的固定直伺服发送次数。"""

    duration_s = config.step_hold_seconds + config.return_settle_seconds
    # 发送时隙为 [0, duration)，因此向上取整；微小浮点误差不应多出一帧。
    return max(1, math.ceil(duration_s * rate_hz - 1e-12))


def _validate_hcx_config(config: HcxConfig) -> tuple[int, int, float]:
    """验证本示例的 100 Hz 轨迹生成和固定 500 Hz 直伺服配置。"""

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
    if config.direct_servo_interpolation != "limited":
        raise ValueError(
            "本测试使用 LimitedInterpolator；请设置 "
            "hcx.direct_servo_interpolation: limited"
        )
    if not config.direct_servo_confirm_unsafe:
        raise ValueError("请在文件顶部明确设置 CONFIRM_DIRECT_SERVO = True")
    if (
        not isinstance(config.direct_servo_rate_hz, int)
        or isinstance(config.direct_servo_rate_hz, bool)
        or not 100 <= config.direct_servo_rate_hz <= 1000
    ):
        raise ValueError("hcx.direct_servo_rate_hz 必须是 100 到 1000 的整数")
    if config.direct_servo_rate_hz != DIRECT_SERVO_RATE_HZ:
        raise ValueError(
            "本 demo 固定以 "
            f"{DIRECT_SERVO_RATE_HZ} Hz 调用 set_target；"
            "请保持 DIRECT_SERVO_RATE_HZ 和 hcx.direct_servo_rate_hz 都为 500"
        )
    watchdog_s = _positive_finite(
        "hcx.direct_servo_watchdog_s", config.direct_servo_watchdog_s
    )
    rate_hz = config.direct_servo_rate_hz
    if watchdog_s <= 1.0 / rate_hz:
        raise ValueError("hcx.direct_servo_watchdog_s 必须大于一个直伺服周期")
    if (
        not isinstance(LIMITED_UPDATE_RATE_HZ, int)
        or isinstance(LIMITED_UPDATE_RATE_HZ, bool)
        or not 1 <= LIMITED_UPDATE_RATE_HZ <= rate_hz
    ):
        raise ValueError(
            "LIMITED_UPDATE_RATE_HZ 必须是大于零且不超过 "
            "DIRECT_SERVO_RATE_HZ 的整数"
        )
    if rate_hz % LIMITED_UPDATE_RATE_HZ != 0:
        raise ValueError(
            "DIRECT_SERVO_RATE_HZ 必须是 LIMITED_UPDATE_RATE_HZ 的整数倍，"
            "以便每批轨迹点能按固定 500 Hz 完整发送"
        )
    _positive_finite(
        "hcx.direct_servo_limited_max_vel_deg_s",
        config.direct_servo_limited_max_vel_deg_s,
    )
    _positive_finite(
        "hcx.direct_servo_limited_max_accel_deg_s2",
        config.direct_servo_limited_max_accel_deg_s2,
    )
    if isinstance(config.direct_servo_limited_lowpass_alpha, bool):
        raise ValueError("hcx.direct_servo_limited_lowpass_alpha 必须在 0 到 1 之间")
    try:
        alpha = float(config.direct_servo_limited_lowpass_alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "hcx.direct_servo_limited_lowpass_alpha 必须在 0 到 1 之间"
        ) from exc
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("hcx.direct_servo_limited_lowpass_alpha 必须在 0 到 1 之间")
    return rate_hz, LIMITED_UPDATE_RATE_HZ, watchdog_s


def _robot_id_for_side(config: HcxConfig, side: HcxSide) -> int:
    return config.left_robot_id if side == "left" else config.right_robot_id


def _configured_csv_path() -> Path | None:
    """返回文件顶部配置的 CSV 输出路径；空字符串表示不导出。"""

    value = TEST_CSV_PATH
    if not isinstance(value, str):
        raise ValueError("TEST_CSV_PATH 必须是字符串")
    normalized = value.strip()
    return Path(normalized) if normalized else None


def _step_target_deg(
    initial_angles_deg: np.ndarray, config: DemoConfig, elapsed_s: float
) -> np.ndarray:
    """在保持阶段给指定关节一个固定阶跃，之后将原始目标切回初始角。"""

    target = initial_angles_deg.copy()
    if elapsed_s < config.step_hold_seconds:
        target[config.joint_index] += config.step_deg
    return target


def _validate_initial_pose(
    arm: Any, initial_angles_deg: np.ndarray, config: DemoConfig
) -> tuple[np.ndarray, np.ndarray]:
    """确认七轴反馈和指定关节阶跃目标均在控制器限位内。"""

    if getattr(arm, "axis_count") != JOINT_COUNT:
        raise RuntimeError(
            f"HCX {config.side} 预期为 {JOINT_COUNT} 轴，"
            f"控制器返回 {getattr(arm, 'axis_count')} 轴"
        )
    if (
        initial_angles_deg.shape != (JOINT_COUNT,)
        or not np.isfinite(initial_angles_deg).all()
    ):
        raise RuntimeError("HCX 当前关节反馈不是有效的七轴度制数组")
    limits_deg = np.asarray(arm.joint_limits_deg, dtype=float)
    if limits_deg.shape != (JOINT_COUNT, 2) or not np.isfinite(limits_deg).all():
        raise RuntimeError("HCX 控制器返回的关节限位无效")
    lower_limit, upper_limit = limits_deg[config.joint_index]
    if lower_limit >= upper_limit:
        raise RuntimeError(f"HCX {config.joint_label} 关节限位无效")
    target = initial_angles_deg[config.joint_index] + config.step_deg
    if target < lower_limit or target > upper_limit:
        raise ValueError(
            f"{config.joint_label} 阶跃目标超出控制器限位："
            f"{target:.3f} 不在 [{lower_limit:.3f}, {upper_limit:.3f}] 内"
        )
    return limits_deg[:, 0].copy(), limits_deg[:, 1].copy()


def _new_limited_interpolator(
    hcx_config: HcxConfig,
    limited_update_rate_hz: int,
    direct_servo_rate_hz: int,
    min_angles_deg: np.ndarray,
    max_angles_deg: np.ndarray,
    initial_angles_deg: np.ndarray,
) -> LimitedInterpolator:
    """创建仅供 100 Hz 生产端使用的 LimitedInterpolator。"""

    interpolator = LimitedInterpolator(
        limited_update_rate_hz,
        direct_servo_rate_hz,
        max_velocity_deg_s=hcx_config.direct_servo_limited_max_vel_deg_s,
        max_acceleration_deg_s2=hcx_config.direct_servo_limited_max_accel_deg_s2,
        lowpass_alpha=hcx_config.direct_servo_limited_lowpass_alpha,
        min_angles_deg=min_angles_deg,
        max_angles_deg=max_angles_deg,
    )
    interpolator.reset(initial_angles_deg)
    return interpolator


def _publish_limited_batch(
    interpolator: LimitedInterpolator,
    mailbox: _LatestTrajectoryBatch,
    raw_target_deg: np.ndarray,
    expected_points_per_batch: int,
) -> None:
    """由生产端生成并原子发布一整批已经限幅的 500 Hz 命令点。"""

    command_targets_deg = interpolator.interpolate(raw_target_deg)
    if command_targets_deg.shape != (expected_points_per_batch, JOINT_COUNT):
        raise RuntimeError(
            "LimitedInterpolator 返回的轨迹点数与 500 Hz 发送比例不一致"
        )
    mailbox.publish(raw_target_deg, command_targets_deg)


def _run_fixed_rate_sender(
    session: Any,
    mailbox: _LatestTrajectoryBatch,
    initial_angles_deg: np.ndarray,
    rate_hz: int,
    stop_event: Any,
    failures: SimpleQueue[BaseException],
    recorder: _SenderRecorder,
    *,
    monotonic: Any = time.monotonic,
    max_ticks: int | None = None,
) -> None:
    """独立 500 Hz 发送循环；循环内只有取点、set_target 和调度统计。

    该函数刻意不接收 LimitedInterpolator。即使生产端没有更新批次，也会在每个
    发送时隙使用最后一条命令调用一次 set_target()，而不会降低通信频率。
    """

    period_s = 1.0 / rate_hz
    sender_started_s = monotonic()
    next_tick_s = sender_started_s
    held_raw_target = initial_angles_deg.copy()
    held_command_target = initial_angles_deg.copy()
    active_batch: _TrajectoryBatch | None = None
    active_batch_index = 0

    while not stop_event.is_set():
        if max_ticks is not None and len(recorder.timestamps_s) >= max_ticks:
            return

        now_s = monotonic()
        remaining_s = next_tick_s - now_s
        if remaining_s > 0.0:
            if stop_event.wait(remaining_s):
                return
            continue

        latest_batch = mailbox.snapshot()
        if latest_batch is not None and (
            active_batch is None or latest_batch.sequence != active_batch.sequence
        ):
            active_batch = latest_batch
            active_batch_index = 0

        if (
            active_batch is not None
            and active_batch_index < active_batch.command_targets_deg.shape[0]
        ):
            raw_target = active_batch.raw_target_deg
            command_target = active_batch.command_targets_deg[active_batch_index]
            limited_batch_id = active_batch.sequence
            active_batch_index += 1
            used_precomputed_target = True
        else:
            # 没有新的预生成点时仍严格保活；不会因静止或生产端延迟而停止发送。
            raw_target = held_raw_target
            command_target = held_command_target
            limited_batch_id = 0
            used_precomputed_target = False

        try:
            session.set_target(command_target.tolist())
        except Exception as exc:
            failures.put(exc)
            stop_event.set()
            return
        completed_s = monotonic()
        recorder.record(
            timestamp_s=completed_s - sender_started_s,
            raw_target_deg=raw_target,
            command_target_deg=command_target,
            limited_batch_id=limited_batch_id,
            used_precomputed_target=used_precomputed_target,
        )
        held_raw_target = raw_target.copy()
        held_command_target = command_target.copy()

        next_tick_s += period_s
        if completed_s >= next_tick_s:
            # 不丢弃任何计划中的 500 Hz 命令。即使本次厂商调用结束得晚，下一轮
            # 也会立即发送已到期的下一个点，直到重新追上固定时间轴。
            recorder.late_send_count += 1
            recorder.maximum_lateness_s = max(
                recorder.maximum_lateness_s,
                completed_s - next_tick_s,
            )


def _run_limited_batch_producer(
    interpolator: LimitedInterpolator,
    mailbox: _LatestTrajectoryBatch,
    initial_angles_deg: np.ndarray,
    config: DemoConfig,
    limited_update_rate_hz: int,
    direct_servo_rate_hz: int,
    stop_event: threading.Event,
) -> int:
    """在调用线程以 100 Hz 生成 limited 批次；此路径不发送 HCX 命令。"""

    limited_update_period_s = 1.0 / limited_update_rate_hz
    expected_points_per_batch = direct_servo_rate_hz // limited_update_rate_hz
    started_s = time.monotonic()
    next_update_s = started_s
    end_s = started_s + config.step_hold_seconds + config.return_settle_seconds
    batch_count = 0

    while not stop_event.is_set():
        now_s = time.monotonic()
        if now_s >= end_s:
            break
        remaining_s = next_update_s - now_s
        if remaining_s > 0.0:
            if stop_event.wait(remaining_s):
                break
            continue

        raw_target = _step_target_deg(
            initial_angles_deg,
            config,
            now_s - started_s,
        )
        _publish_limited_batch(
            interpolator,
            mailbox,
            raw_target,
            expected_points_per_batch,
        )
        batch_count += 1

        next_update_s += limited_update_period_s
        if now_s >= next_update_s:
            skipped_updates = int(
                (now_s - next_update_s) // limited_update_period_s
            ) + 1
            next_update_s += skipped_updates * limited_update_period_s

    return batch_count


def _raise_sender_failure(failures: SimpleQueue[BaseException]) -> None:
    """将独立发送线程的异常在主调用线程中重新抛出。"""

    try:
        failure = failures.get_nowait()
    except Empty:
        return
    raise RuntimeError(f"HCX 500 Hz 直伺服发送失败: {failure}") from failure


def run_demo(
    hcx_config: HcxConfig,
    config: DemoConfig,
) -> DemoReport:
    """连接真实 HCX，执行 100 Hz 生产和独立 500 Hz 发送测试。"""

    config.validate()
    rate_hz, limited_update_rate_hz, watchdog_s = _validate_hcx_config(hcx_config)
    planned_command_count = _planned_command_count(config, rate_hz)
    robot_id = _robot_id_for_side(hcx_config, config.side)
    connection = HcxConnection(HcxConnectionConfig.from_runtime_config(hcx_config))
    arm: Any | None = None
    session: Any | None = None
    sender_thread: threading.Thread | None = None
    sender_stop_event = threading.Event()
    sender_failures: SimpleQueue[BaseException] = SimpleQueue()
    protection_disabled_by_demo = False
    try:
        # acquire() 在此处通过 HcxConnection 懒加载 hcx_sdk.RobotClient。
        arm = connection.acquire(robot_id)
        connection.prepare_for_motion(robot_id)

        if bool(arm.protection_enabled):
            if not config.disable_protection:
                raise RuntimeError(
                    "HCX 单臂防护仍开启。请先按现场规程关闭，或在确认安全后设置 "
                    "TEST_DISABLE_PROTECTION = True，让本程序暂时关闭并在退出时恢复。"
                )
            arm.set_protection(False, confirm_unsafe=True)
            protection_disabled_by_demo = True

        initial_angles_deg = np.asarray(arm.joint_angles(), dtype=float)
        min_angles_deg, max_angles_deg = _validate_initial_pose(
            arm, initial_angles_deg, config
        )
        with arm.start_direct_servo(
            rate_hz=rate_hz,
            watchdog_s=watchdog_s,
            confirm_unsafe=True,
        ) as session:
            interpolator = _new_limited_interpolator(
                hcx_config,
                limited_update_rate_hz,
                rate_hz,
                min_angles_deg,
                max_angles_deg,
                initial_angles_deg,
            )
            mailbox = _LatestTrajectoryBatch()
            recorder = _SenderRecorder.create(initial_angles_deg)
            sender_thread = threading.Thread(
                target=_run_fixed_rate_sender,
                args=(
                    session,
                    mailbox,
                    initial_angles_deg,
                    rate_hz,
                    sender_stop_event,
                    sender_failures,
                    recorder,
                ),
                name=f"hcx-{config.side}-fixed-{rate_hz}hz-sender",
                daemon=True,
                kwargs={"max_ticks": planned_command_count},
            )
            sender_thread.start()
            try:
                limited_update_count = _run_limited_batch_producer(
                    interpolator,
                    mailbox,
                    initial_angles_deg,
                    config,
                    limited_update_rate_hz,
                    rate_hz,
                    sender_stop_event,
                )
            except BaseException:
                sender_stop_event.set()
                sender_thread.join(timeout=max(1.0, 2.0 / rate_hz))
                raise

            # 生产端停止后，发送端仍要补完本测试已计划的全部 500 Hz 时隙。
            sender_thread.join(timeout=max(1.0, 2.0 / rate_hz))

            if sender_thread.is_alive():
                sender_stop_event.set()
                sender_thread.join(timeout=max(1.0, 2.0 / rate_hz))
                raise RuntimeError("HCX 500 Hz 发送线程未能在停止请求后退出")
            _raise_sender_failure(sender_failures)
            trace = recorder.trace()
            if trace.command_targets_deg.shape[0] != planned_command_count:
                raise RuntimeError(
                    "HCX 固定 500 Hz 发送未完成计划帧数："
                    f"{trace.command_targets_deg.shape[0]}/"
                    f"{planned_command_count}"
                )
            state = session.state
            if not state.running or state.faulted:
                raise RuntimeError(
                    "HCX 直伺服会话异常："
                    f"running={state.running}, faulted={state.faulted}, "
                    f"sent_count={state.sent_count}, error={state.error!r}"
                )
            return DemoReport(
                robot_id=robot_id,
                side=config.side,
                joint_number=config.joint_number,
                rate_hz=rate_hz,
                limited_update_rate_hz=limited_update_rate_hz,
                watchdog_s=watchdog_s,
                max_velocity_deg_s=float(hcx_config.direct_servo_limited_max_vel_deg_s),
                max_acceleration_deg_s2=float(
                    hcx_config.direct_servo_limited_max_accel_deg_s2
                ),
                planned_command_count=planned_command_count,
                submitted_count=trace.command_targets_deg.shape[0],
                limited_update_count=limited_update_count,
                held_target_count=int(
                    trace.precomputed_target_mask.size
                    - np.count_nonzero(trace.precomputed_target_mask)
                ),
                deadline_miss_count=recorder.deadline_miss_count,
                late_send_count=recorder.late_send_count,
                maximum_lateness_s=recorder.maximum_lateness_s,
                observed_rate_hz=recorder.observed_rate_hz(),
                native_sent_count=state.sent_count,
                native_running=bool(state.running),
                native_faulted=bool(state.faulted),
                native_error=state.error,
                trace=trace,
            )
    finally:
        sender_stop_event.set()
        if sender_thread is not None and sender_thread.is_alive():
            sender_thread.join(timeout=max(1.0, 2.0 / rate_hz))
        if session is not None:
            try:
                session.stop()
            except (RuntimeError, TypeError, ValueError) as exc:
                print(f"[WARN] 停止 HCX 直伺服会话失败: {exc}", file=sys.stderr)
        if protection_disabled_by_demo and arm is not None:
            try:
                arm.set_protection(True)
            except (RuntimeError, TypeError, ValueError) as exc:
                print(f"[WARN] 恢复 HCX 单臂防护失败: {exc}", file=sys.stderr)
        if arm is not None:
            try:
                connection.release(robot_id)
            except (RuntimeError, TypeError, ValueError) as exc:
                print(f"[WARN] 关闭 HCX 连接失败: {exc}", file=sys.stderr)


def write_csv(path: Path, trace: DemoTrace) -> None:
    """写出实际发送的 500 Hz 命令及其 limited 批次来源。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["time_s", "limited_batch_id", "precomputed_target"]
    header.extend(f"raw_j{axis + 1}_deg" for axis in range(JOINT_COUNT))
    header.extend(f"command_j{axis + 1}_deg" for axis in range(JOINT_COUNT))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        for timestamp, batch_id, precomputed_target, raw, command in zip(
            trace.timestamps_s,
            trace.limited_batch_ids,
            trace.precomputed_target_mask,
            trace.raw_targets_deg,
            trace.command_targets_deg,
            strict=True,
        ):
            writer.writerow(
                (
                    timestamp,
                    int(batch_id),
                    int(precomputed_target),
                    *raw.tolist(),
                    *command.tolist(),
                )
            )


def _print_report(report: DemoReport) -> None:
    joint_index = report.joint_number - 1
    joint_label = f"J{report.joint_number}"
    velocity = report.trace.joint_velocity_deg_s(joint_index, report.rate_hz)
    acceleration = report.trace.joint_acceleration_deg_s2(
        joint_index, report.rate_hz
    )
    print("-" * 72)
    print(f"  机械臂: {report.side} (robot_id={report.robot_id})")
    print(f"  limited 批次生成频率: {report.limited_update_rate_hz} Hz")
    print(f"  每批生成点数: {report.rate_hz // report.limited_update_rate_hz}")
    print(f"  固定 set_target 发送频率: {report.rate_hz} Hz")
    print(
        "  Python set_target 次数: "
        f"{report.submitted_count}/{report.planned_command_count}"
    )
    print(f"  LimitedInterpolator 批次次数: {report.limited_update_count}")
    print(f"  无新点时重发上一命令次数: {report.held_target_count}")
    observed_rate = (
        "--"
        if report.observed_rate_hz is None
        else f"{report.observed_rate_hz:.1f}"
    )
    print(f"  实际完成发送频率: {observed_rate}/{report.rate_hz} Hz")
    print(f"  薄原生 PluseToServo 成功调用次数: {report.native_sent_count}")
    print(f"  丢弃发送时隙次数: {report.deadline_miss_count}")
    print(f"  发送完成晚于下一时隙次数: {report.late_send_count}")
    print(f"  最大周期滞后: {report.maximum_lateness_s * 1000.0:.3f} ms")
    print(
        f"  最大 |{joint_label} 速度|: "
        f"{np.max(np.abs(velocity)):.6f} deg/s "
        f"(限制 {report.max_velocity_deg_s:.6f})"
    )
    print(
        f"  最大 |{joint_label} 加速度|: "
        f"{np.max(np.abs(acceleration)):.6f} deg/s^2 "
        f"(限制 {report.max_acceleration_deg_s2:.6f})"
    )
    print(
        f"  最后 {joint_label} 原始/命令: "
        f"{report.trace.raw_targets_deg[-1, joint_index]:.6f} / "
        f"{report.trace.command_targets_deg[-1, joint_index]:.6f} deg"
    )


def main() -> int:
    """使用文件顶部的显式 HCX 参数执行真实直伺服测试。"""

    try:
        csv_path = _configured_csv_path()
        report = run_demo(TEST_HCX_CONFIG, TEST_CONFIG)
    except KeyboardInterrupt:
        print("\n[STOP] 收到退出请求，已停止 HCX 软件侧直伺服下发。")
        return 130
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f"[ERROR] HCX limited 直伺服测试失败: {exc}", file=sys.stderr)
        return 1

    print("=" * 72)
    print("    HCX LimitedInterpolator 分段轨迹直伺服测试")
    print("=" * 72)
    print(
        f"  {TEST_CONFIG.joint_label} 阶跃: 初始角 -> 初始角 + "
        f"{TEST_CONFIG.step_deg:g} deg，"
        f"保持 {TEST_CONFIG.step_hold_seconds:g} s 后返回初始角，"
        f"再等待 {TEST_CONFIG.return_settle_seconds:g} s"
    )
    print(
        f"  控制链: {TEST_CONFIG.joint_label} step target -> "
        f"LimitedInterpolator ({LIMITED_UPDATE_RATE_HZ} Hz) -> "
        f"hcx_sdk.set_target ({DIRECT_SERVO_RATE_HZ} Hz)"
    )
    print(
        "  独立 500 Hz 线程只发送已生成点；没有新点时重发上一条命令，"
        "不在发送线程执行限幅算法。"
    )
    print(
        f"  不经过控制器规划运动；除 {TEST_CONFIG.joint_label} 外的关节"
        "始终保持读取到的当前角度。"
    )
    _print_report(report)
    if csv_path is not None:
        try:
            write_csv(csv_path, report.trace)
        except OSError as exc:
            print(f"[ERROR] 写入 CSV 失败: {exc}", file=sys.stderr)
            return 1
        print(f"[INFO] 已写入 CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
