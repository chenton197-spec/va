#!/usr/bin/env python3
"""持续读取双 OpenArm Mini 状态，并输出关节滤波和虚拟弹簧阻尼结果。

每侧主臂分别执行角度连续化、One Euro 滤波、固定低通和弹簧阻尼/前瞻预测。该程序
只用于观察主臂侧算法输出：不连接从臂、不应用主从轴映射、不发送任何运动命令。
夹爪保持原始归一化开合值，不参与滤波或弹簧阻尼。

  - raw_deg：原始主臂角度
  - unwrapped_deg：消除 ±180° 跨越跳变后的角度
  - one_euro_deg：一级滤波结果
  - low_pass_deg：一级加二级低通后的结果
  - spring_output_deg：在二级滤波结果上经过弹簧阻尼及可选前瞻后的最终观测值
  - gripper_raw：夹爪原始值，未滤波


"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from teleop_sdk.adapters import OpenArmMiniLeaderArm
from teleop_sdk.algorithms import AngleUnwrapper, SpringDamper
from teleop_sdk.config import OpenArmMiniLeaderConfig, TeleopConfig, load_runtime_config
from teleop_sdk.filters import LowPassFilter, OneEuroFilter

# 与 TeleopConfig 的默认控制频率一致，保证默认 spring_omega=20 的离散弹簧模型稳定。
READ_RATE_HZ = 125.0
READ_TIMEOUT_S = 0.1
# 125 Hz 下逐帧打印会拖慢实际读取循环；每 10 帧打印一次，便于观察且不干扰采样。
PRINT_EVERY_N_SAMPLES = 10
VIRTUAL_JOINT_LIMIT_DEG = 360.0


@dataclass(frozen=True)
class _ProcessedJointFrame:
    """单侧主臂同一采样帧经过各处理阶段的关节结果。"""

    raw_joint_angles: np.ndarray
    unwrapped_joint_angles: np.ndarray
    one_euro_joint_angles: np.ndarray
    low_pass_joint_angles: np.ndarray
    spring_joint_angles: np.ndarray
    gripper_opening: float


class _JointProcessingPipeline:
    """一侧主臂独占的关节滤波和虚拟弹簧阻尼状态。"""

    def __init__(self, n_joints: int, config: TeleopConfig):
        if n_joints <= 0:
            raise ValueError("主臂关节数量必须为正数")

        self._config = config
        self._unwrapper = AngleUnwrapper(n_joints)
        self._one_euro: OneEuroFilter | None = None
        self._low_pass: LowPassFilter | None = None
        if config.filter_enabled:
            self._one_euro = OneEuroFilter(
                n_joints=n_joints,
                mincutoff=config.filter_mincutoff_hz,
                beta=config.filter_beta,
            )
            self._low_pass = LowPassFilter(
                n_joints=n_joints,
                cutoff_hz=config.tremor_cutoff_hz,
            )
        # 仅用于可视化。没有从臂限位时使用宽松的虚拟范围，避免掩盖滤波结果。
        self._spring_damper: SpringDamper | None = None
        if config.spring_enabled:
            self._spring_damper = SpringDamper(
                rate_hz=READ_RATE_HZ,
                omega=config.spring_omega,
                jump_threshold_deg=config.jump_threshold_deg,
                max_accel_deg_s2=config.max_accel_deg_s2,
                max_vel_deg_s=config.max_vel_deg_s,
                min_angles_deg=np.full(n_joints, -VIRTUAL_JOINT_LIMIT_DEG),
                max_angles_deg=np.full(n_joints, VIRTUAL_JOINT_LIMIT_DEG),
            )

    def step(
        self, raw_joint_angles: np.ndarray, timestamp: float
    ) -> tuple[np.ndarray, ...]:
        """返回原始、连续化、一级、二级和最终弹簧输出。"""

        raw = np.asarray(raw_joint_angles, dtype=float).copy()
        unwrapped = self._unwrapper.step(raw)
        one_euro = (
            self._one_euro.step(unwrapped, timestamp)
            if self._one_euro is not None
            else unwrapped.copy()
        )
        low_pass = (
            self._low_pass.step(one_euro, timestamp)
            if self._low_pass is not None
            else one_euro.copy()
        )
        spring = low_pass.copy()
        if self._spring_damper is not None:
            spring = self._spring_damper.step(low_pass, low_pass)
            spring = self._spring_damper.predict(
                self._config.predict_lookahead_ms / 1000.0
            )
        return raw, unwrapped, one_euro, low_pass, spring


def _validate_config(config: OpenArmMiniLeaderConfig) -> None:
    """在打开串口前检查状态读取所需的配置和标定文件。"""

    if not config.port_left.strip() or not config.port_right.strip():
        raise ValueError("openarm_mini.port_left 和 port_right 均不能为空")
    if config.port_left == config.port_right:
        raise ValueError("openarm_mini.port_left 和 port_right 必须是两条不同串口")
    if not config.calibration_path.strip():
        raise ValueError("请在 teleop.yaml 设置 openarm_mini.calibration_path")
    if not isinstance(config.baudrate, int) or config.baudrate <= 0:
        raise ValueError("openarm_mini.baudrate 必须为正整数")
    if not Path(config.calibration_path).expanduser().is_file():
        raise ValueError(
            f"找不到 OpenArm Mini 标定文件: {config.calibration_path}；请先运行标定示例"
        )


def _create_leaders(config: OpenArmMiniLeaderConfig) -> list[OpenArmMiniLeaderArm]:
    """创建两个严格只读的 OpenArm Mini 状态读取器。"""

    return [
        OpenArmMiniLeaderArm(
            port=config.port_left,
            calibration_path=config.calibration_path,
            side="left",
            baudrate=config.baudrate,
            read_only=True,
        ),
        OpenArmMiniLeaderArm(
            port=config.port_right,
            calibration_path=config.calibration_path,
            side="right",
            baudrate=config.baudrate,
            read_only=True,
        ),
    ]


def _process_frame(
    frame: tuple[np.ndarray, float] | None,
    pipeline: _JointProcessingPipeline,
    timestamp: float,
) -> _ProcessedJointFrame | None:
    """将一帧读取结果送入对应侧的关节处理管线。"""

    if frame is None:
        return None
    joint_angles, gripper_opening = frame
    raw, unwrapped, one_euro, low_pass, spring = pipeline.step(joint_angles, timestamp)
    return _ProcessedJointFrame(
        raw_joint_angles=raw,
        unwrapped_joint_angles=unwrapped,
        one_euro_joint_angles=one_euro,
        low_pass_joint_angles=low_pass,
        spring_joint_angles=spring,
        gripper_opening=float(gripper_opening),
    )


def _format_processed_frame(frame: _ProcessedJointFrame | None) -> str:
    """将处理结果格式化为多行终端输出。"""

    if frame is None:
        return "unavailable"
    return (
        f"raw_deg={np.round(frame.raw_joint_angles, 2).tolist()}\n"
        f"    unwrapped_deg={np.round(frame.unwrapped_joint_angles, 2).tolist()}\n"
        f"    one_euro_deg={np.round(frame.one_euro_joint_angles, 2).tolist()}\n"
        f"    low_pass_deg={np.round(frame.low_pass_joint_angles, 2).tolist()}\n"
        f"    spring_output_deg={np.round(frame.spring_joint_angles, 2).tolist()}\n"
        f"    gripper_raw={frame.gripper_opening:.3f}"
    )


def main() -> int:
    """持续打印左右主臂经过滤波和虚拟弹簧阻尼后的关节输出。"""

    if READ_RATE_HZ <= 0.0:
        raise ValueError("READ_RATE_HZ 必须为正数")
    if PRINT_EVERY_N_SAMPLES <= 0:
        raise ValueError("PRINT_EVERY_N_SAMPLES 必须为正整数")

    runtime = load_runtime_config()
    openarm_config = runtime.openarm_mini
    teleop_config = runtime.teleop
    leaders: list[OpenArmMiniLeaderArm] = []

    try:
        _validate_config(openarm_config)
    except ValueError as exc:
        print(f"[ERROR] OpenArm Mini 滤波状态读取配置无效: {exc}")
        return 2

    period_s = 1.0 / READ_RATE_HZ
    next_deadline = time.perf_counter()
    sample_index = 0

    try:
        leaders = _create_leaders(openarm_config)
        left, right = leaders
        if left.joint_count != right.joint_count:
            raise ValueError("左右 OpenArm Mini 关节数量不一致")
        left_pipeline = _JointProcessingPipeline(left.joint_count, teleop_config)
        right_pipeline = _JointProcessingPipeline(right.joint_count, teleop_config)
        left.connect()
        right.connect()

        print("=" * 60)
        print("    OpenArm Mini 双主臂滤波与弹簧阻尼观测")
        print("=" * 60)
        print("  流程: 角度连续化 -> One Euro -> 固定低通 -> 弹簧阻尼/前瞻预测")
        print("  夹爪保持原始归一化值；不参与滤波或弹簧阻尼。")
        print("  本程序只读取电机位置；不会连接从臂或发送运动命令。")
        print(f"  读取频率: {READ_RATE_HZ:.1f} Hz；按 Ctrl+C 退出")
        print("-" * 60)

        while True:
            capture_monotonic_ns = time.perf_counter_ns()
            timestamp = capture_monotonic_ns / 1_000_000_000.0
            left_frame = _process_frame(
                left.read_joint_angles_and_gripper_opening(READ_TIMEOUT_S),
                left_pipeline,
                timestamp,
            )
            right_frame = _process_frame(
                right.read_joint_angles_and_gripper_opening(READ_TIMEOUT_S),
                right_pipeline,
                timestamp,
            )
            if sample_index % PRINT_EVERY_N_SAMPLES == 0:
                print(
                    f"state[{sample_index}] capture_monotonic_ns={capture_monotonic_ns}\n"
                    f"  left:  {_format_processed_frame(left_frame)}\n"
                    f"  right: {_format_processed_frame(right_frame)}",
                    flush=True,
                )

            sample_index += 1
            next_deadline += period_s
            delay_s = next_deadline - time.perf_counter()
            if delay_s > 0.0:
                time.sleep(delay_s)
            else:
                next_deadline = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[STOP] 停止读取 OpenArm Mini 滤波状态")
        return 130
    except Exception as exc:
        print(f"[ERROR] OpenArm Mini 滤波状态读取异常: {exc}")
        return 1
    finally:
        for leader in reversed(leaders):
            leader.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
