#!/usr/bin/env python3
"""用 Alicia-D 末端扳机直接控制 Gloria-M 夹爪。"""

from __future__ import annotations

import time

from leobot_scripts import CallableGripperFeedbackSource, load_recording_config
from teleop_sdk.adapters import AliciaLeaderArm, GloriaMGripperFollower
from teleop_sdk.config import load_runtime_config


PRINT_EVERY_N_SAMPLES = 30
READ_TIMEOUT_S = 0.1


def _clamp_opening(value: float) -> float:
    """将 Alicia-D 触发器读数限制在统一的 0-1 开合范围。"""

    return max(0.0, min(1.0, value))


def main() -> None:
    """按控制频率下发扳机目标，并按采集频率读取实际夹爪反馈。"""

    if PRINT_EVERY_N_SAMPLES <= 0 or READ_TIMEOUT_S <= 0.0:
        raise ValueError("PRINT_EVERY_N_SAMPLES 和 READ_TIMEOUT_S 必须为正数")

    runtime = load_runtime_config()
    recording = load_recording_config()
    if not runtime.gloria_m.enabled:
        raise ValueError("请先在 teleop.yaml 中将 gloria_m.enabled 设为 true")

    leader = AliciaLeaderArm(
        port=runtime.alicia.port,
        gripper_type=runtime.alicia.gripper_type,
        connect_retries=runtime.alicia.connect_retries,
        connect_retry_delay_s=runtime.alicia.connect_retry_delay_s,
    )
    gripper = GloriaMGripperFollower(runtime.gloria_m)
    feedback_source = CallableGripperFeedbackSource(gripper.read_normalized_opening)
    control_period_s = 1.0 / runtime.teleop.rate_hz
    feedback_period_s = 1.0 / recording.fps
    next_control_deadline = time.perf_counter()
    next_feedback_deadline = next_control_deadline
    control_index = 0
    feedback_index = 0

    try:
        leader.connect()
        gripper.connect()
        print(
            "[INFO] Alicia-D 末端扳机控制 Gloria-M："
            f"控制 {runtime.teleop.rate_hz:g} Hz，反馈采样 {recording.fps} Hz；按 Ctrl+C 退出"
        )
        while True:
            trigger_opening = leader.read_gripper_opening(READ_TIMEOUT_S)
            if trigger_opening is None:
                if control_index % PRINT_EVERY_N_SAMPLES == 0:
                    print(f"[WARN] Alicia-D 扳机读数不可用，跳过第 {control_index} 次控制")
            else:
                target_opening = _clamp_opening(float(trigger_opening))
                gripper.send_normalized(target_opening)

            now = time.perf_counter()
            if now >= next_feedback_deadline:
                feedback = feedback_source.read_gripper_opening()
                if feedback is None:
                    feedback_text = "unavailable"
                else:
                    feedback_text = (
                        f"{feedback.value[0]:.4f} "
                        f"capture_monotonic_ns={feedback.capture_monotonic_ns}"
                    )
                print(f"observation.gripper[{feedback_index}] = {feedback_text}")
                feedback_index += 1
                next_feedback_deadline += feedback_period_s
                if next_feedback_deadline <= now:
                    next_feedback_deadline = now + feedback_period_s

            control_index += 1
            next_control_deadline += control_period_s
            delay_s = next_control_deadline - time.perf_counter()
            if delay_s > 0.0:
                time.sleep(delay_s)
            else:
                next_control_deadline = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[STOP] 停止 Alicia-D 末端扳机夹爪控制")
    finally:
        gripper.disable()
        gripper.disconnect()
        leader.disconnect()


if __name__ == "__main__":
    main()
