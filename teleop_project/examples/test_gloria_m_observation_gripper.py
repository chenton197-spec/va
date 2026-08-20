#!/usr/bin/env python3
"""独立读取 Gloria-M 的 ``observation.gripper`` 反馈。"""

from __future__ import annotations

import time

from leobot_scripts import CallableGripperFeedbackSource, load_recording_config
from teleop_sdk.adapters import GloriaMGripperFollower
from teleop_sdk.config import load_runtime_config


PRINT_EVERY_N_SAMPLES = 1
# 保持适配器连接后的默认使能状态；需要完全失能再手动拨动时改为 True。
DISABLE_MOTOR_FOR_MANUAL_TEST = False


def main() -> None:
    """以固定频率打印归一化 Gloria-M 开合反馈，不发送开合目标。"""

    if PRINT_EVERY_N_SAMPLES <= 0:
        raise ValueError("PRINT_EVERY_N_SAMPLES 必须为正数")

    runtime = load_runtime_config()
    recording = load_recording_config()
    if not runtime.gloria_m.enabled:
        raise ValueError("请先在 teleop.yaml 中将 gloria_m.enabled 设为 true")
    gripper = GloriaMGripperFollower(runtime.gloria_m)
    # 与 demo 使用同一个回调：每次读取消费上一帧回包并请求下一帧状态。
    source = CallableGripperFeedbackSource(gripper.read_normalized_opening)
    period_s = 1.0 / recording.fps
    next_deadline = time.perf_counter()
    sample_index = 0

    try:
        gripper.connect()
        if DISABLE_MOTOR_FOR_MANUAL_TEST:
            gripper.disable()
            print("[INFO] Gloria-M 已失能，可手动开合；每次采样都会请求新的位置反馈")
        print(f"[INFO] 开始读取 observation.gripper，频率 {recording.fps} Hz；按 Ctrl+C 退出")
        while True:
            sample = source.read_gripper_opening()
            if sample_index % PRINT_EVERY_N_SAMPLES == 0:
                if sample is None:
                    print(f"observation.gripper[{sample_index}] = unavailable")
                else:
                    print(
                        f"observation.gripper[{sample_index}] = {sample.value[0]:.4f} "
                        f"capture_monotonic_ns={sample.capture_monotonic_ns}"
                    )
            sample_index += 1
            next_deadline += period_s
            delay_s = next_deadline - time.perf_counter()
            if delay_s > 0.0:
                time.sleep(delay_s)
            else:
                next_deadline = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[STOP] 停止读取 Gloria-M 夹爪反馈")
    finally:
        gripper.disable()
        gripper.disconnect()


if __name__ == "__main__":
    main()
