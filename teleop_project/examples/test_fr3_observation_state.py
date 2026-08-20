#!/usr/bin/env python3
"""独立读取 FR3 的 ``observation.state`` 关节反馈。"""

from __future__ import annotations

import time

import numpy as np

from leobot_scripts import load_recording_config
from teleop_sdk.adapters import FairinoFR3Follower
from teleop_sdk.config import load_runtime_config


PRINT_EVERY_N_SAMPLES = 1


def main() -> None:
    """以固定频率打印 FR3 的角度制关节反馈，不启动 ServoJ。"""

    if PRINT_EVERY_N_SAMPLES <= 0:
        raise ValueError("PRINT_EVERY_N_SAMPLES 必须为正数")

    runtime = load_runtime_config()
    recording = load_recording_config()
    follower = FairinoFR3Follower(runtime.fr3.robot_ip)
    period_s = 1.0 / recording.fps
    next_deadline = time.perf_counter()
    sample_index = 0

    try:
        follower.connect()
        print(f"[INFO] 开始读取 observation.state，频率 {recording.fps} Hz；按 Ctrl+C 退出")
        while True:
            capture_monotonic_ns = time.perf_counter_ns()
            state = follower.read_joint_angles_deg()
            if sample_index % PRINT_EVERY_N_SAMPLES == 0:
                print(
                    f"observation.state[{sample_index}] = "
                    f"{np.round(state, 3).tolist()} "
                    f"capture_monotonic_ns={capture_monotonic_ns}"
                )
            sample_index += 1
            next_deadline += period_s
            delay_s = next_deadline - time.perf_counter()
            if delay_s > 0.0:
                time.sleep(delay_s)
            else:
                next_deadline = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[STOP] 停止读取 FR3 状态")
    finally:
        follower.stop_servo()
        follower.disconnect()


if __name__ == "__main__":
    main()
