"""控制循环的本地定时辅助函数。"""

from __future__ import annotations

import time


def precise_sleep(duration_s: float, spin_threshold_s: float) -> None:
    """先休眠再短暂忙等，使控制周期行为不依赖任意厂商 SDK。"""
    if duration_s <= 0.0:
        return
    deadline = time.perf_counter() + duration_s
    sleep_time = duration_s - spin_threshold_s
    if sleep_time > 0.0:
        time.sleep(sleep_time)
    while time.perf_counter() < deadline:
        pass
