"""关节角度控制的公开数据类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class MotionResult:
    """一次规划关节运动的完成状态。"""

    robot_id: int
    sequence: int
    succeeded: bool
    cancelled: bool = False


@dataclass(frozen=True)
class DirectServoState:
    """直伺服会话的状态快照。"""

    running: bool
    faulted: bool
    sent_count: int
    error: Optional[str]
    axis_count: int


JointAngles = Tuple[float, ...]
