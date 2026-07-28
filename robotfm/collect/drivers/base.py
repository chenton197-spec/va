"""遥操作驱动抽象：将人类输入映射为环境 action。

不同平台实现不同子类：
- PushTMouseDriver: 鼠标位置 -> 2D 目标点
- 未来: RealTeleopDriver 手柄/键盘 -> 关节或末端指令
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from robotfm.types import Observation


class TeleopDriver(ABC):
    """遥操作驱动基类。"""

    @abstractmethod
    def get_action(self, observation: Observation) -> np.ndarray | None:
        """根据当前观测（及内部输入设备状态）返回本步要执行的动作。

        返回 None 表示本步不发送新动作（可配合 hold 上一动作）。
        """

    def on_reset(self) -> None:
        """环境 reset 时调用，用于重置驱动内部状态。"""
        return None

    def on_episode_end(self) -> None:
        """一条 episode 结束保存或丢弃后调用。"""
        return None

    def poll_events(self) -> str | None:
        """轮询 UI 事件，返回控制令牌或 None。

        令牌:
            'quit'  - 退出采集
            'reset' - 丢弃当前轨迹并重置环境
            'save'  - 尽快保存当前轨迹
        """
        return None
