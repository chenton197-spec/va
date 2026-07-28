"""环境抽象基类。

所有后端（PushT 仿真、真实机器人）都必须实现 BaseEnv 接口。
采集循环、评估脚本只依赖此接口，不感知具体后端实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from robotfm.types import Observation, StepResult


class BaseEnv(ABC):
    """机器人环境统一接口（类似 gymnasium.Env，但观测格式统一）。

    子类必须设置以下类属性或在 __init__ 中赋值：
        observation_cameras: 相机名列表
        state_dim:           状态向量维度
        action_dim:          动作向量维度
        fps:                 控制频率（Hz）
    """

    observation_cameras: list[str]
    state_dim: int
    action_dim: int
    fps: int

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        """重置环境，返回初始观测。"""

    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        """执行一步动作，返回 StepResult（含新观测、奖励、结束标志）。"""

    @abstractmethod
    def close(self) -> None:
        """释放环境资源（关闭窗口、断开硬件连接等）。"""

    def render_rgb(self) -> np.ndarray | None:
        """可选：返回当前帧 RGB 图像 (H,W,3) uint8，用于录像或调试。

        默认返回 None；子类可覆盖以支持评估时保存视频。
        """
        return None
