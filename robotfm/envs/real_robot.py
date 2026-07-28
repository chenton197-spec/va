"""真实机器人环境占位实现（Phase 2+）。

接入真机时需要子类化或重写本类，实现：
1. reset():  读取各相机图像 -> Observation.images；读取关节/位姿 -> Observation.state
2. step():   将 action 发送给机械臂控制器，再读取新观测
3. close():  断开相机和机械臂连接

数据格式与 PushT 完全一致，训练代码无需修改。
"""

from __future__ import annotations

import numpy as np

from robotfm.envs.base import BaseEnv
from robotfm.types import Observation, StepResult


class RealRobotEnv(BaseEnv):
    """真机后端占位。接入硬件前请勿直接实例化。"""

    def __init__(
        self,
        camera_names: list[str],
        state_dim: int,
        action_dim: int,
        fps: int = 10,
    ) -> None:
        self.observation_cameras = camera_names
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fps = fps

    def reset(self, seed: int | None = None) -> Observation:
        raise NotImplementedError(
            "RealRobotEnv 是占位类。请先实现相机驱动和机械臂接口后再使用。"
        )

    def step(self, action: np.ndarray) -> StepResult:
        raise NotImplementedError("RealRobotEnv.step 尚未实现。")

    def close(self) -> None:
        return None
