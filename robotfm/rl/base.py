"""强化学习模块占位（Phase 3）。

计划在 BC 预训练的 FlowMatchingPolicy 基础上做在线微调，例如：
- PPO / advantage-weighted regression
- 与 flow policy 兼容的 log-prob 或重参数化采样

当前仅定义接口，不实现具体算法。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn

from robotfm.envs.base import BaseEnv


class RLAlgorithm(ABC):
    """在线 RL 算法抽象。"""

    @abstractmethod
    def rollout(self, env: BaseEnv, policy: nn.Module, num_steps: int) -> dict[str, Any]:
        """在环境中收集 rollout 数据（obs, action, reward, done 等）。"""

    @abstractmethod
    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """用一批 rollout 数据更新策略，返回 loss 等指标。"""


class RLStub(RLAlgorithm):
    """占位实现：调用即提示尚未实现。"""

    def rollout(self, env: BaseEnv, policy: nn.Module, num_steps: int) -> dict[str, Any]:
        raise NotImplementedError(
            "RL rollout 尚未实现。请先从 FlowMatchingPolicy BC checkpoint 开始。"
        )

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        raise NotImplementedError("RL update 尚未实现。")
