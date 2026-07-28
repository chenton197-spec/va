from __future__ import annotations

import torch.nn as nn


class MultiTaskDiTPolicy(nn.Module):
    """Phase 2 占位：带语言条件的 Multitask DiT。

    预期扩展方向：
    1. 在 `FlowMatchingPolicy` 的基础上增加文本 / 任务描述编码；
    2. 把 `task` 字段转成 task embedding 或语言 token；
    3. 让动作 DiT 在视觉、状态、语言三种条件下生成动作；
    4. 支持多任务、多机器人形态共享训练。

    当前故意保持为占位实现，避免在 Phase 1 把工程范围拉得过大。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "MultiTaskDiTPolicy is reserved for phase 2. "
            "Reuse FlowMatchingPolicy and add a text encoder + task tokens."
        )
