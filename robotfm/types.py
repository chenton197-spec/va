"""核心数据结构定义。

本模块定义仿真与真机共用的观测、步进结果、数据集元信息。
所有环境后端（PushT、RealRobot）都必须输出这些统一类型，
这样采集、训练、评估代码就不需要感知具体机器人类型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Observation:
    """单步观测，仿真与真机统一格式。

    属性:
        images: 多相机图像字典，key 为相机名（如 "top"、"wrist"），
                value 为 HWC uint8 数组，形状 (H, W, 3)
        state:  低维本体感知，如关节角、末端位姿、PushT 中的 agent_pos，
                形状 (state_dim,)
        timestamp: 可选时间戳，真机采集时可用于对齐多传感器
    """

    images: dict[str, np.ndarray]
    state: np.ndarray
    timestamp: float | None = None
    depths: dict[str, np.ndarray] | None = None

    def validate(self, camera_names: list[str], state_dim: int) -> None:
        """校验观测是否符合 meta.json 中声明的相机和状态维度。"""
        for name in camera_names:
            if name not in self.images:
                raise KeyError(f"Missing camera '{name}' in observation.images")
            img = self.images[name]
            if img.ndim != 3 or img.shape[-1] != 3:
                raise ValueError(f"Camera '{name}' must be HWC, got {img.shape}")
        if self.state.shape != (state_dim,):
            raise ValueError(f"state must be ({state_dim},), got {self.state.shape}")


@dataclass
class StepResult:
    """环境执行一步动作后的返回结果。

    与 gymnasium 的 (obs, reward, terminated, truncated, info) 对应，
    但观测已统一封装为 Observation 类型。
    """

    observation: Observation
    reward: float
    terminated: bool  # 任务自然结束（如 PushT 成功推到目标）
    truncated: bool   # 超时等非自然结束
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """episode 是否结束（terminated 或 truncated 任一为 True）。"""
        return self.terminated or self.truncated


@dataclass
class EpisodeMeta:
    """一次数据采集运行（run）的全局元信息。

    保存在 run_dir/meta.json，训练时据此知道：
    - 有哪些相机、分辨率多少
    - state/action 各多少维
    - 采集频率、任务描述等

    同一 run 内所有 episode 的相机集合和维度必须一致。
    """

    backend: str          # 后端类型，如 "pusht"、"real_robot"
    embodiment: str       # 机器人形态标识，如 "pusht_sim"、"so101"
    fps: int              # 采集/控制频率（Hz）
    cameras: dict[str, dict[str, int]]  # 相机名 -> {height, width, channels}
    state_dim: int
    action_dim: int
    state_names: list[str]   # 状态各维语义名，便于调试和可视化
    action_names: list[str]  # 动作各维语义名
    num_episodes: int = 0    # 已保存的 episode 数量（写入时自动更新）
    task: str = ""           # 任务描述，预留给 Multitask DiT 语言条件
    created_at: str = ""     # ISO 格式创建时间

    @property
    def camera_names(self) -> list[str]:
        """返回相机名列表，顺序与 cameras 字典 key 一致。"""
        return list(self.cameras.keys())
