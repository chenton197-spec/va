"""环境工厂：根据配置 backend 字段创建对应的环境实例。

新增后端时在此注册，例如：
    if cfg.backend == "my_robot":
        return MyRobotEnv(...)
"""

from __future__ import annotations

from robotfm.config import RobotFMConfig
from robotfm.envs.base import BaseEnv
from robotfm.envs.pusht import PushTEnv
from robotfm.envs.real_robot import RealRobotEnv


def make_env(
    cfg: RobotFMConfig,
    render_mode: str | None = None,
    window_title: str | None = None,
) -> BaseEnv:
    """根据 RobotFMConfig 创建环境。

    参数:
        cfg: 顶层配置
        render_mode: 可覆盖 cfg.env.render_mode（评估时常用 "rgb_array"）
        window_title: human 模式下 pygame 窗口标题

    返回:
        实现了 BaseEnv 接口的环境实例
    """
    mode = render_mode or cfg.env.render_mode
    if cfg.backend == "pusht":
        title = window_title or "PushT"
        return PushTEnv(
            camera_names=cfg.cameras,
            render_size=cfg.env.render_size,
            max_episode_steps=cfg.env.max_episode_steps,
            render_mode=mode,
            fps=cfg.fps,
            window_title=title,
        )
    if cfg.backend == "real_robot":
        return RealRobotEnv(
            camera_names=cfg.cameras,
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            fps=cfg.fps,
        )
    raise ValueError(f"未知 backend: {cfg.backend}")
