"""PushT 仿真环境适配器。

将 gym_pusht 的原始观测映射为 robotfm 统一的 Observation 格式：
- pixels -> images["top"]
- agent_pos -> state

PushT 是一个 2D 推块任务：智能体（圆）需将 T 形块推到目标区域。
动作为 2D 目标位置 (x, y)，观测包含 96x96 RGB 图像和 2D 末端位置。
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import gym_pusht  # noqa: F401  # 注册 gym_pusht 环境
import numpy as np

from robotfm.envs.base import BaseEnv
from robotfm.types import Observation, StepResult


class PushTEnv(BaseEnv):
    """PushT 仿真后端，封装 gym_pusht/PushT-v0。"""

    def __init__(
        self,
        camera_names: list[str] | None = None,
        render_size: int = 96,
        max_episode_steps: int = 300,
        render_mode: str = "human",
        fps: int = 10,
        window_title: str = "PushT",
    ) -> None:
        self.observation_cameras = camera_names or ["top"]
        self.state_dim = 2
        self.action_dim = 2
        self.fps = fps
        self._render_mode = render_mode
        self._window_title = window_title
        self._caption_set = False

        # 创建 gym 环境；obs_type="pixels_agent_pos" 同时返回图像和 agent 位置
        self._env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            observation_width=render_size,
            observation_height=render_size,
            # 可视化窗口可以比观测分辨率大，便于人类 teleop
            visualization_width=max(680, render_size * 4),
            visualization_height=max(680, render_size * 4),
        )
        self._last_raw_obs: dict[str, Any] | None = None
        self._render_size = render_size

    def _ensure_raw_obs(self, raw_obs: Any) -> dict[str, Any]:
        """归一化并补全 gym_pusht 返回的观测字段。"""
        if isinstance(raw_obs, dict):
            obs = dict(raw_obs)
        else:
            obs = {}

        env = self._env.unwrapped
        if "agent_pos" not in obs:
            agent_pos = getattr(getattr(env, "agent", None), "position", None)
            if agent_pos is not None:
                obs["agent_pos"] = np.asarray(agent_pos, dtype=np.float32)
            else:
                obs["agent_pos"] = np.zeros(2, dtype=np.float32)

        if obs.get("pixels") is None:
            # gym_pusht 在 human 模式下 get_obs()/step() 会给 pixels=None；
            # 这里直接使用底层 draw + get_img 生成观测帧。
            if hasattr(env, "_draw") and hasattr(env, "_get_img"):
                screen = env._draw()  # type: ignore[attr-defined]
                obs["pixels"] = env._get_img(  # type: ignore[attr-defined]
                    screen,
                    width=self._render_size,
                    height=self._render_size,
                    render_action=False,
                )

        pixels = obs.get("pixels")
        if pixels is None:
            raise RuntimeError("PushT returned empty pixels observation; cannot build image observation.")

        # human 模式下还必须调用 render()，否则 pygame 窗口不会创建/刷新
        if self._render_mode == "human":
            self._env.render()
            if not self._caption_set:
                try:
                    import pygame

                    if pygame.display.get_init():
                        pygame.display.set_caption(self._window_title)
                        self._caption_set = True
                except Exception:
                    pass
        return obs

    def _to_observation(self, raw_obs: dict[str, Any]) -> Observation:
        """将 gym_pusht 原始 dict 观测转为统一 Observation。"""
        pixels = raw_obs["pixels"]
        if pixels.dtype != np.uint8:
            pixels = np.clip(pixels, 0, 255).astype(np.uint8)
        # 单相机映射到 images["top"]；多相机时 PushT 只有 top 视角
        images = {self.observation_cameras[0]: pixels}
        state = np.asarray(raw_obs["agent_pos"], dtype=np.float32)
        return Observation(images=images, state=state)

    def reset(self, seed: int | None = None) -> Observation:
        raw_obs, _ = self._env.reset(seed=seed)
        raw_obs = self._ensure_raw_obs(raw_obs)
        self._last_raw_obs = raw_obs
        return self._to_observation(raw_obs)

    def step(self, action: np.ndarray) -> StepResult:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        raw_obs, reward, terminated, truncated, info = self._env.step(action)
        raw_obs = self._ensure_raw_obs(raw_obs)
        self._last_raw_obs = raw_obs
        return StepResult(
            observation=self._to_observation(raw_obs),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
        )

    def render_rgb(self) -> np.ndarray | None:
        """返回当前可视化帧，用于评估录像。"""
        frame = self._env.render()
        if frame is None:
            return None
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def close(self) -> None:
        self._env.close()
