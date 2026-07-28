"""PushT 鼠标遥操作：将 pygame 窗口内鼠标坐标映射为 2D 目标动作。

PushT 的动作空间是末端目标位置 (x, y)，范围约 [0, 512]^2。
鼠标在窗口中的相对位置线性映射到该范围。
"""

from __future__ import annotations

import numpy as np
import pygame

from robotfm.collect.drivers.base import TeleopDriver
from robotfm.types import Observation


class PushTMouseDriver(TeleopDriver):
    """鼠标控制 PushT 智能体目标位置。"""

    def __init__(
        self,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        hold_on_missing: bool = True,
    ) -> None:
        self.action_low = action_low if action_low is not None else np.array([0.0, 0.0], dtype=np.float32)
        self.action_high = action_high if action_high is not None else np.array([512.0, 512.0], dtype=np.float32)
        self.hold_on_missing = hold_on_missing
        self._last_action = np.array([256.0, 256.0], dtype=np.float32)
        self._window_size: tuple[int, int] | None = None

    def on_reset(self) -> None:
        self._last_action = np.array([256.0, 256.0], dtype=np.float32)

    def _map_mouse_to_action(self) -> np.ndarray | None:
        """读取鼠标像素坐标，线性映射到 action_low ~ action_high。"""
        if not pygame.get_init():
            return None
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if self._window_size is None:
            info = pygame.display.Info()
            self._window_size = (max(info.current_w, 1), max(info.current_h, 1))
        w, h = self._window_size
        x = self.action_low[0] + (mouse_x / max(w - 1, 1)) * (self.action_high[0] - self.action_low[0])
        y = self.action_low[1] + (mouse_y / max(h - 1, 1)) * (self.action_high[1] - self.action_low[1])
        action = np.array([x, y], dtype=np.float32)
        self._last_action = action
        return action

    def get_action(self, observation: Observation) -> np.ndarray | None:
        del observation  # PushT 鼠标 teleop 不依赖观测
        action = self._map_mouse_to_action()
        if action is None and self.hold_on_missing:
            return self._last_action.copy()
        return action

    def poll_events(self) -> str | None:
        """处理 pygame 键盘/窗口事件。"""
        if not pygame.get_init():
            return None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return "quit"
                if event.key == pygame.K_r:
                    return "reset"
                if event.key == pygame.K_s:
                    return "save"
        return None
