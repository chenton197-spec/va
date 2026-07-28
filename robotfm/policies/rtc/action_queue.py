"""Action queue management for Real-Time Chunking (RTC).

Ported from LeRobot ActionQueue with full merge / leftover semantics.
"""

from __future__ import annotations

import logging
from threading import Lock

import torch
from torch import Tensor

from robotfm.policies.rtc.configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


class ActionQueue:
    """Thread-safe queue for managing action chunks in real-time control.

    Modes:
    1. RTC-enabled: Replaces the entire queue, accounting for inference delay
    2. RTC-disabled: Appends new actions to the queue
    """

    def __init__(self, cfg: RTCConfig):
        self.queue = None
        self.original_queue = None
        self.lock = Lock()
        self.last_index = 0
        self.cfg = cfg

    def get(self) -> Tensor | None:
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            action = self.queue[self.last_index]
            self.last_index += 1
            return action.clone()

    def clear(self) -> None:
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.last_index = 0

    def qsize(self) -> int:
        with self.lock:
            if self.queue is None:
                return 0
            return len(self.queue) - self.last_index

    def empty(self) -> bool:
        with self.lock:
            if self.queue is None:
                return True
            return len(self.queue) - self.last_index <= 0

    def get_action_index(self) -> int:
        with self.lock:
            return self.last_index

    def get_left_over(self) -> Tensor | None:
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :].clone()

    def get_processed_left_over(self) -> Tensor | None:
        with self.lock:
            if self.queue is None:
                return None
            return self.queue[self.last_index :].clone()

    def merge(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = None,
    ):
        with self.lock:
            delay = self._check_and_resolve_delays(real_delay, action_index_before_inference)
            if self.cfg.enabled:
                self._replace_actions_queue(original_actions, processed_actions, delay)
                return
            self._append_actions_queue(original_actions, processed_actions)

    def _replace_actions_queue(self, original_actions: Tensor, processed_actions: Tensor, real_delay: int):
        clamped_delay = max(0, min(real_delay, len(original_actions), len(processed_actions)))
        self.original_queue = original_actions[clamped_delay:].clone()
        self.queue = processed_actions[clamped_delay:].clone()
        logger.debug("original_actions shape: %s", self.original_queue.shape)
        logger.debug("processed_actions shape: %s", self.queue.shape)
        logger.debug("real_delay: %d, clamped_delay: %d", real_delay, clamped_delay)
        self.last_index = 0

    def _append_actions_queue(self, original_actions: Tensor, processed_actions: Tensor):
        if self.queue is None:
            self.original_queue = original_actions.clone()
            self.queue = processed_actions.clone()
            return

        self.original_queue = torch.cat([self.original_queue, original_actions.clone()])
        self.original_queue = self.original_queue[self.last_index :]

        self.queue = torch.cat([self.queue, processed_actions.clone()])
        self.queue = self.queue[self.last_index :]

        self.last_index = 0

    def _check_and_resolve_delays(
        self, real_delay: int, action_index_before_inference: int | None = None
    ) -> int:
        effective_delay = max(0, real_delay)

        if action_index_before_inference is not None:
            indexes_diff = max(0, self.last_index - action_index_before_inference)
            if indexes_diff != real_delay:
                logger.warning(
                    "Indexes diff is not equal to real delay. indexes_diff=%d, real_delay=%d",
                    indexes_diff,
                    real_delay,
                )
                return real_delay

        return effective_delay
