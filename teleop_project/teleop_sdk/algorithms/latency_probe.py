"""遥操作目标提交时序诊断。"""

from __future__ import annotations

from collections import deque

import numpy as np


class LatencyProbe:
    """按轴测量主臂明显位移到从臂目标提交变化的时间。

    该诊断不会读取从臂实际关节反馈，不能作为端到端机械响应延迟使用。
    """

    def __init__(self, rate_hz: float, threshold_deg: float, quiescent_deg: float):
        self.threshold_deg = threshold_deg
        self.quiescent_deg = quiescent_deg
        self._history: deque[np.ndarray] = deque(maxlen=max(1, int(0.2 * rate_hz)))
        self._baseline: np.ndarray | None = None
        self._active: np.ndarray | None = None
        self._onset_time: np.ndarray | None = None
        self._onset_sent: np.ndarray | None = None

    def step(
        self,
        leader_raw_deg: np.ndarray,
        sent_angles_deg: np.ndarray,
        axis_order: tuple[int, ...],
        now: float,
    ) -> list[tuple[int, float]]:
        """推进诊断并返回本帧完成的 ``(从臂轴号, 目标提交毫秒数)``。"""
        leader = np.asarray(leader_raw_deg, dtype=float)
        sent = np.asarray(sent_angles_deg, dtype=float)
        self._history.append(leader.copy())
        if self._baseline is None:
            self._baseline = leader.copy()
            self._active = np.zeros(len(axis_order), dtype=bool)
            self._onset_time = np.zeros(len(axis_order), dtype=float)
            self._onset_sent = sent.copy()
            return []
        if len(self._history) < self._history.maxlen:
            return []

        assert self._active is not None
        assert self._onset_time is not None
        assert self._onset_sent is not None
        completed: list[tuple[int, float]] = []
        window_start = self._history[0]
        for follower_axis, leader_axis in enumerate(axis_order):
            if not self._active[follower_axis]:
                if abs(leader[leader_axis] - window_start[leader_axis]) < self.quiescent_deg:
                    self._baseline[leader_axis] = leader[leader_axis]
                if abs(leader[leader_axis] - self._baseline[leader_axis]) > self.threshold_deg:
                    self._active[follower_axis] = True
                    self._onset_time[follower_axis] = now
                    self._onset_sent[follower_axis] = sent[follower_axis]
            elif abs(sent[follower_axis] - self._onset_sent[follower_axis]) > self.threshold_deg:
                completed.append((follower_axis, (now - self._onset_time[follower_axis]) * 1000.0))
                self._active[follower_axis] = False
                self._baseline[leader_axis] = leader[leader_axis]
        return completed
