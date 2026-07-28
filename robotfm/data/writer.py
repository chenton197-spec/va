"""Episode 写入器：将采集到的一条轨迹落盘为 NPZ。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from robotfm.data.schema import episode_path, image_key, meta_path, save_meta
from robotfm.types import EpisodeMeta


class EpisodeWriter:
    """负责将单条 episode 写入 run_dir，并维护 meta.json 中的 num_episodes。"""

    def __init__(self, run_dir: Path, meta: EpisodeMeta) -> None:
        self.run_dir = run_dir
        self.meta = meta
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "episodes").mkdir(parents=True, exist_ok=True)
        # 首次创建 run 时写入 meta 和创建时间
        if not meta_path(run_dir).exists():
            self.meta.created_at = datetime.now(timezone.utc).isoformat()
            save_meta(run_dir, self.meta)

    def write_episode(
        self,
        episode_index: int,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        success: bool,
        task: str = "",
    ) -> Path:
        """将一条完整轨迹写入 ep_XXXXXX.npz。

        参数中各数组时间维 T 必须一致；images 每个相机为 (T,H,W,3)。
        """
        arrays: dict[str, np.ndarray] = {
            "state": state.astype(np.float32),
            "action": action.astype(np.float32),
            "reward": reward.astype(np.float32),
            "done": done.astype(bool),
        }
        for cam, frames in images.items():
            arrays[image_key(cam)] = frames.astype(np.uint8)

        path = episode_path(self.run_dir, episode_index)
        np.savez_compressed(path, success=success, task=task, **arrays)

        self.meta.num_episodes = max(self.meta.num_episodes, episode_index + 1)
        save_meta(self.run_dir, self.meta)
        return path
