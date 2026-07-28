"""数据格式定义与校验。

磁盘布局（每个 run）:
    run_dir/
        meta.json          # EpisodeMeta，全局维度与相机声明
        stats.json         # state/action 的 mean/std，训练归一化用
        episodes/
            ep_000000.npz  # 单条轨迹
            ep_000001.npz
            ...

每个 NPZ 内 key 约定:
    images/<camera>  (T, H, W, 3) uint8
    state            (T, state_dim) float32
    action           (T, action_dim) float32
    reward           (T,) float32
    done             (T,) bool
    success          标量 bool
    task             标量 str（可选）
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from robotfm.types import EpisodeMeta

# NPZ 中图像 key 的前缀，例如 "images/top"
IMAGE_PREFIX = "images/"


def meta_path(run_dir: Path) -> Path:
    return run_dir / "meta.json"


def stats_path(run_dir: Path) -> Path:
    return run_dir / "stats.json"


def episode_path(run_dir: Path, episode_index: int) -> Path:
    """单条 episode 文件路径，索引零填充 6 位。"""
    return run_dir / "episodes" / f"ep_{episode_index:06d}.npz"


def save_meta(run_dir: Path, meta: EpisodeMeta) -> None:
    """将 EpisodeMeta 写入 meta.json。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    with meta_path(run_dir).open("w") as f:
        json.dump(asdict(meta), f, indent=2)


def load_meta(run_dir: Path) -> EpisodeMeta:
    """从 meta.json 加载 EpisodeMeta。"""
    with meta_path(run_dir).open() as f:
        raw = json.load(f)
    return EpisodeMeta(**raw)


def image_key(camera: str) -> str:
    """相机名 -> NPZ 内的 key，如 "top" -> "images/top"。"""
    return f"{IMAGE_PREFIX}{camera}"


def validate_episode_arrays(
    arrays: dict[str, np.ndarray],
    meta: EpisodeMeta,
) -> None:
    """校验一条 episode 内各数组形状是否与 meta 一致。

    用于写入后或训练前检查，避免维度不匹配导致隐蔽错误。
    """
    t = None
    for cam in meta.camera_names:
        key = image_key(cam)
        if key not in arrays:
            raise KeyError(f"Missing {key} in episode")
        arr = arrays[key]
        spec = meta.cameras[cam]
        expected = (spec["height"], spec["width"], spec["channels"])
        if arr.ndim != 4 or arr.shape[1:] != expected:
            raise ValueError(f"{key} shape mismatch: {arr.shape} vs (T, {expected})")
        if t is None:
            t = arr.shape[0]
        elif arr.shape[0] != t:
            raise ValueError("Camera frame counts differ across keys")

    for key, dim in (("state", meta.state_dim), ("action", meta.action_dim)):
        arr = arrays[key]
        if arr.ndim != 2 or arr.shape[1] != dim:
            raise ValueError(f"{key} must be (T, {dim}), got {arr.shape}")
        if t is not None and arr.shape[0] != t:
            raise ValueError(f"{key} length mismatch")

    for key in ("reward", "done"):
        arr = arrays[key]
        if arr.ndim != 1 or (t is not None and arr.shape[0] != t):
            raise ValueError(f"{key} must be (T,)")


def load_episode(path: Path) -> dict[str, Any]:
    """加载单个 NPZ episode，分离数组与元字段 success/task。"""
    with np.load(path, allow_pickle=True) as data:
        arrays = {k: data[k] for k in data.files}
    success = bool(arrays.pop("success", False))
    task = str(arrays.pop("task", ""))
    return {"arrays": arrays, "success": success, "task": task}


def validate_episode(path: Path, meta: EpisodeMeta) -> None:
    """加载并校验指定 episode 文件。"""
    payload = load_episode(path)
    validate_episode_arrays(payload["arrays"], meta)
