"""数据集归一化统计：计算并读写 state/action/image 的统计量。

支持模式（与 Diffusion Policy ``LinearNormalizer`` 对齐）:
- ``gaussian``: ``(x - mean) / std``
- ``gaussian_2std``: ``(x - mean) / (2 * std)``（约将 ±2σ 映到 ±1）
- ``limits``: 线性映射到 ``[-1, 1]``（按 min/max）
- ``limits_01``: 线性映射到 ``[0, 1]``（按 min/max）

图像（可选，对齐 LeRobot ACT VISUAL MEAN_STD）:
- ``image_mean`` / ``image_std``: 全局 RGB，形状 ``(3,)``，在 ``[0, 1]`` 像素上统计

训练时对 state 和 action 做归一化，推理时再反归一化回物理量。
stats.json 与 meta.json 同目录，一次 run 一份统计量。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from robotfm.data.schema import image_key, load_episode, load_meta, stats_path, validate_episode_arrays

NormMode = Literal["gaussian", "gaussian_2std", "limits", "limits_01"]
ImageNormMode = Literal["imagenet", "dataset"]
NORM_MODES: tuple[str, ...] = ("gaussian", "gaussian_2std", "limits", "limits_01")
IMAGE_NORM_MODES: tuple[str, ...] = ("imagenet", "dataset")
LIMITS_MODES: tuple[str, ...] = ("limits", "limits_01")
GAUSSIAN_MODES: tuple[str, ...] = ("gaussian", "gaussian_2std")
_RANGE_EPS = 1e-4
_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def is_limits_mode(mode: str) -> bool:
    return mode in LIMITS_MODES


def is_gaussian_mode(mode: str) -> bool:
    return mode in GAUSSIAN_MODES


def _gaussian_std_scale(mode: NormMode) -> float:
    """``gaussian_2std`` 用 2×std 作除数，使约 ±2σ 落在 ±1。"""
    return 2.0 if mode == "gaussian_2std" else 1.0


def validate_image_norm_mode(mode: str) -> ImageNormMode:
    if mode not in IMAGE_NORM_MODES:
        raise ValueError(f"image_norm_mode must be one of {IMAGE_NORM_MODES}, got {mode!r}")
    return mode  # type: ignore[return-value]


def _stats_from_state_action(
    states: list[np.ndarray],
    actions: list[np.ndarray],
    *,
    image_mean: np.ndarray | None = None,
    image_std: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if not states or not actions:
        raise ValueError("No state/action arrays to aggregate")
    state_all = np.concatenate(states, axis=0)
    action_all = np.concatenate(actions, axis=0)
    if image_mean is None or image_std is None:
        ref = imagenet_image_stats()
        image_mean = ref["image_mean"]
        image_std = ref["image_std"]
    return {
        "state_mean": state_all.mean(axis=0).astype(np.float32),
        "state_std": (state_all.std(axis=0) + 1e-6).astype(np.float32),
        "state_min": state_all.min(axis=0).astype(np.float32),
        "state_max": state_all.max(axis=0).astype(np.float32),
        "action_mean": action_all.mean(axis=0).astype(np.float32),
        "action_std": (action_all.std(axis=0) + 1e-6).astype(np.float32),
        "action_min": action_all.min(axis=0).astype(np.float32),
        "action_max": action_all.max(axis=0).astype(np.float32),
        "image_mean": np.asarray(image_mean, dtype=np.float32),
        "image_std": np.asarray(image_std, dtype=np.float32),
    }


def _merge_episode_stats_jsonl(run_dir: Path) -> dict[str, np.ndarray] | None:
    """Weighted merge of ``meta/episodes_stats.jsonl`` into robotfm stats.

    Supports single gripper (``*.gripper``), left/right-only (``*.left_gripper``
    or ``*.right_gripper``), and dual grippers (``*.left_gripper`` +
    ``*.right_gripper``). Returns None if the file is missing, incomplete, or
    the arm vector length disagrees with ``meta/info.json``.
    """
    path = Path(run_dir) / "meta" / "episodes_stats.jsonl"
    if not path.is_file():
        return None

    info_path = Path(run_dir) / "meta" / "info.json"
    gripper_suffixes: tuple[str, ...] = ()
    arm_state_dim: int | None = None
    arm_action_dim: int | None = None
    if info_path.is_file():
        with info_path.open() as f:
            features = json.load(f).get("features", {})
        from robotfm.data.lerobot_dataset import resolve_gripper_suffixes

        try:
            gripper_suffixes = resolve_gripper_suffixes(features)
        except ValueError:
            return None
        try:
            arm_state_dim = int(features["observation.state"]["shape"][0])
            arm_action_dim = int(features["action"]["shape"][0])
        except (KeyError, TypeError, IndexError, ValueError):
            arm_state_dim = None
            arm_action_dim = None

    # Accumulators for mean / min / max / second moment via count-weighted merge
    keys = (
        ("observation.state", "observation", "state"),
        ("action", "action", "action"),
    )
    acc: dict[str, dict[str, np.ndarray | float]] = {}
    for arm_key, _prefix, prefix in keys:
        acc[prefix] = {
            "count": 0.0,
            "sum": None,
            "sq_sum": None,
            "min": None,
            "max": None,
        }

    def _as_vec(
        arm: dict, grips: list[dict]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        arm_mean = np.asarray(arm["mean"], dtype=np.float64)
        arm_std = np.asarray(arm["std"], dtype=np.float64)
        arm_min = np.asarray(arm["min"], dtype=np.float64)
        arm_max = np.asarray(arm["max"], dtype=np.float64)
        count = float(arm["count"])
        if grips:
            g_mean = np.asarray([g["mean"] for g in grips], dtype=np.float64)
            g_std = np.asarray([g["std"] for g in grips], dtype=np.float64)
            g_min = np.asarray([g["min"] for g in grips], dtype=np.float64)
            g_max = np.asarray([g["max"] for g in grips], dtype=np.float64)
            mean = np.concatenate([arm_mean, g_mean])
            std = np.concatenate([arm_std, g_std])
            vmin = np.concatenate([arm_min, g_min])
            vmax = np.concatenate([arm_max, g_max])
        else:
            mean, std, vmin, vmax = arm_mean, arm_std, arm_min, arm_max
        return mean, std, vmin, vmax, count

    n_lines = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            ep_stats = raw["stats"]
            n_lines += 1
            for arm_key, feat_prefix, prefix in keys:
                if arm_key not in ep_stats:
                    return None
                grips: list[dict] = []
                for suffix in gripper_suffixes:
                    grip = ep_stats.get(f"{feat_prefix}.{suffix}")
                    if grip is None or not isinstance(grip, dict):
                        return None
                    grips.append(grip)
                mean, std, vmin, vmax, count = _as_vec(ep_stats[arm_key], grips)
                expected_arm = arm_state_dim if prefix == "state" else arm_action_dim
                if expected_arm is not None:
                    arm_len = int(np.asarray(ep_stats[arm_key]["mean"]).size)
                    if arm_len != int(expected_arm):
                        # Stale episodes_stats (e.g. 14-D dual-arm leftover on 7-D left-only).
                        return None
                bucket = acc[prefix]
                sq = np.square(std) + np.square(mean)  # E[x^2] = var + mean^2
                if bucket["sum"] is None:
                    bucket["sum"] = mean * count
                    bucket["sq_sum"] = sq * count
                    bucket["min"] = vmin.copy()
                    bucket["max"] = vmax.copy()
                    bucket["count"] = count
                else:
                    bucket["sum"] = bucket["sum"] + mean * count
                    bucket["sq_sum"] = bucket["sq_sum"] + sq * count
                    bucket["min"] = np.minimum(bucket["min"], vmin)
                    bucket["max"] = np.maximum(bucket["max"], vmax)
                    bucket["count"] = float(bucket["count"]) + count

    if n_lines == 0:
        return None

    out: dict[str, np.ndarray] = {}
    for prefix in ("state", "action"):
        bucket = acc[prefix]
        count = float(bucket["count"])
        if count <= 0 or bucket["sum"] is None:
            return None
        mean = np.asarray(bucket["sum"], dtype=np.float64) / count
        ex2 = np.asarray(bucket["sq_sum"], dtype=np.float64) / count
        var = np.maximum(ex2 - np.square(mean), 0.0)
        std = np.sqrt(var) + 1e-6
        out[f"{prefix}_mean"] = mean.astype(np.float32)
        out[f"{prefix}_std"] = std.astype(np.float32)
        out[f"{prefix}_min"] = np.asarray(bucket["min"], dtype=np.float32)
        out[f"{prefix}_max"] = np.asarray(bucket["max"], dtype=np.float32)

    ref = imagenet_image_stats()
    out["image_mean"] = ref["image_mean"]
    out["image_std"] = ref["image_std"]
    return out


def compute_lerobot_stats(run_dir: Path) -> dict[str, np.ndarray]:
    """Compute stats for a leobot image-sequence dataset root.

    Prefers weighted merge of ``meta/episodes_stats.jsonl``; falls back to
    scanning parquet state/action. Image mean/std default to ImageNet (FM
    does not require a full pixel scan).
    """
    merged = _merge_episode_stats_jsonl(run_dir)
    if merged is not None:
        return merged

    from robotfm.data.lerobot_dataset import iter_lerobot_state_action

    _meta, states, actions = iter_lerobot_state_action(run_dir)
    return _stats_from_state_action(states, actions)


def compute_stats(run_dir: Path) -> dict[str, np.ndarray]:
    """遍历 run 下所有 episode，计算 state/action/image 的全局统计量。

    返回字典:
        state_mean, state_std, state_min, state_max,
        action_mean, action_std, action_min, action_max,
        image_mean, image_std  （RGB，形状 (3,)，像素已 /255 → [0,1]）
    std 上加 1e-6 防止除零。min/max 供 ``limits`` / ``limits_01`` 模式使用。

    LeRobot image-sequence roots (``meta/info.json``) use parquet / episodes_stats
    and skip a full image pixel scan (ImageNet image stats by default).
    """
    run_dir = Path(run_dir)
    from robotfm.data.lerobot_dataset import is_lerobot_image_sequence_root

    if is_lerobot_image_sequence_root(run_dir):
        return compute_lerobot_stats(run_dir)

    meta = load_meta(run_dir)
    episode_files = sorted((run_dir / "episodes").glob("ep_*.npz"))
    if not episode_files:
        raise FileNotFoundError(f"No episodes found in {run_dir / 'episodes'}")

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for ep_file in episode_files:
        payload = load_episode(ep_file)
        validate_episode_arrays(payload["arrays"], meta)
        arrays = payload["arrays"]
        states.append(arrays["state"])
        actions.append(arrays["action"])
        for cam in meta.camera_names:
            frames = arrays[image_key(cam)].astype(np.float64) / 255.0  # (T,H,W,3)
            flat = frames.reshape(-1, 3)
            pixel_sum += flat.sum(axis=0)
            pixel_sq_sum += np.square(flat).sum(axis=0)
            pixel_count += flat.shape[0]

    if pixel_count <= 0:
        raise ValueError(f"No image pixels found in {run_dir}")

    image_mean = pixel_sum / pixel_count
    image_var = pixel_sq_sum / pixel_count - np.square(image_mean)
    image_std = np.sqrt(np.maximum(image_var, 0.0)) + 1e-6
    return _stats_from_state_action(
        states,
        actions,
        image_mean=image_mean.astype(np.float32),
        image_std=image_std.astype(np.float32),
    )


def save_stats(run_dir: Path, stats: dict[str, np.ndarray]) -> None:
    """将统计量写入 stats.json（list 格式便于 JSON 序列化）。"""
    serializable = {k: v.tolist() for k, v in stats.items()}
    with stats_path(run_dir).open("w") as f:
        json.dump(serializable, f, indent=2)


def load_stats(run_dir: Path) -> dict[str, np.ndarray]:
    """从 stats.json 加载统计量，返回 numpy float32 数组。"""
    with stats_path(run_dir).open() as f:
        raw = json.load(f)
    return {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}


def ensure_stats(
    run_dir: Path,
    norm_mode: str = "gaussian",
    *,
    require_image_stats: bool = False,
) -> dict[str, np.ndarray]:
    """加载 stats；若缺失所需键，则重新计算并保存。"""
    mode = validate_norm_mode(norm_mode)
    path = stats_path(run_dir)
    if path.exists():
        stats = load_stats(run_dir)
        need_limits = is_limits_mode(mode) and not _has_limits_keys(stats)
        need_image = require_image_stats and not _has_image_keys(stats)
        if not need_limits and not need_image:
            return stats
    stats = compute_stats(run_dir)
    save_stats(run_dir, stats)
    return stats


def validate_norm_mode(mode: str) -> NormMode:
    if mode not in NORM_MODES:
        raise ValueError(f"norm_mode must be one of {NORM_MODES}, got {mode!r}")
    return mode  # type: ignore[return-value]


def _has_limits_keys(stats: dict[str, np.ndarray]) -> bool:
    return all(
        k in stats
        for k in ("state_min", "state_max", "action_min", "action_max")
    )


def _has_image_keys(stats: dict[str, np.ndarray]) -> bool:
    return "image_mean" in stats and "image_std" in stats


def imagenet_image_stats() -> dict[str, np.ndarray]:
    """ImageNet RGB mean/std（与旧 ACT 硬编码一致）。"""
    return {
        "image_mean": _IMAGENET_MEAN.copy(),
        "image_std": _IMAGENET_STD.copy(),
    }


def resolve_image_stats(
    stats: dict[str, np.ndarray] | None,
    image_norm_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """按 ``image_norm_mode`` 返回 ``(mean, std)``，形状均为 ``(3,)``。"""
    mode = validate_image_norm_mode(image_norm_mode)
    if mode == "imagenet":
        ref = imagenet_image_stats()
        return ref["image_mean"], ref["image_std"]
    if stats is None or not _has_image_keys(stats):
        raise ValueError(
            "image_norm_mode='dataset' requires stats with image_mean/image_std; "
            "call ensure_stats(..., require_image_stats=True) first"
        )
    mean = np.asarray(stats["image_mean"], dtype=np.float32).reshape(3)
    std = np.asarray(stats["image_std"], dtype=np.float32).reshape(3)
    return mean, std


def normalize_images(
    x: np.ndarray | torch.Tensor,
    stats: dict[str, np.ndarray],
) -> np.ndarray | torch.Tensor:
    """对 ``[..., 3, H, W]`` 图像做 channel-wise ``(x - mean) / std``。"""
    if "image_mean" not in stats or "image_std" not in stats:
        raise KeyError("stats must contain image_mean and image_std")
    if isinstance(x, torch.Tensor):
        mean = torch.as_tensor(stats["image_mean"], dtype=x.dtype, device=x.device)
        std = torch.as_tensor(stats["image_std"], dtype=x.dtype, device=x.device)
        view = (*([1] * (x.ndim - 3)), 3, 1, 1)
        return (x - mean.view(*view)) / std.view(*view)
    mean = np.asarray(stats["image_mean"], dtype=np.float32).reshape(*([1] * (x.ndim - 3)), 3, 1, 1)
    std = np.asarray(stats["image_std"], dtype=np.float32).reshape(*([1] * (x.ndim - 3)), 3, 1, 1)
    return ((x - mean) / std).astype(np.float32, copy=False)


def normalize(
    x: np.ndarray,
    stats: dict[str, np.ndarray],
    *,
    prefix: str,
    mode: str = "gaussian",
) -> np.ndarray:
    """按 ``mode`` 归一化 ``prefix``（``state`` / ``action``）数据。"""
    mode = validate_norm_mode(mode)
    if is_gaussian_mode(mode):
        scale = _gaussian_std_scale(mode)
        return (x - stats[f"{prefix}_mean"]) / (scale * stats[f"{prefix}_std"])

    lo = stats[f"{prefix}_min"]
    hi = stats[f"{prefix}_max"]
    rng = hi - lo
    ignore = rng < _RANGE_EPS
    if mode == "limits_01":
        rng = np.where(ignore, 1.0, rng)
        out = (x - lo) / rng
    else:
        rng = np.where(ignore, 2.0, rng)  # scale=1 → 常数维映射到 0
        out = 2.0 * (x - lo) / rng - 1.0
    if np.any(ignore):
        out = np.where(ignore, 0.0, out)
    return out.astype(np.float32, copy=False)


def denormalize(
    x: np.ndarray | torch.Tensor,
    stats: dict[str, np.ndarray],
    *,
    prefix: str,
    mode: str = "gaussian",
) -> np.ndarray | torch.Tensor:
    """反归一化；numpy / torch 均可。"""
    mode = validate_norm_mode(mode)
    if isinstance(x, torch.Tensor):
        return _denormalize_torch(x, stats, prefix=prefix, mode=mode)
    return _denormalize_numpy(x, stats, prefix=prefix, mode=mode)


def _denormalize_numpy(
    x: np.ndarray,
    stats: dict[str, np.ndarray],
    *,
    prefix: str,
    mode: NormMode,
) -> np.ndarray:
    if is_gaussian_mode(mode):
        scale = _gaussian_std_scale(mode)
        return x * (scale * stats[f"{prefix}_std"]) + stats[f"{prefix}_mean"]

    lo = stats[f"{prefix}_min"]
    hi = stats[f"{prefix}_max"]
    rng = hi - lo
    ignore = rng < _RANGE_EPS
    if mode == "limits_01":
        rng = np.where(ignore, 1.0, rng)
        out = x * rng + lo
    else:
        rng = np.where(ignore, 2.0, rng)
        out = (x + 1.0) * 0.5 * rng + lo
    if np.any(ignore):
        out = np.where(ignore, lo, out)
    return out.astype(np.float32, copy=False)


def _denormalize_torch(
    x: torch.Tensor,
    stats: dict[str, np.ndarray],
    *,
    prefix: str,
    mode: NormMode,
) -> torch.Tensor:
    def _t(key: str) -> torch.Tensor:
        return torch.as_tensor(stats[key], dtype=x.dtype, device=x.device)

    if is_gaussian_mode(mode):
        scale = _gaussian_std_scale(mode)
        return x * (scale * _t(f"{prefix}_std")) + _t(f"{prefix}_mean")

    lo = _t(f"{prefix}_min")
    hi = _t(f"{prefix}_max")
    rng = hi - lo
    ignore = rng < _RANGE_EPS
    if mode == "limits_01":
        rng = torch.where(ignore, torch.full_like(rng, 1.0), rng)
        out = x * rng + lo
    else:
        rng = torch.where(ignore, torch.full_like(rng, 2.0), rng)
        out = (x + 1.0) * 0.5 * rng + lo
    if torch.any(ignore):
        out = torch.where(ignore, lo, out)
    return out
