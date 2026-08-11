"""PyTorch Dataset：按帧索引采样，构造 action chunk 训练 batch。

每个样本对应 (episode_id, 时间步 t)：
- 输入：过去 n_obs_steps 帧的多相机图像 + 状态
- 标签：从 t 开始的 horizon 步动作序列（不足则 padding + mask）

重要：索引的是「每一帧」，不是整段 episode。
默认丢弃每个 episode 末尾 ``n_action_steps`` 帧，保证每条样本至少有
``n_action_steps`` 步真实未来动作（与闭环实际执行长度对齐），避免
短 chunk + pad 经 UNet 时序卷积泄漏，干扰精细对齐。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from robotfm.data.schema import image_key, load_episode, load_meta
from robotfm.data.stats import is_limits_mode, normalize, validate_norm_mode


def build_episode_dataset(run_dir: Path, **kwargs):
    """Build NPZ ``EpisodeDataset`` or LeRobot image-sequence dataset.

    Auto-detects ``run_dir/meta/info.json`` with ``image_storage=image_sequence``.
    """
    from robotfm.data.lerobot_dataset import (
        LeRobotImageSequenceDataset,
        is_lerobot_image_sequence_root,
    )

    run_dir = Path(run_dir)
    if is_lerobot_image_sequence_root(run_dir):
        return LeRobotImageSequenceDataset(run_dir=run_dir, **kwargs)
    return EpisodeDataset(run_dir=run_dir, **kwargs)


def resize_images(images: torch.Tensor, size: int | None) -> torch.Tensor:
    """双线性缩放到 ``size×size``。

    支持形状:
      - (T, 3, H, W)
      - (Cams, T, 3, H, W)
    """
    if size is None:
        return images
    *lead, c, h, w = images.shape
    if h == size and w == size:
        return images
    flat = images.reshape(-1, c, h, w)
    flat = torch.nn.functional.interpolate(
        flat, size=(size, size), mode="bilinear", align_corners=False
    )
    return flat.reshape(*lead, c, size, size)


def crop_images(images: torch.Tensor, crop_size: int | None, random: bool) -> torch.Tensor:
    """对 CHW 图像张量做空间裁剪。

    支持形状:
      - (T, 3, H, W)
      - (Cams, T, 3, H, W)

    ``crop_size is None`` 或等于 H/W 时原样返回。
    """
    if crop_size is None:
        return images
    *lead, c, h, w = images.shape
    if h == crop_size and w == crop_size:
        return images
    if h < crop_size or w < crop_size:
        raise ValueError(f"Cannot crop {h}x{w} to {crop_size}")

    if random:
        # 训练：随机裁剪做数据增强
        top = int(torch.randint(0, h - crop_size + 1, (1,)).item())
        left = int(torch.randint(0, w - crop_size + 1, (1,)).item())
    else:
        # 评估：中心裁剪，保证可复现
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2

    return images[..., top : top + crop_size, left : left + crop_size]


def color_jitter_images(
    images: torch.Tensor,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
) -> torch.Tensor:
    """对 [0,1] CHW 图像做光度增强；同一条样本共用一组随机因子。

    支持形状:
      - (T, 3, H, W)
      - (Cams, T, 3, H, W)

    强度为相对幅度（如 brightness=0.2 → 因子均匀采样自 [0.8, 1.2]）。
    全部为 0 时原样返回。评估路径不应调用本函数。
    """
    if brightness <= 0 and contrast <= 0 and saturation <= 0 and hue <= 0:
        return images

    from torchvision.transforms import functional as TF

    *lead, c, h, w = images.shape
    flat = images.reshape(-1, c, h, w)

    if brightness > 0:
        factor = float(torch.empty(1).uniform_(max(0.0, 1.0 - brightness), 1.0 + brightness))
        flat = TF.adjust_brightness(flat, factor)
    if contrast > 0:
        factor = float(torch.empty(1).uniform_(max(0.0, 1.0 - contrast), 1.0 + contrast))
        flat = TF.adjust_contrast(flat, factor)
    if saturation > 0:
        factor = float(torch.empty(1).uniform_(max(0.0, 1.0 - saturation), 1.0 + saturation))
        flat = TF.adjust_saturation(flat, factor)
    if hue > 0:
        # torchvision hue 因子范围 (-0.5, 0.5)
        factor = float(torch.empty(1).uniform_(-hue, hue))
        flat = TF.adjust_hue(flat, max(-0.5, min(0.5, factor)))

    return flat.clamp(0.0, 1.0).reshape(*lead, c, h, w)


def crop_images_batch(
    images: torch.Tensor, crop_size: int | None, random: bool
) -> torch.Tensor:
    """Batched spatial crop for ``(B, Cams, T, 3, H, W)``.

    Each batch item gets its own crop window; cams/T within an item share it.
    """
    if crop_size is None:
        return images
    if images.ndim != 6:
        raise ValueError(f"Expected (B,Cams,T,3,H,W), got shape {tuple(images.shape)}")
    b, cams, t, c, h, w = images.shape
    if h == crop_size and w == crop_size:
        return images
    if h < crop_size or w < crop_size:
        raise ValueError(f"Cannot crop {h}x{w} to {crop_size}")

    if random:
        tops = torch.randint(0, h - crop_size + 1, (b,), device=images.device)
        lefts = torch.randint(0, w - crop_size + 1, (b,), device=images.device)
    else:
        tops = torch.full(
            (b,), (h - crop_size) // 2, device=images.device, dtype=torch.long
        )
        lefts = torch.full(
            (b,), (w - crop_size) // 2, device=images.device, dtype=torch.long
        )

    out = images.new_empty(b, cams, t, c, crop_size, crop_size)
    for i in range(b):
        top = int(tops[i])
        left = int(lefts[i])
        out[i] = images[i, ..., top : top + crop_size, left : left + crop_size]
    return out


def color_jitter_images_batch(
    images: torch.Tensor,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
) -> torch.Tensor:
    """Batched photometric jitter for ``(B, Cams, T, 3, H, W)`` on any device.

    Each batch item samples one factor set shared across its cams/T.
    """
    if brightness <= 0 and contrast <= 0 and saturation <= 0 and hue <= 0:
        return images
    if images.ndim != 6:
        raise ValueError(f"Expected (B,Cams,T,3,H,W), got shape {tuple(images.shape)}")

    from torchvision.transforms import functional as TF

    b, cams, t, c, h, w = images.shape
    out = images
    # Apply per-sample so factors match CPU EpisodeDataset semantics.
    pieces: list[torch.Tensor] = []
    for i in range(b):
        flat = out[i].reshape(-1, c, h, w)
        if brightness > 0:
            factor = float(
                torch.empty(1, device=images.device).uniform_(
                    max(0.0, 1.0 - brightness), 1.0 + brightness
                )
            )
            flat = TF.adjust_brightness(flat, factor)
        if contrast > 0:
            factor = float(
                torch.empty(1, device=images.device).uniform_(
                    max(0.0, 1.0 - contrast), 1.0 + contrast
                )
            )
            flat = TF.adjust_contrast(flat, factor)
        if saturation > 0:
            factor = float(
                torch.empty(1, device=images.device).uniform_(
                    max(0.0, 1.0 - saturation), 1.0 + saturation
                )
            )
            flat = TF.adjust_saturation(flat, factor)
        if hue > 0:
            factor = float(
                torch.empty(1, device=images.device).uniform_(-hue, hue)
            )
            flat = TF.adjust_hue(flat, max(-0.5, min(0.5, factor)))
        pieces.append(flat.clamp(0.0, 1.0).reshape(cams, t, c, h, w))
    return torch.stack(pieces, dim=0)


def apply_image_augments_batch(
    images: torch.Tensor,
    *,
    crop_size: int | None,
    random_crop: bool,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
) -> torch.Tensor:
    """GPU/CPU batch crop (+ optional color jitter when ``random_crop``)."""
    images = crop_images_batch(images, crop_size, random=random_crop)
    if random_crop:
        images = color_jitter_images_batch(
            images,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
    return images


def camera_dropout_prob(
    step: int,
    total_steps: int,
    *,
    schedule_steps: int | None = None,
    early_frac: float = 0.30,
    mid_frac: float = 0.40,
    early_prob: float = 0.40,
    mid_prob: float = 0.25,
    late_prob: float = 0.12,
) -> float:
    """三阶段全局相机遮挡概率：前期 / 中期 / 后期。

    进度按 ``horizon = schedule_steps or total_steps``：
    ``progress = min(1, step / horizon)``。超出日程后固定 ``late_prob``，
    续训加大 ``train.steps`` 时不会回落到中期概率。
    """
    horizon = int(schedule_steps) if schedule_steps else int(total_steps)
    if horizon <= 0:
        return float(late_prob)
    progress = min(1.0, float(step) / float(horizon))
    early_end = float(early_frac)
    mid_end = early_end + float(mid_frac)
    if progress < early_end:
        return float(early_prob)
    if progress < mid_end:
        return float(mid_prob)
    return float(late_prob)


def apply_camera_dropout(
    images: torch.Tensor,
    p: float,
    *,
    keep_at_least_one: bool = True,
) -> torch.Tensor:
    """按相机独立 Bernoulli 整路置零。

    ``images``: (B, Cams, T, 3, H, W)。``p<=0`` 时原样返回。
    若 ``keep_at_least_one`` 且某样本所有相机都被抽中，随机放回一路。
    """
    if p <= 0.0:
        return images
    if images.ndim != 6:
        raise ValueError(
            f"apply_camera_dropout expects (B, Cams, T, 3, H, W), got {tuple(images.shape)}"
        )
    b, cams = images.shape[:2]
    if cams == 0:
        return images
    device = images.device
    drop = torch.rand(b, cams, device=device) < float(p)
    if keep_at_least_one and cams > 0:
        all_dropped = drop.all(dim=1)
        if bool(all_dropped.any()):
            # 对全灭样本随机保留一路
            keep_idx = torch.randint(0, cams, (int(all_dropped.sum().item()),), device=device)
            drop[all_dropped, :] = True
            drop[all_dropped, keep_idx] = False
    # (B, Cams, 1, 1, 1, 1) 广播到整路时空
    mask = (~drop).to(dtype=images.dtype).view(b, cams, 1, 1, 1, 1)
    return images * mask


class EpisodeDataset(Dataset):
    """帧级索引的数据集，支持多相机与 action chunking。

    索引方式：内部维护 (ep_idx, t) 列表，__getitem__(idx) 取第 idx 个 (ep, t)。

    默认 ``drop_n_last_frames = 0``：不丢末尾帧；未来动作不够 ``horizon`` 时 0-pad，
    并用 ``action_mask`` 标出有效步。
    """

    def __init__(
        self,
        run_dir: Path,
        n_obs_steps: int = 2,
        horizon: int = 16,
        n_action_steps: int | None = None,
        drop_n_last_frames: int | None = None,
        stats: dict[str, np.ndarray] | None = None,
        normalize: bool = True,
        norm_mode: str = "gaussian",
        resize_size: int | None = None,
        crop_size: int | None = 84,
        random_crop: bool = True,
        color_jitter_brightness: float = 0.0,
        color_jitter_contrast: float = 0.0,
        color_jitter_saturation: float = 0.0,
        color_jitter_hue: float = 0.0,
        defer_augment: bool = False,
        uint8_cache: bool = False,
        uint8_cache_dir: str | Path | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.meta = load_meta(self.run_dir)
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.stats = stats
        self.normalize = normalize
        self.norm_mode = validate_norm_mode(norm_mode)
        if self.normalize and self.stats is not None and is_limits_mode(self.norm_mode):
            missing = [
                k
                for k in ("state_min", "state_max", "action_min", "action_max")
                if k not in self.stats
            ]
            if missing:
                raise ValueError(
                    f"norm_mode={self.norm_mode!r} requires {missing} in stats; "
                    "delete stats.json and recompute, or call ensure_stats(..., "
                    f"{self.norm_mode!r})"
                )
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.color_jitter_brightness = color_jitter_brightness
        self.color_jitter_contrast = color_jitter_contrast
        self.color_jitter_saturation = color_jitter_saturation
        self.color_jitter_hue = color_jitter_hue
        self.defer_augment = bool(defer_augment)
        if uint8_cache:
            raise ValueError("uint8_cache is only supported for LeRobot image-sequence datasets")
        _ = uint8_cache_dir

        if drop_n_last_frames is None:
            drop_n_last_frames = 0
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")
        if n_action_steps is not None and n_action_steps > horizon:
            raise ValueError(f"n_action_steps ({n_action_steps}) must be <= horizon ({horizon})")
        self.n_action_steps = n_action_steps
        self.drop_n_last_frames = drop_n_last_frames

        self.episode_files = sorted((self.run_dir / "episodes").glob("ep_*.npz"))
        if not self.episode_files:
            raise FileNotFoundError(f"No episodes in {self.run_dir / 'episodes'}")

        # self.index: 扁平化后的「所有 episode × 保留帧」
        # 每个元素是 (第几个 episode, 该 episode 内的时间步 t)
        # 默认不丢末尾帧；未来不够 horizon 时在 __getitem__ 里 0-pad
        self.index: list[tuple[int, int]] = []
        self._episode_lengths: list[int] = []
        for ep_idx, ep_file in enumerate(self.episode_files):
            payload = load_episode(ep_file)
            length = payload["arrays"]["state"].shape[0]
            self._episode_lengths.append(length)
            usable = length - self.drop_n_last_frames
            if usable <= 0:
                continue
            for t in range(usable):
                self.index.append((ep_idx, t))
        if not self.index:
            raise ValueError(
                f"No usable frames after drop_n_last_frames={self.drop_n_last_frames}; "
                "check episode lengths / n_action_steps."
            )

    def __len__(self) -> int:
        return len(self.index)

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        if not self.normalize or self.stats is None:
            return state
        return normalize(state, self.stats, prefix="state", mode=self.norm_mode)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        if not self.normalize or self.stats is None:
            return action
        return normalize(action, self.stats, prefix="action", mode=self.norm_mode)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """返回一个训练样本。

        返回张量:
            obs_images:      (Cams, T_obs, 3, H', W')  float32 [0,1]（可选裁剪）
            obs_state:       (T_obs, state_dim)
            action:          (horizon, action_dim)  已归一化；末尾不足则 0-pad
            action_mask:     (horizon, 1)  有效步为 1，padding 为 0（loss 只算有效步）
        """
        # ---- 1) 定位到某个 episode 的某一帧 t ----
        ep_idx, t = self.index[idx]
        payload = load_episode(self.episode_files[ep_idx])
        arrays = payload["arrays"]
        length = arrays["state"].shape[0]

        # ---- 2) 构造观测历史：过去 n_obs_steps 帧（含当前 t）----
        # 例 n_obs_steps=2, t=10 → obs_indices=[9, 10]
        # 例 n_obs_steps=2, t=0  → 历史不够，用第 0 帧重复填充 → [0, 0]
        obs_start = max(0, t - self.n_obs_steps + 1)
        obs_indices = list(range(obs_start, t + 1))
        while len(obs_indices) < self.n_obs_steps:
            obs_indices.insert(0, obs_indices[0])  # 开头重复第一帧

        # ---- 3) 读多相机图像，归一化到 [0,1]，再做空间裁剪 ----
        images = []
        for cam in self.meta.camera_names:
            cam_frames = arrays[image_key(cam)][obs_indices]
            # HWC → CHW，并 /255
            cam_frames = np.transpose(cam_frames, (0, 3, 1, 2)).astype(np.float32) / 255.0
            images.append(torch.from_numpy(cam_frames))

        obs_images = torch.stack(images, dim=0)  # (Cams, T_obs, 3, H, W)
        # 可选：先放大再裁（SlowFast 常用 resize→crop）
        obs_images = resize_images(obs_images, self.resize_size)
        if not self.defer_augment:
            # 训练时 random_crop=True：随机裁到 crop_size×crop_size
            obs_images = crop_images(obs_images, self.crop_size, random=self.random_crop)
            if self.random_crop:
                obs_images = color_jitter_images(
                    obs_images,
                    brightness=self.color_jitter_brightness,
                    contrast=self.color_jitter_contrast,
                    saturation=self.color_jitter_saturation,
                    hue=self.color_jitter_hue,
                )

        # ---- 4) 状态取 obs_indices，并用 stats 归一化 ----
        state = self._normalize_state(arrays["state"][obs_indices].astype(np.float32))

        # ---- 5) 动作标签：从当前 t 起往后取 horizon 步 ----
        # 末尾不够则 0-pad，action_mask 标出有效步（不丢帧）。
        #
        # 例 length=100, horizon=8, t=95:
        #   action[95:100] → valid_len=5，后 3 步 0-pad，mask=[1,1,1,1,1,0,0,0]
        action_end = min(t + self.horizon, length)
        valid_len = action_end - t  # 本样本里「真实动作」有几步
        actions = arrays["action"][t:action_end].astype(np.float32)

        # mask: 前 valid_len 步=1（参与 loss），后面 pad 步=0（不参与 loss）
        mask = np.zeros((self.horizon, 1), dtype=np.float32)
        mask[:valid_len] = 1.0
        if valid_len < self.horizon:
            pad = np.zeros((self.horizon - valid_len, self.meta.action_dim), dtype=np.float32)
            actions = np.concatenate([actions, pad], axis=0)
        actions = self._normalize_action(actions)

        return {
            "obs_images": obs_images,
            "obs_state": torch.from_numpy(state),
            "action": torch.from_numpy(actions),
            "action_mask": torch.from_numpy(mask),
        }
