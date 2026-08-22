"""PyTorch Dataset：按帧索引采样，构造 action chunk 训练 batch。

每个样本对应 (episode_id, 时间步 t)：
- 输入：过去 n_obs_steps 帧的多相机图像 + 状态
- 标签：从 t 开始的 horizon 步动作序列

未来动作不够 ``horizon`` 时，按原版 A2A / Diffusion Policy sampler
**重复最后一帧动作**补齐（不是 0-pad）。``action_mask`` 仍标出真实步 vs
补齐步，供 ACT 等策略可选屏蔽；A2A 重建损失按原版对整段 chunk 计算。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from robotfm.data.schema import image_key, load_episode, load_meta
from robotfm.data.action_delta import (
    flow_history_from_phys,
    joint_mask_from_names,
    overlay_joint_delta_action_stats,
    subtract_joint_pose,
)
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


def resize_images(images: torch.Tensor, size: int | None, mode: str = "bilinear") -> torch.Tensor:
    """缩放到 ``size×size``。

    支持形状:
      - (T, C, H, W)
      - (Cams, T, C, H, W)
    """
    if size is None:
        return images
    *lead, c, h, w = images.shape
    if h == size and w == size:
        return images
    flat = images.reshape(-1, c, h, w)
    if mode == "nearest":
        flat = torch.nn.functional.interpolate(flat, size=(size, size), mode="nearest")
    else:
        flat = torch.nn.functional.interpolate(
            flat, size=(size, size), mode="bilinear", align_corners=False
        )
    return flat.reshape(*lead, c, size, size)


def spatial_preprocess_images(
    images: torch.Tensor,
    *,
    pre_crop_size: int | None = None,
    resize_size: int | None = None,
    crop_size: int | None = None,
    random_crop: bool = False,
    resize_mode: str = "bilinear",
) -> torch.Tensor:
    """统一空间预处理：中心 pre_crop → resize → 可选 crop。

    ``pre_crop_size`` 始终中心裁（保比例方裁，如 1280×720 → 720×720）。
    ``crop_size`` 在 resize 之后；``random_crop=True`` 时随机裁，否则中心裁。
    """
    if pre_crop_size is not None:
        images = crop_images(images, pre_crop_size, random=False)
    images = resize_images(images, resize_size, mode=resize_mode)
    if crop_size is not None:
        images = crop_images(images, crop_size, random=random_crop)
    return images


def crop_hw_box(h: int, w: int, crop_size: int, random: bool) -> tuple[int, int]:
    if h < crop_size or w < crop_size:
        raise ValueError(f"Cannot crop {h}x{w} to {crop_size}")
    if random:
        top = int(torch.randint(0, h - crop_size + 1, (1,)).item())
        left = int(torch.randint(0, w - crop_size + 1, (1,)).item())
    else:
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
    return top, left


def apply_hw_crop(images: torch.Tensor, top: int, left: int, crop_size: int) -> torch.Tensor:
    return images[..., top : top + crop_size, left : left + crop_size]


def crop_images(images: torch.Tensor, crop_size: int | None, random: bool) -> torch.Tensor:
    """对 CHW 图像张量做空间裁剪。

    支持形状:
      - (T, C, H, W)
      - (Cams, T, C, H, W)

    ``crop_size is None`` 或等于 H/W 时原样返回。
    """
    if crop_size is None:
        return images
    *lead, c, h, w = images.shape
    if h == crop_size and w == crop_size:
        return images
    top, left = crop_hw_box(h, w, crop_size, random)
    return apply_hw_crop(images, top, left, crop_size)


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


def crop_offsets_batch(
    batch_size: int,
    h: int,
    w: int,
    crop_size: int,
    random: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if h < crop_size or w < crop_size:
        raise ValueError(f"Cannot crop {h}x{w} to {crop_size}")
    if random:
        tops = torch.randint(0, h - crop_size + 1, (batch_size,), device=device)
        lefts = torch.randint(0, w - crop_size + 1, (batch_size,), device=device)
    else:
        tops = torch.full((batch_size,), (h - crop_size) // 2, device=device, dtype=torch.long)
        lefts = torch.full((batch_size,), (w - crop_size) // 2, device=device, dtype=torch.long)
    return tops, lefts


def apply_crop_offsets_batch(
    images: torch.Tensor, tops: torch.Tensor, lefts: torch.Tensor, crop_size: int
) -> torch.Tensor:
    b = images.shape[0]
    out = images.new_empty(*images.shape[:-2], crop_size, crop_size)
    for i in range(b):
        top = int(tops[i])
        left = int(lefts[i])
        out[i] = images[i, ..., top : top + crop_size, left : left + crop_size]
    return out


def crop_images_batch(
    images: torch.Tensor, crop_size: int | None, random: bool
) -> torch.Tensor:
    """Batched spatial crop for ``(B, Cams, T, C, H, W)``.

    Each batch item gets its own crop window; cams/T within an item share it.
    """
    if crop_size is None:
        return images
    if images.ndim != 6:
        raise ValueError(f"Expected (B,Cams,T,C,H,W), got shape {tuple(images.shape)}")
    b, _cams, _t, _c, h, w = images.shape
    if h == crop_size and w == crop_size:
        return images
    tops, lefts = crop_offsets_batch(b, h, w, crop_size, random, images.device)
    return apply_crop_offsets_batch(images, tops, lefts, crop_size)


_LUMA_WEIGHTS = (0.2989, 0.5870, 0.1140)


def _jitter_factor(images: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Per-sample factor shaped ``(B,1,1,1,1,1)``; no CPU sync."""
    b = images.shape[0]
    return torch.empty(
        b, 1, 1, 1, 1, 1, device=images.device, dtype=images.dtype
    ).uniform_(low, high)


def _blend_broadcast(img: torch.Tensor, other: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
    """``ratio * img + (1 - ratio) * other``, matching torchvision ``_blend``."""
    return (ratio * img + (1.0 - ratio) * other).clamp(0.0, 1.0)


def _luma(images: torch.Tensor) -> torch.Tensor:
    """Rec. 601 luma; ``images`` is ``(..., 3, H, W)`` → ``(..., H, W)``."""
    r, g, b = images.unbind(dim=-3)
    return _LUMA_WEIGHTS[0] * r + _LUMA_WEIGHTS[1] * g + _LUMA_WEIGHTS[2] * b


def _rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    """Vectorized RGB→HSV; channel dim is -3 (torchvision-compatible)."""
    r, g, b = img.unbind(dim=-3)
    maxc = torch.max(img, dim=-3).values
    minc = torch.min(img, dim=-3).values
    eqc = maxc == minc
    cr = maxc - minc
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    cr_safe = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_safe
    gc = (maxc - g) / cr_safe
    bc = (maxc - b) / cr_safe
    hr = (bc - gc).div(6).add(1).remainder(1)
    hg = rc.sub(bc).div(6).add(2.0 / 6)
    hb = gc.sub(rc).div(6).add(4.0 / 6)
    h = torch.where((maxc == r) & ~eqc, hr, ones)
    h = torch.where((maxc == g) & ~eqc, hg, h)
    h = torch.where((maxc == b) & ~eqc, hb, h)
    return torch.stack((h, s, maxc), dim=-3)


def _hsv_to_rgb(img: torch.Tensor) -> torch.Tensor:
    """Vectorized HSV→RGB without the 6-sector stack (keeps 8GB GPUs alive at 512)."""
    h, s, v = img.unbind(dim=-3)
    h6 = h * 6.0
    i = torch.remainder(torch.floor(h6), 6.0).to(dtype=torch.int64)
    f = h6 - torch.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r, g, b = v, t, p  # i == 0
    m = i == 1
    r, g, b = torch.where(m, q, r), torch.where(m, v, g), torch.where(m, p, b)
    m = i == 2
    r, g, b = torch.where(m, p, r), torch.where(m, v, g), torch.where(m, t, b)
    m = i == 3
    r, g, b = torch.where(m, p, r), torch.where(m, q, g), torch.where(m, v, b)
    m = i == 4
    r, g, b = torch.where(m, t, r), torch.where(m, p, g), torch.where(m, v, b)
    m = i == 5
    r, g, b = torch.where(m, v, r), torch.where(m, p, g), torch.where(m, q, b)
    return torch.stack((r, g, b), dim=-3)


def color_jitter_images_batch(
    images: torch.Tensor,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
) -> torch.Tensor:
    """Batched photometric jitter for ``(B, Cams, T, 3, H, W)`` on any device.

    Each batch item samples one factor set shared across its cams/T.
    Factors stay on-device (no ``float(gpu_tensor)`` sync / Python loop).
    """
    if brightness <= 0 and contrast <= 0 and saturation <= 0 and hue <= 0:
        return images
    if images.ndim != 6:
        raise ValueError(f"Expected (B,Cams,T,3,H,W), got shape {tuple(images.shape)}")
    if images.shape[-3] != 3:
        raise ValueError(f"Expected 3-channel RGB, got C={images.shape[-3]}")

    out = images
    if brightness > 0:
        factor = _jitter_factor(out, max(0.0, 1.0 - brightness), 1.0 + brightness)
        out = (out * factor).clamp(0.0, 1.0)
    if contrast > 0:
        mean = _luma(out).mean(dim=(-2, -1), keepdim=True).unsqueeze(-3)
        factor = _jitter_factor(out, max(0.0, 1.0 - contrast), 1.0 + contrast)
        out = _blend_broadcast(out, mean, factor)
    if saturation > 0:
        gray = _luma(out).unsqueeze(-3).expand_as(out)
        factor = _jitter_factor(out, max(0.0, 1.0 - saturation), 1.0 + saturation)
        out = _blend_broadcast(out, gray, factor)
    if hue > 0:
        hue_delta = _jitter_factor(out, -hue, hue).clamp(-0.5, 0.5)
        # Per-camera to cap HSV intermediates on 8GB (512×3cams×8T).
        bsz = out.shape[0]
        delta = hue_delta.view(bsz, 1, 1, 1)
        cam_out = []
        for cam_i in range(out.shape[1]):
            hsv = _rgb_to_hsv(out[:, cam_i])
            h_ch, s_ch, v_ch = hsv.unbind(dim=-3)
            h_ch = (h_ch + delta) % 1.0
            cam_out.append(_hsv_to_rgb(torch.stack((h_ch, s_ch, v_ch), dim=-3)))
        out = torch.stack(cam_out, dim=1).clamp(0.0, 1.0)
    return out


def images_to_float01(images: torch.Tensor) -> torch.Tensor:
    """Convert uint8 ``obs_images`` to float32 in ``[0, 1]``.

    Float tensors are returned unchanged. Call after H2D so the DataLoader can
    keep uint8 (4× less host RAM / PCIe) when ``defer_augment`` is set.
    """
    if images.dtype == torch.uint8:
        return images.float().div_(255.0)
    return images


def apply_image_augments_batch(
    images: torch.Tensor,
    *,
    crop_size: int | None,
    random_crop: bool,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
    depths: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """GPU/CPU batch crop (+ optional color jitter when ``random_crop``).

    If ``depths`` is given, it is cropped with the same per-sample window and
    returned as the second value (no color jitter).
    """
    if crop_size is not None and images.ndim == 6:
        h, w = images.shape[-2], images.shape[-1]
        if h != crop_size or w != crop_size:
            tops, lefts = crop_offsets_batch(
                images.shape[0], h, w, crop_size, random_crop, images.device
            )
            images = apply_crop_offsets_batch(images, tops, lefts, crop_size)
            if depths is not None:
                depths = apply_crop_offsets_batch(depths, tops, lefts, crop_size)
        elif depths is not None and (
            depths.shape[-2] != crop_size or depths.shape[-1] != crop_size
        ):
            depths = crop_images_batch(depths, crop_size, random=False)
    else:
        images = crop_images_batch(images, crop_size, random=random_crop)
        if depths is not None:
            depths = crop_images_batch(depths, crop_size, random=False)
    if random_crop:
        images = color_jitter_images_batch(
            images,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
    if depths is None:
        return images
    return images, depths


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


def state_group_masks(
    state_names: list[str] | None,
    state_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按名字拆关节 / 夹爪掩码，形状均为 ``(D,)`` bool。

    夹爪维：名字（忽略大小写）含 ``gripper``。
    无夹爪名且 ``state_dim >= 7`` 时，沿用单臂惯例把最后一维当夹爪。
    """
    dim = int(state_dim)
    names = list(state_names or [])
    if len(names) != dim:
        names = [f"s{i}" for i in range(dim)]
    gripper = torch.tensor(
        ["gripper" in name.lower() for name in names],
        dtype=torch.bool,
    )
    if dim > 0 and not bool(gripper.any()) and dim >= 7:
        gripper[-1] = True
    return ~gripper, gripper


def apply_state_dropout(
    state: torch.Tensor,
    *,
    whole_p: float,
    joint_p: float,
    gripper_p: float,
    joint_mask: torch.Tensor,
    gripper_mask: torch.Tensor,
    keep_at_least_one: bool = True,
) -> torch.Tensor:
    """全状态 / 关节组 / 夹爪组按样本 Bernoulli 置零。

    ``state``: (B, T, D)。同一组对所有历史帧一起遮挡。所有概率都 ``<=0`` 时原样返回。
    若 ``keep_at_least_one`` 且某样本两组都被抽中，随机放回一组。
    ``whole_p`` 抽中的样本始终全部置零，不受 ``keep_at_least_one`` 影响。
    """
    if whole_p <= 0.0 and joint_p <= 0.0 and gripper_p <= 0.0:
        return state
    if state.ndim != 3:
        raise ValueError(
            f"apply_state_dropout expects (B, T, D), got {tuple(state.shape)}"
        )
    batch, _time, dim = state.shape
    device = state.device
    joint_mask = joint_mask.to(device=device, dtype=torch.bool).reshape(-1)
    gripper_mask = gripper_mask.to(device=device, dtype=torch.bool).reshape(-1)
    if joint_mask.numel() != dim or gripper_mask.numel() != dim:
        raise ValueError(
            f"state_dropout masks must have length {dim}, got "
            f"joint={tuple(joint_mask.shape)} gripper={tuple(gripper_mask.shape)}"
        )

    groups: list[tuple[torch.Tensor, torch.Tensor]] = []
    if joint_p > 0.0 and bool(joint_mask.any()):
        groups.append((torch.rand(batch, device=device) < float(joint_p), joint_mask))
    if gripper_p > 0.0 and bool(gripper_mask.any()):
        groups.append((torch.rand(batch, device=device) < float(gripper_p), gripper_mask))
    group_drop_mask = torch.zeros(batch, dim, device=device, dtype=torch.bool)
    for group_drop, group_mask in groups:
        group_drop_mask = group_drop_mask | (
            group_drop.unsqueeze(1) & group_mask.unsqueeze(0)
        )

    if keep_at_least_one and groups:
        stacked = torch.stack([group_drop for group_drop, _ in groups], dim=1)
        all_dropped = stacked.all(dim=1)
        if bool(all_dropped.any()):
            n_restore = int(all_dropped.sum().item())
            keep_group = torch.randint(0, len(groups), (n_restore,), device=device)
            for group_idx, (_group_drop, group_mask) in enumerate(groups):
                restore = all_dropped.clone()
                restore[all_dropped] = keep_group == group_idx
                if bool(restore.any()):
                    group_drop_mask[restore] = group_drop_mask[restore] & ~group_mask

    whole_drop = torch.rand(batch, device=device) < float(whole_p)
    drop = group_drop_mask | whole_drop.unsqueeze(1)

    mask = (~drop).to(dtype=state.dtype).view(batch, 1, dim)
    return state * mask


class EpisodeDataset(Dataset):
    """帧级索引的数据集，支持多相机与 action chunking。

    索引方式：内部维护 (ep_idx, t) 列表，__getitem__(idx) 取第 idx 个 (ep, t)。

    默认 ``drop_n_last_frames = 0``：不丢末尾帧；未来动作不够 ``horizon`` 时
    重复最后一帧动作补齐，并用 ``action_mask`` 标出真实步 vs 补齐步。
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
        pre_crop_size: int | None = None,
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
        predict_joint_delta: bool = False,
        depth_cameras: tuple[str, ...] | list[str] = (),
        depth_min_mm: float = 50.0,
        depth_max_mm: float = 500.0,
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
        self.pre_crop_size = pre_crop_size
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.color_jitter_brightness = color_jitter_brightness
        self.color_jitter_contrast = color_jitter_contrast
        self.color_jitter_saturation = color_jitter_saturation
        self.color_jitter_hue = color_jitter_hue
        self.defer_augment = bool(defer_augment)
        self.predict_joint_delta = bool(predict_joint_delta)
        _ = depth_cameras
        _ = depth_min_mm
        _ = depth_max_mm
        self._joint_mask = joint_mask_from_names(self.meta.action_names, self.meta.action_dim)
        if self.predict_joint_delta and self.meta.state_dim != self.meta.action_dim:
            raise ValueError(
                "predict_joint_delta requires state_dim == action_dim "
                f"(got {self.meta.state_dim} vs {self.meta.action_dim})"
            )
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
        # 默认不丢末尾帧；未来不够 horizon 时在 __getitem__ 里重复最后动作
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
        if self.predict_joint_delta:
            if self.stats is None:
                raise ValueError("predict_joint_delta requires stats")
            ep_states = []
            ep_actions = []
            for ep_file in self.episode_files:
                arrays = load_episode(ep_file)["arrays"]
                ep_states.append(arrays["state"])
                ep_actions.append(arrays["action"])
            overlay_joint_delta_action_stats(
                self.stats,
                ep_states,
                ep_actions,
                horizon=self.horizon,
                joint_mask=self._joint_mask,
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
            action:          (horizon, action_dim)  已归一化；末尾不足则重复最后一帧
            action_mask:     (horizon, 1)  真实步为 1，末尾重复补齐为 0
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
        # pre_crop（中心）→ resize；post-crop / color_jitter 可 defer 到 GPU
        if self.defer_augment:
            obs_images = spatial_preprocess_images(
                obs_images,
                pre_crop_size=self.pre_crop_size,
                resize_size=self.resize_size,
                crop_size=None,
                random_crop=False,
            )
        else:
            obs_images = spatial_preprocess_images(
                obs_images,
                pre_crop_size=self.pre_crop_size,
                resize_size=self.resize_size,
                crop_size=self.crop_size,
                random_crop=self.random_crop,
            )
            if self.random_crop:
                obs_images = color_jitter_images(
                    obs_images,
                    brightness=self.color_jitter_brightness,
                    contrast=self.color_jitter_contrast,
                    saturation=self.color_jitter_saturation,
                    hue=self.color_jitter_hue,
                )

        # ---- 4) 状态取 obs_indices，并用 stats 归一化 ----
        state_phys = arrays["state"][obs_indices].astype(np.float32)
        state = self._normalize_state(state_phys)
        if self.predict_joint_delta:
            if self.stats is None:
                raise ValueError("predict_joint_delta requires stats")
            flow_hist = flow_history_from_phys(
                state_phys,
                self.stats,
                self.norm_mode,
                predict_joint_delta=True,
                joint_mask=self._joint_mask,
            )
        else:
            flow_hist = state

        # ---- 5) 动作标签：从当前 t 起往后取 horizon 步 ----
        # 末尾不够则重复最后一帧（原版 A2A sampler），action_mask 标出真实步。
        #
        # 例 length=100, horizon=8, t=95:
        #   action[95:100] → valid_len=5，后 3 步 = a99，mask=[1,1,1,1,1,0,0,0]
        action_end = min(t + self.horizon, length)
        valid_len = action_end - t  # 本样本里「真实动作」有几步
        actions = arrays["action"][t:action_end].astype(np.float32)
        if self.predict_joint_delta:
            q_now = arrays["state"][t].astype(np.float32)
            actions = subtract_joint_pose(actions, q_now, self._joint_mask)

        # mask: 前 valid_len 步=1（真实），后面 pad 步=0（重复最后动作补齐）
        mask = np.zeros((self.horizon, 1), dtype=np.float32)
        mask[:valid_len] = 1.0
        if valid_len < self.horizon:
            # 原版 sampler：末尾用最后一帧重复填充，而非 0
            last = actions[-1:]
            pad = np.repeat(last, self.horizon - valid_len, axis=0)
            actions = np.concatenate([actions, pad], axis=0)
        actions = self._normalize_action(actions)

        return {
            "obs_images": obs_images,
            "obs_state": torch.from_numpy(state),
            "obs_history": torch.from_numpy(np.asarray(flow_hist, dtype=np.float32)),
            "action": torch.from_numpy(actions),
            "action_mask": torch.from_numpy(mask),
        }
