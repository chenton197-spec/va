"""SlowFast 视频视觉编码器（与 encoders.py 的 ResNet 路径隔离）。

接口约定（便于日后换 VideoMAE 等）:
  SlowFastVideoEncoder.forward(obs) : (B, T, 3, H, W) -> (B, out_dim)
  MultiCameraSlowFastEncoder 与 MultiCameraEncoder 同形:
    forward(obs_images, obs_state) -> cond (B, cond_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from robotfm.policies.encoders import StateEncoder

# SlowFast R50 融合后通道（slow 2048 + fast 256）
_SLOWFAST_FEAT_DIM = 2304
# Kinetics SlowFast-R50 8x8 预训练默认 clip 长度
_SLOWFAST_NUM_FRAMES = 32


def _pack_pathway(frames: torch.Tensor, alpha: int = 4) -> list[torch.Tensor]:
    """将 (B, C, T, H, W) 拆成 SlowFast 的 [slow, fast] 两路。"""
    fast = frames
    t = frames.shape[2]
    n_slow = max(t // alpha, 1)
    idx = torch.linspace(0, t - 1, steps=n_slow, device=frames.device).long()
    slow = torch.index_select(frames, 2, idx)
    return [slow, fast]


def _resample_temporal(frames: torch.Tensor, target_t: int) -> torch.Tensor:
    """将 (B, C, T, H, W) 时间维重采样到 ``target_t``（适配预训练 clip 长度）。"""
    if frames.shape[2] == target_t:
        return frames
    return F.interpolate(
        frames,
        size=(target_t, frames.shape[3], frames.shape[4]),
        mode="trilinear",
        align_corners=False,
    )


def _load_slowfast_r50(pretrained: bool) -> nn.Module:
    """Torch Hub 加载 Kinetics 预训练 SlowFast-R50，去掉分类头。"""
    model = torch.hub.load(
        "facebookresearch/pytorchvideo",
        "slowfast_r50",
        pretrained=pretrained,
        trust_repo=True,
    )
    # blocks[-1] 是 ResNetBasicHead: pool → dropout → proj(→num_classes)
    head = model.blocks[-1]
    head.proj = nn.Identity()
    head.activation = None
    return model


class SlowFastVideoEncoder(nn.Module):
    """Kinetics 预训练 SlowFast-R50 视频编码器。

    输入: obs (B, T, 3, H, W) in [0, 1]
    输出: feat (B, out_dim)

    内部将时间维重采样到 32 帧以匹配 SlowFast 8x8 预训练池化核。
    """

    def __init__(
        self,
        out_dim: int = 128,
        pretrained: bool = True,
        alpha: int = 4,
        num_frames: int = _SLOWFAST_NUM_FRAMES,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.num_frames = num_frames
        self.backbone = _load_slowfast_r50(pretrained=pretrained)
        self.proj = nn.Sequential(
            nn.Linear(_SLOWFAST_FEAT_DIM, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        # Kinetics / pytorchvideo SlowFast 预处理
        self.register_buffer(
            "img_mean",
            torch.tensor([0.45, 0.45, 0.45], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "img_std",
            torch.tensor([0.225, 0.225, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, T, 3, H, W)
        returns: (B, out_dim)
        """
        # (B, T, 3, H, W) → (B, 3, T, H, W)
        x = (obs - self.img_mean) / self.img_std
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = _resample_temporal(x, self.num_frames)
        pathways = _pack_pathway(x, alpha=self.alpha)
        feat = self.backbone(pathways)
        if feat.ndim > 2:
            feat = feat.flatten(1)
        return self.proj(feat)


class MultiCameraSlowFastEncoder(nn.Module):
    """多相机 SlowFast + 多步状态 → global cond。

    与 ``MultiCameraEncoder`` 输出约定一致，供 FlowMatchingPolicy 直接替换。
    """

    def __init__(
        self,
        num_cameras: int,
        state_dim: int,
        n_obs_steps: int,
        image_out_dim: int = 128,
        state_out_dim: int = 128,
        cond_dim: int = 256,
        pretrained_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.n_obs_steps = n_obs_steps

        self.image_encoder = SlowFastVideoEncoder(
            out_dim=image_out_dim,
            pretrained=pretrained_encoder,
        )
        self.state_encoder = StateEncoder(state_dim=state_dim, out_dim=state_out_dim)

        fused_dim = num_cameras * image_out_dim + state_out_dim * n_obs_steps
        self.proj = nn.Sequential(
            nn.Linear(fused_dim, cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs_images: torch.Tensor, obs_state: torch.Tensor) -> torch.Tensor:
        """
        obs_images: (B, Cams, T, 3, H, W)
        obs_state: (B, T, state_dim)
        returns: cond (B, cond_dim)

        Train: per-camera serial; eval: batch cameras (B*Cams) for one backbone pass.
        """
        b, cams, t, _, _, _ = obs_images.shape

        if self.training:
            cam_feats = [self.image_encoder(obs_images[:, c]) for c in range(cams)]
            img_feat = torch.cat(cam_feats, dim=-1)
        else:
            flat = obs_images.reshape(b * cams, t, *obs_images.shape[3:])
            cam_feats = self.image_encoder(flat)
            img_feat = cam_feats.reshape(b, cams, -1).flatten(1)

        state = obs_state.reshape(b * t, -1)
        state_feat = self.state_encoder(state).reshape(b, -1)

        fused = torch.cat([img_feat, state_feat], dim=-1)
        return self.proj(fused)
