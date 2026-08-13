"""ViT-B/16 视觉编码器（与 encoders.py / video_encoders.py 隔离）。

接口约定:
  ViTEncoder.forward(obs) : (B, T, 3, H, W) -> (B, out_dim)
  MultiCameraViTEncoder 与 MultiCameraEncoder 同形:
    forward(obs_images, obs_state) -> cond (B, cond_dim)

预训练：torchvision ImageNet-1k ViT-B/16（非 Google JAX .npz）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16

from robotfm.policies.encoders import StateEncoder

_VIT_FEAT_DIM = 768
_VIT_IMAGE_SIZE = 224


def _adapt_vit_patch_conv(backbone: nn.Module, in_channels: int) -> nn.Module:
    """扩展 ViT patch embed（conv_proj），支持多帧 / 帧差通道堆叠。"""
    old = backbone.conv_proj
    if not isinstance(old, nn.Conv2d):
        raise TypeError(f"Expected Conv2d conv_proj, got {type(old)}")
    if old.in_channels == in_channels:
        return backbone

    new = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        new.weight.zero_()
        for c in range(in_channels):
            new.weight[:, c] = old.weight[:, c % old.in_channels]
        new.weight *= old.in_channels / float(in_channels)
        if old.bias is not None and new.bias is not None:
            new.bias.copy_(old.bias)
    backbone.conv_proj = new
    return backbone


class ViTEncoder(nn.Module):
    """torchvision ViT-B/16 时序视觉编码器。

    - ImageNet-1k 预训练（可关）；去掉分类头，取 CLS token
    - ImageNet mean/std 归一化
    - 可选 frame diff：``[I0, I1-I0, ...]`` 通道拼接后一次前向
    - 输入空间尺寸双线性缩放到 224（匹配预训练 pos embed）

    输入: obs (B, T, 3, H, W)
    输出: feat (B, out_dim)
    """

    def __init__(
        self,
        n_obs_steps: int = 2,
        out_dim: int = 128,
        pretrained: bool = True,
        use_frame_diff: bool = True,
        image_size: int = _VIT_IMAGE_SIZE,
    ) -> None:
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.use_frame_diff = use_frame_diff and n_obs_steps >= 2
        self.image_size = image_size
        self.pretrained = pretrained

        in_channels = 3 * n_obs_steps
        if pretrained:
            backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            backbone = vit_b_16(weights=None)

        backbone = _adapt_vit_patch_conv(backbone, in_channels)
        backbone.heads = nn.Identity()
        self.backbone = backbone

        self.proj = nn.Sequential(
            nn.Linear(_VIT_FEAT_DIM, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

        self.register_buffer(
            "img_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "img_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def _prepare_images(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, T, 3, H, W) in [0,1] -> (B, T*3, H, W)."""
        obs = (obs - self.img_mean) / self.img_std
        if self.use_frame_diff:
            frames = [obs[:, 0]]
            for t in range(1, obs.shape[1]):
                frames.append(obs[:, t] - obs[:, t - 1])
            x = torch.cat(frames, dim=1)
        else:
            x = obs.flatten(1, 2)
        if x.shape[-2] != self.image_size or x.shape[-1] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, T, 3, H, W)
        returns: (B, out_dim)
        """
        x = self._prepare_images(obs)
        cls = self.backbone(x)
        if cls.ndim > 2:
            cls = cls.flatten(1)
        return self.proj(cls)


class MultiCameraViTEncoder(nn.Module):
    """多相机 ViT-B/16 + 多步状态 → global cond。

    与 ``MultiCameraEncoder`` 输出约定一致，供 FlowMatching / A2A / VITA 替换。

    ``share_image_encoder``: True 共用一份 ViT；False 每相机独立权重。
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
        use_frame_diff: bool = True,
        share_image_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.n_obs_steps = n_obs_steps
        self.share_image_encoder = share_image_encoder

        def _make_image_encoder() -> ViTEncoder:
            return ViTEncoder(
                n_obs_steps=n_obs_steps,
                out_dim=image_out_dim,
                pretrained=pretrained_encoder,
                use_frame_diff=use_frame_diff,
            )

        if share_image_encoder:
            self.image_encoder = _make_image_encoder()
        else:
            self.image_encoders = nn.ModuleList(
                [_make_image_encoder() for _ in range(num_cameras)]
            )
        self.state_encoder = StateEncoder(state_dim=state_dim, out_dim=state_out_dim)

        fused_dim = num_cameras * image_out_dim + state_out_dim * n_obs_steps
        self.proj = nn.Sequential(
            nn.Linear(fused_dim, cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(inplace=True),
        )

    def vision_parameters(self):
        """Yield visual backbone parameters（供 optimizer 分组）。"""
        if self.share_image_encoder:
            yield from self.image_encoder.parameters()
        else:
            yield from self.image_encoders.parameters()

    def _encode_images(self, obs_images: torch.Tensor) -> torch.Tensor:
        """obs_images (B, Cams, T, 3, H, W) -> (B, Cams * image_out_dim)."""
        b, cams, t, _, _, _ = obs_images.shape
        if cams != self.num_cameras:
            raise ValueError(
                f"Expected {self.num_cameras} cameras, got {cams}"
            )

        if not self.share_image_encoder:
            cam_feats = [
                self.image_encoders[c](obs_images[:, c]) for c in range(cams)
            ]
            return torch.cat(cam_feats, dim=-1)

        if self.training:
            cam_feats = [self.image_encoder(obs_images[:, c]) for c in range(cams)]
            return torch.cat(cam_feats, dim=-1)

        flat = obs_images.reshape(b * cams, t, *obs_images.shape[3:])
        cam_feats = self.image_encoder(flat)
        return cam_feats.reshape(b, cams, -1).flatten(1)

    def forward(self, obs_images: torch.Tensor, obs_state: torch.Tensor) -> torch.Tensor:
        """
        obs_images: (B, Cams, T, 3, H, W)
        obs_state: (B, T, state_dim)
        returns: cond (B, cond_dim)

        Shared: train serial / eval batched; separate encoders always serial.
        """
        b, _, t, _, _, _ = obs_images.shape
        img_feat = self._encode_images(obs_images)

        state = obs_state.reshape(b * t, -1)
        state_feat = self.state_encoder(state).reshape(b, -1)

        fused = torch.cat([img_feat, state_feat], dim=-1)
        return self.proj(fused)
