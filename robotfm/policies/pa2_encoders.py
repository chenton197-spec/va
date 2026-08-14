"""PA2 YOLO 视觉编码器（与 encoders.py / vit_encoders.py 隔离）。

接口约定:
  PA2Encoder.forward(obs) : (B, T, 3, H, W) -> (B, out_dim)
  MultiCameraPA2Encoder 与 MultiCameraEncoder 同形:
    forward(obs_images, obs_state) -> cond (B, cond_dim)

骨干对齐 PA2 ``casbotPA2-backbone.yaml`` 的图像部分（C3k2 / C2PSA / SPPF）。
图像保持 [0, 1]（不做 ImageNet mean/std）；可选 frame diff 通道堆叠。
状态只走外面的 ``StateEncoder``，不进入图像骨干。
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from robotfm.policies.encoders import StateEncoder
from robotfm.policies.pa2_yolo import PA2VisionBackbone


class PA2Encoder(nn.Module):
    """PA2 YOLO 时序视觉编码器（只编码图像）。

    输入: obs (B, T, 3, H, W) in [0, 1]
    输出: feat (B, out_dim)
    """

    def __init__(
        self,
        n_obs_steps: int = 2,
        out_dim: int = 128,
        pretrained: bool = False,
        use_frame_diff: bool = True,
        scale: str = "n",
    ) -> None:
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.use_frame_diff = use_frame_diff and n_obs_steps >= 2
        if pretrained:
            warnings.warn(
                "PA2 YOLO backbone has no ImageNet checkpoint in va; "
                "pretrained_encoder=True is ignored (train from scratch).",
                stacklevel=2,
            )

        in_channels = 3 * n_obs_steps
        self.backbone = PA2VisionBackbone(in_channels=in_channels, scale=scale)
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.out_channels, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _prepare_images(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, T, 3, H, W) in [0,1] -> (B, T*3, H, W)."""
        if self.use_frame_diff:
            frames = [obs[:, 0]]
            for t in range(1, obs.shape[1]):
                frames.append(obs[:, t] - obs[:, t - 1])
            return torch.cat(frames, dim=1)
        return obs.flatten(1, 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, T, 3, H, W)
        returns: (B, out_dim)
        """
        x = self._prepare_images(obs)
        feats = self.backbone(x)
        appearance = feats.mean(dim=(2, 3))
        return self.proj(appearance)


class MultiCameraPA2Encoder(nn.Module):
    """多相机 PA2 YOLO + 多步状态 → global cond。

    与 ``MultiCameraEncoder`` 输出约定一致：图像骨干只看 RGB，状态由
    ``StateEncoder`` 单独编码后再拼接。
    """

    def __init__(
        self,
        num_cameras: int,
        state_dim: int,
        n_obs_steps: int,
        image_out_dim: int = 128,
        state_out_dim: int = 128,
        cond_dim: int = 256,
        pretrained_encoder: bool = False,
        use_frame_diff: bool = True,
        share_image_encoder: bool = True,
        scale: str = "n",
    ) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.n_obs_steps = n_obs_steps
        self.share_image_encoder = share_image_encoder

        def _make_image_encoder() -> PA2Encoder:
            return PA2Encoder(
                n_obs_steps=n_obs_steps,
                out_dim=image_out_dim,
                pretrained=pretrained_encoder,
                use_frame_diff=use_frame_diff,
                scale=scale,
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
            raise ValueError(f"Expected {self.num_cameras} cameras, got {cams}")

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
        """
        b, _, t, _, _, _ = obs_images.shape
        img_feat = self._encode_images(obs_images)

        state = obs_state.reshape(b * t, -1)
        state_feat = self.state_encoder(state).reshape(b, -1)

        fused = torch.cat([img_feat, state_feat], dim=-1)
        return self.proj(fused)
