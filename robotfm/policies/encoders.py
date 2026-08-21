from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class SpatialSoftmax(nn.Module):
    """将特征图压缩为关键点坐标（Diffusion Policy / robomimic 风格）。"""

    def __init__(self, in_channels: int, num_kp: int = 32) -> None:
        super().__init__()
        self.num_kp = num_kp
        self.conv = nn.Conv2d(in_channels, num_kp, kernel_size=1)
        self._cached_hw: tuple[int, int] | None = None
        self.register_buffer("_pos_grid", torch.zeros(2, 1), persistent=False)

    def _make_grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._cached_hw == (h, w) and self._pos_grid.device == device and self._pos_grid.dtype == dtype:
            return self._pos_grid
        ys = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=0)
        self._pos_grid = pos
        self._cached_hw = (h, w)
        return pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        b, k, h, w = feat.shape
        flat = feat.view(b, k, h * w)
        attn = F.softmax(flat, dim=-1)
        pos = self._make_grid(h, w, feat.device, feat.dtype)
        expected = torch.matmul(attn, pos.t())
        return expected.reshape(b, k * 2)


def _adapt_resnet_first_conv(
    backbone: nn.Module,
    in_channels: int,
    *,
    zero_init_last: int = 0,
) -> nn.Module:
    """扩展 ResNet stem，支持多帧 / 帧差通道堆叠（deep-operation 配方）。

    ``zero_init_last``：末尾若干输入通道权重保持 0（用于 CoordConv xy）。
    """
    old = backbone.conv1
    if old.in_channels == in_channels:
        return backbone
    if zero_init_last < 0 or zero_init_last > in_channels:
        raise ValueError(f"zero_init_last={zero_init_last} invalid for in_channels={in_channels}")

    new = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    rgb_channels = in_channels - zero_init_last
    with torch.no_grad():
        new.weight.zero_()
        for c in range(rgb_channels):
            new.weight[:, c] = old.weight[:, c % old.in_channels]
        if rgb_channels > 0:
            new.weight[:, :rgb_channels] *= old.in_channels / float(rgb_channels)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    backbone.conv1 = new
    return backbone


class ResNet18Encoder(nn.Module):
    """ResNet-18 时序视觉编码器（对齐 deep-operation）。

    - ImageNet 预训练（可关），保留 BatchNorm
    - ImageNet mean/std 归一化
    - 可选 frame diff：``[I0, I1-I0, ...]`` 通道拼接后一次前向
    - 可选 CoordConv：在 stem 前拼归一化 xy（``use_coord_conv``）
    - SpatialSoftmax keypoints + GAP 外观特征

    输入: obs (B, T, 3, H, W)
    输出: feat (B, out_dim)
    """

    def __init__(
        self,
        n_obs_steps: int = 2,
        out_dim: int = 128,
        num_kp: int = 32,
        pretrained: bool = True,
        use_frame_diff: bool = True,
        use_coord_conv: bool = False,
        feature_layer: str = "layer3",
    ) -> None:
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.use_frame_diff = use_frame_diff and n_obs_steps >= 2
        self.use_coord_conv = use_coord_conv
        self.feature_layer = feature_layer
        self.pretrained = pretrained

        in_channels = 3 * n_obs_steps + (2 if use_coord_conv else 0)
        if pretrained:
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            backbone = resnet18(weights=None)

        backbone = _adapt_resnet_first_conv(
            backbone,
            in_channels,
            zero_init_last=2 if use_coord_conv else 0,
        )
        backbone.fc = nn.Identity()
        backbone.avgpool = nn.Identity()
        self.backbone = backbone

        channel_map = {"layer2": 128, "layer3": 256, "layer4": 512}
        if feature_layer not in channel_map:
            raise ValueError(f"feature_layer must be one of {list(channel_map)}")
        feat_channels = channel_map[feature_layer]

        self.spatial_softmax = SpatialSoftmax(in_channels=feat_channels, num_kp=num_kp)
        in_dim = num_kp * 2 + feat_channels
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
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
        self._coord_cached_hw: tuple[int, int] | None = None
        self.register_buffer("_coord_grid", torch.zeros(2, 1, 1), persistent=False)

    def _coord_channels(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """归一化 xy 网格 (2, H, W)，范围 [-1, 1]。"""
        if (
            self._coord_cached_hw == (h, w)
            and self._coord_grid.device == device
            and self._coord_grid.dtype == dtype
        ):
            return self._coord_grid
        ys = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([grid_x, grid_y], dim=0)
        self._coord_grid = pos
        self._coord_cached_hw = (h, w)
        return pos

    def _prepare_images(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, T, 3, H, W) in [0,1] -> (B, T*3[+2], H, W)."""
        obs = (obs - self.img_mean) / self.img_std
        if self.use_frame_diff:
            frames = [obs[:, 0]]
            for t in range(1, obs.shape[1]):
                frames.append(obs[:, t] - obs[:, t - 1])
            x = torch.cat(frames, dim=1)
        else:
            x = obs.flatten(1, 2)
        if self.use_coord_conv:
            b, _, h, w = x.shape
            xy = self._coord_channels(h, w, x.device, x.dtype).unsqueeze(0).expand(b, -1, -1, -1)
            x = torch.cat([x, xy], dim=1)
        return x

    def _extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        if self.feature_layer == "layer2":
            return x
        x = self.backbone.layer3(x)
        if self.feature_layer == "layer3":
            return x
        return self.backbone.layer4(x)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, T, 3, H, W)
        returns: (B, out_dim)
        """
        x = self._prepare_images(obs)
        feats = self._extract_feature_map(x)
        keypoints = self.spatial_softmax(feats)
        appearance = feats.mean(dim=(2, 3))
        return self.proj(torch.cat([keypoints, appearance], dim=-1))


class StateEncoder(nn.Module):
    """对低维状态做 MLP 编码。"""

    def __init__(self, state_dim: int, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiCameraEncoder(nn.Module):
    """多相机 + 多步状态编码器。

    每个相机将 T 帧堆叠（可选帧差）经 ResNet18 编码一次，再与状态特征融合。

    ``share_image_encoder=True``（默认）：所有相机共用一份 ResNet 权重。
    ``share_image_encoder=False``：每相机独立一份权重（参数量约 ×Cams）。
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
        use_coord_conv: bool = False,
        share_image_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.n_obs_steps = n_obs_steps
        self.share_image_encoder = share_image_encoder

        def _make_image_encoder() -> ResNet18Encoder:
            return ResNet18Encoder(
                n_obs_steps=n_obs_steps,
                out_dim=image_out_dim,
                pretrained=pretrained_encoder,
                use_frame_diff=use_frame_diff,
                use_coord_conv=use_coord_conv,
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
            # 串行按相机 forward：5060 Ti 上比 B*Cams 一次过更快（激活更贴合 L2），
            # 且 BN 统计按 batch=B，不混相机。eval 仍用下方 batched 路径。
            cam_feats = [self.image_encoder(obs_images[:, c]) for c in range(cams)]
            return torch.cat(cam_feats, dim=-1)

        # Shared + eval: (B, Cams, ...) -> (B*Cams, ...) for one ResNet pass.
        flat = obs_images.reshape(b * cams, t, *obs_images.shape[3:])
        cam_feats = self.image_encoder(flat)
        return cam_feats.reshape(b, cams, -1).flatten(1)

    def forward(self, obs_images: torch.Tensor, obs_state: torch.Tensor) -> torch.Tensor:
        """
        obs_images: (B, Cams, T, 3, H, W)
        obs_state: (B, T, state_dim)
        returns: cond (B, cond_dim)

        Shared encoder — train: per-camera serial (BN batch = B);
        eval: cameras stacked on batch (B*Cams). Separate encoders always serial.
        """
        b, _, t, _, _, _ = obs_images.shape
        img_feat = self._encode_images(obs_images)

        state = obs_state.reshape(b * t, -1)
        state_feat = self.state_encoder(state).reshape(b, -1)

        fused = torch.cat([img_feat, state_feat], dim=-1)
        return self.proj(fused)


def build_multi_camera_encoder(
    vision_backbone: str,
    *,
    num_cameras: int,
    state_dim: int,
    n_obs_steps: int,
    image_out_dim: int = 128,
    state_out_dim: int = 128,
    cond_dim: int = 256,
    pretrained_encoder: bool = True,
    use_frame_diff: bool = True,
    use_coord_conv: bool = False,
    share_image_encoder: bool = True,
) -> nn.Module:
    """按 ``vision_backbone`` 构造多相机观测编码器（输出 ``cond``）。

    支持：``resnet18`` / ``slowfast_r50`` / ``vit_b_16`` / ``pa2``。
    ``use_coord_conv`` 仅作用于 ResNet 路径。
    """
    backbone = vision_backbone.lower()
    if backbone in {"resnet18", "resnet"}:
        return MultiCameraEncoder(
            num_cameras=num_cameras,
            state_dim=state_dim,
            n_obs_steps=n_obs_steps,
            image_out_dim=image_out_dim,
            state_out_dim=state_out_dim,
            cond_dim=cond_dim,
            pretrained_encoder=pretrained_encoder,
            use_frame_diff=use_frame_diff,
            use_coord_conv=use_coord_conv,
            share_image_encoder=share_image_encoder,
        )
    if backbone in {"slowfast_r50", "slowfast"}:
        from robotfm.policies.video_encoders import MultiCameraSlowFastEncoder

        return MultiCameraSlowFastEncoder(
            num_cameras=num_cameras,
            state_dim=state_dim,
            n_obs_steps=n_obs_steps,
            image_out_dim=image_out_dim,
            state_out_dim=state_out_dim,
            cond_dim=cond_dim,
            pretrained_encoder=pretrained_encoder,
            share_image_encoder=share_image_encoder,
        )
    if backbone in {"vit_b_16", "vit"}:
        from robotfm.policies.vit_encoders import MultiCameraViTEncoder

        return MultiCameraViTEncoder(
            num_cameras=num_cameras,
            state_dim=state_dim,
            n_obs_steps=n_obs_steps,
            image_out_dim=image_out_dim,
            state_out_dim=state_out_dim,
            cond_dim=cond_dim,
            pretrained_encoder=pretrained_encoder,
            use_frame_diff=use_frame_diff,
            share_image_encoder=share_image_encoder,
        )
    if backbone in {"pa2", "casbot_pa2", "casbotpa2"}:
        from robotfm.policies.pa2_encoders import MultiCameraPA2Encoder

        return MultiCameraPA2Encoder(
            num_cameras=num_cameras,
            state_dim=state_dim,
            n_obs_steps=n_obs_steps,
            image_out_dim=image_out_dim,
            state_out_dim=state_out_dim,
            cond_dim=cond_dim,
            pretrained_encoder=pretrained_encoder,
            use_frame_diff=use_frame_diff,
            share_image_encoder=share_image_encoder,
        )
    raise ValueError(
        f"Unknown vision_backbone={vision_backbone!r}; "
        "expected 'resnet18', 'slowfast_r50', 'vit_b_16', or 'pa2'"
    )
