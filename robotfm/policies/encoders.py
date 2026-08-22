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
        extra_channels: int = 0,
        token_grid: int = 0,
        token_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.use_frame_diff = use_frame_diff and n_obs_steps >= 2
        self.use_coord_conv = use_coord_conv
        self.feature_layer = feature_layer
        self.pretrained = pretrained
        self.extra_channels = int(extra_channels)
        self.token_grid = int(token_grid)
        self.channels_per_frame = 3 + self.extra_channels

        in_channels = self.channels_per_frame * n_obs_steps + (2 if use_coord_conv else 0)
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
        tok_dim = int(token_dim) if token_dim is not None else out_dim
        if self.token_grid > 0:
            self.token_proj = nn.Sequential(
                nn.Conv2d(feat_channels, tok_dim, kernel_size=1, bias=False),
                nn.GroupNorm(8, tok_dim),
                nn.SiLU(),
            )
        else:
            self.token_proj = None

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
        """obs: (B, T, 3[+extra], H, W) in [0,1] -> (B, T*C[+2], H, W)."""
        if self.extra_channels > 0:
            rgb = obs[:, :, :3]
            extra = obs[:, :, 3:]
            rgb = (rgb - self.img_mean) / self.img_std
            extra = (extra - 0.5) / 0.5
            obs = torch.cat([rgb, extra], dim=2)
        else:
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

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self._prepare_images(obs)
        feats = self._extract_feature_map(x)
        keypoints = self.spatial_softmax(feats)
        appearance = feats.mean(dim=(2, 3))
        global_emb = self.proj(torch.cat([keypoints, appearance], dim=-1))
        tokens = None
        if self.token_proj is not None:
            pooled = F.adaptive_avg_pool2d(
                feats, output_size=(self.token_grid, self.token_grid)
            )
            tokens = self.token_proj(pooled).flatten(2).transpose(1, 2).contiguous()
        return global_emb, tokens

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        global_emb, _ = self.encode(obs)
        return global_emb


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


class ProprioEncoder(nn.Module):
    def __init__(self, proprio_dim: int, n_obs_steps: int, cond_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proprio_dim * n_obs_steps, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.mlp(proprio.reshape(proprio.shape[0], -1))


class MultiCameraEncoder(nn.Module):
    """多相机 + 多步状态编码器。

    每个相机将 T 帧堆叠（可选帧差）经 ResNet18 编码一次，再与状态特征融合。

    ``share_image_encoder=True``（默认）：所有相机共用一份 ResNet 权重。
    ``share_image_encoder=False``：每相机独立一份权重（参数量约 ×Cams）。
    ``depth_cameras`` 非空时禁止共享编码器（RGB-D stem 与 RGB 不同）。
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
        cameras: tuple[str, ...] | list[str] | None = None,
        depth_cameras: tuple[str, ...] | list[str] = (),
        arm_aware: bool = False,
        token_grid: int = 0,
    ) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.n_obs_steps = n_obs_steps
        self.cond_dim = cond_dim
        self.token_grid = int(token_grid)
        self.cameras = tuple(
            cameras if cameras is not None else [f"cam{i}" for i in range(num_cameras)]
        )
        if len(self.cameras) != num_cameras:
            raise ValueError(
                f"cameras length {len(self.cameras)} != num_cameras {num_cameras}"
            )
        self.depth_cameras = tuple(depth_cameras)
        self.depth_camera_set = set(self.depth_cameras)
        if self.depth_cameras and share_image_encoder:
            raise ValueError(
                "share_image_encoder=True is incompatible with depth_cameras"
            )
        self.share_image_encoder = share_image_encoder
        cam_set = set(self.cameras)
        self.dual_arm_aware = bool(
            arm_aware
            and state_dim == 16
            and cam_set >= {"left_hand", "right_hand", "head"}
        )
        self.left_arm_aware = bool(
            arm_aware
            and not self.dual_arm_aware
            and state_dim == 8
            and cam_set >= {"left_hand", "head"}
            and "right_hand" not in cam_set
        )
        self.arm_aware = self.dual_arm_aware or self.left_arm_aware
        img_dim = cond_dim if self.arm_aware else image_out_dim

        def _make_image_encoder(name: str) -> ResNet18Encoder:
            return ResNet18Encoder(
                n_obs_steps=n_obs_steps,
                out_dim=img_dim,
                pretrained=pretrained_encoder,
                use_frame_diff=use_frame_diff,
                use_coord_conv=use_coord_conv,
                extra_channels=1 if name in self.depth_camera_set else 0,
                token_grid=self.token_grid,
                token_dim=cond_dim if self.token_grid > 0 else None,
            )

        if share_image_encoder:
            self.image_encoder = _make_image_encoder(self.cameras[0] if self.cameras else "cam0")
        else:
            self.image_encoders = nn.ModuleList(
                [_make_image_encoder(name) for name in self.cameras]
            )
        if self.token_grid > 0:
            self.camera_embed = nn.Parameter(torch.zeros(len(self.cameras), cond_dim))
            nn.init.normal_(self.camera_embed, std=0.02)
        else:
            self.camera_embed = None

        if self.dual_arm_aware:
            half = state_dim // 2
            self.left_proprio = ProprioEncoder(half, n_obs_steps, cond_dim)
            self.right_proprio = ProprioEncoder(half, n_obs_steps, cond_dim)
            self.arm_fuse = nn.ModuleDict(
                {
                    "left": nn.Sequential(
                        nn.Linear(cond_dim * 2, cond_dim),
                        nn.LayerNorm(cond_dim),
                        nn.SiLU(),
                        nn.Linear(cond_dim, cond_dim),
                    ),
                    "right": nn.Sequential(
                        nn.Linear(cond_dim * 2, cond_dim),
                        nn.LayerNorm(cond_dim),
                        nn.SiLU(),
                        nn.Linear(cond_dim, cond_dim),
                    ),
                }
            )
            self.fuse = nn.Sequential(
                nn.Linear(cond_dim * 3, cond_dim * 2),
                nn.LayerNorm(cond_dim * 2),
                nn.SiLU(),
                nn.Linear(cond_dim * 2, cond_dim),
            )
            self.state_encoder = None
            self.proj = None
        elif self.left_arm_aware:
            self.left_proprio = ProprioEncoder(state_dim, n_obs_steps, cond_dim)
            self.right_proprio = None
            self.arm_fuse = nn.ModuleDict(
                {
                    "left": nn.Sequential(
                        nn.Linear(cond_dim * 2, cond_dim),
                        nn.LayerNorm(cond_dim),
                        nn.SiLU(),
                        nn.Linear(cond_dim, cond_dim),
                    ),
                }
            )
            self.fuse = nn.Sequential(
                nn.Linear(cond_dim * 2, cond_dim * 2),
                nn.LayerNorm(cond_dim * 2),
                nn.SiLU(),
                nn.Linear(cond_dim * 2, cond_dim),
            )
            self.state_encoder = None
            self.proj = None
        else:
            self.left_proprio = None
            self.right_proprio = None
            self.arm_fuse = None
            self.fuse = None
            self.state_encoder = StateEncoder(state_dim=state_dim, out_dim=state_out_dim)
            fused_dim = num_cameras * img_dim + state_out_dim * n_obs_steps
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

    def _encoder_for(self, name: str) -> ResNet18Encoder:
        if self.share_image_encoder:
            return self.image_encoder
        return self.image_encoders[self.cameras.index(name)]

    def _camera_input(
        self,
        obs_images: torch.Tensor,
        cam_idx: int,
        name: str,
        obs_depth: torch.Tensor | None,
    ) -> torch.Tensor:
        rgb = obs_images[:, cam_idx]
        if name not in self.depth_camera_set:
            return rgb
        if obs_depth is None:
            raise KeyError(f"Missing obs_depth for depth camera {name!r}")
        d_idx = self.depth_cameras.index(name)
        return torch.cat([rgb, obs_depth[:, d_idx]], dim=2)

    def encode_obs(
        self,
        obs_images: torch.Tensor,
        obs_state: torch.Tensor,
        obs_depth: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, cams, t, _, _, _ = obs_images.shape
        if cams != self.num_cameras:
            raise ValueError(f"Expected {self.num_cameras} cameras, got {cams}")

        cam_globals: list[torch.Tensor] = []
        cam_tokens: list[torch.Tensor] = []
        use_tokens = self.token_grid > 0
        can_batch = (
            self.share_image_encoder
            and not self.training
            and not self.depth_camera_set
        )
        if can_batch:
            flat = obs_images.reshape(b * cams, t, *obs_images.shape[3:])
            g, tok = self.image_encoder.encode(flat)
            g = g.reshape(b, cams, -1)
            for i in range(cams):
                cam_globals.append(g[:, i])
                if use_tokens and tok is not None:
                    n_tok = tok.shape[1]
                    cam_tok = tok.reshape(b, cams, n_tok, tok.shape[-1])[:, i]
                    cam_tokens.append(cam_tok + self.camera_embed[i].view(1, 1, -1))
        else:
            for i, name in enumerate(self.cameras):
                inp = self._camera_input(obs_images, i, name, obs_depth)
                g, tok = self._encoder_for(name).encode(inp)
                cam_globals.append(g)
                if use_tokens and tok is not None:
                    cam_tokens.append(tok + self.camera_embed[i].view(1, 1, -1))

        vision_tokens = torch.cat(cam_tokens, dim=1) if cam_tokens else None

        if self.dual_arm_aware:
            by_name = {name: g for name, g in zip(self.cameras, cam_globals)}
            left_p = self.left_proprio(obs_state[:, :, :8])
            right_p = self.right_proprio(obs_state[:, :, 8:])
            left = self.arm_fuse["left"](torch.cat([by_name["left_hand"], left_p], dim=-1))
            right = self.arm_fuse["right"](torch.cat([by_name["right_hand"], right_p], dim=-1))
            global_emb = self.fuse(torch.cat([by_name["head"], left, right], dim=-1))
        elif self.left_arm_aware:
            by_name = {name: g for name, g in zip(self.cameras, cam_globals)}
            left_p = self.left_proprio(obs_state)
            left = self.arm_fuse["left"](torch.cat([by_name["left_hand"], left_p], dim=-1))
            global_emb = self.fuse(torch.cat([by_name["head"], left], dim=-1))
        else:
            img_feat = torch.cat(cam_globals, dim=-1)
            state = obs_state.reshape(b * t, -1)
            state_feat = self.state_encoder(state).reshape(b, -1)
            global_emb = self.proj(torch.cat([img_feat, state_feat], dim=-1))
        return global_emb, vision_tokens

    def forward(
        self,
        obs_images: torch.Tensor,
        obs_state: torch.Tensor,
        obs_depth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        global_emb, _ = self.encode_obs(obs_images, obs_state, obs_depth)
        return global_emb


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
    cameras: tuple[str, ...] | list[str] | None = None,
    depth_cameras: tuple[str, ...] | list[str] = (),
    arm_aware: bool = False,
    token_grid: int = 0,
) -> nn.Module:
    """按 ``vision_backbone`` 构造多相机观测编码器（输出 ``cond``）。

    支持：``resnet18`` / ``slowfast_r50`` / ``vit_b_16`` / ``pa2``。
    ``use_coord_conv`` / RGB-D / arm_aware / tokens 仅作用于 ResNet 路径。
    """
    backbone = vision_backbone.lower()
    depth_cameras = tuple(depth_cameras)
    if depth_cameras and backbone not in {"resnet18", "resnet"}:
        raise ValueError("RGB-D early fusion is only supported for vision_backbone=resnet18")
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
            cameras=cameras,
            depth_cameras=depth_cameras,
            arm_aware=arm_aware,
            token_grid=token_grid,
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
