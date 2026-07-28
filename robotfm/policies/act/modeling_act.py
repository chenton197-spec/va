"""Action Chunking Transformer (ACT) — 对齐 LeRobot / 原版 ACT。

参考:
- LeRobot: ``lerobot/policies/act/modeling_act.py``
- 原版: https://github.com/tonyzhaozh/act
- 论文: https://huggingface.co/papers/2304.13705

相对 LeRobot 的适配:
- 输入 batch 使用 robotfm 约定的 ``obs_images`` / ``obs_state`` / ``action`` / ``action_mask``
- 对外接口为 ``compute_loss`` / ``sample_actions``（与 FlowMatchingPolicy 一致）
- 图像在 ``_to_act_batch`` 做 VISUAL MEAN_STD（对齐 LeRobot processor；
  ``image_norm_mode=dataset`` 用数据集统计，``imagenet`` 保留旧硬编码路径）
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from robotfm.policies.act.configuration_act import ACTConfig


class ACTPolicy(nn.Module):
    """ACT 策略封装：训练 ``compute_loss``，推理 ``sample_actions``。"""

    def __init__(
        self,
        config: ACTConfig,
        *,
        image_mean: np.ndarray | Tensor | None = None,
        image_std: np.ndarray | Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = ACT(config)
        # Processor 级图像归一化参数（不进入 CVAE 语义；随 state_dict 保存以保证 eval 一致）
        if image_mean is None:
            image_mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        if image_std is None:
            image_std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        self.register_buffer(
            "image_mean",
            torch.as_tensor(image_mean, dtype=torch.float32).reshape(3),
            persistent=True,
        )
        self.register_buffer(
            "image_std",
            torch.as_tensor(image_std, dtype=torch.float32).reshape(3),
            persistent=True,
        )
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )
        self.reset()

    def _normalize_images(self, images: Tensor) -> Tensor:
        """``images``: (..., 3, H, W) in [0, 1] → dataset / ImageNet MEAN_STD。"""
        mean = self.image_mean.view(*([1] * (images.ndim - 3)), 3, 1, 1)
        std = self.image_std.view(*([1] * (images.ndim - 3)), 3, 1, 1)
        return (images - mean) / std

    def reset(self) -> None:
        """环境 reset 时清空 action queue / temporal ensembler。"""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """L1 重建 + ``kl_weight * KL``（VAE 开启时）。"""
        act_batch = self._to_act_batch(batch, include_actions=True)
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(act_batch)

        abs_err = F.l1_loss(act_batch["action"], actions_hat, reduction="none")
        valid_mask = ~act_batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

        if self.config.use_vae and log_sigma_x2_hat is not None:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp()))
                .sum(-1)
                .mean()
            )
            return l1_loss + mean_kld * self.config.kl_weight
        return l1_loss

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, Tensor],
        **kwargs,
    ) -> Tensor:
        """推理：返回完整 action chunk ``(B, chunk_size, action_dim)``。

        ``kwargs`` 保留以兼容 eval 中可能传入的 RTC 参数（ACT 忽略）。
        """
        del kwargs
        self.eval()
        act_batch = self._to_act_batch(batch, include_actions=False)
        actions = self.model(act_batch)[0]
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """单步动作（队列 / temporal ensemble），供逐步闭环使用。"""
        self.eval()
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.sample_actions(batch)
            return self.temporal_ensembler.update(actions)

        if len(self._action_queue) == 0:
            actions = self.sample_actions(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def _to_act_batch(
        self, batch: dict[str, Tensor], *, include_actions: bool
    ) -> dict[str, Tensor]:
        """robotfm batch → ACT 内部 batch（单帧观测 + pad mask）。"""
        obs_images = batch["obs_images"]  # (B, Cams, T_obs, 3, H, W)
        obs_state = batch["obs_state"]  # (B, T_obs, state_dim)

        # ACT 只用当前帧（最后一帧）；MEAN_STD 对齐 LeRobot VISUAL processor
        images_bt = obs_images[:, :, -1]  # (B, Cams, 3, H, W)
        images_bt = self._normalize_images(images_bt)
        # 拆成每相机一张列表，对齐 LeRobot
        images = [images_bt[:, i] for i in range(images_bt.shape[1])]
        state = obs_state[:, -1]  # (B, state_dim)

        out: dict[str, Tensor] = {
            "observation.images": images,
            "observation.state": state,
        }
        if include_actions:
            action = batch["action"]
            # robotfm: action_mask 1=valid；ACT: action_is_pad True=pad
            mask = batch["action_mask"]
            if mask.ndim == 3:
                mask = mask.squeeze(-1)
            action_is_pad = mask < 0.5
            out["action"] = action
            out["action_is_pad"] = action_is_pad
        return out


class ACTTemporalEnsembler:
    """Algorithm 2 of ACT paper：在线指数加权 temporal ensembling。"""

    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self) -> None:
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[
                self.ensembled_actions_count
            ]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(
                self.ensembled_actions_count + 1, max=self.chunk_size
            )
            self.ensembled_actions = torch.cat(
                [self.ensembled_actions, actions[:, -1:]], dim=1
            )
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACT(nn.Module):
    """ACT 核心网络（CVAE encoder + transformer encoder/decoder）。"""

    def __init__(self, config: ACTConfig) -> None:
        super().__init__()
        self.config = config

        if config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            self.vae_encoder_robot_state_input_proj = nn.Linear(config.state_dim, config.dim_model)
            self.vae_encoder_action_input_proj = nn.Linear(config.action_dim, config.dim_model)
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_input_token_encoder = 1 + 1 + config.chunk_size  # cls + state + actions
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        weights = "IMAGENET1K_V1" if config.pretrained_backbone else None
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
            weights=weights,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        self.encoder_robot_state_input_proj = nn.Linear(config.state_dim, config.dim_model)
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        self.encoder_img_feat_input_proj = nn.Conv2d(
            backbone_model.fc.in_features, config.dim_model, kernel_size=1
        )
        # latent + robot_state
        self.encoder_1d_feature_pos_embed = nn.Embedding(2, config.dim_model)
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, config.action_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, tuple[Tensor | None, Tensor | None]]:
        images: list[Tensor] = batch["observation.images"]
        batch_size = images[0].shape[0]
        device = images[0].device

        if self.config.use_vae and "action" in batch and self.training:
            cls_embed = self.vae_encoder_cls_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
            robot_state_embed = self.vae_encoder_robot_state_input_proj(
                batch["observation.state"]
            ).unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch["action"])
            vae_encoder_input = torch.cat([cls_embed, robot_state_embed, action_embed], dim=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.zeros(
                (batch_size, 2), dtype=torch.bool, device=device
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], dim=1)

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros(
                batch_size, self.config.latent_dim, dtype=torch.float32, device=device
            )

        encoder_in_tokens: list[Tensor] = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed: list[Tensor] = list(
            self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)
        )
        encoder_in_tokens.append(
            self.encoder_robot_state_input_proj(batch["observation.state"])
        )

        for img in images:
            cam_features = self.backbone(img)["feature_map"]
            # 2D pos embed 返回 (1, C, H, W)，与 LeRobot 一致，attention 里广播到 batch
            cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            cam_features = self.encoder_img_feat_input_proj(cam_features)
            # (B, C, H, W) → (H*W, B, C)；pos → (H*W, 1, C)
            b, c, h, w = cam_features.shape
            cam_features = cam_features.permute(2, 3, 0, 1).reshape(h * w, b, c)
            cam_pos_embed = cam_pos_embed.permute(2, 3, 0, 1).reshape(h * w, 1, c)
            encoder_in_tokens.extend(list(cam_features))
            encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens_t = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed_t = torch.stack(encoder_in_pos_embed, dim=0)

        encoder_out = self.encoder(encoder_in_tokens_t, pos_embed=encoder_in_pos_embed_t)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed_t.dtype,
            device=device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed_t,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)
        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False) -> None:
        super().__init__()
        num_layers = config.n_vae_encoder_layers if is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self,
        x: Tensor,
        pos_embed: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        return self.norm(x)


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(
        self,
        x: Tensor,
        pos_embed: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
            )
        return self.norm(x)


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(
            config.dim_model, config.n_heads, dropout=config.dropout
        )
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    def get_position_angle_vec(position: int) -> list[float]:
        return [
            position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)
        ]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, :1])
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi
        inverse_frequency = self._temperature ** (
            2
            * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2)
            / self.dimension
        )
        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency
        pos_embed_x = torch.stack(
            (x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1
        ).flatten(3)
        pos_embed_y = torch.stack(
            (y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1
        ).flatten(3)
        return torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)


def get_activation_fn(activation: str) -> Callable:
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}")
