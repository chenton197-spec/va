from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Beta

from robotfm.policies.encoders import arm_dim_indices, build_multi_camera_encoder
from robotfm.policies.rtc import RTCConfig, RTCProcessor
from robotfm.policies.unet1d import ConditionalUnet1D


@dataclass
class FlowMatchingConfig:
    num_cameras: int
    state_dim: int
    action_dim: int
    horizon: int
    n_obs_steps: int
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_inference_steps: int = 10
    beta_alpha: float = 1.5
    beta_beta: float = 1.0
    noise_s: float = 0.999
    down_dims: tuple[int, ...] = (256, 512, 1024)
    diffusion_step_embed_dim: int = 256
    kernel_size: int = 5
    n_groups: int = 8
    pretrained_encoder: bool = True
    use_frame_diff: bool = True
    use_coord_conv: bool = False
    share_image_encoder: bool = True
    vision_backbone: str = "resnet18"
    rtc: RTCConfig | None = None
    cameras: tuple[str, ...] | list[str] | None = None
    depth_cameras: tuple[str, ...] | list[str] = ()
    arm_aware: bool = False
    token_grid: int = 8
    use_temporal_attn: bool = True
    use_cross_attn: bool = True
    action_names: tuple[str, ...] | list[str] | None = None
    state_names: tuple[str, ...] | list[str] | None = None
    split_arm_unet: bool = False


class FlowMatchingPolicy(nn.Module):
    def __init__(self, cfg: FlowMatchingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.split_arm_unet = bool(cfg.split_arm_unet)
        token_grid = int(cfg.token_grid) if cfg.use_cross_attn else 0
        state_names = cfg.state_names if cfg.state_names is not None else cfg.action_names
        self.encoder = build_multi_camera_encoder(
            cfg.vision_backbone,
            num_cameras=cfg.num_cameras,
            state_dim=cfg.state_dim,
            n_obs_steps=cfg.n_obs_steps,
            cond_dim=cfg.hidden_dim,
            pretrained_encoder=cfg.pretrained_encoder,
            use_frame_diff=cfg.use_frame_diff,
            use_coord_conv=cfg.use_coord_conv,
            share_image_encoder=cfg.share_image_encoder,
            cameras=cfg.cameras,
            depth_cameras=tuple(cfg.depth_cameras),
            arm_aware=bool(cfg.arm_aware),
            token_grid=token_grid,
            split_arm_unet=self.split_arm_unet,
            state_names=state_names,
        )
        unet_kwargs = dict(
            global_cond_dim=cfg.hidden_dim,
            diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
            down_dims=tuple(cfg.down_dims),
            kernel_size=cfg.kernel_size,
            n_groups=cfg.n_groups,
            cond_predict_scale=True,
            use_temporal_attn=cfg.use_temporal_attn,
            use_cross_attn=cfg.use_cross_attn,
            vision_dim=cfg.hidden_dim,
        )
        if self.split_arm_unet:
            left_idx, right_idx = arm_dim_indices(cfg.action_names, cfg.action_dim)
            self.register_buffer(
                "left_action_index", torch.tensor(left_idx, dtype=torch.long)
            )
            self.register_buffer(
                "right_action_index", torch.tensor(right_idx, dtype=torch.long)
            )
            self.unet_left = ConditionalUnet1D(input_dim=len(left_idx), **unet_kwargs)
            self.unet_right = ConditionalUnet1D(input_dim=len(right_idx), **unet_kwargs)
        else:
            self.unet = ConditionalUnet1D(input_dim=cfg.action_dim, **unet_kwargs)
        self._beta = Beta(cfg.beta_alpha, cfg.beta_beta)
        self.rtc_processor: RTCProcessor | None = None
        if cfg.rtc is not None and cfg.rtc.enabled:
            self.rtc_processor = RTCProcessor(cfg.rtc)

    def _rtc_enabled(self) -> bool:
        return self.rtc_processor is not None and self.cfg.rtc is not None and self.cfg.rtc.enabled

    def _rtc_guidance_enabled(self) -> bool:
        return self._rtc_enabled() and bool(self.cfg.rtc.guidance_enabled)

    def _merge_arm_actions(
        self, left: torch.Tensor, right: torch.Tensor, like: torch.Tensor
    ) -> torch.Tensor:
        out = torch.zeros_like(like)
        out[..., self.left_action_index] = left
        out[..., self.right_action_index] = right
        return out

    def _unet_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        tokens: torch.Tensor | tuple[torch.Tensor | None, torch.Tensor | None] | None,
    ) -> torch.Tensor:
        if not self.split_arm_unet:
            return self.unet(x, t, cond, vision_tokens=tokens)
        left_cond, right_cond = cond
        left_tok, right_tok = tokens if tokens is not None else (None, None)
        pred_l = self.unet_left(
            x.index_select(-1, self.left_action_index),
            t,
            left_cond,
            vision_tokens=left_tok,
        )
        pred_r = self.unet_right(
            x.index_select(-1, self.right_action_index),
            t,
            right_cond,
            vision_tokens=right_tok,
        )
        return self._merge_arm_actions(pred_l, pred_r, x)

    def _obs_cond(self, batch: dict[str, torch.Tensor]):
        depth = batch.get("obs_depth")
        if self.split_arm_unet:
            left_cond, right_cond, left_tok, right_tok = self.encoder.encode_arm_obs(
                batch["obs_images"], batch["obs_state"], depth
            )
            return (left_cond, right_cond), (left_tok, right_tok)
        if hasattr(self.encoder, "encode_obs"):
            return self.encoder.encode_obs(batch["obs_images"], batch["obs_state"], depth)
        cond = self.encoder(batch["obs_images"], batch["obs_state"])
        return cond, None

    def sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sample = self._beta.sample((batch_size,)).to(device=device, dtype=dtype)
        return (self.cfg.noise_s - sample) / self.cfg.noise_s

    def _cfm_weight(
        self, horizon: int, act_dim: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        t = torch.linspace(1.0, 0.35, horizon, device=device, dtype=dtype)
        w = t[:, None].expand(horizon, act_dim).clone()
        names = list(self.cfg.action_names or [])
        grip = [i for i, n in enumerate(names) if "gripper" in str(n).lower()]
        if not grip and act_dim >= 2:
            grip = [act_dim - 1]
        for i in grip:
            if i < act_dim:
                w[:, i] = w[:, i] * 2.0
        return w

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        actions = batch["action"]
        mask = batch["action_mask"]
        cond, tokens = self._obs_cond(batch)
        noise = torch.randn_like(actions)
        t = self.sample_time(actions.shape[0], actions.device, actions.dtype)
        t_view = t[:, None, None]
        x_t = (1.0 - t_view) * noise + t_view * actions
        target_v = actions - noise
        pred_v = self._unet_forward(x_t, t, cond, tokens)
        weight = self._cfm_weight(actions.shape[1], actions.shape[2], actions.device, actions.dtype)
        per = (pred_v.float() - target_v.float()) ** 2 * weight * mask
        phase_w = batch.get("action_loss_w")
        if phase_w is not None:
            per = per * phase_w.to(device=per.device, dtype=per.dtype)
        return per.sum() / mask.expand_as(pred_v).sum().clamp_min(1.0)

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, torch.Tensor],
        *,
        prev_chunk_left_over: torch.Tensor | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
    ) -> torch.Tensor:
        b = batch["obs_state"].shape[0]
        device = batch["obs_state"].device
        dtype = batch["obs_state"].dtype
        cond, tokens = self._obs_cond(batch)
        x = torch.randn(b, self.cfg.horizon, self.cfg.action_dim, device=device, dtype=dtype)
        steps = self.cfg.num_inference_steps
        dt = 1.0 / steps
        use_guidance = self._rtc_guidance_enabled()
        if inference_delay is None and self.cfg.rtc is not None:
            inference_delay = self.cfg.rtc.inference_delay
        if execution_horizon is None and self.cfg.rtc is not None:
            execution_horizon = self.cfg.rtc.execution_horizon
        for i in range(steps):
            t_val = i / steps
            t = torch.full((b,), t_val, device=device, dtype=dtype)

            def denoise_step_partial(
                input_x_t,
                current_t=t,
                current_cond=cond,
                current_tokens=tokens,
            ):
                return self._unet_forward(
                    input_x_t, current_t, current_cond, current_tokens
                )

            if use_guidance:
                assert self.rtc_processor is not None
                v = self.rtc_processor.denoise_step(
                    x_t=x,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=t_val,
                    original_denoise_step_partial=denoise_step_partial,
                    execution_horizon=execution_horizon,
                )
            else:
                v = denoise_step_partial(x)
            x = x + dt * v
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=t_val, x_t=x, v_t=v)
        return x
