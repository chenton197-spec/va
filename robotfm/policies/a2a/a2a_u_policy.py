from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from robotfm.policies.encoders import build_multi_camera_encoder
from robotfm.policies.rtc import RTCConfig, RTCProcessor
from robotfm.policies.unet1d import ConditionalUnet1D


@dataclass
class A2AUConfig:
    num_cameras: int
    state_dim: int
    action_dim: int
    horizon: int
    n_obs_steps: int
    n_action_steps: int
    hidden_dim: int = 256
    num_inference_steps: int = 6
    history_noise_std: float = 0.0
    use_ot_matcher: bool = False
    pretrained_encoder: bool = True
    use_frame_diff: bool = True
    use_coord_conv: bool = False
    share_image_encoder: bool = False
    vision_backbone: str = "resnet18"
    rtc: RTCConfig | None = None
    cameras: tuple[str, ...] | list[str] | None = None
    depth_cameras: tuple[str, ...] | list[str] = ()
    arm_aware: bool = True
    token_grid: int = 8
    use_temporal_attn: bool = True
    use_cross_attn: bool = True
    down_dims: tuple[int, ...] | list[int] = field(default_factory=lambda: (128, 256, 512))
    diffusion_step_embed_dim: int = 256
    kernel_size: int = 5
    n_groups: int = 8
    action_names: tuple[str, ...] | list[str] | None = None


class A2AUPolicy(nn.Module):
    def __init__(self, cfg: A2AUConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.n_action_steps > cfg.horizon:
            raise ValueError(
                f"n_action_steps ({cfg.n_action_steps}) must be <= horizon ({cfg.horizon})"
            )
        if cfg.rtc is not None and cfg.rtc.enabled and cfg.n_action_steps != cfg.horizon:
            raise ValueError(
                "A2A-U RTC requires n_action_steps == horizon "
                f"(got n_action_steps={cfg.n_action_steps}, horizon={cfg.horizon})"
            )
        token_grid = int(cfg.token_grid) if cfg.use_cross_attn else 0
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
        )
        self.unet = ConditionalUnet1D(
            input_dim=cfg.action_dim,
            global_cond_dim=cfg.hidden_dim,
            diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
            down_dims=tuple(cfg.down_dims),
            kernel_size=cfg.kernel_size,
            n_groups=cfg.n_groups,
            use_temporal_attn=cfg.use_temporal_attn,
            use_cross_attn=cfg.use_cross_attn,
            vision_dim=cfg.hidden_dim,
        )
        self.hist_proj = (
            nn.Linear(cfg.state_dim, cfg.action_dim)
            if cfg.state_dim != cfg.action_dim
            else None
        )
        self.flow_matcher = None
        if cfg.use_ot_matcher:
            from robotfm.policies.a2a.flow_matchers import (
                ExactOptimalTransportConditionalFlowMatcher,
            )

            self.flow_matcher = ExactOptimalTransportConditionalFlowMatcher(
                sigma=0.0,
                num_sampling_steps=cfg.num_inference_steps,
            )
        self.rtc_processor: RTCProcessor | None = None
        if cfg.rtc is not None and cfg.rtc.enabled:
            self.rtc_processor = RTCProcessor(cfg.rtc)

    def _rtc_enabled(self) -> bool:
        return self.rtc_processor is not None and self.cfg.rtc is not None and self.cfg.rtc.enabled

    def _rtc_guidance_enabled(self) -> bool:
        return self._rtc_enabled() and bool(self.cfg.rtc.guidance_enabled)

    def _add_history_noise(self, history_states: torch.Tensor) -> torch.Tensor:
        std = self.cfg.history_noise_std
        if std > 0:
            return history_states + torch.randn_like(history_states) * std
        return history_states

    def _obs_cond(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        depth = batch.get("obs_depth")
        if hasattr(self.encoder, "encode_obs"):
            return self.encoder.encode_obs(batch["obs_images"], batch["obs_state"], depth)
        cond = self.encoder(batch["obs_images"], batch["obs_state"])
        return cond, None

    def _history_as_x0(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        history = batch["obs_history"] if "obs_history" in batch else batch["obs_state"]
        history = self._add_history_noise(history)
        if self.hist_proj is not None:
            history = self.hist_proj(history)
        b, t, d = history.shape
        horizon = self.cfg.horizon
        if t == horizon:
            return history
        if t > horizon:
            return history[:, -horizon:]
        last = history[:, -1:, :].expand(b, horizon - t, d)
        return torch.cat([history, last], dim=1)

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

    def compute_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1 = batch["action"]
        mask = batch.get("action_mask")
        if mask is None:
            mask = torch.ones(x1.shape[0], x1.shape[1], 1, device=x1.device, dtype=x1.dtype)
        x0 = self._history_as_x0(batch)
        b = x1.shape[0]
        if self.flow_matcher is not None:
            x0f = x0.reshape(b, -1)
            x1f = x1.reshape(b, -1)
            t, xtf, utf = self.flow_matcher.fm.sample_location_and_conditional_flow(x0f, x1f)
            xt = xtf.reshape_as(x0)
            ut = utf.reshape_as(x0)
        else:
            t = torch.rand(b, device=x1.device, dtype=x1.dtype)
            t_view = t[:, None, None]
            xt = (1.0 - t_view) * x0 + t_view * x1
            ut = x1 - x0
        cond, tokens = self._obs_cond(batch)
        pred = self.unet(xt, t, cond, vision_tokens=tokens)
        weight = self._cfm_weight(x1.shape[1], x1.shape[2], x1.device, x1.dtype)
        per = (pred.float() - ut.float()) ** 2 * weight * mask
        flow_loss = per.sum() / mask.expand_as(pred).sum().clamp_min(1.0)
        zero = flow_loss.new_zeros(())
        return flow_loss, {
            "flow": flow_loss,
            "consistency": zero,
            "enc_recon": zero,
            "flow_recon": zero,
        }

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
        x = self._history_as_x0(batch)
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
                return self.unet(
                    input_x_t, current_t, current_cond, vision_tokens=current_tokens
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
