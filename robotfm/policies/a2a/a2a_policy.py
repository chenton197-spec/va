"""Action-to-Action Flow Matching Policy (A2A / N-A2A).

Ported from A2A_Flow_Matching; wired to va batch keys and MultiCameraEncoder.

Architecture (aligned with original A2A):
    obs_state (agent_pos history) --encode--> history_latents (x0)
    obs_images + obs_state --encode--> obs_latents (condition)
    Flow Matching: x0 --flow(condition)--> x1 (future_action_latents)
    x1 --decode--> future actions
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from robotfm.policies.a2a.action_ae import SimpleActionDecoder, make_action_encoder
from robotfm.policies.a2a.flow_matchers import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TorchFlowMatcher,
)
from robotfm.policies.a2a.flow_net import SimpleFlowNet
from robotfm.policies.encoders import MultiCameraEncoder
from robotfm.policies.rtc import RTCConfig, RTCProcessor


@dataclass
class A2AConfig:
    """A2A / N-A2A hyper-parameters."""

    num_cameras: int
    state_dim: int
    action_dim: int
    horizon: int
    n_obs_steps: int
    n_action_steps: int
    latent_dim: int = 512
    hidden_dim: int = 256  # vision cond dim before projector
    num_inference_steps: int = 6
    consistency_weight: float = 1.0
    enc_recon_weight: float = 0.5
    flow_recon_weight: float = 0.5
    enc_contrastive_weight: float = 0.0
    flow_contrastive_weight: float = 0.0
    history_noise_std: float = 0.0
    use_ot_matcher: bool = False
    decode_flow_latents: bool = True
    flow_hidden_dim: int = 512
    flow_num_layers: int = 4
    flow_mlp_ratio: float = 4.0
    flow_dropout: float = 0.0
    ae_enc_hidden_dim: int = 512
    ae_dec_hidden_dim: int = 512
    ae_num_layers: int = 4
    ae_dropout: float = 0.0
    pretrained_encoder: bool = True
    use_frame_diff: bool = True
    rtc: RTCConfig | None = None


class A2APolicy(nn.Module):
    """A2A / N-A2A: flow from state history to future actions (optional history noise)."""

    def __init__(self, cfg: A2AConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.n_action_steps > cfg.horizon:
            raise ValueError(
                f"n_action_steps ({cfg.n_action_steps}) must be <= horizon ({cfg.horizon})"
            )
        if cfg.rtc is not None and cfg.rtc.enabled and cfg.n_action_steps != cfg.horizon:
            raise ValueError(
                "A2A RTC requires n_action_steps == horizon "
                f"(got n_action_steps={cfg.n_action_steps}, horizon={cfg.horizon})"
            )

        self.encoder = MultiCameraEncoder(
            num_cameras=cfg.num_cameras,
            state_dim=cfg.state_dim,
            n_obs_steps=cfg.n_obs_steps,
            cond_dim=cfg.hidden_dim,
            pretrained_encoder=cfg.pretrained_encoder,
            use_frame_diff=cfg.use_frame_diff,
        )
        self.obs_projector = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        # Flow source encoder: agent_pos / state history (original A2A), not commanded actions.
        # Attribute name kept for checkpoint key compatibility within this module.
        self.history_action_encoder = make_action_encoder(
            seq_length=cfg.n_obs_steps,
            action_dim=cfg.state_dim,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.ae_enc_hidden_dim,
        )
        self.action_encoder = make_action_encoder(
            seq_length=cfg.n_action_steps,
            action_dim=cfg.action_dim,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.ae_enc_hidden_dim,
        )
        self.action_decoder = SimpleActionDecoder(
            dec_hidden_dim=cfg.ae_dec_hidden_dim,
            latent_dim=cfg.latent_dim,
            pred_horizon=cfg.n_action_steps,
            action_dim=cfg.action_dim,
            num_layers=cfg.ae_num_layers,
            dropout=cfg.ae_dropout,
        )

        self.flow_net = SimpleFlowNet(
            input_dim=cfg.latent_dim,
            hidden_dim=cfg.flow_hidden_dim,
            output_dim=cfg.latent_dim,
            num_layers=cfg.flow_num_layers,
            mlp_ratio=cfg.flow_mlp_ratio,
            dropout=cfg.flow_dropout,
            condition_dim=cfg.latent_dim,
        )

        matcher_cls = (
            ExactOptimalTransportConditionalFlowMatcher
            if cfg.use_ot_matcher
            else ConditionalFlowMatcher
        )
        self.flow_matcher: TorchFlowMatcher = matcher_cls(
            sigma=0.0,
            num_sampling_steps=cfg.num_inference_steps,
        )

        self.rtc_processor: RTCProcessor | None = None
        if cfg.rtc is not None and cfg.rtc.enabled:
            self.rtc_processor = RTCProcessor(cfg.rtc)

    def _rtc_enabled(self) -> bool:
        return self.rtc_processor is not None and self.cfg.rtc is not None and self.cfg.rtc.enabled

    def _add_history_noise(self, history_states: torch.Tensor) -> torch.Tensor:
        std = self.cfg.history_noise_std
        if std > 0:
            return history_states + torch.randn_like(history_states) * std
        return history_states

    def _encode_obs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cond = self.encoder(batch["obs_images"], batch["obs_state"])
        return self.obs_projector(cond)

    def _encode_history(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode state history as flow source (original A2A: agent_pos)."""
        history = self._add_history_noise(batch["obs_state"])
        return self.history_action_encoder(history)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """CFM + consistency + recon losses (A2A_Flow_Matching defaults).

        batch:
            obs_images: (B, Cams, T_obs, 3, H, W)
            obs_state: (B, T_obs, state_dim)  — also used as flow source
            action: (B, horizon, action_dim)
            action_mask: (B, horizon, 1)
        """
        actions = batch["action"]
        mask = batch["action_mask"]
        batch_size = actions.shape[0]
        n_act = self.cfg.n_action_steps

        future_actions = actions[:, :n_act]
        future_mask = mask[:, :n_act]

        obs_latents = self._encode_obs(batch)
        history_latents = self._encode_history(batch)
        future_action_latents = self.action_encoder(future_actions)

        flow_loss, _metrics = self.flow_matcher.compute_loss(
            self.flow_net,
            target=future_action_latents,
            start=history_latents,
            global_cond=obs_latents,
        )
        loss = flow_loss

        if self.cfg.enc_contrastive_weight > 0:
            loss = loss + self.cfg.enc_contrastive_weight * self._compute_contrastive_loss(
                obs_latents.view(batch_size, -1),
                future_action_latents.view(batch_size, -1),
            )

        if self.cfg.decode_flow_latents:
            action_latents_pred = self.flow_matcher.sample(
                self.flow_net,
                shape=(batch_size, self.cfg.latent_dim),
                device=obs_latents.device,
                start=history_latents,
                num_steps=self.cfg.num_inference_steps,
                global_cond=obs_latents,
            )

            if self.cfg.consistency_weight > 0:
                consistency_loss = F.mse_loss(action_latents_pred, future_action_latents)
                loss = loss + self.cfg.consistency_weight * consistency_loss

            if self.cfg.flow_contrastive_weight > 0:
                loss = loss + self.cfg.flow_contrastive_weight * self._compute_contrastive_loss(
                    obs_latents.view(batch_size, -1),
                    action_latents_pred.view(batch_size, -1),
                )

            if self.cfg.flow_recon_weight > 0:
                actions_recon = self.action_decoder(action_latents_pred)
                recon = F.l1_loss(actions_recon, future_actions, reduction="none") * future_mask
                loss = loss + self.cfg.flow_recon_weight * (
                    recon.sum() / future_mask.sum().clamp_min(1.0)
                )

        if self.cfg.enc_recon_weight > 0:
            actions_recon = self.action_decoder(future_action_latents)
            recon = F.l1_loss(actions_recon, future_actions, reduction="none") * future_mask
            loss = loss + self.cfg.enc_recon_weight * (
                recon.sum() / future_mask.sum().clamp_min(1.0)
            )

        return loss

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, torch.Tensor],
        *,
        prev_chunk_left_over: torch.Tensor | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
    ) -> torch.Tensor:
        """Euler ODE from state-history latents → decode to (B, horizon, action_dim).

        RTC (optional): each latent Euler step guides via ``decode_x1`` so prefix
        actions match ``prev_chunk_left_over`` (same kwargs as FlowMatchingPolicy).
        """
        b = batch["obs_state"].shape[0]
        device = batch["obs_state"].device
        dtype = batch["obs_state"].dtype

        obs_latents = self._encode_obs(batch)
        history_latents = self._encode_history(batch)

        rtc_enabled = self._rtc_enabled()
        if rtc_enabled:
            if inference_delay is None and self.cfg.rtc is not None:
                inference_delay = self.cfg.rtc.inference_delay
            if execution_horizon is None and self.cfg.rtc is not None:
                execution_horizon = self.cfg.rtc.execution_horizon

            x = history_latents
            steps = self.cfg.num_inference_steps
            dt = 1.0 / steps
            for i in range(steps):
                t_val = i / steps
                t = torch.full((b,), t_val, device=device, dtype=dtype)

                def denoise_step_partial(
                    input_x_t,
                    current_t=t,
                    current_cond=obs_latents,
                ):
                    return self.flow_net(input_x_t, current_t, global_cond=current_cond)

                v = self.rtc_processor.denoise_step(
                    x_t=x,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=t_val,
                    original_denoise_step_partial=denoise_step_partial,
                    execution_horizon=execution_horizon,
                    decode_x1=self.action_decoder,
                )
                x = x + dt * v

                if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                    self.rtc_processor.track(time=t_val, x_t=x, v_t=v)

            action_latents_pred = x
        else:
            action_latents_pred = self.flow_matcher.sample(
                self.flow_net,
                shape=(b, self.cfg.latent_dim),
                device=device,
                num_steps=self.cfg.num_inference_steps,
                start=history_latents,
                global_cond=obs_latents,
                return_traces=False,
            )

        action_pred = self.action_decoder(action_latents_pred)  # (B, n_action_steps, A)

        # Pad to horizon if n_action_steps < horizon (FM interface compatibility).
        if action_pred.shape[1] < self.cfg.horizon:
            pad = torch.zeros(
                b,
                self.cfg.horizon - action_pred.shape[1],
                self.cfg.action_dim,
                device=device,
                dtype=action_pred.dtype,
            )
            action_pred = torch.cat([action_pred, pad], dim=1)
        return action_pred

    @staticmethod
    def _compute_contrastive_loss(image_features, action_features, temperature=0.07):
        batch_size = image_features.size(0)
        image_features = F.normalize(image_features, dim=1)
        action_features = F.normalize(action_features, dim=1)

        logits = torch.matmul(image_features, action_features.T) / temperature
        labels = torch.arange(batch_size, device=logits.device)
        loss_i2a = F.cross_entropy(logits, labels)
        loss_a2i = F.cross_entropy(logits.T, labels)
        return (loss_i2a + loss_a2i) / 2
