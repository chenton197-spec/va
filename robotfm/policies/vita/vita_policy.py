"""Vision-to-Action Flow Matching Policy (VITA).

Ported from A2A_Flow_Matching; wired to va batch keys and MultiCameraEncoder.
Reuses A2A action AE / SimpleFlowNet / TorchCFM matchers.

Architecture (aligned with original VITA):
    obs_images + obs_state --encode--> obs_latents (x0)
    actions (full horizon) --encode--> action_latents (x1)
    Flow Matching: x0 --flow(no cond)--> x1
    x1 --decode--> actions
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
from robotfm.policies.encoders import build_multi_camera_encoder


@dataclass
class VITAConfig:
    """VITA hyper-parameters."""

    num_cameras: int
    state_dim: int
    action_dim: int
    horizon: int
    n_obs_steps: int
    n_action_steps: int
    latent_dim: int = 512
    hidden_dim: int = 256  # vision dim before projector
    num_inference_steps: int = 6
    consistency_weight: float = 1.0
    enc_recon_weight: float = 0.5
    flow_recon_weight: float = 0.5
    enc_contrastive_weight: float = 1e-4
    flow_contrastive_weight: float = 0.0
    use_ot_matcher: bool = True
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
    share_image_encoder: bool = True
    vision_backbone: str = "resnet18"  # resnet18 | slowfast_r50 | vit_b_16 | pa2


class VITAPolicy(nn.Module):
    """VITA: flow from visual latents to action latents (no global_cond)."""

    def __init__(self, cfg: VITAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.n_action_steps > cfg.horizon:
            raise ValueError(
                f"n_action_steps ({cfg.n_action_steps}) must be <= horizon ({cfg.horizon})"
            )

        self.encoder = build_multi_camera_encoder(
            cfg.vision_backbone,
            num_cameras=cfg.num_cameras,
            state_dim=cfg.state_dim,
            n_obs_steps=cfg.n_obs_steps,
            cond_dim=cfg.hidden_dim,
            pretrained_encoder=cfg.pretrained_encoder,
            use_frame_diff=cfg.use_frame_diff,
            share_image_encoder=cfg.share_image_encoder,
        )
        self.obs_projector = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        # VITA AE covers the full action horizon (not just n_action_steps).
        self.action_encoder = make_action_encoder(
            seq_length=cfg.horizon,
            action_dim=cfg.action_dim,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.ae_enc_hidden_dim,
        )
        self.action_decoder = SimpleActionDecoder(
            dec_hidden_dim=cfg.ae_dec_hidden_dim,
            latent_dim=cfg.latent_dim,
            pred_horizon=cfg.horizon,
            action_dim=cfg.action_dim,
            num_layers=cfg.ae_num_layers,
            dropout=cfg.ae_dropout,
        )

        # No condition_dim: vision is the flow source, not a condition.
        self.flow_net = SimpleFlowNet(
            input_dim=cfg.latent_dim,
            hidden_dim=cfg.flow_hidden_dim,
            output_dim=cfg.latent_dim,
            num_layers=cfg.flow_num_layers,
            mlp_ratio=cfg.flow_mlp_ratio,
            dropout=cfg.flow_dropout,
            condition_dim=None,
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

    def _encode_obs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cond = self.encoder(batch["obs_images"], batch["obs_state"])
        return self.obs_projector(cond)

    def compute_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """CFM + consistency + recon losses (upstream VITA defaults).

        Returns ``(loss, terms)`` where ``terms`` are **unweighted** scalars:
        ``flow``, ``consistency``, ``enc_recon``, ``flow_recon``.

        batch:
            obs_images: (B, Cams, T_obs, 3, H, W)
            obs_state: (B, T_obs, state_dim)
            action: (B, horizon, action_dim)
            action_mask: (B, horizon, 1)
        """
        actions = batch["action"]
        mask = batch["action_mask"]
        batch_size = actions.shape[0]

        obs_latents = self._encode_obs(batch)
        action_latents = self.action_encoder(actions)

        flow_loss, _metrics = self.flow_matcher.compute_loss(
            self.flow_net,
            target=action_latents,
            start=obs_latents,
        )
        zero = flow_loss.new_zeros(())
        consistency_loss = zero
        flow_recon_loss = zero
        enc_recon_loss = zero
        loss = flow_loss

        if self.cfg.enc_contrastive_weight > 0:
            loss = loss + self.cfg.enc_contrastive_weight * self._compute_contrastive_loss(
                obs_latents.view(batch_size, -1),
                action_latents.view(batch_size, -1),
            )

        if self.cfg.decode_flow_latents:
            action_latents_pred = self.flow_matcher.sample(
                self.flow_net,
                shape=(batch_size, self.cfg.latent_dim),
                device=obs_latents.device,
                start=obs_latents,
                num_steps=self.cfg.num_inference_steps,
            )

            if self.cfg.consistency_weight > 0:
                consistency_loss = F.mse_loss(action_latents_pred, action_latents)
                loss = loss + self.cfg.consistency_weight * consistency_loss

            if self.cfg.flow_contrastive_weight > 0:
                loss = loss + self.cfg.flow_contrastive_weight * self._compute_contrastive_loss(
                    obs_latents.view(batch_size, -1),
                    action_latents_pred.view(batch_size, -1),
                )

            if self.cfg.flow_recon_weight > 0:
                actions_recon = self.action_decoder(action_latents_pred)
                recon = F.l1_loss(actions_recon, actions, reduction="none") * mask
                flow_recon_loss = recon.sum() / mask.expand_as(recon).sum().clamp_min(1.0)
                loss = loss + self.cfg.flow_recon_weight * flow_recon_loss

        if self.cfg.enc_recon_weight > 0:
            actions_recon = self.action_decoder(action_latents)
            recon = F.l1_loss(actions_recon, actions, reduction="none") * mask
            enc_recon_loss = recon.sum() / mask.expand_as(recon).sum().clamp_min(1.0)
            loss = loss + self.cfg.enc_recon_weight * enc_recon_loss

        terms = {
            "flow": flow_loss,
            "consistency": consistency_loss,
            "enc_recon": enc_recon_loss,
            "flow_recon": flow_recon_loss,
        }
        return loss, terms

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, torch.Tensor], **_kwargs) -> torch.Tensor:
        """Euler ODE from visual latents → decode to (B, horizon, action_dim).

        Extra kwargs (e.g. RTC) are ignored for VITA v1.
        """
        b = batch["obs_state"].shape[0]
        device = batch["obs_state"].device

        obs_latents = self._encode_obs(batch)
        action_latents_pred = self.flow_matcher.sample(
            self.flow_net,
            shape=(b, self.cfg.latent_dim),
            device=device,
            num_steps=self.cfg.num_inference_steps,
            start=obs_latents,
            return_traces=False,
        )
        return self.action_decoder(action_latents_pred)  # (B, horizon, A)

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
