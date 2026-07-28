"""Real-Time Chunking (RTC) processor for robotfm.

Ported from LeRobot RTCProcessor with time-convention adapted for robotfm's
flow-matching schedule (t: 0 noise → 1 data), matching the PI original convention.

Reference:
  https://www.physicalintelligence.company/download/real_time_chunking.pdf
  https://github.com/Physical-Intelligence/real-time-chunking-kinetix
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor

from robotfm.policies.rtc.configuration_rtc import RTCAttentionSchedule, RTCConfig
from robotfm.policies.rtc.debug_tracker import Tracker

logger = logging.getLogger(__name__)


class RTCProcessor:
    """Real-Time Chunking processor for action chunking policies."""

    def __init__(self, rtc_config: RTCConfig):
        self.rtc_config = rtc_config
        self.tracker = None
        if rtc_config.debug:
            self.tracker = Tracker(
                enabled=rtc_config.debug,
                maxlen=rtc_config.debug_maxlen,
            )

    def track(
        self,
        time: float | Tensor,
        x_t: Tensor | None = None,
        v_t: Tensor | None = None,
        x1_t: Tensor | None = None,
        correction: Tensor | None = None,
        err: Tensor | None = None,
        weights: Tensor | None = None,
        guidance_weight: float | Tensor | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
        **metadata,
    ) -> None:
        if self.tracker is not None:
            self.tracker.track(
                time=time,
                x_t=x_t,
                v_t=v_t,
                x1_t=x1_t,
                correction=correction,
                err=err,
                weights=weights,
                guidance_weight=guidance_weight,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
                **metadata,
            )

    def get_all_debug_steps(self) -> list:
        if self.tracker is not None:
            return self.tracker.get_all_steps()
        return []

    def is_debug_enabled(self) -> bool:
        return self.tracker is not None and self.tracker.enabled

    def reset_tracker(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()

    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        original_denoise_step_partial,
        execution_horizon=None,
    ) -> Tensor:
        """RTC guidance wrapper around an existing denoiser.

        robotfm uses t: 0→1 (noise→data), so:
          tau = time
          x1_t = x_t + (1 - time) * v_t
        which is equivalent to LeRobot's inversion of their t: 1→0 schedule.
        """
        # robotfm / PI original: time goes from 0 (noise) to 1 (data)
        tau = time

        if prev_chunk_left_over is None:
            v_t = original_denoise_step_partial(x_t)
            return v_t

        x_t = x_t.clone().detach()

        squeezed = False
        if len(x_t.shape) < 3:
            x_t = x_t.unsqueeze(0)
            squeezed = True

        if len(prev_chunk_left_over.shape) < 3:
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)

        if execution_horizon is None:
            execution_horizon = self.rtc_config.execution_horizon

        if execution_horizon > prev_chunk_left_over.shape[1]:
            execution_horizon = prev_chunk_left_over.shape[1]

        batch_size = x_t.shape[0]
        action_chunk_size = x_t.shape[1]
        action_dim = x_t.shape[2]

        if prev_chunk_left_over.shape[1] < action_chunk_size or prev_chunk_left_over.shape[2] < action_dim:
            padded = torch.zeros(batch_size, action_chunk_size, action_dim, device=x_t.device, dtype=x_t.dtype)
            padded[:, : prev_chunk_left_over.shape[1], : prev_chunk_left_over.shape[2]] = prev_chunk_left_over
            prev_chunk_left_over = padded

        assert prev_chunk_left_over.shape == x_t.shape, (
            "The padded previous chunk must be the same size as the input tensor"
        )

        weights = (
            self.get_prefix_weights(inference_delay, execution_horizon, action_chunk_size)
            .to(device=x_t.device, dtype=x_t.dtype)
            .unsqueeze(0)
            .unsqueeze(-1)
        )

        with torch.enable_grad():
            v_t = original_denoise_step_partial(x_t)
            x_t.requires_grad_(True)

            # Clean-action estimate under OT-CFM with t: 0→1, v ≈ x1 - x0
            x1_t = x_t + (1.0 - time) * v_t  # noqa: N806
            err = (prev_chunk_left_over - x1_t) * weights
            grad_outputs = err.clone().detach()
            correction = torch.autograd.grad(x1_t, x_t, grad_outputs, retain_graph=False)[0]

        max_guidance_weight = torch.as_tensor(self.rtc_config.max_guidance_weight, device=x_t.device, dtype=x_t.dtype)
        tau_tensor = torch.as_tensor(tau, device=x_t.device, dtype=x_t.dtype)
        squared_one_minus_tau = (1 - tau_tensor) ** 2
        inv_r2 = (squared_one_minus_tau + tau_tensor**2) / (squared_one_minus_tau)
        c = torch.nan_to_num((1 - tau_tensor) / tau_tensor, posinf=max_guidance_weight)
        guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
        guidance_weight = torch.minimum(guidance_weight, max_guidance_weight)

        # LeRobot integrates with dt < 0 (t: 1→0) and uses ``v - gw * correction``.
        # robotfm integrates with dt > 0 (t: 0→1), so the guidance term sign flips.
        result = v_t + guidance_weight * correction

        if squeezed:
            result = result.squeeze(0)
            correction = correction.squeeze(0)
            x1_t = x1_t.squeeze(0)
            err = err.squeeze(0)

        self.track(
            time=time,
            x1_t=x1_t,
            correction=correction,
            err=err,
            weights=weights,
            guidance_weight=guidance_weight,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )

        return result

    def get_prefix_weights(self, start, end, total):
        start = min(start, end)
        schedule = self.rtc_config.prefix_attention_schedule

        if schedule == RTCAttentionSchedule.ZEROS:
            weights = torch.zeros(total)
            weights[:start] = 1.0
        elif schedule == RTCAttentionSchedule.ONES:
            weights = torch.ones(total)
            weights[end:] = 0.0
        elif schedule == RTCAttentionSchedule.LINEAR:
            lin_weights = self._linweights(start, end, total)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)
        elif schedule == RTCAttentionSchedule.EXP:
            lin_weights = self._linweights(start, end, total)
            lin_weights = lin_weights * torch.expm1(lin_weights).div(math.e - 1)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)
        else:
            raise ValueError(f"Unknown prefix_attention_schedule: {schedule}")

        return weights

    def _linweights(self, start, end, total):
        skip_steps_at_end = max(total - end, 0)
        linspace_steps = total - skip_steps_at_end - start

        if end <= start or linspace_steps <= 0:
            return torch.tensor([])

        return torch.linspace(1, 0, linspace_steps + 2)[1:-1]

    def _add_trailing_zeros(self, weights, total, end):
        zeros_len = total - end
        if zeros_len <= 0:
            return weights
        zeros = torch.zeros(zeros_len)
        return torch.cat([weights, zeros])

    def _add_leading_ones(self, weights, start, total):
        ones_len = min(start, total)
        if ones_len <= 0:
            return weights
        ones = torch.ones(ones_len)
        return torch.cat([ones, weights])
