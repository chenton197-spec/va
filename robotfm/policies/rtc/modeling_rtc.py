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
from collections.abc import Callable

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
        # Cache prefix weights / A2A decoder output shape across Euler steps.
        self._prefix_weight_cache: dict[tuple, Tensor] = {}
        self._latent_action_hw: tuple[int, int] | None = None

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

    @staticmethod
    def _guidance_weight(tau: float | Tensor, max_guidance_weight: float, device, dtype) -> Tensor:
        max_gw = torch.as_tensor(max_guidance_weight, device=device, dtype=dtype)
        tau_tensor = torch.as_tensor(tau, device=device, dtype=dtype)
        squared_one_minus_tau = (1 - tau_tensor) ** 2
        inv_r2 = (squared_one_minus_tau + tau_tensor**2) / (squared_one_minus_tau)
        c = torch.nan_to_num((1 - tau_tensor) / tau_tensor, posinf=max_gw)
        guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_gw)
        return torch.minimum(guidance_weight, max_gw)

    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        original_denoise_step_partial,
        execution_horizon=None,
        decode_x1: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        """RTC guidance wrapper around an existing denoiser.

        robotfm uses t: 0→1 (noise→data), so:
          tau = time
          x1_t = x_t + (1 - time) * v_t
        which is equivalent to LeRobot's inversion of their t: 1→0 schedule.

        Args:
            decode_x1: Optional map from clean latent estimate ``x1_latent`` to
                action-space ``(B, H, A)``. When set, ``x_t`` is latent ``(B, D)``
                and leftover / prefix weights live in action space (A2A path).
                When ``None``, ``x_t`` and leftover share action-space shape (FM).
        """
        # robotfm / PI original: time goes from 0 (noise) to 1 (data)
        tau = time

        if prev_chunk_left_over is None:
            v_t = original_denoise_step_partial(x_t)
            return v_t

        if decode_x1 is not None:
            return self._denoise_step_latent(
                x_t=x_t,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
                time=time,
                tau=tau,
                original_denoise_step_partial=original_denoise_step_partial,
                execution_horizon=execution_horizon,
                decode_x1=decode_x1,
            )

        return self._denoise_step_action(
            x_t=x_t,
            prev_chunk_left_over=prev_chunk_left_over,
            inference_delay=inference_delay,
            time=time,
            tau=tau,
            original_denoise_step_partial=original_denoise_step_partial,
            execution_horizon=execution_horizon,
        )

    def _denoise_step_action(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        tau,
        original_denoise_step_partial,
        execution_horizon,
    ) -> Tensor:
        """Action-space RTC (Flow Matching): x_t and leftover are (B, H, A).

        ``v_t`` is computed with ``x_t`` detached, so ``x1 = x_t + (1-t)*v_t`` has
        ``∂x1/∂x_t = I``. Correction equals ``err`` (identity VJP); skip autograd.
        """
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

        weights = self._prefix_weights_cached(
            inference_delay, execution_horizon, action_chunk_size, x_t.device, x_t.dtype
        )

        v_t = original_denoise_step_partial(x_t)
        # Clean-action estimate under OT-CFM with t: 0→1, v ≈ x1 - x0
        x1_t = x_t + (1.0 - time) * v_t  # noqa: N806
        err = (prev_chunk_left_over - x1_t) * weights
        # Identity VJP: v_t is not connected to leaf x_t, so grad(x1, x_t, err) == err.
        correction = err

        guidance_weight = self._guidance_weight(
            tau, self.rtc_config.max_guidance_weight, x_t.device, x_t.dtype
        )

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

    def _denoise_step_latent(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        tau,
        original_denoise_step_partial,
        execution_horizon,
        decode_x1: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """Latent-space RTC (A2A): guide via decode_x1(x1_latent) vs action leftover."""
        x_t = x_t.clone().detach()
        if x_t.ndim != 2:
            raise ValueError(f"decode_x1 path expects x_t shape (B, D), got {tuple(x_t.shape)}")

        if prev_chunk_left_over.ndim == 2:
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)
        elif prev_chunk_left_over.ndim != 3:
            raise ValueError(
                f"leftover expects (H, A) or (B, H, A), got {tuple(prev_chunk_left_over.shape)}"
            )

        batch_size = x_t.shape[0]
        if prev_chunk_left_over.shape[0] == 1 and batch_size > 1:
            prev_chunk_left_over = prev_chunk_left_over.expand(batch_size, -1, -1)
        elif prev_chunk_left_over.shape[0] != batch_size:
            raise ValueError(
                f"leftover batch {prev_chunk_left_over.shape[0]} != x_t batch {batch_size}"
            )

        action_chunk_size, action_dim = self._resolve_latent_action_shape(x_t, decode_x1)

        if execution_horizon is None:
            execution_horizon = self.rtc_config.execution_horizon
        if execution_horizon > prev_chunk_left_over.shape[1]:
            execution_horizon = prev_chunk_left_over.shape[1]
        # Cap soft-blend region to decoder horizon as well.
        if execution_horizon > action_chunk_size:
            execution_horizon = action_chunk_size

        if (
            prev_chunk_left_over.shape[1] != action_chunk_size
            or prev_chunk_left_over.shape[2] != action_dim
        ):
            padded = torch.zeros(
                batch_size,
                action_chunk_size,
                action_dim,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            h = min(prev_chunk_left_over.shape[1], action_chunk_size)
            a = min(prev_chunk_left_over.shape[2], action_dim)
            padded[:, :h, :a] = prev_chunk_left_over[:, :h, :a]
            prev_chunk_left_over = padded

        weights = self._prefix_weights_cached(
            inference_delay, execution_horizon, action_chunk_size, x_t.device, x_t.dtype
        )

        with torch.enable_grad():
            v_t = original_denoise_step_partial(x_t).detach()
            x_t = x_t.detach().requires_grad_(True)

            x1_latent = x_t + (1.0 - time) * v_t
            x1_actions = decode_x1(x1_latent)
            err = (prev_chunk_left_over - x1_actions) * weights
            grad_outputs = err.clone().detach()
            correction = torch.autograd.grad(x1_actions, x_t, grad_outputs, retain_graph=False)[0]

        guidance_weight = self._guidance_weight(
            tau, self.rtc_config.max_guidance_weight, x_t.device, x_t.dtype
        )
        result = v_t + guidance_weight * correction

        self.track(
            time=time,
            x1_t=x1_actions,
            correction=correction,
            err=err,
            weights=weights,
            guidance_weight=guidance_weight,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )

        return result

    def _resolve_latent_action_shape(
        self,
        x_t: Tensor,
        decode_x1: Callable[[Tensor], Tensor],
    ) -> tuple[int, int]:
        """Return (H, A) for A2A decoder; probe at most once per processor."""
        if self._latent_action_hw is not None:
            return self._latent_action_hw
        with torch.no_grad():
            probe = decode_x1(x_t)
        if probe.ndim != 3:
            raise ValueError(f"decode_x1 must return (B, H, A), got {tuple(probe.shape)}")
        self._latent_action_hw = (int(probe.shape[1]), int(probe.shape[2]))
        return self._latent_action_hw

    def _prefix_weights_cached(
        self,
        start: int,
        end: int,
        total: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Prefix weights shaped (1, H, 1), cached across Euler steps."""
        key = (
            int(start),
            int(end),
            int(total),
            str(self.rtc_config.prefix_attention_schedule),
            device.type,
            device.index,
            str(dtype),
        )
        cached = self._prefix_weight_cache.get(key)
        if cached is None:
            cached = (
                self.get_prefix_weights(start, end, total)
                .to(device=device, dtype=dtype)
                .unsqueeze(0)
                .unsqueeze(-1)
            )
            self._prefix_weight_cache[key] = cached
        return cached

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
