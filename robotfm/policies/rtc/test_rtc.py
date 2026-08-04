"""Unit tests for robotfm RTC guidance and ActionQueue semantics."""

from __future__ import annotations

import torch

from robotfm.policies.rtc import ActionQueue, RTCAttentionSchedule, RTCConfig, RTCProcessor


def _euler_integrate(denoise_fn, noise, num_steps, *, rtc_processor=None, rtc_enabled=False, **rtc_kw):
    """Minimal Euler loop mirroring FlowMatchingPolicy.sample_actions (t: 0→1)."""
    x_t = noise
    dt = 1.0 / num_steps
    for step in range(num_steps):
        time = step / num_steps

        def denoise_step_partial(input_x_t, current_time=time):
            return denoise_fn(input_x_t, current_time)

        if rtc_enabled:
            v_t = rtc_processor.denoise_step(
                x_t=x_t,
                time=time,
                original_denoise_step_partial=denoise_step_partial,
                **rtc_kw,
            )
        else:
            v_t = denoise_step_partial(x_t)
        x_t = x_t + dt * v_t
    return x_t


def test_rtc_first_chunk_without_leftover_matches_unguided():
    processor = RTCProcessor(RTCConfig(execution_horizon=4, enabled=True))
    noise = torch.randn(2, 4, 2)

    def denoise_fn(x_t, time):
        return -0.5 * x_t

    torch.manual_seed(0)
    unguided = _euler_integrate(denoise_fn, noise.clone(), num_steps=4)

    torch.manual_seed(0)
    first_chunk = _euler_integrate(
        denoise_fn,
        noise.clone(),
        num_steps=4,
        rtc_processor=processor,
        rtc_enabled=True,
        prev_chunk_left_over=None,
        inference_delay=2,
        execution_horizon=4,
    )

    assert torch.allclose(unguided, first_chunk)


def test_rtc_guidance_pulls_prefix_toward_previous_chunk():
    processor = RTCProcessor(
        RTCConfig(
            execution_horizon=4,
            max_guidance_weight=10.0,
            prefix_attention_schedule=RTCAttentionSchedule.ONES,
            enabled=True,
        )
    )
    # Zero velocity: unguided sample stays at init noise; RTC should still pull prefix to leftover.
    def denoise_fn(x_t, time):
        return torch.zeros_like(x_t)

    noise = torch.randn(2, 4, 2)
    prev_chunk = torch.full((2, 4, 2), 1.5)

    unguided = _euler_integrate(denoise_fn, noise.clone(), num_steps=8)
    guided = _euler_integrate(
        denoise_fn,
        noise.clone(),
        num_steps=8,
        rtc_processor=processor,
        rtc_enabled=True,
        prev_chunk_left_over=prev_chunk,
        inference_delay=2,
        execution_horizon=4,
    )

    guided_dist = (guided[:, :2] - prev_chunk[:, :2]).abs().mean()
    unguided_dist = (unguided[:, :2] - prev_chunk[:, :2]).abs().mean()
    assert guided_dist < 0.5 * unguided_dist
    assert torch.isfinite(guided).all()


def test_fm_closed_form_correction_matches_identity_autograd():
    """Frozen-v path: autograd through x1=x+(1-t)*v equals correction=err."""
    processor = RTCProcessor(
        RTCConfig(
            execution_horizon=4,
            max_guidance_weight=10.0,
            prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
            enabled=True,
        )
    )
    torch.manual_seed(0)
    x_t = torch.randn(2, 4, 2)
    leftover = torch.randn(2, 4, 2)
    time = 0.25
    inference_delay, execution_horizon = 2, 4

    def denoise_fn(x):
        return -0.3 * x + 0.1

    # Reference: previous identity-autograd formulation.
    x_ref = x_t.clone().detach()
    weights = (
        processor.get_prefix_weights(inference_delay, execution_horizon, x_ref.shape[1])
        .to(dtype=x_ref.dtype)
        .unsqueeze(0)
        .unsqueeze(-1)
    )
    with torch.enable_grad():
        v_ref = denoise_fn(x_ref)
        x_leaf = x_ref.detach().requires_grad_(True)
        x1_ref = x_leaf + (1.0 - time) * v_ref
        err_ref = (leftover - x1_ref) * weights
        corr_ref = torch.autograd.grad(x1_ref, x_leaf, err_ref.clone().detach())[0]

    # Current closed-form path via denoise_step.
    v_new = processor.denoise_step(
        x_t=x_t.clone(),
        prev_chunk_left_over=leftover,
        inference_delay=inference_delay,
        time=time,
        original_denoise_step_partial=denoise_fn,
        execution_horizon=execution_horizon,
    )
    gw = processor._guidance_weight(
        time, processor.rtc_config.max_guidance_weight, x_t.device, x_t.dtype
    )
    # Reconstruct closed-form result: v + gw * err with same v/x1 as frozen-v.
    x_cf = x_t.clone().detach()
    v_cf = denoise_fn(x_cf)
    x1_cf = x_cf + (1.0 - time) * v_cf
    err_cf = (leftover - x1_cf) * weights
    expected = v_cf + gw * err_cf

    assert torch.allclose(corr_ref, err_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(v_new, expected, atol=1e-6, rtol=1e-5)


def test_action_queue_merge_skips_inference_delay():
    cfg = RTCConfig(enabled=True, execution_horizon=4, inference_delay=2)
    queue = ActionQueue(cfg)

    original = torch.arange(8, dtype=torch.float32).view(8, 1)
    processed = original.clone()
    queue.merge(original, processed, real_delay=2)

    assert queue.qsize() == 6
    assert torch.allclose(queue.get(), torch.tensor([2.0]))
    leftover = queue.get_left_over()
    assert leftover is not None
    assert leftover.shape[0] == 5
    assert torch.allclose(leftover[0], torch.tensor([3.0]))


def test_guidance_disabled_matches_unguided_even_with_leftover():
    """guidance_enabled=False → leftover ignored; same as unguided Euler."""
    processor = RTCProcessor(
        RTCConfig(
            execution_horizon=4,
            max_guidance_weight=10.0,
            prefix_attention_schedule=RTCAttentionSchedule.ONES,
            enabled=True,
            guidance_enabled=False,
        )
    )
    noise = torch.randn(2, 4, 2)
    leftover = torch.full((2, 4, 2), 3.0)

    def denoise_fn(x_t, time):
        return -0.5 * x_t

    torch.manual_seed(0)
    unguided = _euler_integrate(denoise_fn, noise.clone(), num_steps=4)

    torch.manual_seed(0)
    no_guide = _euler_integrate(
        denoise_fn,
        noise.clone(),
        num_steps=4,
        rtc_processor=processor,
        rtc_enabled=True,
        prev_chunk_left_over=leftover,
        inference_delay=2,
        execution_horizon=4,
    )
    assert torch.allclose(unguided, no_guide)


def test_prefix_attention_schedules_shapes():
    for schedule in RTCAttentionSchedule:
        processor = RTCProcessor(
            RTCConfig(enabled=True, execution_horizon=6, prefix_attention_schedule=schedule)
        )
        weights = processor.get_prefix_weights(2, 6, 8)
        assert weights.shape == (8,)
        assert torch.allclose(weights[:2], torch.ones(2))


def _linear_decode_x1(latent_dim: int, horizon: int, action_dim: int):
    """Differentiable fake decoder: reshape/project latent → (B, H, A)."""
    out_dim = horizon * action_dim
    # Well-conditioned map so RTC correction reaches action space strongly.
    weight = torch.eye(out_dim, latent_dim) if latent_dim >= out_dim else torch.randn(out_dim, latent_dim)
    if latent_dim < out_dim:
        weight = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
    bias = torch.zeros(out_dim)

    def decode(z: torch.Tensor) -> torch.Tensor:
        flat = torch.nn.functional.linear(z, weight.to(z.device, z.dtype), bias.to(z.device, z.dtype))
        return flat.view(z.shape[0], horizon, action_dim)

    return decode


def test_rtc_decode_x1_without_leftover_matches_unguided():
    processor = RTCProcessor(RTCConfig(execution_horizon=4, enabled=True))
    latent_dim, horizon, action_dim = 8, 4, 2
    decode = _linear_decode_x1(latent_dim, horizon, action_dim)
    noise = torch.randn(2, latent_dim)

    def denoise_fn(x_t, time):
        return -0.5 * x_t

    torch.manual_seed(0)
    unguided = _euler_integrate(denoise_fn, noise.clone(), num_steps=4)

    torch.manual_seed(0)
    first_chunk = _euler_integrate(
        denoise_fn,
        noise.clone(),
        num_steps=4,
        rtc_processor=processor,
        rtc_enabled=True,
        prev_chunk_left_over=None,
        inference_delay=2,
        execution_horizon=4,
        decode_x1=decode,
    )

    assert torch.allclose(unguided, first_chunk)


def test_rtc_decode_x1_guidance_pulls_decoded_prefix():
    processor = RTCProcessor(
        RTCConfig(
            execution_horizon=4,
            max_guidance_weight=10.0,
            prefix_attention_schedule=RTCAttentionSchedule.ONES,
            enabled=True,
        )
    )
    latent_dim, horizon, action_dim = 8, 4, 2
    decode = _linear_decode_x1(latent_dim, horizon, action_dim)

    def denoise_fn(x_t, time):
        return torch.zeros_like(x_t)

    noise = torch.randn(2, latent_dim)
    prev_chunk = torch.full((2, horizon, action_dim), 1.5)

    unguided_lat = _euler_integrate(denoise_fn, noise.clone(), num_steps=8)
    guided_lat = _euler_integrate(
        denoise_fn,
        noise.clone(),
        num_steps=8,
        rtc_processor=processor,
        rtc_enabled=True,
        prev_chunk_left_over=prev_chunk,
        inference_delay=2,
        execution_horizon=4,
        decode_x1=decode,
    )

    guided_actions = decode(guided_lat)
    unguided_actions = decode(unguided_lat)
    guided_dist = (guided_actions[:, :2] - prev_chunk[:, :2]).abs().mean()
    unguided_dist = (unguided_actions[:, :2] - prev_chunk[:, :2]).abs().mean()
    assert guided_dist < 0.5 * unguided_dist
    assert torch.isfinite(guided_lat).all()
    assert torch.isfinite(guided_actions).all()
