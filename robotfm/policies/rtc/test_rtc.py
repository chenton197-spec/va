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


def test_prefix_attention_schedules_shapes():
    for schedule in RTCAttentionSchedule:
        processor = RTCProcessor(
            RTCConfig(enabled=True, execution_horizon=6, prefix_attention_schedule=schedule)
        )
        weights = processor.get_prefix_weights(2, 6, 8)
        assert weights.shape == (8,)
        assert torch.allclose(weights[:2], torch.ones(2))
