"""MultiCameraEncoder eval batched cameras vs serial reference."""

from __future__ import annotations

import torch

from robotfm.policies.encoders import MultiCameraEncoder


def _serial_img_feat(enc: MultiCameraEncoder, obs_images: torch.Tensor) -> torch.Tensor:
    b, cams, t, _, _, _ = obs_images.shape
    assert b >= 1 and cams >= 1
    cam_feats = [enc.image_encoder(obs_images[:, c]) for c in range(cams)]
    return torch.cat(cam_feats, dim=-1)


def test_eval_batched_cameras_match_serial():
    torch.manual_seed(0)
    enc = MultiCameraEncoder(
        num_cameras=2,
        state_dim=7,
        n_obs_steps=2,
        pretrained_encoder=False,
        use_frame_diff=True,
    )
    enc.eval()

    obs_images = torch.rand(2, 2, 2, 3, 64, 64)
    obs_state = torch.randn(2, 2, 7)

    with torch.no_grad():
        out_batched = enc(obs_images, obs_state)

        # Reconstruct serial path (same as training branch) under eval BN.
        img_feat = _serial_img_feat(enc, obs_images)
        b, _, t, _, _, _ = obs_images.shape
        state = obs_state.reshape(b * t, -1)
        state_feat = enc.state_encoder(state).reshape(b, -1)
        out_serial = enc.proj(torch.cat([img_feat, state_feat], dim=-1))

    max_diff = (out_batched - out_serial).abs().max().item()
    assert max_diff < 1e-5, f"batched vs serial max abs diff={max_diff}"


def test_train_still_uses_per_camera_forward():
    """Smoke: training mode still runs (BN batch = B per camera)."""
    enc = MultiCameraEncoder(
        num_cameras=2,
        state_dim=7,
        n_obs_steps=2,
        pretrained_encoder=False,
    )
    enc.train()
    obs_images = torch.rand(4, 2, 2, 3, 64, 64)
    obs_state = torch.randn(4, 2, 7)
    out = enc(obs_images, obs_state)
    assert out.shape == (4, 256)


if __name__ == "__main__":
    test_eval_batched_cameras_match_serial()
    test_train_still_uses_per_camera_forward()
    print("ok")
