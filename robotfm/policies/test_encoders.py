"""MultiCameraEncoder: shared eval batching + separate-weight smoke tests."""

from __future__ import annotations

import torch

from robotfm.policies.encoders import MultiCameraEncoder, ResNet18Encoder


def _serial_img_feat_shared(enc: MultiCameraEncoder, obs_images: torch.Tensor) -> torch.Tensor:
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
        share_image_encoder=True,
    )
    enc.eval()

    obs_images = torch.rand(2, 2, 2, 3, 64, 64)
    obs_state = torch.randn(2, 2, 7)

    with torch.no_grad():
        out_batched = enc(obs_images, obs_state)

        # Reconstruct serial path (same as training branch) under eval BN.
        img_feat = _serial_img_feat_shared(enc, obs_images)
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
        share_image_encoder=True,
    )
    enc.train()
    obs_images = torch.rand(4, 2, 2, 3, 64, 64)
    obs_state = torch.randn(4, 2, 7)
    out = enc(obs_images, obs_state)
    assert out.shape == (4, 256)


def test_separate_encoders_have_independent_weights():
    enc = MultiCameraEncoder(
        num_cameras=2,
        state_dim=7,
        n_obs_steps=2,
        pretrained_encoder=False,
        share_image_encoder=False,
    )
    assert not enc.share_image_encoder
    assert len(enc.image_encoders) == 2
    assert not hasattr(enc, "image_encoder")

    p0 = next(enc.image_encoders[0].parameters())
    p1 = next(enc.image_encoders[1].parameters())
    assert p0 is not p1
    assert id(p0) != id(p1)

    # Mutating cam0 must not change cam1.
    with torch.no_grad():
        p0.add_(1.0)
    assert not torch.equal(p0, p1)

    enc.train()
    out = enc(torch.rand(2, 2, 2, 3, 64, 64), torch.randn(2, 2, 7))
    assert out.shape == (2, 256)

    enc.eval()
    out_e = enc(torch.rand(2, 2, 2, 3, 64, 64), torch.randn(2, 2, 7))
    assert out_e.shape == (2, 256)

    n_vision = sum(p.numel() for p in enc.vision_parameters())
    shared = MultiCameraEncoder(
        num_cameras=2,
        state_dim=7,
        n_obs_steps=2,
        pretrained_encoder=False,
        share_image_encoder=True,
    )
    n_shared = sum(p.numel() for p in shared.vision_parameters())
    assert n_vision == 2 * n_shared


def test_coord_conv_off_keeps_stem_channels():
    t = 2
    enc = ResNet18Encoder(
        n_obs_steps=t,
        pretrained=False,
        use_frame_diff=False,
        use_coord_conv=False,
    )
    assert enc.backbone.conv1.in_channels == 3 * t
    out = enc(torch.rand(2, t, 3, 64, 64))
    assert out.shape == (2, 128)


def test_coord_conv_on_adds_xy_channels():
    t = 2
    enc = ResNet18Encoder(
        n_obs_steps=t,
        pretrained=False,
        use_frame_diff=False,
        use_coord_conv=True,
    )
    assert enc.backbone.conv1.in_channels == 3 * t + 2
    # RGB slice filled from stem adapt; last two (xy) stay zero-init.
    assert torch.count_nonzero(enc.backbone.conv1.weight[:, -2:]).item() == 0
    out = enc(torch.rand(2, t, 3, 64, 64))
    assert out.shape == (2, 128)

    multi = MultiCameraEncoder(
        num_cameras=2,
        state_dim=7,
        n_obs_steps=t,
        pretrained_encoder=False,
        use_coord_conv=True,
        share_image_encoder=True,
    )
    assert multi.image_encoder.backbone.conv1.in_channels == 3 * t + 2
    out_m = multi(torch.rand(2, 2, t, 3, 32, 32), torch.randn(2, t, 7))
    assert out_m.shape == (2, 256)


if __name__ == "__main__":
    test_eval_batched_cameras_match_serial()
    test_train_still_uses_per_camera_forward()
    test_separate_encoders_have_independent_weights()
    test_coord_conv_off_keeps_stem_channels()
    test_coord_conv_on_adds_xy_channels()
    print("ok")
