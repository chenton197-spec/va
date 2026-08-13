"""MultiCameraViTEncoder: shared / separate smoke tests (no pretrained download)."""

from __future__ import annotations

import torch

from robotfm.policies.encoders import build_multi_camera_encoder
from robotfm.policies.vit_encoders import MultiCameraViTEncoder


def test_vit_shared_train_eval_shapes():
    enc = MultiCameraViTEncoder(
        num_cameras=2,
        state_dim=7,
        n_obs_steps=2,
        pretrained_encoder=False,
        use_frame_diff=True,
        share_image_encoder=True,
    )
    obs_images = torch.rand(2, 2, 2, 3, 64, 64)
    obs_state = torch.randn(2, 2, 7)

    enc.train()
    out_tr = enc(obs_images, obs_state)
    assert out_tr.shape == (2, 256)

    enc.eval()
    with torch.no_grad():
        out_ev = enc(obs_images, obs_state)
    assert out_ev.shape == (2, 256)


def test_vit_separate_encoders():
    enc = MultiCameraViTEncoder(
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
    assert id(p0) != id(p1)

    with torch.no_grad():
        p0.add_(1.0)
    assert not torch.equal(p0, p1)

    out = enc(torch.rand(2, 2, 2, 3, 48, 48), torch.randn(2, 2, 7))
    assert out.shape == (2, 256)

    n_vision = sum(p.numel() for p in enc.vision_parameters())
    shared = MultiCameraViTEncoder(
        num_cameras=2,
        state_dim=7,
        n_obs_steps=2,
        pretrained_encoder=False,
        share_image_encoder=True,
    )
    n_shared = sum(p.numel() for p in shared.vision_parameters())
    assert n_vision == 2 * n_shared


def test_factory_vit_b_16():
    enc = build_multi_camera_encoder(
        "vit_b_16",
        num_cameras=1,
        state_dim=4,
        n_obs_steps=2,
        cond_dim=128,
        pretrained_encoder=False,
        share_image_encoder=True,
    )
    assert isinstance(enc, MultiCameraViTEncoder)
    out = enc(torch.rand(1, 1, 2, 3, 32, 32), torch.randn(1, 2, 4))
    assert out.shape == (1, 128)


if __name__ == "__main__":
    test_vit_shared_train_eval_shapes()
    test_vit_separate_encoders()
    test_factory_vit_b_16()
    print("ok")
