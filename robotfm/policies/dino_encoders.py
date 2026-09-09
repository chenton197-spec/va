from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

_DINO_IMAGE_SIZE = 224


class DinoV2Encoder(nn.Module):
    def __init__(
        self,
        name: str = "dinov2_vits14",
        freeze: bool = True,
        image_size: int = _DINO_IMAGE_SIZE,
    ) -> None:
        super().__init__()
        self.name = name
        self.image_size = int(image_size)
        self.freeze = bool(freeze)
        self.base_model = torch.hub.load("facebookresearch/dinov2:b48308a", name)
        self.emb_dim = int(self.base_model.num_features)
        self.patch_size = int(self.base_model.patch_size)
        if self.freeze:
            for p in self.base_model.parameters():
                p.requires_grad = False
        self.register_buffer(
            "img_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "img_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.base_model.eval()
        return self

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.freeze:
            with torch.no_grad():
                return self._forward_features(obs)
        return self._forward_features(obs)

    def _forward_features(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = obs[:, -1, :3]
        if x.shape[-2] != self.image_size or x.shape[-1] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        x = (x - self.img_mean) / self.img_std
        feats = self.base_model.forward_features(x)
        cls = feats["x_norm_clstoken"]
        patches = feats["x_norm_patchtokens"]
        b, n, e = patches.shape
        gh = self.image_size // self.patch_size
        gw = n // gh
        patch_map = patches.transpose(1, 2).reshape(b, e, gh, gw)
        return cls, patch_map
