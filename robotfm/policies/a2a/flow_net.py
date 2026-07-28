"""SimpleFlowNet ported from A2A_Flow_Matching (VITA-style AdaLN MLP)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from robotfm.policies.a2a.layers import Mlp


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FlowNetLayer(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=approx_gelu,
            drop=dropout,
        )

        self.time_modulator = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        self.dim = dim

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.time_modulator[-1].weight, 0)
        nn.init.constant_(self.time_modulator[-1].bias, 0)

    def forward(self, x, t):
        B = x.shape[0]
        features = self.time_modulator(t).view(B, 3, self.dim).unbind(1)
        gamma, scale, shift = features

        x_norm = self.norm(x)
        x_norm = x_norm.mul(scale.add(1)).add_(shift)
        x = x + self.mlp(x_norm).mul_(gamma)

        return x


class SimpleFlowNet(nn.Module):
    """A simple flow network with only MLP layers (Gao et al. 2025 / A2A)."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        mlp_ratio=4.0,
        dropout=0.0,
        time_embed_dim=256,
        condition_dim=None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, hidden_dim),
        )

        if condition_dim is not None:
            self.cond_embed = nn.Linear(condition_dim, hidden_dim)
        else:
            self.cond_embed = None

        self.layers = nn.ModuleList(
            [
                FlowNetLayer(dim=hidden_dim, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.time_embed[1].weight, std=0.02)
        nn.init.normal_(self.time_embed[3].weight, std=0.02)

    def forward(self, x, t, global_cond=None):
        x = self.input_proj(x)
        t = self.time_embed(t)

        if global_cond is not None and self.cond_embed is not None:
            c = self.cond_embed(global_cond)
            t = t + c

        for block in self.layers:
            x = block(x, t)

        x = self.norm(x)
        x = self.out_proj(x)
        return x
