"""Conditional 1D UNet with FiLM — Diffusion Policy / RecFlow 风格动作骨干。

对 action chunk ``(B, horizon, action_dim)`` 预测 Flow Matching 速度场。
观测条件与时间经 FiLM 注入每个 residual block。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """连续时间 / diffusion step 的正弦位置编码。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish。"""

    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(min(n_groups, out_channels), out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """带 FiLM 条件调制的 1D residual block。

    FiLM: ``out = scale * h + bias``，其中 scale/bias 由全局条件预测。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.cond_predict_scale = cond_predict_scale
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        cond: (B, cond_dim)
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed.unsqueeze(-1)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """条件 1D UNet：预测 action chunk 上的速度场。

    输入约定（与 ActionDiT 对齐，便于直接替换）：
        sample: (B, horizon, action_dim)
        timestep: (B,)
        global_cond: (B, global_cond_dim)

    输出:
        (B, horizon, action_dim)
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: list[int] | tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ) -> None:
        super().__init__()
        all_dims = [input_dim, *list(down_dims)]
        start_dim = down_dims[0]

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    down_dims[-1],
                    down_dims[-1],
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    down_dims[-1],
                    down_dims[-1],
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
            ]
        )

        down_modules: list[nn.ModuleList] = []
        for ind, (dim_in, dim_out) in enumerate[tuple[int, int]](in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.down_modules = nn.ModuleList(down_modules)

        up_modules: list[nn.ModuleList] = []
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.up_modules = nn.ModuleList(up_modules)

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """前向：对噪声动作序列预测速度场（或噪声残差）。

        参数:
            sample: (B, T, input_dim)
                当前时刻的 action chunk（Flow Matching 中的 x_t，或 diffusion 中的
                噪声动作）。T = horizon，input_dim = action_dim。
            timestep: (B,)
                连续/离散时间步 t，经正弦编码后注入各 residual block。
            global_cond: (B, global_cond_dim)
                观测条件（如图像/状态编码器输出），与时间嵌入拼接成 FiLM 条件。

        返回:
            (B, T, input_dim) — 与 sample 同形状的速度场预测。
        """
        # ------------------------------------------------------------------
        # 1) 布局转换：外部约定 (B, T, C)，Conv1d 需要 (B, C, T)
        #    T 是时间/horizon 轴，C 是动作维度；UNet 在 C 上做通道、在 T 上做卷积。
        # ------------------------------------------------------------------
        x = sample.transpose(1, 2)

        # ------------------------------------------------------------------
        # 2) 构造全局条件向量 cond
        #    - t_emb: 正弦位置编码 + MLP，把标量 t 映射到 diffusion_step_embed_dim
        #    - 与观测条件拼接 → (B, diffusion_step_embed_dim + global_cond_dim)
        #    后续每个 ConditionalResidualBlock1D 用 FiLM（scale/bias）调制特征。
        # ------------------------------------------------------------------
        t_emb = self.diffusion_step_encoder(timestep)
        cond = torch.cat([t_emb, global_cond], dim=-1)

        # ------------------------------------------------------------------
        # 3) Encoder（下采样路径）
        #    每级：两个条件 residual block → 缓存特征到 h（供 skip connection）
        #          → Downsample1d 将时间分辨率减半（最后一级为 Identity）。
        #    通道数沿 down_dims 递增（如 256→512→1024），感受野变大、语义更抽象。
        # ------------------------------------------------------------------
        h: list[torch.Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, cond)
            x = resnet2(x, cond)
            h.append(x)          # 下采样前的特征，解码时按相反顺序弹出
            x = downsample(x)

        # ------------------------------------------------------------------
        # 4) Bottleneck（中间层）
        #    在最低时间分辨率、最高通道数上再过两个 residual block，
        #    充分融合时间与观测条件后再开始上采样重建。
        # ------------------------------------------------------------------
        for mid in self.mid_modules:
            x = mid(x, cond)

        # ------------------------------------------------------------------
        # 5) Decoder（上采样路径）
        #    每级：与对应 encoder 特征在通道维 concat（U-Net skip）
        #          → 两个 residual block（输入通道为 skip 拼接后的 2×）
        #          → Upsample1d 将时间分辨率加倍（最后一级为 Identity）。
        #    h.pop() 保证「最深 encoder 特征最先与 bottleneck 拼接」。
        # ------------------------------------------------------------------
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat([x, h.pop()], dim=1)  # dim=1 即通道维
            x = resnet(x, cond)
            x = resnet2(x, cond)
            x = upsample(x)

        # ------------------------------------------------------------------
        # 6) 输出头 + 布局还原
        #    final_conv: Conv1dBlock 精炼特征 → 1×1 Conv 投影回 action_dim。
        #    transpose 回 (B, T, C)，与 ActionDiT 等骨干接口对齐。
        # ------------------------------------------------------------------
        x = self.final_conv(x)
        return x.transpose(1, 2)
