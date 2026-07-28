from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """把连续时间 `t` 编码成正弦位置向量。

    Flow Matching 中，网络不仅要看当前的噪声动作 `x_t`，
    还要知道“现在在从噪声走向真实动作的哪一个时刻”。
    这里采用和 Transformer 位置编码类似的正余弦编码方式。

    输入:
        t: (B,)
    输出:
        emb: (B, dim)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        # 构造一组从低频到高频的频率，用于表达不同时间尺度。
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class DiTBlock(nn.Module):
    """最小化的 DiT block。

    结构:
        LayerNorm -> Self-Attention -> Residual
        LayerNorm -> FFN           -> Residual

    这里没有加入更复杂的 adaLN / cross-attention / gating，
    因为当前项目先追求结构清晰与可维护。
    """

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先做自注意力，让不同动作 token（未来不同时间步）之间交互。
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        # 再做前馈网络，增强逐 token 的非线性表达能力。
        x = x + self.ff(self.norm2(x))
        return x


class ActionDiT(nn.Module):
    """动作生成器：条件化的 DiT。

    作用：
    - 输入当前时刻的 noised action trajectory `x_t`
    - 输入时间 `t`
    - 输入从图像+状态编码得到的条件向量 `cond`
    - 输出每个动作 token 对应的速度场预测 `v_theta`

    张量约定：
        x_t:  (B, horizon, action_dim)
        t:    (B,)
        cond: (B, cond_dim)
        out:  (B, horizon, action_dim)
    """

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        cond_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim

        # 把每一个动作向量映射到 Transformer hidden space。
        self.in_proj = nn.Linear(action_dim, hidden_dim)

        # 时间编码：连续时间 t -> hidden_dim，再经过一个小 MLP 提升表达能力。
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 条件向量投影到和 token 相同的 hidden space。
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # 可学习的位置编码，用来区分动作 chunk 内的不同步。
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, hidden_dim))
        nn.init.normal_(self.pos_emb, std=0.02)

        # 多层 DiT block 负责建模动作序列内部的时序相关性。
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads) for _ in range(num_layers)])

        # 输出回每个时间步的动作速度维度。
        self.out_proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """前向传播。

        整体思路：
        1. 把动作 token 投影到 hidden space；
        2. 加上动作位置编码；
        3. 把时间编码和条件编码广播到整个动作序列；
        4. 经过多层 DiT block；
        5. 输出与动作同形状的速度场预测。
        """

        h = self.in_proj(x_t) + self.pos_emb[:, : x_t.shape[1]]
        t_emb = self.time_mlp(self.time_emb(t)).unsqueeze(1)
        c_emb = self.cond_proj(cond).unsqueeze(1)

        # 这里使用“直接相加”的条件注入方式。
        # 后续如果模型规模变大，可替换成 adaLN / FiLM / cross-attention。
        h = h + t_emb + c_emb
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h)
