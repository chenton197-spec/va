"""PA2 YOLO 视觉骨干（对齐 ``casbotPA2-backbone.yaml`` 的图像部分）。

从 PA2 ``network_builder.nn.modules`` 抽出：Conv / C3k2 / C2PSA / SPPF。
原 YAML 中途的 SensorFusion 换成 1×1 Conv，骨干只编码图像。

scale ``n``: depth=0.75, width=0.5, max_channels=1024。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

_PA2_SCALES = {
    "n": (0.75, 0.5, 1024),
    "s": (0.75, 0.5, 1024),
    "m": (0.75, 1.0, 512),
    "l": (1.0, 1.0, 512),
    "x": (1.0, 1.5, 512),
}


def _autopad(k: int | tuple[int, ...], p: int | None = None, d: int = 1) -> int | list[int]:
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def _make_divisible(x: float, divisor: int = 8) -> int:
    return math.ceil(x / divisor) * divisor


def _scaled_channels(ch: int, width: float, max_channels: int) -> int:
    return _make_divisible(min(ch, max_channels) * width, 8)


def _scaled_depth(n: int, depth: float) -> int:
    return max(round(n * depth), 1) if n > 1 else n


class Conv(nn.Module):
    """Conv + BN + SiLU。"""

    default_act = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if act is True:
            self.act = self.default_act
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(x)


class PSABlock(nn.Module):
    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C3k2(C2f):
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C2PSA(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5) -> None:
        super().__init__()
        if c1 != c2:
            raise ValueError(f"C2PSA requires c1==c2, got {c1} vs {c2}")
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        heads = max(self.c // 64, 1)
        self.m = nn.Sequential(
            *(PSABlock(self.c, attn_ratio=0.5, num_heads=heads) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class SPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False) -> None:
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y


class PA2VisionBackbone(nn.Module):
    """casbotPA2-backbone.yaml 的图像 Sequential（空 head，无 SensorFusion）。

    输入: ``x (B, C, H, W)``
    输出: P5 特征图 ``(B, out_channels, H/32, W/32)``
    """

    def __init__(self, in_channels: int, scale: str = "n") -> None:
        super().__init__()
        if scale not in _PA2_SCALES:
            raise ValueError(f"Unknown PA2 scale={scale!r}; expected one of {list(_PA2_SCALES)}")
        depth, width, max_channels = _PA2_SCALES[scale]

        def ch(raw: int) -> int:
            return _scaled_channels(raw, width, max_channels)

        def nrep(raw: int) -> int:
            return _scaled_depth(raw, depth)

        c64, c128, c256, c512, c1024 = ch(64), ch(128), ch(256), ch(512), ch(1024)
        n2 = nrep(2)

        self.layers = nn.Sequential(
            Conv(in_channels, c64, 3, 2),  # 0 P1/2
            Conv(c64, c128, 3, 2),  # 1 P2/4
            C3k2(c128, c256, n2, False, 0.25),  # 2
            Conv(c256, c256, 3, 2),  # 3 P3/8
            C3k2(c256, c512, n2, False, 0.25),  # 4
            Conv(c512, c1024, 1),  # 5 原 SensorFusion 处，仅升通道
            Conv(c1024, c512, 3, 2),  # 6 P4/16
            C2PSA(c512, c512, n2),  # 7
            C3k2(c512, c512, n2, True),  # 8
            Conv(c512, c1024, 3, 2),  # 9 P5/32
            C2PSA(c1024, c1024, n2),  # 10
            C3k2(c1024, c1024, n2, True),  # 11
            SPPF(c1024, c1024, 5, 3, True),  # 12
            C2PSA(c1024, c1024, n2),  # 13
        )
        self.out_channels = c1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
