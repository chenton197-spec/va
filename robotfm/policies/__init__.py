"""策略模块总入口。

当前包含：
- `encoders.py`: 多相机图像（预训练 ResNet18）与状态编码；`build_multi_camera_encoder`
- `video_encoders.py`: SlowFast-R50 视频编码（与 ResNet 分文件）
- `vit_encoders.py`: ViT-B/16（torchvision ImageNet-1k CLS）
- `pa2_encoders.py`: PA2 YOLO（C3k2 / C2PSA / SPPF，仅图像）
- `unet1d.py`: ConditionalUnet1D + FiLM 动作骨干
- `dit.py`: 旧版 DiT（保留作消融对照）
- `flow_matching.py`: OT-CFM 训练损失与 Euler 采样（可选 RTC）
- `a2a/`: Action-to-Action / N-A2A（torchcfm；obs_state history → future actions；可选 RTC）
- `vita/`: Vision-to-Action Flow Matching（视觉潜变量 → 动作潜变量；复用 a2a AE/flow）
- `act/`: Action Chunking Transformer（对齐 LeRobot / 原版 ACT）
- `rtc/`: Real-Time Chunking 推理引导
- `multitask_dit.py`: 第二阶段语言条件预留

建议阅读顺序：
1. `ARCHITECTURE.md`
2. `flow_matching.py` / `a2a/` / `vita/` / `act/`
3. `rtc/`
4. `encoders.py` / `video_encoders.py` / `vit_encoders.py` / `pa2_encoders.py`
5. `unet1d.py`
"""

from __future__ import annotations

from typing import Any

from robotfm.policies.act import ACTConfig, ACTPolicy
from robotfm.policies.flow_matching import FlowMatchingConfig, FlowMatchingPolicy
from robotfm.policies.rtc import ActionQueue, RTCConfig, RTCProcessor
from robotfm.policies.vita import VITAConfig, VITAPolicy

__all__ = [
    "A2AConfig",
    "A2APolicy",
    "A2AUConfig",
    "A2AUPolicy",
    "ACTConfig",
    "ACTPolicy",
    "ActionQueue",
    "FlowMatchingConfig",
    "FlowMatchingPolicy",
    "RTCConfig",
    "RTCProcessor",
    "VITAConfig",
    "VITAPolicy",
]


def __getattr__(name: str) -> Any:
    """Lazy-import A2A so FM / ACT / dataset paths work without torchcfm."""
    if name in {"A2AConfig", "A2APolicy"}:
        from robotfm.policies.a2a import A2AConfig, A2APolicy

        return A2AConfig if name == "A2AConfig" else A2APolicy
    if name in {"A2AUConfig", "A2AUPolicy"}:
        from robotfm.policies.a2a.a2a_u_policy import A2AUConfig, A2AUPolicy

        return A2AUConfig if name == "A2AUConfig" else A2AUPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
