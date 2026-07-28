"""ACT 超参数配置（对齐 LeRobot ``ACTConfig`` / 原版 ACT）。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ACTConfig:
    """Action Chunking Transformer 配置。

    默认值对齐 LeRobot ``ACTConfig``（原版 ACT 论文设定）：
    - ``n_decoder_layers=1``：匹配原仓库 decoder 实际只生效 1 层的行为
      （https://github.com/tonyzhaozh/act/issues/25）
    - ``n_obs_steps`` 必须为 1（单帧观测）
    """

    num_cameras: int
    state_dim: int
    action_dim: int
    chunk_size: int = 100
    n_action_steps: int = 100
    n_obs_steps: int = 1

    # Vision backbone
    vision_backbone: str = "resnet18"
    pretrained_backbone: bool = True
    replace_final_stride_with_dilation: bool = False

    # Transformer
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    dropout: float = 0.1

    # VAE
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    kl_weight: float = 10.0

    # Inference
    temporal_ensemble_coeff: float | None = None

    def __post_init__(self) -> None:
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be a ResNet variant, got {self.vision_backbone!r}"
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"ACT only supports n_obs_steps=1, got {self.n_obs_steps}"
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})"
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "temporal ensembling requires n_action_steps=1 "
                "(policy must be queried every env step)"
            )
