"""配置系统：从 YAML 加载训练/采集/评估的全部超参数。

使用嵌套 dataclass 组织配置，通过 load_config() 从 YAML 文件加载。
各子配置块职责清晰，便于为不同任务（PushT、真机）编写不同 yaml。
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from robotfm.policies.rtc import RTCConfig


@dataclass
class EnvConfig:
    """仿真/环境相关配置。"""

    render_size: int = 96           # 观测图像边长（正方形）
    max_episode_steps: int = 300      # 单 episode 最大步数
    render_mode: str = "human"        # "human" 显示窗口 / "rgb_array" 仅返回数组


@dataclass
class DatasetConfig:
    """数据集与 action chunking 相关配置。"""

    n_obs_steps: int = 2    # 策略输入的历史观测帧数
    horizon: int = 16       # 每次预测的未来动作步数（action chunk 长度）
    run_name: str = "pusht_demos"  # 数据子目录名，完整路径 = data_root / run_name
    resize_size: int | None = None  # 裁剪前先双线性缩放到该边长；None 表示不缩放
    crop_size: int | None = 84  # 图像裁剪边长；None 表示不裁剪
    eval_fixed_crop: bool = True  # 评估时用中心裁剪（训练用 random crop）
    # state/action 归一化: gaussian=(x-mean)/std；limits→[-1,1]；limits_01→[0,1]
    norm_mode: str = "gaussian"
    # ACT 图像归一化: imagenet=旧硬编码；dataset=LeRobot VISUAL MEAN_STD（stats.json）
    # FM/A2A 忽略此字段（编码器内仍用 ImageNet）
    image_norm_mode: str = "imagenet"


@dataclass
class PolicyConfig:
    """策略结构与采样超参数（flow_matching / a2a / n_a2a / act）。"""

    type: str = "flow_matching"  # flow_matching | a2a | n_a2a | act
    vision_backbone: str = "resnet18"  # resnet18 | slowfast_r50
    hidden_dim: int = 256           # 条件向量 / 全局 cond 维度
    num_layers: int = 4               # 旧 DiT 字段，保留兼容
    num_heads: int = 4                # 旧 DiT 字段，保留兼容；ACT 用 n_heads
    num_inference_steps: int = 10     # 推理时 Euler 积分步数
    n_action_steps: int = 8           # 闭环执行步数；训练时同步丢弃末尾同名帧数
    beta_alpha: float = 1.5           # 训练时间 t 的 Beta 分布参数 alpha
    beta_beta: float = 1.0            # 训练时间 t 的 Beta 分布参数 beta
    noise_s: float = 0.999            # 时间采样上界缩放，避免 t 取到极端端点
    down_dims: list[int] = field(default_factory=lambda: [256, 512, 1024])
    diffusion_step_embed_dim: int = 256
    kernel_size: int = 5
    n_groups: int = 8
    pretrained_encoder: bool = True  # ImageNet / Kinetics 预训练（微调）
    use_frame_diff: bool = True  # ResNet 路径：[I0, I1-I0, ...] 通道堆叠
    # A2A / N-A2A（flow_matching 忽略）
    latent_dim: int = 512
    consistency_weight: float = 1.0
    enc_recon_weight: float = 0.5
    flow_recon_weight: float = 0.5
    enc_contrastive_weight: float = 0.0
    flow_contrastive_weight: float = 0.0
    history_noise_std: float = 0.0  # >0 启用 N-A2A 历史加噪
    flow_hidden_dim: int = 512
    flow_num_layers: int = 4
    flow_mlp_ratio: float = 4.0
    flow_dropout: float = 0.0
    ae_enc_hidden_dim: int = 512
    ae_dec_hidden_dim: int = 512
    ae_num_layers: int = 4
    ae_dropout: float = 0.0
    decode_flow_latents: bool = True
    # ACT（对齐 LeRobot ACTConfig；其它策略忽略）
    dim_model: int = 512
    dim_feedforward: int = 3200
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1  # 匹配原版 ACT 实际只生效 1 层 decoder
    n_vae_encoder_layers: int = 4
    kl_weight: float = 10.0
    dropout: float = 0.1
    use_vae: bool = True
    pre_norm: bool = False
    feedforward_activation: str = "relu"
    replace_final_stride_with_dilation: bool = False
    temporal_ensemble_coeff: float | None = None
    # RTC 仅影响推理；默认关闭，保持与旧配置行为一致
    rtc: RTCConfig = field(default_factory=lambda: RTCConfig(enabled=False))


@dataclass
class TrainConfig:
    """训练循环超参数。"""

    batch_size: int = 64
    steps: int = 50_000
    lr: float = 1e-4
    weight_decay: float = 1e-6
    log_freq: int = 100       # 每隔多少步打印 loss
    save_freq: int = 5_000    # 每隔多少步保存 checkpoint
    device: str = "cuda"
    num_workers: int = 2      # DataLoader 工作进程数
    cosine_lr: bool = True    # cosine 衰减（含短 warmup）
    warmup_steps: int = 500
    encoder_lr_scale: float = 0.1  # 视觉 backbone lr = lr * scale
    max_grad_norm: float = 1.0


@dataclass
class EvalConfig:
    """评估相关配置。"""

    num_episodes: int = 50
    save_video: bool = False
    video_dir: str = "outputs/eval_videos"
    # 成功判定：episode 内最大真实 coverage（非 reward）达到该阈值。
    # gym_pusht 默认 env success 为 coverage>0.95；评估可单独放宽。
    success_coverage: float = 0.85


@dataclass
class CollectConfig:
    """数据采集（遥操作）相关配置。"""

    target_episodes: int = 30   # 目标采集成功 episode 数
    save_all: bool = False      # True 时保存所有 episode，False 时默认只存成功的
    task: str = "push the T block to the target"  # 写入数据的任务描述


@dataclass
class RobotFMConfig:
    """顶层配置，聚合上述所有子配置。

    同时包含跨模块的通用字段：backend、cameras、state/action 维度等。
    这些字段决定数据格式和策略输入输出形状，必须与 meta.json 一致。
    """

    backend: str = "pusht"
    embodiment: str = "pusht_sim"
    cameras: list[str] = field(default_factory=lambda: ["top"])
    state_dim: int = 2
    action_dim: int = 2
    state_names: list[str] = field(default_factory=lambda: ["x", "y"])
    action_names: list[str] = field(default_factory=lambda: ["x", "y"])
    fps: int = 10
    data_root: str = "data/demos"       # 数据根目录
    output_dir: str = "outputs/fm_run"  # checkpoint 输出目录
    env: EnvConfig = field(default_factory=EnvConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)

    @property
    def camera_specs(self) -> dict[str, dict[str, int]]:
        """根据 render_size 自动生成各相机的分辨率规格，用于 meta.json。"""
        size = self.env.render_size
        return {name: {"height": size, "width": size, "channels": 3} for name in self.cameras}


def _merge_dataclass(obj: Any, data: dict[str, Any]) -> None:
    """递归将 YAML 字典合并到 dataclass 实例（支持嵌套子配置）。"""
    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(obj, key, value)


def _normalize_rtc_config(rtc: RTCConfig) -> RTCConfig:
    """Re-construct RTCConfig so string enums from YAML are coerced."""
    return RTCConfig(
        enabled=rtc.enabled,
        prefix_attention_schedule=rtc.prefix_attention_schedule,
        max_guidance_weight=rtc.max_guidance_weight,
        execution_horizon=rtc.execution_horizon,
        inference_delay=rtc.inference_delay,
        debug=rtc.debug,
        debug_maxlen=rtc.debug_maxlen,
    )


def load_config(path: str | Path) -> RobotFMConfig:
    """从 YAML 文件加载配置，返回 RobotFMConfig 实例。"""
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    cfg = RobotFMConfig()
    _merge_dataclass(cfg, raw)
    cfg.policy.rtc = _normalize_rtc_config(cfg.policy.rtc)
    return cfg


def resolve_path(base: Path, value: str) -> Path:
    """将相对路径解析为基于 base 的绝对路径；已是绝对路径则原样返回。"""
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def _yaml_friendly(obj: Any) -> Any:
    """把 Enum / dataclass 嵌套结构转成可 ``yaml.safe_dump`` 的纯 Python 对象。"""
    if isinstance(obj, dict):
        return {k: _yaml_friendly(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_friendly(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    return obj


def backup_train_config(
    cfg: RobotFMConfig,
    output_dir: Path,
    *,
    source_config_path: str | Path | None = None,
) -> None:
    """把训练配置备份到 checkpoint 目录。

    - ``config.yaml``: 实际生效配置（含 CLI 覆盖后的值）
    - ``config_source.yaml``: 原始 YAML 副本（若提供了源路径）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_path = output_dir / "config.yaml"
    with effective_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            _yaml_friendly(asdict(cfg)),
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    if source_config_path is None:
        return
    src = Path(source_config_path)
    if src.is_file():
        shutil.copy2(src, output_dir / "config_source.yaml")
