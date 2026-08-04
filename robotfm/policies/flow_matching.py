from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from robotfm.policies.encoders import MultiCameraEncoder
from robotfm.policies.rtc import RTCConfig, RTCProcessor
from robotfm.policies.unet1d import ConditionalUnet1D
from robotfm.policies.video_encoders import MultiCameraSlowFastEncoder


@dataclass
class FlowMatchingConfig:
    """Flow Matching 策略的关键超参数。

    说明：
    - `num_cameras` / `state_dim` / `action_dim` 决定输入输出维度；
    - `horizon` / `n_obs_steps` 决定动作 chunk 和观测历史长度；
    - `hidden_dim` 同时作为条件向量维度；
    - `down_dims` 控制 ConditionalUnet1D 容量；
    - `vision_backbone`: ``resnet18``（帧差）或 ``slowfast_r50``（视频）；
    - `num_inference_steps` 决定推理时 Euler 积分步数；
    - `beta_alpha` / `beta_beta` / `noise_s` 控制训练时采样的时间分布；
    - `rtc` 为可选 RTC 推理配置（训练不受影响）。
    """

    num_cameras: int
    state_dim: int
    action_dim: int
    horizon: int
    n_obs_steps: int
    hidden_dim: int = 256
    num_layers: int = 4  # 保留字段以兼容旧配置；UNet 不再使用
    num_heads: int = 4  # 保留字段以兼容旧配置；UNet 不再使用
    num_inference_steps: int = 10
    beta_alpha: float = 1.5
    beta_beta: float = 1.0
    noise_s: float = 0.999
    down_dims: tuple[int, ...] = (256, 512, 1024)
    diffusion_step_embed_dim: int = 256
    kernel_size: int = 5
    n_groups: int = 8
    pretrained_encoder: bool = True
    use_frame_diff: bool = True
    vision_backbone: str = "resnet18"
    rtc: RTCConfig | None = None


class FlowMatchingPolicy(nn.Module):
    """基于 Flow Matching 的动作生成策略。

    架构拆分为两部分：
    1. `encoder`:
       把多相机图像 + 历史状态编码成条件向量 `cond`
       （ResNet18 或 SlowFast-R50，由 ``vision_backbone`` 选择）
    2. `unet`:
       在给定 `cond` 和时间 `t` 的前提下，对 noised action chunk
       预测速度场 `v_theta`（ConditionalUnet1D + FiLM）
    训练目标：
        给定真实动作 `x1` 和高斯噪声 `x0`
        构造插值:
            x_t = (1 - t) * x0 + t * x1
        目标速度:
            v* = x1 - x0
        网络学习:
            v_theta(x_t, t, cond) ~= v*

    推理过程：
        从纯噪声 x0 出发，利用 Euler 积分逐步更新：
            x <- x + dt * v_theta(x, t, cond)
        若启用 RTC，则每步速度场经 `RTCProcessor.denoise_step` 做前缀引导。
        最终得到未来 horizon 步动作序列。
    """

    def __init__(self, cfg: FlowMatchingConfig) -> None:
        super().__init__()
        self.cfg = cfg

        backbone = cfg.vision_backbone.lower()
        if backbone in {"resnet18", "resnet"}:
            self.encoder = MultiCameraEncoder(
                num_cameras=cfg.num_cameras,
                state_dim=cfg.state_dim,
                n_obs_steps=cfg.n_obs_steps,
                cond_dim=cfg.hidden_dim,
                pretrained_encoder=cfg.pretrained_encoder,
                use_frame_diff=cfg.use_frame_diff,
            )
        elif backbone in {"slowfast_r50", "slowfast"}:
            self.encoder = MultiCameraSlowFastEncoder(
                num_cameras=cfg.num_cameras,
                state_dim=cfg.state_dim,
                n_obs_steps=cfg.n_obs_steps,
                cond_dim=cfg.hidden_dim,
                pretrained_encoder=cfg.pretrained_encoder,
            )
        else:
            raise ValueError(
                f"Unknown vision_backbone={cfg.vision_backbone!r}; "
                "expected 'resnet18' or 'slowfast_r50'"
            )

        self.unet = ConditionalUnet1D(
            input_dim=cfg.action_dim,
            global_cond_dim=cfg.hidden_dim,
            diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
            down_dims=cfg.down_dims,
            kernel_size=cfg.kernel_size,
            n_groups=cfg.n_groups,
            cond_predict_scale=True,
        )

        self._beta = Beta(cfg.beta_alpha, cfg.beta_beta)
        self.rtc_processor: RTCProcessor | None = None
        if cfg.rtc is not None and cfg.rtc.enabled:
            self.rtc_processor = RTCProcessor(cfg.rtc)

    def _rtc_enabled(self) -> bool:
        return self.rtc_processor is not None and self.cfg.rtc is not None and self.cfg.rtc.enabled

    def _rtc_guidance_enabled(self) -> bool:
        return self._rtc_enabled() and bool(self.cfg.rtc.guidance_enabled)

    def sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """采样训练时间 t。

        先从 Beta 分布采样，再按 `noise_s` 缩放，避免极端端点数值问题。
        """
        sample = self._beta.sample((batch_size,)).to(device=device, dtype=dtype)
        return (self.cfg.noise_s - sample) / self.cfg.noise_s

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """计算 Flow Matching 训练损失。

        batch 约定：
            obs_images:  (B, Cams, T_obs, 3, H, W)
            obs_state:   (B, T_obs, state_dim)
            action:      (B, horizon, action_dim)
            action_mask: (B, horizon, 1)
        """
        actions = batch["action"]
        mask = batch["action_mask"]

        cond = self.encoder(batch["obs_images"], batch["obs_state"])

        noise = torch.randn_like(actions)
        t = self.sample_time(actions.shape[0], actions.device, actions.dtype)
        t_view = t[:, None, None]

        x_t = (1.0 - t_view) * noise + t_view * actions
        target_v = actions - noise
        pred_v = self.unet(x_t, t, cond)

        # action_mask：有效动作步=1，episode 末尾 padding 步=0
        # 乘上 mask 后，末尾 pad 不进 loss；真实短 chunk 动作仍会训练到
        loss = F.mse_loss(pred_v, target_v, reduction="none") * mask
        return loss.sum() / mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, torch.Tensor],
        *,
        prev_chunk_left_over: torch.Tensor | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
    ) -> torch.Tensor:
        """推理生成动作 chunk（Euler 积分，可选 RTC 前缀引导）。

        RTC 参数与 LeRobot ``euler_integrate`` 钩子对齐：
        - ``prev_chunk_left_over``: 上一 chunk 未执行尾部
        - ``inference_delay``: 前缀硬约束步数
        - ``execution_horizon``: soft blend 区域终点

        ``rtc.guidance_enabled=False`` 时仍可在外层用 ActionQueue 做 ahead+discard，
        但本函数不做前缀引导。
        """
        b = batch["obs_state"].shape[0]
        device = batch["obs_state"].device
        dtype = batch["obs_state"].dtype
        cond = self.encoder(batch["obs_images"], batch["obs_state"])

        x = torch.randn(b, self.cfg.horizon, self.cfg.action_dim, device=device, dtype=dtype)
        steps = self.cfg.num_inference_steps
        dt = 1.0 / steps

        use_guidance = self._rtc_guidance_enabled()
        if inference_delay is None and self.cfg.rtc is not None:
            inference_delay = self.cfg.rtc.inference_delay
        if execution_horizon is None and self.cfg.rtc is not None:
            execution_horizon = self.cfg.rtc.execution_horizon

        for i in range(steps):
            t_val = i / steps
            t = torch.full((b,), t_val, device=device, dtype=dtype)

            def denoise_step_partial(input_x_t, current_t=t, current_cond=cond):
                return self.unet(input_x_t, current_t, current_cond)

            if use_guidance:
                assert self.rtc_processor is not None
                v = self.rtc_processor.denoise_step(
                    x_t=x,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=t_val,
                    original_denoise_step_partial=denoise_step_partial,
                    execution_horizon=execution_horizon,
                )
            else:
                v = denoise_step_partial(x)

            x = x + dt * v

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=t_val, x_t=x, v_t=v)

        return x
