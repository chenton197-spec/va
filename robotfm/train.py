"""策略训练入口。

流程:
1. 加载/计算 stats.json
2. 构建 EpisodeDataset + DataLoader
3. 按 policy.type 实例化 FlowMatchingPolicy / A2APolicy / VITAPolicy / ACTPolicy
4. AdamW（encoder 小 lr）+ grad clip，定期保存 checkpoint
5. 将训练日志写入 ``output_dir/train.log``

Checkpoint 含 ``step`` / ``policy_state_dict`` / ``optimizer_state_dict`` /
``config`` / ``stats``，可用 ``--resume`` 续训（``train.steps`` 为全局总步数）。
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from robotfm.config import (
    RobotFMConfig,
    _normalize_rtc_config,
    backup_train_config,
    resolve_path,
)
from robotfm.collect.loop import get_run_dir
from robotfm.data.dataset import (
    apply_camera_dropout,
    apply_image_augments_batch,
    build_episode_dataset,
    camera_dropout_prob,
)
from robotfm.data.stats import ensure_stats, is_limits_mode, resolve_image_stats
from robotfm.policies.act import ACTConfig, ACTPolicy
from robotfm.policies.flow_matching import FlowMatchingConfig, FlowMatchingPolicy
from robotfm.policies.vita import VITAConfig, VITAPolicy

# A2A is optional (torchcfm); import lazily in _build_a2a_policy.
PolicyModule = FlowMatchingPolicy | VITAPolicy | ACTPolicy | torch.nn.Module


class _TrainLogger:
    """同时写控制台与 ``train.log``（文件为可读行，不含 tqdm \\r）。"""

    def __init__(self, log_path: Path, *, append: bool = False) -> None:
        self.path = log_path
        self._fh = log_path.open("a" if append else "w", encoding="utf-8")

    def log(self, msg: str, *, also_print: bool = True) -> None:
        line = msg if msg.endswith("\n") else msg + "\n"
        self._fh.write(line)
        self._fh.flush()
        if also_print:
            print(msg, flush=True)

    def close(self) -> None:
        self._fh.close()


def _checkpoint_payload(
    *,
    step: int,
    policy: PolicyModule,
    optim: torch.optim.Optimizer,
    cfg: RobotFMConfig,
    stats: dict,
) -> dict:
    """统一 checkpoint 字段，支持续训（含 optimizer / step）。"""
    return {
        "step": step,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "config": cfg,
        "stats": stats,
    }


def _load_resume_checkpoint(
    resume_path: Path,
    policy: PolicyModule,
    optim: torch.optim.Optimizer,
    logger: _TrainLogger,
    *,
    load_optimizer: bool = True,
) -> tuple[int, dict | None]:
    """从 checkpoint 恢复权重 / optimizer / step。

    返回 ``(start_step, stats_or_none)``；stats 优先用 ckpt 内的，保证归一化一致。
    """
    if not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    if "policy_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing policy_state_dict: {resume_path}")
    policy.load_state_dict(ckpt["policy_state_dict"])
    start_step = int(ckpt.get("step", 0))
    if (
        load_optimizer
        and "optimizer_state_dict" in ckpt
        and ckpt["optimizer_state_dict"] is not None
    ):
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        # AdamW 自定义 base_lr 若旧 ckpt 缺失则回填，避免 cosine 调度 KeyError
        for group in optim.param_groups:
            if "base_lr" not in group:
                group["base_lr"] = float(group["lr"])
        logger.log(f"resume: loaded optimizer_state_dict from {resume_path}")
    elif load_optimizer:
        logger.log(
            f"resume: WARNING no optimizer_state_dict in {resume_path}; "
            "optimizer starts fresh (weights/step still restored)"
        )
    else:
        logger.log("resume: skip optimizer_state_dict (fresh optimizer)")
    logger.log(f"resume: start_step={start_step} from {resume_path}")
    stats = ckpt.get("stats")
    return start_step, stats if stats is not None else None


def _build_a2a_policy(cfg: RobotFMConfig) -> torch.nn.Module:
    try:
        from robotfm.policies.a2a import A2AConfig, A2APolicy
    except ImportError as exc:
        raise ImportError(
            "policy.type a2a/n_a2a requires torchcfm; pip install torchcfm"
        ) from exc

    ptype = cfg.policy.type.lower()
    history_noise_std = cfg.policy.history_noise_std
    use_ot = ptype == "n_a2a"
    if ptype == "n_a2a" and history_noise_std <= 0:
        history_noise_std = 0.1
    rtc = _normalize_rtc_config(cfg.policy.rtc)
    a2a_cfg = A2AConfig(
        num_cameras=len(cfg.cameras),
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        horizon=cfg.dataset.horizon,
        n_obs_steps=cfg.dataset.n_obs_steps,
        n_action_steps=cfg.policy.n_action_steps,
        latent_dim=cfg.policy.latent_dim,
        hidden_dim=cfg.policy.hidden_dim,
        num_inference_steps=cfg.policy.num_inference_steps,
        consistency_weight=cfg.policy.consistency_weight,
        enc_recon_weight=cfg.policy.enc_recon_weight,
        flow_recon_weight=cfg.policy.flow_recon_weight,
        enc_contrastive_weight=cfg.policy.enc_contrastive_weight,
        flow_contrastive_weight=cfg.policy.flow_contrastive_weight,
        history_noise_std=history_noise_std,
        use_ot_matcher=use_ot or cfg.policy.use_ot_matcher,
        decode_flow_latents=cfg.policy.decode_flow_latents,
        flow_hidden_dim=cfg.policy.flow_hidden_dim,
        flow_num_layers=cfg.policy.flow_num_layers,
        flow_mlp_ratio=cfg.policy.flow_mlp_ratio,
        flow_dropout=cfg.policy.flow_dropout,
        ae_enc_hidden_dim=cfg.policy.ae_enc_hidden_dim,
        ae_dec_hidden_dim=cfg.policy.ae_dec_hidden_dim,
        ae_num_layers=cfg.policy.ae_num_layers,
        ae_dropout=cfg.policy.ae_dropout,
        pretrained_encoder=cfg.policy.pretrained_encoder,
        use_frame_diff=cfg.policy.use_frame_diff,
        share_image_encoder=cfg.policy.share_image_encoder,
        rtc=rtc if rtc.enabled else None,
    )
    return A2APolicy(a2a_cfg)


def _build_vita_policy(cfg: RobotFMConfig) -> VITAPolicy:
    """构造 VITA（视觉潜变量 → 动作潜变量）。"""
    vita_cfg = VITAConfig(
        num_cameras=len(cfg.cameras),
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        horizon=cfg.dataset.horizon,
        n_obs_steps=cfg.dataset.n_obs_steps,
        n_action_steps=cfg.policy.n_action_steps,
        latent_dim=cfg.policy.latent_dim,
        hidden_dim=cfg.policy.hidden_dim,
        num_inference_steps=cfg.policy.num_inference_steps,
        consistency_weight=cfg.policy.consistency_weight,
        enc_recon_weight=cfg.policy.enc_recon_weight,
        flow_recon_weight=cfg.policy.flow_recon_weight,
        enc_contrastive_weight=cfg.policy.enc_contrastive_weight,
        flow_contrastive_weight=cfg.policy.flow_contrastive_weight,
        use_ot_matcher=cfg.policy.use_ot_matcher,
        decode_flow_latents=cfg.policy.decode_flow_latents,
        flow_hidden_dim=cfg.policy.flow_hidden_dim,
        flow_num_layers=cfg.policy.flow_num_layers,
        flow_mlp_ratio=cfg.policy.flow_mlp_ratio,
        flow_dropout=cfg.policy.flow_dropout,
        ae_enc_hidden_dim=cfg.policy.ae_enc_hidden_dim,
        ae_dec_hidden_dim=cfg.policy.ae_dec_hidden_dim,
        ae_num_layers=cfg.policy.ae_num_layers,
        ae_dropout=cfg.policy.ae_dropout,
        pretrained_encoder=cfg.policy.pretrained_encoder,
        use_frame_diff=cfg.policy.use_frame_diff,
        share_image_encoder=cfg.policy.share_image_encoder,
    )
    return VITAPolicy(vita_cfg)


def _build_act_policy(
    cfg: RobotFMConfig,
    stats: dict | None = None,
) -> ACTPolicy:
    """构造 ACT（chunk_size = dataset.horizon）。"""
    if cfg.dataset.n_obs_steps != 1:
        raise ValueError(
            f"ACT requires dataset.n_obs_steps=1, got {cfg.dataset.n_obs_steps}"
        )
    image_mean, image_std = resolve_image_stats(stats, cfg.dataset.image_norm_mode)
    act_cfg = ACTConfig(
        num_cameras=len(cfg.cameras),
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        chunk_size=cfg.dataset.horizon,
        n_action_steps=cfg.policy.n_action_steps,
        n_obs_steps=cfg.dataset.n_obs_steps,
        vision_backbone=cfg.policy.vision_backbone,
        pretrained_backbone=cfg.policy.pretrained_encoder,
        replace_final_stride_with_dilation=cfg.policy.replace_final_stride_with_dilation,
        pre_norm=cfg.policy.pre_norm,
        dim_model=cfg.policy.dim_model,
        n_heads=cfg.policy.n_heads,
        dim_feedforward=cfg.policy.dim_feedforward,
        feedforward_activation=cfg.policy.feedforward_activation,
        n_encoder_layers=cfg.policy.n_encoder_layers,
        n_decoder_layers=cfg.policy.n_decoder_layers,
        dropout=cfg.policy.dropout,
        use_vae=cfg.policy.use_vae,
        latent_dim=cfg.policy.latent_dim,
        n_vae_encoder_layers=cfg.policy.n_vae_encoder_layers,
        kl_weight=cfg.policy.kl_weight,
        temporal_ensemble_coeff=cfg.policy.temporal_ensemble_coeff,
    )
    return ACTPolicy(act_cfg, image_mean=image_mean, image_std=image_std)


def build_policy(cfg: RobotFMConfig, stats: dict | None = None) -> PolicyModule:
    """根据 RobotFMConfig.policy.type 构造策略。

    ``stats`` 在 ACT + ``image_norm_mode=dataset`` 时必需（提供 image_mean/std）。
    """
    ptype = cfg.policy.type.lower()
    if ptype in {"a2a", "n_a2a"}:
        return _build_a2a_policy(cfg)
    if ptype == "vita":
        return _build_vita_policy(cfg)
    if ptype == "act":
        return _build_act_policy(cfg, stats)
    if ptype != "flow_matching":
        raise ValueError(f"Unknown policy.type={cfg.policy.type!r}")

    down_dims = tuple(cfg.policy.down_dims)
    rtc = _normalize_rtc_config(cfg.policy.rtc)
    fm_cfg = FlowMatchingConfig(
        num_cameras=len(cfg.cameras),
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        horizon=cfg.dataset.horizon,
        n_obs_steps=cfg.dataset.n_obs_steps,
        hidden_dim=cfg.policy.hidden_dim,
        num_layers=cfg.policy.num_layers,
        num_heads=cfg.policy.num_heads,
        num_inference_steps=cfg.policy.num_inference_steps,
        beta_alpha=cfg.policy.beta_alpha,
        beta_beta=cfg.policy.beta_beta,
        noise_s=cfg.policy.noise_s,
        down_dims=down_dims,
        diffusion_step_embed_dim=cfg.policy.diffusion_step_embed_dim,
        kernel_size=cfg.policy.kernel_size,
        n_groups=cfg.policy.n_groups,
        pretrained_encoder=cfg.policy.pretrained_encoder,
        use_frame_diff=cfg.policy.use_frame_diff,
        share_image_encoder=cfg.policy.share_image_encoder,
        vision_backbone=cfg.policy.vision_backbone,
        rtc=rtc if rtc.enabled else None,
    )
    return FlowMatchingPolicy(fm_cfg)


def _build_optimizer(policy: PolicyModule, cfg: RobotFMConfig) -> torch.optim.Optimizer:
    """视觉 backbone 用更小 lr，其余参数用完整 lr。"""
    if isinstance(policy, ACTPolicy):
        backbone_ids = {id(p) for p in policy.model.backbone.parameters()}
        encoder_params = [p for p in policy.parameters() if id(p) in backbone_ids]
        other_params = [p for p in policy.parameters() if id(p) not in backbone_ids]
    else:
        vision_fn = getattr(policy.encoder, "vision_parameters", None)
        if callable(vision_fn):
            encoder_ids = {id(p) for p in vision_fn()}
        else:
            encoder_ids = {id(p) for p in policy.encoder.image_encoder.parameters()}
        encoder_params = [p for p in policy.parameters() if id(p) in encoder_ids]
        other_params = [p for p in policy.parameters() if id(p) not in encoder_ids]
    encoder_lr = cfg.train.lr * cfg.train.encoder_lr_scale
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr, "base_lr": encoder_lr},
            {"params": other_params, "lr": cfg.train.lr, "base_lr": cfg.train.lr},
        ],
        weight_decay=cfg.train.weight_decay,
    )


def _set_cosine_lr(optim: torch.optim.Optimizer, step: int, cfg: RobotFMConfig) -> None:
    """线性 warmup + cosine 衰减；各组按各自 base_lr 缩放。"""
    warmup = max(cfg.train.warmup_steps, 1)
    total = max(cfg.train.steps, 1)
    if step < warmup:
        scale = (step + 1) / warmup
    else:
        progress = (step - warmup) / max(total - warmup, 1)
        scale = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    for group in optim.param_groups:
        base = group.get("base_lr", cfg.train.lr)
        group["lr"] = base * scale


def train_flow_matching(
    cfg: RobotFMConfig,
    base_dir: Path,
    *,
    resume_path: Path | None = None,
    reset_step: bool = False,
    source_config_path: Path | None = None,
) -> Path:
    """执行行为克隆训练（Flow Matching / A2A / N-A2A / ACT）。

    ``cfg.train.steps`` 为 **全局总步数**。续训时从 checkpoint 的 ``step`` 接着跑到该值。
    例如已训到 30000、再训 30k → 设 ``steps: 60000`` 并 ``--resume checkpoint_final.pt``。

    ``reset_step=True``：只加载权重（可选 optimizer），从 step=0 再跑 ``train.steps``
    （适合换数据 finetune 固定再训 N 步）。
    """
    base_dir = base_dir.resolve()
    run_dir = get_run_dir(cfg, base_dir)

    output_dir = resolve_path(base_dir, cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_train_config(cfg, output_dir, source_config_path=source_config_path)
    log_path = output_dir / "train.log"
    append_log = bool(resume_path) and log_path.exists()
    logger = _TrainLogger(log_path, append=append_log)
    if append_log:
        logger.log("--- resume ---")
    logger.log(f"start_time: {datetime.now().isoformat(timespec='seconds')}")
    logger.log(f"output_dir: {output_dir}")
    logger.log(f"config_backup: {output_dir / 'config.yaml'}")
    logger.log(f"policy.type: {cfg.policy.type}")
    logger.log(f"dataset.run_name: {cfg.dataset.run_name}")
    logger.log(
        f"train: steps={cfg.train.steps} batch_size={cfg.train.batch_size} "
        f"lr={cfg.train.lr} log_freq={cfg.train.log_freq} save_freq={cfg.train.save_freq}"
    )
    if resume_path is not None:
        logger.log(f"resume_path: {resume_path}")

    try:
        require_image = (
            cfg.policy.type.lower() == "act"
            and getattr(cfg.dataset, "image_norm_mode", "imagenet") == "dataset"
        )
        stats = ensure_stats(
            run_dir, cfg.dataset.norm_mode, require_image_stats=require_image
        )
        logger.log(f"dataset.norm_mode: {cfg.dataset.norm_mode}")
        logger.log(
            f"dataset.image_norm_mode: {getattr(cfg.dataset, 'image_norm_mode', 'imagenet')}"
        )
        logger.log(f"train.cosine_lr: {cfg.train.cosine_lr}")

        resolved_resume: Path | None = None
        if resume_path is not None:
            resolved_resume = resume_path if resume_path.is_absolute() else (base_dir / resume_path)
            if not resolved_resume.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resolved_resume}")
            ckpt_head = torch.load(resolved_resume, map_location="cpu", weights_only=False)
            ckpt_cfg = ckpt_head.get("config")
            ckpt_run = None
            if ckpt_cfg is not None and hasattr(ckpt_cfg, "dataset"):
                ckpt_run = getattr(ckpt_cfg.dataset, "run_name", None)
            # 同数据集续训：沿用 ckpt stats；换数据 finetune：用当前 run 的 stats
            if (
                ckpt_head.get("stats") is not None
                and ckpt_run is not None
                and ckpt_run == cfg.dataset.run_name
            ):
                stats = ckpt_head["stats"]
                logger.log(
                    f"resume: using stats from checkpoint (same dataset run_name={ckpt_run})"
                )
            elif ckpt_run is not None and ckpt_run != cfg.dataset.run_name:
                logger.log(
                    f"resume: dataset changed ({ckpt_run} -> {cfg.dataset.run_name}); "
                    "using current dataset stats.json"
                )

        # limits / limits_01 需要 min/max；旧 stats/ckpt 可能没有，自动补齐
        if is_limits_mode(cfg.dataset.norm_mode):
            needed = ("state_min", "state_max", "action_min", "action_max")
            if any(k not in stats for k in needed):
                stats = ensure_stats(run_dir, cfg.dataset.norm_mode)
                logger.log(
                    f"stats: recomputed with min/max for norm_mode={cfg.dataset.norm_mode}"
                )

        # ACT dataset 图像归一化：ckpt/旧 stats 可能缺 image_*，补齐
        if require_image and ("image_mean" not in stats or "image_std" not in stats):
            stats = ensure_stats(
                run_dir, cfg.dataset.norm_mode, require_image_stats=True
            )
            logger.log("stats: recomputed with image_mean/image_std for ACT dataset norm")

        gpu_augment = bool(cfg.dataset.gpu_augment)
        dataset = build_episode_dataset(
            run_dir=run_dir,
            n_obs_steps=cfg.dataset.n_obs_steps,
            horizon=cfg.dataset.horizon,
            n_action_steps=cfg.policy.n_action_steps,
            drop_n_last_frames=0,  # 末尾不足 0-pad + mask，不丢帧
            stats=stats,
            normalize=True,
            norm_mode=cfg.dataset.norm_mode,
            resize_size=cfg.dataset.resize_size,
            crop_size=cfg.dataset.crop_size,
            random_crop=True,
            color_jitter_brightness=cfg.dataset.color_jitter_brightness,
            color_jitter_contrast=cfg.dataset.color_jitter_contrast,
            color_jitter_saturation=cfg.dataset.color_jitter_saturation,
            color_jitter_hue=cfg.dataset.color_jitter_hue,
            defer_augment=gpu_augment,
            uint8_cache=bool(cfg.dataset.uint8_cache),
            uint8_cache_dir=cfg.dataset.uint8_cache_dir,
        )
        logger.log(f"dataset: run_dir={run_dir} num_samples={len(dataset)}")
        if cfg.dataset.uint8_cache:
            cache = getattr(dataset, "_image_cache", None)
            cache_dir = getattr(cache, "cache_dir", None)
            logger.log(f"dataset.uint8_cache: enabled path={cache_dir}")
        if gpu_augment:
            logger.log(
                "dataset.gpu_augment: crop/color_jitter deferred to GPU batch path"
            )
        loader_kwargs: dict = {
            "batch_size": cfg.train.batch_size,
            "shuffle": True,
            "num_workers": cfg.train.num_workers,
            "pin_memory": True,
            "drop_last": True,
        }
        if cfg.train.num_workers > 0:
            # Avoid worker respawn cost; keep prefetch small for 16GB RAM.
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        loader = DataLoader(dataset, **loader_kwargs)

        device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
        logger.log(f"device: {device}")
        policy = build_policy(cfg, stats).to(device)
        optim = _build_optimizer(policy, cfg)

        step = 0
        if resolved_resume is not None:
            step, _ = _load_resume_checkpoint(
                resolved_resume,
                policy,
                optim,
                logger,
                load_optimizer=not reset_step,
            )
            # 把 Adam 状态迁到训练 device
            for state in optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            if reset_step:
                logger.log(f"resume: reset_step enabled (was {step} -> 0)")
                step = 0

        if step >= cfg.train.steps:
            raise ValueError(
                f"Resume step ({step}) >= train.steps ({cfg.train.steps}). "
                "Increase train.steps (total global steps) to continue, "
                "e.g. steps=60000 to train another 30k after a 30k run."
            )

        pbar = tqdm(total=cfg.train.steps, initial=step, desc="train")
        while step < cfg.train.steps:
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                if gpu_augment:
                    batch["obs_images"] = apply_image_augments_batch(
                        batch["obs_images"],
                        crop_size=cfg.dataset.crop_size,
                        random_crop=True,
                        brightness=cfg.dataset.color_jitter_brightness,
                        contrast=cfg.dataset.color_jitter_contrast,
                        saturation=cfg.dataset.color_jitter_saturation,
                        hue=cfg.dataset.color_jitter_hue,
                    )
                cam_drop_p = 0.0
                cam_drop_cfg = cfg.dataset.camera_dropout
                if cam_drop_cfg.enabled:
                    cam_drop_p = camera_dropout_prob(
                        step,
                        cfg.train.steps,
                        schedule_steps=cam_drop_cfg.schedule_steps,
                        early_frac=cam_drop_cfg.early_frac,
                        mid_frac=cam_drop_cfg.mid_frac,
                        early_prob=cam_drop_cfg.early_prob,
                        mid_prob=cam_drop_cfg.mid_prob,
                        late_prob=cam_drop_cfg.late_prob,
                    )
                    batch["obs_images"] = apply_camera_dropout(
                        batch["obs_images"],
                        cam_drop_p,
                        keep_at_least_one=cam_drop_cfg.keep_at_least_one,
                    )
                if cfg.train.cosine_lr:
                    _set_cosine_lr(optim, step, cfg)
                loss = policy.compute_loss(batch)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                # clip_grad_norm_ 返回值为 clip 前的总梯度范数
                grad_norm_v = float("nan")
                if cfg.train.max_grad_norm > 0:
                    grad_norm_v = float(
                        torch.nn.utils.clip_grad_norm_(
                            policy.parameters(), cfg.train.max_grad_norm
                        )
                    )
                optim.step()

                lr = float(optim.param_groups[-1]["lr"])
                loss_v = float(loss.item())
                if step % cfg.train.log_freq == 0:
                    clipped = (
                        cfg.train.max_grad_norm > 0
                        and grad_norm_v == grad_norm_v  # not NaN
                        and grad_norm_v > cfg.train.max_grad_norm
                    )
                    postfix = {
                        "loss": loss_v,
                        "lr": lr,
                        "gnorm": grad_norm_v,
                        "clip": int(clipped),
                    }
                    if cam_drop_cfg.enabled:
                        postfix["cam_drop_p"] = cam_drop_p
                    pbar.set_postfix(**postfix)
                    # 只写文件，避免打断 tqdm 进度条
                    drop_msg = (
                        f" cam_drop_p={cam_drop_p:.4g}" if cam_drop_cfg.enabled else ""
                    )
                    logger.log(
                        f"step={step}/{cfg.train.steps} loss={loss_v:.6f} "
                        f"lr={lr:.8g} grad_norm={grad_norm_v:.6f} "
                        f"max_grad_norm={cfg.train.max_grad_norm:g} "
                        f"clipped={int(clipped)}{drop_msg}",
                        also_print=False,
                    )
                if step > 0 and step % cfg.train.save_freq == 0:
                    ckpt = output_dir / f"checkpoint_{step:06d}.pt"
                    torch.save(
                        _checkpoint_payload(
                            step=step, policy=policy, optim=optim, cfg=cfg, stats=stats
                        ),
                        ckpt,
                    )
                    logger.log(f"saved: {ckpt}", also_print=False)
                step += 1
                pbar.update(1)
                if step >= cfg.train.steps:
                    break

        final_ckpt = output_dir / "checkpoint_final.pt"
        torch.save(
            _checkpoint_payload(
                step=step, policy=policy, optim=optim, cfg=cfg, stats=stats
            ),
            final_ckpt,
        )
        pbar.close()
        logger.log(f"saved: {final_ckpt}")
        logger.log(f"end_time: {datetime.now().isoformat(timespec='seconds')}")
        return final_ckpt
    finally:
        logger.close()
