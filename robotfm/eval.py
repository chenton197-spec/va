"""Flow Matching / ACT 策略仿真评估。

闭环控制逻辑:
- 默认（RTC off）: action chunking + replan
  1. 维护最近 n_obs_steps 帧观测历史
  2. 当当前 chunk 执行完时，调用 policy.sample_actions 预测新 chunk
  3. 反归一化后依次执行前 n_action_steps 步，再 replan
- ACT temporal ensemble（``temporal_ensemble_coeff`` 非空）:
  每步 ``select_action``（指数加权），episode 开始时 ``policy.reset()``
- RTC on: 用 ActionQueue 管理 leftover / inference_delay，采样时前缀引导
  1. leftover = queue.get_left_over()
  2. sample_actions(..., prev_chunk_left_over=leftover, inference_delay=...)
  3. queue.merge(original, processed, real_delay)
  4. 逐步 queue.get() → env.step
- 统计 success_rate、avg_reward、mean_max_coverage，可选保存 rollout 视频
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from robotfm.config import RobotFMConfig, _normalize_rtc_config, resolve_path
from robotfm.collect.loop import get_run_dir
from robotfm.data.action_delta import (
    denormalize_predicted_action,
    flow_history_from_phys,
    joint_mask_from_names,
)
from robotfm.data.dataset import spatial_preprocess_images
from robotfm.data.stats import ensure_stats, normalize
from robotfm.envs.registry import make_env
from robotfm.policies.rtc import ActionQueue
from robotfm.train import build_policy


def _pace_step(step_start: float, fps: int) -> None:
    """Sleep so wall-clock step rate roughly matches ``fps``."""
    if fps <= 0:
        return
    elapsed = time.perf_counter() - step_start
    remain = (1.0 / fps) - elapsed
    if remain > 0:
        time.sleep(remain)


def _denormalize_chunk(
    pred: torch.Tensor,
    stats: dict[str, np.ndarray],
    cfg: RobotFMConfig,
    q_now_phys: np.ndarray,
) -> torch.Tensor:
    """Denormalize a (T, A) action chunk tensor for env execution."""
    return denormalize_predicted_action(
        pred,
        stats,
        cfg.dataset.norm_mode,
        q_now_phys=q_now_phys,
        predict_joint_delta=bool(cfg.policy.predict_joint_delta),
        joint_mask=joint_mask_from_names(cfg.action_names, cfg.action_dim),
    )


def _denormalize_action(
    action: np.ndarray,
    stats: dict[str, np.ndarray],
    cfg: RobotFMConfig,
    q_now_phys: np.ndarray,
) -> np.ndarray:
    """将归一化动作还原为环境物理量。"""
    out = denormalize_predicted_action(
        action,
        stats,
        cfg.dataset.norm_mode,
        q_now_phys=q_now_phys,
        predict_joint_delta=bool(cfg.policy.predict_joint_delta),
        joint_mask=joint_mask_from_names(cfg.action_names, cfg.action_dim),
    )
    return np.asarray(out, dtype=np.float32)


def _normalize_state(
    state: np.ndarray, stats: dict[str, np.ndarray], norm_mode: str
) -> np.ndarray:
    """与训练时一致的状态归一化。"""
    return normalize(state, stats, prefix="state", mode=norm_mode)


def _build_obs_batch(
    obs_history: list,
    cfg: RobotFMConfig,
    stats: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """将观测历史列表转为策略输入 batch（batch_size=1）。

    注意 obs_images 形状为 (1, Cams, T_obs, 3, H, W)，与 Dataset 单样本一致。
    评估使用固定中心裁剪（``eval_fixed_crop``）。
    历史不足时重复最早一帧（与训练 / 原版 A2A 的 state pad 一致）。
    A2A / N-A2A 的 flow 起点是 ``obs_state``（agent_pos），不再需要 action history。
    """
    n_obs = cfg.dataset.n_obs_steps
    obs_history = obs_history[-n_obs:]
    while len(obs_history) < n_obs:
        obs_history.insert(0, obs_history[0])

    camera_histories = []
    for cam in cfg.cameras:
        frames = []
        for obs in obs_history:
            img = obs.images[cam].astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            frames.append(torch.from_numpy(img))
        camera_histories.append(torch.stack(frames, dim=0))

    norm_mode = cfg.dataset.norm_mode
    states = [
        torch.from_numpy(_normalize_state(obs.state.astype(np.float32), stats, norm_mode))
        for obs in obs_history
    ]

    obs_images = torch.stack(camera_histories, dim=0)
    obs_images = spatial_preprocess_images(
        obs_images,
        pre_crop_size=cfg.dataset.pre_crop_size,
        resize_size=cfg.dataset.resize_size,
        crop_size=cfg.dataset.crop_size if cfg.dataset.eval_fixed_crop else None,
        random_crop=False,
    )

    obs_images = obs_images.unsqueeze(0).to(device)
    obs_state = torch.stack(states, dim=0).unsqueeze(0).to(device)
    state_phys = np.stack(
        [np.asarray(obs.state, dtype=np.float32) for obs in obs_history], axis=0
    )
    obs_history_n = flow_history_from_phys(
        state_phys,
        stats,
        norm_mode,
        predict_joint_delta=bool(cfg.policy.predict_joint_delta),
        action_names=list(cfg.action_names) if cfg.action_names else None,
    )
    return {
        "obs_images": obs_images,
        "obs_state": obs_state,
        "obs_history": torch.from_numpy(obs_history_n).unsqueeze(0).to(device),
    }


def evaluate_flow_matching(
    cfg: RobotFMConfig,
    checkpoint: Path,
    base_dir: Path,
    render: bool = False,
    pace_realtime: bool = False,
) -> dict[str, float]:
    """在仿真环境中评估 checkpoint。

    参数:
        render: True 时用 human 窗口实时观看；False 用 rgb_array（可录视频）
        pace_realtime: True 时按 ``cfg.fps`` 限速，便于人眼观看

    返回:
        success_rate: 成功 episode 比例（max coverage >= cfg.eval.success_coverage）
        avg_reward:   平均累计奖励
        mean_max_coverage: 各 episode 最大真实 coverage 的均值

    注: coverage 达到 ``cfg.eval.success_coverage`` 时 early-stop，
    避免策略在已达标后继续推块把对齐破坏掉。
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stats = ckpt["stats"]
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    # ACT + dataset 图像归一化：旧 ckpt stats 可能缺 image_*，从 run_dir 补齐
    require_image = (
        cfg.policy.type.lower() == "act"
        and getattr(cfg.dataset, "image_norm_mode", "imagenet") == "dataset"
    )
    if require_image and ("image_mean" not in stats or "image_std" not in stats):
        run_dir = get_run_dir(cfg, base_dir)
        stats = ensure_stats(run_dir, cfg.dataset.norm_mode, require_image_stats=True)

    policy = build_policy(cfg, stats)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()

    title = None
    if render:
        title = "PushT Eval (RTC)" if cfg.policy.rtc.enabled else "PushT Eval"
    env = make_env(cfg, render_mode="human" if render else "rgb_array", window_title=title)

    video_dir = resolve_path(base_dir, cfg.eval.video_dir)
    if cfg.eval.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    rtc_cfg = _normalize_rtc_config(cfg.policy.rtc)
    rtc_enabled = rtc_cfg.enabled
    use_temporal_ensemble = (
        cfg.policy.temporal_ensemble_coeff is not None and hasattr(policy, "select_action")
    )
    pace_fps = cfg.fps if pace_realtime else 0
    success_thresh = float(cfg.eval.success_coverage)

    successes = 0
    rewards = []
    max_coverages = []

    for ep in range(cfg.eval.num_episodes):
        obs = env.reset(seed=ep)
        obs_history = [obs]
        ep_reward = 0.0
        max_coverage = 0.0
        done = False
        frames = []
        step = None
        if use_temporal_ensemble and hasattr(policy, "reset"):
            policy.reset()

        def _apply_step(step_result) -> bool:
            """更新 reward/coverage；env done 或 coverage 达标则结束 episode。"""
            nonlocal ep_reward, max_coverage, obs, done
            ep_reward += step_result.reward
            cov = float(step_result.info.get("coverage", step_result.reward))
            max_coverage = max(max_coverage, cov)
            obs = step_result.observation
            obs_history.append(obs)
            # Early-stop at eval success_coverage so the policy does not keep pushing.
            done = step_result.done or max_coverage >= success_thresh
            return done

        if rtc_enabled:
            action_queue = ActionQueue(rtc_cfg)

            while not done:
                # Replan when remaining steps <= inference_delay (think-while-moving).
                if action_queue.qsize() <= rtc_cfg.inference_delay:
                    leftover = action_queue.get_left_over()
                    if leftover is not None and leftover.shape[0] == 0:
                        leftover = None

                    batch = _build_obs_batch(obs_history, cfg, stats, device)
                    with torch.no_grad():
                        pred = policy.sample_actions(
                            batch,
                            prev_chunk_left_over=leftover,
                            inference_delay=rtc_cfg.inference_delay,
                            execution_horizon=rtc_cfg.execution_horizon,
                        )[0]

                    # Simulate latency: consume leftover delay steps before swapping chunks.
                    delay = 0 if leftover is None else rtc_cfg.inference_delay
                    for _ in range(min(delay, action_queue.qsize())):
                        if done:
                            break
                        t0 = time.perf_counter()
                        action_t = action_queue.get()
                        if action_t is None:
                            break
                        step = env.step(action_t.numpy())
                        _apply_step(step)
                        frame = env.render_rgb()
                        if frame is not None:
                            frames.append(frame)
                        _pace_step(t0, pace_fps)

                    if done:
                        break

                    processed = _denormalize_chunk(
                        pred, stats, cfg, obs_history[-1].state.astype(np.float32)
                    )
                    action_queue.merge(pred.cpu(), processed.cpu(), real_delay=delay)

                t0 = time.perf_counter()
                action_t = action_queue.get()
                if action_t is None:
                    break
                step = env.step(action_t.numpy())
                _apply_step(step)

                frame = env.render_rgb()
                if frame is not None:
                    frames.append(frame)
                _pace_step(t0, pace_fps)
        elif use_temporal_ensemble:
            # ACT 标准部署：每步查询 + 指数加权 temporal ensembling
            while not done:
                batch = _build_obs_batch(obs_history, cfg, stats, device)
                with torch.no_grad():
                    action_t = policy.select_action(batch)
                action_np = action_t[0].detach().cpu().numpy()
                action = _denormalize_action(
                    action_np, stats, cfg, obs_history[-1].state.astype(np.float32)
                )

                t0 = time.perf_counter()
                step = env.step(action)
                _apply_step(step)

                frame = env.render_rgb()
                if frame is not None:
                    frames.append(frame)
                _pace_step(t0, pace_fps)
        else:
            chunk_actions: list[np.ndarray] = []
            chunk_idx = 0

            while not done:
                if chunk_idx >= len(chunk_actions):
                    batch = _build_obs_batch(obs_history, cfg, stats, device)
                    with torch.no_grad():
                        pred = policy.sample_actions(batch)[0].cpu().numpy()
                    chunk_actions = [
                        _denormalize_action(
                            a, stats, cfg, obs_history[-1].state.astype(np.float32)
                        )
                        for a in pred[: cfg.policy.n_action_steps]
                    ]
                    chunk_idx = 0

                t0 = time.perf_counter()
                action = chunk_actions[chunk_idx]
                chunk_idx += 1
                step = env.step(action)
                _apply_step(step)

                frame = env.render_rgb()
                if frame is not None:
                    frames.append(frame)
                _pace_step(t0, pace_fps)

        success = max_coverage >= success_thresh
        successes += int(success)
        rewards.append(ep_reward)
        max_coverages.append(max_coverage)

        if cfg.eval.save_video and frames:
            out = video_dir / f"episode_{ep:03d}.mp4"
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps, (w, h))
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()

    env.close()
    return {
        "success_rate": successes / max(cfg.eval.num_episodes, 1),
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_max_coverage": float(np.mean(max_coverages)) if max_coverages else 0.0,
        "max_coverages": max_coverages,
        "episode_rewards": rewards,
    }
