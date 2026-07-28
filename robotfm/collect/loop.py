"""遥操作采集主循环。

与具体环境、具体 teleop 驱动解耦：
- env:    BaseEnv 实现（PushT / 真机）
- driver: TeleopDriver 实现（鼠标 / 手柄 / 键盘）

循环按 cfg.fps 限速，将 (obs, action, reward, done) 逐帧缓存，
episode 结束或用户按 S 时写入 NPZ，最后计算 stats.json。
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np

from robotfm.collect.drivers.base import TeleopDriver
from robotfm.config import RobotFMConfig, resolve_path
from robotfm.data.stats import compute_stats, save_stats
from robotfm.data.writer import EpisodeWriter
from robotfm.envs.base import BaseEnv
from robotfm.types import EpisodeMeta


def timestamped_run_name(run_name: str, when: datetime | None = None) -> str:
    """为 run_name 追加 YYMMDDHHMM 时间后缀，避免多次采集互相覆盖。"""
    stamp = (when or datetime.now()).strftime("%y%m%d%H%M")
    return f"{run_name}_{stamp}"


def get_run_dir(cfg: RobotFMConfig, base_dir: Path) -> Path:
    """数据 run 的完整路径：base_dir / data_root / dataset.run_name。"""
    data_root = resolve_path(base_dir, cfg.data_root)
    return data_root / cfg.dataset.run_name


def _stack_episode(
    frames: list[dict[str, np.ndarray | float | bool]],
    camera_names: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """将逐帧 dict 列表堆叠为整条轨迹的 numpy 数组。"""
    images = {cam: np.stack([f["images"][cam] for f in frames], axis=0) for cam in camera_names}
    state = np.stack([f["state"] for f in frames], axis=0).astype(np.float32)
    action = np.stack([f["action"] for f in frames], axis=0).astype(np.float32)
    reward = np.asarray([f["reward"] for f in frames], dtype=np.float32)
    done = np.asarray([f["done"] for f in frames], dtype=bool)
    success = bool(frames[-1].get("success", False))
    return images, state, action, reward, done, success


def collect_demos(
    cfg: RobotFMConfig,
    env: BaseEnv,
    driver: TeleopDriver,
    base_dir: Path,
) -> Path:
    """主采集循环，直到存满 target_episodes 条或用户退出。

    控制说明（由 driver.poll_events 解析）:
        鼠标: teleop 目标动作
        R:    丢弃当前 episode，重新 reset
        S:    提前保存当前 episode
        Q/Esc: 退出采集

    返回:
        run_dir 路径
    """
    cfg.dataset.run_name = timestamped_run_name(cfg.dataset.run_name)
    run_dir = get_run_dir(cfg, base_dir)
    meta = EpisodeMeta(
        backend=cfg.backend,
        embodiment=cfg.embodiment,
        fps=cfg.fps,
        cameras=cfg.camera_specs,
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        state_names=cfg.state_names,
        action_names=cfg.action_names,
        task=cfg.collect.task,
    )
    writer = EpisodeWriter(run_dir, meta)

    saved = 0
    episode_index = 0
    seed = 0
    pending_save = False  # 用户按 S 后下一帧结束即保存
    frames: list[dict] = []

    obs = env.reset(seed=seed)
    driver.on_reset()
    dt = 1.0 / cfg.fps
    last_time = time.time()

    print("Controls: mouse=teleop, R=reset/discard, S=save episode, Q=quit")

    while saved < cfg.collect.target_episodes:
        token = driver.poll_events()
        if token == "quit":
            break
        if token == "reset":
            frames.clear()
            seed += 1
            obs = env.reset(seed=seed)
            driver.on_reset()
            pending_save = False
            continue
        if token == "save":
            pending_save = True

        action = driver.get_action(obs)
        if action is None:
            time.sleep(dt)
            continue

        # BC 数据约定：存 (o_t, a_t)，即在 step 之前的观测与即将执行的动作
        images = {name: obs.images[name].copy() for name in env.observation_cameras}
        frames.append(
            {
                "images": images,
                "state": obs.state.copy(),
                "action": action.copy(),
            }
        )

        step = env.step(action)
        obs = step.observation

        # 把 step 后的 reward/done/success 写回最后一帧
        if frames:
            frames[-1]["reward"] = step.reward
            frames[-1]["done"] = step.done
            frames[-1]["success"] = step.info.get("success", step.terminated)

        # 按 fps 限速
        elapsed = time.time() - last_time
        if elapsed < dt:
            time.sleep(dt - elapsed)
        last_time = time.time()

        if step.done or pending_save:
            # gym_pusht 用 is_success；部分环境用 success
            success = bool(step.info.get("success", step.info.get("is_success", step.terminated)))
            if cfg.collect.save_all or success or pending_save:
                images, state, action_arr, reward, done, _ = _stack_episode(frames, env.observation_cameras)
                writer.write_episode(
                    episode_index=episode_index,
                    images=images,
                    state=state,
                    action=action_arr,
                    reward=reward,
                    done=done,
                    success=success,
                    task=cfg.collect.task,
                )
                saved += 1
                episode_index += 1
                print(f"Saved episode {episode_index - 1} (success={success}). Total saved: {saved}")
            else:
                print("Episode discarded (not successful). Press R to retry.")

            frames.clear()
            pending_save = False
            seed += 1
            obs = env.reset(seed=seed)
            driver.on_reset()

    if saved > 0:
        stats = compute_stats(run_dir)
        save_stats(run_dir, stats)
    env.close()
    return run_dir
