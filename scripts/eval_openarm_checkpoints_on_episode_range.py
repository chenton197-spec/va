#!/usr/bin/env python3
"""
Batch open-loop offline eval (pred vs GT actions) for multiple checkpoints.

This is a trimmed version of `eval_episode_openloop.py`:
- Computes MAE/RMSE metrics for an episode index range.
- Avoids per-episode plots / npz to reduce disk + wall time.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from robotfm.collect.loop import get_run_dir
from robotfm.config import load_config
from robotfm.data.dataset import spatial_preprocess_images
from robotfm.data.action_delta import denormalize_predicted_action, flow_history_from_phys, joint_mask_from_names
from robotfm.data.lerobot_dataset import load_episode_arrays_from_parquet, load_lerobot_info
from robotfm.data.stats import normalize
from robotfm.train import build_policy


def _action_names_from_cfg(cfg: Any) -> list[str]:
    names = list(cfg.action_names)
    if len(names) == int(cfg.action_dim):
        return names
    # YAML often omits names; defaults stay PushT ["x","y"] while action_dim=7.
    return [f"j{i + 1}" for i in range(int(cfg.action_dim) - 1)] + ["gripper"]


def _resolve_train_config(ckpt_path: Path, config_arg: str | None, base_dir: Path) -> Path | None:
    """Resolve training YAML: CLI > config_source.yaml > config.yaml beside ckpt."""
    if config_arg:
        p = Path(config_arg)
        if not p.is_absolute():
            p = base_dir / p
        return p.resolve()
    for name in ("config_source.yaml", "config.yaml"):
        cand = ckpt_path.parent / name
        if cand.is_file():
            return cand
    return None


@dataclass(frozen=True)
class EpisodeMetric:
    mae_joints: float
    mae_gripper: float
    mae_all: float
    rmse_joints: float
    rmse_gripper: float
    rmse_all: float
    n_valid_steps: int


def _parse_ckpt_label(ckpt_path: Path) -> str:
    m = re.search(r"checkpoint_(\d+)\.pt$", ckpt_path.name)
    if m:
        return m.group(1)
    if ckpt_path.name == "checkpoint_latest.pt":
        return "latest"
    return ckpt_path.name


def _mae_rmse(pred_arr: np.ndarray, gt_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_m = np.isfinite(pred_arr).all(axis=1)
    err_m = pred_arr[valid_m] - gt_arr[valid_m]
    mae = np.mean(np.abs(err_m), axis=0)
    rmse = np.sqrt(np.mean(err_m**2, axis=0))
    return mae, rmse, valid_m


def evaluate_one_checkpoint(
    *,
    ckpt_path: Path,
    episode_indices: Iterable[int],
    episode_eval_dir: Path,
    config_arg: str | None,
    base_dir: Path,
    run_dir_override: Path | None,
) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_cfg_path = _resolve_train_config(ckpt_path, config_arg, base_dir)
    if train_cfg_path is not None:
        cfg = load_config(train_cfg_path)
    else:
        cfg = ckpt["config"]

    stats = ckpt["stats"]
    norm_mode = cfg.dataset.norm_mode

    n_obs = cfg.dataset.n_obs_steps
    horizon = cfg.dataset.horizon
    n_action_steps = int(cfg.policy.n_action_steps)
    exec_steps = n_action_steps
    num_inference_steps = int(cfg.policy.num_inference_steps)
    history_noise_std = float(cfg.policy.history_noise_std)

    action_names = _action_names_from_cfg(cfg)
    cameras = list(cfg.cameras)
    delta_joint_mask = joint_mask_from_names(action_names, int(cfg.action_dim))
    reanchor = False

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    policy = build_policy(cfg, stats)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()

    run_dir = run_dir_override if run_dir_override is not None else get_run_dir(cfg, base_dir)
    info = load_lerobot_info(run_dir)
    features = info["features"]
    # In practice, eval_episode_openloop uses `_short_camera_name` mapping.
    # `features` keys look like: observation.images.<short>.
    from robotfm.data.lerobot_dataset import _short_camera_name  # local import to avoid circular

    cam_feat_keys = [next(k for k in features if _short_camera_name(k) == cam) for cam in cameras]

    episode_metrics: list[EpisodeMetric] = []
    episode_details: list[dict[str, Any]] = []

    # Reuse eval_episode_openloop's `_build_obs_batch` so image decoding +
    # preprocessing match exactly.
    import importlib.util

    eval_script_path = base_dir / "scripts" / "eval_episode_openloop.py"
    spec = importlib.util.spec_from_file_location("eval_episode_openloop", eval_script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {eval_script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _orig_build_obs_batch = getattr(mod, "_build_obs_batch")

    for ep in episode_indices:
        payload = load_episode_arrays_from_parquet(run_dir, ep, info, cam_feat_keys)
        states = payload["state"]
        actions_gt = payload["action"]
        image_paths = payload["image_paths"]
        length = int(payload["length"])

        pred_actions = np.full_like(actions_gt, np.nan, dtype=np.float32)
        replan_ts: list[int] = []

        t = 0
        while t < length:
            batch = _orig_build_obs_batch(
                run_dir=run_dir,
                image_paths=image_paths,
                states=states,
                cameras=cameras,
                t=t,
                n_obs_steps=n_obs,
                pre_crop_size=cfg.dataset.pre_crop_size,
                resize_size=cfg.dataset.resize_size,
                crop_size=cfg.dataset.crop_size,
                stats=stats,
                norm_mode=norm_mode,
                device=device,
            )
            with torch.no_grad():
                pred_norm = policy.sample_actions(batch)[0].cpu()
            q_now = states[t].astype(np.float32)
            pred_phys = np.asarray(
                denormalize_predicted_action(
                    pred_norm,
                    stats,
                    norm_mode,
                    q_now_phys=q_now,
                    predict_joint_delta=bool(cfg.policy.predict_joint_delta),
                    joint_mask=delta_joint_mask,
                )
            )

            take = min(exec_steps, length - t, pred_phys.shape[0])
            chunk_raw = pred_phys[:take]
            if reanchor:
                chunk = np.array(chunk_raw, dtype=np.float32, copy=True)
                offset = q_now[delta_joint_mask] - chunk[0, delta_joint_mask]
                chunk[:, delta_joint_mask] = chunk[:, delta_joint_mask] + offset
                pred_actions[t : t + take] = chunk
            else:
                pred_actions[t : t + take] = chunk_raw

            replan_ts.append(t)
            t += take

        # Metrics
        mae, rmse, valid_m = _mae_rmse(pred_actions, actions_gt)

        gripper_mask = np.array(["gripper" in n.lower() for n in action_names], dtype=bool)
        if not gripper_mask.any() and len(mae) >= 7:
            gripper_mask = np.zeros(len(mae), dtype=bool)
            gripper_mask[-1] = True
        joint_mask = ~gripper_mask

        n_valid_steps = int(valid_m.sum())

        mae_joints = float(np.mean(mae[joint_mask])) if joint_mask.any() else float("nan")
        mae_gripper = float(np.mean(mae[gripper_mask])) if gripper_mask.any() else float("nan")
        mae_all = float(np.mean(mae))
        rmse_joints = float(np.mean(rmse[joint_mask])) if joint_mask.any() else float("nan")
        rmse_gripper = float(np.mean(rmse[gripper_mask])) if gripper_mask.any() else float("nan")
        rmse_all = float(np.mean(rmse))

        metric = EpisodeMetric(
            mae_joints=mae_joints,
            mae_gripper=mae_gripper,
            mae_all=mae_all,
            rmse_joints=rmse_joints,
            rmse_gripper=rmse_gripper,
            rmse_all=rmse_all,
            n_valid_steps=n_valid_steps,
        )
        episode_metrics.append(metric)

        episode_details.append(
            {
                "episode": ep,
                "length": length,
                "n_valid_steps": n_valid_steps,
                "n_replans": len(replan_ts),
                "mae_joints": mae_joints,
                "mae_gripper": mae_gripper,
                "mae_all": mae_all,
                "rmse_joints": rmse_joints,
                "rmse_gripper": rmse_gripper,
                "rmse_all": rmse_all,
                "mae_per_dim": {n: float(v) for n, v in zip(action_names, mae)},
                "rmse_per_dim": {n: float(v) for n, v in zip(action_names, rmse)},
            }
        )

    # Aggregate
    def _mean(xs: list[EpisodeMetric], f: str) -> float:
        return float(np.mean([getattr(x, f) for x in xs])) if xs else float("nan")

    agg = {
        "checkpoint": str(ckpt_path),
        "ckpt_label": _parse_ckpt_label(ckpt_path),
        "run_dir": str(run_dir),
        "episodes": [int(x) for x in episode_indices],
        "n_episodes": len(list(episode_indices)),
        "mae_joints_mean": _mean(episode_metrics, "mae_joints"),
        "mae_gripper_mean": _mean(episode_metrics, "mae_gripper"),
        "mae_all_mean": _mean(episode_metrics, "mae_all"),
        "rmse_joints_mean": _mean(episode_metrics, "rmse_joints"),
        "rmse_gripper_mean": _mean(episode_metrics, "rmse_gripper"),
        "rmse_all_mean": _mean(episode_metrics, "rmse_all"),
    }

    episode_eval_dir.mkdir(parents=True, exist_ok=True)
    (episode_eval_dir / "episode_details.json").write_text(json.dumps(episode_details, indent=2) + "\n")
    (episode_eval_dir / "metrics_summary.json").write_text(json.dumps(agg, indent=2) + "\n")
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate openarm checkpoints on episode index range")
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        required=True,
        help="checkpoint directory containing checkpoint_*.pt and/or checkpoint_latest.pt",
    )
    parser.add_argument("--ckpt-glob", type=str, default="checkpoint_*.pt", help="glob pattern relative to ckpt-dir")
    parser.add_argument("--include-latest", action="store_true", help="also evaluate checkpoint_latest.pt if present")
    parser.add_argument("--config", type=str, default=None, help="override training config YAML (optional)")
    parser.add_argument("--run-dir", type=str, default=None, help="override dataset root run_dir")
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Comma-separated episode indices, e.g. '15,85'. If set, ignores --episode-start/--episode-end.",
    )
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="default: <ckpt-dir>/eval_epXXXX-YYYY_sweep")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = (base_dir / ckpt_dir).resolve()

    run_dir_override = Path(args.run_dir).resolve() if args.run_dir else None

    if args.episodes:
        episode_indices = [int(x.strip()) for x in args.episodes.split(",") if x.strip()]
    else:
        if args.episode_start is None or args.episode_end is None:
            raise ValueError("Need either --episodes or both --episode-start and --episode-end")
        episode_indices = list(range(int(args.episode_start), int(args.episode_end) + 1))

    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else ckpt_dir
        / (
            f"eval_eps_{args.episodes.replace(',','_')}"
            if args.episodes
            else f"eval_ep{args.episode_start:04d}-{args.episode_end:04d}_range"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(ckpt_dir.glob(args.ckpt_glob))
    if args.include_latest:
        latest = ckpt_dir / "checkpoint_latest.pt"
        if latest.is_file():
            ckpts = [latest] + [p for p in ckpts if p != latest]

    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched in {ckpt_dir} (glob={args.ckpt_glob})")

    results: list[dict[str, Any]] = []
    for ckpt_path in ckpts:
        label = _parse_ckpt_label(ckpt_path)
        episode_eval_dir = out_dir / f"eval_{label}"
        agg = evaluate_one_checkpoint(
            ckpt_path=ckpt_path,
            episode_indices=episode_indices,
            episode_eval_dir=episode_eval_dir,
            config_arg=args.config,
            base_dir=base_dir,
            run_dir_override=run_dir_override,
        )
        results.append(agg)

    # Sort results by numeric ckpt step when possible.
    def _sort_key(x: dict[str, Any]) -> float:
        lab = x.get("ckpt_label", "")
        if lab.isdigit():
            return float(lab)
        return float("inf") if lab != "latest" else -1.0

    results = sorted(results, key=_sort_key)
    (out_dir / "checkpoints_summary.json").write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(f"Saved summary: {out_dir / 'checkpoints_summary.json'}")


if __name__ == "__main__":
    main()

