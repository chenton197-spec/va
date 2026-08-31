#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "eval_episode_openloop", BASE / "scripts" / "eval_episode_openloop.py"
)
ev = importlib.util.module_from_spec(spec)
sys.modules["eval_episode_openloop"] = ev
spec.loader.exec_module(ev)

from robotfm.collect.loop import get_run_dir
from robotfm.config import load_config
from robotfm.data.action_delta import denormalize_predicted_action, flow_history_from_phys, joint_mask_from_names
from robotfm.data.lerobot_dataset import (
    _depth_rel_from_image_rel,
    _load_depth_sources,
    _load_packed_depth,
    _short_camera_name,
    load_episode_arrays_from_parquet,
    load_lerobot_info,
)
from robotfm.data.stats import normalize
from robotfm.data.uint8_cache import Uint8ImageCache, resolve_cache_dir
from robotfm.train import build_policy

RUNS = [
    (
        "statedelta",
        BASE
        / "outputs/fm_openarm_hcx_dual_arm_with_out_room_s_depth_nobs1_h60_nact16_statedelta_260830124458/checkpoint_best_val.pt",
    ),
    (
        "reachw",
        BASE
        / "outputs/fm_openarm_hcx_dual_arm_with_out_room_s_depth_nobs1_h60_nact16_statedelta_reachw_260830213549/checkpoint_best_val.pt",
    ),
]
EPISODES = [30, 37]
OUT_ROOT = BASE / "outputs/eval_joint_error_reachw_260831"
BATCH_SIZE = 4


def _plot_error(steps, err, names, mae, title, path):
    n_dim = len(names)
    ncols = 2
    nrows = int(np.ceil(n_dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(3.0 * nrows, 6.0)), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for i, name in enumerate(names):
        ax = axes[i]
        ax.plot(steps, err[:, i], color="#d62728", linewidth=0.9)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{name}  MAE={mae[i]:.4f}")
    for j in range(n_dim, len(axes)):
        axes[j].set_visible(False)
    for ax in axes[max(0, n_dim - ncols) : n_dim]:
        ax.set_xlabel("frame")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_compare(steps, errs, maes, names, title, path):
    n_dim = len(names)
    ncols = 2
    nrows = int(np.ceil(n_dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(3.0 * nrows, 6.0)), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    colors = {"statedelta": "#d62728", "reachw": "#2ca02c"}
    labels = {"statedelta": "statedelta", "reachw": "reachw"}
    for i, name in enumerate(names):
        ax = axes[i]
        for tag, err in errs.items():
            ax.plot(
                steps,
                err[:, i],
                color=colors[tag],
                linewidth=0.9,
                label=f"{labels[tag]} MAE={maes[tag][i]:.3f}",
                alpha=0.85,
            )
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.set_title(name)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    for j in range(n_dim, len(axes)):
        axes[j].set_visible(False)
    for ax in axes[max(0, n_dim - ncols) : n_dim]:
        ax.set_xlabel("frame")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _depth_meta(run_dir: Path, depth_cameras: list[str]):
    sources = _load_depth_sources(run_dir)
    meta = {}
    for camera in depth_cameras:
        feature = f"observation.depth.{camera}"
        src = sources[feature]
        meta[camera] = {
            "scale": float(src["scale_mm_per_raw_unit"]),
            "invalid_raw": int(src.get("invalid_raw_value", src.get("invalid_value", 0))),
        }
    return meta


def _load_rgb_idx(cache: Uint8ImageCache, ep_local: int, cameras: list[str], idx: np.ndarray) -> torch.Tensor:
    b, n_obs = idx.shape
    flat = idx.reshape(-1).tolist()
    cams = []
    for cam in cameras:
        hwc = cache.load_cam_frames(ep_local, cam, flat)
        chw = np.transpose(hwc, (0, 3, 1, 2))
        cams.append(chw.reshape(b, n_obs, 3, chw.shape[2], chw.shape[3]))
    return torch.from_numpy(np.stack(cams, axis=1))


def _load_depth_idx(
    run_dir: Path,
    image_paths: dict[str, list[str]],
    depth_cameras: list[str],
    meta: dict,
    idx: np.ndarray,
    image_size,
    min_mm: float,
    max_mm: float,
) -> torch.Tensor:
    b, n_obs = idx.shape
    cams = []
    for camera in depth_cameras:
        m = meta[camera]
        paths = image_paths[camera]
        batch = []
        for bi in range(b):
            frames = []
            for fi in idx[bi].tolist():
                rel = _depth_rel_from_image_rel(paths[int(fi)], camera)
                frames.append(
                    torch.from_numpy(
                        _load_packed_depth(
                            run_dir / rel,
                            scale_mm=m["scale"],
                            invalid_raw=m["invalid_raw"],
                            min_mm=min_mm,
                            max_mm=max_mm,
                            image_size=image_size,
                        )
                    )
                )
            batch.append(torch.stack(frames, dim=0))
        cams.append(torch.stack(batch, dim=0))
    return torch.stack(cams, dim=1)


def _infer(
    *,
    policy,
    cache: Uint8ImageCache,
    ep_local: int,
    cameras: list[str],
    states: np.ndarray,
    stats: dict,
    cfg,
    device: torch.device,
    dest_offset: int,
    run_dir: Path,
    image_paths: dict[str, list[str]],
    depth_cameras: list[str],
    dmeta: dict,
):
    action_names = ev._action_names_from_cfg(cfg)
    n_obs = cfg.dataset.n_obs_steps
    n_action_steps = int(cfg.policy.n_action_steps)
    exec_steps = n_action_steps
    predict_joint_delta = bool(cfg.policy.predict_joint_delta)
    delta_joint_mask = joint_mask_from_names(action_names, int(cfg.action_dim))
    length = int(states.shape[0])
    action_dim = int(states.shape[1])
    infer_ts = list(range(0, length, exec_steps))
    n_infer = len(infer_ts)
    obs_idx = np.array([ev._obs_indices(t, n_obs) for t in infer_ts], dtype=np.int64)
    pred = np.full((length, action_dim), np.nan, dtype=np.float32)
    bs = BATCH_SIZE
    dest_offset = int(dest_offset)
    t0 = time.perf_counter()
    for start in range(0, n_infer, bs):
        sl = slice(start, min(start + bs, n_infer))
        ts = infer_ts[sl]
        idx = obs_idx[sl]
        obs_images = _load_rgb_idx(cache, ep_local, cameras, idx).to(device, non_blocking=True)
        if obs_images.dtype == torch.uint8:
            obs_images = obs_images.float().div_(255.0)
        state_win = states[idx].astype(np.float32)
        state_norm = normalize(state_win, stats, prefix="state", mode=cfg.dataset.norm_mode)
        if predict_joint_delta:
            hist = np.stack(
                [
                    flow_history_from_phys(
                        states[row].astype(np.float32),
                        stats,
                        cfg.dataset.norm_mode,
                        predict_joint_delta=True,
                        joint_mask=delta_joint_mask,
                    )
                    for row in idx
                ],
                axis=0,
            )
        else:
            hist = state_norm
        obs_depth = _load_depth_idx(
            run_dir,
            image_paths,
            depth_cameras,
            dmeta,
            idx,
            cfg.dataset.image_size,
            float(cfg.dataset.depth_min_mm),
            float(cfg.dataset.depth_max_mm),
        ).to(device, non_blocking=True)
        batch = {
            "obs_images": obs_images,
            "obs_state": torch.from_numpy(state_norm).to(device, non_blocking=True),
            "obs_history": torch.from_numpy(np.ascontiguousarray(hist)).to(device, non_blocking=True),
            "obs_depth": obs_depth,
        }
        with torch.no_grad():
            pred_norm = policy.sample_actions(batch)[:, :n_action_steps].float().cpu()
        q_now = states[np.asarray(ts, dtype=np.int64)].astype(np.float32)
        pred_phys = np.asarray(
            denormalize_predicted_action(
                pred_norm,
                stats,
                cfg.dataset.norm_mode,
                q_now_phys=q_now,
                predict_joint_delta=predict_joint_delta,
                joint_mask=delta_joint_mask,
            )
        )
        for i, t in enumerate(ts):
            dest = int(t) + dest_offset
            if dest >= length:
                continue
            take = min(exec_steps, length - dest, pred_phys.shape[1])
            if take <= 0:
                continue
            pred[dest : dest + take] = np.asarray(pred_phys[i, :take], dtype=np.float32)
        if start == 0 or (start // bs) % 20 == 0:
            done = min(start + bs, n_infer)
            print(f"  infer {done}/{n_infer} {time.perf_counter() - t0:.1f}s", flush=True)
    return pred, action_names, n_action_steps


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print(f"device={device} batch_size={BATCH_SIZE}", flush=True)

    cfg0 = load_config(ev._resolve_train_config(RUNS[0][1], None, BASE))
    run_dir = get_run_dir(cfg0, BASE)
    info = load_lerobot_info(run_dir)
    cameras = list(cfg0.cameras)
    depth_cameras = list(cfg0.dataset.depth_cameras)
    dmeta = _depth_meta(run_dir, depth_cameras)
    cam_feat_keys = [
        next(k for k in info["features"] if _short_camera_name(k) == cam) for cam in cameras
    ]
    cache_dir = resolve_cache_dir(run_dir, image_size=cfg0.dataset.image_size)
    cache = Uint8ImageCache(cache_dir)
    print(f"uint8_cache {cache_dir}", flush=True)

    for ep in EPISODES:
        payload = load_episode_arrays_from_parquet(run_dir, ep, info, cam_feat_keys)
        states = payload["state"]
        image_paths = payload["image_paths"]
        length = int(payload["length"])
        ep_local = cache.episode_ids.index(ep)
        print(f"\n=== episode {ep} length={length} ===", flush=True)

        errs = {}
        maes = {}
        names_ref = None
        steps = np.arange(length)
        for tag, ckpt_path in RUNS:
            print(f"load {tag}", flush=True)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = load_config(ev._resolve_train_config(ckpt_path, None, BASE))
            stats = ckpt["stats"]
            state_dict = ckpt["policy_state_dict"]
            del ckpt
            dest_offset = 1 if bool(getattr(cfg.policy, "predict_state_delta", False)) else 0
            policy = build_policy(cfg, stats)
            policy.load_state_dict(state_dict)
            del state_dict
            policy.to(device)
            policy.eval()
            t_inf = time.perf_counter()
            pred, action_names, n_action_steps = _infer(
                policy=policy,
                cache=cache,
                ep_local=ep_local,
                cameras=cameras,
                states=states,
                stats=stats,
                cfg=cfg,
                device=device,
                dest_offset=dest_offset,
                run_dir=run_dir,
                image_paths=image_paths,
                depth_cameras=depth_cameras,
                dmeta=dmeta,
            )
            infer_s = time.perf_counter() - t_inf
            del policy, stats
            if device.type == "cuda":
                torch.cuda.empty_cache()
            names_ref = action_names
            valid = np.isfinite(pred).all(axis=1)
            err = pred - states
            mae = np.mean(np.abs(err[valid]), axis=0)
            gripper_mask = np.array(["gripper" in n.lower() for n in action_names], dtype=bool)
            joint_mask = ~gripper_mask
            mae_j = float(np.mean(mae[joint_mask]))
            mae_g = float(np.mean(mae[gripper_mask]))
            print(
                f"{tag} infer={infer_s:.1f}s dest_offset={dest_offset} "
                f"nact={n_action_steps} mae_j={mae_j:.4f} mae_g={mae_g:.4f}",
                flush=True,
            )
            out_dir = OUT_ROOT / tag / f"ep{ep:04d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_dir / "pred_vs_state.npz",
                pred=pred,
                gt=states,
                err=err,
                action_names=np.asarray(action_names),
            )
            metrics = {
                "tag": tag,
                "checkpoint": str(ckpt_path),
                "episode": ep,
                "gt": "state",
                "dest_offset": dest_offset,
                "n_action_steps": n_action_steps,
                "mae_per_dim": {n: float(v) for n, v in zip(action_names, mae)},
                "mae_joints": mae_j,
                "mae_gripper": mae_g,
            }
            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            _plot_error(
                steps,
                err,
                action_names,
                mae,
                f"ep{ep} pred−state  |  {tag}  |  mae_j={mae_j:.4f} mae_g={mae_g:.4f}",
                out_dir / "joint_error.png",
            )
            errs[tag] = err
            maes[tag] = mae

        cmp_dir = OUT_ROOT / "compare" / f"ep{ep:04d}"
        cmp_dir.mkdir(parents=True, exist_ok=True)
        _plot_compare(
            steps,
            errs,
            maes,
            names_ref,
            f"ep{ep} pred−state  statedelta vs reachw",
            cmp_dir / "joint_error_compare.png",
        )
        print(f"saved compare {cmp_dir / 'joint_error_compare.png'}", flush=True)
        del payload, states, errs, maes
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\ndone: {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
