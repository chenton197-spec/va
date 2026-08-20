#!/usr/bin/env python3
"""Batched teacher-forced action MSE on a LeRobot val split.

CPU workers decode images; GPU (or CPU) runs policy.sample_actions.
Reports normalized-space MSE plus physical-unit MSE / MAE / RMSE.

Usage (from va/ with conda env lerobot)::

    PYTHONPATH=. python scripts/eval_val_mse.py \\
      --checkpoint model/dadi/outputs/.../checkpoint_final.pt \\
      --run-dir data/openarm_hcx_dual_arm_val \\
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from robotfm.config import load_config
from robotfm.data.action_delta import denormalize_predicted_action, joint_mask_from_names
from robotfm.data.dataset import build_episode_dataset, images_to_float01
from robotfm.data.stats import denormalize
from robotfm.train import build_policy


def _resolve_train_config(ckpt_path: Path, config_arg: str | None, base_dir: Path) -> Path | None:
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


def _resolve_checkpoint(raw: str, base_dir: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if path.is_dir():
        for name in ("checkpoint_final.pt", "checkpoint_latest.pt"):
            cand = path / name
            if cand.is_file():
                return cand
        numbered = sorted(path.glob("checkpoint_*.pt"))
        numbered = [p for p in numbered if p.name not in {"checkpoint_final.pt", "checkpoint_latest.pt"}]
        if numbered:
            return numbered[-1]
        raise FileNotFoundError(f"No checkpoint_*.pt under {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _action_names(cfg) -> list[str]:
    names = list(cfg.action_names)
    if len(names) == int(cfg.action_dim):
        return names
    return [f"j{i + 1}" for i in range(int(cfg.action_dim) - 1)] + ["gripper"]


def _masks(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    grip = np.array(["gripper" in n.lower() for n in names], dtype=bool)
    if not grip.any() and len(names) >= 7:
        grip = np.zeros(len(names), dtype=bool)
        grip[-1] = True
    return ~grip, grip


def _pick_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] / (1024**2)
        device = torch.device("cuda" if free >= 2500 else "cpu")
        print(f"cuda_free_mib={free:.0f} -> device={device}")
        return device
    return torch.device("cpu")


def _summarize_err(
    err: np.ndarray,
    names: list[str],
    joint_mask: np.ndarray,
    grip_mask: np.ndarray,
) -> dict:
    """err: (N, T, A) — already masked to valid steps."""
    sq = err * err
    abs_e = np.abs(err)
    mse_dim = sq.mean(axis=(0, 1))
    mae_dim = abs_e.mean(axis=(0, 1))
    rmse_dim = np.sqrt(mse_dim)
    out = {
        "mse_per_dim": {n: float(v) for n, v in zip(names, mse_dim)},
        "mae_per_dim": {n: float(v) for n, v in zip(names, mae_dim)},
        "rmse_per_dim": {n: float(v) for n, v in zip(names, rmse_dim)},
        "mse_all": float(sq.mean()),
        "mae_all": float(abs_e.mean()),
        "rmse_all": float(np.sqrt(sq.mean())),
    }
    if joint_mask.any():
        out["mse_joints"] = float(sq[..., joint_mask].mean())
        out["mae_joints"] = float(abs_e[..., joint_mask].mean())
        out["rmse_joints"] = float(np.sqrt(sq[..., joint_mask].mean()))
    if grip_mask.any():
        out["mse_gripper"] = float(sq[..., grip_mask].mean())
        out["mae_gripper"] = float(abs_e[..., grip_mask].mean())
        out["rmse_gripper"] = float(np.sqrt(sq[..., grip_mask].mean()))
    return out


def _list_sweep_checkpoints(ckpt_dir: Path) -> list[Path]:
    """Numbered ckpts in step order, then latest/final if present."""
    numbered: list[tuple[int, Path]] = []
    for path in ckpt_dir.glob("checkpoint_*.pt"):
        if path.name in {"checkpoint_latest.pt", "checkpoint_final.pt"}:
            continue
        stem = path.stem.replace("checkpoint_", "")
        if stem.isdigit():
            numbered.append((int(stem), path))
    numbered.sort()
    out = [path for _step, path in numbered]
    for name in ("checkpoint_latest.pt", "checkpoint_final.pt"):
        cand = ckpt_dir / name
        if cand.is_file():
            out.append(cand)
    return out


def _subset_indices_for_episodes(dataset, episode_ids: set[int] | None) -> list[int]:
    if not episode_ids:
        return list(range(len(dataset)))
    keep: list[int] = []
    for i, (ep_local, _t) in enumerate(dataset.index):
        if int(dataset._episode_ids[ep_local]) in episode_ids:
            keep.append(i)
    if not keep:
        raise ValueError(f"No windows for episodes {sorted(episode_ids)}")
    return keep


@torch.no_grad()
def _eval_policy(
    *,
    policy,
    loader: DataLoader,
    dataset,
    keep_idx: list[int],
    device: torch.device,
    stats: dict,
    cfg,
    names: list[str],
    joint_mask: np.ndarray,
    grip_mask: np.ndarray,
    delta_joint_mask: np.ndarray,
    horizon: int,
    predict_joint_delta: bool,
) -> tuple[dict, dict, list[dict], int, float]:
    err_norm_chunks: list[np.ndarray] = []
    err_phys_chunks: list[np.ndarray] = []
    ep_err_norm: dict[int, list[np.ndarray]] = defaultdict(list)
    ep_err_phys: dict[int, list[np.ndarray]] = defaultdict(list)
    n_windows = 0
    cursor = 0
    t_inf = time.time()
    first_batch = True

    for batch in loader:
        obs_images = images_to_float01(batch["obs_images"].to(device, non_blocking=True))
        obs_state = batch["obs_state"].to(device, non_blocking=True)
        obs_history = batch["obs_history"].to(device, non_blocking=True)
        gt_norm = batch["action"].to(device, non_blocking=True)
        mask = batch["action_mask"].to(device, non_blocking=True)
        pred_norm = policy.sample_actions(
            {
                "obs_images": obs_images,
                "obs_state": obs_state,
                "obs_history": obs_history,
            }
        )
        pred_norm = pred_norm[:, :horizon]
        q_now = denormalize(
            obs_state[:, -1], stats, prefix="state", mode=cfg.dataset.norm_mode
        )
        pred_phys = denormalize_predicted_action(
            pred_norm,
            stats,
            cfg.dataset.norm_mode,
            q_now_phys=q_now,
            predict_joint_delta=predict_joint_delta,
            joint_mask=delta_joint_mask,
        )
        gt_phys = denormalize(gt_norm, stats, prefix="action", mode=cfg.dataset.norm_mode)
        if predict_joint_delta:
            from robotfm.data.action_delta import add_joint_pose

            gt_phys = add_joint_pose(gt_phys, q_now, delta_joint_mask)
        err_n = ((pred_norm - gt_norm) * mask).cpu().numpy()
        err_p = ((pred_phys - gt_phys) * mask).cpu().numpy()
        valid = mask.cpu().numpy()
        if first_batch:
            extra = ""
            if device.type == "cuda":
                extra = f"  vram={torch.cuda.memory_allocated() / 1024**2:.0f}MiB"
            print(f"  first_batch ok  b={err_n.shape[0]}  {time.time() - t_inf:.1f}s{extra}")
            first_batch = False
        bsz = err_n.shape[0]
        for i in range(bsz):
            ds_i = keep_idx[cursor + i]
            ep_local, _t = dataset.index[ds_i]
            ep_id = int(dataset._episode_ids[ep_local])
            rows = valid[i, :, 0] > 0.5
            if not rows.any():
                continue
            e_n = err_n[i, rows]
            e_p = err_p[i, rows]
            err_norm_chunks.append(e_n)
            err_phys_chunks.append(e_p)
            ep_err_norm[ep_id].append(e_n)
            ep_err_phys[ep_id].append(e_p)
            n_windows += 1
        cursor += bsz

    elapsed = time.time() - t_inf
    err_norm = np.concatenate(err_norm_chunks, axis=0)[:, None, :]
    err_phys = np.concatenate(err_phys_chunks, axis=0)[:, None, :]
    metrics_norm = _summarize_err(err_norm, names, joint_mask, grip_mask)
    metrics_phys = _summarize_err(err_phys, names, joint_mask, grip_mask)
    per_ep = []
    for ep_id in sorted(ep_err_phys):
        e_n = np.concatenate(ep_err_norm[ep_id], axis=0)[:, None, :]
        e_p = np.concatenate(ep_err_phys[ep_id], axis=0)[:, None, :]
        per_ep.append(
            {
                "episode": ep_id,
                "n_steps": int(e_p.shape[0]),
                "norm": _summarize_err(e_n, names, joint_mask, grip_mask),
                "phys": _summarize_err(e_p, names, joint_mask, grip_mask),
            }
        )
    return metrics_norm, metrics_phys, per_ep, n_windows, elapsed


def _hard_easy_split(per_ep: list[dict], hard_eps: set[int]) -> dict:
    def _weighted(rows: list[dict], key: str) -> float:
        num = 0.0
        den = 0.0
        for row in rows:
            n = float(row["n_steps"])
            num += float(row["phys"][key]) * n
            den += n
        return num / den if den else float("nan")

    hard = [r for r in per_ep if int(r["episode"]) in hard_eps]
    easy = [r for r in per_ep if int(r["episode"]) not in hard_eps]
    out = {}
    for label, rows in (("hard", hard), ("other", easy)):
        if not rows:
            continue
        out[f"mae_joints_{label}"] = _weighted(rows, "mae_joints")
        out[f"mse_all_{label}"] = _weighted(rows, "mse_all")
        out[f"n_steps_{label}"] = int(sum(r["n_steps"] for r in rows))
    return out


def _print_block(title: str, metrics: dict, names: list[str]) -> None:
    print(f"\n===== {title} =====")
    print(
        f"  all       MSE={metrics['mse_all']:.6f}  MAE={metrics['mae_all']:.6f}  "
        f"RMSE={metrics['rmse_all']:.6f}"
    )
    if "mse_joints" in metrics:
        print(
            f"  joints    MSE={metrics['mse_joints']:.6f}  MAE={metrics['mae_joints']:.6f}  "
            f"RMSE={metrics['rmse_joints']:.6f}"
        )
    if "mse_gripper" in metrics:
        print(
            f"  gripper   MSE={metrics['mse_gripper']:.6f}  MAE={metrics['mae_gripper']:.6f}  "
            f"RMSE={metrics['rmse_gripper']:.6f}"
        )
    for name in names:
        print(
            f"  {name:16s}  MSE={metrics['mse_per_dim'][name]:.6f}  "
            f"MAE={metrics['mae_per_dim'][name]:.6f}  "
            f"RMSE={metrics['rmse_per_dim'][name]:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batched val-set action MSE")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Treat --checkpoint as a run dir and eval every checkpoint_*.pt",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--run-dir",
        type=str,
        default="data/openarm_hcx_dual_arm_val",
        help="LeRobot dataset root (default: val split)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth window (1=all frames; n_action_steps matches chunking)",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Comma-separated episode ids to keep (default: all)",
    )
    parser.add_argument(
        "--history-noise-std",
        type=float,
        default=0.0,
        help="Inference history noise (default 0 for deterministic eval)",
    )
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Sweep: skip a ckpt if its eval json already exists",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    ckpt_arg = Path(args.checkpoint)
    if not ckpt_arg.is_absolute():
        ckpt_arg = base_dir / ckpt_arg
    ckpt_arg = ckpt_arg.resolve()
    if args.sweep:
        if not ckpt_arg.is_dir():
            raise NotADirectoryError(f"--sweep expects a directory, got {ckpt_arg}")
        ckpt_paths = _list_sweep_checkpoints(ckpt_arg)
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint_*.pt under {ckpt_arg}")
    else:
        ckpt_paths = [_resolve_checkpoint(args.checkpoint, base_dir)]

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (base_dir / run_dir).resolve()
    episode_filter = None
    if args.episodes:
        episode_filter = {int(x.strip()) for x in args.episodes.split(",") if x.strip()}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    first_path = ckpt_paths[0]
    t_load = time.time()
    first_ckpt = torch.load(first_path, map_location="cpu", weights_only=False)
    first_ckpt.pop("optimizer_state_dict", None)
    train_cfg_path = _resolve_train_config(first_path, args.config, base_dir)
    if train_cfg_path is not None:
        cfg = load_config(train_cfg_path)
        print(f"train_config: {train_cfg_path}")
    else:
        cfg = first_ckpt["config"]
        print("train_config: checkpoint embedded config")

    stats = first_ckpt["stats"]
    names = _action_names(cfg)
    joint_mask, grip_mask = _masks(names)
    delta_joint_mask = joint_mask_from_names(names, int(cfg.action_dim))
    n_obs = int(cfg.dataset.n_obs_steps)
    n_act = int(cfg.policy.n_action_steps)
    horizon = int(cfg.dataset.horizon)
    predict_joint_delta = bool(cfg.policy.predict_joint_delta)
    cfg.policy.history_noise_std = float(args.history_noise_std)
    hard_eps = {15, 85}

    device = _pick_device(args.device)
    policy = build_policy(cfg, stats)
    policy.to(device)
    policy.eval()
    print(
        f"run_dir: {run_dir}  device={device}  n_obs={n_obs} horizon={horizon} "
        f"n_act={n_act} resize={cfg.dataset.resize_size} "
        f"predict_joint_delta={predict_joint_delta} "
        f"history_noise_std={cfg.policy.history_noise_std}  "
        f"n_ckpts={len(ckpt_paths)}  build={time.time() - t_load:.1f}s"
    )

    t_ds = time.time()
    dataset = build_episode_dataset(
        run_dir=run_dir,
        n_obs_steps=n_obs,
        horizon=horizon,
        n_action_steps=n_act,
        drop_n_last_frames=n_act,
        stats=stats,
        normalize=True,
        norm_mode=cfg.dataset.norm_mode,
        resize_size=cfg.dataset.resize_size,
        pre_crop_size=cfg.dataset.pre_crop_size,
        crop_size=None,
        random_crop=False,
        color_jitter_brightness=0.0,
        color_jitter_contrast=0.0,
        color_jitter_saturation=0.0,
        color_jitter_hue=0.0,
        defer_augment=True,
        uint8_cache=False,
        predict_joint_delta=predict_joint_delta,
    )
    keep_idx = _subset_indices_for_episodes(dataset, episode_filter)
    stride = max(int(args.stride), 1)
    if stride > 1:
        keep_idx = keep_idx[::stride]
    subset: torch.utils.data.Dataset = (
        dataset if keep_idx == list(range(len(dataset))) else Subset(dataset, keep_idx)
    )
    print(
        f"dataset: n={len(dataset)} keep={len(keep_idx)} stride={stride} "
        f"episodes={len(dataset._episode_ids)} "
        f"filter={sorted(episode_filter) if episode_filter else 'all'} "
        f"load={time.time() - t_ds:.1f}s"
    )

    loader = DataLoader(
        subset,
        batch_size=max(int(args.batch_size), 1),
        shuffle=False,
        num_workers=max(int(args.num_workers), 0),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
    )

    summary_rows: list[dict] = []
    for ckpt_i, ckpt_path in enumerate(ckpt_paths):
        out_path = (
            Path(args.out_json)
            if args.out_json and len(ckpt_paths) == 1
            else ckpt_path.parent
            / f"eval_val_mse_{device.type}_stepPLACEHOLDER.json"
        )
        ckpt = first_ckpt if ckpt_i == 0 and ckpt_path == first_path else torch.load(
            ckpt_path, map_location="cpu", weights_only=False
        )
        ckpt.pop("optimizer_state_dict", None)
        step = int(ckpt.get("step") or 0)
        if args.out_json and len(ckpt_paths) == 1:
            out_path = Path(args.out_json)
        else:
            out_path = ckpt_path.parent / f"eval_val_mse_{device.type}_step{step:06d}.json"
        if args.skip_existing and out_path.is_file() and len(ckpt_paths) > 1:
            print(f"\n[{ckpt_i + 1}/{len(ckpt_paths)}] skip existing {out_path.name}")
            prev = json.loads(out_path.read_text())
            split = _hard_easy_split(prev.get("per_episode", []), hard_eps)
            summary_rows.append(
                {
                    "checkpoint": str(ckpt_path),
                    "step": prev.get("step", step),
                    "mse_all_norm": prev["norm"]["mse_all"],
                    "mse_all_phys": prev["phys"]["mse_all"],
                    "mae_joints": prev["phys"]["mae_joints"],
                    "mae_gripper": prev["phys"].get("mae_gripper"),
                    **split,
                    "skipped": True,
                }
            )
            continue

        print(f"\n[{ckpt_i + 1}/{len(ckpt_paths)}] {ckpt_path.name} step={step}")
        policy.load_state_dict(ckpt["policy_state_dict"])
        policy.eval()
        try:
            metrics_norm, metrics_phys, per_ep, n_windows, elapsed = _eval_policy(
                policy=policy,
                loader=loader,
                dataset=dataset,
                keep_idx=keep_idx,
                device=device,
                stats=stats,
                cfg=cfg,
                names=names,
                joint_mask=joint_mask,
                grip_mask=grip_mask,
                delta_joint_mask=delta_joint_mask,
                horizon=horizon,
                predict_joint_delta=predict_joint_delta,
            )
        except torch.cuda.OutOfMemoryError:
            raise SystemExit(
                f"CUDA OOM at batch_size={args.batch_size}. Retry with --batch-size 4 or 2."
            ) from None

        split = _hard_easy_split(per_ep, hard_eps)
        print(
            f"  infer {elapsed:.1f}s  {n_windows / max(elapsed, 1e-6):.2f} win/s  "
            f"MSE_norm={metrics_norm['mse_all']:.4f}  "
            f"MAE_j={metrics_phys['mae_joints']:.4f}  "
            f"MAE_j_15+85={split.get('mae_joints_hard', float('nan')):.4f}  "
            f"MAE_j_other={split.get('mae_joints_other', float('nan')):.4f}"
        )
        if len(ckpt_paths) == 1:
            _print_block("normalized action  (gaussian_2std space)", metrics_norm, names)
            _print_block("physical action  (joints deg, gripper opening)", metrics_phys, names)
            print("\n===== per-episode physical MSE / MAE =====")
            for row in per_ep:
                p = row["phys"]
                print(
                    f"  ep{row['episode']:04d}  n={row['n_steps']:5d}  "
                    f"MSE={p['mse_all']:.4f}  MAE_j={p.get('mae_joints', float('nan')):.4f}  "
                    f"MAE_g={p.get('mae_gripper', float('nan')):.4f}"
                )

        out = {
            "checkpoint": str(ckpt_path),
            "step": step,
            "run_dir": str(run_dir),
            "device": str(device),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "stride": stride,
            "episodes": sorted(episode_filter) if episode_filter else None,
            "n_windows": n_windows,
            "n_obs_steps": n_obs,
            "horizon": horizon,
            "n_action_steps": n_act,
            "norm_mode": cfg.dataset.norm_mode,
            "history_noise_std": float(cfg.policy.history_noise_std),
            "predict_joint_delta": predict_joint_delta,
            "seconds": elapsed,
            "windows_per_sec": n_windows / max(elapsed, 1e-6),
            "norm": metrics_norm,
            "phys": metrics_phys,
            "per_episode": per_ep,
            **split,
        }
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  wrote {out_path}")
        summary_rows.append(
            {
                "checkpoint": str(ckpt_path),
                "step": step,
                "mse_all_norm": metrics_norm["mse_all"],
                "mse_all_phys": metrics_phys["mse_all"],
                "mae_joints": metrics_phys["mae_joints"],
                "mae_gripper": metrics_phys.get("mae_gripper"),
                **split,
                "seconds": elapsed,
            }
        )

    if len(summary_rows) > 1:
        summary_path = ckpt_arg / f"eval_val_mse_{device.type}_ckpt_sweep_summary.json"
        if not ckpt_arg.is_dir():
            summary_path = ckpt_paths[0].parent / f"eval_val_mse_{device.type}_ckpt_sweep_summary.json"
        summary_path.write_text(json.dumps({"rows": summary_rows}, indent=2))
        print("\n===== ckpt sweep (physical joint MAE) =====")
        best = min(summary_rows, key=lambda r: r.get("mae_joints_hard", r["mae_joints"]))
        for row in summary_rows:
            mark = "  <-- best 15+85" if row is best else ""
            print(
                f"  step={row['step']:6d}  MAE_j={row['mae_joints']:.4f}  "
                f"15+85={row.get('mae_joints_hard', float('nan')):.4f}  "
                f"other={row.get('mae_joints_other', float('nan')):.4f}  "
                f"MSE_n={row['mse_all_norm']:.4f}{mark}"
            )
        print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
