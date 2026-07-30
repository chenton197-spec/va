#!/usr/bin/env python3
"""小批量超参扫描：短训对比 lr / batch_size，选出 loss 下降更稳的配置。

用法（在 va/ 目录）::

    python scripts/sweep_train_params.py --config configs/pusht_fm.yaml
    python scripts/sweep_train_params.py --steps 800 --run-name pusht_demos_merged_2607211809_1829

结果写到 ``outputs/sweeps/<timestamp>/summary.json``，并打印推荐参数。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from robotfm.config import RobotFMConfig, load_config, resolve_path
from robotfm.collect.loop import get_run_dir
from robotfm.data.dataset import build_episode_dataset
from robotfm.data.stats import ensure_stats
from robotfm.train import build_policy


def _ema(values: list[float], alpha: float = 0.1) -> float:
    """指数滑动平均，用于平滑末段 loss。"""
    if not values:
        return float("nan")
    avg = values[0]
    for v in values[1:]:
        avg = alpha * v + (1.0 - alpha) * avg
    return float(avg)


def run_one(
    cfg: RobotFMConfig,
    base_dir: Path,
    steps: int,
    log_every: int,
) -> dict:
    """跑一次短训，返回 loss 曲线与汇总指标。"""
    run_dir = get_run_dir(cfg, base_dir)
    stats = ensure_stats(run_dir, cfg.dataset.norm_mode)

    dataset = build_episode_dataset(
        run_dir=run_dir,
        n_obs_steps=cfg.dataset.n_obs_steps,
        horizon=cfg.dataset.horizon,
        n_action_steps=cfg.policy.n_action_steps,
        stats=stats,
        normalize=True,
        norm_mode=cfg.dataset.norm_mode,
        resize_size=cfg.dataset.resize_size,
        crop_size=cfg.dataset.crop_size,
        random_crop=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.device.startswith("cuda"),
        drop_last=True,
    )

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    policy = build_policy(cfg).to(device)
    optim = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    history: list[dict[str, float]] = []
    raw_losses: list[float] = []
    step = 0
    t0 = time.time()
    pbar = tqdm(total=steps, desc=f"lr={cfg.train.lr:g} bs={cfg.train.batch_size}", leave=False)

    while step < steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = policy.compute_loss(batch)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            # 记录梯度范数，便于发现 lr 过大导致爆炸
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1e9)
            optim.step()

            loss_f = float(loss.detach().item())
            raw_losses.append(loss_f)
            if step % log_every == 0 or step + 1 == steps:
                history.append(
                    {
                        "step": float(step),
                        "loss": loss_f,
                        "grad_norm": float(grad_norm),
                    }
                )
                pbar.set_postfix(loss=f"{loss_f:.3f}")

            step += 1
            pbar.update(1)
            if step >= steps:
                break

    pbar.close()
    elapsed = time.time() - t0

    # 用前/后窗口对比：相对下降越大越好；末段 EMA 越小越好
    warm = max(1, steps // 10)
    head = raw_losses[:warm]
    tail = raw_losses[-warm:]
    head_mean = sum(head) / len(head)
    tail_mean = sum(tail) / len(tail)
    tail_ema = _ema(tail)
    rel_drop = (head_mean - tail_mean) / max(head_mean, 1e-8)
    # 末段是否发散：后半相对前半回升
    mid = len(raw_losses) // 2
    mid_mean = sum(raw_losses[mid : mid + warm]) / max(1, len(raw_losses[mid : mid + warm]))
    diverged = tail_mean > mid_mean * 1.25 and rel_drop < 0.05
    # 综合分：优先相对下降，其次末段 loss 低；发散则重罚
    score = rel_drop * 2.0 - math.log1p(max(tail_ema, 0.0)) * 0.15
    if diverged or not math.isfinite(tail_ema) or tail_ema > head_mean * 2:
        score -= 10.0

    return {
        "lr": cfg.train.lr,
        "batch_size": cfg.train.batch_size,
        "weight_decay": cfg.train.weight_decay,
        "steps": steps,
        "device": str(device),
        "elapsed_sec": elapsed,
        "head_loss": head_mean,
        "tail_loss": tail_mean,
        "tail_ema": tail_ema,
        "rel_drop": rel_drop,
        "diverged": diverged,
        "score": score,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep train lr/batch_size with short runs")
    parser.add_argument("--config", type=str, default="configs/pusht_fm.yaml")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="覆盖 dataset.run_name，例如 pusht_demos_merged_2607211809_1829",
    )
    parser.add_argument("--steps", type=int, default=800, help="每组超参短训步数")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        default=[3e-5, 1e-4, 3e-4, 1e-3],
        help="学习率候选",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="batch size 候选",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="默认自动：有 CUDA 用 cuda，否则 cpu",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    base_cfg = load_config(base_dir / args.config)
    if args.run_name:
        base_cfg.dataset.run_name = args.run_name
    base_cfg.train.num_workers = args.num_workers
    if args.device:
        base_cfg.train.device = args.device
    elif not torch.cuda.is_available():
        base_cfg.train.device = "cpu"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = resolve_path(base_dir, "outputs/sweeps") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    combos = [(lr, bs) for lr in args.lrs for bs in args.batch_sizes]
    print(
        f"Sweep {len(combos)} runs | data={base_cfg.dataset.run_name} "
        f"| steps={args.steps} | device={base_cfg.train.device}"
    )

    for lr, bs in combos:
        cfg = deepcopy(base_cfg)
        cfg.train.lr = lr
        cfg.train.batch_size = bs
        # 避免短训写满磁盘：每个 trial 单独目录但不强制频繁 save
        cfg.output_dir = str(out_dir / f"lr{lr:g}_bs{bs}")
        try:
            result = run_one(cfg, base_dir, steps=args.steps, log_every=args.log_every)
        except Exception as exc:  # noqa: BLE001 — 扫参时单次失败继续
            result = {
                "lr": lr,
                "batch_size": bs,
                "error": str(exc),
                "score": -1e9,
                "diverged": True,
            }
            print(f"  FAIL lr={lr:g} bs={bs}: {exc}")
        results.append(result)
        if "error" not in result:
            print(
                f"  lr={lr:g} bs={bs}: "
                f"head={result['head_loss']:.3f} tail={result['tail_loss']:.3f} "
                f"drop={result['rel_drop']:.1%} score={result['score']:.3f} "
                f"({'DIVERGED' if result['diverged'] else 'ok'}, {result['elapsed_sec']:.0f}s)"
            )

    ranked = sorted(results, key=lambda r: r.get("score", -1e9), reverse=True)
    best = ranked[0]
    summary = {
        "created_at": stamp,
        "run_name": base_cfg.dataset.run_name,
        "steps_per_run": args.steps,
        "device": base_cfg.train.device,
        "ranked": [
            {
                k: v
                for k, v in r.items()
                if k != "history"
            }
            for r in ranked
        ],
        "best": {k: v for k, v in best.items() if k != "history"},
        "recommend": {
            "train.lr": best.get("lr"),
            "train.batch_size": best.get("batch_size"),
            "train.weight_decay": best.get("weight_decay", base_cfg.train.weight_decay),
            "dataset.run_name": base_cfg.dataset.run_name,
            "note": "基于短训 loss 下降选出；正式训练仍建议用更多 steps 并做 eval",
        },
        "base_config": asdict(base_cfg),
    }
    # 附带完整 history 便于画图
    summary["histories"] = {
        f"lr{r.get('lr')}_bs{r.get('batch_size')}": r.get("history", []) for r in results
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Sweep done ===")
    print(f"summary: {summary_path}")
    if "error" in best:
        print("No successful run.")
        return
    print(
        f"Recommend: lr={best['lr']:g}, batch_size={best['batch_size']}, "
        f"rel_drop={best['rel_drop']:.1%}, tail_loss={best['tail_loss']:.3f}"
    )
    print("Suggested yaml snippet:")
    print("dataset:")
    print(f"  run_name: {base_cfg.dataset.run_name}")
    print("train:")
    print(f"  lr: {best['lr']}")
    print(f"  batch_size: {best['batch_size']}")
    print(f"  weight_decay: {best.get('weight_decay', base_cfg.train.weight_decay)}")
    print(f"  device: {base_cfg.train.device}")


if __name__ == "__main__":
    main()
