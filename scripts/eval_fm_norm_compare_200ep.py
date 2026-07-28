#!/usr/bin/env python3
"""Eval FM limits vs gaussian at 20k/25k/30k, 200 ep, coverage>=0.92."""

from __future__ import annotations

from pathlib import Path

import torch

from robotfm.eval import evaluate_flow_matching


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    thresh = 0.92
    num_episodes = 200

    runs = [
        ("limits", base_dir / "outputs/fm_pusht_limits_260725203614", "limits"),
        ("gaussian", base_dir / "outputs/fm_pusht_gaussian_260726004557", "gaussian"),
    ]

    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"metric: success @ coverage>={thresh}, {num_episodes}ep")
    print()

    all_summary: list[tuple[str, str, dict]] = []
    for run_name, out, expected_norm in runs:
        ckpts = [
            ("020000", out / "checkpoint_020000.pt"),
            ("025000", out / "checkpoint_025000.pt"),
            ("030000/final", out / "checkpoint_final.pt"),
        ]
        print(f"########## RUN {run_name} ({out.name}) ##########")
        results: dict[str, dict] = {}
        for label, path in ckpts:
            if not path.is_file():
                raise FileNotFoundError(path)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            cfg = ckpt["config"]
            cfg.dataset.norm_mode = expected_norm
            cfg.eval.num_episodes = num_episodes
            cfg.eval.success_coverage = thresh
            print(f"===== {run_name} {label} ({path.name}) =====")
            print(
                f"policy.type={cfg.policy.type} "
                f"norm_mode={cfg.dataset.norm_mode} "
                f"success_coverage>={thresh} num_episodes={num_episodes}"
            )
            metrics = evaluate_flow_matching(
                cfg, path, base_dir, render=False, pace_realtime=False
            )
            results[label] = metrics
            sr = float(metrics["success_rate"])
            ar = float(metrics["avg_reward"])
            mc = float(metrics["mean_max_coverage"])
            print(f"Success rate: {sr:.3f}")
            print(f"Avg reward: {ar:.3f}")
            print(f"Mean max coverage: {mc:.3f}")
            print()
            (out / f"eval_{path.stem}_cov92_200ep.log").write_text(
                f"checkpoint: {path}\n"
                f"norm_mode: {cfg.dataset.norm_mode}\n"
                f"success_coverage>={thresh}\n"
                f"num_episodes: {num_episodes}\n"
                f"Success rate: {sr:.3f}\n"
                f"Avg reward: {ar:.3f}\n"
                f"Mean max coverage: {mc:.3f}\n"
            )
            all_summary.append((run_name, label, metrics))

        print(f"===== SUMMARY {run_name} (cov>={thresh}, {num_episodes}ep) =====")
        for label, metrics in results.items():
            print(
                f"{label}: success={float(metrics['success_rate']):.3f}  "
                f"mean_max_cov={float(metrics['mean_max_coverage']):.3f}"
            )
        print()

    print("===== OVERALL SUMMARY =====")
    for run_name, label, metrics in all_summary:
        print(
            f"{run_name:8s} {label:12s}  "
            f"success={float(metrics['success_rate']):.3f}  "
            f"mean_max_cov={float(metrics['mean_max_coverage']):.3f}"
        )


if __name__ == "__main__":
    main()
