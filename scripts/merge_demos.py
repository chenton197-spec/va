#!/usr/bin/env python3
"""合并多个 demo run 目录为一个。

CLI 用法
--------
在 conda 环境 ``lerobot`` 下，从 ``va/`` 目录运行::

    conda activate lerobot
    cd /path/to/va
    python scripts/merge_demos.py \\
        data/demos/pusht_demos_2607211809 \\
        data/demos/pusht_demos_2607211829 \\
        --output data/demos/pusht_demos_merged

命令行参数
~~~~~~~~~~
``run_dirs``
    一个或多个源 run 目录（含 ``meta.json`` 与 ``episodes/ep_*.npz``）。
    按输入顺序依次拷贝，episode 索引从 0 连续重新编号。

``--output PATH``
    合并后的输出目录。默认 ``data/demos/<首个 run 名>_merged_YYMMDDHHMM``。

``--overwrite``
    若输出目录已存在则先清空再写入；默认目录已存在时报错。

``--allow-fps-mismatch``
    允许源 run 的 ``fps`` 不一致（仍写入第一个 run 的 fps，并打印警告）。
    其它 schema 字段仍必须一致。

约束
~~~~
所有源 run 的 schema 必须一致（backend / embodiment / cameras /
state_dim / action_dim / state_names / action_names；默认也要求 fps）。
``task`` 取第一个非空值；``created_at`` 取最早时间；``num_episodes`` 为合并后总数。

合并完成后重新计算 ``stats.json``（全局 state/action mean/std）。
"""

from __future__ import annotations

import argparse
import shutil
import warnings
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from robotfm.data.schema import episode_path, load_meta, save_meta
from robotfm.data.stats import compute_stats, save_stats
from robotfm.types import EpisodeMeta

# Schema 字段：合并时必须完全一致（fps 可用 --allow-fps-mismatch 放宽）
_SCHEMA_FIELDS = (
    "backend",
    "embodiment",
    "fps",
    "cameras",
    "state_dim",
    "action_dim",
    "state_names",
    "action_names",
)


def _resolve_run_dir(path: Path) -> Path:
    """相对路径相对 va/ 根目录解析。"""
    if path.is_absolute():
        return path
    va_root = Path(__file__).resolve().parents[1]
    return (va_root / path).resolve()


def _assert_compatible(
    base: EpisodeMeta,
    other: EpisodeMeta,
    other_dir: Path,
    *,
    allow_fps_mismatch: bool = False,
) -> None:
    base_d = asdict(base)
    other_d = asdict(other)
    for field in _SCHEMA_FIELDS:
        if base_d[field] == other_d[field]:
            continue
        if field == "fps" and allow_fps_mismatch:
            warnings.warn(
                f"fps mismatch ignored: base={base_d['fps']!r}, "
                f"other ({other_dir})={other_d['fps']!r}; "
                f"output meta keeps base fps={base_d['fps']!r}",
                stacklevel=2,
            )
            continue
        raise ValueError(
            f"Schema mismatch on {field!r}:\n"
            f"  base:  {base_d[field]!r}\n"
            f"  other ({other_dir}): {other_d[field]!r}"
        )


def _list_episodes(run_dir: Path) -> list[Path]:
    episodes_dir = run_dir / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"Missing episodes/ in {run_dir}")
    files = sorted(episodes_dir.glob("ep_*.npz"))
    if not files:
        raise FileNotFoundError(f"No ep_*.npz in {episodes_dir}")
    return files


def _pick_task(metas: list[EpisodeMeta]) -> str:
    for meta in metas:
        if meta.task:
            return meta.task
    return ""


def _earliest_created_at(metas: list[EpisodeMeta]) -> str:
    stamps = [m.created_at for m in metas if m.created_at]
    return min(stamps) if stamps else datetime.now().astimezone().isoformat()


def merge_demos(
    run_dirs: list[Path],
    output_dir: Path,
    overwrite: bool = False,
    allow_fps_mismatch: bool = False,
) -> Path:
    """将多个 run 合并到 output_dir，返回输出路径。"""
    if len(run_dirs) < 1:
        raise ValueError("Need at least one run directory")

    for d in run_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"Run directory not found: {d}")
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"Missing meta.json in {d}")

    metas = [load_meta(d) for d in run_dirs]
    base_meta = metas[0]
    for meta, d in zip(metas[1:], run_dirs[1:]):
        _assert_compatible(
            base_meta,
            meta,
            d,
            allow_fps_mismatch=allow_fps_mismatch,
        )

    episode_lists = [_list_episodes(d) for d in run_dirs]
    total = sum(len(files) for files in episode_lists)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    episodes_out = output_dir / "episodes"
    episodes_out.mkdir(parents=True, exist_ok=True)

    out_meta = replace(
        base_meta,
        num_episodes=0,
        task=_pick_task(metas),
        created_at=_earliest_created_at(metas),
    )
    save_meta(output_dir, out_meta)

    out_index = 0
    for src_dir, files in zip(run_dirs, episode_lists):
        print(f"Copying {len(files)} episodes from {src_dir}")
        for src in files:
            dst = episode_path(output_dir, out_index)
            shutil.copy2(src, dst)
            out_index += 1

    out_meta.num_episodes = total
    save_meta(output_dir, out_meta)

    print(f"Computing stats for {total} episodes...")
    save_stats(output_dir, compute_stats(output_dir))
    print(f"Merged {len(run_dirs)} runs -> {output_dir} ({total} episodes)")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple demo run directories into one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/merge_demos.py \\\n"
            "      data/demos/pusht_demos_2607211809 \\\n"
            "      data/demos/pusht_demos_2607211829 \\\n"
            "      --output data/demos/pusht_demos_merged\n"
        ),
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=str,
        help="Source run directories to merge (order preserved)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output run directory (default: <first>_merged_YYMMDDHHMM)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directory if it already exists",
    )
    parser.add_argument(
        "--allow-fps-mismatch",
        action="store_true",
        help="Allow merging runs with different fps (keeps first run's fps)",
    )
    args = parser.parse_args()

    run_dirs = [_resolve_run_dir(Path(p)) for p in args.run_dirs]

    if args.output:
        output_dir = _resolve_run_dir(Path(args.output))
    else:
        stamp = datetime.now().strftime("%y%m%d%H%M")
        output_dir = run_dirs[0].parent / f"{run_dirs[0].name}_merged_{stamp}"

    merge_demos(
        run_dirs,
        output_dir,
        overwrite=args.overwrite,
        allow_fps_mismatch=args.allow_fps_mismatch,
    )


if __name__ == "__main__":
    main()
