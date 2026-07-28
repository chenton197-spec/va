#!/usr/bin/env python3
"""按目标 fps 对 demo run 均匀抽帧（不插值）。

示例（30fps -> 10fps，stride=3）::

    python scripts/resample_demos.py \\
        data/demos/pusht_demos_merged_2607211809_1829 \\
        --target-fps 10 \\
        --output data/demos/pusht_demos_merged_2607211809_1829_10fps
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from robotfm.data.schema import (
    episode_path,
    load_episode,
    load_meta,
    save_meta,
    validate_episode_arrays,
)
from robotfm.data.stats import compute_stats, save_stats


def _resolve_run_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    va_root = Path(__file__).resolve().parents[1]
    return (va_root / path).resolve()


def _subsample_indices(length: int, stride: int, *, keep_last: bool) -> np.ndarray:
    if length <= 0:
        raise ValueError("Episode length must be positive")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    indices = list(range(0, length, stride))
    if keep_last and indices[-1] != length - 1:
        indices.append(length - 1)
    return np.asarray(indices, dtype=np.int64)


def _subsample_arrays(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, arr in arrays.items():
        out[key] = arr[indices]
    # 抽帧后仅末帧标记 episode 结束
    if "done" in out:
        done = np.zeros(len(indices), dtype=bool)
        done[-1] = True
        out["done"] = done
    return out


def resample_demos(
    run_dir: Path,
    output_dir: Path,
    target_fps: int,
    *,
    overwrite: bool = False,
    keep_last: bool = True,
) -> Path:
    meta = load_meta(run_dir)
    src_fps = int(meta.fps)
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if src_fps % target_fps != 0:
        raise ValueError(
            f"src fps={src_fps} is not divisible by target_fps={target_fps}; "
            "uniform integer stride required"
        )
    stride = src_fps // target_fps
    if stride == 1 and output_dir.resolve() == run_dir.resolve():
        raise ValueError("Nothing to do: stride=1 and output == input")

    episode_files = sorted((run_dir / "episodes").glob("ep_*.npz"))
    if not episode_files:
        raise FileNotFoundError(f"No ep_*.npz in {run_dir / 'episodes'}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\nPass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    episodes_out = output_dir / "episodes"
    episodes_out.mkdir(parents=True, exist_ok=True)

    out_meta = replace(meta, fps=target_fps, num_episodes=0)
    save_meta(output_dir, out_meta)

    total_src = 0
    total_dst = 0
    for i, src in enumerate(episode_files):
        payload = load_episode(src)
        arrays = payload["arrays"]
        t = arrays["state"].shape[0]
        total_src += t
        indices = _subsample_indices(t, stride, keep_last=keep_last)
        sub = _subsample_arrays(arrays, indices)
        validate_episode_arrays(sub, out_meta)
        total_dst += indices.shape[0]

        path = episode_path(output_dir, i)
        np.savez_compressed(
            path,
            success=payload["success"],
            task=payload["task"],
            **sub,
        )

    out_meta.num_episodes = len(episode_files)
    save_meta(output_dir, out_meta)
    save_stats(output_dir, compute_stats(output_dir))

    print(
        f"Resampled {run_dir} -> {output_dir}: "
        f"fps {src_fps}->{target_fps} (stride={stride}), "
        f"{len(episode_files)} episodes, frames {total_src}->{total_dst}"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Uniformly subsample demo run fps.")
    parser.add_argument("run_dir", type=str, help="Source run directory")
    parser.add_argument("--target-fps", type=int, required=True, help="Target fps (must divide source fps)")
    parser.add_argument("--output", type=str, default=None, help="Output run directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--drop-last",
        action="store_true",
        help="Do not force-include the final frame when it falls off the stride grid",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(Path(args.run_dir))
    if args.output:
        output_dir = _resolve_run_dir(Path(args.output))
    else:
        output_dir = run_dir.parent / f"{run_dir.name}_{args.target_fps}fps"

    resample_demos(
        run_dir,
        output_dir,
        args.target_fps,
        overwrite=args.overwrite,
        keep_last=not args.drop_last,
    )


if __name__ == "__main__":
    main()
