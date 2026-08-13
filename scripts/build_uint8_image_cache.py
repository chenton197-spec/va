#!/usr/bin/env python3
"""Build a uint8 RGB memmap cache for a LeRobot image-sequence dataset.

Example::

    cd /home/casbot/ct/va
    python scripts/build_uint8_image_cache.py \\
        --run-dir /home/casbot/ct/data/shine_shoes_fr3_s256_prefix_speedup \\
        --resize-size 256 \\
        --num-workers 8

    # 1280×720 → center 720² → 512²
    python scripts/build_uint8_image_cache.py \\
        --run-dir data/openarm_hcx_dual_arm \\
        --pre-crop-size 720 \\
        --resize-size 512 \\
        --num-workers 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robotfm.data.uint8_cache import build_uint8_image_cache


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Dataset root (contains meta/ + images/)",
    )
    p.add_argument("--resize-size", type=int, default=256)
    p.add_argument(
        "--pre-crop-size",
        type=int,
        default=None,
        help="Center-crop to this square size before resize (e.g. 720 for 1280x720)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: {run_dir}/cache/uint8_rgb_{H}x{W}[_pcN]",
    )
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = build_uint8_image_cache(
        args.run_dir,
        resize_size=args.resize_size,
        pre_crop_size=args.pre_crop_size,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    print(f"done: {out}")


if __name__ == "__main__":
    main()
