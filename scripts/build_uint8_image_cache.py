#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robotfm.data.uint8_cache import build_uint8_image_cache


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = build_uint8_image_cache(
        args.run_dir,
        image_size=[args.height, args.width],
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    print(f"done: {out}")


if __name__ == "__main__":
    main()
