#!/usr/bin/env python3
"""将 LeRobot image-sequence 数据集的 JPEG 预缩放到新目录（不改原数据）。

拷贝/链接 sidecar（data、meta、stats），把 images 写成 ``size×size``，
并更新输出集 ``meta/info.json`` 的图像 shape。

用法:
  cd /home/casbotskill/ct/va
  conda activate lerobot
  python scripts/preresize_images.py \\
    --dataset-root shine_shoes_fr3 \\
    --output-root shine_shoes_fr3_s256 \\
    --size 256
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
from tqdm import tqdm


def _symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
        print(f"symlink {dst} -> {src.resolve()}")
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"copied {src} -> {dst}")


def copy_sidecars(src_root: Path, dst_root: Path) -> None:
    """链接 data/stats；meta 逐文件链接，但 info.json 必须实体拷贝以便改 shape。"""
    data_src = src_root / "data"
    if data_src.exists():
        _symlink_or_copy(data_src, dst_root / "data")
    else:
        print(f"warn: missing {data_src}, skip")

    stats_src = src_root / "stats.json"
    if stats_src.is_file():
        _symlink_or_copy(stats_src, dst_root / "stats.json")

    meta_src = src_root / "meta"
    meta_dst = dst_root / "meta"
    if not meta_src.is_dir():
        raise SystemExit(f"Missing meta dir: {meta_src}")
    meta_dst.mkdir(parents=True, exist_ok=True)
    for child in meta_src.iterdir():
        if child.name == "info.json":
            continue
        _symlink_or_copy(child, meta_dst / child.name)

    info_src = meta_src / "info.json"
    if not info_src.is_file():
        raise SystemExit(f"Missing info.json: {info_src}")
    shutil.copy2(info_src, meta_dst / "info.json")
    print(f"copied {info_src} -> {meta_dst / 'info.json'}")


def collect_image_jobs(src_images: Path, dst_images: Path) -> list[tuple[str, str]]:
    """返回 (src_path, dst_path)，保持相对路径不变。"""
    jobs: list[tuple[str, str]] = []
    for jpg in sorted(src_images.rglob("*.jpg")):
        if not jpg.is_file() or jpg.name.startswith(".") or ".tmp.jpg" in jpg.name:
            continue
        rel = jpg.relative_to(src_images)
        jobs.append((str(jpg), str(dst_images / rel)))
    return jobs


def _resize_one(args: tuple[str, str, int, int]) -> tuple[str, str]:
    """Return (dst_path, status)."""
    src_path, dst_path, size, jpeg_quality = args
    try:
        bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return dst_path, "error:read_failed"
        h, w = bgr.shape[:2]
        if h != size or w != size:
            bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
        out = Path(dst_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(
            dst_path,
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            return dst_path, "error:write_failed"
        return dst_path, "ok" if (h != size or w != size) else "skip"
    except Exception as exc:  # noqa: BLE001
        return dst_path, f"error:{exc}"


def update_info_shapes(info_path: Path, size: int, source_root: Path) -> None:
    info = json.loads(info_path.read_text())
    for key, feat in info.get("features", {}).items():
        if not key.startswith("observation.images."):
            continue
        if isinstance(feat, dict) and "shape" in feat:
            feat["shape"] = [size, size, 3]
    info["preresize"] = {
        "size": size,
        "source_dataset": str(source_root),
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-resize LeRobot JPEGs into a NEW dataset root (source untouched)"
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="源数据集根目录")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="输出数据集根目录（必须不同于源）",
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    src_root = args.dataset_root.resolve()
    dst_root = args.output_root.resolve()
    if dst_root == src_root:
        raise SystemExit("error: --output-root must differ from --dataset-root")
    if dst_root.is_relative_to(src_root) or src_root.is_relative_to(dst_root):
        raise SystemExit("error: refuse nested source/output dataset roots")

    src_images = src_root / "images"
    if not src_images.is_dir():
        raise SystemExit(f"Missing images dir: {src_images}")
    if not (src_root / "meta" / "info.json").is_file():
        raise SystemExit(f"Missing info.json under {src_root / 'meta'}")

    dst_root.mkdir(parents=True, exist_ok=True)
    copy_sidecars(src_root, dst_root)

    dst_images = dst_root / "images"
    jobs = collect_image_jobs(src_images, dst_images)
    if not jobs:
        raise SystemExit(f"No jpg files under {src_images}")

    print(f"source: {src_root}")
    print(f"output: {dst_root}")
    print(f"images: {len(jobs)}  size={args.size}  workers={args.workers}")

    ok = skip = err = 0
    errors: list[str] = []
    payload = [
        (src, dst, args.size, args.jpeg_quality) for src, dst in jobs
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_resize_one, item) for item in payload]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="preresize"):
            path, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                if len(errors) < 20:
                    errors.append(f"{path}: {status}")

    info_out = dst_root / "meta" / "info.json"
    update_info_shapes(info_out, args.size, src_root)

    print(f"done: ok={ok} skip={skip} err={err}")
    print(f"updated: {info_out}")
    print(f"source untouched: {src_root}")
    if errors:
        print("sample errors:")
        for line in errors:
            print(" ", line)
    if err:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
