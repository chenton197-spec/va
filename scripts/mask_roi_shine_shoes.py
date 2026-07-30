#!/usr/bin/env python3
"""对 shine_shoes_fr3 的 head/hand 图像做四点多边形 ROI mask（ROI 外置黑）。

IMG1 = observation.images.head
IMG2 = observation.images.hand

用法:
  # 交互选 4 点，写出新数据集（推荐，不改原图）
  conda activate lerobot
  cd /home/casbotskill/ct/va
  python scripts/mask_roi_shine_shoes.py \\
    --dataset-root shine_shoes_fr3 \\
    --output-root shine_shoes_fr3_roi_masked \\
    --cameras all

  # 只设置 / mask IMG1（head）
  python scripts/mask_roi_shine_shoes.py --cameras IMG1 ...

  # 只设置 / mask IMG2（hand）
  python scripts/mask_roi_shine_shoes.py --cameras IMG2 ...

  # 复用已保存的 ROI
  python scripts/mask_roi_shine_shoes.py \\
    --dataset-root shine_shoes_fr3 \\
    --output-root shine_shoes_fr3_roi_masked \\
    --roi-json shine_shoes_fr3_roi_masked/meta/roi_params.json \\
    --cameras all

  # 原地覆盖 jpg（危险，先备份）
  python scripts/mask_roi_shine_shoes.py --dataset-root shine_shoes_fr3 --inplace --cameras IMG1

操作（选点窗口）:
  - 左键依次点 4 个顶点
  - r: 重置当前相机
  - c: 确认（需恰好 4 点）
  - ESC: 取消并退出
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

CAMERA_ALIASES = {
    "IMG1": "observation.images.head",
    "IMG2": "observation.images.hand",
    "head": "observation.images.head",
    "hand": "observation.images.hand",
}
ALL_CAMERAS = ("observation.images.head", "observation.images.hand")
CAMERA_TO_ALIAS = {
    "observation.images.head": "IMG1",
    "observation.images.hand": "IMG2",
}


def resolve_cameras(choice: str) -> tuple[str, ...]:
    """``IMG1`` / ``IMG2`` / ``all``（也接受 head/hand）。"""
    key = choice.strip()
    if key.lower() == "all":
        return ALL_CAMERAS
    if key in CAMERA_ALIASES:
        return (CAMERA_ALIASES[key],)
    if key in ALL_CAMERAS:
        return (key,)
    raise SystemExit(
        f"Unknown --cameras {choice!r}; use IMG1, IMG2, or all"
    )


def prompt_cameras() -> tuple[str, ...]:
    print("选择要设置 ROI 的相机:")
    print("  1) IMG1  (observation.images.head)")
    print("  2) IMG2  (observation.images.hand)")
    print("  3) all   (IMG1 + IMG2)")
    while True:
        raw = input("请输入 [1/2/3] (默认 3): ").strip() or "3"
        if raw in ("1", "IMG1", "img1", "head"):
            return resolve_cameras("IMG1")
        if raw in ("2", "IMG2", "img2", "hand"):
            return resolve_cameras("IMG2")
        if raw in ("3", "all", "ALL"):
            return resolve_cameras("all")
        print("无效输入，请重新选择。")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mask outside 4-point ROI for shine_shoes_fr3 head/hand images."
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("shine_shoes_fr3"),
        help="源数据集根目录",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="输出数据集根目录（默认 <dataset-root>_roi_masked）；与 --inplace 互斥",
    )
    p.add_argument(
        "--inplace",
        action="store_true",
        help="原地覆盖 images/ 下 jpg（不复制 meta/data）",
    )
    p.add_argument(
        "--cameras",
        type=str,
        default=None,
        choices=["IMG1", "IMG2", "all", "head", "hand", "ask"],
        help="处理哪路相机: IMG1 / IMG2 / all；ask=启动时交互选择（默认 ask）",
    )
    p.add_argument(
        "--roi-json",
        type=Path,
        default=None,
        help="已有 ROI json；不提供则交互选点",
    )
    p.add_argument(
        "--episode",
        type=int,
        default=0,
        help="用于选点的 episode 索引",
    )
    p.add_argument(
        "--frame",
        type=int,
        default=0,
        help="用于选点的 frame 索引",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行处理进程数",
    )
    p.add_argument(
        "--preview-only",
        action="store_true",
        help="只选点/预览并写 roi json，不批量处理",
    )
    p.add_argument(
        "--skip-copy-meta",
        action="store_true",
        help="不复制/链接 meta 与 data（仅写 masked images）",
    )
    return p.parse_args()


def _find_sample_image(images_root: Path, camera: str, episode: int, frame: int) -> Path:
    chunk_dirs = sorted(images_root.glob("chunk-*"))
    if not chunk_dirs:
        raise FileNotFoundError(f"No chunk-* under {images_root}")
    ep_name = f"episode_{episode:06d}"
    frame_name = f"frame_{frame:06d}.jpg"
    for chunk in chunk_dirs:
        path = chunk / camera / ep_name / frame_name
        if path.is_file():
            return path
    # fallback: first available frame for that camera
    for chunk in chunk_dirs:
        cam_root = chunk / camera
        if not cam_root.is_dir():
            continue
        for ep_dir in sorted(cam_root.glob("episode_*")):
            frames = sorted(ep_dir.glob("frame_*.jpg"))
            if frames:
                return frames[0]
    raise FileNotFoundError(f"No sample image for camera={camera}")


def select_quad_roi(img_bgr: np.ndarray, window_title: str) -> list[list[int]] | None:
    """交互点击 4 个点，返回 [[x,y], ...] 或 None（取消）。"""
    h, w = img_bgr.shape[:2]
    max_side = 1280
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
    disp_w, disp_h = int(round(w * scale)), int(round(h * scale))

    base = cv2.resize(img_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    points: list[tuple[int, int]] = []

    def redraw() -> np.ndarray:
        canvas = base.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(
                canvas,
                str(i + 1),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if len(points) >= 2:
            for a, b in zip(points, points[1:]):
                cv2.line(canvas, a, b, (0, 255, 0), 2)
        if len(points) == 4:
            cv2.line(canvas, points[-1], points[0], (0, 255, 0), 2)
            overlay = canvas.copy()
            pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], (0, 180, 0))
            canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        hint = "L-click 4 pts | r=reset | c=confirm | ESC=cancel"
        cv2.putText(
            canvas,
            f"{window_title}  ({len(points)}/4)  {hint}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def on_mouse(event, x, y, _flags, _param) -> None:
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((int(x), int(y)))
            cv2.imshow(window_title, redraw())

    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_title, on_mouse)
    cv2.imshow(window_title, redraw())

    print(f"\n[{window_title}] 左键点 4 个顶点；r 重置；c 确认；ESC 取消")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            cv2.destroyWindow(window_title)
            return None
        if key in (ord("r"), ord("R")):
            points = []
            cv2.imshow(window_title, redraw())
        if key in (ord("c"), ord("C")):
            if len(points) != 4:
                print(f"  需要恰好 4 个点，当前 {len(points)}")
                continue
            break

    cv2.destroyWindow(window_title)
    inv = 1.0 / scale
    return [[int(round(x * inv)), int(round(y * inv))] for x, y in points]


def apply_polygon_mask(img_bgr: np.ndarray, polygon_xy: list[list[int]]) -> np.ndarray:
    """ROI 内保留原图，外置 0。"""
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    pts = np.asarray(polygon_xy, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    out = np.zeros_like(img_bgr)
    out[mask == 255] = img_bgr[mask == 255]
    return out


def preview_masked(img_bgr: np.ndarray, polygon_xy: list[list[int]], title: str) -> None:
    masked = apply_polygon_mask(img_bgr, polygon_xy)
    # draw poly on original for side-by-side
    annotated = img_bgr.copy()
    pts = np.asarray(polygon_xy, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
    pair = np.concatenate([annotated, masked], axis=1)
    max_w = 1600
    if pair.shape[1] > max_w:
        s = max_w / pair.shape[1]
        pair = cv2.resize(pair, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imshow(title, pair)
    print(f"[{title}] 左=原图+ROI，右=masked；按任意键继续")
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def collect_image_jobs(src_images: Path, dst_images: Path, cameras: tuple[str, ...]) -> list[tuple[str, str, str]]:
    """返回 (camera, src_path, dst_path) 列表。"""
    jobs: list[tuple[str, str, str]] = []
    for chunk in sorted(src_images.glob("chunk-*")):
        for camera in cameras:
            cam_dir = chunk / camera
            if not cam_dir.is_dir():
                continue
            for jpg in cam_dir.rglob("frame_*.jpg"):
                rel = jpg.relative_to(src_images)
                dst = dst_images / rel
                jobs.append((camera, str(jpg), str(dst)))
    return jobs


def _mask_one(args: tuple[str, str, list[list[int]], int]) -> str:
    src_path, dst_path, polygon, jpeg_quality = args
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read {src_path}")
    out = apply_polygon_mask(img, polygon)
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(dst_path, out, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError(f"Failed to write {dst_path}")
    return dst_path


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


def copy_or_link_sidecar(src_root: Path, dst_root: Path) -> None:
    """链接 data；meta 按文件链接（保留本地 roi_params.json，避免写穿源目录）。"""
    data_src = src_root / "data"
    if data_src.exists():
        _symlink_or_copy(data_src, dst_root / "data")
    else:
        print(f"warn: missing {data_src}, skip")

    meta_src = src_root / "meta"
    meta_dst = dst_root / "meta"
    if meta_src.is_dir():
        meta_dst.mkdir(parents=True, exist_ok=True)
        for child in meta_src.iterdir():
            if child.name == "roi_params.json":
                continue
            _symlink_or_copy(child, meta_dst / child.name)
    else:
        print(f"warn: missing {meta_src}, skip")

    stats_src = src_root / "stats.json"
    if stats_src.is_file():
        _symlink_or_copy(stats_src, dst_root / "stats.json")


def link_unselected_cameras(
    src_images: Path,
    dst_images: Path,
    selected: tuple[str, ...],
) -> None:
    """新数据集里，未选相机目录直接 symlink 到源，保证输出集完整。"""
    selected_set = set(selected)
    for chunk in sorted(src_images.glob("chunk-*")):
        for cam in ALL_CAMERAS:
            if cam in selected_set:
                continue
            src = chunk / cam
            if not src.is_dir():
                continue
            dst = dst_images / chunk.name / cam
            if dst.exists() or dst.is_symlink():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                dst.symlink_to(src.resolve(), target_is_directory=True)
                print(f"symlink unselected {dst} -> {src.resolve()}")
            except OSError as exc:
                print(f"warn: cannot symlink {dst}: {exc}")


def load_rois_from_json(path: Path, cameras: tuple[str, ...]) -> dict[str, list[list[int]]]:
    with path.open() as f:
        raw = json.load(f)
    rois: dict[str, list[list[int]]] = {}
    for cam in cameras:
        if cam in raw and raw[cam] is not None:
            rois[cam] = raw[cam]
            continue
        alias = CAMERA_TO_ALIAS[cam]
        if alias in raw and raw[alias] is not None:
            rois[cam] = raw[alias]
    missing = [c for c in cameras if c not in rois]
    if missing:
        raise SystemExit(f"ROI json missing cameras: {missing} (file={path})")
    return rois


def merge_roi_payload(
    existing_path: Path | None,
    new_rois: dict[str, list[list[int]]],
    dataset_root: Path,
    episode: int,
    frame: int,
    selected: tuple[str, ...],
) -> dict:
    payload: dict = {
        "format": "polygon_xy_4pts",
        "mask_outside": True,
        "fill_value": 0,
        "source_dataset": str(dataset_root),
        "sample_episode": episode,
        "sample_frame": frame,
        "selected_cameras": [CAMERA_TO_ALIAS[c] for c in selected],
    }
    if existing_path is not None and existing_path.is_file():
        try:
            with existing_path.open() as f:
                old = json.load(f)
            for cam in ALL_CAMERAS:
                alias = CAMERA_TO_ALIAS[cam]
                if cam in old and old[cam] is not None:
                    payload[cam] = old[cam]
                    payload[alias] = old[cam]
                elif alias in old and old[alias] is not None:
                    payload[cam] = old[alias]
                    payload[alias] = old[alias]
        except (json.JSONDecodeError, OSError) as exc:
            print(f"warn: cannot merge old ROI json: {exc}")

    for cam, poly in new_rois.items():
        alias = CAMERA_TO_ALIAS[cam]
        payload[cam] = poly
        payload[alias] = poly
    return payload


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    images_root = dataset_root / "images"
    if not images_root.is_dir():
        raise SystemExit(f"images/ not found under {dataset_root}")

    if args.inplace and args.output_root is not None:
        raise SystemExit("--inplace 与 --output-root 不能同时使用")

    if args.inplace:
        output_root = dataset_root
        dst_images = images_root
    else:
        output_root = (
            args.output_root.resolve()
            if args.output_root is not None
            else dataset_root.parent / f"{dataset_root.name}_roi_masked"
        )
        dst_images = output_root / "images"
        output_root.mkdir(parents=True, exist_ok=True)

    cam_choice = args.cameras if args.cameras is not None else "ask"
    cameras = prompt_cameras() if cam_choice == "ask" else resolve_cameras(cam_choice)
    print(
        "Selected cameras: "
        + ", ".join(f"{CAMERA_TO_ALIAS[c]} ({c})" for c in cameras)
    )

    rois: dict[str, list[list[int]]] = {}
    if args.roi_json is not None:
        rois = load_rois_from_json(args.roi_json, cameras)
        print(f"Loaded ROI from {args.roi_json}")
    else:
        for cam in cameras:
            alias = CAMERA_TO_ALIAS[cam]
            sample = _find_sample_image(images_root, cam, args.episode, args.frame)
            print(f"Sample for {alias} ({cam}): {sample}")
            img = cv2.imread(str(sample), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f"Cannot read {sample}")
            poly = select_quad_roi(img, f"{alias} {cam}")
            if poly is None:
                raise SystemExit("ROI selection cancelled")
            rois[cam] = poly
            preview_masked(img, poly, f"preview {alias}")

    meta_out = output_root / "meta"
    meta_out.mkdir(parents=True, exist_ok=True)
    roi_path = meta_out / "roi_params.json"
    # merge with existing roi file / --roi-json so partial updates keep the other cam
    merge_src = roi_path if roi_path.is_file() else args.roi_json
    roi_payload = merge_roi_payload(
        merge_src,
        rois,
        dataset_root,
        args.episode,
        args.frame,
        cameras,
    )
    with roi_path.open("w", encoding="utf-8") as f:
        json.dump(roi_payload, f, indent=2)
    print(f"Wrote {roi_path}")
    for cam in cameras:
        print(f"  {CAMERA_TO_ALIAS[cam]} ({cam}): {rois[cam]}")

    if args.preview_only:
        print("--preview-only: stop before batch mask")
        return

    if not args.inplace and not args.skip_copy_meta:
        copy_or_link_sidecar(dataset_root, output_root)
        link_unselected_cameras(images_root, dst_images, cameras)

    jobs_meta = collect_image_jobs(images_root, dst_images, cameras)
    if not jobs_meta:
        raise SystemExit(f"No jpg found under {images_root} for {cameras}")

    work: list[tuple[str, str, list[list[int]], int]] = [
        (src, dst, rois[cam], 95) for cam, src, dst in jobs_meta
    ]
    print(f"Masking {len(work)} images -> {dst_images}  (workers={args.workers})")

    if args.workers <= 1:
        for item in tqdm(work, desc="mask"):
            _mask_one(item)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_mask_one, item) for item in work]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="mask"):
                fut.result()

    print("Done.")
    print(f"  cameras: {', '.join(CAMERA_TO_ALIAS[c] for c in cameras)}")
    print(f"  output:  {output_root}")
    print(f"  roi:     {roi_path}")


if __name__ == "__main__":
    main()
