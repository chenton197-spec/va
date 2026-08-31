"""Disk-backed uint8 RGB image cache for LeRobot image-sequence datasets.

Layout under ``{run_dir}/cache/uint8_rgb_{H}x{W}/``::

    meta.json          cameras, episode offsets, shape, image_size
    {camera}.dat       memmap (total_frames, H, W, 3) uint8 RGB

Build once with ``scripts/build_uint8_image_cache.py``, then train with
``dataset.uint8_cache: true`` to skip per-step JPEG decode.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from robotfm.data.dataset import parse_image_hw

CACHE_VERSION = 1
CACHE_DIR_PREFIX = "uint8_rgb_"


def cache_dir_for(run_dir: Path, height: int, width: int) -> Path:
    name = f"{CACHE_DIR_PREFIX}{height}x{width}"
    return Path(run_dir) / "cache" / name


def resolve_cache_dir(
    run_dir: Path,
    *,
    image_size: int | tuple[int, int] | list[int] | None,
    cache_dir: str | Path | None = None,
) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    hw = parse_image_hw(image_size)
    if hw is None:
        raise ValueError(
            "uint8 cache requires image_size (or an explicit cache_dir) "
            "so the on-disk resolution is known"
        )
    return cache_dir_for(run_dir, hw[0], hw[1])


def _resize_hw(
    arr: np.ndarray,
    height: int,
    width: int,
    interpolation: int,
) -> np.ndarray:
    h, w = arr.shape[:2]
    if h == height and w == width:
        return arr
    return cv2.resize(arr, (width, height), interpolation=interpolation)


def _load_image_rgb(path: Path, height: int, width: int) -> np.ndarray:
    path_str = str(path)
    bgr = None
    half = cv2.imread(path_str, cv2.IMREAD_REDUCED_COLOR_2)
    if half is not None and min(half.shape[:2]) >= min(height, width):
        bgr = half
    if bgr is None:
        bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    bgr = _resize_hw(bgr, height, width, cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _decode_job(
    args: tuple[str, str, int, int, int],
) -> tuple[str, int, np.ndarray]:
    cam, path_str, flat_index, height, width = args
    rgb = _load_image_rgb(Path(path_str), height, width)
    if rgb.shape[0] != height or rgb.shape[1] != width:
        raise ValueError(
            f"decoded shape {rgb.shape} != ({height},{width},3) for {path_str}"
        )
    return cam, flat_index, np.ascontiguousarray(rgb, dtype=np.uint8)


class Uint8ImageCache:
    """Read-only view over a built uint8 RGB memmap cache."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing cache meta: {meta_path}")
        with meta_path.open() as f:
            self.meta: dict[str, Any] = json.load(f)
        if int(self.meta.get("version", 0)) != CACHE_VERSION:
            raise ValueError(
                f"Unsupported cache version {self.meta.get('version')} "
                f"(expected {CACHE_VERSION}) at {meta_path}"
            )
        self.height = int(self.meta["height"])
        self.width = int(self.meta["width"])
        self.cameras: list[str] = list(self.meta["cameras"])
        self.total_frames = int(self.meta["total_frames"])
        self.episode_ids: list[int] = [int(x) for x in self.meta["episode_ids"]]
        self.episode_offsets: list[int] = [int(x) for x in self.meta["episode_offsets"]]
        self.episode_lengths: list[int] = [int(x) for x in self.meta["episode_lengths"]]

        self._maps: dict[str, np.memmap] = {}
        for cam in self.cameras:
            path = self.cache_dir / f"{cam}.dat"
            if not path.is_file():
                raise FileNotFoundError(f"Missing cache file: {path}")
            self._maps[cam] = np.memmap(
                path,
                dtype=np.uint8,
                mode="r",
                shape=(self.total_frames, self.height, self.width, 3),
            )

    def flat_indices(self, ep_local: int, frame_indices: list[int]) -> np.ndarray:
        base = self.episode_offsets[ep_local]
        length = self.episode_lengths[ep_local]
        idx = np.asarray(frame_indices, dtype=np.int64)
        if (idx < 0).any() or (idx >= length).any():
            raise IndexError(
                f"frame indices {frame_indices} out of range for episode "
                f"local={ep_local} length={length}"
            )
        return base + idx

    def load_cam_frames(
        self, ep_local: int, cam: str, frame_indices: list[int]
    ) -> np.ndarray:
        """Return (T, H, W, 3) uint8 RGB (copied out of memmap)."""
        flat = self.flat_indices(ep_local, frame_indices)
        return np.asarray(self._maps[cam][flat], dtype=np.uint8)

    def matches_episodes(self, episode_ids: list[int], lengths: list[int]) -> bool:
        return self.episode_ids == list(episode_ids) and self.episode_lengths == list(
            lengths
        )


def build_uint8_image_cache(
    run_dir: Path,
    *,
    image_size: int | tuple[int, int] | list[int],
    output_dir: Path | None = None,
    num_workers: int = 8,
    overwrite: bool = False,
) -> Path:
    """Decode all JPEGs into uint8 memmaps; return cache directory."""
    from robotfm.data.lerobot_dataset import (
        IMAGE_FEATURE_PREFIX,
        _short_camera_name,
        build_episode_meta_from_info,
        list_episode_indices,
        load_episode_arrays_from_parquet,
        load_lerobot_info,
    )

    hw = parse_image_hw(image_size)
    if hw is None:
        raise ValueError("image_size is required to build uint8 cache")
    height, width = hw

    run_dir = Path(run_dir)
    info = load_lerobot_info(run_dir)
    if info.get("image_storage") != "image_sequence":
        raise ValueError(f"{run_dir} is not an image_sequence dataset")

    meta = build_episode_meta_from_info(info)
    cameras = list(meta.camera_names)
    features = info["features"]
    short_to_feat = {
        _short_camera_name(k): k
        for k in features
        if k.startswith(IMAGE_FEATURE_PREFIX)
    }
    camera_feature_keys = [short_to_feat[name] for name in cameras]

    out_dir = (
        Path(output_dir) if output_dir is not None else cache_dir_for(run_dir, height, width)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    if meta_path.is_file() and not overwrite:
        existing = Uint8ImageCache(out_dir)
        print(
            f"cache already exists at {out_dir} "
            f"({existing.total_frames} frames); skip"
        )
        return out_dir

    episode_ids = list_episode_indices(run_dir, info)
    episode_offsets: list[int] = []
    episode_lengths: list[int] = []
    decode_args: list[tuple[str, str, int, int, int]] = []
    cursor = 0

    print(f"indexing episodes under {run_dir} ...")
    for ep_id in tqdm(episode_ids, desc="index"):
        payload = load_episode_arrays_from_parquet(
            run_dir, ep_id, info, camera_feature_keys
        )
        length = int(payload["length"])
        episode_offsets.append(cursor)
        episode_lengths.append(length)
        for t in range(length):
            flat = cursor + t
            for cam in cameras:
                rel = payload["image_paths"][cam][t]
                decode_args.append((cam, str(run_dir / rel), flat, height, width))
        cursor += length

    total_frames = cursor
    gib = total_frames * height * width * 3 * len(cameras) / (1024**3)
    print(
        f"building cache: episodes={len(episode_ids)} frames={total_frames} "
        f"cams={len(cameras)} {height}x{width} → {gib:.2f} GiB at {out_dir}"
    )

    maps: dict[str, np.memmap] = {}
    for cam in cameras:
        path = out_dir / f"{cam}.dat"
        if path.exists():
            path.unlink()
        maps[cam] = np.memmap(
            path,
            dtype=np.uint8,
            mode="w+",
            shape=(total_frames, height, width, 3),
        )

    workers = max(1, int(num_workers))
    chunk = max(workers * 64, 256)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        with tqdm(total=len(decode_args), desc="decode") as pbar:
            for start in range(0, len(decode_args), chunk):
                batch = decode_args[start : start + chunk]
                futures = [pool.submit(_decode_job, arg) for arg in batch]
                for fut in as_completed(futures):
                    cam, flat_index, rgb = fut.result()
                    maps[cam][flat_index] = rgb
                    pbar.update(1)
                if (start // chunk) % 20 == 0:
                    for m in maps.values():
                        m.flush()

    for m in maps.values():
        m.flush()

    meta_obj = {
        "version": CACHE_VERSION,
        "height": height,
        "width": width,
        "cameras": cameras,
        "total_frames": total_frames,
        "dtype": "uint8",
        "layout": "NHWC",
        "image_size": [height, width],
        "source_run_dir": str(run_dir.resolve()),
        "episode_ids": episode_ids,
        "episode_offsets": episode_offsets,
        "episode_lengths": episode_lengths,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_obj, f, indent=2)
        f.write("\n")
    print(f"wrote {meta_path}")
    return out_dir
