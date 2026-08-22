"""LeRobot / leobot image-sequence dataset loader for robotfm training.

Reads parquet + on-disk JPEG sequences (``leobot_image_sequence_v1``) and
returns the same batch keys as ``EpisodeDataset``:
``obs_images``, ``obs_state``, ``action``, ``action_mask``.

State/action concatenate the arm vector with gripper scalar(s):
- single-arm: ``observation.gripper`` / ``action.gripper`` → +1-D
- left-only: ``*.left_gripper`` (no ``right_gripper``) → +1-D
- dual-arm: ``*.left_gripper`` + ``*.right_gripper`` → +2-D
Camera feature keys ``observation.images.<name>`` map to short names ``<name>``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from robotfm.data.dataset import apply_hw_crop, color_jitter_images, crop_hw_box
from robotfm.data.action_delta import (
    flow_history_from_phys,
    joint_mask_from_names,
    overlay_joint_delta_action_stats,
    subtract_joint_pose,
)
from robotfm.data.stats import is_limits_mode, normalize, validate_norm_mode
from robotfm.data.uint8_cache import Uint8ImageCache, resolve_cache_dir
from robotfm.types import EpisodeMeta

IMAGE_FEATURE_PREFIX = "observation.images."
DEPTH_FEATURE_PREFIX = "observation.depth."

# Gripper feature layouts supported by this loader / stats merge.
_SINGLE_GRIPPER = ("gripper",)
_LEFT_GRIPPER = ("left_gripper",)
_RIGHT_GRIPPER = ("right_gripper",)
_DUAL_GRIPPER = ("left_gripper", "right_gripper")


def _load_depth_sources(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "meta" / "depth_sources.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        doc = json.load(f)
    return doc.get("sources", {})


def _decode_packed_depth_png(packed_rgb: np.ndarray) -> np.ndarray:
    if packed_rgb.ndim != 3 or packed_rgb.shape[2] != 3:
        raise ValueError(f"packed depth must be HxWx3, got {packed_rgb.shape}")
    high = packed_rgb[..., 0].astype(np.uint16)
    low = packed_rgb[..., 1].astype(np.uint16)
    return (high << 8) | low


def _raw_depth_to_normalized(
    raw: np.ndarray,
    scale_mm_per_raw_unit: float,
    min_depth_mm: float,
    max_depth_mm: float,
    invalid_raw_value: int = 0,
) -> np.ndarray:
    depth_mm = raw.astype(np.float32) * float(scale_mm_per_raw_unit)
    denom = max(max_depth_mm - min_depth_mm, 1e-6)
    norm = np.clip((depth_mm - min_depth_mm) / denom, 0.0, 1.0).astype(np.float32)
    norm[raw == invalid_raw_value] = 0.0
    return norm[None, ...]


def _center_crop_hw(arr: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h == crop_size and w == crop_size:
        return arr
    if h < crop_size or w < crop_size:
        raise ValueError(f"Cannot crop {h}x{w} to {crop_size}")
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return arr[top : top + crop_size, left : left + crop_size]


def _depth_rel_from_image_rel(img_rel: str, camera: str) -> str:
    rel = img_rel.replace("images/", "depth/", 1)
    rel = rel.replace(f"observation.images.{camera}", f"observation.depth.{camera}", 1)
    return str(Path(rel).with_suffix(".png"))


def _load_packed_depth(
    path: Path,
    *,
    scale_mm: float,
    invalid_raw: int,
    min_mm: float,
    max_mm: float,
    resize_size: int | None,
    pre_crop_size: int | None,
) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read depth: {path}")
    packed = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raw = _decode_packed_depth_png(packed)
    if pre_crop_size is not None:
        raw = _center_crop_hw(raw, pre_crop_size)
    if resize_size is not None:
        h, w = raw.shape[:2]
        if h != resize_size or w != resize_size:
            raw = cv2.resize(
                raw, (resize_size, resize_size), interpolation=cv2.INTER_NEAREST
            )
    return _raw_depth_to_normalized(raw, scale_mm, min_mm, max_mm, invalid_raw)


def is_lerobot_image_sequence_root(run_dir: Path) -> bool:
    """True if ``run_dir`` looks like a leobot/LeRobot image-sequence dataset."""
    info_path = Path(run_dir) / "meta" / "info.json"
    if not info_path.is_file():
        return False
    try:
        with info_path.open() as f:
            info = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return info.get("image_storage") == "image_sequence"


def _short_camera_name(feature_key: str) -> str:
    if feature_key.startswith(IMAGE_FEATURE_PREFIX):
        return feature_key[len(IMAGE_FEATURE_PREFIX) :]
    return feature_key


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    if chunks_size <= 0:
        raise ValueError(f"chunks_size must be > 0, got {chunks_size}")
    return episode_index // chunks_size


def _format_data_path(template: str, episode_index: int, chunks_size: int) -> str:
    chunk = _episode_chunk(episode_index, chunks_size)
    return template.format(episode_chunk=chunk, episode_index=episode_index)


def load_lerobot_info(run_dir: Path) -> dict[str, Any]:
    with (Path(run_dir) / "meta" / "info.json").open() as f:
        return json.load(f)


def resolve_gripper_suffixes(features: dict[str, Any]) -> tuple[str, ...]:
    """Return gripper name suffixes present for both state and action.

    Supports:
    - ``()``: no gripper columns
    - ``(\"gripper\",)``: single-arm
    - ``(\"left_gripper\",)``: left arm only (e.g. with_depth left subset)
    - ``(\"right_gripper\",)``: right arm only
    - ``(\"left_gripper\", \"right_gripper\")``: dual-arm (openarm etc.)
    """
    def _present(prefix: str, suffixes: tuple[str, ...]) -> bool:
        return all(f"{prefix}.{s}" in features for s in suffixes)

    dual_state = _present("observation", _DUAL_GRIPPER)
    dual_action = _present("action", _DUAL_GRIPPER)
    single_state = _present("observation", _SINGLE_GRIPPER)
    single_action = _present("action", _SINGLE_GRIPPER)
    left_state = _present("observation", _LEFT_GRIPPER)
    left_action = _present("action", _LEFT_GRIPPER)
    right_state = _present("observation", _RIGHT_GRIPPER)
    right_action = _present("action", _RIGHT_GRIPPER)

    if dual_state or dual_action:
        if dual_state != dual_action:
            raise ValueError(
                "Dual-arm grippers require both observation.{left,right}_gripper "
                "and action.{left,right}_gripper"
            )
        if single_state or single_action:
            raise ValueError(
                "Cannot mix single gripper (*.gripper) with dual "
                "(*.left_gripper / *.right_gripper)"
            )
        return _DUAL_GRIPPER

    if single_state != single_action:
        raise ValueError(
            "action.gripper and observation.gripper must both exist or both be absent"
        )
    if single_state:
        if left_state or left_action or right_state or right_action:
            raise ValueError(
                "Cannot mix single gripper (*.gripper) with "
                "*.left_gripper / *.right_gripper"
            )
        return _SINGLE_GRIPPER

    if left_state != left_action:
        raise ValueError(
            "action.left_gripper and observation.left_gripper must both exist "
            "or both be absent"
        )
    if right_state != right_action:
        raise ValueError(
            "action.right_gripper and observation.right_gripper must both exist "
            "or both be absent"
        )
    if left_state:
        return _LEFT_GRIPPER
    if right_state:
        return _RIGHT_GRIPPER
    return ()


def build_episode_meta_from_info(
    info: dict[str, Any],
    *,
    backend: str = "real_robot",
    embodiment: str | None = None,
) -> EpisodeMeta:
    """Build robotfm ``EpisodeMeta`` from LeRobot ``meta/info.json``."""
    features = info["features"]
    cameras: dict[str, dict[str, int]] = {}
    for key, spec in features.items():
        if not key.startswith(IMAGE_FEATURE_PREFIX):
            continue
        shape = spec["shape"]
        cameras[_short_camera_name(key)] = {
            "height": int(shape[0]),
            "width": int(shape[1]),
            "channels": int(shape[2]) if len(shape) > 2 else 3,
        }
    if not cameras:
        raise ValueError("No observation.images.* features in info.json")

    action_dim_arm = int(features["action"]["shape"][0])
    state_dim_arm = int(features["observation.state"]["shape"][0])
    gripper_suffixes = resolve_gripper_suffixes(features)
    gripper_extra = len(gripper_suffixes)
    action_dim = action_dim_arm + gripper_extra
    state_dim = state_dim_arm + gripper_extra

    state_names = [f"state_{i}" for i in range(state_dim_arm)]
    action_names = [f"action_{i}" for i in range(action_dim_arm)]
    state_names.extend(gripper_suffixes)
    action_names.extend(gripper_suffixes)

    robot_type = str(info.get("robot_type") or "unknown")
    return EpisodeMeta(
        backend=backend,
        embodiment=embodiment or robot_type,
        fps=int(info["fps"]),
        cameras=cameras,
        state_dim=state_dim,
        action_dim=action_dim,
        state_names=state_names,
        action_names=action_names,
        num_episodes=int(info.get("total_episodes", 0)),
        task="",
        created_at="",
    )


def _center_crop_hwc(img: np.ndarray, crop_size: int) -> np.ndarray:
    """Center-crop HWC array to ``crop_size×crop_size``."""
    h, w = img.shape[:2]
    if h == crop_size and w == crop_size:
        return img
    if h < crop_size or w < crop_size:
        raise ValueError(f"Cannot crop {h}x{w} to {crop_size}")
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return img[top : top + crop_size, left : left + crop_size]


def _load_image_rgb(
    path: Path,
    resize_size: int | None = None,
    pre_crop_size: int | None = None,
) -> np.ndarray:
    """Decode JPEG via OpenCV (faster than PIL); optional center pre-crop then resize.

    When only ``resize_size`` is set (no pre-crop), try half-resolution JPEG decode
    first (``IMREAD_REDUCED_COLOR_2``), then ``cv2.resize`` to target. This avoids
    bilinear interpolate on full 1280x720 float tensors (dual-cam n_obs hot path).

    With ``pre_crop_size`` (e.g. 720 on 1280×720), always decode full-res so the
    square crop is exact, then resize to ``resize_size``.
    """
    path_str = str(path)
    bgr = None
    # Half-res decode only when we are not center-cropping first (half of 720 < 720).
    if resize_size is not None and pre_crop_size is None:
        half = cv2.imread(path_str, cv2.IMREAD_REDUCED_COLOR_2)
        if half is not None and min(half.shape[:2]) >= resize_size:
            bgr = half
    if bgr is None:
        bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if pre_crop_size is not None:
        bgr = _center_crop_hwc(bgr, pre_crop_size)
    if resize_size is not None:
        h, w = bgr.shape[:2]
        if h != resize_size or w != resize_size:
            bgr = cv2.resize(
                bgr, (resize_size, resize_size), interpolation=cv2.INTER_AREA
            )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _vector_column_to_array(values: list) -> np.ndarray:
    """Convert parquet fixed_size_list / list column values to (T, D) float32."""
    return np.asarray(values, dtype=np.float32)


def _gripper_column_to_array(values: list) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def load_episode_arrays_from_parquet(
    run_dir: Path,
    episode_index: int,
    info: dict[str, Any],
    camera_feature_keys: list[str],
) -> dict[str, Any]:
    """Load one episode: state/action arrays + per-camera relative image paths."""
    data_path = _format_data_path(
        info["data_path"], episode_index, int(info["chunks_size"])
    )
    parquet_path = Path(run_dir) / data_path
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing parquet: {parquet_path}")

    table = pq.read_table(parquet_path)
    data = table.to_pydict()
    t = table.num_rows

    state = _vector_column_to_array(data["observation.state"])
    action = _vector_column_to_array(data["action"])
    gripper_suffixes = resolve_gripper_suffixes(info["features"])
    if gripper_suffixes:
        state_grips = [
            _gripper_column_to_array(data[f"observation.{s}"]) for s in gripper_suffixes
        ]
        action_grips = [
            _gripper_column_to_array(data[f"action.{s}"]) for s in gripper_suffixes
        ]
        state = np.concatenate([state, *state_grips], axis=1)
        action = np.concatenate([action, *action_grips], axis=1)

    image_paths: dict[str, list[str]] = {}
    for feat_key in camera_feature_keys:
        short = _short_camera_name(feat_key)
        col = data[feat_key]
        paths = [row["path"] if isinstance(row, dict) else row[0] for row in col]
        if len(paths) != t:
            raise ValueError(f"{feat_key} path count {len(paths)} != T={t}")
        image_paths[short] = paths

    return {
        "state": state.astype(np.float32, copy=False),
        "action": action.astype(np.float32, copy=False),
        "image_paths": image_paths,
        "length": t,
    }


def list_episode_indices(run_dir: Path, info: dict[str, Any]) -> list[int]:
    """Episode indices present on disk (prefer episodes.jsonl, else 0..N-1).

    Skips entries whose parquet is missing so partial datasets still load.
    """
    run_dir = Path(run_dir)
    chunks_size = int(info["chunks_size"])
    data_path_tmpl = info["data_path"]
    episodes_path = run_dir / "meta" / "episodes.jsonl"
    if episodes_path.is_file():
        candidates: list[int] = []
        with episodes_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                candidates.append(int(json.loads(line)["episode_index"]))
    else:
        candidates = list(range(int(info.get("total_episodes", 0))))

    indices: list[int] = []
    missing = 0
    for ep_id in candidates:
        parquet_path = run_dir / _format_data_path(data_path_tmpl, ep_id, chunks_size)
        if parquet_path.is_file():
            indices.append(ep_id)
        else:
            missing += 1
    if missing:
        print(
            f"warning: skipped {missing} episode(s) with missing parquet under {run_dir}"
        )
    return indices


class LeRobotImageSequenceDataset(Dataset):
    """Frame-indexed dataset over a leobot image-sequence root."""

    def __init__(
        self,
        run_dir: Path,
        n_obs_steps: int = 2,
        horizon: int = 16,
        n_action_steps: int | None = None,
        drop_n_last_frames: int | None = None,
        stats: dict[str, np.ndarray] | None = None,
        normalize: bool = True,
        norm_mode: str = "gaussian",
        pre_crop_size: int | None = None,
        resize_size: int | None = None,
        crop_size: int | None = 84,
        random_crop: bool = True,
        color_jitter_brightness: float = 0.0,
        color_jitter_contrast: float = 0.0,
        color_jitter_saturation: float = 0.0,
        color_jitter_hue: float = 0.0,
        defer_augment: bool = False,
        uint8_cache: bool = False,
        uint8_cache_dir: str | Path | None = None,
        predict_joint_delta: bool = False,
        depth_cameras: tuple[str, ...] | list[str] = (),
        depth_min_mm: float = 50.0,
        depth_max_mm: float = 500.0,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.info = load_lerobot_info(self.run_dir)
        if self.info.get("image_storage") != "image_sequence":
            raise ValueError(
                f"{self.run_dir} is not an image_sequence dataset "
                f"(image_storage={self.info.get('image_storage')!r})"
            )
        self.meta = build_episode_meta_from_info(self.info)
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.stats = stats
        self.normalize = normalize
        self.norm_mode = validate_norm_mode(norm_mode)
        if self.normalize and self.stats is not None and is_limits_mode(self.norm_mode):
            missing = [
                k
                for k in ("state_min", "state_max", "action_min", "action_max")
                if k not in self.stats
            ]
            if missing:
                raise ValueError(
                    f"norm_mode={self.norm_mode!r} requires {missing} in stats; "
                    "delete stats.json and recompute, or call ensure_stats(..., "
                    f"{self.norm_mode!r})"
                )
        self.pre_crop_size = pre_crop_size
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.color_jitter_brightness = color_jitter_brightness
        self.color_jitter_contrast = color_jitter_contrast
        self.color_jitter_saturation = color_jitter_saturation
        self.color_jitter_hue = color_jitter_hue
        self.defer_augment = bool(defer_augment)
        self.predict_joint_delta = bool(predict_joint_delta)
        self.depth_cameras = tuple(depth_cameras)
        self.depth_min_mm = float(depth_min_mm)
        self.depth_max_mm = float(depth_max_mm)
        self._depth_meta: dict[str, dict[str, Any]] = {}
        if self.depth_cameras:
            sources = _load_depth_sources(self.run_dir)
            for cam in self.depth_cameras:
                feature = f"{DEPTH_FEATURE_PREFIX}{cam}"
                if feature not in sources:
                    raise KeyError(
                        f"depth camera {cam!r} missing from meta/depth_sources.json "
                        f"(expected {feature})"
                    )
                meta = sources[feature]
                storage = meta.get("storage") or meta.get("raw_format")
                if storage not in (None, "rgb8_y16_pack_png", "uint16"):
                    raise ValueError(
                        f"Unsupported depth storage for {feature}: {storage!r}"
                    )
                scale = meta.get("scale_mm_per_raw_unit")
                if scale is None:
                    raise KeyError(
                        f"{feature} missing scale_mm_per_raw_unit in depth_sources.json"
                    )
                self._depth_meta[cam] = {
                    "scale_mm_per_raw_unit": float(scale),
                    "invalid_raw_value": int(
                        meta.get("invalid_raw_value", meta.get("invalid_value", 0))
                    ),
                }
        self._joint_mask = joint_mask_from_names(self.meta.action_names, self.meta.action_dim)
        if self.predict_joint_delta and self.meta.state_dim != self.meta.action_dim:
            raise ValueError(
                "predict_joint_delta requires state_dim == action_dim "
                f"(got {self.meta.state_dim} vs {self.meta.action_dim})"
            )
        self._image_cache: Uint8ImageCache | None = None

        if drop_n_last_frames is None:
            drop_n_last_frames = 0
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")
        if n_action_steps is not None and n_action_steps > horizon:
            raise ValueError(f"n_action_steps ({n_action_steps}) must be <= horizon ({horizon})")
        self.n_action_steps = n_action_steps
        self.drop_n_last_frames = drop_n_last_frames

        features = self.info["features"]
        self._camera_feature_keys = [
            k for k in features if k.startswith(IMAGE_FEATURE_PREFIX)
        ]
        # Preserve meta.camera_names order
        short_to_feat = {_short_camera_name(k): k for k in self._camera_feature_keys}
        self._camera_feature_keys = [short_to_feat[name] for name in self.meta.camera_names]

        episode_indices = list_episode_indices(self.run_dir, self.info)
        if not episode_indices:
            raise FileNotFoundError(f"No episodes listed under {self.run_dir}")

        self._episode_ids: list[int] = []
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._image_paths: list[dict[str, list[str]]] = []
        self._episode_lengths: list[int] = []
        self.index: list[tuple[int, int]] = []

        for ep_id in episode_indices:
            payload = load_episode_arrays_from_parquet(
                self.run_dir, ep_id, self.info, self._camera_feature_keys
            )
            length = int(payload["length"])
            if payload["state"].shape[1] != self.meta.state_dim:
                raise ValueError(
                    f"episode {ep_id} state dim {payload['state'].shape[1]} "
                    f"!= meta.state_dim {self.meta.state_dim}"
                )
            if payload["action"].shape[1] != self.meta.action_dim:
                raise ValueError(
                    f"episode {ep_id} action dim {payload['action'].shape[1]} "
                    f"!= meta.action_dim {self.meta.action_dim}"
                )
            ep_local = len(self._episode_ids)
            self._episode_ids.append(ep_id)
            self._states.append(payload["state"])
            self._actions.append(payload["action"])
            self._image_paths.append(payload["image_paths"])
            self._episode_lengths.append(length)
            usable = length - self.drop_n_last_frames
            if usable <= 0:
                continue
            for t in range(usable):
                self.index.append((ep_local, t))

        if not self.index:
            raise ValueError(
                f"No usable frames after drop_n_last_frames={self.drop_n_last_frames}; "
                "check episode lengths / n_action_steps."
            )
        self.meta.num_episodes = len(self._episode_ids)
        if self.predict_joint_delta:
            if self.stats is None:
                raise ValueError("predict_joint_delta requires stats")
            overlay_joint_delta_action_stats(
                self.stats,
                self._states,
                self._actions,
                horizon=self.horizon,
                joint_mask=self._joint_mask,
            )

        if uint8_cache:
            cache_path = resolve_cache_dir(
                self.run_dir,
                resize_size=self.resize_size,
                pre_crop_size=self.pre_crop_size,
                cache_dir=uint8_cache_dir,
            )
            if not (cache_path / "meta.json").is_file():
                raise FileNotFoundError(
                    f"uint8_cache enabled but missing {cache_path / 'meta.json'}. "
                    "Build it with: python scripts/build_uint8_image_cache.py "
                    f"--run-dir {self.run_dir} --resize-size {self.resize_size}"
                    + (
                        f" --pre-crop-size {self.pre_crop_size}"
                        if self.pre_crop_size is not None
                        else ""
                    )
                )
            self._image_cache = Uint8ImageCache(cache_path)
            if self._image_cache.cameras != list(self.meta.camera_names):
                raise ValueError(
                    f"cache cameras {self._image_cache.cameras} != "
                    f"dataset cameras {list(self.meta.camera_names)}"
                )
            if not self._image_cache.matches_episodes(
                self._episode_ids, self._episode_lengths
            ):
                raise ValueError(
                    f"uint8 cache at {cache_path} does not match episode ids/lengths "
                    f"under {self.run_dir}; rebuild with --overwrite"
                )
            if self.resize_size is not None and (
                self._image_cache.height != self.resize_size
                or self._image_cache.width != self.resize_size
            ):
                raise ValueError(
                    f"cache HxW={self._image_cache.height}x{self._image_cache.width} "
                    f"!= resize_size={self.resize_size}"
                )
            cached_pre = self._image_cache.meta.get("pre_crop_size")
            if cached_pre != self.pre_crop_size and (
                cached_pre is not None or self.pre_crop_size is not None
            ):
                raise ValueError(
                    f"cache pre_crop_size={cached_pre} != "
                    f"dataset pre_crop_size={self.pre_crop_size}; rebuild cache"
                )

    def __len__(self) -> int:
        return len(self.index)

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        if not self.normalize or self.stats is None:
            return state
        return normalize(state, self.stats, prefix="state", mode=self.norm_mode)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        if not self.normalize or self.stats is None:
            return action
        return normalize(action, self.stats, prefix="action", mode=self.norm_mode)

    def _load_cam_frames(self, ep_local: int, cam: str, frame_indices: list[int]) -> np.ndarray:
        if self._image_cache is not None:
            return self._image_cache.load_cam_frames(ep_local, cam, frame_indices)
        paths = self._image_paths[ep_local][cam]
        frames = []
        for fi in frame_indices:
            img_path = self.run_dir / paths[fi]
            frames.append(
                _load_image_rgb(
                    img_path,
                    resize_size=self.resize_size,
                    pre_crop_size=self.pre_crop_size,
                )
            )
        return np.stack(frames, axis=0)

    def _load_depth_cam_window(
        self, ep_local: int, camera: str, frame_indices: list[int]
    ) -> np.ndarray:
        meta = self._depth_meta[camera]
        paths = self._image_paths[ep_local][camera]
        frames = []
        for fi in frame_indices:
            rel = _depth_rel_from_image_rel(paths[fi], camera)
            frames.append(
                _load_packed_depth(
                    self.run_dir / rel,
                    scale_mm=meta["scale_mm_per_raw_unit"],
                    invalid_raw=meta["invalid_raw_value"],
                    min_mm=self.depth_min_mm,
                    max_mm=self.depth_max_mm,
                    resize_size=self.resize_size,
                    pre_crop_size=self.pre_crop_size,
                )
            )
        return np.stack(frames, axis=0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_local, t = self.index[idx]
        length = self._episode_lengths[ep_local]
        state_all = self._states[ep_local]
        action_all = self._actions[ep_local]

        obs_start = max(0, t - self.n_obs_steps + 1)
        obs_indices = list(range(obs_start, t + 1))
        while len(obs_indices) < self.n_obs_steps:
            obs_indices.insert(0, obs_indices[0])

        images = []
        for cam in self.meta.camera_names:
            cam_frames = self._load_cam_frames(ep_local, cam, obs_indices)
            cam_frames = np.transpose(cam_frames, (0, 3, 1, 2))
            if self.defer_augment:
                # Keep uint8; train loop does /255 on GPU after H2D.
                cam_frames = np.ascontiguousarray(cam_frames)
            else:
                cam_frames = cam_frames.astype(np.float32) / 255.0
            images.append(torch.from_numpy(cam_frames))

        obs_images = torch.stack(images, dim=0)
        obs_depth = None
        if self.depth_cameras:
            depth_cams = []
            for cam in self.depth_cameras:
                depth_cams.append(
                    torch.from_numpy(
                        self._load_depth_cam_window(ep_local, cam, obs_indices)
                    )
                )
            obs_depth = torch.stack(depth_cams, dim=0)
        if not self.defer_augment:
            if self.crop_size is not None:
                _, _, h, w = obs_images.shape[-4:]
                if h != self.crop_size or w != self.crop_size:
                    top, left = crop_hw_box(h, w, self.crop_size, self.random_crop)
                    obs_images = apply_hw_crop(obs_images, top, left, self.crop_size)
                    if obs_depth is not None:
                        obs_depth = apply_hw_crop(obs_depth, top, left, self.crop_size)
            if self.random_crop:
                obs_images = color_jitter_images(
                    obs_images,
                    brightness=self.color_jitter_brightness,
                    contrast=self.color_jitter_contrast,
                    saturation=self.color_jitter_saturation,
                    hue=self.color_jitter_hue,
                )

        state_phys = state_all[obs_indices].astype(np.float32)
        state = self._normalize_state(state_phys)
        if self.predict_joint_delta:
            if self.stats is None:
                raise ValueError("predict_joint_delta requires stats")
            flow_hist = flow_history_from_phys(
                state_phys,
                self.stats,
                self.norm_mode,
                predict_joint_delta=True,
                joint_mask=self._joint_mask,
            )
        else:
            flow_hist = state

        action_end = min(t + self.horizon, length)
        valid_len = action_end - t
        actions = action_all[t:action_end].astype(np.float32)
        if self.predict_joint_delta:
            q_now = state_all[t].astype(np.float32)
            actions = subtract_joint_pose(actions, q_now, self._joint_mask)

        # 原版 A2A sampler：末尾不够时重复最后一帧，而非 0-pad
        mask = np.zeros((self.horizon, 1), dtype=np.float32)
        mask[:valid_len] = 1.0
        if valid_len < self.horizon:
            last = actions[-1:]
            pad = np.repeat(last, self.horizon - valid_len, axis=0)
            actions = np.concatenate([actions, pad], axis=0)
        actions = self._normalize_action(actions)

        out = {
            "obs_images": obs_images,
            "obs_state": torch.from_numpy(state),
            "obs_history": torch.from_numpy(np.asarray(flow_hist, dtype=np.float32)),
            "action": torch.from_numpy(actions),
            "action_mask": torch.from_numpy(mask),
        }
        if obs_depth is not None:
            out["obs_depth"] = obs_depth
        return out


def iter_lerobot_state_action(
    run_dir: Path,
) -> tuple[EpisodeMeta, list[np.ndarray], list[np.ndarray]]:
    """Yield meta plus per-episode state/action arrays (for stats)."""
    run_dir = Path(run_dir)
    info = load_lerobot_info(run_dir)
    meta = build_episode_meta_from_info(info)
    features = info["features"]
    camera_keys = [k for k in features if k.startswith(IMAGE_FEATURE_PREFIX)]
    short_to_feat = {_short_camera_name(k): k for k in camera_keys}
    camera_keys = [short_to_feat[name] for name in meta.camera_names]

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for ep_id in list_episode_indices(run_dir, info):
        payload = load_episode_arrays_from_parquet(run_dir, ep_id, info, camera_keys)
        states.append(payload["state"])
        actions.append(payload["action"])
    return meta, states, actions
