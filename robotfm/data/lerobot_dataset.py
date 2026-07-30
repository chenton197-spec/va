"""LeRobot / leobot image-sequence dataset loader for robotfm training.

Reads parquet + on-disk JPEG sequences (``leobot_image_sequence_v1``) and
returns the same batch keys as ``EpisodeDataset``:
``obs_images``, ``obs_state``, ``action``, ``action_mask``.

State/action concatenate the 6-D arm vector with the scalar gripper → 7-D.
Camera feature keys ``observation.images.<name>`` map to short names ``<name>``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from robotfm.data.dataset import color_jitter_images, crop_images, resize_images
from robotfm.data.stats import is_limits_mode, normalize, validate_norm_mode
from robotfm.types import EpisodeMeta

IMAGE_FEATURE_PREFIX = "observation.images."


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
    has_action_gripper = "action.gripper" in features
    has_state_gripper = "observation.gripper" in features
    if has_action_gripper != has_state_gripper:
        raise ValueError("action.gripper and observation.gripper must both exist or both be absent")
    gripper_extra = 1 if has_action_gripper else 0
    action_dim = action_dim_arm + gripper_extra
    state_dim = state_dim_arm + gripper_extra

    state_names = [f"state_{i}" for i in range(state_dim_arm)]
    action_names = [f"action_{i}" for i in range(action_dim_arm)]
    if gripper_extra:
        state_names.append("gripper")
        action_names.append("gripper")

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


def _load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


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
    if "observation.gripper" in data:
        state = np.concatenate([state, _gripper_column_to_array(data["observation.gripper"])], axis=1)
    if "action.gripper" in data:
        action = np.concatenate([action, _gripper_column_to_array(data["action.gripper"])], axis=1)

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
    """Episode indices present on disk (prefer episodes.jsonl, else 0..N-1)."""
    episodes_path = Path(run_dir) / "meta" / "episodes.jsonl"
    if episodes_path.is_file():
        indices: list[int] = []
        with episodes_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                indices.append(int(json.loads(line)["episode_index"]))
        return indices
    total = int(info.get("total_episodes", 0))
    return list(range(total))


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
        resize_size: int | None = None,
        crop_size: int | None = 84,
        random_crop: bool = True,
        color_jitter_brightness: float = 0.0,
        color_jitter_contrast: float = 0.0,
        color_jitter_saturation: float = 0.0,
        color_jitter_hue: float = 0.0,
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
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.color_jitter_brightness = color_jitter_brightness
        self.color_jitter_contrast = color_jitter_contrast
        self.color_jitter_saturation = color_jitter_saturation
        self.color_jitter_hue = color_jitter_hue

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
        paths = self._image_paths[ep_local][cam]
        frames = []
        for fi in frame_indices:
            img_path = self.run_dir / paths[fi]
            frames.append(_load_image_rgb(img_path))
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
            cam_frames = np.transpose(cam_frames, (0, 3, 1, 2)).astype(np.float32) / 255.0
            images.append(torch.from_numpy(cam_frames))

        obs_images = torch.stack(images, dim=0)
        obs_images = resize_images(obs_images, self.resize_size)
        obs_images = crop_images(obs_images, self.crop_size, random=self.random_crop)
        if self.random_crop:
            obs_images = color_jitter_images(
                obs_images,
                brightness=self.color_jitter_brightness,
                contrast=self.color_jitter_contrast,
                saturation=self.color_jitter_saturation,
                hue=self.color_jitter_hue,
            )

        state = self._normalize_state(state_all[obs_indices].astype(np.float32))

        action_end = min(t + self.horizon, length)
        valid_len = action_end - t
        actions = action_all[t:action_end].astype(np.float32)

        mask = np.zeros((self.horizon, 1), dtype=np.float32)
        mask[:valid_len] = 1.0
        if valid_len < self.horizon:
            pad = np.zeros((self.horizon - valid_len, self.meta.action_dim), dtype=np.float32)
            actions = np.concatenate([actions, pad], axis=0)
        actions = self._normalize_action(actions)

        return {
            "obs_images": obs_images,
            "obs_state": torch.from_numpy(state),
            "action": torch.from_numpy(actions),
            "action_mask": torch.from_numpy(mask),
        }


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
