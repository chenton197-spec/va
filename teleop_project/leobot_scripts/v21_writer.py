"""A small writer for LeRobot v2.1 video and local PNG sequence datasets."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
import json
import math
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .sources import CameraFrame, DepthFrame, DepthMetadata
from .video import encode_episode_video

V21 = "v2.1"
# Legacy image-sequence metadata values accepted when reopening an existing
# dataset. New PNG and JPG datasets use IMAGE_SEQUENCE so they can coexist.
PNG_SEQUENCE = "leobot_png_v1"
JPG_SEQUENCE = "leobot_jpg_v1"
IMAGE_SEQUENCE = "leobot_image_sequence_v1"
CHUNK_SIZE = 1000
DEPTH_SIDECAR_VERSION = 1
DEPTH_CHUNK_TARGET_BYTES = 8 * 1024 * 1024
PNG_WORKERS_PER_CAMERA = 6
MAX_PNG_WORKERS = 12
PNG_PENDING_TASKS_PER_WORKER = 16
ImageStorage = Literal["video", "png", "jpg"]


@dataclass(frozen=True)
class CameraSpec:
    """The fixed schema of one RGB feature stored as video or a PNG sequence."""

    feature_name: str
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or self.shape[2] != 3:
            raise ValueError("Camera features must use an H x W x 3 RGB shape")


@dataclass(frozen=True)
class DepthCameraSpec:
    """The fixed schema and static metadata of one raw depth sidecar."""

    feature_name: str
    rgb_feature_name: str | None
    shape: tuple[int, int]
    metadata: DepthMetadata

    def __post_init__(self) -> None:
        if not self.feature_name.startswith("observation.depth."):
            raise ValueError("Depth feature names must use observation.depth.<name>")
        if self.rgb_feature_name is not None and not self.rgb_feature_name.startswith("observation.images."):
            raise ValueError("Depth sidecars must reference an observation.images.<name> feature")
        if len(self.shape) != 2 or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("Depth features must use a positive H x W shape")
        if not isinstance(self.metadata.raw_format, str) or not self.metadata.raw_format.strip():
            raise ValueError("Depth raw_format must be a non-empty string")
        if isinstance(self.metadata.invalid_value, bool) or not isinstance(
            self.metadata.invalid_value, (int, np.integer)
        ):
            raise ValueError("Depth invalid_value must be an integer")
        if not 0 <= int(self.metadata.invalid_value) <= np.iinfo(np.uint16).max:
            raise ValueError("Depth invalid_value must fit uint16")
        if self.rgb_feature_name is None and self.metadata.aligned_to_rgb:
            raise ValueError("Standalone depth sources cannot be marked aligned_to_rgb")
        if self.rgb_feature_name is not None and not self.metadata.aligned_to_rgb:
            raise ValueError("RGB-D depth sources must be marked aligned_to_rgb")
        if self.metadata.color_intrinsics is not None:
            intrinsics = np.asarray(self.metadata.color_intrinsics, dtype=float)
            if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
                raise ValueError("color_intrinsics must be a finite 3 x 3 matrix")


@dataclass(frozen=True)
class WriterConfig:
    root: Path
    robot_type: str
    fps: int
    joint_count: int
    include_gripper: bool = False
    scalar_actuator_names: tuple[str, ...] = ()
    cameras: tuple[CameraSpec, ...] = ()
    depth_cameras: tuple[DepthCameraSpec, ...] = ()
    image_storage: ImageStorage = "video"
    quality: int = 75

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.joint_count <= 0:
            raise ValueError("joint_count must be positive")
        if self.image_storage not in {"video", "png", "jpg"}:
            raise ValueError("image_storage must be 'video', 'png', or 'jpg'")
        if isinstance(self.quality, bool) or not isinstance(self.quality, int) or not 1 <= self.quality <= 100:
            raise ValueError("quality must be an integer from 1 to 100")
        if not isinstance(self.scalar_actuator_names, tuple):
            raise ValueError("scalar_actuator_names must be a tuple")
        if any(
            not isinstance(name, str) or not name.strip()
            for name in self.scalar_actuator_names
        ):
            raise ValueError("scalar_actuator_names must contain non-empty strings")
        if len(self.scalar_actuator_names) != len(set(self.scalar_actuator_names)):
            raise ValueError("scalar_actuator_names must be unique")
        if "gripper" in self.scalar_actuator_names:
            raise ValueError("scalar_actuator_names must not contain the legacy name 'gripper'")
        names = [camera.feature_name for camera in self.cameras]
        if len(names) != len(set(names)):
            raise ValueError("Camera feature names must be unique")
        depth_names = [camera.feature_name for camera in self.depth_cameras]
        if len(depth_names) != len(set(depth_names)):
            raise ValueError("Depth feature names must be unique")
        camera_shapes = {camera.feature_name: camera.shape for camera in self.cameras}
        for depth_camera in self.depth_cameras:
            if depth_camera.rgb_feature_name is None:
                continue
            rgb_shape = camera_shapes.get(depth_camera.rgb_feature_name)
            if rgb_shape is None:
                raise ValueError("Every depth sidecar must reference a configured RGB camera")
            if rgb_shape[:2] != depth_camera.shape:
                raise ValueError("Aligned RGB and depth shapes must have the same height and width")


@dataclass
class RecordedFrame:
    state: np.ndarray
    action: np.ndarray
    # Real episode-relative time when a camera-driven recorder owns the
    # timeline. Fixed-rate robot-only recordings leave it unset.
    timestamp_s: float | None = None
    cameras: dict[str, CameraFrame] = field(default_factory=dict)
    # Per-camera episode-relative times. A paired camera may legitimately be
    # slightly older than the master frame that emitted this row.
    camera_timestamps_s: dict[str, float] = field(default_factory=dict)
    depths: dict[str, DepthFrame] = field(default_factory=dict)
    gripper_state: float | None = None
    gripper_action: float | None = None
    actuator_states: dict[str, float] = field(default_factory=dict)
    actuator_actions: dict[str, float] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)


class V21DatasetWriter:
    """Write complete episodes atomically without importing ``lerobot``.

    ``video`` mode writes the LeRobotDataset v2.1 video layout. ``png`` and
    ``jpg`` write repository-specific image sequences; both can coexist in one
    dataset because every Parquet row stores its exact image path. A failed or
    discarded episode remains below ``.staging``.
    """

    def __init__(self, config: WriterConfig):
        self.config = config
        self.root = config.root
        self.meta_dir = self.root / "meta"
        self.staging_dir = self.root / ".staging"
        self._info = self._load_or_create_info()
        self._tasks = self._load_tasks()
        self._depth_sources = self._load_or_validate_depth_sources()
        self._active: _EpisodeBuffer | None = None

    @property
    def active(self) -> bool:
        return self._active is not None

    def begin_episode(self, task: str) -> int:
        if self._active is not None:
            raise RuntimeError("An episode is already active")
        if not task.strip():
            raise ValueError("task must be a non-empty string")
        episode_index = int(self._info["total_episodes"])
        stage = Path(tempfile.mkdtemp(prefix=f"episode_{episode_index:06d}_", dir=self.staging_dir))
        active = _EpisodeBuffer(episode_index=episode_index, task=task, stage=stage)
        try:
            self._initialize_depth_stores(active)
            self._start_png_writers(active)
        except Exception:
            self._shutdown_png_writers(active, cancel=True)
            shutil.rmtree(stage, ignore_errors=True)
            raise
        self._active = active
        return episode_index

    def append_frame(self, frame: RecordedFrame) -> int:
        active = self._require_active()
        self._raise_completed_png_errors(active)
        frame_index = len(active.frames)
        state = _float_vector(frame.state, self.config.joint_count, "state")
        action = _float_vector(frame.action, self.config.joint_count, "action")
        timestamp_s = _record_timestamp_s(frame.timestamp_s, frame_index, self.config.fps)
        if active.frames:
            previous_timestamp = active.frames[-1].timestamp_s
            assert previous_timestamp is not None
            if timestamp_s < previous_timestamp:
                raise ValueError("Recorded frame timestamps must be nondecreasing")
        if self.config.include_gripper:
            if frame.gripper_state is None or frame.gripper_action is None:
                raise ValueError("A gripper-enabled dataset requires state and action for every frame")
            frame.gripper_state = _normalized(frame.gripper_state, "gripper_state")
            frame.gripper_action = _normalized(frame.gripper_action, "gripper_action")
        actuator_states = _named_normalized_values(
            frame.actuator_states,
            self.config.scalar_actuator_names,
            "actuator_states",
        )
        actuator_actions = _named_normalized_values(
            frame.actuator_actions,
            self.config.scalar_actuator_names,
            "actuator_actions",
        )
        self._write_camera_images(active, frame_index, frame.cameras)
        self._write_depth_frames(active, frame_index, frame.depths)
        camera_timestamps_s = {
            name: _camera_timestamp_s(sample.timestamp_s, timestamp_s)
            for name, sample in frame.cameras.items()
        }
        camera_timestamps_s.update(frame.camera_timestamps_s)
        for name, value in camera_timestamps_s.items():
            if not math.isfinite(value):
                raise ValueError(f"Camera timestamp for {name} must be finite")
        audit = {**frame.audit, "frame_index": frame_index, "emitted": True}
        # Image and depth arrays are already owned by their write paths. Keep
        # only the small numeric record needed for Parquet and statistics.
        recorded = RecordedFrame(
            state=state,
            action=action,
            timestamp_s=timestamp_s,
            camera_timestamps_s=camera_timestamps_s,
            gripper_state=frame.gripper_state,
            gripper_action=frame.gripper_action,
            actuator_states=actuator_states,
            actuator_actions=actuator_actions,
            audit=audit,
        )
        active.frames.append(recorded)
        active.audit.append(audit)
        return frame_index

    def append_skipped_tick(self, audit: dict[str, Any]) -> None:
        active = self._require_active()
        active.audit.append({**audit, "frame_index": None, "emitted": False})

    def finish_episode(self) -> int:
        active = self._require_active()
        if not active.frames:
            audit_path = self._preserve_failed_audit(active)
            reason_summary = _skip_reason_summary(active.audit)
            raise RuntimeError(
                f"episode {active.episode_index:06d} 没有完整帧；"
                f"候选帧={len(active.audit)}；跳过原因: {reason_summary}；"
                f"审计文件: {audit_path}"
            )
        try:
            self._flush_depth_stores(active)
            self._wait_for_png_writes(active)
            if self.config.image_storage == "video":
                self._encode_videos(active)
            self._write_parquet(active)
            self._write_audit(active)
            self._commit_episode(active)
            return active.episode_index
        except Exception:
            # Keep the stage directory for diagnosis and an explicit retry.
            raise
        finally:
            self._shutdown_png_writers(active, cancel=False)
            self._active = None

    def discard_episode(self) -> None:
        active = self._require_active()
        try:
            self._shutdown_png_writers(active, cancel=True)
        finally:
            shutil.rmtree(active.stage, ignore_errors=True)
            self._active = None

    def _load_or_create_info(self) -> dict[str, Any]:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        info_path = self.meta_dir / "info.json"
        if info_path.exists():
            with info_path.open(encoding="utf-8") as file:
                info = json.load(file)
            if self._validate_info(info):
                _write_json_atomic(info_path, info)
            return info

        features = _default_features(self.config)
        info: dict[str, Any] = {
            "codebase_version": _dataset_version(self.config.image_storage),
            "image_storage": _info_image_storage(self.config.image_storage),
            "robot_type": self.config.robot_type,
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "total_chunks": 0,
            "chunks_size": CHUNK_SIZE,
            "fps": self.config.fps,
            "splits": {},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": features,
        }
        if self.config.image_storage == "video":
            info["total_videos"] = 0
            info["video_path"] = (
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
                if self.config.cameras
                else None
            )
        else:
            info["total_images"] = 0
        _write_json_atomic(info_path, info)
        return info

    def _validate_info(self, info: dict[str, Any]) -> bool:
        if self.config.image_storage in {"png", "jpg"}:
            return self._validate_image_sequence_info(info)

        expected_version = _dataset_version(self.config.image_storage)
        if info.get("codebase_version") != expected_version:
            raise ValueError(
                "Existing dataset media layout cannot be mixed: "
                "video and image sequences use different layouts"
            )
        # v2.1 datasets created before image_storage was introduced are video datasets.
        if info.get("image_storage", "video") != self.config.image_storage:
            raise ValueError("Existing dataset image_storage does not match the recorder configuration")
        if int(info.get("fps", 0)) != self.config.fps:
            raise ValueError("Existing dataset FPS does not match the recorder configuration")
        expected = _default_features(self.config)
        if info.get("features") != expected:
            raise ValueError("Existing dataset features do not match the recorder configuration")
        if self.config.cameras:
            if not isinstance(info.get("video_path"), str):
                raise ValueError("Existing dataset is missing video_path")
        return False

    def _validate_image_sequence_info(self, info: dict[str, Any]) -> bool:
        if info.get("codebase_version") not in {PNG_SEQUENCE, JPG_SEQUENCE, IMAGE_SEQUENCE}:
            raise ValueError(
                "Existing dataset media layout cannot be mixed: "
                "video and image sequences use different layouts"
            )
        if info.get("image_storage") not in {"png", "jpg", "image_sequence"}:
            raise ValueError("Existing dataset does not use an image-sequence layout")
        if int(info.get("fps", 0)) != self.config.fps:
            raise ValueError("Existing dataset FPS does not match the recorder configuration")

        features = info.get("features")
        if not isinstance(features, dict):
            raise ValueError("Existing dataset is missing feature metadata")
        normalized = {name: dict(feature) for name, feature in features.items() if isinstance(feature, dict)}
        if len(normalized) != len(features):
            raise ValueError("Existing dataset features do not match the recorder configuration")
        for camera in self.config.cameras:
            feature = normalized.get(camera.feature_name)
            if feature is None or feature.get("dtype") not in {
                "png_sequence",
                "jpg_sequence",
                "image_sequence",
            }:
                raise ValueError("Existing dataset features do not match the recorder configuration")
            feature["dtype"] = "image_sequence"
        expected = _default_features(self.config)
        if normalized != expected:
            raise ValueError("Existing dataset features do not match the recorder configuration")

        changed = (
            info.get("codebase_version") != IMAGE_SEQUENCE
            or info.get("image_storage") != "image_sequence"
            or info.get("features") != expected
            or "image_path" in info
        )
        info["codebase_version"] = IMAGE_SEQUENCE
        info["image_storage"] = "image_sequence"
        info["features"] = expected
        info.pop("image_path", None)
        return changed

    def _load_tasks(self) -> dict[str, int]:
        path = self.meta_dir / "tasks.jsonl"
        if not path.exists():
            return {}
        tasks: dict[str, int] = {}
        for item in _read_jsonl(path):
            tasks[str(item["task"])] = int(item["task_index"])
        return tasks

    def _load_or_validate_depth_sources(self) -> dict[str, Any] | None:
        expected = _depth_sources_document(self.config)
        path = self.meta_dir / "depth_sources.json"
        if path.exists():
            with path.open(encoding="utf-8") as file:
                existing = json.load(file)
            if existing != expected:
                raise ValueError("Existing depth sidecar metadata does not match the recorder configuration")
            return existing
        if self.config.depth_cameras:
            return expected
        return None

    def _initialize_depth_stores(self, active: _EpisodeBuffer) -> None:
        if not self.config.depth_cameras:
            return
        try:
            import zarr
            from numcodecs import Blosc
        except ImportError as exc:
            raise RuntimeError(
                "RGB-D collection requires zarr and numcodecs. Install the checked-in requirements."
            ) from exc

        for camera in self.config.depth_cameras:
            path = active.stage / "depth" / camera.feature_name / "episode.zarr"
            path.parent.mkdir(parents=True, exist_ok=True)
            group = zarr.open_group(str(path), mode="w")
            group.attrs.update(_depth_group_attributes(camera))
            chunk_frames = _depth_chunk_frames(camera.shape)
            compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
            raw = group.create_dataset(
                "depth_raw",
                shape=(0, *camera.shape),
                chunks=(chunk_frames, *camera.shape),
                dtype=np.uint16,
                compressor=compressor,
            )
            frame_index = group.create_dataset(
                "frame_index",
                shape=(0,),
                chunks=(chunk_frames,),
                dtype=np.int64,
                compressor=compressor,
            )
            capture_monotonic_ns = group.create_dataset(
                "capture_monotonic_ns",
                shape=(0,),
                chunks=(chunk_frames,),
                dtype=np.int64,
                compressor=compressor,
            )
            source_timestamp_ns = group.create_dataset(
                "source_timestamp_ns",
                shape=(0,),
                chunks=(chunk_frames,),
                dtype=np.int64,
                compressor=compressor,
            )
            source_frame_index = group.create_dataset(
                "source_frame_index",
                shape=(0,),
                chunks=(chunk_frames,),
                dtype=np.int64,
                compressor=compressor,
            )
            meters_per_raw_unit = group.create_dataset(
                "meters_per_raw_unit",
                shape=(0,),
                chunks=(chunk_frames,),
                dtype=np.float32,
                compressor=compressor,
            )
            active.depth_stores[camera.feature_name] = _DepthStore(
                spec=camera,
                chunk_frames=chunk_frames,
                raw=raw,
                frame_index=frame_index,
                capture_monotonic_ns=capture_monotonic_ns,
                source_timestamp_ns=source_timestamp_ns,
                source_frame_index=source_frame_index,
                meters_per_raw_unit=meters_per_raw_unit,
            )

    def _write_depth_frames(
        self, active: _EpisodeBuffer, frame_index: int, depths: dict[str, DepthFrame]
    ) -> None:
        expected = set(active.depth_stores)
        if set(depths) != expected:
            raise ValueError("Every configured depth source must provide one depth frame per dataset frame")
        for name, store in active.depth_stores.items():
            sample = depths[name]
            raw = np.asarray(sample.raw)
            if raw.shape != store.spec.shape or raw.dtype != np.uint16:
                raise ValueError(f"Depth {name} does not match its configured uint16 shape")
            scale = float(sample.meters_per_raw_unit)
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"Depth {name} meters_per_raw_unit must be finite and positive")
            pending_index = int(store.raw.shape[0]) + len(store.pending)
            if pending_index != frame_index:
                raise RuntimeError(f"Depth {name} frame index is not contiguous")
            store.pending.append(
                _PendingDepthFrame(
                    raw=raw,
                    meters_per_raw_unit=scale,
                    capture_monotonic_ns=(
                        -1 if sample.capture_monotonic_ns is None else int(sample.capture_monotonic_ns)
                    ),
                    source_timestamp_ns=(
                        -1 if sample.source_timestamp_ns is None else int(sample.source_timestamp_ns)
                    ),
                    source_frame_index=(
                        -1 if sample.source_frame_index is None else int(sample.source_frame_index)
                    ),
                )
            )
            if len(store.pending) >= store.chunk_frames:
                self._flush_depth_store(store)

    def _flush_depth_stores(self, active: _EpisodeBuffer) -> None:
        for store in active.depth_stores.values():
            self._flush_depth_store(store)

    @staticmethod
    def _flush_depth_store(store: _DepthStore) -> None:
        if not store.pending:
            return
        pending = store.pending
        start = int(store.raw.shape[0])
        end = start + len(pending)
        store.raw.resize((end, *store.spec.shape))
        store.raw[start:end] = np.stack([sample.raw for sample in pending])
        store.frame_index.resize((end,))
        store.frame_index[start:end] = np.arange(start, end, dtype=np.int64)
        store.capture_monotonic_ns.resize((end,))
        store.capture_monotonic_ns[start:end] = np.asarray(
            [sample.capture_monotonic_ns for sample in pending], dtype=np.int64
        )
        store.source_timestamp_ns.resize((end,))
        store.source_timestamp_ns[start:end] = np.asarray(
            [sample.source_timestamp_ns for sample in pending], dtype=np.int64
        )
        store.source_frame_index.resize((end,))
        store.source_frame_index[start:end] = np.asarray(
            [sample.source_frame_index for sample in pending], dtype=np.int64
        )
        store.meters_per_raw_unit.resize((end,))
        store.meters_per_raw_unit[start:end] = np.asarray(
            [sample.meters_per_raw_unit for sample in pending], dtype=np.float32
        )
        store.pending.clear()

    def _write_camera_images(
        self, active: _EpisodeBuffer, frame_index: int, cameras: dict[str, CameraFrame]
    ) -> None:
        expected = {camera.feature_name: camera for camera in self.config.cameras}
        if set(cameras) != set(expected):
            raise ValueError("Every configured camera must provide one frame for each dataset frame")
        if not cameras:
            return
        for name, sample in cameras.items():
            array = np.asarray(sample.rgb)
            if array.shape != expected[name].shape or array.dtype != np.uint8:
                raise ValueError(f"Camera {name} does not match its configured RGB uint8 shape")
        if self.config.image_storage in {"png", "jpg"}:
            self._queue_png_images(active, frame_index, cameras)
            return
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Camera collection requires Pillow") from exc
        for name, sample in cameras.items():
            array = np.asarray(sample.rgb)
            path = active.stage / "images" / name / f"frame_{frame_index:06d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array, mode="RGB").save(path)

    def _start_png_writers(self, active: _EpisodeBuffer) -> None:
        if self.config.image_storage not in {"png", "jpg"} or not self.config.cameras:
            return
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Camera collection requires Pillow") from exc
        del Image
        workers = min(
            MAX_PNG_WORKERS,
            max(1, PNG_WORKERS_PER_CAMERA * len(self.config.cameras)),
        )
        active.png_slots = threading.BoundedSemaphore(workers * PNG_PENDING_TASKS_PER_WORKER)
        active.png_executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="leobot-png",
        )

    def _queue_png_images(
        self,
        active: _EpisodeBuffer,
        frame_index: int,
        cameras: dict[str, CameraFrame],
    ) -> None:
        executor = active.png_executor
        slots = active.png_slots
        if executor is None or slots is None:
            raise RuntimeError("PNG writer is not initialized")
        self._raise_completed_png_errors(active)
        for name, sample in cameras.items():
            path = (
                active.stage
                / "images"
                / name
                / f"frame_{frame_index:06d}.{self.config.image_storage}"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            # This only blocks the collection process after its bounded image
            # backlog is full; it never blocks the ServoJ parent process.
            slots.acquire()
            future = executor.submit(
                _save_image_sequence_frame,
                path,
                np.asarray(sample.rgb),
                self.config.image_storage,
                self.config.quality,
            )
            active.png_futures.add(future)
            future.add_done_callback(lambda _future, semaphore=slots: semaphore.release())

    @staticmethod
    def _raise_completed_png_errors(active: _EpisodeBuffer) -> None:
        for future in tuple(active.png_futures):
            if not future.done():
                continue
            active.png_futures.remove(future)
            future.result()

    @staticmethod
    def _wait_for_png_writes(active: _EpisodeBuffer) -> None:
        while active.png_futures:
            future = active.png_futures.pop()
            future.result()

    @staticmethod
    def _shutdown_png_writers(active: _EpisodeBuffer, *, cancel: bool) -> None:
        executor = active.png_executor
        if executor is None:
            return
        if cancel:
            for future in active.png_futures:
                future.cancel()
        executor.shutdown(wait=True, cancel_futures=cancel)
        active.png_futures.clear()
        active.png_executor = None
        active.png_slots = None

    def _encode_videos(self, active: _EpisodeBuffer) -> None:
        for camera in self.config.cameras:
            image_dir = active.stage / "images" / camera.feature_name
            output = active.stage / "videos" / camera.feature_name / "episode.mp4"
            encode_episode_video(image_dir, output, self.config.fps)

    def _write_parquet(self, active: _EpisodeBuffer) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Dataset collection requires pyarrow") from exc

        length = len(active.frames)
        start_index = int(self._info["total_frames"])
        task_index = self._task_index(active.task)
        arrays: list[Any] = [
            pa.array(range(start_index, start_index + length), type=pa.int64()),
            pa.array([active.episode_index] * length, type=pa.int64()),
            pa.array(range(length), type=pa.int64()),
            pa.array([float(frame.timestamp_s) for frame in active.frames], type=pa.float32()),
            pa.array([task_index] * length, type=pa.int64()),
            _fixed_vector_array(pa, [frame.action for frame in active.frames], self.config.joint_count),
            _fixed_vector_array(pa, [frame.state for frame in active.frames], self.config.joint_count),
        ]
        names = [
            "index",
            "episode_index",
            "frame_index",
            "timestamp",
            "task_index",
            "action",
            "observation.state",
        ]
        if self.config.include_gripper:
            arrays.extend(
                [
                    pa.array([frame.gripper_action for frame in active.frames], type=pa.float32()),
                    pa.array([frame.gripper_state for frame in active.frames], type=pa.float32()),
                ]
            )
            names.extend(["action.gripper", "observation.gripper"])
        for actuator_name in self.config.scalar_actuator_names:
            arrays.extend(
                [
                    pa.array(
                        [frame.actuator_actions[actuator_name] for frame in active.frames],
                        type=pa.float32(),
                    ),
                    pa.array(
                        [frame.actuator_states[actuator_name] for frame in active.frames],
                        type=pa.float32(),
                    ),
                ]
            )
            names.extend(
                [
                    f"action.{actuator_name}",
                    f"observation.{actuator_name}",
                ]
            )
        for camera in self.config.cameras:
            relative_paths = [
                self._camera_relative_path(active.episode_index, camera.feature_name, frame_index)
                for frame_index in range(length)
            ]
            arrays.append(
                pa.StructArray.from_arrays(
                    [
                        pa.array(relative_paths, type=pa.string()),
                        pa.array(
                            [
                                frame.camera_timestamps_s.get(camera.feature_name, float(frame.timestamp_s))
                                for frame in active.frames
                            ],
                            type=pa.float32(),
                        ),
                    ],
                    names=["path", "timestamp"],
                )
            )
            names.append(camera.feature_name)
        table = pa.Table.from_arrays(arrays, names=names)
        target = active.stage / "data.parquet"
        pq.write_table(table, target)

    def _write_audit(self, active: _EpisodeBuffer) -> None:
        path = active.stage / "audit.jsonl"
        _write_jsonl_atomic(path, active.audit)

    def _preserve_failed_audit(self, active: _EpisodeBuffer) -> Path:
        destination = (
            self.meta_dir
            / "failed_recording_audit"
            / f"{active.stage.name}.jsonl"
        )
        _write_jsonl_atomic(destination, active.audit)
        return destination

    def _commit_episode(self, active: _EpisodeBuffer) -> None:
        chunk = active.episode_index // CHUNK_SIZE
        parquet_destination = self.root / "data" / f"chunk-{chunk:03d}" / f"episode_{active.episode_index:06d}.parquet"
        parquet_destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(active.stage / "data.parquet", parquet_destination)
        for camera in self.config.cameras:
            if self.config.image_storage == "video":
                destination = (
                    self.root
                    / "videos"
                    / f"chunk-{chunk:03d}"
                    / camera.feature_name
                    / f"episode_{active.episode_index:06d}.mp4"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(active.stage / "videos" / camera.feature_name / "episode.mp4", destination)
            else:
                destination = (
                    self.root
                    / "images"
                    / f"chunk-{chunk:03d}"
                    / camera.feature_name
                    / f"episode_{active.episode_index:06d}"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(active.stage / "images" / camera.feature_name, destination)
        for camera in self.config.depth_cameras:
            destination = (
                self.root
                / "depth"
                / f"chunk-{chunk:03d}"
                / camera.feature_name
                / f"episode_{active.episode_index:06d}.zarr"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(active.stage / "depth" / camera.feature_name / "episode.zarr", destination)
        audit_destination = self.meta_dir / "recording_audit" / f"episode_{active.episode_index:06d}.jsonl"
        audit_destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(active.stage / "audit.jsonl", audit_destination)

        if self._depth_sources is not None:
            _write_json_atomic(self.meta_dir / "depth_sources.json", self._depth_sources)

        episode = {
            "episode_index": active.episode_index,
            "tasks": [active.task],
            "length": len(active.frames),
        }
        stats = _episode_stats(active.frames, self.config)
        _append_jsonl_atomic(self.meta_dir / "episodes.jsonl", episode)
        _append_jsonl_atomic(
            self.meta_dir / "episodes_stats.jsonl",
            {"episode_index": active.episode_index, "stats": stats},
        )
        if active.task not in self._tasks:
            task_index = int(self._info["total_tasks"])
            self._tasks[active.task] = task_index
            self._info["total_tasks"] = task_index + 1
            _append_jsonl_atomic(
                self.meta_dir / "tasks.jsonl",
                {"task_index": task_index, "task": active.task},
            )
        self._info["total_episodes"] += 1
        self._info["total_frames"] += len(active.frames)
        if self.config.image_storage == "video":
            self._info["total_videos"] += len(self.config.cameras)
        else:
            self._info["total_images"] += len(self.config.cameras) * len(active.frames)
        self._info["total_chunks"] = max(self._info["total_chunks"], chunk + 1)
        self._info["splits"] = {"train": f"0:{self._info['total_episodes']}"}
        _write_json_atomic(self.meta_dir / "info.json", self._info)
        shutil.rmtree(active.stage, ignore_errors=True)

    def _task_index(self, task: str) -> int:
        existing = self._tasks.get(task)
        if existing is not None:
            return existing
        return int(self._info["total_tasks"])

    def _video_relative_path(self, episode_index: int, feature_name: str) -> str:
        return self._info["video_path"].format(
            episode_chunk=episode_index // CHUNK_SIZE,
            video_key=feature_name,
            episode_index=episode_index,
        )

    def _image_relative_path(self, episode_index: int, feature_name: str, frame_index: int) -> str:
        return (
            f"images/chunk-{episode_index // CHUNK_SIZE:03d}/{feature_name}/"
            f"episode_{episode_index:06d}/frame_{frame_index:06d}.{self.config.image_storage}"
        )

    def _camera_relative_path(self, episode_index: int, feature_name: str, frame_index: int) -> str:
        if self.config.image_storage == "video":
            return self._video_relative_path(episode_index, feature_name)
        return self._image_relative_path(episode_index, feature_name, frame_index)

    def _require_active(self) -> _EpisodeBuffer:
        if self._active is None:
            raise RuntimeError("No active episode")
        return self._active


@dataclass
class _EpisodeBuffer:
    episode_index: int
    task: str
    stage: Path
    frames: list[RecordedFrame] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)
    depth_stores: dict[str, _DepthStore] = field(default_factory=dict)
    png_executor: ThreadPoolExecutor | None = None
    png_slots: threading.BoundedSemaphore | None = None
    png_futures: set[Future[None]] = field(default_factory=set)


@dataclass(frozen=True)
class _PendingDepthFrame:
    raw: np.ndarray
    meters_per_raw_unit: float
    capture_monotonic_ns: int
    source_timestamp_ns: int
    source_frame_index: int


@dataclass
class _DepthStore:
    spec: DepthCameraSpec
    chunk_frames: int
    raw: Any
    frame_index: Any
    capture_monotonic_ns: Any
    source_timestamp_ns: Any
    source_frame_index: Any
    meters_per_raw_unit: Any
    pending: list[_PendingDepthFrame] = field(default_factory=list)


def _save_image_sequence_frame(
    path: Path, rgb: np.ndarray, image_storage: ImageStorage, quality: int
) -> None:
    """Write one RGB image independently so multiple camera frames encode in parallel."""

    from PIL import Image

    image = Image.fromarray(rgb, mode="RGB")
    if image_storage == "png":
        image.save(path, compress_level=1)
        return
    if image_storage == "jpg":
        image.save(path, format="JPEG", quality=quality)
        return
    raise ValueError(f"Image sequence storage must be 'png' or 'jpg', got {image_storage!r}")


def _default_features(config: WriterConfig) -> dict[str, dict[str, Any]]:
    scalar_int = {"dtype": "int64", "shape": [1], "names": None}
    features: dict[str, dict[str, Any]] = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": scalar_int,
        "episode_index": scalar_int,
        "index": scalar_int,
        "task_index": scalar_int,
        "action": {"dtype": "float32", "shape": [config.joint_count], "names": None},
        "observation.state": {"dtype": "float32", "shape": [config.joint_count], "names": None},
    }
    if config.include_gripper:
        features["action.gripper"] = {"dtype": "float32", "shape": [1], "names": None}
        features["observation.gripper"] = {"dtype": "float32", "shape": [1], "names": None}
    for actuator_name in config.scalar_actuator_names:
        features[f"action.{actuator_name}"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        }
        features[f"observation.{actuator_name}"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        }
    image_dtype = "video" if config.image_storage == "video" else "image_sequence"
    for camera in config.cameras:
        features[camera.feature_name] = {
            "dtype": image_dtype,
            "shape": list(camera.shape),
            "names": ["height", "width", "channels"],
        }
    return features


def _dataset_version(image_storage: ImageStorage) -> str:
    return V21 if image_storage == "video" else IMAGE_SEQUENCE


def _info_image_storage(image_storage: ImageStorage) -> str:
    return "video" if image_storage == "video" else "image_sequence"


def _depth_sources_document(config: WriterConfig) -> dict[str, Any]:
    return {
        "schema_version": DEPTH_SIDECAR_VERSION,
        "sources": {
            camera.feature_name: _depth_source_entry(camera) for camera in config.depth_cameras
        },
    }


def _depth_source_entry(camera: DepthCameraSpec) -> dict[str, Any]:
    metadata = camera.metadata
    entry: dict[str, Any] = {
        "rgb_feature": camera.rgb_feature_name,
        "shape": list(camera.shape),
        "raw_dtype": "uint16",
        "raw_format": metadata.raw_format,
        "invalid_value": int(metadata.invalid_value),
        "aligned_to_rgb": bool(metadata.aligned_to_rgb),
    }
    if metadata.color_intrinsics is not None:
        entry["color_intrinsics"] = np.asarray(metadata.color_intrinsics, dtype=float).tolist()
    if metadata.camera_model is not None:
        entry["camera_model"] = metadata.camera_model
    if metadata.serial_number is not None:
        entry["serial_number"] = metadata.serial_number
    if metadata.source_timestamp_clock is not None:
        entry["source_timestamp_clock"] = metadata.source_timestamp_clock
    return entry


def _depth_group_attributes(camera: DepthCameraSpec) -> dict[str, Any]:
    return {
        "schema_version": DEPTH_SIDECAR_VERSION,
        "depth_feature": camera.feature_name,
        **_depth_source_entry(camera),
    }


def _depth_chunk_frames(shape: tuple[int, int]) -> int:
    frame_bytes = int(np.prod(shape)) * np.dtype(np.uint16).itemsize
    return max(1, min(32, DEPTH_CHUNK_TARGET_BYTES // frame_bytes))


def _float_vector(value: np.ndarray, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {result.shape}")
    return result.copy()


def _record_timestamp_s(value: float | None, frame_index: int, fps: int) -> float:
    """Return a finite nonnegative row timestamp with legacy FPS fallback."""

    timestamp_s = frame_index / fps if value is None else float(value)
    if not math.isfinite(timestamp_s) or timestamp_s < 0.0:
        raise ValueError("Recorded frame timestamp_s must be finite and nonnegative")
    return timestamp_s


def _camera_timestamp_s(value: float | None, row_timestamp_s: float) -> float:
    """Return one finite camera timestamp, defaulting to its row time."""

    timestamp_s = row_timestamp_s if value is None else float(value)
    if not math.isfinite(timestamp_s):
        raise ValueError("Camera timestamp_s must be finite")
    return timestamp_s


def _normalized(value: float, name: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be normalized to [0, 1]")
    return result


def _named_normalized_values(
    values: dict[str, float],
    expected_names: tuple[str, ...],
    field_name: str,
) -> dict[str, float]:
    if not isinstance(values, dict):
        raise ValueError(f"{field_name} must be a dictionary")
    actual_names = set(values)
    expected = set(expected_names)
    if actual_names != expected:
        missing = sorted(expected - actual_names)
        unexpected = sorted(actual_names - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError(
            f"{field_name} names do not match the configured actuators "
            f"({'; '.join(details)})"
        )
    return {
        name: _normalized(values[name], f"{field_name}.{name}")
        for name in expected_names
    }


def _fixed_vector_array(pa: Any, values: list[np.ndarray], length: int) -> Any:
    return pa.array([value.tolist() for value in values], type=pa.list_(pa.float32(), length))


def _skip_reason_summary(audit: list[dict[str, Any]]) -> str:
    if not audit:
        return "未收到头部相机候选帧"
    counts = Counter(
        str(item.get("skip_reason") or "未标明原因")
        for item in audit
    )
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{reason}={count}" for reason, count in ordered)


def _episode_stats(
    frames: list[RecordedFrame], config: WriterConfig
) -> dict[str, dict[str, Any]]:
    features: dict[str, np.ndarray] = {
        "action": np.stack([frame.action for frame in frames]),
        "observation.state": np.stack([frame.state for frame in frames]),
    }
    if config.include_gripper:
        features["action.gripper"] = np.asarray([frame.gripper_action for frame in frames], dtype=np.float32)
        features["observation.gripper"] = np.asarray([frame.gripper_state for frame in frames], dtype=np.float32)
    for actuator_name in config.scalar_actuator_names:
        features[f"action.{actuator_name}"] = np.asarray(
            [frame.actuator_actions[actuator_name] for frame in frames],
            dtype=np.float32,
        )
        features[f"observation.{actuator_name}"] = np.asarray(
            [frame.actuator_states[actuator_name] for frame in frames],
            dtype=np.float32,
        )
    stats: dict[str, dict[str, Any]] = {}
    for name, values in features.items():
        stats[name] = {
            "min": np.min(values, axis=0).tolist(),
            "max": np.max(values, axis=0).tolist(),
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
            "count": int(values.shape[0]),
        }
    return stats


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for value in values:
            file.write(json.dumps(value, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _append_jsonl_atomic(path: Path, value: dict[str, Any]) -> None:
    values = _read_jsonl(path) if path.exists() else []
    values.append(value)
    _write_jsonl_atomic(path, values)
