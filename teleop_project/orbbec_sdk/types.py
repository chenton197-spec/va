"""Vendor-neutral data and configuration types for the Orbbec integration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import math

import numpy as np


class CameraMode(str, Enum):
    """Streams exposed by one physical camera."""

    RGB = "rgb"
    DEPTH = "depth"
    RGBD = "rgbd"


class AlignmentMode(str, Enum):
    """Depth-to-color alignment strategy for an RGB-D pipeline."""

    NONE = "none"
    SOFTWARE = "software"
    HARDWARE = "hardware"


class CameraStatus(str, Enum):
    """Observable lifecycle state of one configured camera."""

    CREATED = "created"
    STARTING = "starting"
    STREAMING = "streaming"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True)
class OrbbecCameraConfig:
    """One camera declaration using explicit resolutions and FPS."""

    name: str
    serial_number: str
    mode: CameraMode = CameraMode.RGBD
    rgb_resolution: tuple[int, int] | list[int] | None = None
    depth_resolution: tuple[int, int] | list[int] | None = None
    fps: int | None = None
    alignment: AlignmentMode | None = None
    frame_timeout_ms: int = 200
    max_consecutive_timeouts: int = 10
    first_frame_timeout_s: float = 5.0
    depth_scale_to_meters: float = 0.001

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.serial_number.strip():
            raise ValueError("Camera name and serial_number must be non-empty")
        if self.frame_timeout_ms <= 0 or self.max_consecutive_timeouts <= 0:
            raise ValueError("Frame timeout settings must be positive")
        if self.first_frame_timeout_s <= 0.0:
            raise ValueError("first_frame_timeout_s must be positive")
        if not math.isfinite(self.depth_scale_to_meters) or self.depth_scale_to_meters <= 0.0:
            raise ValueError("depth_scale_to_meters must be finite and positive")

        rgb_resolution = _resolution(self.rgb_resolution, "rgb_resolution")
        depth_resolution = _resolution(self.depth_resolution, "depth_resolution")
        object.__setattr__(self, "rgb_resolution", rgb_resolution)
        object.__setattr__(self, "depth_resolution", depth_resolution)

        has_explicit_resolution = rgb_resolution is not None or depth_resolution is not None
        if has_explicit_resolution and self.fps is None:
            raise ValueError("fps is required when a stream resolution is specified")
        if not has_explicit_resolution and self.fps is not None:
            raise ValueError("fps requires an rgb_resolution or depth_resolution")
        if self.fps is not None and (
            isinstance(self.fps, bool) or not isinstance(self.fps, (int, np.integer)) or self.fps <= 0
        ):
            raise ValueError("fps must be a positive integer")
        if self.fps is not None:
            object.__setattr__(self, "fps", int(self.fps))
        if self.mode is CameraMode.RGBD and (rgb_resolution is None) != (depth_resolution is None):
            raise ValueError("RGB-D mode requires both resolutions or neither")
        if self.mode is CameraMode.DEPTH and rgb_resolution is not None:
            raise ValueError("Depth mode cannot define rgb_resolution")

        alignment = self.alignment
        if alignment is None:
            alignment = AlignmentMode.SOFTWARE if self.mode is CameraMode.RGBD else AlignmentMode.NONE
            object.__setattr__(self, "alignment", alignment)
        if self.mode is CameraMode.RGBD and alignment is AlignmentMode.NONE:
            raise ValueError("RGB-D mode requires an alignment strategy")
        if self.mode is not CameraMode.RGBD and alignment is not AlignmentMode.NONE:
            raise ValueError("Only RGB-D mode can enable depth-to-color alignment")


def _resolution(value: tuple[int, int] | list[int] | None, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a [width, height] pair")
    try:
        width, height = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a [width, height] pair") from exc
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, (int, np.integer))
        or not isinstance(height, (int, np.integer))
        or width <= 0
        or height <= 0
    ):
        raise ValueError(f"{name} must contain positive integer dimensions")
    return int(width), int(height)


@dataclass(frozen=True)
class DeviceDescriptor:
    """A discovered device and the backend handle required to open it."""

    serial_number: str
    model: str | None
    handle: Any


@dataclass(frozen=True)
class CameraMetadata:
    """Stable metadata learned when a pipeline receives its first frame."""

    serial_number: str
    camera_model: str | None
    rgb_shape: tuple[int, int, int] | None = None
    depth_shape: tuple[int, int] | None = None
    color_intrinsics: np.ndarray | None = None
    aligned_to_rgb: bool = False
    source_timestamp_clock: str = "orbbec_global_timestamp_ns"


@dataclass(frozen=True)
class OrbbecFrame:
    """One new camera frame shared by preview and dataset collection.

    Most callers only need ``rgb`` and ``depth``. ``depth`` is the unmodified
    Y16 ``uint16`` image; the remaining fields preserve its scale and timing
    for dataset collection.
    """

    rgb: np.ndarray | None
    depth_raw: np.ndarray | None
    meters_per_raw_unit: float | None
    capture_monotonic_ns: int
    rgb_source_timestamp_ns: int | None = None
    depth_source_timestamp_ns: int | None = None
    rgb_source_frame_index: int | None = None
    depth_source_frame_index: int | None = None
    color_intrinsics: np.ndarray | None = None
    aligned_to_rgb: bool = False

    @property
    def depth(self) -> np.ndarray | None:
        """Return the unmodified Y16 depth image."""

        return self.depth_raw


@dataclass(frozen=True)
class BackendSession:
    """Opaque backend pipeline handle paired with its configured device."""

    descriptor: DeviceDescriptor
    config: OrbbecCameraConfig
    handle: Any
