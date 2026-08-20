"""Optional, device-neutral sources used by the collection SDK."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class TimedValue:
    """A numeric sample with its source monotonic-clock timestamp."""

    value: np.ndarray
    capture_monotonic_ns: int | None = None


@dataclass(frozen=True)
class CameraFrame:
    """An RGB frame supplied by a camera adapter.

    ``capture_monotonic_ns`` measures host-side arrival. Optional source clock
    values are preserved separately in the audit sidecar when available.
    """

    rgb: np.ndarray
    # Episode-relative time set by a recorder when it has a shared timeline.
    # Direct camera users normally leave this unset.
    timestamp_s: float | None = None
    capture_monotonic_ns: int | None = None
    source_timestamp_ns: int | None = None
    source_frame_index: int | None = None


@dataclass(frozen=True)
class DepthFrame:
    """One lossless depth sample supplied by an RGB-D camera adapter.

    ``raw`` preserves the camera's native unsigned 16-bit depth values.  Its
    physical interpretation is explicit: ``raw * meters_per_raw_unit`` gives
    depth in metres, while the configured invalid value remains invalid.
    """

    raw: np.ndarray
    meters_per_raw_unit: float
    capture_monotonic_ns: int | None = None
    source_timestamp_ns: int | None = None
    source_frame_index: int | None = None


@dataclass(frozen=True)
class DepthMetadata:
    """Static information for one raw depth source.

    Intrinsics and device identity are optional because 2D RGB-D policies do
    not need them.  They are retained when a downstream user needs a point
    cloud or reproducible camera calibration.
    """

    raw_format: str = "Y16"
    invalid_value: int = 0
    aligned_to_rgb: bool = False
    color_intrinsics: np.ndarray | None = None
    camera_model: str | None = None
    serial_number: str | None = None
    source_timestamp_clock: str | None = None


@dataclass(frozen=True)
class RGBDMetadata(DepthMetadata):
    """Static metadata for a depth stream spatially aligned to RGB."""

    aligned_to_rgb: bool = True


@dataclass(frozen=True)
class RGBDFrame:
    """An RGB image and spatially aligned raw depth from one camera frame set."""

    rgb: np.ndarray
    depth: DepthFrame
    rgb_capture_monotonic_ns: int | None = None
    rgb_source_timestamp_ns: int | None = None
    rgb_source_frame_index: int | None = None


class GripperFeedbackSource(Protocol):
    """Read normalized follower-gripper feedback for one collection frame."""

    def read_gripper_opening(self) -> TimedValue | None:
        """Return a normalized 0-1 opening, or ``None`` when unavailable."""
        ...


class CallableGripperFeedbackSource:
    """Adapt a normalized scalar feedback callback for the collection SDK."""

    def __init__(self, read_opening: Callable[[], float | None]) -> None:
        self._read_opening = read_opening

    def read_gripper_opening(self) -> TimedValue | None:
        opening = self._read_opening()
        if opening is None:
            return None
        return TimedValue(
            value=np.array([float(opening)], dtype=float),
            capture_monotonic_ns=time.perf_counter_ns(),
        )
