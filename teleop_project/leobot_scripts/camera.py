"""Device-neutral RGB and RGB-D camera adapter contracts for collection workers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .sources import CameraFrame, RGBDFrame, RGBDMetadata


@dataclass(frozen=True)
class CameraSourceHealth:
    """Device-neutral health snapshot for one camera source."""

    status: str
    latest_capture_monotonic_ns: int | None = None
    error: str | None = None


@runtime_checkable
class CameraFrameSource(Protocol):
    """Timestamped RGB frames from one logical camera.

    A source owns only its cursor state. It must not consume frames globally:
    the recorder and diagnostic tools can keep independent cursors over the
    same device buffer.
    """

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the RGB shape as ``(height, width, 3)``."""
        ...

    def latest_sequence(self) -> int:
        """Return the cursor that skips all frames present before an episode."""
        ...

    def next_frame_after(self, sequence: int) -> tuple[int, CameraFrame | RGBDFrame] | None:
        """Return the oldest retained frame after ``sequence``."""
        ...

    def frame_at_or_before(self, target_monotonic_ns: int) -> CameraFrame | RGBDFrame | None:
        """Return the newest frame whose capture time is not in the future."""
        ...


@runtime_checkable
class RGBDFrameSource(CameraFrameSource, Protocol):
    """Timestamped RGB-D frames from one logical camera.

    A source owns only its cursor state. It must not consume frames globally:
    the recorder and diagnostic tools can keep independent cursors over the
    same device buffer.
    """

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the RGB shape as ``(height, width, 3)``."""
        ...

    @property
    def depth_shape(self) -> tuple[int, int]:
        """Return the raw depth shape as ``(height, width)``."""
        ...

    @property
    def metadata(self) -> RGBDMetadata:
        """Return static raw-depth and calibration metadata."""
        ...

    def latest_sequence(self) -> int:
        """Return the cursor that skips all frames present before an episode."""
        ...

    def next_frame_after(self, sequence: int) -> tuple[int, RGBDFrame] | None:
        """Return the oldest retained frame after ``sequence``."""
        ...

    def frame_at_or_before(self, target_monotonic_ns: int) -> RGBDFrame | None:
        """Return the newest frame whose capture time is not in the future."""
        ...


class CameraAdapterSession(Protocol):
    """Opened cameras owned entirely by one collection worker process."""

    @property
    def sources(self) -> Mapping[str, CameraFrameSource]:
        """Map declared camera names to their timestamped RGB-capable sources."""
        ...

    def close(self) -> None:
        """Release vendor resources owned by this worker process."""
        ...


class CameraAdapterConfig(Protocol):
    """Pickle-safe declaration that opens one adapter in a spawned worker."""

    @property
    def camera_names(self) -> tuple[str, ...]:
        """Return the declared logical camera names before a worker starts."""
        ...

    def open(self) -> CameraAdapterSession:
        """Open configured cameras and return their worker-owned session."""
        ...
