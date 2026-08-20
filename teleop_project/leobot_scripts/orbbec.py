"""Orbbec implementation of the device-neutral camera adapter contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from orbbec_sdk import CameraMode, OrbbecCamera, OrbbecCameraConfig, OrbbecFrame, OrbbecManager

from .camera import CameraAdapterSession, CameraFrameSource, CameraSourceHealth
from .sources import CameraFrame, DepthFrame, DepthMetadata, RGBDFrame, RGBDMetadata


class OrbbecRGBSource:
    """Use an Orbbec RGB or RGB-D camera as a nonblocking RGB source."""

    def __init__(self, camera: OrbbecCamera):
        self._camera = camera
        self._last_sequence = 0

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._camera.rgb_shape

    def latest_frame(self) -> CameraFrame | None:
        item = self._camera.get_frame_after(self._last_sequence)
        if item is None:
            return None
        sequence, frame = item
        self._last_sequence = sequence
        return self._to_camera_frame(frame)

    def next_frame_after(self, sequence: int) -> tuple[int, CameraFrame] | None:
        """Return the next retained RGB frame after a recorder-owned cursor."""

        item = self._camera.get_next_frame_after(sequence)
        if item is None:
            return None
        next_sequence, frame = item
        rgb_frame = self._to_camera_frame(frame)
        if rgb_frame is None:
            return None
        return next_sequence, rgb_frame

    def latest_sequence(self) -> int:
        """Return the camera cursor used to start a fresh camera-driven episode."""

        return self._camera.latest_sequence()

    @staticmethod
    def _to_camera_frame(frame: OrbbecFrame) -> CameraFrame | None:
        if frame.rgb is None:
            return None
        return CameraFrame(
            rgb=frame.rgb,
            capture_monotonic_ns=frame.capture_monotonic_ns,
            source_timestamp_ns=frame.rgb_source_timestamp_ns,
            source_frame_index=frame.rgb_source_frame_index,
        )

    def frame_at_or_before(self, target_monotonic_ns: int) -> CameraFrame | None:
        """Return the RGB image selected for one causal recording timestamp."""

        frame = self._camera.get_frame_at_or_before(target_monotonic_ns)
        if frame is None or frame.rgb is None:
            return None
        return CameraFrame(
            rgb=frame.rgb,
            capture_monotonic_ns=frame.capture_monotonic_ns,
            source_timestamp_ns=frame.rgb_source_timestamp_ns,
            source_frame_index=frame.rgb_source_frame_index,
        )


class OrbbecDepthSource:
    """Use an Orbbec depth or RGB-D camera as a nonblocking depth-only source."""

    def __init__(self, camera: OrbbecCamera):
        self._camera = camera
        self._last_sequence = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self._camera.depth_shape

    @property
    def metadata(self) -> DepthMetadata:
        metadata = self._camera.metadata
        return DepthMetadata(
            raw_format="Y16",
            invalid_value=0,
            aligned_to_rgb=False,
            camera_model=metadata.camera_model,
            serial_number=metadata.serial_number,
            source_timestamp_clock=metadata.source_timestamp_clock,
        )

    def latest_frame(self) -> DepthFrame | None:
        item = self._camera.get_frame_after(self._last_sequence)
        if item is None:
            return None
        sequence, frame = item
        self._last_sequence = sequence
        if frame is None or frame.depth is None or frame.meters_per_raw_unit is None:
            return None
        return DepthFrame(
            raw=frame.depth,
            meters_per_raw_unit=frame.meters_per_raw_unit,
            capture_monotonic_ns=frame.capture_monotonic_ns,
            source_timestamp_ns=frame.depth_source_timestamp_ns,
            source_frame_index=frame.depth_source_frame_index,
        )

    def frame_at_or_before(self, target_monotonic_ns: int) -> DepthFrame | None:
        """Return the raw depth image selected for one causal recording timestamp."""

        frame = self._camera.get_frame_at_or_before(target_monotonic_ns)
        if frame is None or frame.depth is None or frame.meters_per_raw_unit is None:
            return None
        return DepthFrame(
            raw=frame.depth,
            meters_per_raw_unit=frame.meters_per_raw_unit,
            capture_monotonic_ns=frame.capture_monotonic_ns,
            source_timestamp_ns=frame.depth_source_timestamp_ns,
            source_frame_index=frame.depth_source_frame_index,
        )


class OrbbecRGBDSource:
    """Use one aligned Orbbec RGB-D camera as an atomic collection source."""

    def __init__(self, camera: OrbbecCamera):
        self._camera = camera
        self._last_sequence = 0

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._camera.rgb_shape

    @property
    def depth_shape(self) -> tuple[int, int]:
        return self._camera.depth_shape

    @property
    def metadata(self) -> RGBDMetadata:
        metadata = self._camera.metadata
        return RGBDMetadata(
            raw_format="Y16",
            invalid_value=0,
            aligned_to_rgb=True,
            color_intrinsics=metadata.color_intrinsics,
            camera_model=metadata.camera_model,
            serial_number=metadata.serial_number,
            source_timestamp_clock=metadata.source_timestamp_clock,
        )

    def latest_frame(self) -> RGBDFrame | None:
        item = self._camera.get_frame_after(self._last_sequence)
        if item is None:
            return None
        sequence, frame = item
        self._last_sequence = sequence
        return self._to_rgbd_frame(frame)

    def next_frame_after(self, sequence: int) -> tuple[int, RGBDFrame] | None:
        """Return the next retained RGB-D frame after a recorder-owned cursor."""

        item = self._camera.get_next_frame_after(sequence)
        if item is None:
            return None
        next_sequence, frame = item
        rgbd_frame = self._to_rgbd_frame(frame)
        if rgbd_frame is None:
            return None
        return next_sequence, rgbd_frame

    def latest_sequence(self) -> int:
        """Return the camera cursor used to start a fresh camera-driven episode."""

        return self._camera.latest_sequence()

    def frame_at_or_before(self, target_monotonic_ns: int) -> RGBDFrame | None:
        """Return one aligned RGB-D frame selected for a recording timestamp."""

        frame = self._camera.get_frame_at_or_before(target_monotonic_ns)
        if frame is None:
            return None
        return self._to_rgbd_frame(frame)

    @staticmethod
    def _to_rgbd_frame(frame: OrbbecFrame) -> RGBDFrame | None:
        if frame.rgb is None or frame.depth is None or frame.meters_per_raw_unit is None:
            return None
        return RGBDFrame(
            rgb=frame.rgb,
            depth=DepthFrame(
                raw=frame.depth,
                meters_per_raw_unit=frame.meters_per_raw_unit,
                capture_monotonic_ns=frame.capture_monotonic_ns,
                source_timestamp_ns=frame.depth_source_timestamp_ns,
                source_frame_index=frame.depth_source_frame_index,
            ),
            rgb_capture_monotonic_ns=frame.capture_monotonic_ns,
            rgb_source_timestamp_ns=frame.rgb_source_timestamp_ns,
            rgb_source_frame_index=frame.rgb_source_frame_index,
        )


@dataclass(frozen=True)
class OrbbecCameraAdapterConfig:
    """Pickle-safe declaration for a worker-owned multi-Orbbec RGB camera session."""

    camera_configs: tuple[OrbbecCameraConfig, ...]

    def __post_init__(self) -> None:
        if not self.camera_configs:
            raise ValueError("OrbbecCameraAdapterConfig requires at least one camera")
        names = tuple(config.name for config in self.camera_configs)
        if len(names) != len(set(names)):
            raise ValueError("Orbbec camera names must be unique")
        if any(config.mode not in {CameraMode.RGB, CameraMode.RGBD} for config in self.camera_configs):
            raise ValueError("CameraProcessDatasetRecorder requires Orbbec RGB or RGB-D cameras")

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(config.name for config in self.camera_configs)

    def open(self) -> "OrbbecCameraAdapterSession":
        """Start the SDK only in the spawned camera-recording worker."""

        session = OrbbecCameraAdapterSession(self.camera_configs)
        session.start()
        return session


class OrbbecCameraAdapterSession:
    """One started Orbbec manager and its RGB or RGB-D sources in a worker process."""

    def __init__(self, camera_configs: tuple[OrbbecCameraConfig, ...]):
        self._camera_configs = camera_configs
        self._manager: OrbbecManager | None = None
        self._sources: dict[str, CameraFrameSource] = {}

    @property
    def sources(self) -> Mapping[str, CameraFrameSource]:
        if self._manager is None:
            raise RuntimeError("Orbbec camera adapter session is not started")
        return self._sources

    def start(self) -> None:
        if self._manager is not None:
            raise RuntimeError("Orbbec camera adapter session is already started")
        manager = OrbbecManager(self._camera_configs)
        try:
            manager.start()
            sources: dict[str, CameraFrameSource] = {}
            for config in self._camera_configs:
                camera = manager.camera(config.name)
                if config.mode is CameraMode.RGB:
                    sources[config.name] = OrbbecRGBSource(camera)
                else:
                    sources[config.name] = OrbbecRGBDSource(camera)
        except BaseException:
            manager.stop()
            raise
        self._manager = manager
        self._sources = sources

    def close(self) -> None:
        manager = self._manager
        self._manager = None
        self._sources = {}
        if manager is not None:
            manager.stop()

    def source_health(self) -> dict[str, CameraSourceHealth]:
        """Return status snapshots without consuming any camera frame."""

        manager = self._manager
        if manager is None:
            return {}
        return {
            config.name: CameraSourceHealth(
                status=manager.camera(config.name).status.value,
                latest_capture_monotonic_ns=(
                    manager.camera(config.name).latest_capture_monotonic_ns
                ),
                error=manager.camera(config.name).last_error,
            )
            for config in self._camera_configs
        }
