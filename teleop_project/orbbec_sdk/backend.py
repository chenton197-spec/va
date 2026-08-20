"""Optional pyorbbecsdk backend kept outside collection-facing code."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from .types import (
    AlignmentMode,
    BackendSession,
    CameraMode,
    DeviceDescriptor,
    OrbbecCameraConfig,
    OrbbecFrame,
)


class OrbbecSdkUnavailableError(RuntimeError):
    """Raised when the optional vendor Python package is not installed."""


class CameraBackend(Protocol):
    """Small backend boundary that lets tests run without a camera or vendor SDK."""

    def create_context(self, xml_config_path: Path | None) -> Any:
        ...

    def list_devices(self, context: Any) -> list[DeviceDescriptor]:
        ...

    def register_device_changed_callback(
        self, context: Any, callback: Callable[[set[str], set[str]], None]
    ) -> None:
        ...

    def open(self, context: Any, device: DeviceDescriptor, config: OrbbecCameraConfig) -> BackendSession:
        ...

    def read(self, session: BackendSession, timeout_ms: int) -> OrbbecFrame | None:
        ...

    def close(self, session: BackendSession) -> None:
        ...

    def close_context(self, context: Any) -> None:
        ...


class PyOrbbecBackend:
    """``pyorbbecsdk2`` implementation loaded only when a manager starts."""

    def __init__(self) -> None:
        self._sdk: Any | None = None

    def create_context(self, xml_config_path: Path | None) -> Any:
        sdk = self._vendor_sdk()
        return sdk.Context() if xml_config_path is None else sdk.Context(str(xml_config_path))

    def list_devices(self, context: Any) -> list[DeviceDescriptor]:
        result: list[DeviceDescriptor] = []
        for device in self._devices_from_list(context.query_devices()):
            try:
                info = device.get_device_info()
                serial_number = str(info.get_serial_number())
                if not serial_number:
                    continue
                result.append(
                    DeviceDescriptor(
                        serial_number=serial_number,
                        model=str(info.get_name()),
                        handle=device,
                    )
                )
            except Exception:
                # A different USB device can be visible but inaccessible to
                # this user. Keep enumerating usable configured cameras.
                continue
        return result

    def register_device_changed_callback(
        self, context: Any, callback: Callable[[set[str], set[str]], None]
    ) -> None:
        def _on_change(removed: Any, added: Any) -> None:
            callback(self._serial_numbers(removed), self._serial_numbers(added))

        context.set_device_changed_callback(_on_change)

    def open(self, context: Any, device: DeviceDescriptor, config: OrbbecCameraConfig) -> BackendSession:
        del context
        sdk = self._vendor_sdk()
        pipeline = sdk.Pipeline(device.handle)
        stream_config = sdk.Config()
        color_profile: Any | None = None
        depth_profile: Any | None = None

        if config.mode in {CameraMode.RGB, CameraMode.RGBD}:
            color_profile = self._select_profile(
                pipeline,
                sdk.OBSensorType.COLOR_SENSOR,
                config.rgb_resolution,
                config.fps,
                "MJPG",
            )
            stream_config.enable_stream(color_profile)

        if config.mode in {CameraMode.DEPTH, CameraMode.RGBD}:
            if config.mode is CameraMode.RGBD and config.alignment is AlignmentMode.HARDWARE:
                assert color_profile is not None
                depth_profile = self._select_hardware_aligned_depth_profile(
                    pipeline, color_profile, config.depth_resolution, config.fps
                )
                stream_config.set_align_mode(sdk.OBAlignMode.HW_MODE)
            else:
                depth_profile = self._select_profile(
                    pipeline,
                    sdk.OBSensorType.DEPTH_SENSOR,
                    config.depth_resolution,
                    config.fps,
                    "Y16",
                )
            stream_config.enable_stream(depth_profile)

        align_filter: Any | None = None
        if config.mode is CameraMode.RGBD:
            stream_config.set_frame_aggregate_output_mode(sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            pipeline.enable_frame_sync()
            if config.alignment is AlignmentMode.SOFTWARE:
                align_filter = sdk.AlignFilter(align_to_stream=sdk.OBStreamType.COLOR_STREAM)

        pipeline.start(stream_config)
        return BackendSession(
            descriptor=device,
            config=config,
            handle={"pipeline": pipeline, "align_filter": align_filter},
        )

    def read(self, session: BackendSession, timeout_ms: int) -> OrbbecFrame | None:
        native = session.handle
        frames = native["pipeline"].wait_for_frames(timeout_ms)
        if not frames:
            return None
        align_filter = native["align_filter"]
        if align_filter is not None:
            frames = align_filter.process(frames)
            if not frames:
                return None
            frames = frames.as_frame_set()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        config = session.config
        if config.mode is CameraMode.RGB and not color_frame:
            return None
        if config.mode is CameraMode.DEPTH and not depth_frame:
            return None
        if config.mode is CameraMode.RGBD and (not color_frame or not depth_frame):
            return None

        rgb = self._color_to_rgb(color_frame) if color_frame else None
        depth_raw: np.ndarray | None = None
        meters_per_raw_unit: float | None = None
        if depth_frame:
            # AlignFilter returns a base Frame in pyorbbecsdk2; restore the
            # depth-specific wrapper before accessing video geometry and scale.
            depth_frame = depth_frame.as_depth_frame()
            sdk = self._vendor_sdk()
            if depth_frame.get_format() != sdk.OBFormat.Y16:
                raise RuntimeError(f"Expected Y16 depth, got {depth_frame.get_format()}")
            depth_raw = self._copy_frame_data(depth_frame, np.uint16).reshape(
                depth_frame.get_height(), depth_frame.get_width()
            )
            meters_per_raw_unit = float(depth_frame.get_depth_scale()) * config.depth_scale_to_meters

        return OrbbecFrame(
            rgb=rgb,
            depth_raw=depth_raw,
            meters_per_raw_unit=meters_per_raw_unit,
            capture_monotonic_ns=time.perf_counter_ns(),
            rgb_source_timestamp_ns=self._global_timestamp_ns(color_frame),
            depth_source_timestamp_ns=self._global_timestamp_ns(depth_frame),
            rgb_source_frame_index=self._frame_index(color_frame),
            depth_source_frame_index=self._frame_index(depth_frame),
            color_intrinsics=self._color_intrinsics(color_frame),
            aligned_to_rgb=config.mode is CameraMode.RGBD,
        )

    def close(self, session: BackendSession) -> None:
        try:
            session.handle["pipeline"].stop()
        except Exception:
            # Pipelines can already be invalid after device removal.
            pass

    def close_context(self, context: Any) -> None:
        try:
            context.unregister_device_changed_callback()
        except Exception:
            pass

    def _vendor_sdk(self) -> Any:
        if self._sdk is None:
            try:
                import pyorbbecsdk
            except ImportError as exc:
                raise OrbbecSdkUnavailableError(
                    "pyorbbecsdk2 is required for Orbbec hardware. "
                    "Install requirements.txt or the matching official wheel."
                ) from exc
            self._sdk = pyorbbecsdk
        return self._sdk

    def _devices_from_list(self, device_list: Any) -> list[Any]:
        try:
            count = int(device_list.get_count())
        except AttributeError:
            try:
                return list(device_list)
            except TypeError:
                return []

        devices: list[Any] = []
        for index in range(count):
            try:
                devices.append(device_list.get_device_by_index(index))
            except Exception:
                continue
        return devices

    def _serial_numbers(self, device_list: Any) -> set[str]:
        serials: set[str] = set()
        for device in self._devices_from_list(device_list):
            try:
                serials.add(str(device.get_device_info().get_serial_number()))
            except Exception:
                continue
        return serials

    def _select_profile(
        self,
        pipeline: Any,
        sensor_type: Any,
        resolution: tuple[int, int] | list[int] | None,
        fps: int | None,
        format_name: str,
    ) -> Any:
        profile_list = pipeline.get_stream_profile_list(sensor_type)
        if resolution is None:
            return profile_list.get_default_video_stream_profile()
        assert fps is not None
        format_value = getattr(self._vendor_sdk().OBFormat, format_name)
        return profile_list.get_video_stream_profile(
            resolution[0], resolution[1], format_value, fps
        )

    def _select_hardware_aligned_depth_profile(
        self,
        pipeline: Any,
        color_profile: Any,
        resolution: tuple[int, int] | list[int] | None,
        fps: int | None,
    ) -> Any:
        sdk = self._vendor_sdk()
        profile_list = pipeline.get_d2c_depth_profile_list(color_profile, sdk.OBAlignMode.HW_MODE)
        if resolution is None:
            return profile_list.get_default_video_stream_profile()
        assert fps is not None
        return profile_list.get_video_stream_profile(
            resolution[0], resolution[1], sdk.OBFormat.Y16, fps
        )

    def _color_to_rgb(self, frame: Any) -> np.ndarray:
        sdk = self._vendor_sdk()
        # The software alignment path yields a base Frame rather than a
        # ColorFrame, so obtain the video-frame API used by the RSDT driver.
        frame = frame.as_video_frame()
        frame_to_read = frame
        if frame.get_format() == sdk.OBFormat.BGR:
            data = self._copy_frame_data(frame, np.uint8).reshape(frame.get_height(), frame.get_width(), 3)
            return data[:, :, ::-1].copy()
        if frame.get_format() != sdk.OBFormat.RGB:
            conversion = self._convert_format(frame.get_format())
            converter = sdk.FormatConvertFilter()
            converter.set_format_convert_format(conversion)
            frame_to_read = converter.process(frame)
            if frame_to_read is None:
                raise RuntimeError(f"Failed to convert color format {frame.get_format()} to RGB")
            frame_to_read = frame_to_read.as_video_frame()
        return self._copy_frame_data(frame_to_read, np.uint8).reshape(
            frame_to_read.get_height(), frame_to_read.get_width(), 3
        )

    def _convert_format(self, source_format: Any) -> Any:
        sdk = self._vendor_sdk()
        formats = {
            sdk.OBFormat.I420: sdk.OBConvertFormat.I420_TO_RGB888,
            sdk.OBFormat.MJPG: sdk.OBConvertFormat.MJPG_TO_RGB888,
            sdk.OBFormat.YUYV: sdk.OBConvertFormat.YUYV_TO_RGB888,
            sdk.OBFormat.NV21: sdk.OBConvertFormat.NV21_TO_RGB888,
            sdk.OBFormat.NV12: sdk.OBConvertFormat.NV12_TO_RGB888,
            sdk.OBFormat.UYVY: sdk.OBConvertFormat.UYVY_TO_RGB888,
        }
        result = formats.get(source_format)
        if result is None:
            raise ValueError(f"Unsupported color format: {source_format}")
        return result

    @staticmethod
    def _copy_frame_data(frame: Any, dtype: Any) -> np.ndarray:
        buffer = frame.get_data()
        try:
            return np.frombuffer(buffer, dtype=dtype).copy()
        except (TypeError, ValueError):
            array = np.asarray(buffer)
            if array.dtype == np.dtype(dtype):
                return array.reshape(-1).copy()
            return np.ascontiguousarray(array).view(np.uint8).view(dtype).reshape(-1).copy()

    @staticmethod
    def _global_timestamp_ns(frame: Any | None) -> int | None:
        if frame is None:
            return None
        value = int(frame.get_global_timestamp_us())
        return value * 1_000 if value > 0 else None

    @staticmethod
    def _frame_index(frame: Any | None) -> int | None:
        return None if frame is None else int(frame.get_index())

    @staticmethod
    def _color_intrinsics(frame: Any | None) -> np.ndarray | None:
        if frame is None:
            return None
        try:
            intrinsics = frame.get_stream_profile().as_video_stream_profile().get_intrinsic()
            return np.array(
                [[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        except Exception:
            return None
