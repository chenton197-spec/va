"""Lifecycle and timestamped-frame buffering for one or more Orbbec cameras."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import replace
import threading
from typing import Any, Iterable

import numpy as np

from .backend import CameraBackend, PyOrbbecBackend
from .config import DEFAULT_XML_CONFIG_PATH
from .types import (
    BackendSession,
    CameraMetadata,
    CameraMode,
    CameraStatus,
    DeviceDescriptor,
    OrbbecCameraConfig,
    OrbbecFrame,
)


class OrbbecError(RuntimeError):
    """Base error for the independent Orbbec SDK."""


class OrbbecStartupError(OrbbecError):
    """Raised when configured cameras cannot all enter streaming state."""


_FRAME_BUFFER_CAPACITY = 8


class OrbbecCamera:
    """One configured device with a background reader and a recent-frame buffer.

    The reader is the only code that talks to the vendor pipeline.  Consumers
    can independently inspect the buffered frames, so a preview, a recorder,
    and a diagnostic tool never consume frames from one another.
    """

    def __init__(self, config: OrbbecCameraConfig):
        self.config = config
        self._state_lock = threading.RLock()
        self._status = CameraStatus.CREATED
        self._last_error: str | None = None
        self._metadata = CameraMetadata(
            serial_number=config.serial_number,
            camera_model=None,
            aligned_to_rgb=config.mode is CameraMode.RGBD,
        )
        self._backend: CameraBackend | None = None
        self._session: BackendSession | None = None
        self._session_closed = False
        self._stop_event = threading.Event()
        self._first_frame_event = threading.Event()
        self._reader: threading.Thread | None = None
        self._frames: deque[tuple[int, OrbbecFrame]] = deque(maxlen=_FRAME_BUFFER_CAPACITY)
        self._sequence = 0

    @property
    def status(self) -> CameraStatus:
        with self._state_lock:
            return self._status

    @property
    def last_error(self) -> str | None:
        with self._state_lock:
            return self._last_error

    @property
    def metadata(self) -> CameraMetadata:
        with self._state_lock:
            return self._metadata

    @property
    def rgb_shape(self) -> tuple[int, int, int]:
        shape = self.metadata.rgb_shape
        if shape is None:
            raise OrbbecError(f"Camera {self.config.name} has not received an RGB frame")
        return shape

    @property
    def depth_shape(self) -> tuple[int, int]:
        shape = self.metadata.depth_shape
        if shape is None:
            raise OrbbecError(f"Camera {self.config.name} has not received a depth frame")
        return shape

    def get_frame(self) -> OrbbecFrame | None:
        """Return the latest complete frame without consuming it.

        Repeated calls may return the same physical frame when the camera has
        not produced another one yet.  Use :meth:`get_frame_after` when a
        caller needs to track its own fresh-frame cursor, or
        :meth:`get_frame_at_or_before` when synchronizing a recording tick.
        """

        with self._state_lock:
            if self._status is not CameraStatus.STREAMING:
                return None
            if not self._frames:
                return None
            return self._frames[-1][1]

    def get_frame_after(self, sequence: int) -> tuple[int, OrbbecFrame] | None:
        """Return the latest frame newer than one caller-owned sequence number.

        ``sequence`` belongs to the caller, rather than to the camera.  This
        keeps fresh-frame delivery independent for every preview or recorder.
        """

        with self._state_lock:
            if self._status is not CameraStatus.STREAMING or not self._frames:
                return None
            latest_sequence, latest_frame = self._frames[-1]
            if latest_sequence <= sequence:
                return None
            return latest_sequence, latest_frame

    def get_next_frame_after(self, sequence: int) -> tuple[int, OrbbecFrame] | None:
        """Return the oldest buffered frame newer than ``sequence``.

        Camera-driven recorders advance their own cursor with this method so
        each buffered physical frame is considered once, in capture order.
        If a consumer falls farther behind than the bounded buffer, the oldest
        retained frame is returned and the source-frame index audit exposes the
        resulting gap.
        """

        with self._state_lock:
            if self._status is not CameraStatus.STREAMING:
                return None
            for frame_sequence, frame in self._frames:
                if frame_sequence > sequence:
                    return frame_sequence, frame
            return None

    def latest_sequence(self) -> int:
        """Return the newest sequence number without consuming its frame."""

        with self._state_lock:
            if self._status is not CameraStatus.STREAMING or not self._frames:
                return 0
            return self._frames[-1][0]

    @property
    def latest_capture_monotonic_ns(self) -> int | None:
        """Return the newest buffered capture time even after the stream fails."""

        with self._state_lock:
            if not self._frames:
                return None
            return self._frames[-1][1].capture_monotonic_ns

    def get_frame_at_or_before(self, target_monotonic_ns: int) -> OrbbecFrame | None:
        """Return the newest buffered frame captured no later than ``target``.

        This is intentionally non-blocking and causal.  A recording sample is
        therefore paired with an image already available at its target time,
        rather than waiting for a future image and shifting the time axis.
        """

        if isinstance(target_monotonic_ns, bool) or not isinstance(target_monotonic_ns, int):
            raise TypeError("target_monotonic_ns must be an integer")
        with self._state_lock:
            if self._status is not CameraStatus.STREAMING:
                return None
            for _, frame in reversed(self._frames):
                if frame.capture_monotonic_ns <= target_monotonic_ns:
                    return frame
            return None

    def _start(self, backend: CameraBackend, session: BackendSession) -> None:
        with self._state_lock:
            if self._status is not CameraStatus.CREATED:
                raise OrbbecError(f"Camera {self.config.name} cannot be started from {self._status.value}")
            self._backend = backend
            self._session = session
            self._metadata = replace(self._metadata, camera_model=session.descriptor.model)
            self._status = CameraStatus.STARTING
            self._reader = threading.Thread(
                target=self._run_reader,
                name=f"orbbec-{self.config.name}",
                daemon=True,
            )
            self._reader.start()

        if not self._first_frame_event.wait(self.config.first_frame_timeout_s):
            self._mark_failed("Timed out waiting for the first complete frame")
            self.stop()
            raise OrbbecStartupError(f"Camera {self.config.name} did not produce a first frame")
        if self.status is not CameraStatus.STREAMING:
            error = self.last_error or "unknown startup error"
            self.stop()
            raise OrbbecStartupError(f"Camera {self.config.name} failed to start: {error}")

    def stop(self) -> None:
        with self._state_lock:
            self._stop_event.set()
            reader = self._reader
        self._close_session()
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=max(1.0, self.config.frame_timeout_ms / 1_000 * 2))
        with self._state_lock:
            if self._status is not CameraStatus.FAILED:
                self._status = CameraStatus.STOPPED
            self._reader = None

    def _mark_failed(self, reason: str) -> None:
        with self._state_lock:
            if self._status is CameraStatus.STOPPED:
                return
            self._status = CameraStatus.FAILED
            self._last_error = reason
            self._stop_event.set()
            self._first_frame_event.set()

    def _run_reader(self) -> None:
        consecutive_timeouts = 0
        try:
            while not self._stop_event.is_set():
                backend, session = self._require_backend_session()
                frame = backend.read(session, self.config.frame_timeout_ms)
                if frame is None:
                    # Startup has its own complete first-frame deadline. Do not
                    # turn a slow SDK/pipeline warm-up into a shorter timeout
                    # merely because several 200 ms reads returned no frame.
                    if self.status is CameraStatus.STREAMING:
                        consecutive_timeouts += 1
                        if consecutive_timeouts >= self.config.max_consecutive_timeouts:
                            self._mark_failed(
                                f"No complete frame for {consecutive_timeouts} consecutive read timeouts"
                            )
                    continue
                consecutive_timeouts = 0
                self._accept_frame(frame)
        except BaseException as exc:
            if not self._stop_event.is_set():
                self._mark_failed(repr(exc))
        finally:
            self._close_session()
            with self._state_lock:
                if self._status is CameraStatus.STARTING and not self._stop_event.is_set():
                    self._mark_failed("Reader stopped before receiving a complete frame")

    def _accept_frame(self, frame: OrbbecFrame) -> None:
        if self.config.mode is CameraMode.RGB and frame.rgb is None:
            raise OrbbecError("RGB camera backend returned no RGB data")
        if self.config.mode is CameraMode.DEPTH and frame.depth_raw is None:
            raise OrbbecError("Depth camera backend returned no depth data")
        if self.config.mode is CameraMode.RGBD and (frame.rgb is None or frame.depth_raw is None):
            raise OrbbecError("RGB-D camera backend returned an incomplete frame")
        if frame.depth_raw is not None and frame.meters_per_raw_unit is None:
            raise OrbbecError("Depth camera backend omitted meters_per_raw_unit")
        if frame.rgb is not None and (
            frame.rgb.dtype != np.uint8 or frame.rgb.ndim != 3 or frame.rgb.shape[2] != 3
        ):
            raise OrbbecError("Camera backend must return H x W x 3 uint8 RGB data")
        if frame.depth_raw is not None:
            if frame.depth_raw.dtype != np.uint16 or frame.depth_raw.ndim != 2:
                raise OrbbecError("Camera backend must return H x W uint16 raw depth")
            assert frame.meters_per_raw_unit is not None
            if not math.isfinite(frame.meters_per_raw_unit) or frame.meters_per_raw_unit <= 0.0:
                raise OrbbecError("Camera backend returned an invalid depth scale")
        if self.config.mode is CameraMode.RGBD and not frame.aligned_to_rgb:
            raise OrbbecError("RGB-D camera backend returned unaligned depth")
        if (
            self.config.mode is CameraMode.RGBD
            and frame.rgb is not None
            and frame.depth_raw is not None
            and frame.rgb.shape[:2] != frame.depth_raw.shape
        ):
            raise OrbbecError("Aligned RGB-D data must use matching RGB and depth dimensions")

        with self._state_lock:
            self._metadata = replace(
                self._metadata,
                rgb_shape=None if frame.rgb is None else tuple(int(v) for v in frame.rgb.shape),
                depth_shape=None if frame.depth_raw is None else tuple(int(v) for v in frame.depth_raw.shape),
                color_intrinsics=frame.color_intrinsics,
                aligned_to_rgb=frame.aligned_to_rgb,
            )
            self._sequence += 1
            self._frames.append((self._sequence, frame))
            self._status = CameraStatus.STREAMING
            self._first_frame_event.set()

    def _require_backend_session(self) -> tuple[CameraBackend, BackendSession]:
        with self._state_lock:
            if self._backend is None or self._session is None:
                raise OrbbecError("Camera reader started without a backend session")
            return self._backend, self._session

    def _close_session(self) -> None:
        with self._state_lock:
            if self._session_closed or self._backend is None or self._session is None:
                return
            backend = self._backend
            session = self._session
            self._session_closed = True
        backend.close(session)


class OrbbecManager:
    """Own all configured serial-number pipelines using the bundled SDK XML."""

    def __init__(
        self,
        cameras: Iterable[OrbbecCameraConfig],
        *,
        backend: CameraBackend | None = None,
    ) -> None:
        configs = tuple(cameras)
        if not configs:
            raise ValueError("At least one Orbbec camera must be configured")
        names = [config.name for config in configs]
        serials = [config.serial_number for config in configs]
        if len(names) != len(set(names)) or len(serials) != len(set(serials)):
            raise ValueError("Orbbec camera names and serial numbers must be unique")
        self._xml_config_path = DEFAULT_XML_CONFIG_PATH
        self._backend = backend or PyOrbbecBackend()
        self._cameras = {config.name: OrbbecCamera(config) for config in configs}
        self._by_serial = {camera.config.serial_number: camera for camera in self._cameras.values()}
        self._context: Any | None = None
        self._state_lock = threading.RLock()

    @property
    def cameras(self) -> dict[str, OrbbecCamera]:
        return dict(self._cameras)

    def camera(self, name: str) -> OrbbecCamera:
        try:
            return self._cameras[name]
        except KeyError as exc:
            raise KeyError(f"Unknown Orbbec camera: {name}") from exc

    def start(self) -> None:
        with self._state_lock:
            if self._context is not None:
                raise OrbbecError("OrbbecManager is already started")
            context = self._backend.create_context(self._xml_config_path)
            self._context = context
        try:
            self._backend.register_device_changed_callback(context, self._on_device_changed)
            devices = {device.serial_number: device for device in self._backend.list_devices(context)}
            missing = sorted(set(self._by_serial) - set(devices))
            if missing:
                detected = ", ".join(sorted(devices)) or "none"
                raise OrbbecStartupError(
                    f"Configured cameras are not connected: {', '.join(missing)}. "
                    f"Detected serial numbers: {detected}"
                )
            for camera in self._cameras.values():
                descriptor: DeviceDescriptor = devices[camera.config.serial_number]
                session = self._backend.open(context, descriptor, camera.config)
                camera._start(self._backend, session)
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        for camera in self._cameras.values():
            camera.stop()
        with self._state_lock:
            context = self._context
            self._context = None
        if context is not None:
            self._backend.close_context(context)

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> OrbbecManager:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

    def _on_device_changed(self, removed_serials: set[str], added_serials: set[str]) -> None:
        del added_serials
        for serial_number in removed_serials:
            camera = self._by_serial.get(serial_number)
            if camera is not None:
                camera._mark_failed("Device disconnected")
