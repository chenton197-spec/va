from __future__ import annotations

import inspect
import queue
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Callable

import numpy as np

import orbbec_sdk
from leobot_scripts.orbbec import (
    OrbbecCameraAdapterConfig,
    OrbbecDepthSource,
    OrbbecRGBDSource,
    OrbbecRGBSource,
)
from orbbec_sdk import (
    CameraMode,
    CameraStatus,
    OrbbecCameraConfig,
    OrbbecManager,
    OrbbecStartupError,
)
from orbbec_sdk.config import DEFAULT_XML_CONFIG_PATH
from orbbec_sdk.types import BackendSession, DeviceDescriptor, OrbbecFrame


class FakeBackend:
    def __init__(self, serials: tuple[str, ...]) -> None:
        self.devices = {
            serial: DeviceDescriptor(serial, f"fake-{serial}", object()) for serial in serials
        }
        self.frames = {serial: queue.Queue() for serial in serials}
        self.context_paths: list[Path | None] = []
        self.opened: list[str] = []
        self.closed: list[str] = []
        self.callback: Callable[[set[str], set[str]], None] | None = None

    def create_context(self, xml_config_path: Path | None) -> dict[str, Path | None]:
        self.context_paths.append(xml_config_path)
        return {"xml": xml_config_path}

    def list_devices(self, context: Any) -> list[DeviceDescriptor]:
        del context
        return list(self.devices.values())

    def register_device_changed_callback(
        self, context: Any, callback: Callable[[set[str], set[str]], None]
    ) -> None:
        del context
        self.callback = callback

    def open(self, context: Any, device: DeviceDescriptor, config: OrbbecCameraConfig) -> BackendSession:
        del context
        self.opened.append(device.serial_number)
        return BackendSession(descriptor=device, config=config, handle=device.serial_number)

    def read(self, session: BackendSession, timeout_ms: int) -> OrbbecFrame | None:
        try:
            return self.frames[session.descriptor.serial_number].get(timeout=timeout_ms / 1_000)
        except queue.Empty:
            return None

    def close(self, session: BackendSession) -> None:
        self.closed.append(session.descriptor.serial_number)

    def close_context(self, context: Any) -> None:
        del context

    def push(self, serial: str, frame: OrbbecFrame) -> None:
        self.frames[serial].put(frame)

    def remove(self, serial: str) -> None:
        assert self.callback is not None
        self.callback({serial}, set())


def _rgbd_frame(index: int) -> OrbbecFrame:
    return OrbbecFrame(
        rgb=np.full((4, 6, 3), index, dtype=np.uint8),
        depth_raw=np.full((4, 6), 1000 + index, dtype=np.uint16),
        meters_per_raw_unit=0.001,
        capture_monotonic_ns=10_000 + index,
        rgb_source_timestamp_ns=1_000_000 + index,
        depth_source_timestamp_ns=1_100_000 + index,
        rgb_source_frame_index=index,
        depth_source_frame_index=index,
        color_intrinsics=np.eye(3),
        aligned_to_rgb=True,
    )


def _depth_frame(index: int) -> OrbbecFrame:
    return OrbbecFrame(
        rgb=None,
        depth_raw=np.full((3, 5), 2000 + index, dtype=np.uint16),
        meters_per_raw_unit=0.001,
        capture_monotonic_ns=20_000 + index,
        depth_source_timestamp_ns=2_000_000 + index,
        depth_source_frame_index=index,
    )


def _rgb_frame(index: int) -> OrbbecFrame:
    return OrbbecFrame(
        rgb=np.full((2, 3, 3), index, dtype=np.uint8),
        depth_raw=None,
        meters_per_raw_unit=None,
        capture_monotonic_ns=40_000 + index,
        rgb_source_timestamp_ns=4_000_000 + index,
        rgb_source_frame_index=index,
    )


def _config(name: str, serial: str, mode: CameraMode) -> OrbbecCameraConfig:
    return OrbbecCameraConfig(
        name=name,
        serial_number=serial,
        mode=mode,
        frame_timeout_ms=20,
        max_consecutive_timeouts=100,
        first_frame_timeout_s=1.0,
    )


def _wait_for(value: Callable[[], object], timeout_s: float = 1.0) -> object:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = value()
        if result is not None:
            return result
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for expected value")


class OrbbecManagerTest(unittest.TestCase):
    def test_public_package_does_not_expose_resolution_presets(self) -> None:
        for name in ("RGB_RESOLUTION_SETTINGS", "DEPTH_RESOLUTION_SETTINGS", "FPS_SETTINGS"):
            self.assertFalse(hasattr(orbbec_sdk, name))
            self.assertNotIn(name, orbbec_sdk.__all__)

    def test_public_manager_signature_does_not_expose_xml(self) -> None:
        self.assertNotIn("xml_config_path", inspect.signature(OrbbecManager).parameters)

    def test_accepts_literal_resolution_lists_and_fps_declaration(self) -> None:
        config = OrbbecCameraConfig(
            name="front",
            serial_number="front-sn",
            mode=CameraMode.RGBD,
            rgb_resolution=[1280, 720],
            depth_resolution=[848, 480],
            fps=15,
        )
        self.assertEqual(config.rgb_resolution, (1280, 720))
        self.assertEqual(config.depth_resolution, (848, 480))
        self.assertEqual(config.fps, 15)

    def test_collection_adapter_accepts_rgb_and_rgbd_camera_names(self) -> None:
        adapter = OrbbecCameraAdapterConfig(
            (
                _config("front", "front-sn", CameraMode.RGB),
                _config("wrist", "wrist-sn", CameraMode.RGBD),
            )
        )

        self.assertEqual(adapter.camera_names, ("front", "wrist"))

    def test_collection_adapter_rejects_depth_only_camera(self) -> None:
        with self.assertRaisesRegex(ValueError, "RGB or RGB-D"):
            OrbbecCameraAdapterConfig((_config("front", "front-sn", CameraMode.DEPTH),))

    def test_rejects_incomplete_rgbd_profile_declaration(self) -> None:
        with self.assertRaisesRegex(ValueError, "both resolutions"):
            OrbbecCameraConfig(
                name="front",
                serial_number="front-sn",
                mode=CameraMode.RGBD,
                rgb_resolution=(848, 480),
                fps=15,
            )

    def test_uses_bundled_xml_by_default(self) -> None:
        backend = FakeBackend(("front-sn",))
        backend.push("front-sn", _rgbd_frame(1))
        manager = OrbbecManager([_config("front", "front-sn", CameraMode.RGBD)], backend=backend)
        manager.start()
        try:
            self.assertEqual(backend.context_paths, [DEFAULT_XML_CONFIG_PATH])
        finally:
            manager.stop()

    def test_startup_waits_for_the_first_frame_deadline_not_timeout_count(self) -> None:
        backend = FakeBackend(("front-sn",))
        config = OrbbecCameraConfig(
            name="front",
            serial_number="front-sn",
            mode=CameraMode.RGBD,
            frame_timeout_ms=20,
            max_consecutive_timeouts=2,
            first_frame_timeout_s=0.3,
        )
        manager = OrbbecManager([config], backend=backend)
        delayed_frame = threading.Timer(0.08, backend.push, args=("front-sn", _rgbd_frame(1)))
        delayed_frame.start()
        try:
            manager.start()
            self.assertEqual(manager.camera("front").status, CameraStatus.STREAMING)
        finally:
            delayed_frame.cancel()
            manager.stop()

    def test_streaming_camera_still_fails_after_consecutive_read_timeouts(self) -> None:
        backend = FakeBackend(("front-sn",))
        backend.push("front-sn", _rgbd_frame(1))
        config = OrbbecCameraConfig(
            name="front",
            serial_number="front-sn",
            mode=CameraMode.RGBD,
            frame_timeout_ms=20,
            max_consecutive_timeouts=2,
            first_frame_timeout_s=0.3,
        )
        manager = OrbbecManager([config], backend=backend)
        manager.start()
        try:
            _wait_for(lambda: manager.camera("front").last_error)
            self.assertEqual(manager.camera("front").status, CameraStatus.FAILED)
            self.assertIn("2 consecutive", manager.camera("front").last_error or "")
        finally:
            manager.stop()

    def test_starts_two_serials_with_one_bundled_xml_context(self) -> None:
        backend = FakeBackend(("front-sn", "top-sn"))
        backend.push("front-sn", _rgbd_frame(1))
        backend.push("top-sn", _depth_frame(2))
        manager = OrbbecManager(
            [_config("front", "front-sn", CameraMode.RGBD), _config("top", "top-sn", CameraMode.DEPTH)],
            backend=backend,
        )
        manager.start()
        try:
            self.assertEqual(backend.context_paths, [DEFAULT_XML_CONFIG_PATH])
            self.assertEqual(set(backend.opened), {"front-sn", "top-sn"})
            self.assertEqual(manager.camera("front").status, CameraStatus.STREAMING)
            rgbd = manager.camera("front").get_frame()
            depth = manager.camera("top").get_frame()
            assert rgbd is not None and depth is not None
            assert rgbd.rgb is not None and depth.depth is not None
            self.assertEqual(int(rgbd.rgb[0, 0, 0]), 1)
            self.assertEqual(int(depth.depth[0, 0]), 2_002)
        finally:
            manager.stop()

    def test_requires_every_configured_serial_before_opening_any_pipeline(self) -> None:
        backend = FakeBackend(("front-sn",))
        manager = OrbbecManager(
            [_config("front", "front-sn", CameraMode.RGBD), _config("top", "top-sn", CameraMode.DEPTH)],
            backend=backend,
        )
        with self.assertRaises(OrbbecStartupError) as raised:
            manager.start()
        self.assertIn("top-sn", str(raised.exception))
        self.assertIn("Detected serial numbers: front-sn", str(raised.exception))
        self.assertEqual(backend.opened, [])

    def test_bridge_owns_its_fresh_frame_cursor_and_preserves_timestamps(self) -> None:
        backend = FakeBackend(("front-sn",))
        backend.push("front-sn", _rgbd_frame(1))
        manager = OrbbecManager([_config("front", "front-sn", CameraMode.RGBD)], backend=backend)
        manager.start()
        try:
            first_source = OrbbecRGBDSource(manager.camera("front"))
            second_source = OrbbecRGBDSource(manager.camera("front"))
            first = first_source.latest_frame()
            self.assertIsNotNone(first)
            self.assertIsNone(first_source.latest_frame())
            self.assertIsNotNone(second_source.latest_frame())
            assert first is not None
            self.assertEqual(first.rgb_source_timestamp_ns, 1_000_001)
            self.assertEqual(first.depth.source_timestamp_ns, 1_100_001)
            self.assertEqual(first_source.metadata.source_timestamp_clock, "orbbec_global_timestamp_ns")

            backend.push("front-sn", _rgbd_frame(2))
            second = _wait_for(first_source.latest_frame)
            self.assertEqual(second.depth.source_frame_index, 2)
        finally:
            manager.stop()

    def test_get_frame_is_non_consuming_and_selects_buffered_frame_by_timestamp(self) -> None:
        backend = FakeBackend(("front-sn",))
        backend.push("front-sn", _rgbd_frame(5))
        manager = OrbbecManager([_config("front", "front-sn", CameraMode.RGBD)], backend=backend)
        manager.start()
        try:
            camera = manager.camera("front")
            frame = camera.get_frame()
            self.assertIs(frame, camera.get_frame())
            assert frame is not None and frame.rgb is not None and frame.depth is not None
            self.assertEqual(int(frame.rgb[0, 0, 0]), 5)
            self.assertEqual(int(frame.depth[0, 0]), 1_005)

            sequence_and_frame = camera.get_frame_after(0)
            assert sequence_and_frame is not None
            sequence, first = sequence_and_frame
            self.assertEqual(first.capture_monotonic_ns, 10_005)
            self.assertIsNone(camera.get_frame_after(sequence))

            backend.push("front-sn", _rgbd_frame(6))
            next_sequence_and_frame = _wait_for(lambda: camera.get_frame_after(sequence))
            assert isinstance(next_sequence_and_frame, tuple)
            next_sequence, second = next_sequence_and_frame
            self.assertGreater(next_sequence, sequence)
            self.assertEqual(second.capture_monotonic_ns, 10_006)

            backend.push("front-sn", _rgbd_frame(7))
            _wait_for(
                lambda: (
                    frame
                    if (frame := camera.get_frame_at_or_before(10_007)) is not None
                    and frame.capture_monotonic_ns == 10_007
                    else None
                )
            )
            next_in_order = camera.get_next_frame_after(sequence)
            assert next_in_order is not None
            in_order_sequence, in_order_frame = next_in_order
            self.assertEqual(in_order_sequence, next_sequence)
            self.assertEqual(in_order_frame.capture_monotonic_ns, 10_006)
            after_in_order = camera.get_next_frame_after(in_order_sequence)
            assert after_in_order is not None
            self.assertEqual(after_in_order[1].capture_monotonic_ns, 10_007)

            selected = camera.get_frame_at_or_before(10_005)
            assert selected is not None
            self.assertEqual(selected.capture_monotonic_ns, 10_005)
            self.assertIsNone(camera.get_frame_at_or_before(10_004))
        finally:
            manager.stop()

    def test_depth_bridge_uses_depth_only_mode(self) -> None:
        backend = FakeBackend(("top-sn",))
        backend.push("top-sn", _depth_frame(3))
        manager = OrbbecManager([_config("top", "top-sn", CameraMode.DEPTH)], backend=backend)
        manager.start()
        try:
            source = OrbbecDepthSource(manager.camera("top"))
            frame = source.latest_frame()
            assert frame is not None
            self.assertEqual(frame.raw.dtype, np.uint16)
            self.assertFalse(source.metadata.aligned_to_rgb)
            self.assertEqual(frame.source_timestamp_ns, 2_000_003)
        finally:
            manager.stop()

    def test_rgb_bridge_uses_rgb_only_mode(self) -> None:
        backend = FakeBackend(("side-sn",))
        backend.push("side-sn", _rgb_frame(4))
        manager = OrbbecManager([_config("side", "side-sn", CameraMode.RGB)], backend=backend)
        manager.start()
        try:
            source = OrbbecRGBSource(manager.camera("side"))
            frame = source.latest_frame()
            assert frame is not None
            self.assertEqual(frame.rgb.shape, (2, 3, 3))
            self.assertEqual(frame.source_timestamp_ns, 4_000_004)
            self.assertIsNone(source.latest_frame())

            backend.push("side-sn", _rgb_frame(5))
            next_frame = _wait_for(lambda: source.next_frame_after(1))
            assert isinstance(next_frame, tuple)
            sequence, frame = next_frame
            self.assertEqual(sequence, 2)
            self.assertEqual(frame.source_frame_index, 5)
            self.assertEqual(source.latest_sequence(), 2)
        finally:
            manager.stop()

    def test_device_removal_marks_the_camera_failed_without_reconnect(self) -> None:
        backend = FakeBackend(("front-sn",))
        backend.push("front-sn", _rgbd_frame(1))
        manager = OrbbecManager([_config("front", "front-sn", CameraMode.RGBD)], backend=backend)
        manager.start()
        try:
            backend.remove("front-sn")
            _wait_for(lambda: manager.camera("front").last_error)
            self.assertEqual(manager.camera("front").status, CameraStatus.FAILED)
            self.assertIn("disconnected", manager.camera("front").last_error or "")
        finally:
            manager.stop()

    def test_rejects_rgbd_frames_with_mismatched_aligned_dimensions(self) -> None:
        backend = FakeBackend(("front-sn",))
        valid = _rgbd_frame(1)
        backend.push(
            "front-sn",
            OrbbecFrame(
                rgb=valid.rgb,
                depth_raw=np.full((3, 6), 1001, dtype=np.uint16),
                meters_per_raw_unit=0.001,
                capture_monotonic_ns=valid.capture_monotonic_ns,
                aligned_to_rgb=True,
            ),
        )
        manager = OrbbecManager([_config("front", "front-sn", CameraMode.RGBD)], backend=backend)

        with self.assertRaisesRegex(OrbbecStartupError, "matching RGB and depth dimensions"):
            manager.start()

    def test_rejects_non_finite_depth_scale_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            OrbbecCameraConfig(
                name="front",
                serial_number="front-sn",
                mode=CameraMode.DEPTH,
                depth_scale_to_meters=float("nan"),
            )


if __name__ == "__main__":
    unittest.main()
