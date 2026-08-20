from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from orbbec_sdk.backend import PyOrbbecBackend
from orbbec_sdk.types import BackendSession, CameraMode, DeviceDescriptor, OrbbecCameraConfig


class _VideoFrame:
    def __init__(self, data: bytes, width: int, height: int, format_value: str) -> None:
        self._data = data
        self._width = width
        self._height = height
        self._format = format_value

    def get_data(self) -> bytes:
        return self._data

    def get_width(self) -> int:
        return self._width

    def get_height(self) -> int:
        return self._height

    def get_format(self) -> str:
        return self._format


class _ColorFrame:
    def __init__(self, video_frame: _VideoFrame) -> None:
        self._video_frame = video_frame

    def as_video_frame(self) -> _VideoFrame:
        return self._video_frame

    def get_global_timestamp_us(self) -> int:
        return 100

    def get_index(self) -> int:
        return 3


class _DepthFrame(_VideoFrame):
    def __init__(self, data: bytes) -> None:
        super().__init__(data, 2, 1, "Y16")

    def get_depth_scale(self) -> float:
        return 1.0

    def get_global_timestamp_us(self) -> int:
        return 200

    def get_index(self) -> int:
        return 4


class _BaseDepthFrame:
    def __init__(self, depth_frame: _DepthFrame) -> None:
        self._depth_frame = depth_frame

    def as_depth_frame(self) -> _DepthFrame:
        return self._depth_frame


class _FrameSet:
    def __init__(self, color_frame: _ColorFrame, depth_frame: _BaseDepthFrame) -> None:
        self._color_frame = color_frame
        self._depth_frame = depth_frame

    def get_color_frame(self) -> _ColorFrame:
        return self._color_frame

    def get_depth_frame(self) -> _BaseDepthFrame:
        return self._depth_frame


class _Pipeline:
    def __init__(self, frames: _FrameSet) -> None:
        self._frames = frames

    def wait_for_frames(self, timeout_ms: int) -> _FrameSet:
        del timeout_ms
        return self._frames


class _Device:
    def __init__(self, serial_number: str) -> None:
        self._serial_number = serial_number

    def get_device_info(self) -> SimpleNamespace:
        return SimpleNamespace(
            get_serial_number=lambda: self._serial_number,
            get_name=lambda: "fake-camera",
        )


class _DeviceList:
    def get_count(self) -> int:
        return 2

    def get_device_by_index(self, index: int) -> _Device:
        if index == 0:
            raise RuntimeError("access denied")
        return _Device("available-sn")


class _Context:
    def query_devices(self) -> _DeviceList:
        return _DeviceList()


class PyOrbbecBackendTest(unittest.TestCase):
    def test_list_devices_skips_an_inaccessible_device(self) -> None:
        backend = PyOrbbecBackend()

        devices = backend.list_devices(_Context())

        self.assertEqual([device.serial_number for device in devices], ["available-sn"])

    def test_read_casts_base_frames_before_decoding(self) -> None:
        backend = PyOrbbecBackend()
        backend._sdk = SimpleNamespace(OBFormat=SimpleNamespace(RGB="RGB", BGR="BGR", Y16="Y16"))

        color = _ColorFrame(_VideoFrame(bytes([1, 2, 3, 4, 5, 6]), 2, 1, "RGB"))
        depth = _BaseDepthFrame(_DepthFrame(np.array([1000, 2000], dtype=np.uint16).tobytes()))
        session = BackendSession(
            descriptor=DeviceDescriptor("camera-sn", "fake", object()),
            config=OrbbecCameraConfig(name="camera", serial_number="camera-sn", mode=CameraMode.RGBD),
            handle={"pipeline": _Pipeline(_FrameSet(color, depth)), "align_filter": None},
        )

        frame = backend.read(session, timeout_ms=200)

        assert frame is not None
        np.testing.assert_array_equal(frame.rgb, np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8))
        np.testing.assert_array_equal(frame.depth, np.array([[1000, 2000]], dtype=np.uint16))
        self.assertEqual(frame.rgb_source_timestamp_ns, 100_000)
        self.assertEqual(frame.depth_source_timestamp_ns, 200_000)


if __name__ == "__main__":
    unittest.main()
