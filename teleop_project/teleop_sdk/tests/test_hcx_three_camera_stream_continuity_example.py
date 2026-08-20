"""无硬件验证 HCX 三相机连续性诊断示例。"""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from examples import test_hcx_three_camera_stream_continuity as example
from leobot_scripts import CameraFrame
from orbbec_sdk import CameraMode, OrbbecCameraConfig


def _frame(index: int, capture_ns: int) -> CameraFrame:
    return CameraFrame(
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        capture_monotonic_ns=capture_ns,
        source_timestamp_ns=capture_ns,
        source_frame_index=index,
    )


class CameraStatsTest(unittest.TestCase):
    def test_tracks_reused_and_skipped_source_frames(self) -> None:
        stats = example.CameraStats()

        self.assertTrue(stats.observe(_frame(10, 1_000), 1_100))
        self.assertTrue(stats.observe(_frame(10, 1_000), 1_200))
        self.assertTrue(stats.observe(_frame(13, 1_300), 1_400))

        self.assertEqual(stats.selected, 3)
        self.assertEqual(stats.reused, 1)
        self.assertEqual(stats.skipped_source_frames, 2)
        self.assertEqual(stats.invalid, 0)
        self.assertEqual(stats.pair_delta_count, 3)

    def test_rejects_missing_future_and_invalid_rgb_frames(self) -> None:
        stats = example.CameraStats()
        invalid_rgb = CameraFrame(
            rgb=np.zeros((2, 3), dtype=np.uint8),
            capture_monotonic_ns=1_000,
            source_frame_index=1,
        )

        self.assertFalse(stats.observe(None, 1_000))
        self.assertFalse(stats.observe(_frame(2, 1_100), 1_000))
        self.assertFalse(stats.observe(invalid_rgb, 1_100))

        self.assertEqual(stats.missing, 1)
        self.assertEqual(stats.invalid, 2)


class CameraConfigTest(unittest.TestCase):
    @staticmethod
    def _configs(fps: int = 15) -> tuple[OrbbecCameraConfig, ...]:
        return tuple(
            OrbbecCameraConfig(
                name=name,
                serial_number=f"serial-{name}",
                mode=CameraMode.RGB,
                rgb_resolution=(1280, 720),
                fps=fps,
            )
            for name in example.CAMERA_NAMES
        )

    def test_selects_hcx_sections_in_required_order(self) -> None:
        configs = tuple(reversed(self._configs()))
        with (
            patch.object(
                example,
                "load_recording_config",
                return_value=SimpleNamespace(fps=15, root=example.Path("datasets/formal")),
            ) as load_recording,
            patch.object(
                example,
                "load_orbbec_camera_configs",
                return_value=configs,
            ) as load_cameras,
        ):
            selected = example._selected_configs()

        self.assertEqual(
            tuple(config.name for config in selected),
            example.CAMERA_NAMES,
        )
        load_recording.assert_called_once_with(section_name="hcx_recording")
        load_cameras.assert_called_once_with(section_name="hcx_orbbec")

    def test_rejects_camera_fps_mismatch(self) -> None:
        with (
            patch.object(
                example,
                "load_recording_config",
                return_value=SimpleNamespace(fps=15, root=example.Path("datasets/formal")),
            ),
            patch.object(
                example,
                "load_orbbec_camera_configs",
                return_value=self._configs(fps=30),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "fps"):
                example._selected_configs()


if __name__ == "__main__":
    unittest.main()
