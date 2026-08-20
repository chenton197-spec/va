"""YAML deployment configuration tests for the independent Orbbec SDK."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orbbec_sdk.config import load_orbbec_camera_configs
from orbbec_sdk.types import CameraMode, OrbbecCameraConfig


class OrbbecYamlConfigTest(unittest.TestCase):
    def test_loads_literal_camera_declarations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """orbbec:
  cameras:
    - name: front
      serial_number: front-sn
      rgb_resolution: [1280, 720]
      depth_resolution: [848, 480]
      fps: 30
    - name: top
      serial_number: top-sn
      mode: depth
      depth_resolution: [640, 480]
      fps: 15
""",
                encoding="utf-8",
            )

            configs = load_orbbec_camera_configs(path)

        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0].rgb_resolution, (1280, 720))
        self.assertEqual(configs[0].depth_resolution, (848, 480))
        self.assertEqual(configs[0].fps, 30)
        self.assertEqual(configs[0].mode, CameraMode.RGBD)
        self.assertEqual(configs[1].mode, CameraMode.DEPTH)

    def test_loads_hcx_orbbec_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """hcx_orbbec:
  cameras:
    - name: head
      serial_number: head-sn
      mode: rgb
      rgb_resolution: [1280, 720]
      fps: 30
    - name: left_hand
      serial_number: left-sn
      mode: rgb
      rgb_resolution: [1280, 720]
      fps: 30
    - name: right_hand
      serial_number: right-sn
      mode: rgb
      rgb_resolution: [1280, 720]
      fps: 30
""",
                encoding="utf-8",
            )

            configs = load_orbbec_camera_configs(path, section_name="hcx_orbbec")

        self.assertEqual([config.name for config in configs], ["head", "left_hand", "right_hand"])
        self.assertTrue(all(config.mode is CameraMode.RGB for config in configs))

    def test_rejects_unknown_camera_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """orbbec:
  cameras:
    - name: front
      serial_number: front-sn
      profile: fast
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "未知字段: profile"):
                load_orbbec_camera_configs(path)

    def test_rgb_mode_keeps_an_inactive_depth_profile(self) -> None:
        config = OrbbecCameraConfig(
            name="front",
            serial_number="front-sn",
            mode=CameraMode.RGB,
            rgb_resolution=[1280, 720],
            depth_resolution=[848, 480],
            fps=30,
        )

        self.assertEqual(config.rgb_resolution, (1280, 720))
        self.assertEqual(config.depth_resolution, (848, 480))

    def test_rejects_unknown_orbbec_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """orbbec:
  camera: []
  cameras:
    - name: front
      serial_number: front-sn
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "orbbec 包含未知字段: camera"):
                load_orbbec_camera_configs(path)

    def test_rejects_duplicate_camera_serial_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """orbbec:
  cameras:
    - name: left
      serial_number: shared-sn
    - name: right
      serial_number: shared-sn
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "必须唯一"):
                load_orbbec_camera_configs(path)


if __name__ == "__main__":
    unittest.main()
