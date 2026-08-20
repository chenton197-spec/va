"""YAML deployment configuration tests for the collection package."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from leobot_scripts.config import load_recording_config


class RecordingYamlConfigTest(unittest.TestCase):
    def test_loads_and_resolves_relative_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """recording:
  root: datasets/demo
  fps: 30
  task: move cube
  image_storage: png
""",
                encoding="utf-8",
            )

            config = load_recording_config(path)

        self.assertEqual(config.root, (path.parent / "datasets/demo").resolve())
        self.assertEqual(config.fps, 30)
        self.assertEqual(config.numeric_sample_fps, 30)
        self.assertIsNone(config.master_camera)
        self.assertEqual(config.enabled_cameras, ())
        self.assertEqual(config.min_free_disk_gb, 10.0)
        self.assertEqual(config.task, "move cube")
        self.assertEqual(config.image_storage, "png")
        self.assertEqual(config.quality, 75)

    def test_defaults_image_storage_to_video_for_legacy_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """recording:
  root: datasets/demo
  fps: 30
  task: move cube
""",
                encoding="utf-8",
            )

            config = load_recording_config(path)

        self.assertEqual(config.image_storage, "video")
        self.assertEqual(config.quality, 75)

    def test_loads_independent_numeric_rate_and_master_camera(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """recording:
  root: datasets/demo
  fps: 30
  numeric_sample_fps: 60
  master_camera: hand
  enabled_cameras: [hand, head]
  min_free_disk_gb: 25
  task: move cube
""",
                encoding="utf-8",
            )

            config = load_recording_config(path)

        self.assertEqual(config.fps, 30)
        self.assertEqual(config.numeric_sample_fps, 60)
        self.assertEqual(config.master_camera, "hand")
        self.assertEqual(config.enabled_cameras, ("hand", "head"))
        self.assertEqual(config.min_free_disk_gb, 25.0)

    def test_loads_hcx_recording_section_without_fixed_numeric_sampler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """hcx_recording:
  root: datasets/hcx
  fps: 30
  task: dual arm task
  master_camera: head
  enabled_cameras: [head, left_hand, right_hand]
""",
                encoding="utf-8",
            )

            config = load_recording_config(path, section_name="hcx_recording")

        self.assertEqual(config.root, (path.parent / "datasets/hcx").resolve())
        self.assertEqual(config.master_camera, "head")
        self.assertEqual(config.enabled_cameras, ("head", "left_hand", "right_hand"))
        # The generic config keeps a compatibility default, but the HCX
        # master-triggered recorder never consumes this value.
        self.assertEqual(config.numeric_sample_fps, 30)

    def test_rejects_unknown_recording_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """recording:
  root: datasets/demo
  fps: 30
  task: move cube
  queue_size: 8
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "未知字段: queue_size"):
                load_recording_config(path)

    def test_rejects_unknown_image_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """recording:
  root: datasets/demo
  fps: 30
  task: move cube
  image_storage: jpeg
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "image_storage"):
                load_recording_config(path)

    def test_accepts_jpg_image_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """recording:
  root: datasets/demo
  fps: 30
  task: move cube
  image_storage: jpg
  quality: 91
""",
                encoding="utf-8",
            )

            config = load_recording_config(path)

        self.assertEqual(config.image_storage, "jpg")
        self.assertEqual(config.quality, 91)

    def test_rejects_invalid_image_quality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """recording:
  root: datasets/demo
  fps: 30
  task: move cube
  quality: 101
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "recording.quality"):
                load_recording_config(path)

    def test_rejects_invalid_numeric_sample_fps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """recording:
  root: datasets/demo
  fps: 30
  numeric_sample_fps: 0
  task: move cube
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "numeric_sample_fps"):
                load_recording_config(path)

    def test_rejects_duplicate_enabled_camera_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.yaml"
            path.write_text(
                """recording:
  root: datasets/demo
  fps: 30
  task: move cube
  enabled_cameras: [hand, hand]
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "不能包含重复名称"):
                load_recording_config(path)


if __name__ == "__main__":
    unittest.main()
