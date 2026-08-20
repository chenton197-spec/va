from __future__ import annotations

import json
import tempfile
import time
import unittest
from dataclasses import fields
from pathlib import Path

import numpy as np

from leobot_scripts import DatasetRecorder, RecorderConfig, RecordingFollower
from teleop_sdk.interfaces import FollowerArm


class FakeFollower(FollowerArm):
    def __init__(self) -> None:
        self.position = np.zeros(2)
        self.sent: list[np.ndarray] = []

    @property
    def joint_count(self) -> int:
        return 2

    @property
    def joint_limits_deg(self) -> tuple[np.ndarray, np.ndarray]:
        return np.full(2, -180.0), np.full(2, 180.0)

    def connect(self) -> None:
        return None

    def read_joint_angles_deg(self) -> np.ndarray:
        return self.position.copy()

    def send_joint_angles_deg(self, angles_deg: np.ndarray, command_time_s: float) -> bool:
        self.position = np.asarray(angles_deg, dtype=float).copy()
        self.sent.append(self.position.copy())
        return True

    def start_servo(self) -> bool:
        return True

    def recover(self) -> bool:
        return True

    def stop_servo(self) -> None:
        return None

    def disconnect(self) -> None:
        return None


class DatasetRecorderTest(unittest.TestCase):
    def test_recorder_config_has_no_camera_source_parameters(self) -> None:
        field_names = {item.name for item in fields(RecorderConfig)}
        self.assertNotIn("cameras", field_names)
        self.assertNotIn("depth_cameras", field_names)
        self.assertNotIn("rgbd_cameras", field_names)
        with self.assertRaises(TypeError):
            RecorderConfig(  # type: ignore[call-arg]
                root=Path("dataset"),
                robot_type="fake",
                cameras={},
            )

    def test_records_fixed_rate_robot_episode_without_image_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            inner = FakeFollower()
            recorder = DatasetRecorder(
                RecorderConfig(
                    root=root,
                    robot_type="fake",
                    fps=30,
                    image_storage="png",
                )
            )
            follower = RecordingFollower(inner, recorder)
            self.assertTrue(follower.send_joint_angles_deg(np.array([5.0, -3.0]), 0.01))
            recorder.start_episode("test motion")
            time.sleep(0.08)
            recorder.stop_episode()

            self.assertGreater(len(inner.sent), 0)
            self.assertFalse(recorder.active)
            self.assertFalse((root / "images").exists())
            info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertNotIn("observation.images.hand", info["features"])

    def test_start_requires_an_accepted_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = DatasetRecorder(
                RecorderConfig(root=Path(directory) / "dataset", robot_type="fake", fps=30)
            )
            RecordingFollower(FakeFollower(), recorder)
            with self.assertRaisesRegex(RuntimeError, "accepts at least one"):
                recorder.start_episode("test motion")


if __name__ == "__main__":
    unittest.main()
