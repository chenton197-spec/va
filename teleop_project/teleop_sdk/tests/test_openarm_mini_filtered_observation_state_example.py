"""OpenArm Mini 滤波与弹簧阻尼观测示例的无硬件测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from examples import test_openarm_mini_filtered_observation_state as filtered_example
from teleop_sdk.config import TeleopConfig


class FakeLeader:
    """替代只读主臂，返回预设状态且不访问串口。"""

    instances: list[FakeLeader] = []
    frames: dict[str, tuple[np.ndarray, float] | None] = {}
    connect_errors: dict[str, Exception] = {}

    def __init__(
        self,
        *,
        port: str,
        calibration_path: str,
        side: str,
        baudrate: int,
        read_only: bool,
    ):
        self.port = port
        self.calibration_path = calibration_path
        self.side = side
        self.baudrate = baudrate
        self.read_only = read_only
        self.connect_calls = 0
        self.read_calls = 0
        self.disconnect_calls = 0
        type(self).instances.append(self)

    @property
    def joint_count(self) -> int:
        return 7

    def connect(self) -> None:
        self.connect_calls += 1
        error = type(self).connect_errors.get(self.side)
        if error is not None:
            raise error

    def read_joint_angles_and_gripper_opening(self, _timeout_s: float):
        self.read_calls += 1
        return type(self).frames[self.side]

    def disconnect(self) -> None:
        self.disconnect_calls += 1


class OpenArmMiniFilteredObservationStateExampleTest(unittest.TestCase):
    """验证双侧算法状态独立、夹爪旁路和串口清理行为。"""

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        calibration_path = Path(self._temporary_directory.name) / "openarm_mini.json"
        calibration_path.write_text("{}", encoding="utf-8")
        self.runtime = SimpleNamespace(
            teleop=TeleopConfig(rate_hz=1.0),
            openarm_mini=SimpleNamespace(
                port_left="/dev/ttyACM1",
                port_right="/dev/ttyACM0",
                calibration_path=str(calibration_path),
                baudrate=1_000_000,
            ),
        )
        FakeLeader.instances = []
        FakeLeader.connect_errors = {}
        FakeLeader.frames = {
            "left": (np.arange(7, dtype=float), 0.25),
            "right": (np.arange(7, dtype=float) + 10.0, 0.75),
        }

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _run_one_sample(self) -> int:
        with (
            patch.object(filtered_example, "load_runtime_config", return_value=self.runtime),
            patch.object(filtered_example, "OpenArmMiniLeaderArm", FakeLeader),
            patch.object(filtered_example, "READ_RATE_HZ", 1.0),
            patch.object(filtered_example.time, "sleep", side_effect=KeyboardInterrupt),
        ):
            return filtered_example.main()

    def test_pipeline_first_frame_keeps_joint_values_and_gripper_raw(self) -> None:
        pipeline = filtered_example._JointProcessingPipeline(
            2,
            TeleopConfig(rate_hz=20.0),
        )
        frame = filtered_example._process_frame(
            (np.array([10.0, -20.0]), 0.25),
            pipeline,
            1.0,
        )

        assert frame is not None
        np.testing.assert_array_equal(frame.raw_joint_angles, [10.0, -20.0])
        np.testing.assert_array_equal(frame.unwrapped_joint_angles, [10.0, -20.0])
        np.testing.assert_array_equal(frame.one_euro_joint_angles, [10.0, -20.0])
        np.testing.assert_array_equal(frame.low_pass_joint_angles, [10.0, -20.0])
        np.testing.assert_array_equal(frame.spring_joint_angles, [10.0, -20.0])
        self.assertEqual(frame.gripper_opening, 0.25)

    def test_pipeline_smooths_a_second_frame_without_changing_gripper(self) -> None:
        pipeline = filtered_example._JointProcessingPipeline(
            2,
            TeleopConfig(rate_hz=20.0),
        )
        filtered_example._process_frame((np.zeros(2), 0.0), pipeline, 1.0)
        frame = filtered_example._process_frame(
            (np.full(2, 60.0), 0.75),
            pipeline,
            1.05,
        )

        assert frame is not None
        self.assertTrue(np.all(frame.one_euro_joint_angles < frame.raw_joint_angles))
        self.assertTrue(np.all(frame.low_pass_joint_angles < frame.one_euro_joint_angles))
        self.assertTrue(np.all(frame.spring_joint_angles < frame.low_pass_joint_angles))
        self.assertEqual(frame.gripper_opening, 0.75)

    def test_reads_both_sides_with_independent_read_only_pipelines(self) -> None:
        result = self._run_one_sample()

        self.assertEqual(result, 130)
        self.assertEqual(len(FakeLeader.instances), 2)
        left, right = FakeLeader.instances
        self.assertEqual((left.port, left.side), ("/dev/ttyACM1", "left"))
        self.assertEqual((right.port, right.side), ("/dev/ttyACM0", "right"))
        self.assertTrue(left.read_only)
        self.assertTrue(right.read_only)
        self.assertEqual([item.read_calls for item in FakeLeader.instances], [1, 1])
        self.assertEqual([item.disconnect_calls for item in FakeLeader.instances], [1, 1])

    def test_rejects_missing_calibration_file_before_creating_leaders(self) -> None:
        self.runtime.openarm_mini.calibration_path = str(
            Path(self._temporary_directory.name) / "missing.json"
        )

        result = self._run_one_sample()

        self.assertEqual(result, 2)
        self.assertEqual(FakeLeader.instances, [])


if __name__ == "__main__":
    unittest.main()
