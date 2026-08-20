"""OpenArm Mini 标定校验示例的无硬件测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from examples import test_openarm_mini_calibration_verify as verify_example


class FakeLeader:
    """替代只读主臂，提供预设的零位和夹爪读数。"""

    instances: list[FakeLeader] = []
    frames: dict[str, tuple[np.ndarray, float, float] | None] = {}
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
        self.frame_calls = 0
        self.gripper_calls = 0
        self.disconnect_calls = 0
        type(self).instances.append(self)

    def connect(self) -> None:
        self.connect_calls += 1
        error = type(self).connect_errors.get(self.side)
        if error is not None:
            raise error

    def read_joint_angles_and_gripper_opening(self, _timeout_s: float):
        self.frame_calls += 1
        frame = type(self).frames[self.side]
        if frame is None:
            return None
        joint_angles, closed_opening, _open_opening = frame
        return joint_angles, closed_opening

    def read_gripper_opening(self, _timeout_s: float) -> float | None:
        self.gripper_calls += 1
        frame = type(self).frames[self.side]
        return None if frame is None else frame[2]

    def disconnect(self) -> None:
        self.disconnect_calls += 1


class OpenArmMiniCalibrationVerifyExampleTest(unittest.TestCase):
    """验证校验示例的阈值、只读构造和清理行为。"""

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        calibration_path = Path(self._temporary_directory.name) / "openarm_mini.json"
        calibration_path.write_text("{}", encoding="utf-8")
        self.runtime = SimpleNamespace(
            openarm_mini=SimpleNamespace(
                port_left="/dev/ttyACM1",
                port_right="/dev/ttyACM0",
                calibration_path=str(calibration_path),
                baudrate=1_000_000,
            )
        )
        FakeLeader.instances = []
        FakeLeader.connect_errors = {}
        FakeLeader.frames = {
            "left": (np.zeros(7), 0.0, 1.0),
            "right": (np.full(7, 4.9), 0.05, 0.95),
        }

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _run(self) -> int:
        with (
            patch.object(verify_example, "load_runtime_config", return_value=self.runtime),
            patch.object(verify_example, "OpenArmMiniLeaderArm", FakeLeader),
            patch("builtins.input", return_value=""),
        ):
            return verify_example.main()

    def test_verifies_both_sides_with_read_only_leaders(self) -> None:
        result = self._run()

        self.assertEqual(result, 0)
        self.assertEqual(len(FakeLeader.instances), 2)
        left, right = FakeLeader.instances
        self.assertEqual((left.port, left.side), ("/dev/ttyACM1", "left"))
        self.assertEqual((right.port, right.side), ("/dev/ttyACM0", "right"))
        self.assertTrue(left.read_only)
        self.assertTrue(right.read_only)
        self.assertEqual([item.connect_calls for item in FakeLeader.instances], [1, 1])
        self.assertEqual([item.frame_calls for item in FakeLeader.instances], [1, 1])
        self.assertEqual([item.gripper_calls for item in FakeLeader.instances], [1, 1])
        self.assertEqual([item.disconnect_calls for item in FakeLeader.instances], [1, 1])

    def test_reports_failure_after_checking_both_sides(self) -> None:
        FakeLeader.frames["left"] = (np.array([5.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.06, 0.94)

        result = self._run()

        self.assertEqual(result, 1)
        self.assertEqual([item.frame_calls for item in FakeLeader.instances], [1, 1])
        self.assertEqual([item.gripper_calls for item in FakeLeader.instances], [1, 1])
        self.assertEqual([item.disconnect_calls for item in FakeLeader.instances], [1, 1])

    def test_rejects_missing_calibration_file_before_creating_leaders(self) -> None:
        self.runtime.openarm_mini.calibration_path = str(
            Path(self._temporary_directory.name) / "missing.json"
        )

        result = self._run()

        self.assertEqual(result, 2)
        self.assertEqual(FakeLeader.instances, [])

    def test_cancellation_disconnects_created_leaders(self) -> None:
        with (
            patch.object(verify_example, "load_runtime_config", return_value=self.runtime),
            patch.object(verify_example, "OpenArmMiniLeaderArm", FakeLeader),
            patch("builtins.input", side_effect=KeyboardInterrupt),
        ):
            result = verify_example.main()

        self.assertEqual(result, 130)
        self.assertEqual([item.disconnect_calls for item in FakeLeader.instances], [1, 1])


if __name__ == "__main__":
    unittest.main()
