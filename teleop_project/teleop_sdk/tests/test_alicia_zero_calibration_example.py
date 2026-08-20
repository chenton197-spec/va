"""Hardware-free checks for the Alicia-D zero-calibration example."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples import test_alicia_zero_calibration as calibration_example


class FakeRobot:
    """Records the example's SDK lifecycle calls without hardware access."""

    def __init__(self, connected: bool = True, calibration_result: bool = True):
        self.connected = connected
        self.calibration_result = calibration_result
        self.calibration_calls = 0
        self.disconnect_calls = 0

    def is_connected(self) -> bool:
        return self.connected

    def zero_calibration(self) -> bool:
        self.calibration_calls += 1
        return self.calibration_result

    def disconnect(self) -> None:
        self.disconnect_calls += 1


class AliciaZeroCalibrationExampleTest(unittest.TestCase):
    """Verify the irreversible calibration example without importing the vendor SDK."""

    def setUp(self) -> None:
        self.runtime = SimpleNamespace(
            alicia=SimpleNamespace(port="/dev/ttyACM0", gripper_type="100mm")
        )

    def test_runs_sdk_calibration_with_runtime_connection_settings(self) -> None:
        robot = FakeRobot()

        with (
            patch.object(calibration_example, "load_runtime_config", return_value=self.runtime),
            patch.object(calibration_example, "_create_robot", return_value=robot) as create_robot,
        ):
            result = calibration_example.main()

        self.assertEqual(result, 0)
        create_robot.assert_called_once_with(port="/dev/ttyACM0", gripper_type="100mm")
        self.assertEqual(robot.calibration_calls, 1)
        self.assertEqual(robot.disconnect_calls, 1)

    def test_refuses_to_calibrate_when_connection_is_not_available(self) -> None:
        robot = FakeRobot(connected=False)

        with (
            patch.object(calibration_example, "load_runtime_config", return_value=self.runtime),
            patch.object(calibration_example, "_create_robot", return_value=robot),
        ):
            result = calibration_example.main()

        self.assertEqual(result, 1)
        self.assertEqual(robot.calibration_calls, 0)
        self.assertEqual(robot.disconnect_calls, 1)

    def test_reports_failed_calibration_and_disconnects(self) -> None:
        robot = FakeRobot(calibration_result=False)

        with (
            patch.object(calibration_example, "load_runtime_config", return_value=self.runtime),
            patch.object(calibration_example, "_create_robot", return_value=robot),
        ):
            result = calibration_example.main()

        self.assertEqual(result, 1)
        self.assertEqual(robot.calibration_calls, 1)
        self.assertEqual(robot.disconnect_calls, 1)

    def test_handles_keyboard_interrupt_and_disconnects(self) -> None:
        robot = FakeRobot()

        with (
            patch.object(calibration_example, "load_runtime_config", return_value=self.runtime),
            patch.object(
                calibration_example,
                "_create_robot",
                return_value=robot,
            ),
            patch.object(robot, "zero_calibration", side_effect=KeyboardInterrupt),
        ):
            result = calibration_example.main()

        self.assertEqual(result, 130)
        self.assertEqual(robot.disconnect_calls, 1)


if __name__ == "__main__":
    unittest.main()
