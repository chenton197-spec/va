"""Hardware-free checks for the Alicia-D joint-position example."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples import test_alicia_move_to_angles as motion_example


class FakeRobot:
    """Records position-command calls without connecting real hardware."""

    def __init__(self, connected: bool = True, motion_result: bool = True):
        self.connected = connected
        self.motion_result = motion_result
        self.motion_calls: list[dict[str, object]] = []
        self.disconnect_calls = 0

    def is_connected(self) -> bool:
        return self.connected

    def set_robot_state(self, **kwargs: object) -> bool:
        self.motion_calls.append(kwargs)
        return self.motion_result

    def disconnect(self) -> None:
        self.disconnect_calls += 1


class AliciaMoveToAnglesExampleTest(unittest.TestCase):
    """Verify fixed in-file motion settings without the vendor SDK."""

    def setUp(self) -> None:
        self.runtime = SimpleNamespace(
            alicia=SimpleNamespace(port="/dev/ttyACM0", gripper_type="100mm")
        )

    def test_sends_fixed_degree_target_with_runtime_connection_settings(self) -> None:
        robot = FakeRobot()

        with (
            patch.object(motion_example, "load_runtime_config", return_value=self.runtime),
            patch.object(motion_example, "_create_robot", return_value=robot) as create_robot,
            patch("builtins.input", return_value=""),
        ):
            result = motion_example.main()

        self.assertEqual(result, 0)
        create_robot.assert_called_once_with(port="/dev/ttyACM0", gripper_type="100mm")
        self.assertEqual(
            robot.motion_calls,
            [
                {
                    "target_joints": list(motion_example.TARGET_JOINTS_DEG),
                    "joint_format": "deg",
                    "speed_deg_s": motion_example.SPEED_DEG_S,
                    "wait_for_completion": True,
                }
            ],
        )
        self.assertEqual(robot.disconnect_calls, 1)

    def test_refuses_to_move_when_connection_is_not_available(self) -> None:
        robot = FakeRobot(connected=False)

        with (
            patch.object(motion_example, "load_runtime_config", return_value=self.runtime),
            patch.object(motion_example, "_create_robot", return_value=robot),
        ):
            result = motion_example.main()

        self.assertEqual(result, 1)
        self.assertEqual(robot.motion_calls, [])
        self.assertEqual(robot.disconnect_calls, 1)

    def test_reports_failed_motion_and_disconnects(self) -> None:
        robot = FakeRobot(motion_result=False)

        with (
            patch.object(motion_example, "load_runtime_config", return_value=self.runtime),
            patch.object(motion_example, "_create_robot", return_value=robot),
            patch("builtins.input", return_value=""),
        ):
            result = motion_example.main()

        self.assertEqual(result, 1)
        self.assertEqual(len(robot.motion_calls), 1)
        self.assertEqual(robot.disconnect_calls, 1)

    def test_cancels_before_sending_target_and_disconnects(self) -> None:
        robot = FakeRobot()

        with (
            patch.object(motion_example, "load_runtime_config", return_value=self.runtime),
            patch.object(motion_example, "_create_robot", return_value=robot),
            patch("builtins.input", side_effect=KeyboardInterrupt),
        ):
            result = motion_example.main()

        self.assertEqual(result, 130)
        self.assertEqual(robot.motion_calls, [])
        self.assertEqual(robot.disconnect_calls, 1)

    def test_rejects_invalid_target_before_connecting(self) -> None:
        with (
            patch.object(motion_example, "TARGET_JOINTS_DEG", (0.0,) * 5),
            patch.object(motion_example, "_create_robot") as create_robot,
        ):
            result = motion_example.main()

        self.assertEqual(result, 2)
        create_robot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
