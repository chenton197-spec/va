"""FR3 适配器的无硬件行为测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from teleop_sdk.adapters.fairino_fr3 import FairinoFR3Follower


class _Package:
    robot_mode = 0
    robot_state = 1
    main_code = 0
    sub_code = 0
    jt_cur_pos = [0.0] * 6


class _Robot:
    def __init__(self) -> None:
        self.robot_state_pkg = _Package()
        self.mode_calls = 0
        self.queue_calls = 0

    def GetSDKVersion(self):
        return 0, "test"

    def Mode(self, mode: int) -> int:
        self.mode_calls += 1
        return 0

    def RobotEnable(self, enabled: int) -> int:
        return 0

    def GetActualJointPosDegree(self, blocking: int):
        return 0, [0.0] * 6

    def ServoJ(self, *args) -> int:
        return 0

    def GetMotionQueueLength(self):
        self.queue_calls += 1
        return 0, 0


class _RobotClass:
    robot = _Robot()

    @classmethod
    def RPC(cls, ip: str) -> _Robot:
        cls.robot = _Robot()
        return cls.robot


class FairinoFR3FollowerTest(unittest.TestCase):
    def test_connect_skips_mode_when_already_automatic(self) -> None:
        follower = FairinoFR3Follower()
        with patch.object(follower, "_load_robot_class", return_value=_RobotClass), patch(
            "teleop_sdk.adapters.fairino_fr3.time.sleep"
        ):
            follower.connect()

        self.assertEqual(_RobotClass.robot.mode_calls, 0)

    def test_queue_length_is_checked_every_twenty_five_frames(self) -> None:
        follower = FairinoFR3Follower()
        robot = _Robot()
        follower._robot = robot
        for _ in range(50):
            self.assertTrue(follower.send_joint_angles_deg(np.zeros(6), 0.008))

        self.assertEqual(robot.queue_calls, 2)

    def test_cached_state_uses_the_sdk_state_package_without_rpc(self) -> None:
        follower = FairinoFR3Follower()
        robot = _Robot()
        robot.robot_state_pkg.jt_cur_pos = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0]
        follower._robot = robot

        cached = follower.read_cached_joint_angles_deg()

        np.testing.assert_array_equal(cached, [1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
