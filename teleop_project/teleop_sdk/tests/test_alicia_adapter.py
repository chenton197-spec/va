"""Alicia-D 专用状态读取的无硬件测试。"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from teleop_sdk.adapters.alicia import AliciaLeaderArm


class FakeAliciaRobot:
    """提供固定 joint_gripper 状态的测试替身。"""

    def __init__(self, status: str, gripper: float, angles: list[float] | None = None):
        self.status = status
        self.gripper = gripper
        self.angles = angles

    def get_robot_state(self, info_type: str, timeout: float):
        self.last_info_type = info_type
        self.last_timeout = timeout
        return SimpleNamespace(
            run_status_text=self.status,
            gripper=self.gripper,
            angles=self.angles,
        )


class AliciaLeaderArmTest(unittest.TestCase):
    """验证 Alicia 专用方法的顺序和状态映射。"""

    def test_sync_locked_gripper_order(self) -> None:
        leader = AliciaLeaderArm()
        leader._robot = FakeAliciaRobot("sync_locked", 625.0)

        result = leader.get_sync_lock_gripper(timeout_s=0.2)

        self.assertEqual(result, (True, True, 625.0))
        self.assertEqual(leader._robot.last_info_type, "joint_gripper")
        self.assertEqual(leader._robot.last_timeout, 0.2)

    def test_idle_state_is_not_synced_or_locked(self) -> None:
        leader = AliciaLeaderArm()
        leader._robot = FakeAliciaRobot("idle", 0.0)

        self.assertEqual(leader.get_sync_lock_gripper(), (False, False, 0.0))

    def test_joint_and_normalized_gripper_are_read_together(self) -> None:
        leader = AliciaLeaderArm()
        leader._robot = FakeAliciaRobot(
            "sync", 500, angles=[0.0, 3.141592653589793]
        )

        result = leader.read_joint_angles_and_gripper_opening()

        assert result is not None
        joints, opening = result
        np.testing.assert_allclose(joints, [0.0, 180.0])
        self.assertEqual(opening, 0.5)

    def test_out_of_range_gripper_value_is_not_preclipped(self) -> None:
        leader = AliciaLeaderArm()
        leader._robot = FakeAliciaRobot("sync", 1200, angles=[0.0, 0.0])

        self.assertEqual(leader.read_gripper_opening(0.1), 1.2)


if __name__ == "__main__":
    unittest.main()
