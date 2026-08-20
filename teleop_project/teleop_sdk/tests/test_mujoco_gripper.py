"""MuJoCo 二指夹爪适配器的无图形测试。"""

from __future__ import annotations

import unittest

import numpy as np

from teleop_sdk.adapters.mujoco_follower import MujocoGripper


class _FakeSimulation:
    def __init__(self) -> None:
        self.acquire_calls = 0
        self.release_calls = 0
        self.sync_calls = 0
        self.addresses = np.array([7, 8], dtype=int)
        self.positions: list[tuple[np.ndarray, np.ndarray]] = []

    def prismatic_joint_limits(
        self, joint_names: tuple[str, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        if joint_names != ("finger_a", "finger_b"):
            raise ValueError("unexpected joint names")
        return np.array([-0.005, 0.0]), np.array([0.0, 0.005])

    def acquire(self) -> None:
        self.acquire_calls += 1

    def release(self) -> None:
        self.release_calls += 1

    def qpos_addresses(self, joint_names: tuple[str, ...]) -> np.ndarray:
        if joint_names != ("finger_a", "finger_b"):
            raise ValueError("unexpected joint names")
        return self.addresses.copy()

    def set_joint_positions(
        self, qpos_addresses: np.ndarray, positions: np.ndarray
    ) -> None:
        self.positions.append((qpos_addresses.copy(), positions.copy()))

    def sync_viewer(self) -> None:
        self.sync_calls += 1


class MujocoGripperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.simulation = _FakeSimulation()
        self.gripper = MujocoGripper(
            self.simulation,
            ("finger_a", "finger_b"),
            closed_positions_m=(0.0, 0.0),
            open_positions_m=(-0.005, 0.005),
        )

    def test_normalized_opening_updates_both_fingers_together(self) -> None:
        self.gripper.connect()

        self.assertTrue(self.gripper.send_normalized(0.0))
        self.assertTrue(self.gripper.send_normalized(0.5))
        self.assertTrue(self.gripper.send_normalized(1.0))

        targets = [positions for _, positions in self.simulation.positions]
        np.testing.assert_allclose(targets[0], [0.0, 0.0])
        np.testing.assert_allclose(targets[1], [-0.0025, 0.0025])
        np.testing.assert_allclose(targets[2], [-0.005, 0.005])
        self.assertEqual(self.simulation.sync_calls, 3)

    def test_clamps_opening_and_releases_the_shared_scene(self) -> None:
        self.gripper.connect()

        self.assertTrue(self.gripper.send_normalized(-1.0))
        self.assertTrue(self.gripper.send_normalized(2.0))
        np.testing.assert_allclose(self.simulation.positions[0][1], [0.0, 0.0])
        np.testing.assert_allclose(self.simulation.positions[1][1], [-0.005, 0.005])

        self.gripper.disable()
        self.assertFalse(self.gripper.send_normalized(0.5))
        self.gripper.disconnect()
        self.assertEqual(self.simulation.acquire_calls, 1)
        self.assertEqual(self.simulation.release_calls, 1)


if __name__ == "__main__":
    unittest.main()
