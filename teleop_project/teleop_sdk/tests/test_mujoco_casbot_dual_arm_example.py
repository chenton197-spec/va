"""CASBOT 双臂 MuJoCo 示例的无图形流程测试。"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from examples import test_mujoco_casbot_dual_arm as example


class _FakeSimulation:
    instances: list["_FakeSimulation"] = []

    def __init__(self, urdf_path: Path) -> None:
        self.urdf_path = urdf_path
        self.open_viewer_calls = 0
        self.sync_viewer_calls = 0
        self.viewer_is_running = False
        self.instances.append(self)

    def open_viewer(self) -> None:
        self.open_viewer_calls += 1

    def sync_viewer(self) -> None:
        self.sync_viewer_calls += 1


class _FakeFollower:
    instances: list["_FakeFollower"] = []

    def __init__(self, simulation: _FakeSimulation, joint_names: tuple[str, ...]) -> None:
        self.simulation = simulation
        self.joint_names = joint_names
        self.connect_calls = 0
        self.start_servo_calls = 0
        self.stop_servo_calls = 0
        self.disconnect_calls = 0
        self.instances.append(self)

    def connect(self) -> None:
        self.connect_calls += 1

    def start_servo(self) -> bool:
        self.start_servo_calls += 1
        return True

    def stop_servo(self) -> None:
        self.stop_servo_calls += 1

    def disconnect(self) -> None:
        self.disconnect_calls += 1


class MujocoCasbotDualArmExampleTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeSimulation.instances = []
        _FakeFollower.instances = []

    def test_example_creates_one_shared_scene_for_left_and_right_arms(self) -> None:
        with patch.object(example, "MujocoSimulation", _FakeSimulation), patch.object(
            example, "MujocoFollower", _FakeFollower
        ):
            example.main()

        self.assertEqual(len(_FakeSimulation.instances), 1)
        self.assertEqual(len(_FakeFollower.instances), 2)
        simulation = _FakeSimulation.instances[0]
        left, right = _FakeFollower.instances
        project_root = Path(example.__file__).resolve().parents[1]
        self.assertEqual(
            simulation.urdf_path,
            project_root / "simulation" / "urdfs" / example.URDF_FILENAME,
        )
        self.assertEqual(left.joint_names, example.LEFT_ARM_JOINT_NAMES)
        self.assertEqual(right.joint_names, example.RIGHT_ARM_JOINT_NAMES)
        self.assertIs(left.simulation, right.simulation)
        self.assertEqual(simulation.open_viewer_calls, 1)
        for follower in (left, right):
            self.assertEqual(follower.connect_calls, 1)
            self.assertEqual(follower.start_servo_calls, 1)
            self.assertEqual(follower.stop_servo_calls, 1)
            self.assertEqual(follower.disconnect_calls, 1)
