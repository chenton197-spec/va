"""OpenArm Mini 遥操 CASBOT 双臂示例的无硬件、无图形测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples import test_openarm_mini_mujoco_casbot_dual_arm as example
from teleop_sdk.config import TeleopConfig


class _FakeSimulation:
    instances: list["_FakeSimulation"] = []

    def __init__(self, urdf_path: Path) -> None:
        self.urdf_path = urdf_path
        self.checkerboard_floor_calls: list[dict[str, float | int]] = []
        self.open_viewer_calls = 0
        self.sync_viewer_calls = 0
        self._viewer_checks = 0
        type(self).instances.append(self)

    def set_mjcf_environment(self, **kwargs: float | bool) -> None:
        self.checkerboard_floor_calls.append(kwargs)

    @property
    def viewer_is_running(self) -> bool:
        self._viewer_checks += 1
        return self._viewer_checks == 1

    def open_viewer(self) -> None:
        self.open_viewer_calls += 1

    def sync_viewer(self) -> None:
        self.sync_viewer_calls += 1


class _FakeFollower:
    instances: list["_FakeFollower"] = []

    def __init__(self, simulation: _FakeSimulation, joint_names: tuple[str, ...]) -> None:
        self.simulation = simulation
        self.joint_names = joint_names
        type(self).instances.append(self)


class _FakeGripper:
    instances: list["_FakeGripper"] = []

    def __init__(
        self,
        simulation: _FakeSimulation,
        joint_names: tuple[str, ...],
        closed_positions_m: tuple[float, ...],
        open_positions_m: tuple[float, ...],
    ) -> None:
        self.simulation = simulation
        self.joint_names = joint_names
        self.closed_positions_m = closed_positions_m
        self.open_positions_m = open_positions_m
        type(self).instances.append(self)


class _FakeLeader:
    instances: list["_FakeLeader"] = []

    def __init__(
        self,
        *,
        port: str,
        calibration_path: str,
        side: str,
        baudrate: int,
        read_only: bool,
    ) -> None:
        self.port = port
        self.calibration_path = calibration_path
        self.side = side
        self.baudrate = baudrate
        self.read_only = read_only
        type(self).instances.append(self)


class _FakeController:
    instances: list["_FakeController"] = []

    def __init__(
        self,
        leader: _FakeLeader,
        follower: _FakeFollower,
        config: TeleopConfig,
        gripper: _FakeGripper | None = None,
    ) -> None:
        self.leader = leader
        self.follower = follower
        self.config = config
        self.gripper = gripper
        self.connect_calls = 0
        self.start_servo_calls = 0
        self.step_timestamps: list[float] = []
        self.shutdown_calls = 0
        type(self).instances.append(self)

    def connect(self) -> None:
        self.connect_calls += 1

    def start_servo(self) -> bool:
        self.start_servo_calls += 1
        return True

    def step(self, timestamp: float) -> bool:
        self.step_timestamps.append(timestamp)
        return True

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class OpenArmMiniMujocoCasbotDualArmExampleTest(unittest.TestCase):
    """验证双臂与二指夹爪示例的装配和主循环生命周期。"""

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.calibration_path = Path(self._temporary_directory.name) / "openarm_mini.json"
        self.calibration_path.write_text("{}", encoding="utf-8")
        self.runtime = SimpleNamespace(
            # 故意给出 Alicia-FR3 的六轴配置，示例必须覆盖为自己的七轴映射。
            teleop=TeleopConfig(rate_hz=125.0, axis_sign=(1.0,) * 6),
            openarm_mini=SimpleNamespace(
                port_left="/dev/ttyACM1",
                port_right="/dev/ttyACM0",
                calibration_path=str(self.calibration_path),
                baudrate=1_000_000,
            ),
        )
        _FakeSimulation.instances = []
        _FakeFollower.instances = []
        _FakeGripper.instances = []
        _FakeLeader.instances = []
        _FakeController.instances = []

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def test_example_builds_two_relative_seven_axis_control_chains(self) -> None:
        with (
            patch.object(example, "load_runtime_config", return_value=self.runtime),
            patch.object(example, "MujocoSimulation", _FakeSimulation),
            patch.object(example, "MujocoFollower", _FakeFollower),
            patch.object(example, "MujocoGripper", _FakeGripper),
            patch.object(example, "OpenArmMiniLeaderArm", _FakeLeader),
            patch.object(example, "TeleopController", _FakeController),
            patch.object(example.time, "sleep"),
        ):
            result = example.main()

        self.assertEqual(result, 0)
        self.assertEqual(len(_FakeSimulation.instances), 1)
        self.assertEqual(len(_FakeFollower.instances), 2)
        self.assertEqual(len(_FakeGripper.instances), 2)
        self.assertEqual(len(_FakeLeader.instances), 2)
        self.assertEqual(len(_FakeController.instances), 2)

        simulation = _FakeSimulation.instances[0]
        left_follower, right_follower = _FakeFollower.instances
        left_gripper, right_gripper = _FakeGripper.instances
        left_leader, right_leader = _FakeLeader.instances
        left_controller, right_controller = _FakeController.instances
        project_root = Path(example.__file__).resolve().parents[1]

        self.assertEqual(
            simulation.urdf_path,
            project_root / "simulation" / "urdfs" / example.URDF_FILENAME,
        )
        self.assertEqual(
            simulation.checkerboard_floor_calls,
            [
                {
                    "floor_z_m": example.CHECKERBOARD_FLOOR_Z_M,
                    "align_model_lowest_point_to_floor": True,
                }
            ],
        )
        self.assertEqual(left_follower.joint_names, example.LEFT_ARM_JOINT_NAMES)
        self.assertEqual(right_follower.joint_names, example.RIGHT_ARM_JOINT_NAMES)
        self.assertIs(left_follower.simulation, right_follower.simulation)
        self.assertEqual(left_gripper.joint_names, example.LEFT_GRIPPER_JOINT_NAMES)
        self.assertEqual(right_gripper.joint_names, example.RIGHT_GRIPPER_JOINT_NAMES)
        self.assertEqual(
            left_gripper.closed_positions_m, example.GRIPPER_CLOSED_POSITIONS_M
        )
        self.assertEqual(
            right_gripper.open_positions_m, example.GRIPPER_OPEN_POSITIONS_M
        )
        self.assertIs(left_gripper.simulation, simulation)
        self.assertIs(right_gripper.simulation, simulation)
        self.assertEqual(left_leader.port, self.runtime.openarm_mini.port_left)
        self.assertEqual(right_leader.port, self.runtime.openarm_mini.port_right)
        self.assertEqual(left_leader.calibration_path, str(self.calibration_path))
        self.assertEqual(right_leader.calibration_path, str(self.calibration_path))
        self.assertEqual((left_leader.side, right_leader.side), ("left", "right"))
        self.assertTrue(left_leader.read_only)
        self.assertTrue(right_leader.read_only)

        self.assertEqual(left_controller.config.axis_order, example.ARM_AXIS_ORDER)
        self.assertEqual(right_controller.config.axis_order, example.ARM_AXIS_ORDER)
        self.assertEqual(left_controller.config.axis_sign, example.LEFT_AXIS_SIGN)
        self.assertEqual(right_controller.config.axis_sign, example.RIGHT_AXIS_SIGN)
        self.assertEqual(example.ARM_CONTROL_MODE, "absolute")
        self.assertFalse(left_controller.config.relative_mode)
        self.assertFalse(right_controller.config.relative_mode)
        self.assertTrue(left_controller.config.filter_enabled)
        self.assertTrue(right_controller.config.filter_enabled)
        self.assertEqual(
            left_controller.config.spring_enabled, example.ENABLE_FILTER_AND_SPRING
        )
        self.assertEqual(
            right_controller.config.spring_enabled, example.ENABLE_FILTER_AND_SPRING
        )
        self.assertIs(left_controller.gripper, left_gripper)
        self.assertIs(right_controller.gripper, right_gripper)
        self.assertEqual(simulation.open_viewer_calls, 1)
        self.assertEqual(simulation.sync_viewer_calls, 1)

        for controller in (left_controller, right_controller):
            self.assertEqual(controller.connect_calls, 1)
            self.assertEqual(controller.start_servo_calls, 1)
            self.assertEqual(len(controller.step_timestamps), 1)
            self.assertEqual(controller.shutdown_calls, 1)
        self.assertEqual(
            left_controller.step_timestamps[0], right_controller.step_timestamps[0]
        )

    def test_arm_control_mode_can_switch_to_relative(self) -> None:
        with patch.object(example, "ARM_CONTROL_MODE", "relative"):
            config = example._control_config_for_side(
                self.runtime.teleop, example.LEFT_AXIS_SIGN
            )

        self.assertTrue(config.relative_mode)

    def test_switch_keeps_filters_and_disables_the_spring_group(self) -> None:
        with patch.object(example, "ENABLE_FILTER_AND_SPRING", False):
            config = example._control_config_for_side(
                self.runtime.teleop,
                example.LEFT_AXIS_SIGN,
            )

        self.assertTrue(config.filter_enabled)
        self.assertFalse(config.spring_enabled)

    def test_switch_enables_the_complete_spring_group(self) -> None:
        with patch.object(example, "ENABLE_FILTER_AND_SPRING", True):
            enabled_config = example._control_config_for_side(
                self.runtime.teleop,
                example.LEFT_AXIS_SIGN,
            )

        self.assertTrue(enabled_config.filter_enabled)
        self.assertTrue(enabled_config.spring_enabled)

    def test_checkerboard_floor_can_be_disabled(self) -> None:
        with (
            patch.object(example, "load_runtime_config", return_value=self.runtime),
            patch.object(example, "MujocoSimulation", _FakeSimulation),
            patch.object(example, "MujocoFollower", _FakeFollower),
            patch.object(example, "MujocoGripper", _FakeGripper),
            patch.object(example, "OpenArmMiniLeaderArm", _FakeLeader),
            patch.object(example, "TeleopController", _FakeController),
            patch.object(example, "ENABLE_CHECKERBOARD_FLOOR", False),
            patch.object(example.time, "sleep"),
        ):
            result = example.main()

        self.assertEqual(result, 0)
        self.assertEqual(len(_FakeSimulation.instances), 1)
        self.assertEqual(_FakeSimulation.instances[0].checkerboard_floor_calls, [])


if __name__ == "__main__":
    unittest.main()
