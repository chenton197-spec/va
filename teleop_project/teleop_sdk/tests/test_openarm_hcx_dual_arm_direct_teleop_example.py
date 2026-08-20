"""双臂 OpenArm -> HCX limited 遥操作示例的无硬件测试。"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import threading
import unittest
from unittest.mock import patch

import numpy as np

from examples import test_openarm_hcx_dual_arm_direct_teleop as example


class _FakeLeader:
    instances: list["_FakeLeader"] = []

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def __init__(
        self,
        port: str,
        calibration_path: object,
        side: str,
        *,
        baudrate: int,
        read_only: bool,
    ) -> None:
        self.port = port
        self.calibration_path = calibration_path
        self.side = side
        self.baudrate = baudrate
        self.read_only = read_only
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.read_calls = 0
        self.__class__.instances.append(self)

    def connect(self) -> None:
        self.connect_calls += 1

    def disconnect(self) -> None:
        self.disconnect_calls += 1

    def read_joint_angles_deg(self, timeout_s: float) -> np.ndarray:
        del timeout_s
        self.read_calls += 1
        base = 10.0 if self.side == "left" else -10.0
        return np.full(example.JOINT_COUNT, base + self.read_calls)


class _FakeSession:
    def __init__(self, start_kwargs: dict[str, object]) -> None:
        self.start_kwargs = start_kwargs
        self.targets: list[list[float]] = []
        self.stopped = False

    def set_target(self, target: list[float]) -> None:
        self.targets.append(target)

    def stop(self) -> None:
        self.stopped = True


class _FakeArm:
    def __init__(self, robot_id: int) -> None:
        self.robot_id = robot_id
        self.sessions: list[_FakeSession] = []
        self.feedback_reads = 0
        self.joint_limits_deg = np.column_stack(
            (
                np.full(example.JOINT_COUNT, -170.0),
                np.full(example.JOINT_COUNT, 170.0),
            )
        )

    def joint_angles(self) -> np.ndarray:
        self.feedback_reads += 1
        return np.full(example.JOINT_COUNT, float(self.robot_id))

    def start_direct_servo(self, **kwargs: object) -> _FakeSession:
        session = _FakeSession(kwargs)
        self.sessions.append(session)
        return session


class _FakeConnection:
    instances: list["_FakeConnection"] = []

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def __init__(self, config: object) -> None:
        self.config = config
        self.arms: dict[int, _FakeArm] = {}
        self.acquired: list[int] = []
        self.prepared: list[int] = []
        self.released: list[int] = []
        self.__class__.instances.append(self)

    def acquire(self, robot_id: int) -> _FakeArm:
        self.acquired.append(robot_id)
        arm = _FakeArm(robot_id)
        self.arms[robot_id] = arm
        return arm

    def prepare_for_motion(self, robot_id: int) -> bool:
        self.prepared.append(robot_id)
        return True

    def release(self, robot_id: int) -> None:
        self.released.append(robot_id)


class OpenArmHcxDualArmDirectTeleopExampleTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeLeader.reset()
        _FakeConnection.reset()

    def test_each_side_uses_its_own_relative_axis_mapping(self) -> None:
        leader_origin = np.zeros(example.JOINT_COUNT)
        lower = np.full(example.JOINT_COUNT, -100.0)
        upper = np.full(example.JOINT_COUNT, 100.0)
        left = example.map_relative_target(
            np.array((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)),
            leader_origin,
            np.full(example.JOINT_COUNT, 10.0),
            example.LEFT_AXIS_SIGN,
            lower,
            upper,
        )
        right = example.map_relative_target(
            np.array((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)),
            leader_origin,
            np.full(example.JOINT_COUNT, -10.0),
            example.RIGHT_AXIS_SIGN,
            lower,
            upper,
        )

        np.testing.assert_allclose(
            left,
            np.array((9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0)),
        )
        np.testing.assert_allclose(
            right,
            np.array((-11.0, -12.0, -13.0, -6.0, -15.0, -16.0, -17.0)),
        )

    def test_demo_uses_the_hard_coded_one_degree_inset_limits(self) -> None:
        arm = _FakeArm(robot_id=2)
        arm.joint_limits_deg = np.array(
            (
                (-170.0, 170.0),
                (-110.0, 110.0),
                (-170.0, 170.0),
                (-140.0, 55.0),
                (-170.0, 170.0),
                (-55.0, 55.0),
                (-60.0, 60.0),
            )
        )

        _, lower, upper = example._read_follower_pose_and_limits(arm)

        np.testing.assert_array_equal(lower, example.HCX_SAFE_MIN_ANGLES_DEG)
        np.testing.assert_array_equal(upper, example.HCX_SAFE_MAX_ANGLES_DEG)

    def test_demo_refuses_to_send_an_initial_target_at_a_raw_limit(self) -> None:
        arm = _FakeArm(robot_id=2)
        arm.joint_limits_deg = np.array(
            (
                (-170.0, 170.0),
                (-110.0, 110.0),
                (-170.0, 170.0),
                (-140.0, 55.0),
                (-170.0, 170.0),
                (-55.0, 55.0),
                (-60.0, 60.0),
            )
        )

        def at_positive_j6_limit() -> np.ndarray:
            return np.array((0.0, 0.0, 0.0, 0.0, 0.0, 55.0, 0.0))

        arm.joint_angles = at_positive_j6_limit  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "固定安全范围外"):
            example._read_follower_pose_and_limits(arm)

    def test_run_demo_uses_one_connection_and_two_independent_arm_pipelines(self) -> None:
        config = replace(
            example.DEMO_CONFIG,
            test_duration_s=0.04,
            feedback_report_rate_hz=100.0,
        )

        with (
            patch.object(example, "OpenArmMiniLeaderArm", _FakeLeader),
            patch.object(example, "HcxConnection", _FakeConnection),
            redirect_stdout(StringIO()),
        ):
            example.run_demo(config)

        self.assertEqual(len(_FakeConnection.instances), 1)
        connection = _FakeConnection.instances[0]
        self.assertEqual(
            connection.acquired,
            [config.left.hcx_robot_id, config.right.hcx_robot_id],
        )
        self.assertEqual(connection.prepared, connection.acquired)
        self.assertEqual(
            connection.released,
            [config.right.hcx_robot_id, config.left.hcx_robot_id],
        )

        self.assertEqual([leader.side for leader in _FakeLeader.instances], ["left", "right"])
        self.assertEqual(
            [leader.port for leader in _FakeLeader.instances],
            [config.left.openarm_port, config.right.openarm_port],
        )
        for leader in _FakeLeader.instances:
            self.assertEqual(leader.connect_calls, 1)
            self.assertEqual(leader.disconnect_calls, 1)
            self.assertGreater(leader.read_calls, 0)
            self.assertTrue(leader.read_only)

        for robot_id in connection.acquired:
            arm = connection.arms[robot_id]
            self.assertEqual(len(arm.sessions), 1)
            session = arm.sessions[0]
            self.assertEqual(
                session.start_kwargs,
                {
                    "rate_hz": config.direct_servo_rate_hz,
                    "watchdog_s": config.direct_servo_watchdog_s,
                    "confirm_unsafe": config.confirm_direct_servo,
                },
            )
            self.assertGreaterEqual(len(session.targets), 1)
            self.assertTrue(session.stopped)
            self.assertGreater(arm.feedback_reads, 1)

    def test_main_uses_top_level_config_without_yaml_or_cli(self) -> None:
        with patch.object(example, "run_demo") as run_demo:
            result = example.main()

        self.assertEqual(result, 0)
        run_demo.assert_called_once_with(example.DEMO_CONFIG)
        self.assertFalse(hasattr(example, "load_runtime_config"))


if __name__ == "__main__":
    unittest.main()
