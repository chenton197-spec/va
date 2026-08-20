"""根目录 OpenArm Mini -> HCX 双臂遥操作入口的无硬件测试。"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import openarm_hcx_dual_arm_teleop as example
from teleop_sdk.config import (
    GloriaMDualGripperConfig,
    GloriaMGripperConfig,
    HcxConfig,
    TeleopConfig,
)


class _FakeConnection:
    instances: list["_FakeConnection"] = []

    def __init__(self, config: object) -> None:
        self.config = config
        type(self).instances.append(self)


class _FakeFollower:
    instances: list["_FakeFollower"] = []

    def __init__(
        self,
        connection: _FakeConnection,
        robot_id: int,
        side: str,
        *,
        direct_servo_config: object,
        on_direct_servo_target_submitted: object | None = None,
    ) -> None:
        self.connection = connection
        self.robot_id = robot_id
        self.side = side
        self.direct_servo_config = direct_servo_config
        self.on_direct_servo_target_submitted = on_direct_servo_target_submitted
        type(self).instances.append(self)

    def read_joint_angles_deg(self) -> np.ndarray:
        return np.full(7, float(self.robot_id))


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
        self.cached_gripper_opening = 0.5
        type(self).instances.append(self)

    def read_cached_gripper_opening(self) -> float:
        return self.cached_gripper_opening


class _FakeController:
    instances: list["_FakeController"] = []
    events: list[str] = []
    start_results = {"left": True, "right": True}

    def __init__(
        self,
        leader: _FakeLeader,
        follower: _FakeFollower,
        config: TeleopConfig,
        gripper: object | None = None,
        leader_gripper: object | None = None,
    ) -> None:
        self.leader = leader
        self.follower = follower
        self.config = config
        self.gripper = gripper
        self.leader_gripper = leader_gripper
        self.connect_calls = 0
        self.start_servo_calls = 0
        self.step_timestamps: list[float] = []
        self.shutdown_calls = 0
        type(self).instances.append(self)

    def connect(self) -> None:
        self.connect_calls += 1
        type(self).events.append(f"{self.leader.side}:connect")

    def start_servo(self) -> bool:
        self.start_servo_calls += 1
        type(self).events.append(f"{self.leader.side}:start")
        return type(self).start_results[self.leader.side]

    def step(self, timestamp: float) -> bool:
        self.step_timestamps.append(timestamp)
        return True

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        type(self).events.append(f"{self.leader.side}:shutdown")


class _LoopController:
    def __init__(self) -> None:
        self.timestamps: list[float] = []

    def step(self, timestamp: float) -> bool:
        self.timestamps.append(timestamp)
        return True


class _ParallelLoopController(_LoopController):
    def __init__(self, barrier: threading.Barrier) -> None:
        super().__init__()
        self._barrier = barrier

    def step(self, timestamp: float) -> bool:
        self.timestamps.append(timestamp)
        self._barrier.wait(timeout=0.5)
        return True


class _FakeFeedbackPoller:
    instances: list["_FakeFeedbackPoller"] = []

    def __init__(self, side: str, follower: _FakeFollower, rate_hz: float) -> None:
        self.side = side
        self.follower = follower
        self.rate_hz = rate_hz
        self.start_calls = 0
        self.stop_calls = 0
        type(self).instances.append(self)

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> bool:
        self.stop_calls += 1
        return True

    def latest(self) -> tuple[np.ndarray, None]:
        return np.full(7, float(self.follower.robot_id)), None


class _FeedbackReader:
    def __init__(self) -> None:
        self.read_event = threading.Event()
        self.thread_names: list[str] = []
        self._lock = threading.Lock()

    def read_joint_angles_deg(self) -> np.ndarray:
        with self._lock:
            self.thread_names.append(threading.current_thread().name)
        self.read_event.set()
        return np.arange(7, dtype=float)


class _FakeFeedbackConsole:
    def __init__(self) -> None:
        self.refresh_calls = 0

    def refresh(self) -> None:
        self.refresh_calls += 1


class _FakeGloriaGripper:
    instances: list["_FakeGloriaGripper"] = []
    failing_ports: set[str] = set()

    def __init__(self, config: GloriaMGripperConfig) -> None:
        self.config = config
        self.connect_calls = 0
        self.disable_calls = 0
        self.disconnect_calls = 0
        type(self).instances.append(self)

    def connect(self) -> None:
        self.connect_calls += 1
        if self.config.port in type(self).failing_ports:
            raise RuntimeError("simulated Gloria-M connection failure")

    def disable(self) -> None:
        self.disable_calls += 1

    def disconnect(self) -> None:
        self.disconnect_calls += 1


class _FakeGloriaWorker:
    instances: list["_FakeGloriaWorker"] = []

    def __init__(
        self,
        side: str,
        leader: _FakeLeader,
        gripper: _FakeGloriaGripper,
        *,
        rate_hz: float,
        status_print_interval_s: float,
    ) -> None:
        self.side = side
        self.leader = leader
        self.gripper = gripper
        self.rate_hz = rate_hz
        self.status_print_interval_s = status_print_interval_s
        self.start_calls = 0
        self.request_stop_calls = 0
        self.close_calls = 0
        type(self).instances.append(self)

    def start(self) -> None:
        self.start_calls += 1

    def request_stop(self) -> None:
        self.request_stop_calls += 1

    def close(self) -> bool:
        self.close_calls += 1
        return True


class _CachedOpeningLeader:
    """只暴露缓存接口；测试中不存在任何 OpenArm 串口读取方法。"""

    def __init__(self, opening: float) -> None:
        self.opening = opening
        self.thread_names: list[str] = []

    def read_cached_gripper_opening(self) -> float:
        self.thread_names.append(threading.current_thread().name)
        return self.opening


class _WorkerGripper:
    def __init__(self) -> None:
        self.targets: list[float] = []
        self.target_event = threading.Event()
        self.disable_calls = 0
        self.disconnect_calls = 0

    def send_normalized(self, target: float) -> bool:
        self.targets.append(target)
        self.target_event.set()
        return True

    def disable(self) -> None:
        self.disable_calls += 1

    def disconnect(self) -> None:
        self.disconnect_calls += 1


class OpenArmHcxDualArmTeleopTest(unittest.TestCase):
    """验证根目录 demo 的双臂装配、独立夹爪边界和关闭路径。"""

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        calibration_path = Path(self._temporary_directory.name) / "openarm_mini.json"
        calibration_path.write_text("{}", encoding="utf-8")
        self.runtime = SimpleNamespace(
            teleop=TeleopConfig(
                rate_hz=100.0,
                axis_sign=(1.0,) * 6,
                filter_enabled=False,
                spring_enabled=True,
            ),
            openarm_mini=SimpleNamespace(
                port_left="/dev/ttyACM1",
                port_right="/dev/ttyACM0",
                calibration_path=str(calibration_path),
                baudrate=1_000_000,
            ),
            hcx=HcxConfig(
                local_ip="192.0.2.10",
                remote_ip="192.0.2.20",
                left_robot_id=2,
                right_robot_id=1,
                left_axis_sign=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0),
                right_axis_sign=(1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0),
                direct_servo_rate_hz=125,
                direct_servo_watchdog_s=0.2,
                direct_servo_confirm_unsafe=True,
            ),
            gloria_m_dual=GloriaMDualGripperConfig(),
        )
        _FakeConnection.instances = []
        _FakeFollower.instances = []
        _FakeLeader.instances = []
        _FakeController.instances = []
        _FakeController.events = []
        _FakeController.start_results = {"left": True, "right": True}
        _FakeFeedbackPoller.instances = []
        _FakeGloriaGripper.instances = []
        _FakeGloriaGripper.failing_ports = set()
        _FakeGloriaWorker.instances = []

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _patched_main_dependencies(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(
            patch.object(example, "load_runtime_config", return_value=self.runtime)
        )
        stack.enter_context(patch.object(example, "HcxConnection", _FakeConnection))
        stack.enter_context(patch.object(example, "HcxFollower", _FakeFollower))
        stack.enter_context(
            patch.object(example, "OpenArmMiniLeaderArm", _FakeLeader)
        )
        stack.enter_context(
            patch.object(example, "TeleopController", _FakeController)
        )
        stack.enter_context(
            patch.object(example, "HcxFeedbackPoller", _FakeFeedbackPoller)
        )
        stack.enter_context(
            patch.object(example, "GloriaMGripperFollower", _FakeGloriaGripper)
        )
        stack.enter_context(
            patch.object(example, "GloriaGripperWorker", _FakeGloriaWorker)
        )
        return stack

    def test_builds_two_read_only_seven_axis_chains_without_grippers(self) -> None:
        with self._patched_main_dependencies() as patched:
            control_loop = patched.enter_context(
                patch.object(example, "_run_control_loop")
            )
            result = example.main()

        self.assertEqual(result, 0)
        self.assertEqual(len(_FakeConnection.instances), 1)
        self.assertEqual(len(_FakeFollower.instances), 2)
        self.assertEqual(len(_FakeLeader.instances), 2)
        self.assertEqual(len(_FakeController.instances), 2)

        connection = _FakeConnection.instances[0]
        left_follower, right_follower = _FakeFollower.instances
        left_leader, right_leader = _FakeLeader.instances
        left_controller, right_controller = _FakeController.instances
        self.assertEqual(connection.config.local_ip, self.runtime.hcx.local_ip)
        self.assertEqual(connection.config.remote_ip, self.runtime.hcx.remote_ip)
        self.assertEqual(connection.config.startup.robot_ids, (2, 1))
        self.assertIs(left_follower.connection, connection)
        self.assertIs(right_follower.connection, connection)
        self.assertEqual((left_follower.robot_id, right_follower.robot_id), (2, 1))
        self.assertEqual((left_follower.side, right_follower.side), ("left", "right"))
        self.assertIs(left_follower.direct_servo_config, right_follower.direct_servo_config)
        self.assertEqual(left_follower.direct_servo_config.rate_hz, 125)
        self.assertEqual(left_follower.direct_servo_config.watchdog_s, 0.2)
        self.assertTrue(left_follower.direct_servo_config.confirm_unsafe)
        self.assertEqual(left_follower.direct_servo_config.interpolation, "direct")

        self.assertEqual((left_leader.side, right_leader.side), ("left", "right"))
        self.assertTrue(left_leader.read_only)
        self.assertTrue(right_leader.read_only)
        self.assertEqual(left_leader.port, self.runtime.openarm_mini.port_left)
        self.assertEqual(right_leader.port, self.runtime.openarm_mini.port_right)

        for controller, axis_sign in (
            (left_controller, self.runtime.hcx.left_axis_sign),
            (right_controller, self.runtime.hcx.right_axis_sign),
        ):
            self.assertEqual(controller.config.axis_order, example.ARM_AXIS_ORDER)
            self.assertEqual(controller.config.axis_sign, axis_sign)
            self.assertTrue(controller.config.relative_mode)
            self.assertFalse(controller.config.filter_enabled)
            self.assertTrue(controller.config.spring_enabled)
            self.assertIsNone(controller.gripper)
            self.assertIsNone(controller.leader_gripper)
            self.assertEqual(controller.connect_calls, 1)
            self.assertEqual(controller.start_servo_calls, 1)
            self.assertEqual(controller.shutdown_calls, 1)

        control_loop.assert_called_once()
        call_args, call_kwargs = control_loop.call_args
        self.assertEqual(call_args, (left_controller, right_controller, 100.0))
        self.assertIsInstance(
            call_kwargs["feedback_console"], example.HcxFeedbackConsole
        )
        self.assertEqual(len(_FakeFeedbackPoller.instances), 2)
        self.assertEqual(
            [(poller.side, poller.rate_hz) for poller in _FakeFeedbackPoller.instances],
            [("left", 30.0), ("right", 30.0)],
        )
        for poller in _FakeFeedbackPoller.instances:
            self.assertEqual(poller.start_calls, 1)
            self.assertEqual(poller.stop_calls, 1)
        self.assertIsNone(left_follower.on_direct_servo_target_submitted)
        self.assertIsNone(right_follower.on_direct_servo_target_submitted)
        self.assertEqual(_FakeGloriaGripper.instances, [])
        self.assertEqual(_FakeGloriaWorker.instances, [])
        self.assertEqual(
            _FakeController.events,
            [
                "left:connect",
                "right:connect",
                "left:start",
                "right:start",
                "right:shutdown",
                "left:shutdown",
            ],
        )

    def test_enabled_gloria_uses_the_existing_matching_openarm_leader(self) -> None:
        self.runtime.gloria_m_dual = GloriaMDualGripperConfig(
            rate_hz=30.0,
            status_print_interval_s=1.0,
            left=GloriaMGripperConfig(enabled=True, port="/dev/ttyACM2"),
            right=GloriaMGripperConfig(enabled=False, port="/dev/ttyACM3"),
        )
        with self._patched_main_dependencies() as patched:
            control_loop = patched.enter_context(
                patch.object(example, "_run_control_loop")
            )
            result = example.main()

        self.assertEqual(result, 0)
        self.assertEqual(len(_FakeGloriaGripper.instances), 1)
        self.assertEqual(len(_FakeGloriaWorker.instances), 1)
        worker = _FakeGloriaWorker.instances[0]
        self.assertEqual(worker.side, "left")
        self.assertIs(worker.leader, _FakeController.instances[0].leader)
        self.assertEqual(worker.gripper.config.port, "/dev/ttyACM2")
        self.assertEqual(worker.rate_hz, 30.0)
        self.assertEqual(worker.start_calls, 1)
        self.assertEqual(worker.request_stop_calls, 1)
        self.assertEqual(worker.close_calls, 1)
        control_loop.assert_called_once()
        self.assertEqual(
            control_loop.call_args.args,
            (_FakeController.instances[0], _FakeController.instances[1], 100.0),
        )

    def test_gloria_connection_failure_does_not_stop_hcx_arm_teleoperation(self) -> None:
        self.runtime.gloria_m_dual = GloriaMDualGripperConfig(
            left=GloriaMGripperConfig(enabled=True, port="/dev/ttyACM2"),
            right=GloriaMGripperConfig(enabled=False, port="/dev/ttyACM3"),
        )
        _FakeGloriaGripper.failing_ports = {"/dev/ttyACM2"}
        with self._patched_main_dependencies() as patched:
            control_loop = patched.enter_context(
                patch.object(example, "_run_control_loop")
            )
            result = example.main()

        self.assertEqual(result, 0)
        self.assertEqual(len(_FakeGloriaGripper.instances), 1)
        self.assertEqual(_FakeGloriaWorker.instances, [])
        control_loop.assert_called_once()
        for controller in _FakeController.instances:
            self.assertEqual(controller.start_servo_calls, 1)

    def test_gloria_worker_uses_only_cached_opening_in_its_own_thread(self) -> None:
        leader = _CachedOpeningLeader(0.25)
        gripper = _WorkerGripper()
        worker = example.GloriaGripperWorker(
            "left",
            leader,  # type: ignore[arg-type]
            gripper,  # type: ignore[arg-type]
            rate_hz=100.0,
            status_print_interval_s=60.0,
        )
        worker.start()
        self.assertTrue(gripper.target_event.wait(timeout=0.5))
        self.assertTrue(worker.close())

        self.assertEqual(gripper.targets[0], 0.25)
        self.assertTrue(
            all(name == "openarm-gloria-left" for name in leader.thread_names)
        )
        self.assertEqual(gripper.disable_calls, 1)
        self.assertEqual(gripper.disconnect_calls, 1)

    def test_start_failure_skips_loop_and_closes_both_chains(self) -> None:
        _FakeController.start_results["left"] = False
        with self._patched_main_dependencies() as patched:
            control_loop = patched.enter_context(
                patch.object(example, "_run_control_loop")
            )
            result = example.main()

        self.assertEqual(result, 1)
        control_loop.assert_not_called()
        self.assertEqual(
            _FakeController.events,
            [
                "left:connect",
                "right:connect",
                "left:start",
                "right:shutdown",
                "left:shutdown",
            ],
        )
        for controller in _FakeController.instances:
            self.assertEqual(controller.shutdown_calls, 1)

    def test_invalid_hcx_pair_does_not_construct_any_device(self) -> None:
        self.runtime.hcx = HcxConfig(
            local_ip="192.0.2.10",
            remote_ip="192.0.2.20",
            left_robot_id=2,
            right_robot_id=2,
        )
        with self._patched_main_dependencies():
            result = example.main()

        self.assertEqual(result, 2)
        self.assertEqual(_FakeConnection.instances, [])
        self.assertEqual(_FakeFollower.instances, [])
        self.assertEqual(_FakeLeader.instances, [])
        self.assertEqual(_FakeController.instances, [])

    def test_invalid_axis_sign_does_not_construct_any_device(self) -> None:
        self.runtime.hcx = HcxConfig(
            local_ip="192.0.2.10",
            remote_ip="192.0.2.20",
            left_robot_id=2,
            right_robot_id=1,
            left_axis_sign=(1.0,) * 6,
        )
        with self._patched_main_dependencies():
            result = example.main()

        self.assertEqual(result, 2)
        self.assertEqual(_FakeConnection.instances, [])
        self.assertEqual(_FakeFollower.instances, [])
        self.assertEqual(_FakeLeader.instances, [])
        self.assertEqual(_FakeController.instances, [])

    def test_missing_direct_servo_confirmation_does_not_construct_any_device(self) -> None:
        self.runtime.hcx = HcxConfig(
            local_ip="192.0.2.10",
            remote_ip="192.0.2.20",
            left_robot_id=2,
            right_robot_id=1,
            direct_servo_confirm_unsafe=False,
        )
        with self._patched_main_dependencies():
            result = example.main()

        self.assertEqual(result, 2)
        self.assertEqual(_FakeConnection.instances, [])
        self.assertEqual(_FakeFollower.instances, [])
        self.assertEqual(_FakeLeader.instances, [])
        self.assertEqual(_FakeController.instances, [])

    def test_linear_direct_servo_uses_the_teleop_rate_as_source_rate(self) -> None:
        self.runtime.hcx = HcxConfig(
            local_ip="192.0.2.10",
            remote_ip="192.0.2.20",
            left_robot_id=2,
            right_robot_id=1,
            direct_servo_rate_hz=1000,
            direct_servo_interpolation="linear",
            direct_servo_confirm_unsafe=True,
        )
        with self._patched_main_dependencies() as patched:
            patched.enter_context(patch.object(example, "_run_control_loop"))
            result = example.main()

        self.assertEqual(result, 0)
        config = _FakeFollower.instances[0].direct_servo_config
        self.assertEqual(config.interpolation, "linear")
        self.assertEqual(config.source_rate_hz, 100)

    def test_limited_direct_servo_uses_the_teleop_rate_as_source_rate(self) -> None:
        self.runtime.hcx = HcxConfig(
            local_ip="192.0.2.10",
            remote_ip="192.0.2.20",
            left_robot_id=2,
            right_robot_id=1,
            direct_servo_rate_hz=800,
            direct_servo_interpolation="limited",
            direct_servo_limited_max_vel_deg_s=20.0,
            direct_servo_limited_max_accel_deg_s2=80.0,
            direct_servo_limited_lowpass_alpha=0.25,
            direct_servo_confirm_unsafe=True,
        )
        with self._patched_main_dependencies() as patched:
            patched.enter_context(patch.object(example, "_run_control_loop"))
            result = example.main()

        self.assertEqual(result, 0)
        config = _FakeFollower.instances[0].direct_servo_config
        self.assertEqual(config.interpolation, "limited")
        self.assertEqual(config.rate_hz, 800)
        self.assertEqual(config.source_rate_hz, 100)
        self.assertEqual(config.limited_max_velocity_deg_s, 20.0)
        self.assertEqual(config.limited_max_acceleration_deg_s2, 80.0)
        self.assertEqual(config.limited_lowpass_alpha, 0.25)

    def test_feedback_poller_reads_in_a_dedicated_thread(self) -> None:
        reader = _FeedbackReader()
        poller = example.HcxFeedbackPoller("left", reader, rate_hz=30.0)
        poller.start()
        self.assertTrue(reader.read_event.wait(timeout=0.5))

        angles_deg, error = poller.latest()
        self.assertTrue(poller.stop())

        np.testing.assert_array_equal(angles_deg, np.arange(7, dtype=float))
        self.assertIsNone(error)
        self.assertTrue(all(name == "hcx-left-feedback" for name in reader.thread_names))

    def test_control_loop_uses_one_timestamp_for_both_sides(self) -> None:
        left = _LoopController()
        right = _LoopController()
        with (
            patch.object(example.time, "perf_counter", side_effect=(0.0, 0.0, 0.0)),
            patch.object(example.time, "sleep", side_effect=KeyboardInterrupt),
        ):
            with self.assertRaises(KeyboardInterrupt):
                example._run_control_loop(left, right, rate_hz=100.0)

        self.assertEqual(left.timestamps, [0.0])
        self.assertEqual(right.timestamps, [0.0])

    def test_control_loop_reads_both_sides_in_parallel(self) -> None:
        barrier = threading.Barrier(2)
        left = _ParallelLoopController(barrier)
        right = _ParallelLoopController(barrier)
        with patch.object(example.time, "sleep", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                example._run_control_loop(left, right, rate_hz=100.0)

        self.assertEqual(len(left.timestamps), 1)
        self.assertEqual(len(right.timestamps), 1)

    def test_control_loop_refreshes_only_the_local_feedback_console(self) -> None:
        left = _LoopController()
        right = _LoopController()
        console = _FakeFeedbackConsole()
        with patch.object(example.time, "sleep", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                example._run_control_loop(
                    left,
                    right,
                    rate_hz=100.0,
                    feedback_console=console,
                )

        self.assertEqual(console.refresh_calls, 1)

if __name__ == "__main__":
    unittest.main()
