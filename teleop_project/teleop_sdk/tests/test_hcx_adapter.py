"""HCX 双七轴从臂适配器的无硬件行为测试。"""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import numpy as np

import teleop_sdk.adapters.hcx as hcx_adapter
from teleop_sdk.adapters.hcx import (
    HcxConnection,
    HcxConnectionConfig,
    HcxDirectServoConfig,
    HcxFollower,
    HcxMoveJointsConfig,
    HcxStartupConfig,
)
from teleop_sdk.config import HcxConfig


def _wait_until(predicate: object, timeout_s: float = 1.0) -> bool:
    """等待异步 Python 发送线程达到测试条件。"""

    if not callable(predicate):
        raise TypeError("predicate must be callable")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return bool(predicate())


class _FakeDirectServoSession:
    _pulse_lock = threading.Lock()
    _active_pulse_calls = 0
    _max_active_pulse_calls = 0
    pulse_delay_s = 0.0

    @classmethod
    def reset(cls) -> None:
        with cls._pulse_lock:
            cls._active_pulse_calls = 0
            cls._max_active_pulse_calls = 0
            cls.pulse_delay_s = 0.0

    def __init__(self) -> None:
        self.running = True
        self.faulted = False
        self.sent_count = 0
        self.error: str | None = None
        self.fail_set_target = False
        self.return_false = False
        self.set_targets: list[list[float]] = []
        self.set_target_timestamps: list[float] = []
        self.stop_calls = 0
        self.target_submitted = threading.Event()
        self.nonzero_target_submitted = threading.Event()
        self.target_failure = threading.Event()

    @property
    def state(self) -> SimpleNamespace:
        return SimpleNamespace(
            running=self.running,
            faulted=self.faulted,
            sent_count=self.sent_count,
            error=self.error,
        )

    def set_target(self, angles_deg: list[float]) -> bool:
        with self._pulse_lock:
            type(self)._active_pulse_calls += 1
            type(self)._max_active_pulse_calls = max(
                type(self)._max_active_pulse_calls,
                type(self)._active_pulse_calls,
            )
        try:
            if type(self).pulse_delay_s > 0.0:
                time.sleep(type(self).pulse_delay_s)
            if self.fail_set_target:
                self.running = False
                self.faulted = True
                self.error = "PluseToServo returned false; commands stopped"
                self.target_failure.set()
                raise RuntimeError(self.error)
            self.set_targets.append(angles_deg)
            self.set_target_timestamps.append(time.monotonic())
            self.sent_count += 1
            self.target_submitted.set()
            if any(abs(value) > 1e-12 for value in angles_deg):
                self.nonzero_target_submitted.set()
            return not self.return_false
        finally:
            with self._pulse_lock:
                type(self)._active_pulse_calls -= 1

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False


class _FakeArm:
    def __init__(
        self,
        robot_id: int,
        axis_count: int,
        joint_limits_deg: tuple[tuple[float, float], ...],
        events: list[str],
    ) -> None:
        self.robot_id = robot_id
        self._axis_count = axis_count
        self.joint_limits_deg = joint_limits_deg
        self._events = events
        self._angles = np.zeros(axis_count, dtype=float)
        self.enabled = True
        self.enable_calls: list[bool] = []
        self.fail_enable = False
        self.enable_applies = True
        self.fail_move = False
        self.move_calls: list[dict[str, object]] = []
        self.clear_route_calls: list[bool] = []
        self.fail_direct_servo_start = False
        self.direct_servo_start_calls: list[dict[str, object]] = []
        self.direct_servo_sessions: list[_FakeDirectServoSession] = []

    @property
    def axis_count(self) -> int:
        return self._axis_count

    def joint_angles(self) -> tuple[float, ...]:
        return tuple(self._angles)

    def move_joints(self, angles_deg: list[float], **kwargs: object) -> object:
        if self.fail_move:
            raise RuntimeError("controller rejected motion")
        self.move_calls.append({"angles_deg": angles_deg, **kwargs})
        self._angles = np.asarray(angles_deg, dtype=float)
        return object()

    def clear_route(self, *, emergency_stop: bool) -> None:
        self.clear_route_calls.append(emergency_stop)

    def start_direct_servo(
        self, *, rate_hz: int, watchdog_s: float, confirm_unsafe: bool
    ) -> _FakeDirectServoSession:
        if self.fail_direct_servo_start:
            raise RuntimeError("direct-servo start rejected")
        session = _FakeDirectServoSession()
        self.direct_servo_start_calls.append(
            {
                "rate_hz": rate_hz,
                "watchdog_s": watchdog_s,
                "confirm_unsafe": confirm_unsafe,
            }
        )
        self.direct_servo_sessions.append(session)
        return session

    def set_enabled(self, enabled: bool) -> None:
        self._events.append(f"arm:{self.robot_id}:set_enabled:{enabled}")
        self.enable_calls.append(enabled)
        if self.fail_enable:
            raise RuntimeError(f"robot {self.robot_id} enable rejected")
        if self.enable_applies:
            self.enabled = enabled


class _FakeRobotClient:
    instances: list["_FakeRobotClient"] = []
    axis_counts: dict[int, int] = {}
    limits_by_robot: dict[int, tuple[tuple[float, float], ...]] = {}

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.axis_counts = {}
        cls.limits_by_robot = {}
        _FakeDirectServoSession.reset()

    def __init__(self, local_ip: str, remote_ip: str, port: int) -> None:
        self.local_ip = local_ip
        self.remote_ip = remote_ip
        self.port = port
        self.connected = False
        self.connect_timeouts: list[float | None] = []
        self.close_calls = 0
        self.events: list[str] = []
        self.active_alarms: tuple[str, ...] = ()
        self.clear_alarms_calls = 0
        self.clear_alarms_applies = True
        self.soft_emergency_stop_normal = True
        self.hmi_detached = True
        self.detach_hmi_calls = 0
        self.detach_hmi_applies = True
        self.global_enabled = True
        self.global_enable_calls: list[bool] = []
        self.global_enable_applies = True
        self.fail_global_enable = False
        self.ethercat_status = {0: True, 1: True}
        self.ethercat_calls: list[int] = []
        self.arms: dict[int, _FakeArm] = {}
        self.instances.append(self)

    def connect(self, *, timeout_s: float | None = None) -> "_FakeRobotClient":
        self.connect_timeouts.append(timeout_s)
        self.connected = True
        return self

    def close(self) -> None:
        self.close_calls += 1
        self.connected = False

    def detach_hmi(self) -> None:
        self.events.append("detach_hmi")
        self.detach_hmi_calls += 1
        if self.detach_hmi_applies:
            self.hmi_detached = True

    def clear_alarms(self) -> None:
        self.events.append("clear_alarms")
        self.clear_alarms_calls += 1
        if self.clear_alarms_applies:
            self.active_alarms = ()

    def set_global_enable(self, enabled: bool) -> None:
        self.events.append(f"set_global_enable:{enabled}")
        self.global_enable_calls.append(enabled)
        if self.fail_global_enable:
            raise RuntimeError("global enable rejected")
        if self.global_enable_applies:
            self.global_enabled = enabled

    def ethercat_master_operational(self, master_index: int) -> bool:
        self.ethercat_calls.append(master_index)
        return self.ethercat_status[master_index]

    def arm(self, robot_id: int) -> _FakeArm:
        if robot_id not in self.arms:
            axis_count = self.axis_counts.get(robot_id, 7)
            limits = self.limits_by_robot.get(robot_id, ((-180.0, 180.0),) * axis_count)
            self.arms[robot_id] = _FakeArm(robot_id, axis_count, limits, self.events)
        return self.arms[robot_id]


class HcxFollowerTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeRobotClient.reset()
        self._loader = patch.object(
            HcxConnection, "_load_robot_client", return_value=_FakeRobotClient
        )
        self._loader.start()

    def tearDown(self) -> None:
        self._loader.stop()

    @staticmethod
    def _connection(startup: HcxStartupConfig | None = None) -> HcxConnection:
        return HcxConnection(
            HcxConnectionConfig(
                local_ip="192.0.2.10",
                remote_ip="192.0.2.20",
                port=12345,
                startup=startup or HcxStartupConfig(),
            )
        )

    @staticmethod
    def _paired_startup(**overrides: object) -> HcxStartupConfig:
        values: dict[str, object] = {
            "robot_ids": (2, 1),
            "controller_initialization_wait_s": 0.0,
            "ethercat_master_indices": (),
            "ethercat_op_timeout_s": 0.1,
            "alarm_clear_retry_count": 1,
            "alarm_clear_retry_interval_s": 0.01,
            "global_enable_retry_count": 1,
            "global_enable_retry_interval_s": 0.01,
            "single_arm_enable_timeout_s": 0.1,
            "enable_status_poll_interval_s": 0.01,
        }
        values.update(overrides)
        return HcxStartupConfig(**values)

    def test_two_followers_share_one_client_and_submit_nonblocking_commands(self) -> None:
        connection = self._connection()
        motion = HcxMoveJointsConfig(
            acceleration_seconds=0.1,
            deceleration_seconds=0.2,
            speed_ratio=0.3,
            smooth=2,
        )
        left = HcxFollower(connection, robot_id=2, side="left", motion_config=motion)
        right = HcxFollower(connection, robot_id=1, side="right", motion_config=motion)

        left.connect()
        right.connect()

        self.assertEqual(len(_FakeRobotClient.instances), 1)
        client = _FakeRobotClient.instances[0]
        self.assertEqual(client.connect_timeouts, [10.0])
        self.assertTrue(left.start_servo())
        self.assertTrue(right.start_servo())
        self.assertTrue(left.send_joint_angles_deg(np.zeros(7), 0.008))
        self.assertTrue(right.send_joint_angles_deg(np.ones(7), 0.008))

        for arm in (client.arms[2], client.arms[1]):
            self.assertEqual(len(arm.move_calls), 1)
            call = arm.move_calls[0]
            self.assertTrue(call["interrupt"])
            self.assertFalse(call["wait"])
            self.assertEqual(call["acceleration_seconds"], 0.1)
            self.assertEqual(call["deceleration_seconds"], 0.2)
            self.assertEqual(call["speed_ratio"], 0.3)
            self.assertEqual(call["smooth"], 2)

        left.stop_servo()
        left.disconnect()
        self.assertEqual(client.arms[2].clear_route_calls, [True])
        self.assertEqual(client.close_calls, 0)
        right.stop_servo()
        right.disconnect()
        self.assertEqual(client.arms[1].clear_route_calls, [True])
        self.assertEqual(client.close_calls, 1)

    def test_direct_servo_mode_uses_independent_sessions_without_clearing_routes(self) -> None:
        connection = self._connection()
        direct = HcxDirectServoConfig(
            rate_hz=125,
            watchdog_s=0.2,
            confirm_unsafe=True,
        )
        left = HcxFollower(
            connection,
            robot_id=2,
            side="left",
            direct_servo_config=direct,
        )
        right = HcxFollower(
            connection,
            robot_id=1,
            side="right",
            direct_servo_config=direct,
        )

        left.connect()
        right.connect()
        self.assertTrue(left.start_servo())
        self.assertTrue(right.start_servo())
        client = _FakeRobotClient.instances[0]
        left_arm = client.arms[2]
        right_arm = client.arms[1]
        self.assertEqual(
            left_arm.direct_servo_start_calls,
            [{"rate_hz": 125, "watchdog_s": 0.2, "confirm_unsafe": True}],
        )
        self.assertEqual(
            right_arm.direct_servo_start_calls,
            [{"rate_hz": 125, "watchdog_s": 0.2, "confirm_unsafe": True}],
        )

        left_session = left_arm.direct_servo_sessions[0]
        right_session = right_arm.direct_servo_sessions[0]
        try:
            # 上游调用只更新最新 Python 目标；每侧输出线程自行按 rate_hz 下发。
            self.assertTrue(left.send_joint_angles_deg(np.zeros(7), 0.008))
            self.assertTrue(right.send_joint_angles_deg(np.ones(7), 0.008))
            self.assertTrue(right_session.nonzero_target_submitted.wait(timeout=1.0))
            self.assertEqual(left_arm.move_calls, [])
            self.assertEqual(right_arm.move_calls, [])
            self.assertTrue(all(target == [0.0] * 7 for target in left_session.set_targets))
            self.assertIn([1.0] * 7, right_session.set_targets)

            left.stop_servo()
            self.assertFalse(left_session.running)
            self.assertTrue(right_session.running)
            self.assertEqual(left_arm.clear_route_calls, [])
            self.assertEqual(right_arm.clear_route_calls, [])
        finally:
            right.stop_servo()
        self.assertFalse(right_session.running)
        self.assertEqual(right_arm.clear_route_calls, [])

    def test_direct_servo_refresh_checks_python_worker_without_dispatching(self) -> None:
        direct = HcxDirectServoConfig(
            rate_hz=125,
            watchdog_s=0.2,
            confirm_unsafe=True,
        )
        follower = HcxFollower(
            self._connection(),
            robot_id=2,
            side="left",
            direct_servo_config=direct,
        )
        follower.connect()
        self.assertTrue(follower.start_servo())
        arm = _FakeRobotClient.instances[0].arms[2]
        session = arm.direct_servo_sessions[0]

        try:
            initial_count = session.sent_count
            self.assertTrue(follower.refresh_servo_target())
            self.assertTrue(follower.refresh_servo_target())
            self.assertTrue(
                _wait_until(lambda: session.sent_count > initial_count, timeout_s=1.0)
            )
            self.assertTrue(follower.send_joint_angles_deg(np.full(7, 1.0), 0.008))
            self.assertTrue(session.nonzero_target_submitted.wait(timeout=1.0))
            self.assertTrue(follower.refresh_servo_target())
            self.assertEqual(arm.move_calls, [])
        finally:
            follower.stop_servo()

    def test_direct_servo_hot_paths_do_not_poll_session_state(self) -> None:
        """100 Hz 控制路径不能等待 500 Hz 厂商调用持有的会话锁。"""

        direct = HcxDirectServoConfig(
            rate_hz=500,
            watchdog_s=0.2,
            confirm_unsafe=True,
        )
        follower = HcxFollower(
            self._connection(),
            robot_id=2,
            side="left",
            direct_servo_config=direct,
        )
        follower.connect()
        self.assertTrue(follower.start_servo())

        try:
            # 若 send_joint_angles_deg() 或 refresh_servo_target() 重新读取
            # session.state，这个测试会立即暴露高频锁竞争回归。
            with patch.object(
                _FakeDirectServoSession,
                "state",
                new_callable=PropertyMock,
                side_effect=AssertionError("hot path must not read session.state"),
            ):
                self.assertTrue(follower.refresh_servo_target())
                self.assertTrue(
                    follower.send_joint_angles_deg(np.full(7, 1.0), 0.01)
                )
        finally:
            follower.stop_servo()

    def test_direct_servo_output_threads_do_not_share_the_connection_lock(self) -> None:
        direct = HcxDirectServoConfig(
            rate_hz=125,
            watchdog_s=0.2,
            confirm_unsafe=True,
        )
        connection = self._connection()
        left = HcxFollower(
            connection,
            robot_id=2,
            side="left",
            direct_servo_config=direct,
        )
        right = HcxFollower(
            connection,
            robot_id=1,
            side="right",
            direct_servo_config=direct,
        )
        left.connect()
        right.connect()
        self.assertTrue(left.start_servo())
        self.assertTrue(right.start_servo())
        _FakeDirectServoSession.pulse_delay_s = 0.025
        try:
            self.assertTrue(left.send_joint_angles_deg(np.full(7, 1.0), 0.008))
            self.assertTrue(right.send_joint_angles_deg(np.full(7, 2.0), 0.008))
            self.assertTrue(
                _wait_until(
                    lambda: _FakeDirectServoSession._max_active_pulse_calls >= 2,
                    timeout_s=1.0,
                )
            )
        finally:
            _FakeDirectServoSession.pulse_delay_s = 0.0
            left.stop_servo()
            right.stop_servo()

    def test_direct_servo_output_stats_report_slow_dispatch_without_dropping_ticks(self) -> None:
        direct = HcxDirectServoConfig(
            rate_hz=500,
            watchdog_s=0.2,
            confirm_unsafe=True,
        )
        follower = HcxFollower(
            self._connection(),
            robot_id=2,
            side="left",
            direct_servo_config=direct,
        )
        follower.connect()
        self.assertTrue(follower.start_servo())
        session = _FakeRobotClient.instances[0].arms[2].direct_servo_sessions[0]

        _FakeDirectServoSession.pulse_delay_s = 0.006
        try:
            self.assertTrue(
                _wait_until(lambda: session.sent_count >= 4, timeout_s=1.0)
            )
            stats = follower.direct_servo_output_stats()
            self.assertIsNotNone(stats)
            assert stats is not None
            self.assertEqual(stats.configured_rate_hz, 500)
            self.assertGreater(stats.successful_command_count, 0)
            self.assertGreater(stats.recent_successful_command_count, 0)
            # 调用比 2 ms 周期更慢时，线程记录迟到，但不会通过重置截止时间
            # 主动跳过任何计划发送时隙。
            self.assertEqual(stats.recent_missed_tick_count, 0)
            self.assertIsNotNone(stats.max_set_target_duration_s)
            assert stats.max_set_target_duration_s is not None
            self.assertGreaterEqual(stats.max_set_target_duration_s, 0.005)
            self.assertIsNotNone(stats.max_start_lateness_s)
            assert stats.max_start_lateness_s is not None
            self.assertGreater(stats.max_start_lateness_s, 0.0)
            self.assertTrue(stats.running)
        finally:
            _FakeDirectServoSession.pulse_delay_s = 0.0
            follower.stop_servo()

    def test_direct_servo_ignores_vendor_false_result(self) -> None:
        """透传输出持续调用；厂商 bool 不参与 Python 调度或恢复判断。"""

        direct = HcxDirectServoConfig(
            rate_hz=125,
            watchdog_s=0.2,
            confirm_unsafe=True,
        )
        follower = HcxFollower(
            self._connection(),
            robot_id=2,
            side="left",
            direct_servo_config=direct,
        )
        follower.connect()
        self.assertTrue(follower.start_servo())
        session = _FakeRobotClient.instances[0].arms[2].direct_servo_sessions[0]
        session.return_false = True
        initial_count = session.sent_count

        try:
            self.assertTrue(
                _wait_until(
                    lambda: session.sent_count >= initial_count + 3,
                    timeout_s=1.0,
                )
            )
            self.assertTrue(follower.refresh_servo_target())
        finally:
            follower.stop_servo()

    def test_direct_direct_servo_does_not_submit_an_interpolated_trajectory(self) -> None:
        direct = HcxDirectServoConfig(
            rate_hz=500,
            watchdog_s=0.2,
            confirm_unsafe=True,
            interpolation="direct",
        )
        follower = HcxFollower(
            self._connection(),
            robot_id=2,
            side="left",
            direct_servo_config=direct,
        )
        follower.connect()
        self.assertTrue(follower.start_servo())
        session = _FakeRobotClient.instances[0].arms[2].direct_servo_sessions[0]

        try:
            self.assertFalse(follower.requires_per_cycle_target_updates)
            self.assertTrue(follower.send_joint_angles_deg(np.full(7, 10.0), 0.01))
            self.assertTrue(session.nonzero_target_submitted.wait(timeout=1.0))

            # direct 模式没有插值队列，发送点只能是初始保持点或最新目标。
            self.assertTrue(
                all(
                    target in ([0.0] * 7, [10.0] * 7)
                    for target in session.set_targets
                )
            )
            self.assertIn([10.0] * 7, session.set_targets)
        finally:
            follower.stop_servo()

    def test_linear_direct_servo_consumes_a_python_interpolation_queue(self) -> None:
        direct = HcxDirectServoConfig(
            rate_hz=500,
            watchdog_s=0.2,
            confirm_unsafe=True,
            interpolation="linear",
            source_rate_hz=100,
        )
        follower = HcxFollower(
            self._connection(),
            robot_id=2,
            side="left",
            direct_servo_config=direct,
        )
        follower.connect()
        self.assertTrue(follower.start_servo())
        session = _FakeRobotClient.instances[0].arms[2].direct_servo_sessions[0]

        try:
            self.assertTrue(follower.requires_per_cycle_target_updates)
            self.assertTrue(follower.send_joint_angles_deg(np.full(7, 10.0), 0.01))
            self.assertTrue(session.nonzero_target_submitted.wait(timeout=1.0))
            self.assertTrue(
                _wait_until(
                    lambda: [10.0] * 7 in session.set_targets,
                    timeout_s=1.0,
                )
            )

            expected_segment = [[2.0] * 7, [4.0] * 7, [6.0] * 7, [8.0] * 7, [10.0] * 7]
            commands = session.set_targets
            segment_start = next(
                index
                for index in range(len(commands) - len(expected_segment) + 1)
                if commands[index : index + len(expected_segment)] == expected_segment
            )
            self.assertGreaterEqual(segment_start, 1)
            self.assertTrue(follower.refresh_servo_target())
        finally:
            follower.stop_servo()

    def test_linear_direct_servo_observer_receives_submitted_interpolation_points(self) -> None:
        observed: list[tuple[np.ndarray, float]] = []
        observed_lock = threading.Lock()
        target_reached = threading.Event()

        def observe(target: np.ndarray, timestamp: float) -> None:
            with observed_lock:
                observed.append((target, timestamp))
            if np.array_equal(target, np.full(7, 10.0)):
                target_reached.set()

        direct = HcxDirectServoConfig(
            rate_hz=500,
            watchdog_s=0.2,
            confirm_unsafe=True,
            interpolation="linear",
            source_rate_hz=100,
        )
        follower = HcxFollower(
            self._connection(),
            robot_id=2,
            side="left",
            direct_servo_config=direct,
            on_direct_servo_target_submitted=observe,
        )
        follower.connect()
        self.assertTrue(follower.start_servo())
        try:
            with observed_lock:
                observed.clear()
            self.assertTrue(follower.send_joint_angles_deg(np.full(7, 10.0), 0.01))
            self.assertTrue(target_reached.wait(timeout=1.0))
            with observed_lock:
                samples = list(observed)

            expected = [np.full(7, value) for value in (2.0, 4.0, 6.0, 8.0, 10.0)]
            for index in range(len(samples) - len(expected) + 1):
                if all(
                    np.array_equal(samples[index + offset][0], point)
                    for offset, point in enumerate(expected)
                ):
                    timestamps = [
                        samples[index + offset][1] for offset in range(len(expected))
                    ]
                    self.assertTrue(all(isinstance(timestamp, float) for timestamp in timestamps))
                    self.assertTrue(
                        all(
                            later > earlier
                            for earlier, later in zip(timestamps, timestamps[1:])
                        )
                    )
                    break
            else:
                self.fail("did not observe the Python linear interpolation segment")
        finally:
            follower.stop_servo()

    def test_limited_direct_servo_outputs_limited_points_at_configured_rate(self) -> None:
        observed: list[tuple[np.ndarray, float]] = []
        observed_nonzero = threading.Event()

        def observe(target: np.ndarray, timestamp_s: float) -> None:
            observed.append((target, timestamp_s))
            if np.any(np.abs(target) > 1e-12):
                observed_nonzero.set()

        direct = HcxDirectServoConfig(
            rate_hz=800,
            watchdog_s=0.2,
            confirm_unsafe=True,
            interpolation="limited",
            source_rate_hz=100,
            limited_max_velocity_deg_s=20.0,
            limited_max_acceleration_deg_s2=80.0,
            limited_lowpass_alpha=1.0,
        )
        follower = HcxFollower(
            self._connection(),
            robot_id=2,
            side="left",
            direct_servo_config=direct,
            on_direct_servo_target_submitted=observe,
        )
        follower.connect()
        self.assertTrue(follower.start_servo())
        session = _FakeRobotClient.instances[0].arms[2].direct_servo_sessions[0]
        arm = _FakeRobotClient.instances[0].arms[2]
        try:
            self.assertTrue(follower.requires_per_cycle_target_updates)
            self.assertEqual(
                follower._direct_servo_limited_interpolator.source_rate_hz,  # type: ignore[union-attr]
                100,
            )
            self.assertEqual(
                follower._direct_servo_limited_interpolator.output_rate_hz,  # type: ignore[union-attr]
                800,
            )
            self.assertTrue(follower.send_joint_angles_deg(np.full(7, 50.0), 0.01))
            self.assertTrue(session.nonzero_target_submitted.wait(timeout=1.0))
            self.assertTrue(observed_nonzero.wait(timeout=1.0))
            self.assertTrue(follower.refresh_servo_target())
            self.assertTrue(
                _wait_until(lambda: len(session.set_targets) >= 12, timeout_s=1.0)
            )
        finally:
            follower.stop_servo()

        self.assertEqual(
            arm.direct_servo_start_calls,
            [{"rate_hz": 800, "watchdog_s": 0.2, "confirm_unsafe": True}],
        )
        self.assertGreater(len(session.set_targets), 1)
        commands = np.asarray(session.set_targets, dtype=float)
        # 限幅点在 100 Hz 生产侧按标称 800 Hz 周期预生成；500 Hz 线程
        # 只发送这些点或重复最后一点，不会放大单步位移。
        max_step_deg = 20.0 / 800.0
        self.assertLessEqual(np.max(np.abs(np.diff(commands, axis=0))), max_step_deg + 1e-9)
        self.assertTrue(observed)
        self.assertTrue(all(isinstance(timestamp, float) for _, timestamp in observed))
        self.assertFalse(session.running)
        self.assertEqual(session.stop_calls, 1)

    def test_limited_interpolation_is_not_run_by_the_high_rate_sender(self) -> None:
        """500 Hz 路径只能发送生产侧已经生成的批次点。"""

        direct = HcxDirectServoConfig(
            rate_hz=500,
            watchdog_s=0.2,
            confirm_unsafe=True,
            interpolation="limited",
            source_rate_hz=100,
            limited_max_velocity_deg_s=20.0,
            limited_max_acceleration_deg_s2=80.0,
            limited_lowpass_alpha=0.25,
        )
        follower = HcxFollower(
            self._connection(),
            robot_id=2,
            side="left",
            direct_servo_config=direct,
        )
        follower.connect()
        self.assertTrue(follower.start_servo())
        session = _FakeRobotClient.instances[0].arms[2].direct_servo_sessions[0]
        interpolator = follower._direct_servo_limited_interpolator
        assert interpolator is not None

        try:
            with patch.object(
                interpolator, "interpolate", wraps=interpolator.interpolate
            ) as interpolate:
                self.assertTrue(
                    follower.send_joint_angles_deg(np.full(7, 20.0), 0.01)
                )
                self.assertEqual(interpolate.call_count, 1)
                self.assertTrue(session.nonzero_target_submitted.wait(timeout=1.0))
                time.sleep(0.03)
                # 这段时间内发送线程至少执行十余个 500 Hz 时隙；调用次数仍
                # 只能是上游这一次 100 Hz 目标更新。
                self.assertEqual(interpolate.call_count, 1)
        finally:
            follower.stop_servo()

    def test_limited_direct_servo_config_requires_limited_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "limited_max_velocity_deg_s"):
            HcxDirectServoConfig(
                rate_hz=1000,
                interpolation="limited",
                source_rate_hz=100,
            )
        with self.assertRaisesRegex(ValueError, "require interpolation='limited'"):
            HcxDirectServoConfig(
                rate_hz=1000,
                interpolation="linear",
                source_rate_hz=100,
                limited_max_velocity_deg_s=20.0,
            )

    def test_direct_servo_observer_requires_direct_servo_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires direct_servo_config"):
            HcxFollower(
                self._connection(),
                robot_id=2,
                side="left",
                on_direct_servo_target_submitted=lambda _target, _timestamp: None,
            )

    def test_interpolated_direct_servo_config_requires_an_integer_rate_ratio(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_rate_hz"):
            HcxDirectServoConfig(interpolation="linear")
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            HcxDirectServoConfig(
                rate_hz=1000,
                interpolation="linear",
                source_rate_hz=300,
            )
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            HcxDirectServoConfig(
                rate_hz=800,
                interpolation="limited",
                source_rate_hz=300,
                limited_max_velocity_deg_s=20.0,
                limited_max_acceleration_deg_s2=80.0,
                limited_lowpass_alpha=0.25,
            )

    def test_limited_output_failure_is_reported_and_recovery_restarts_worker(self) -> None:
        direct = HcxDirectServoConfig(
            rate_hz=500,
            watchdog_s=0.2,
            confirm_unsafe=True,
            interpolation="limited",
            source_rate_hz=100,
            limited_max_velocity_deg_s=20.0,
            limited_max_acceleration_deg_s2=80.0,
            limited_lowpass_alpha=0.25,
        )
        follower = HcxFollower(
            self._connection(),
            robot_id=2,
            side="left",
            direct_servo_config=direct,
        )
        follower.connect()
        self.assertTrue(follower.start_servo())
        arm = _FakeRobotClient.instances[0].arms[2]
        failed_session = arm.direct_servo_sessions[0]
        failed_session.fail_set_target = True

        try:
            with patch("builtins.print") as print_mock:
                self.assertTrue(follower.send_joint_angles_deg(np.full(7, 10.0), 0.01))
                self.assertTrue(failed_session.target_failure.wait(timeout=1.0))
                self.assertFalse(follower.refresh_servo_target())

            messages = "\n".join(
                str(call.args[0]) for call in print_mock.call_args_list
            )
            self.assertIn("直伺服高频输出失败", messages)
            self.assertIn("PluseToServo returned false", messages)
            self.assertIn("faulted=True", messages)

            self.assertTrue(follower.recover())
            self.assertFalse(failed_session.running)
            self.assertEqual(failed_session.stop_calls, 1)
            self.assertEqual(len(arm.direct_servo_sessions), 2)
            self.assertTrue(arm.direct_servo_sessions[1].running)
        finally:
            follower.stop_servo()

    def test_direct_servo_refresh_reports_native_fault_state(self) -> None:
        direct = HcxDirectServoConfig(
            rate_hz=125,
            watchdog_s=0.2,
            confirm_unsafe=True,
        )
        follower = HcxFollower(
            self._connection(),
            robot_id=2,
            side="left",
            direct_servo_config=direct,
        )
        follower.connect()
        self.assertTrue(follower.start_servo())
        session = _FakeRobotClient.instances[0].arms[2].direct_servo_sessions[0]
        session.fail_set_target = True

        with patch("builtins.print") as print_mock:
            self.assertTrue(session.target_failure.wait(timeout=1.0))
            self.assertFalse(follower.refresh_servo_target())

        messages = "\n".join(str(call.args[0]) for call in print_mock.call_args_list)
        self.assertIn("PluseToServo returned false", messages)
        self.assertIn("faulted=True", messages)
        self.assertIn("sent_count=", messages)
        follower.stop_servo()

    def test_direct_servo_requires_explicit_confirmation_before_startup_actions(self) -> None:
        connection = self._connection(
            self._paired_startup(
                auto_detach_hmi=True,
                auto_clear_alarms=True,
                auto_enable=True,
            )
        )
        follower = HcxFollower(
            connection,
            robot_id=2,
            side="left",
            direct_servo_config=HcxDirectServoConfig(confirm_unsafe=False),
        )
        follower.connect()
        client = _FakeRobotClient.instances[0]
        client.hmi_detached = False
        client.active_alarms = ("startup alarm",)
        client.global_enabled = False
        client.arms[2].enabled = False
        client.arm(1).enabled = False

        self.assertFalse(follower.start_servo())

        self.assertEqual(client.detach_hmi_calls, 0)
        self.assertEqual(client.clear_alarms_calls, 0)
        self.assertEqual(client.global_enable_calls, [])
        self.assertEqual(client.arms[2].enable_calls, [])
        self.assertEqual(client.arm(1).enable_calls, [])
        self.assertEqual(client.arms[2].direct_servo_start_calls, [])

    def test_direct_servo_state_recheck_stops_an_existing_session_when_not_ready(self) -> None:
        direct = HcxDirectServoConfig(
            rate_hz=125,
            watchdog_s=0.2,
            confirm_unsafe=True,
        )
        follower = HcxFollower(
            self._connection(),
            robot_id=2,
            side="left",
            direct_servo_config=direct,
        )
        follower.connect()
        self.assertTrue(follower.start_servo())
        client = _FakeRobotClient.instances[0]
        arm = client.arms[2]
        session = arm.direct_servo_sessions[0]
        client.global_enabled = False

        self.assertFalse(follower.start_servo())

        self.assertFalse(session.running)
        self.assertEqual(session.stop_calls, 1)
        self.assertEqual(arm.clear_route_calls, [])

    def test_direct_servo_recover_recreates_only_the_faulted_side_session(self) -> None:
        direct = HcxDirectServoConfig(
            rate_hz=125,
            watchdog_s=0.2,
            confirm_unsafe=True,
        )
        follower = HcxFollower(
            self._connection(),
            robot_id=2,
            side="left",
            direct_servo_config=direct,
        )
        follower.connect()
        self.assertTrue(follower.start_servo())
        arm = _FakeRobotClient.instances[0].arms[2]
        first_session = arm.direct_servo_sessions[0]
        first_session.fail_set_target = True

        try:
            # 下游调用由异步 Python 线程执行，因此上游目标写入成功后再等待
            # 原生单次调用返回失败。
            self.assertTrue(follower.send_joint_angles_deg(np.zeros(7), 0.008))
            self.assertTrue(first_session.target_failure.wait(timeout=1.0))
            self.assertTrue(follower.recover())

            self.assertFalse(first_session.running)
            self.assertEqual(first_session.stop_calls, 1)
            self.assertEqual(len(arm.direct_servo_sessions), 2)
            self.assertTrue(arm.direct_servo_sessions[1].running)
            self.assertEqual(arm.clear_route_calls, [])
        finally:
            follower.stop_servo()

    def test_direct_servo_config_rejects_invalid_vendor_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "rate_hz"):
            HcxDirectServoConfig(rate_hz=99)
        with self.assertRaisesRegex(ValueError, "watchdog_s"):
            HcxDirectServoConfig(watchdog_s=0.0)
        with self.assertRaisesRegex(ValueError, "confirm_unsafe"):
            HcxDirectServoConfig(confirm_unsafe=1)  # type: ignore[arg-type]

    def test_both_sides_use_the_supplied_degree_joint_limits(self) -> None:
        connection = self._connection()
        left = HcxFollower(connection, robot_id=2, side="left")
        right = HcxFollower(connection, robot_id=1, side="right")

        left_minimum, left_maximum = left.joint_limits_deg
        right_minimum, right_maximum = right.joint_limits_deg
        expected_minimum = np.array(
            [-169.0, -109.0, -169.0, -139.0, -169.0, -54.0, -59.0]
        )
        expected_maximum = np.array(
            [169.0, 109.0, 169.0, 54.0, 169.0, 54.0, 59.0]
        )

        np.testing.assert_array_equal(left_minimum, expected_minimum)
        np.testing.assert_array_equal(left_maximum, expected_maximum)
        np.testing.assert_array_equal(right_minimum, expected_minimum)
        np.testing.assert_array_equal(right_maximum, expected_maximum)

    def test_connect_rejects_wrong_axis_count_and_limits_outside_controller_range(self) -> None:
        _FakeRobotClient.axis_counts[2] = 6
        with self.assertRaisesRegex(RuntimeError, "must expose 7 axes"):
            HcxFollower(self._connection(), robot_id=2, side="left").connect()

        _FakeRobotClient.reset()
        _FakeRobotClient.limits_by_robot[2] = ((-100.0, 100.0),) * 7
        with self.assertRaisesRegex(RuntimeError, "exceed controller joint limits"):
            HcxFollower(self._connection(), robot_id=2, side="left").connect()

    def test_start_servo_only_checks_state_without_changing_controller_state(self) -> None:
        follower = HcxFollower(self._connection(), robot_id=2, side="left")
        follower.connect()
        client = _FakeRobotClient.instances[0]
        arm = client.arms[2]

        client.active_alarms = ("servo alarm",)
        self.assertFalse(follower.start_servo())
        client.active_alarms = ()
        client.global_enabled = False
        self.assertFalse(follower.start_servo())
        client.global_enabled = True
        arm.enabled = False
        self.assertFalse(follower.start_servo())
        arm.enabled = True
        client.soft_emergency_stop_normal = False
        self.assertFalse(follower.start_servo())
        client.soft_emergency_stop_normal = True
        self.assertTrue(follower.start_servo())
        self.assertEqual(arm.move_calls, [])
        self.assertEqual(arm.clear_route_calls, [])

    def test_pair_startup_runs_authorized_steps_once_before_motion(self) -> None:
        startup = self._paired_startup(
            auto_detach_hmi=True,
            auto_clear_alarms=True,
            auto_enable=True,
            ethercat_master_indices=(0,),
        )
        connection = self._connection(startup)
        left = HcxFollower(connection, robot_id=2, side="left")
        left.connect()
        client = _FakeRobotClient.instances[0]
        client.hmi_detached = False
        client.active_alarms = ("startup alarm",)
        client.global_enabled = False
        client.arms[2].enabled = False
        client.arm(1).enabled = False

        self.assertTrue(left.start_servo())

        self.assertEqual(client.detach_hmi_calls, 1)
        self.assertEqual(client.clear_alarms_calls, 1)
        self.assertEqual(client.global_enable_calls, [True])
        self.assertEqual(client.arms[2].enable_calls, [True])
        self.assertEqual(client.arms[1].enable_calls, [True])
        self.assertEqual(
            client.events,
            [
                "detach_hmi",
                "clear_alarms",
                "set_global_enable:True",
                "arm:2:set_enabled:True",
                "arm:1:set_enabled:True",
            ],
        )
        self.assertTrue(client.ethercat_calls)
        self.assertTrue(all(index == 0 for index in client.ethercat_calls))
        self.assertEqual(client.arms[2].move_calls, [])
        self.assertEqual(client.arms[1].move_calls, [])

        right = HcxFollower(connection, robot_id=1, side="right")
        right.connect()
        self.assertTrue(right.start_servo())
        self.assertEqual(client.detach_hmi_calls, 1)
        self.assertEqual(client.clear_alarms_calls, 1)
        self.assertEqual(client.global_enable_calls, [True])
        self.assertEqual(client.arms[2].enable_calls, [True])
        self.assertEqual(client.arms[1].enable_calls, [True])

        left.stop_servo()
        self.assertNotIn(False, client.global_enable_calls)
        self.assertNotIn(False, client.arms[2].enable_calls)
        self.assertNotIn(False, client.arms[1].enable_calls)

    def test_pair_startup_requires_explicit_authorization_before_state_changes(self) -> None:
        connection = self._connection(self._paired_startup())
        left = HcxFollower(connection, robot_id=2, side="left")
        left.connect()
        client = _FakeRobotClient.instances[0]
        client.global_enabled = False
        client.arms[2].enabled = False
        client.arm(1).enabled = False

        self.assertFalse(left.start_servo())

        self.assertEqual(client.detach_hmi_calls, 0)
        self.assertEqual(client.clear_alarms_calls, 0)
        self.assertEqual(client.global_enable_calls, [])
        self.assertEqual(client.arms[2].enable_calls, [])
        self.assertEqual(client.arms[1].enable_calls, [])
        self.assertEqual(client.arms[2].move_calls, [])
        self.assertEqual(client.arms[1].move_calls, [])

    def test_pair_startup_blocks_motion_when_one_arm_enable_fails(self) -> None:
        connection = self._connection(self._paired_startup(auto_enable=True))
        left = HcxFollower(connection, robot_id=2, side="left")
        left.connect()
        client = _FakeRobotClient.instances[0]
        client.global_enabled = False
        client.arms[2].enabled = False
        right_arm = client.arm(1)
        right_arm.enabled = False
        right_arm.fail_enable = True

        self.assertFalse(left.start_servo())
        self.assertFalse(connection.motion_ready(2))
        self.assertEqual(client.global_enable_calls, [True])
        self.assertEqual(client.arms[2].enable_calls, [True])
        self.assertEqual(right_arm.enable_calls, [True])
        self.assertEqual(client.arms[2].move_calls, [])
        self.assertEqual(right_arm.move_calls, [])

    def test_pair_ethercat_timeout_blocks_startup_without_enabling(self) -> None:
        startup = self._paired_startup(
            auto_enable=True,
            ethercat_master_indices=(0,),
            ethercat_op_timeout_s=0.03,
            enable_status_poll_interval_s=0.01,
        )
        connection = self._connection(startup)
        left = HcxFollower(connection, robot_id=2, side="left")
        left.connect()
        client = _FakeRobotClient.instances[0]
        client.ethercat_status[0] = False
        elapsed_s = 0.0

        def monotonic() -> float:
            return elapsed_s

        def sleep(seconds: float) -> None:
            nonlocal elapsed_s
            elapsed_s += seconds

        with (
            patch.object(hcx_adapter.time, "monotonic", side_effect=monotonic),
            patch.object(hcx_adapter.time, "sleep", side_effect=sleep),
        ):
            self.assertFalse(left.start_servo())

        self.assertGreaterEqual(len(client.ethercat_calls), 2)
        self.assertEqual(client.global_enable_calls, [])
        self.assertEqual(client.arms[2].enable_calls, [])
        self.assertEqual(client.arm(1).enable_calls, [])

    def test_recover_only_rechecks_pair_state_without_replaying_setup(self) -> None:
        connection = self._connection(self._paired_startup(auto_enable=True))
        left = HcxFollower(connection, robot_id=2, side="left")
        left.connect()
        client = _FakeRobotClient.instances[0]
        client.global_enabled = False
        client.arms[2].enabled = False
        client.arm(1).enabled = False

        self.assertTrue(left.start_servo())
        events_before_recovery = list(client.events)
        client.global_enabled = False

        self.assertFalse(left.recover())
        self.assertEqual(client.events, events_before_recovery)
        self.assertEqual(client.global_enable_calls, [True])

    def test_connection_config_factory_uses_hcx_runtime_settings(self) -> None:
        runtime = HcxConfig(
            local_ip="192.0.2.10",
            remote_ip="192.0.2.20",
            port=12345,
            connect_timeout_s=3.0,
            left_robot_id=4,
            right_robot_id=5,
            direct_servo_rate_hz=250,
            direct_servo_watchdog_s=0.3,
            direct_servo_confirm_unsafe=True,
            auto_detach_hmi=True,
            auto_clear_alarms=True,
            auto_enable=True,
            controller_initialization_wait_s=1.5,
            ethercat_master_indices=(0, 1),
            ethercat_op_timeout_s=12.0,
            alarm_clear_retry_count=2,
            alarm_clear_retry_interval_s=0.2,
            global_enable_retry_count=3,
            global_enable_retry_interval_s=0.3,
            single_arm_enable_timeout_s=4.0,
            enable_status_poll_interval_s=0.4,
        )

        config = HcxConnectionConfig.from_runtime_config(runtime)

        self.assertEqual(config.local_ip, "192.0.2.10")
        self.assertEqual(config.remote_ip, "192.0.2.20")
        self.assertEqual(config.connect_timeout_s, 3.0)
        self.assertEqual(config.startup.robot_ids, (4, 5))
        self.assertTrue(config.startup.auto_detach_hmi)
        self.assertTrue(config.startup.auto_clear_alarms)
        self.assertTrue(config.startup.auto_enable)
        self.assertEqual(config.startup.ethercat_master_indices, (0, 1))
        self.assertEqual(config.startup.global_enable_retry_count, 3)
        direct = HcxDirectServoConfig.from_runtime_config(runtime)
        self.assertEqual(direct.rate_hz, 250)
        self.assertEqual(direct.watchdog_s, 0.3)
        self.assertTrue(direct.confirm_unsafe)

    def test_invalid_or_rejected_targets_fail_without_motion(self) -> None:
        follower = HcxFollower(self._connection(), robot_id=2, side="left")
        follower.connect()
        self.assertTrue(follower.start_servo())
        arm = _FakeRobotClient.instances[0].arms[2]

        self.assertFalse(follower.send_joint_angles_deg(np.zeros(6), 0.008))
        self.assertFalse(follower.send_joint_angles_deg(np.full(7, np.nan), 0.008))
        self.assertFalse(follower.send_joint_angles_deg(np.full(7, 999.0), 0.008))
        arm.fail_move = True
        self.assertFalse(follower.send_joint_angles_deg(np.zeros(7), 0.008))
        self.assertEqual(arm.move_calls, [])

    def test_recover_only_rechecks_state_and_duplicate_robot_ids_are_rejected(self) -> None:
        connection = self._connection()
        left = HcxFollower(connection, robot_id=2, side="left")
        duplicate = HcxFollower(connection, robot_id=2, side="right")
        left.connect()
        with self.assertRaisesRegex(RuntimeError, "already in use"):
            duplicate.connect()

        client = _FakeRobotClient.instances[0]
        client.active_alarms = ("servo alarm",)
        self.assertFalse(left.recover())
        self.assertEqual(client.arms[2].clear_route_calls, [])
        self.assertEqual(client.arms[2].move_calls, [])
