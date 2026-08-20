"""控制器与滤波器的无硬件测试。"""

from __future__ import annotations

import threading
import unittest

import numpy as np

from teleop_sdk import TeleopConfig, TeleopController
from teleop_sdk.adapters import MockFollower
from teleop_sdk.filters import LowPassFilter, OneEuroFilter
from teleop_sdk.interfaces import GripperActuator, LeaderArm, LeaderArmWithGripper, LeaderGripperInput


class FakeLeader(LeaderArm):
    """按顺序提供角度制关节读数的测试主臂。"""

    def __init__(self, frames: list[np.ndarray]):
        self._frames = [frame.astype(float).copy() for frame in frames]
        self._index = 0
        self.connected = False

    @property
    def joint_count(self) -> int:
        return len(self._frames[0])

    def connect(self) -> None:
        self.connected = True

    def read_joint_angles_deg(self, timeout_s: float) -> np.ndarray | None:
        if not self.connected or not self._frames:
            return None
        frame = self._frames[min(self._index, len(self._frames) - 1)]
        self._index += 1
        return frame.copy()

    def disconnect(self) -> None:
        self.connected = False


class FakeGripperActuator(GripperActuator):
    def __init__(self) -> None:
        self.openings: list[float] = []

    def connect(self) -> None:
        return None

    def send_normalized(self, opening: float) -> bool:
        self.openings.append(opening)
        return True

    def disable(self) -> None:
        return None

    def disconnect(self) -> None:
        return None


class FakeCombinedLeader(LeaderArmWithGripper):
    def __init__(self) -> None:
        self.connected = False
        self.combined_reads = 0

    @property
    def joint_count(self) -> int:
        return 6

    def connect(self) -> None:
        self.connected = True

    def read_joint_angles_deg(self, timeout_s: float) -> np.ndarray | None:
        raise AssertionError("联合读取时不应调用独立关节读取")

    def read_gripper_opening(self, timeout_s: float) -> float | None:
        raise AssertionError("联合读取时不应调用独立夹爪读取")

    def read_joint_angles_and_gripper_opening(self, timeout_s: float):
        self.combined_reads += 1
        return np.zeros(6), 0.5

    def disconnect(self) -> None:
        self.connected = False


class FakeLeaderGripper(LeaderGripperInput):
    def read_gripper_opening(self, timeout_s: float) -> float | None:
        return 0.25


class StopOnServoFollower(MockFollower):
    """在控制器进入伺服后触发外部停止的无硬件从臂。"""

    def __init__(self, stop_event: threading.Event) -> None:
        super().__init__()
        self._stop_event = stop_event
        self.stop_calls = 0
        self.disconnect_calls = 0

    def start_servo(self) -> bool:
        started = super().start_servo()
        self._stop_event.set()
        return started

    def stop_servo(self) -> None:
        self.stop_calls += 1
        super().stop_servo()

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        super().disconnect()


class HeartbeatFollower(MockFollower):
    """记录控制器目标保活调用的无硬件从臂。"""

    def __init__(self) -> None:
        super().__init__()
        self.refresh_calls = 0
        self.refresh_result = True
        self.recover_calls = 0

    def refresh_servo_target(self) -> bool:
        self.refresh_calls += 1
        return self.refresh_result

    def recover(self) -> bool:
        self.recover_calls += 1
        return super().recover()


class ContinuousTargetFollower(MockFollower):
    """声明需要完整低频采样序列的无硬件从臂。"""

    @property
    def requires_per_cycle_target_updates(self) -> bool:
        return True


class FilterTest(unittest.TestCase):
    """验证迁移后滤波器的首次采样行为。"""

    def test_first_sample_is_returned_unchanged(self) -> None:
        sample = np.arange(6, dtype=float)
        self.assertTrue(np.array_equal(OneEuroFilter().step(sample, 1.0), sample))
        self.assertTrue(np.array_equal(LowPassFilter().step(sample, 1.0), sample))


class ControllerTest(unittest.TestCase):
    """验证接口驱动的控制器不需要真实机械臂。"""

    def test_relative_mode_moves_after_baseline(self) -> None:
        leader = FakeLeader(
            [np.full(6, 10.0), np.full(6, 11.0), np.full(6, 11.0)]
        )
        follower = MockFollower()
        config = TeleopConfig(
            filter_enabled=False,
            axis_sign=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )
        controller = TeleopController(leader, follower, config)
        controller.connect()
        controller._follower_ready = follower.start_servo()

        controller.step(1.0)
        controller.step(1.008)
        controller.step(1.016)

        current, _, command_count = follower.get_state()
        self.assertGreater(command_count, 0)
        self.assertTrue(np.all(current > 0.0))

    def test_filter_switch_controls_both_filter_stages(self) -> None:
        leader = FakeLeader([np.zeros(6)])
        follower = MockFollower()

        enabled = TeleopController(leader, follower, TeleopConfig(filter_enabled=True))
        disabled = TeleopController(leader, follower, TeleopConfig(filter_enabled=False))

        self.assertIsNotNone(enabled._leader_filter)
        self.assertIsNotNone(enabled._tremor_filter)
        self.assertIsNone(disabled._leader_filter)
        self.assertIsNone(disabled._tremor_filter)

    def test_relative_limit_holds_until_leader_reenters_mapped_range(self) -> None:
        leader = FakeLeader([np.zeros(6)])
        follower = MockFollower()
        config = TeleopConfig(
            filter_enabled=False,
            min_angles_deg=(-1, -1, -1, -1, -1, -1),
            max_angles_deg=(1, 1, 1, 1, 1, 1),
            axis_sign=(1.0, -1.0, 1.0, -1.0, 1.0, -1.0),
        )
        controller = TeleopController(leader, follower, config)
        controller.connect()
        controller._leader_start = np.zeros(6)

        at_limit = controller._limit(
            controller._map_to_follower(np.array((3.0, -3.0, -3.0, 3.0, 0.0, 0.0)))
        )
        still_outside = controller._limit(
            controller._map_to_follower(np.array((2.0, -2.0, -2.0, 2.0, 0.0, 0.0)))
        )
        reentered = controller._limit(
            controller._map_to_follower(np.array((0.5, -0.5, -0.5, 0.5, 0.0, 0.0)))
        )

        np.testing.assert_array_equal(at_limit, (1.0, 1.0, -1.0, -1.0, 0.0, 0.0))
        np.testing.assert_array_equal(still_outside, at_limit)
        np.testing.assert_array_equal(reentered, (0.5, 0.5, -0.5, -0.5, 0.0, 0.0))
        np.testing.assert_array_equal(controller._leader_start, np.zeros(6))

    def test_mapping_and_limits_match_configuration(self) -> None:
        leader = FakeLeader([np.zeros(6)])
        follower = MockFollower()
        config = TeleopConfig(
            filter_enabled=False,
            relative_mode=False,
            min_angles_deg=(-170, -265, -145, -265, -170, -355),
            max_angles_deg=(170, 85, 145, 85, 170, 355),
            axis_order=(1, 0, 2, 3, 4, 5),
            axis_sign=(-1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )
        controller = TeleopController(leader, follower, config)
        controller.connect()
        mapped = controller._map_to_follower(np.array([20, 30, 0, 0, 0, 0], dtype=float))
        limited = controller._limit(np.array([999, -999, 0, 0, 0, 0], dtype=float))

        self.assertEqual(mapped[0], -30.0)
        self.assertEqual(mapped[1], 20.0)
        self.assertEqual(limited[0], 170.0)
        self.assertEqual(limited[1], -265.0)

    def test_absolute_mode_maps_leader_zero_to_follower_zero(self) -> None:
        leader = FakeLeader([np.zeros(6)])
        follower = MockFollower()
        # 模拟控制器连接前从臂停在非零姿态。
        follower._angles[:] = 25.0
        controller = TeleopController(
            leader,
            follower,
            TeleopConfig(filter_enabled=False, relative_mode=False),
        )
        controller.connect()

        target = controller._map_to_follower(np.zeros(6))

        np.testing.assert_array_equal(target, np.zeros(6))

    def test_different_joint_counts_are_rejected_before_connect(self) -> None:
        leader = FakeLeader([np.zeros(6)])
        follower = MockFollower(n_joints=7)

        with self.assertRaisesRegex(ValueError, "关节数量不一致"):
            TeleopController(leader, follower, TeleopConfig())

    def test_combined_leader_gripper_reading_is_preferred(self) -> None:
        leader = FakeCombinedLeader()
        follower = MockFollower()
        gripper = FakeGripperActuator()
        controller = TeleopController(
            leader, follower, TeleopConfig(filter_enabled=False), gripper=gripper
        )
        controller.connect()
        controller.step(1.0)

        self.assertEqual(leader.combined_reads, 1)
        self.assertEqual(gripper.openings, [0.5])

    def test_independent_leader_gripper_input_is_supported(self) -> None:
        leader = FakeLeader([np.zeros(6)])
        follower = MockFollower()
        gripper = FakeGripperActuator()
        controller = TeleopController(
            leader,
            follower,
            TeleopConfig(filter_enabled=False),
            gripper=gripper,
            leader_gripper=FakeLeaderGripper(),
        )
        controller.connect()
        controller.step(1.0)

        self.assertEqual(gripper.openings, [0.25])

    def test_out_of_range_gripper_input_is_clamped_without_filtering(self) -> None:
        leader = FakeLeader([np.zeros(6)])
        follower = MockFollower()
        gripper = FakeGripperActuator()
        controller = TeleopController(leader, follower, TeleopConfig(), gripper=gripper)

        controller._step_gripper(1.2)
        controller._step_gripper(-0.1)

        # 夹爪不经过任何滤波，异常输入只在执行器边界裁剪。
        self.assertEqual(gripper.openings, [1.0, 0.0])

    def test_start_servo_supports_an_external_control_scheduler(self) -> None:
        leader = FakeLeader([np.zeros(6)])
        follower = MockFollower()
        controller = TeleopController(leader, follower, TeleopConfig(filter_enabled=False))
        controller.connect()

        self.assertTrue(controller.start_servo())
        self.assertTrue(controller.follower_ready)
        self.assertTrue(controller.wait_for_servo_start(0.001))

    def test_disabled_spring_path_sends_mapped_target_without_lag(self) -> None:
        leader = FakeLeader([np.zeros(6), np.full(6, 30.0)])
        follower = MockFollower()
        controller = TeleopController(
            leader,
            follower,
            TeleopConfig(
                filter_enabled=False,
                spring_enabled=False,
            ),
        )
        controller.connect()
        self.assertTrue(controller.start_servo())

        controller.step(1.0)
        controller.step(1.008)

        current, _, command_count = follower.get_state()
        self.assertEqual(command_count, 1)
        np.testing.assert_array_equal(current, np.full(6, 30.0))

    def test_joint_target_observer_receives_only_accepted_targets(self) -> None:
        observed: list[tuple[np.ndarray, float]] = []
        leader = FakeLeader([np.zeros(6), np.full(6, 30.0)])
        follower = MockFollower()
        controller = TeleopController(
            leader,
            follower,
            TeleopConfig(filter_enabled=False, spring_enabled=False),
            on_joint_target_submitted=lambda angles, timestamp: observed.append(
                (angles, timestamp)
            ),
        )
        controller.connect()
        self.assertTrue(controller.start_servo())

        controller.step(1.0)
        controller.step(1.01)

        self.assertEqual(len(observed), 1)
        np.testing.assert_array_equal(observed[0][0], np.full(6, 30.0))
        self.assertIsInstance(observed[0][1], float)

    def test_generated_target_observer_runs_before_dead_zone_suppression(self) -> None:
        observed: list[tuple[np.ndarray, float]] = []
        leader = FakeLeader([np.zeros(6), np.full(6, 30.0)])
        follower = MockFollower()
        controller = TeleopController(
            leader,
            follower,
            TeleopConfig(filter_enabled=False, spring_enabled=False),
            on_joint_target_generated=lambda angles, timestamp: observed.append(
                (angles, timestamp)
            ),
        )
        controller.connect()
        self.assertTrue(controller.start_servo())

        controller.step(1.0)
        controller.step(1.01)

        self.assertEqual(len(observed), 2)
        np.testing.assert_array_equal(observed[0][0], np.zeros(6))
        np.testing.assert_array_equal(observed[1][0], np.full(6, 30.0))
        self.assertIsInstance(observed[0][1], float)

    def test_dead_zone_still_refreshes_the_follower_servo_target(self) -> None:
        leader = FakeLeader([np.zeros(6)])
        follower = HeartbeatFollower()
        controller = TeleopController(
            leader, follower, TeleopConfig(filter_enabled=False)
        )
        controller.connect()
        self.assertTrue(controller.start_servo())

        controller.step(1.0)

        _, _, command_count = follower.get_state()
        self.assertEqual(command_count, 0)
        self.assertEqual(follower.refresh_calls, 2)

    def test_continuous_target_follower_receives_dead_zone_samples(self) -> None:
        leader = FakeLeader([np.zeros(6)])
        follower = ContinuousTargetFollower()
        controller = TeleopController(
            leader,
            follower,
            TeleopConfig(filter_enabled=False, spring_enabled=False),
        )
        controller.connect()
        self.assertTrue(controller.start_servo())

        controller.step(1.0)

        _, _, command_count = follower.get_state()
        self.assertEqual(command_count, 1)

    def test_failed_servo_target_refresh_uses_existing_recovery_path(self) -> None:
        leader = FakeLeader([np.zeros(6)])
        follower = HeartbeatFollower()
        follower.refresh_result = False
        controller = TeleopController(
            leader, follower, TeleopConfig(filter_enabled=False)
        )
        controller.connect()
        self.assertTrue(controller.start_servo())

        controller.step(1.0)

        self.assertEqual(follower.recover_calls, 1)

    def test_stop_event_exits_run_and_runs_normal_cleanup(self) -> None:
        stop_event = threading.Event()
        leader = FakeLeader([np.zeros(6)])
        follower = StopOnServoFollower(stop_event)
        controller = TeleopController(leader, follower, TeleopConfig(filter_enabled=False))
        controller.connect()

        controller.run(stop_event)

        self.assertFalse(leader.connected)
        self.assertGreaterEqual(follower.stop_calls, 1)
        self.assertGreaterEqual(follower.disconnect_calls, 1)

    def test_session_can_defer_cleanup_until_recording_sampler_stops(self) -> None:
        stop_event = threading.Event()
        leader = FakeLeader([np.zeros(6)])
        follower = StopOnServoFollower(stop_event)
        controller = TeleopController(leader, follower, TeleopConfig(filter_enabled=False))
        controller.connect()

        controller.run(stop_event, cleanup_on_exit=False)

        self.assertTrue(leader.connected)
        self.assertEqual(follower.stop_calls, 0)
        self.assertEqual(follower.disconnect_calls, 0)
        controller.shutdown()
        self.assertFalse(leader.connected)
        self.assertEqual(follower.stop_calls, 1)
        self.assertEqual(follower.disconnect_calls, 1)


if __name__ == "__main__":
    unittest.main()
