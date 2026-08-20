"""Gloria-M 适配器的无硬件测试。"""

from __future__ import annotations

import unittest
from teleop_sdk.adapters.gloria_m import GloriaMGripperFollower
from teleop_sdk.config import GloriaMGripperConfig


class _Motion:
    def __init__(self) -> None:
        self.torque: float | None = None

    def send_mit(self, **kwargs: float) -> None:
        self.torque = kwargs["tau"]


class _State:
    position = 0.0


class _Gripper:
    state = _State()
    motion = _Motion()


class _FeedbackState:
    def __init__(self, position: float) -> None:
        self.position = position


class _FeedbackMotor:
    def __init__(self) -> None:
        self.poll_count = 0
        self.request_count = 0

    def poll(self) -> None:
        self.poll_count += 1

    def request_state(self) -> None:
        self.request_count += 1


class _FeedbackGripper:
    def __init__(self, position: float) -> None:
        self.state = _FeedbackState(position)
        self.motor = _FeedbackMotor()


class _ContactState:
    def __init__(self, position: float, updated_at: float) -> None:
        self.position = position
        self.updated_at = updated_at


class _ContactGripper:
    def __init__(self, position: float, updated_at: float) -> None:
        self.state = _ContactState(position, updated_at)
        self.motion = _Motion()


class GloriaMGripperFollowerTest(unittest.TestCase):
    def test_normalized_target_is_torque_limited(self) -> None:
        config = GloriaMGripperConfig(open_q_rad=2.5, max_torque_nm=0.75, stiffness_nm_per_rad=6.0)
        follower = GloriaMGripperFollower(config)
        follower._gripper = _Gripper()

        self.assertTrue(follower.send_normalized(1.0))
        self.assertEqual(follower._gripper.motion.torque, 0.75)

    def test_feedback_is_polled_normalized_and_clamped(self) -> None:
        follower = GloriaMGripperFollower(GloriaMGripperConfig(close_q_rad=0.5, open_q_rad=2.5))
        gripper = _FeedbackGripper(position=1.5)
        follower._gripper = gripper

        self.assertEqual(follower.read_normalized_opening(), 0.5)
        gripper.state.position = 3.0
        self.assertEqual(follower.read_normalized_opening(), 1.0)
        gripper.state.position = -1.0
        self.assertEqual(follower.read_normalized_opening(), 0.0)
        self.assertEqual(gripper.motor.poll_count, 3)
        self.assertEqual(gripper.motor.request_count, 3)

    def test_cached_feedback_does_not_poll_or_request_serial_state(self) -> None:
        follower = GloriaMGripperFollower(GloriaMGripperConfig(close_q_rad=0.5, open_q_rad=2.5))
        gripper = _FeedbackGripper(position=1.5)
        follower._gripper = gripper

        self.assertEqual(follower.read_cached_normalized_opening(), 0.5)
        self.assertEqual(gripper.motor.poll_count, 0)
        self.assertEqual(gripper.motor.request_count, 0)

    def test_rejects_identical_open_and_close_calibration(self) -> None:
        with self.assertRaisesRegex(ValueError, "必须不同"):
            GloriaMGripperFollower(GloriaMGripperConfig(open_q_rad=1.0, close_q_rad=1.0))

    def test_contact_switches_to_hold_until_opening_is_requested(self) -> None:
        config = GloriaMGripperConfig(
            contact_torque_nm=0.5,
            contact_stall_duration_s=0.12,
            contact_position_tolerance_rad=0.01,
            hold_torque_nm=0.2,
            contact_release_hysteresis_rad=0.05,
        )
        follower = GloriaMGripperFollower(config)
        gripper = _ContactGripper(position=1.0, updated_at=1.0)
        follower._gripper = gripper

        self.assertTrue(follower.send_normalized(0.0))
        self.assertEqual(gripper.motion.torque, -0.5)
        gripper.state.updated_at = 1.05
        self.assertTrue(follower.send_normalized(0.0))
        self.assertFalse(follower.contact_active)

        # 相同反馈帧不能重复计入停滞时长。
        self.assertTrue(follower.send_normalized(0.0))
        self.assertFalse(follower.contact_active)

        gripper.state.updated_at = 1.18
        self.assertTrue(follower.send_normalized(0.0))
        self.assertTrue(follower.contact_active)
        self.assertEqual(gripper.motion.torque, -0.2)

        # 持续按紧不会重新下发最大闭合扭矩。
        self.assertTrue(follower.send_normalized(0.0))
        self.assertEqual(gripper.motion.torque, -0.2)

        # 明确张开后退出保压并恢复普通目标跟随。
        self.assertTrue(follower.send_normalized(0.5))
        self.assertFalse(follower.contact_active)
        self.assertEqual(gripper.motion.torque, 0.75)

    def test_rejects_hold_torque_above_contact_torque(self) -> None:
        with self.assertRaisesRegex(ValueError, "hold_torque_nm"):
            GloriaMGripperFollower(
                GloriaMGripperConfig(contact_torque_nm=0.3, hold_torque_nm=0.4)
            )

    def test_moving_gripper_does_not_trigger_contact_hold(self) -> None:
        config = GloriaMGripperConfig(
            contact_stall_duration_s=0.12,
            contact_position_tolerance_rad=0.01,
        )
        follower = GloriaMGripperFollower(config)
        gripper = _ContactGripper(position=1.0, updated_at=1.0)
        follower._gripper = gripper

        for position, updated_at in ((0.98, 1.06), (0.96, 1.12), (0.94, 1.18)):
            gripper.state.position = position
            gripper.state.updated_at = updated_at
            self.assertTrue(follower.send_normalized(0.0))

        self.assertFalse(follower.contact_active)
