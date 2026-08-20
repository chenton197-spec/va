"""无硬件验证 OpenArm -> HCX 头部相机驱动采集入口的旁路边界。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

import openarm_hcx_dual_arm_record as example
from leobot_scripts import (
    CameraRecorderHealth,
    CameraSourceHealth,
    MasterFrameRequest,
    MasterFrameSnapshot,
)


class _FakeHcxFollower:
    def __init__(self, values: np.ndarray | None = None) -> None:
        self.values = np.arange(7, dtype=float) if values is None else values
        self.sent: list[np.ndarray] = []
        self.read_count = 0

    @property
    def joint_count(self) -> int:
        return 7

    @property
    def joint_limits_deg(self) -> tuple[np.ndarray, np.ndarray]:
        return np.full(7, -100.0), np.full(7, 100.0)

    @property
    def requires_per_cycle_target_updates(self) -> bool:
        return True

    def send_joint_angles_deg(self, target: np.ndarray, command_time_s: float) -> bool:
        del command_time_s
        self.sent.append(np.asarray(target, dtype=float).copy())
        return True

    def read_joint_angles_deg(self) -> np.ndarray:
        self.read_count += 1
        return self.values.copy()

    def connect(self) -> None:
        return None

    def start_servo(self) -> bool:
        return True

    def refresh_servo_target(self) -> bool:
        return True

    def recover(self) -> bool:
        return True

    def stop_servo(self) -> None:
        return None

    def disconnect(self) -> None:
        return None


class _FakeGripper:
    def __init__(self, opening: float) -> None:
        self.opening = opening
        self.sent: list[float] = []
        self.read_count = 0

    def send_normalized(self, opening: float) -> bool:
        self.sent.append(float(opening))
        return True

    def read_normalized_opening(self) -> float:
        self.read_count += 1
        return self.opening

    def disable(self) -> None:
        return None

    def disconnect(self) -> None:
        return None


class OpenArmHcxDualArmRecordTest(unittest.TestCase):
    def test_any_failed_camera_source_reports_immediately(self) -> None:
        health = CameraRecorderHealth(
            state="recording",
            worker_alive=True,
            active=True,
            last_heartbeat_monotonic_ns=9_900_000_000,
            last_master_capture_monotonic_ns=9_900_000_000,
            error=None,
            source_health={
                "head": CameraSourceHealth(status="streaming"),
                "left_hand": CameraSourceHealth(status="streaming"),
                "right_hand": CameraSourceHealth(
                    status="failed",
                    error="Device disconnected",
                ),
            },
        )

        message = example._camera_source_failure_message(
            health,
            now_ns=10_000_000_000,
        )

        assert message is not None
        self.assertIn("right_hand status=failed", message)
        self.assertIn("Device disconnected", message)
        self.assertIn("head(status=streaming", message)

    def test_stale_camera_message_reports_worker_and_sdk_reason(self) -> None:
        health = CameraRecorderHealth(
            state="recording",
            worker_alive=True,
            active=True,
            last_heartbeat_monotonic_ns=9_800_000_000,
            last_master_capture_monotonic_ns=6_000_000_000,
            error=None,
            source_health={
                "head": CameraSourceHealth(
                    status="failed",
                    latest_capture_monotonic_ns=6_000_000_000,
                    error="No complete frame for 10 consecutive read timeouts",
                ),
                "left_hand": CameraSourceHealth(
                    status="streaming",
                    latest_capture_monotonic_ns=9_900_000_000,
                ),
                "right_hand": CameraSourceHealth(
                    status="streaming",
                    latest_capture_monotonic_ns=9_850_000_000,
                ),
            },
        )

        message = example._camera_stale_message(health, now_ns=10_000_000_000)

        assert message is not None
        self.assertIn("4.000 秒未更新", message)
        self.assertIn("heartbeat-age=0.200s", message)
        self.assertIn("likely-cause=head camera FAILED", message)
        self.assertIn("head(status=failed", message)
        self.assertIn("No complete frame for 10 consecutive read timeouts", message)
        self.assertIn("left_hand(status=streaming", message)

    def test_fresh_camera_has_no_stale_message(self) -> None:
        health = CameraRecorderHealth(
            state="ready",
            worker_alive=True,
            active=False,
            last_heartbeat_monotonic_ns=9_900_000_000,
            last_master_capture_monotonic_ns=9_500_000_000,
            error=None,
        )

        self.assertIsNone(
            example._camera_stale_message(health, now_ns=10_000_000_000)
        )

    def test_upstream_proxies_record_only_successfully_accepted_actions(self) -> None:
        tracker = example._DualActionTracker()
        left_inner = _FakeHcxFollower()
        right_inner = _FakeHcxFollower(np.arange(7, dtype=float) + 10.0)
        left = example._TrackedHcxFollower(left_inner, "left", tracker)
        right = example._TrackedHcxFollower(right_inner, "right", tracker)
        left_gripper = example._TrackedGloriaGripper(_FakeGripper(0.2), "left", tracker)
        right_gripper = example._TrackedGloriaGripper(_FakeGripper(0.8), "right", tracker)

        self.assertTrue(left.send_joint_angles_deg(np.arange(7, dtype=float), 0.01))
        self.assertTrue(right.send_joint_angles_deg(np.arange(7, dtype=float) + 20.0, 0.01))
        self.assertTrue(left_gripper.send_normalized(0.25))
        self.assertTrue(right_gripper.send_normalized(0.75))

        snapshot = tracker.snapshot()
        assert snapshot is not None
        np.testing.assert_array_equal(
            snapshot.action_deg,
            np.concatenate((np.arange(7, dtype=float), np.arange(7, dtype=float) + 20.0)),
        )
        self.assertEqual(snapshot.left_gripper, 0.25)
        self.assertEqual(snapshot.right_gripper, 0.75)

    def test_head_event_reads_each_physical_feedback_source_once(self) -> None:
        session = example.OpenArmHcxRecordingSession(
            SimpleNamespace(),
            SimpleNamespace(),
            (),
            Path("dataset"),
        )
        left_follower = _FakeHcxFollower(np.arange(7, dtype=float) + 1.0)
        right_follower = _FakeHcxFollower(np.arange(7, dtype=float) + 11.0)
        left_gripper = _FakeGripper(0.2)
        right_gripper = _FakeGripper(0.8)
        session._followers = {"left": left_follower, "right": right_follower}  # type: ignore[assignment]
        session._grippers = {"left": left_gripper, "right": right_gripper}  # type: ignore[assignment]
        session._tracker.note_joint_action("left", np.arange(7, dtype=float))
        session._tracker.note_joint_action("right", np.arange(7, dtype=float) + 20.0)
        session._tracker.note_gripper_action("left", 0.3)
        session._tracker.note_gripper_action("right", 0.9)

        result = session._snapshot_for_master_frame(MasterFrameRequest(1, 100, 110))

        self.assertIsInstance(result, MasterFrameSnapshot)
        assert isinstance(result, MasterFrameSnapshot)
        self.assertEqual(left_follower.read_count, 1)
        self.assertEqual(right_follower.read_count, 1)
        self.assertEqual(left_gripper.read_count, 1)
        self.assertEqual(right_gripper.read_count, 1)
        np.testing.assert_array_equal(
            result.state,
            np.concatenate((np.arange(7, dtype=float) + 1.0, np.arange(7, dtype=float) + 11.0)),
        )
        self.assertEqual(result.actuator_states, {"left_gripper": 0.2, "right_gripper": 0.8})

    def test_recording_entrypoint_does_not_start_the_legacy_feedback_poller(self) -> None:
        source = Path(example.__file__).read_text(encoding="utf-8")
        self.assertNotIn("HcxFeedbackPoller", source)
        self.assertNotIn("on_direct_servo_target_submitted", source)


if __name__ == "__main__":
    unittest.main()
