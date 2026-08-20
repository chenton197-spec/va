"""OpenArm Mini -> Gloria-M 双侧夹爪示例的无硬件测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

import examples.test_openarm_gloria_m_dual_gripper_teleop as example
from teleop_sdk.config import GloriaMDualGripperConfig, GloriaMGripperConfig


class _FakeLeader:
    def __init__(self, opening: object) -> None:
        self.opening = opening
        self.timeouts: list[float] = []

    def read_gripper_opening(self, timeout_s: float) -> float | None:
        self.timeouts.append(timeout_s)
        return self.opening  # type: ignore[return-value]


class _FakeGripper:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.targets: list[float] = []

    def send_normalized(self, opening: float) -> bool:
        self.targets.append(opening)
        return self.result


class _FakeWorker:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def join(self) -> None:
        self._events.append("worker.join")


class _FakeLifecycleDevice:
    def __init__(self, name: str, events: list[str]) -> None:
        self._name = name
        self._events = events

    def disable(self) -> None:
        self._events.append(f"{self._name}.disable")

    def disconnect(self) -> None:
        self._events.append(f"{self._name}.disconnect")


class OpenArmGloriaMDualGripperTeleopExampleTest(unittest.TestCase):
    def test_each_openarm_gripper_only_sends_to_its_matching_gloria_gripper(self) -> None:
        left_leader = _FakeLeader(0.25)
        right_leader = _FakeLeader(0.75)
        left_gripper = _FakeGripper()
        right_gripper = _FakeGripper()

        left_target, left_sent = example._send_gripper_target(
            left_leader, left_gripper, 0.05
        )
        right_target, right_sent = example._send_gripper_target(
            right_leader, right_gripper, 0.05
        )

        self.assertEqual((left_target, left_sent), (0.25, True))
        self.assertEqual((right_target, right_sent), (0.75, True))
        self.assertEqual(left_gripper.targets, [0.25])
        self.assertEqual(right_gripper.targets, [0.75])
        self.assertEqual(left_leader.timeouts, [0.05])
        self.assertEqual(right_leader.timeouts, [0.05])

    def test_missing_or_invalid_leader_reading_does_not_send_a_gripper_command(self) -> None:
        gripper = _FakeGripper()

        for opening in (None, float("nan"), "not-a-number"):
            with self.subTest(opening=opening):
                target, sent = example._send_gripper_target(
                    _FakeLeader(opening), gripper, 0.05
                )
                self.assertIsNone(target)
                self.assertIsNone(sent)

        self.assertEqual(gripper.targets, [])

    def test_target_is_clamped_to_the_normalized_gripper_range(self) -> None:
        gripper = _FakeGripper()

        low_target, _ = example._send_gripper_target(_FakeLeader(-0.3), gripper, 0.05)
        high_target, _ = example._send_gripper_target(_FakeLeader(1.3), gripper, 0.05)

        self.assertEqual(low_target, 0.0)
        self.assertEqual(high_target, 1.0)
        self.assertEqual(gripper.targets, [0.0, 1.0])

    def test_validation_allows_one_enabled_side_and_rejects_shared_active_ports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "openarm_mini.json"
            calibration_path.write_text("{}", encoding="utf-8")
            openarm = SimpleNamespace(
                port_left="/dev/ttyACM0",
                port_right="/dev/ttyACM1",
                calibration_path=str(calibration_path),
                baudrate=1_000_000,
            )
            enabled = GloriaMGripperConfig(enabled=True, port="/dev/ttyACM2")
            disabled_dual = GloriaMDualGripperConfig(
                left=enabled,
                right=GloriaMGripperConfig(enabled=False, port="/dev/ttyACM3"),
            )
            self.assertEqual(example._validate_config(openarm, disabled_dual), ("left",))

            shared_port_dual = GloriaMDualGripperConfig(
                left=enabled,
                right=GloriaMGripperConfig(enabled=True, port="/dev/ttyACM2"),
            )
            with self.assertRaisesRegex(ValueError, "已启用的 OpenArm"):
                example._validate_config(openarm, shared_port_dual)

    def test_validation_requires_at_least_one_enabled_side(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "openarm_mini.json"
            calibration_path.write_text("{}", encoding="utf-8")
            openarm = SimpleNamespace(
                port_left="/dev/ttyACM0",
                port_right="/dev/ttyACM1",
                calibration_path=str(calibration_path),
                baudrate=1_000_000,
            )
            disabled = GloriaMDualGripperConfig(
                left=GloriaMGripperConfig(enabled=False, port="/dev/ttyACM2"),
                right=GloriaMGripperConfig(enabled=False, port="/dev/ttyACM3"),
            )

            with self.assertRaisesRegex(ValueError, "至少"):
                example._validate_config(openarm, disabled)

    def test_shutdown_stops_workers_before_disabling_and_disconnects_in_reverse_order(
        self,
    ) -> None:
        events: list[str] = []
        stop_event = threading.Event()
        left_gripper = _FakeLifecycleDevice("left_gripper", events)
        right_gripper = _FakeLifecycleDevice("right_gripper", events)
        left_leader = _FakeLifecycleDevice("left_leader", events)
        right_leader = _FakeLifecycleDevice("right_leader", events)

        example._shutdown(
            [_FakeWorker(events)],  # type: ignore[list-item]
            stop_event,
            [left_gripper, right_gripper],  # type: ignore[list-item]
            [left_leader, right_leader],  # type: ignore[list-item]
        )

        self.assertTrue(stop_event.is_set())
        self.assertEqual(
            events,
            [
                "worker.join",
                "right_gripper.disable",
                "right_gripper.disconnect",
                "left_gripper.disable",
                "left_gripper.disconnect",
                "right_leader.disconnect",
                "left_leader.disconnect",
            ],
        )


if __name__ == "__main__":
    unittest.main()
