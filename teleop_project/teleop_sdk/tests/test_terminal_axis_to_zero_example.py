"""J1 后置规划运动示例的无硬件测试。"""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples import test_terminal_axis_to_zero as example


class _FakeMotion:
    def __init__(self, arm: "_FakeArm", sequence: int) -> None:
        self._arm = arm
        self.sequence = sequence

    def wait(self, *, timeout_s: float) -> SimpleNamespace:
        raise AssertionError("阶段切换不得依赖 moveJoints2 回调等待")


class _FakeArm:
    def __init__(self, robot_id: int) -> None:
        self.robot_id = robot_id
        self.submitted_targets: list[tuple[float, ...]] = []
        self.feedback_angles_deg = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0)
        self.feedback_samples: list[tuple[float, ...]] = []
        self.feedback_read_count = 0

    def move_joints(self, target: tuple[float, ...], **kwargs: object) -> _FakeMotion:
        self.submitted_targets.append(target)
        self.last_kwargs = kwargs
        self.feedback_angles_deg = target
        return _FakeMotion(self, len(self.submitted_targets))

    def joint_angles(self) -> tuple[float, ...]:
        self.feedback_read_count += 1
        if self.feedback_samples:
            self.feedback_angles_deg = self.feedback_samples.pop(0)
        return self.feedback_angles_deg


class TerminalAxisToZeroExampleTest(unittest.TestCase):
    @staticmethod
    def _full_plan(arm: _FakeArm, name: str) -> example.JointTargetPlan:
        return example.JointTargetPlan(
            arm_name=name,
            arm=arm,
            current_angles_deg=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0),
            target_angles_deg=(11.0, -90.0, 31.0, 41.0, 51.0, 61.0, 71.0),
            changed_axis_indices=(0, 1, 2, 3, 4, 5, 6),
        )

    def test_first_stage_holds_j1_and_second_stage_only_changes_j1(self) -> None:
        arm = _FakeArm(robot_id=2)
        full_plan = self._full_plan(arm, "左臂")

        first_stage = example._build_non_j1_plan(full_plan)
        second_stage = example._build_j1_only_plan(full_plan, first_stage)

        self.assertEqual(first_stage.target_angles_deg[example.J1_AXIS_INDEX], 10.0)
        self.assertNotIn(example.J1_AXIS_INDEX, first_stage.changed_axis_indices)
        self.assertEqual(first_stage.changed_axis_indices, (1, 2, 3, 4, 5, 6))
        self.assertEqual(second_stage.current_angles_deg, first_stage.target_angles_deg)
        self.assertEqual(second_stage.target_angles_deg, full_plan.target_angles_deg)
        self.assertEqual(second_stage.changed_axis_indices, (example.J1_AXIS_INDEX,))

    def test_both_first_stage_feedbacks_confirm_before_j1_stage_is_submitted(self) -> None:
        left_arm = _FakeArm(robot_id=2)
        right_arm = _FakeArm(robot_id=1)
        full_plans = (
            self._full_plan(left_arm, "左臂"),
            self._full_plan(right_arm, "右臂"),
        )
        first_stage_plans = tuple(
            example._build_non_j1_plan(plan) for plan in full_plans
        )

        first_stage_reports = example._submit_plans_then_confirm_feedback(
            first_stage_plans
        )
        self.assertEqual(len(first_stage_reports), 2)
        self.assertGreater(left_arm.feedback_read_count, 0)
        self.assertGreater(right_arm.feedback_read_count, 0)
        self.assertEqual(
            first_stage_reports[0]["状态"], "实际关节反馈已确认到位"
        )
        self.assertEqual(
            first_stage_reports[1]["状态"], "实际关节反馈已确认到位"
        )

        second_stage_plans = tuple(
            example._build_j1_only_plan(full_plan, first_stage_plan)
            for full_plan, first_stage_plan in zip(full_plans, first_stage_plans)
        )
        example._submit_plans_then_confirm_feedback(second_stage_plans)

        for arm, full_plan in zip((left_arm, right_arm), full_plans):
            self.assertEqual(len(arm.submitted_targets), 2)
            self.assertEqual(
                arm.submitted_targets[0][example.J1_AXIS_INDEX],
                full_plan.current_angles_deg[example.J1_AXIS_INDEX],
            )
            self.assertEqual(arm.submitted_targets[1], full_plan.target_angles_deg)
            self.assertEqual(
                arm.last_kwargs,
                {
                    "interrupt": False,
                    "acceleration_seconds": example.ACCELERATION_SECONDS,
                    "deceleration_seconds": example.DECELERATION_SECONDS,
                    "speed_ratio": example.MOTION_SPEED_RATIO,
                    "smooth": example.SMOOTH,
                    "wait": False,
                },
            )

    def test_feedback_confirmation_polls_until_target_is_reached(self) -> None:
        arm = _FakeArm(robot_id=2)
        plan = self._full_plan(arm, "左臂")
        first_stage = example._build_non_j1_plan(plan)
        arm.feedback_samples = [
            (10.0, -89.0, 31.0, 41.0, 51.0, 61.0, 71.0),
            first_stage.target_angles_deg,
        ]

        with patch.object(example, "FEEDBACK_CONFIRM_POLL_INTERVAL_S", 0.0001):
            reports = example._confirm_submitted_plans_by_feedback(
                [(first_stage, _FakeMotion(arm, sequence=1))]
            )

        self.assertEqual(arm.feedback_read_count, 2)
        self.assertEqual(reports[0]["状态"], "实际关节反馈已确认到位")

    def test_j1_already_at_target_does_not_submit_a_second_stage_motion(self) -> None:
        arm = _FakeArm(robot_id=2)
        full_plan = example.JointTargetPlan(
            arm_name="左臂",
            arm=arm,
            current_angles_deg=(11.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0),
            target_angles_deg=(11.0, -90.0, 31.0, 41.0, 51.0, 61.0, 71.0),
            changed_axis_indices=(1, 2, 3, 4, 5, 6),
        )
        first_stage = example._build_non_j1_plan(full_plan)
        second_stage = example._build_j1_only_plan(full_plan, first_stage)

        self.assertEqual(second_stage.changed_axis_indices, ())
        self.assertIsNone(example._submit_plan(second_stage))
        self.assertEqual(arm.submitted_targets, [])


if __name__ == "__main__":
    unittest.main()
