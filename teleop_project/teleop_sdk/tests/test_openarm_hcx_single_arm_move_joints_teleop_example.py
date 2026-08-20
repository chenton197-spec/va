"""单臂 OpenArm -> HCX move_joints 遥操作示例的无硬件测试。"""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from examples import test_openarm_hcx_single_arm_move_joints_teleop as example


class _FakeArm:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def move_joints(self, angles_deg: list[float], **kwargs: object) -> object:
        self.calls.append({"angles_deg": angles_deg, **kwargs})
        return object()


class OpenArmHcxSingleArmMoveJointsTeleopExampleTest(unittest.TestCase):
    def test_relative_mapping_applies_axis_sign_and_joint_limits(self) -> None:
        leader_origin = np.zeros(example.JOINT_COUNT)
        follower_origin = np.full(example.JOINT_COUNT, 10.0)
        leader = np.array((2.0, 3.0, -4.0, 0.0, 8.0, -9.0, 1.0))
        lower = np.full(example.JOINT_COUNT, 0.0)
        upper = np.full(example.JOINT_COUNT, 12.0)

        target = example.map_relative_target(
            leader,
            leader_origin,
            follower_origin,
            (1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            lower,
            upper,
        )

        np.testing.assert_allclose(
            target,
            np.array((12.0, 7.0, 6.0, 10.0, 12.0, 12.0, 11.0)),
        )

    def test_submit_uses_nonblocking_interruptible_planned_motion(self) -> None:
        arm = _FakeArm()
        target = np.arange(example.JOINT_COUNT, dtype=float)

        example._submit_move_joints(arm, target, example.DEMO_CONFIG)

        self.assertEqual(len(arm.calls), 1)
        call = arm.calls[0]
        self.assertEqual(call["angles_deg"], target.tolist())
        self.assertTrue(call["interrupt"])
        self.assertFalse(call["wait"])
        self.assertEqual(
            call["acceleration_seconds"],
            example.DEMO_CONFIG.move_joint_acceleration_seconds,
        )
        self.assertEqual(
            call["deceleration_seconds"],
            example.DEMO_CONFIG.move_joint_deceleration_seconds,
        )
        self.assertEqual(
            call["speed_ratio"], example.DEMO_CONFIG.move_joint_speed_ratio
        )
        self.assertEqual(call["smooth"], example.DEMO_CONFIG.move_joint_smooth)

    def test_config_rejects_invalid_planned_command_rate(self) -> None:
        invalid = replace(example.DEMO_CONFIG, move_joint_command_rate_hz=0.0)

        with self.assertRaisesRegex(ValueError, "move_joint_command_rate_hz"):
            invalid.validate()

    def test_main_uses_top_level_config_without_yaml_or_cli(self) -> None:
        with patch.object(example, "run_demo") as run_demo:
            result = example.main()

        self.assertEqual(result, 0)
        run_demo.assert_called_once_with(example.DEMO_CONFIG)
        self.assertFalse(hasattr(example, "load_runtime_config"))


if __name__ == "__main__":
    unittest.main()
