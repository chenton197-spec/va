"""单臂 OpenArm -> HCX limited 遥操作示例的无硬件测试。"""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from examples import test_openarm_hcx_single_arm_direct_teleop as example
from teleop_sdk.algorithms import LimitedInterpolator


class OpenArmHcxSingleArmDirectTeleopExampleTest(unittest.TestCase):
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

    def test_latest_target_is_processed_by_500_hz_limited_interpolator(self) -> None:
        latest = example.LatestTarget(np.zeros(example.JOINT_COUNT))
        target = np.full(example.JOINT_COUNT, 10.0)
        latest.publish(target)
        snapshot = latest.snapshot()

        interpolator = LimitedInterpolator(
            500,
            500,
            max_velocity_deg_s=120.0,
            max_acceleration_deg_s2=80.0,
            lowpass_alpha=0.2,
        )
        interpolator.reset(np.zeros(example.JOINT_COUNT))
        points = np.asarray([interpolator.step(snapshot) for _ in range(5)])
        positions = np.concatenate((np.zeros(1), points[:, 0]))
        velocity = np.diff(positions) * 500.0
        acceleration = np.diff(np.concatenate((np.zeros(1), velocity))) * 500.0

        self.assertTrue(np.all(np.diff(points[:, 0]) > 0.0))
        self.assertLessEqual(np.max(np.abs(velocity)), 120.0)
        self.assertLessEqual(np.max(np.abs(acceleration)), 80.0 + 1e-9)

    def test_config_rejects_invalid_limited_lowpass_alpha(self) -> None:
        invalid = replace(example.DEMO_CONFIG, limited_lowpass_alpha=1.1)

        with self.assertRaisesRegex(ValueError, "0 到 1"):
            invalid.validate()

    def test_main_uses_top_level_config_without_yaml_or_cli(self) -> None:
        with patch.object(example, "run_demo") as run_demo:
            result = example.main()

        self.assertEqual(result, 0)
        run_demo.assert_called_once_with(example.DEMO_CONFIG)
        self.assertFalse(hasattr(example, "load_runtime_config"))


if __name__ == "__main__":
    unittest.main()
