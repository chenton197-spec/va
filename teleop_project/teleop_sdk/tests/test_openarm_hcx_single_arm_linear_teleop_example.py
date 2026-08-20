"""单臂 OpenArm -> HCX 线性插值 demo 的无硬件测试。"""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from examples import test_openarm_hcx_single_arm_linear_teleop as example


class OpenArmHcxSingleArmLinearTeleopExampleTest(unittest.TestCase):
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

    def test_linear_stream_generates_five_continuous_500_hz_points(self) -> None:
        stream = example.LinearTargetStream(
            source_rate_hz=100,
            output_rate_hz=500,
            initial_angles_deg=np.zeros(example.JOINT_COUNT),
        )
        target = np.full(example.JOINT_COUNT, 10.0)

        points = np.asarray([stream.step(target) for _ in range(5)])

        self.assertEqual(stream.samples_per_interval, 5)
        np.testing.assert_allclose(points[:, 0], (2.0, 4.0, 6.0, 8.0, 10.0))
        np.testing.assert_allclose(stream.step(target), target)

        next_target = np.full(example.JOINT_COUNT, 20.0)
        next_points = np.asarray([stream.step(next_target) for _ in range(5)])
        np.testing.assert_allclose(
            next_points[:, 0],
            (12.0, 14.0, 16.0, 18.0, 20.0),
        )

    def test_config_rejects_non_multiple_linear_rates(self) -> None:
        invalid = replace(example.DEMO_CONFIG, leader_sample_rate_hz=120.0)

        with self.assertRaisesRegex(ValueError, "整数倍"):
            invalid.validate()

    def test_main_uses_top_level_config_without_yaml_or_cli(self) -> None:
        with patch.object(example, "run_demo") as run_demo:
            result = example.main()

        self.assertEqual(result, 0)
        run_demo.assert_called_once_with(example.DEMO_CONFIG)
        self.assertFalse(hasattr(example, "load_runtime_config"))


if __name__ == "__main__":
    unittest.main()
