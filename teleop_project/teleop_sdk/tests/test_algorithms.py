"""新增通用遥操作算法测试。"""

from __future__ import annotations

import unittest

import numpy as np

from teleop_sdk.algorithms import (
    AngleUnwrapper,
    LatencyProbe,
    LimitedInterpolator,
    LinearInterpolator,
)


class AngleUnwrapperTest(unittest.TestCase):
    def test_crossing_boundary_stays_continuous(self) -> None:
        unwrapper = AngleUnwrapper(2)
        np.testing.assert_array_equal(unwrapper.step(np.array([179.0, -179.0])), [179.0, -179.0])
        np.testing.assert_array_equal(unwrapper.step(np.array([-179.0, 179.0])), [181.0, -181.0])


class LatencyProbeTest(unittest.TestCase):
    def test_reports_per_axis_latency(self) -> None:
        probe = LatencyProbe(rate_hz=10.0, threshold_deg=1.0, quiescent_deg=0.1)
        probe.step(np.zeros(2), np.zeros(2), (0, 1), 0.0)
        probe.step(np.zeros(2), np.zeros(2), (0, 1), 0.1)
        probe.step(np.array([2.0, 0.0]), np.zeros(2), (0, 1), 0.2)
        result = probe.step(np.array([2.0, 0.0]), np.array([2.0, 0.0]), (0, 1), 0.3)
        self.assertEqual(result[0][0], 0)
        self.assertAlmostEqual(result[0][1], 100.0)


class LinearInterpolatorTest(unittest.TestCase):
    def test_resamples_a_100_hz_segment_to_ten_1000_hz_targets(self) -> None:
        interpolator = LinearInterpolator(source_rate_hz=100, output_rate_hz=1000)

        targets = interpolator.interpolate(
            np.array([0.0, 10.0]), np.array([10.0, -10.0])
        )

        self.assertEqual(targets.shape, (10, 2))
        np.testing.assert_allclose(targets[0], [1.0, 8.0])
        np.testing.assert_allclose(targets[-1], [10.0, -10.0])

    def test_rejects_non_integer_rate_ratios_and_invalid_vectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            LinearInterpolator(source_rate_hz=100, output_rate_hz=950)

        interpolator = LinearInterpolator(source_rate_hz=100, output_rate_hz=1000)
        with self.assertRaisesRegex(ValueError, "matching non-empty"):
            interpolator.interpolate(np.array([0.0]), np.array([0.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "finite"):
            interpolator.interpolate(np.array([0.0]), np.array([float("nan")]))


class LimitedInterpolatorTest(unittest.TestCase):
    def test_generates_output_rate_points_with_velocity_and_acceleration_limits(self) -> None:
        interpolator = LimitedInterpolator(
            source_rate_hz=100,
            output_rate_hz=1000,
            max_velocity_deg_s=20.0,
            max_acceleration_deg_s2=80.0,
            lowpass_alpha=1.0,
        )
        interpolator.reset(np.array([0.0]))

        targets = interpolator.interpolate(np.array([100.0]))[:, 0]
        velocity = np.diff(np.concatenate((np.array([0.0]), targets))) / 0.001
        acceleration = np.diff(np.concatenate((np.array([0.0]), velocity))) / 0.001

        self.assertEqual(targets.shape, (10,))
        self.assertLess(targets[-1], 100.0)
        self.assertLessEqual(np.max(np.abs(velocity)), 20.0 + 1e-9)
        self.assertLessEqual(np.max(np.abs(acceleration)), 80.0 + 1e-9)

    def test_braking_distance_limit_reaches_a_static_target_without_overshoot(self) -> None:
        interpolator = LimitedInterpolator(
            source_rate_hz=100,
            output_rate_hz=1000,
            max_velocity_deg_s=20.0,
            max_acceleration_deg_s2=80.0,
            lowpass_alpha=1.0,
        )
        interpolator.reset(np.array([0.0]))

        points = np.concatenate(
            [interpolator.interpolate(np.array([10.0]))[:, 0] for _ in range(100)]
        )
        held_points = interpolator.interpolate(np.array([10.0]))[:, 0]
        velocity = np.diff(np.concatenate((np.array([0.0]), points))) / 0.001
        acceleration = np.diff(np.concatenate((np.array([0.0]), velocity))) / 0.001

        self.assertTrue(np.all(np.diff(points) >= -1e-9))
        self.assertLessEqual(np.max(points), 10.0 + 1e-9)
        self.assertLessEqual(np.max(np.abs(acceleration)), 80.0 + 1e-9)
        self.assertAlmostEqual(points[-1], 10.0, places=8)
        np.testing.assert_allclose(held_points, np.full(10, 10.0))

    def test_preserves_state_between_source_intervals(self) -> None:
        interpolator = LimitedInterpolator(
            source_rate_hz=1,
            output_rate_hz=4,
            max_velocity_deg_s=1.0,
            max_acceleration_deg_s2=100.0,
            lowpass_alpha=1.0,
        )
        interpolator.reset(np.array([0.0]))

        first = interpolator.interpolate(np.array([10.0]))[:, 0]
        second = interpolator.interpolate(np.array([10.0]))[:, 0]

        np.testing.assert_allclose(first, [0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(second, [1.25, 1.5, 1.75, 2.0])

    def test_lowpass_alpha_zero_holds_the_reset_target(self) -> None:
        interpolator = LimitedInterpolator(
            source_rate_hz=100,
            output_rate_hz=1000,
            max_velocity_deg_s=100.0,
            max_acceleration_deg_s2=1000.0,
            lowpass_alpha=0.0,
        )
        interpolator.reset(np.array([3.0, -2.0]))

        targets = interpolator.interpolate(np.array([30.0, -20.0]))

        np.testing.assert_allclose(targets, np.array([[3.0, -2.0]] * 10))

    def test_clamps_overshoot_at_configured_joint_limits(self) -> None:
        interpolator = LimitedInterpolator(
            source_rate_hz=1,
            output_rate_hz=4,
            max_velocity_deg_s=100.0,
            max_acceleration_deg_s2=1000.0,
            lowpass_alpha=1.0,
            min_angles_deg=np.array([-1.0]),
            max_angles_deg=np.array([1.0]),
        )
        interpolator.reset(np.array([0.0]))

        targets = interpolator.interpolate(np.array([100.0]))[:, 0]

        np.testing.assert_allclose(targets, np.ones(4))

    def test_step_respects_actual_elapsed_time(self) -> None:
        interpolator = LimitedInterpolator(
            source_rate_hz=1000,
            output_rate_hz=1000,
            max_velocity_deg_s=100.0,
            max_acceleration_deg_s2=80.0,
            lowpass_alpha=1.0,
        )
        interpolator.reset(np.array([0.0]))

        first = interpolator.step(np.array([100.0]), elapsed_s=0.002)[0]
        second = interpolator.step(np.array([100.0]), elapsed_s=0.004)[0]
        first_velocity = first / 0.002
        second_velocity = (second - first) / 0.004

        self.assertLessEqual(abs(first_velocity), 100.0 + 1e-12)
        self.assertLessEqual(abs(second_velocity), 100.0 + 1e-12)
        self.assertAlmostEqual((second_velocity - first_velocity) / 0.004, 80.0)
        with self.assertRaisesRegex(ValueError, "elapsed_s"):
            interpolator.step(np.array([100.0]), elapsed_s=0.0)

    def test_requires_reset_and_valid_rate_ratio(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            LimitedInterpolator(
                source_rate_hz=100,
                output_rate_hz=950,
                max_velocity_deg_s=1.0,
                max_acceleration_deg_s2=1.0,
                lowpass_alpha=0.25,
            )

        interpolator = LimitedInterpolator(
            source_rate_hz=100,
            output_rate_hz=1000,
            max_velocity_deg_s=1.0,
            max_acceleration_deg_s2=1.0,
            lowpass_alpha=0.25,
        )
        with self.assertRaisesRegex(RuntimeError, "reset"):
            interpolator.interpolate(np.array([1.0]))
