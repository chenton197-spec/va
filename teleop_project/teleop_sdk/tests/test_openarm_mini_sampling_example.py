"""OpenArm Mini 采样诊断示例的无硬件测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples import test_openarm_mini_sampling as example


def _config() -> example.SamplingConfig:
    return example.SamplingConfig(
        rate_hz=100.0,
        read_timeout_s=0.1,
        test_duration_s=1.0,
        summary_interval_s=1.0,
        same_frame_tolerance_deg=0.01,
        large_joint_step_deg=5.0,
        gap_warning_multiplier=2.0,
        minimum_acceptable_rate_ratio=0.95,
    )


def _outcome(
    completed_at_s: float,
    angles_deg: np.ndarray | None,
    *,
    status: str = "ok",
) -> example.ReadOutcome:
    return example.ReadOutcome(
        completed_at_s=completed_at_s,
        duration_s=0.002,
        status=status,
        angles_deg=angles_deg,
    )


class _FakeLeader:
    joint_count = example.JOINT_COUNT

    def __init__(self, angles_deg: np.ndarray) -> None:
        self.angles_deg = angles_deg
        self.read_count = 0

    def read_joint_angles_deg(self, _timeout_s: float) -> np.ndarray:
        self.read_count += 1
        return self.angles_deg.copy()


class OpenArmMiniSamplingExampleTest(unittest.TestCase):
    def test_records_same_frames_and_large_joint_steps(self) -> None:
        config = _config()
        stats = example.ArmSamplingStats("left")
        zero = np.zeros(example.JOINT_COUNT)
        stepped = zero.copy()
        stepped[3] = 6.0

        stats.record(_outcome(0.0, zero), config)
        stats.record(_outcome(0.01, zero), config)
        stats.record(_outcome(0.02, stepped), config)

        self.assertEqual(stats.valid_count, 3)
        self.assertEqual(stats.same_frame_count, 1)
        self.assertEqual(stats.changed_frame_count, 1)
        self.assertEqual(stats.large_joint_step_count, 1)
        self.assertEqual(stats.gap_warning_count, 0)
        self.assertAlmostEqual(stats.observed_rate_hz or 0.0, 100.0)

    def test_records_none_frame_and_long_gap_between_valid_samples(self) -> None:
        config = _config()
        stats = example.ArmSamplingStats("right")
        zero = np.zeros(example.JOINT_COUNT)

        stats.record(_outcome(0.0, zero), config)
        stats.record(_outcome(0.01, None, status="none"), config)
        stats.record(_outcome(0.04, zero), config)

        self.assertEqual(stats.valid_count, 2)
        self.assertEqual(stats.none_frame_count, 1)
        self.assertEqual(stats.gap_warning_count, 1)
        self.assertAlmostEqual(stats.max_sample_gap_s, 0.04)

    def test_write_csv_exports_ok_and_none_records(self) -> None:
        config = _config()
        left = example.ArmSamplingStats("left")
        right = example.ArmSamplingStats("right")
        zero = np.zeros(example.JOINT_COUNT)
        left.record(_outcome(0.0, zero), config)
        right.record(_outcome(0.0, None, status="none"), config)
        report = example.SamplingReport(
            config=config,
            elapsed_s=0.01,
            cycle_count=1,
            missed_tick_count=0,
            maximum_lateness_s=0.0,
            mean_pair_completion_skew_s=0.0,
            maximum_pair_completion_skew_s=0.0,
            interrupted=False,
            left=left,
            right=right,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sampling.csv"
            example.write_csv(output, report)
            contents = output.read_text(encoding="utf-8")

        self.assertIn("completion_time_s", contents)
        self.assertIn("left", contents)
        self.assertIn("right", contents)
        self.assertIn("none", contents)

    def test_runs_one_parallel_dual_arm_sampling_cycle(self) -> None:
        config = _config()
        config = example.SamplingConfig(
            rate_hz=config.rate_hz,
            read_timeout_s=config.read_timeout_s,
            test_duration_s=0.001,
            summary_interval_s=config.summary_interval_s,
            same_frame_tolerance_deg=config.same_frame_tolerance_deg,
            large_joint_step_deg=config.large_joint_step_deg,
            gap_warning_multiplier=config.gap_warning_multiplier,
            minimum_acceptable_rate_ratio=config.minimum_acceptable_rate_ratio,
        )
        left = _FakeLeader(np.zeros(example.JOINT_COUNT))
        right = _FakeLeader(np.ones(example.JOINT_COUNT))

        report = example.run_sampling_test(left, right, config)

        self.assertGreaterEqual(report.cycle_count, 1)
        self.assertEqual(report.left.valid_count, report.cycle_count)
        self.assertEqual(report.right.valid_count, report.cycle_count)
        self.assertEqual(left.read_count, report.cycle_count)
        self.assertEqual(right.read_count, report.cycle_count)

    def test_rejects_invalid_sampling_rate(self) -> None:
        invalid = example.SamplingConfig(
            rate_hz=0.0,
            read_timeout_s=0.1,
            test_duration_s=1.0,
            summary_interval_s=1.0,
            same_frame_tolerance_deg=0.01,
            large_joint_step_deg=5.0,
            gap_warning_multiplier=2.0,
            minimum_acceptable_rate_ratio=0.95,
        )

        with self.assertRaisesRegex(ValueError, "rate_hz"):
            invalid.validate()


if __name__ == "__main__":
    unittest.main()
