"""OpenArm Mini 单关节采样连续性 demo 的无硬件测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from examples import test_openarm_mini_joint_continuity as example


def _config() -> example.ContinuityConfig:
    return example.ContinuityConfig(
        joint_number=1,
        sample_rate_hz=100.0,
        read_timeout_s=0.01,
        test_duration_s=1.0,
        summary_interval_s=1.0,
        sample_deadband_deg=0.10,
        trend_start_deg=0.30,
        reversal_confirm_deg=0.50,
        large_sample_step_deg=3.0,
        gap_warning_multiplier=2.0,
    )


class OpenArmMiniJointContinuityExampleTest(unittest.TestCase):
    def _record_values(
        self,
        values: tuple[float, ...],
        config: example.ContinuityConfig,
    ) -> example.JointContinuityStats:
        stats = example.JointContinuityStats(config.joint_index)
        for index, value in enumerate(values):
            angles = np.zeros(example.JOINT_COUNT)
            angles[config.joint_index] = value
            stats.record(
                completed_at_s=index * 0.01,
                read_duration_s=0.001,
                status="ok",
                angles_deg=angles,
                detail="",
                config=config,
            )
        return stats

    def test_monotonic_trace_does_not_report_a_reversal(self) -> None:
        stats = self._record_values((0.0, 0.1, 0.3, 0.5, 0.7, 0.9), _config())

        self.assertEqual(stats.trend_direction, 1)
        self.assertEqual(stats.events, [])
        self.assertAlmostEqual(stats.max_opposite_excursion_deg, 0.0)

    def test_sustained_backtrack_reports_one_reversal(self) -> None:
        stats = self._record_values((0.0, 0.3, 0.7, 1.0, 0.8, 0.4), _config())

        self.assertEqual(len(stats.events), 1)
        event = stats.events[0]
        self.assertEqual(event.old_direction, 1)
        self.assertEqual(event.new_direction, -1)
        self.assertAlmostEqual(event.extreme_angle_deg, 1.0)
        self.assertAlmostEqual(event.observed_angle_deg, 0.4)
        self.assertAlmostEqual(event.backtrack_deg, 0.6)

    def test_small_backtrack_is_retained_but_not_confirmed(self) -> None:
        stats = self._record_values((0.0, 0.4, 0.8, 0.6, 0.9), _config())

        self.assertEqual(stats.events, [])
        self.assertAlmostEqual(stats.max_opposite_excursion_deg, 0.2)

    def test_csv_contains_raw_axes_and_reversal_columns(self) -> None:
        config = _config()
        stats = self._record_values((0.0, 0.4, 1.0, 0.4), config)
        report = example.ContinuityReport(
            config=config,
            elapsed_s=0.03,
            cycle_count=4,
            missed_tick_count=0,
            maximum_lateness_s=0.0,
            interrupted=False,
            stats=stats,
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.csv"
            example.write_csv(output, report)
            contents = output.read_text(encoding="utf-8")

        self.assertIn("j7_deg", contents)
        self.assertIn("reversal_backtrack_deg", contents)
        self.assertIn("0.6", contents)

    def test_config_rejects_a_confirmation_threshold_below_noise_deadband(self) -> None:
        invalid = replace(_config(), reversal_confirm_deg=0.05)

        with self.assertRaisesRegex(ValueError, "reversal_confirm_deg"):
            invalid.validate()

    def test_main_uses_top_level_config_without_yaml_or_cli(self) -> None:
        with patch.object(example, "run_continuity_test") as run_test:
            run_test.return_value = example.ContinuityReport(
                config=example.CONTINUITY_CONFIG,
                elapsed_s=0.0,
                cycle_count=0,
                missed_tick_count=0,
                maximum_lateness_s=0.0,
                interrupted=False,
                stats=example.JointContinuityStats(
                    example.CONTINUITY_CONFIG.joint_index
                ),
            )
            with patch.object(example, "_validate_connection_config"), patch.object(
                example, "_configured_csv_path", return_value=None
            ), patch.object(example, "OpenArmMiniLeaderArm") as leader_class:
                leader = leader_class.return_value
                leader.joint_count = example.JOINT_COUNT
                result = example.main()

        self.assertEqual(result, 0)
        self.assertTrue(leader.connect.called)
        self.assertTrue(leader.disconnect.called)
        self.assertFalse(hasattr(example, "load_runtime_config"))


if __name__ == "__main__":
    unittest.main()
