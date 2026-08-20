"""OpenArm -> HCX 三段链路诊断示例的无硬件测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from examples import test_openarm_hcx_teleop_diagnostics as example


def _target(timestamp_s: float, joint_value_deg: float, joint_index: int = 2) -> example.TargetSample:
    angles_deg = np.zeros(example.JOINT_COUNT, dtype=float)
    angles_deg[joint_index] = joint_value_deg
    return example.TargetSample(timestamp_s, angles_deg)


def _stream_metrics(
    samples: tuple[example.TargetSample, ...] | list[example.TargetSample],
    rate_hz: float,
) -> example.StreamMetrics:
    return example.analyze_stream(
        samples,
        expected_rate_hz=rate_hz,
        same_tolerance_deg=0.01,
        reversal_tolerance_deg=0.02,
        gap_warning_multiplier=1.5,
    )


def _run_config(**overrides: object) -> example.DiagnosticRunConfig:
    return replace(
        example.DIAGNOSTIC_RUN_CONFIG,
        test_duration_s=1.0,
        trace_history_s=1.0,
        **overrides,
    )


class OpenArmHcxTeleopDiagnosticsExampleTest(unittest.TestCase):
    def test_platform_ratio_excludes_static_prefix_and_suffix(self) -> None:
        # J3 在静止后以两个台阶移动。只有两次有效变化之间的零增量会进入平台比例。
        values = [0.0] * 5 + [1.0] * 10 + [2.0] * 5
        metrics = _stream_metrics(
            [_target(index * 0.01, value) for index, value in enumerate(values)],
            100.0,
        )

        self.assertAlmostEqual(metrics.observed_rate_hz or 0.0, 100.0)
        self.assertGreater(metrics.same_ratio[2], 0.75)
        self.assertEqual(metrics.gap_warning_count, 0)

    def test_conclusion_identifies_source_platform_before_hardware_output(self) -> None:
        source_values = [0.0] * 5 + [1.0] * 10 + [2.0] * 5
        generated = _stream_metrics(
            [_target(index * 0.01, value) for index, value in enumerate(source_values)],
            100.0,
        )
        transmitted = _stream_metrics(
            [_target(index / 500.0, index * 0.01) for index in range(101)],
            500.0,
        )
        conclusions = example.diagnose_side(
            generated=generated,
            transmitted=transmitted,
            feedback=None,
            direct_servo_stats=None,
            config=_run_config(feedback_enabled=False),
        )

        self.assertIn("#1 活动关节的生成目标存在较多平台", conclusions[0])
        self.assertIn("#2 本次成功 set_target 流接近设定频率", conclusions[1])
        self.assertIn("FEEDBACK_ENABLED=False", conclusions[2])

    def test_conclusion_identifies_direct_servo_output_gap(self) -> None:
        generated = _stream_metrics(
            [_target(index * 0.01, index * 0.1) for index in range(20)], 100.0
        )
        transmitted = _stream_metrics(
            [_target(0.000, 0.0), _target(0.002, 0.1), _target(0.020, 0.2)],
            500.0,
        )
        conclusions = example.diagnose_side(
            generated=generated,
            transmitted=transmitted,
            feedback=None,
            direct_servo_stats=None,
            config=_run_config(feedback_enabled=False),
        )

        self.assertGreater(transmitted.gap_warning_count, 0)
        self.assertIn("#2 Python 直伺服输出存在降频、间隙或 miss", conclusions[1])

    def test_feedback_tracking_error_matches_latest_successful_target(self) -> None:
        transmitted = [_target(0.0, 0.0, joint_index=3), _target(0.1, 10.0, joint_index=3)]
        feedback = [
            example.FeedbackSample(
                timestamp_s=0.12,
                started_at_s=0.119,
                duration_s=0.001,
                angles_deg=np.zeros(example.JOINT_COUNT),
                error=None,
            ),
            example.FeedbackSample(
                timestamp_s=0.22,
                started_at_s=0.219,
                duration_s=0.001,
                angles_deg=np.zeros(example.JOINT_COUNT),
                error=None,
            ),
        ]
        metrics = example.analyze_feedback(
            feedback, transmitted, expected_rate_hz=10.0
        )
        generated = _stream_metrics(
            [_target(index * 0.01, index * 0.1) for index in range(20)], 100.0
        )
        transmitted_metrics = _stream_metrics(transmitted, 10.0)
        conclusions = example.diagnose_side(
            generated=generated,
            transmitted=transmitted_metrics,
            feedback=metrics,
            direct_servo_stats=None,
            config=_run_config(feedback_enabled=True),
        )

        self.assertEqual(metrics.matched_target_count, 2)
        self.assertIsNotNone(metrics.p95_absolute_error_deg)
        assert metrics.p95_absolute_error_deg is not None
        self.assertAlmostEqual(metrics.p95_absolute_error_deg[3], 10.0)
        self.assertIn("#3 实际反馈相对最近成功提交目标的 P95 误差较大", conclusions[2])

    def test_trace_recorder_and_csv_keep_all_three_stream_names(self) -> None:
        recorder = example.TraceRecorder(100.0, 500, 10.0, 1.0)
        recorder.generated_callback("left")(np.zeros(example.JOINT_COUNT), 1.0)
        recorder.transmitted_callback("left")(np.ones(example.JOINT_COUNT), 1.001)
        recorder.record_feedback(
            "left",
            timestamp_s=1.01,
            started_at_s=1.009,
            duration_s=0.001,
            angles_deg=np.full(example.JOINT_COUNT, 0.5),
            error=None,
        )
        traces = recorder.snapshot()
        generated = _stream_metrics(traces.generated["left"], 100.0)
        transmitted = _stream_metrics(traces.transmitted["left"], 500.0)
        feedback = example.analyze_feedback(
            traces.feedback["left"], traces.transmitted["left"], expected_rate_hz=10.0
        )
        side_report = example.SideDiagnosticReport(
            side="left",
            generated=generated,
            transmitted=transmitted,
            feedback=feedback,
            feedback_worker=None,
            direct_servo_stats=None,
            conclusions=(),
        )
        report = example.DiagnosticReport(
            elapsed_s=1.0,
            interrupted=False,
            control_loop=example.ControlLoopStats(100, 0, 0.0),
            left=side_report,
            right=replace(side_report, side="right"),
            traces=example.TraceSnapshot(
                generated={"left": traces.generated["left"], "right": ()},
                transmitted={"left": traces.transmitted["left"], "right": ()},
                feedback={"left": traces.feedback["left"], "right": ()},
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "diagnostic.csv"
            example.write_csv(output, report)
            contents = output.read_text(encoding="utf-8")

        self.assertIn("generated", contents)
        self.assertIn("set_target_success", contents)
        self.assertIn("joint_feedback", contents)

    def test_main_uses_top_level_config_without_loading_yaml_or_cli(self) -> None:
        report = object()
        with (
            patch.object(example, "run_diagnostic", return_value=report) as run_diagnostic,
            patch.object(example, "print_report") as print_report,
            patch.object(example, "DIAGNOSTIC_CSV_PATH", ""),
        ):
            result = example.main()

        self.assertEqual(result, 0)
        self.assertFalse(hasattr(example, "load_runtime_config"))
        run_diagnostic.assert_called_once_with(
            example.DIAGNOSTIC_HCX_CONFIG,
            example.DIAGNOSTIC_TELEOP_CONFIG,
            example.DIAGNOSTIC_RUN_CONFIG,
        )
        print_report.assert_called_once_with(report)

    def test_rejects_unconfirmed_direct_servo_before_connecting(self) -> None:
        unconfirmed = replace(
            example.DIAGNOSTIC_HCX_CONFIG, direct_servo_confirm_unsafe=False
        )

        with self.assertRaisesRegex(ValueError, "CONFIRM_DIRECT_SERVO"):
            example._build_direct_servo_config(unconfirmed, 100.0)


if __name__ == "__main__":
    unittest.main()
