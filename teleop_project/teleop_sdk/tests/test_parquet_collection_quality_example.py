"""无硬件验证 Parquet 采集质量分析示例。"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import unittest

import numpy as np

from examples import test_parquet_collection_quality as example


def _table(
    *,
    frame_indices: list[int],
    timestamps: list[float],
    action: list[list[float]],
    state: list[list[float]],
):
    import pyarrow as pa

    row_count = len(frame_indices)
    return pa.table(
        {
            "index": pa.array(range(100, 100 + row_count), type=pa.int64()),
            "episode_index": pa.array([7] * row_count, type=pa.int64()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "action": pa.array(action, type=pa.list_(pa.float32(), 2)),
            "observation.state": pa.array(state, type=pa.list_(pa.float32(), 2)),
            "action.left_gripper": pa.array([0.0] * row_count, type=pa.float32()),
            "observation.left_gripper": pa.array([0.0] * row_count, type=pa.float32()),
            "observation.images.head": pa.array(
                [
                    {"path": f"head/{index}.jpg", "timestamp": timestamp}
                    for index, timestamp in enumerate(timestamps)
                ],
                type=pa.struct([("path", pa.string()), ("timestamp", pa.float32())]),
            ),
        }
    )


class ParquetCollectionQualityExampleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = replace(
            example.ANALYSIS_CONFIG,
            expected_frame_rate_hz=10.0,
            joint_step_discontinuity_deg=2.0,
            max_reported_discontinuities=4,
        )

    def test_global_index_is_checked_relative_to_its_first_value(self) -> None:
        metrics = example.analyze_sequence(
            "index", [100, 101, 103], expected_start=None
        )

        self.assertEqual(metrics.missing_count, 1)
        self.assertEqual(metrics.mismatch_count, 1)

    def test_regular_rows_are_continuous(self) -> None:
        table = _table(
            frame_indices=[0, 1, 2, 3],
            timestamps=[0.0, 0.1, 0.2, 0.3],
            action=[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]],
            state=[[0.0, 0.0], [0.4, 0.0], [0.9, 0.0], [1.4, 0.0]],
        )

        report = example.analyze_table(table, Path("episode.parquet"), self.config)

        assert report.frame_index is not None
        assert report.timeline is not None
        self.assertEqual(report.row_count, 4)
        self.assertEqual(report.frame_index.mismatch_count, 0)
        self.assertEqual(report.timeline.gap_warning_count, 0)
        self.assertEqual(report.vectors[0].discontinuity_count.tolist(), [0, 0])
        self.assertEqual(len(report.scalars), 2)
        self.assertIsNotNone(report.alignment)
        assert report.alignment is not None
        np.testing.assert_allclose(
            report.alignment.maximum_absolute_error_deg,
            [0.1, 0.0],
            atol=1e-6,
        )

    def test_reports_frame_gap_time_gap_and_joint_jump(self) -> None:
        table = _table(
            frame_indices=[0, 1, 3, 4],
            timestamps=[0.0, 0.1, 0.5, 0.6],
            action=[[0.0, 0.0], [0.5, 0.0], [5.0, 0.0], [5.5, 0.0]],
            state=[[0.0, 0.0], [0.5, 0.0], [5.0, 0.0], [5.5, 0.0]],
        )

        report = example.analyze_table(table, Path("episode.parquet"), self.config)

        assert report.frame_index is not None
        assert report.timeline is not None
        self.assertEqual(report.frame_index.missing_count, 1)
        self.assertEqual(report.frame_index.mismatch_count, 2)
        self.assertEqual(report.timeline.gap_warning_count, 1)
        action = next(item for item in report.vectors if item.name == "action")
        self.assertEqual(action.discontinuity_count.tolist(), [1, 0])
        self.assertEqual(len(action.discontinuities), 1)
        self.assertEqual(action.discontinuities[0].previous_frame_row, 1)
        self.assertEqual(action.discontinuities[0].frame_row, 2)

    def test_camera_future_timestamp_and_duplicate_path_are_reported(self) -> None:
        table = _table(
            frame_indices=[0, 1],
            timestamps=[0.0, 0.1],
            action=[[0.0, 0.0], [0.0, 0.0]],
            state=[[0.0, 0.0], [0.0, 0.0]],
        )
        import pyarrow as pa

        camera = pa.array(
            [
                {"path": "head/0.jpg", "timestamp": 0.02},
                {"path": "head/0.jpg", "timestamp": 0.10},
            ],
            type=pa.struct([("path", pa.string()), ("timestamp", pa.float32())]),
        )
        table = table.set_column(
            table.column_names.index("observation.images.head"),
            "observation.images.head",
            camera,
        )

        report = example.analyze_table(table, Path("episode.parquet"), self.config)

        self.assertEqual(len(report.cameras), 1)
        self.assertEqual(report.cameras[0].duplicate_path_count, 1)
        self.assertEqual(report.cameras[0].future_timestamp_count, 1)

    def test_report_prints_summary_and_per_joint_rows(self) -> None:
        table = _table(
            frame_indices=[0, 1],
            timestamps=[0.0, 0.1],
            action=[[0.0, 0.0], [0.5, 0.0]],
            state=[[0.0, 0.0], [0.5, 0.0]],
        )
        report = example.analyze_table(table, Path("episode.parquet"), self.config)
        output = io.StringIO()

        with redirect_stdout(output):
            example.print_report(report, self.config)

        text = output.getvalue()
        self.assertIn("[SUMMARY]", text)
        self.assertIn("frame_index 连续", text)
        self.assertIn("关节      范围°", text)
        self.assertIn("J1", text)


if __name__ == "__main__":
    unittest.main()
