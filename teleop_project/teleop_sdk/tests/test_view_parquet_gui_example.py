"""Parquet 图形查看器数据格式化逻辑的无 GUI 测试。"""

from __future__ import annotations

import unittest

import pyarrow as pa

from examples import view_parquet_gui as example


class ViewParquetGuiExampleTest(unittest.TestCase):
    def test_full_text_formats_nested_values_as_readable_json(self) -> None:
        text = example._full_text({"angles": [1.0, 2.0], "valid": True})
        self.assertIn('"angles": [', text)
        self.assertIn('"valid": true', text)

    def test_cell_preview_is_single_line_and_truncated(self) -> None:
        preview = example._cell_preview(list(range(20)), limit=20)
        self.assertNotIn("\n", preview)
        self.assertEqual(len(preview), 20)
        self.assertTrue(preview.endswith("..."))

    def test_timestamp_summary_reports_rate_and_intervals(self) -> None:
        table = pa.table({"timestamp": [0.0, 0.1, 0.2], "value": [1, 2, 3]})
        summary = example._timestamp_summary(table)
        self.assertIn("10.00 Hz", summary)
        self.assertIn("时长 0.200 s", summary)

    def test_schema_text_lists_field_names_and_types(self) -> None:
        table = pa.table({"frame_index": [0], "action": [[1.0, 2.0]]})
        text = example._schema_text(table)
        self.assertIn("frame_index", text)
        self.assertIn("action", text)
        self.assertIn("list<item: double>", text)


if __name__ == "__main__":
    unittest.main()
