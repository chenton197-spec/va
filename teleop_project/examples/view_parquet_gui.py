#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用一个独立 Qt 窗口清晰查看 Parquet 的结构和逐帧数据。"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any

# 修改为需要查看的 Parquet 文件，不使用命令行参数。
PARQUET_PATH = Path(
    "datasets/openarm_hcx_dual_arm/data/chunk-000/episode_000000.parquet"
)
# 表格单元格只显示摘要；选中单元格后，下方会显示不截断的完整值。
CELL_PREVIEW_LENGTH = 120


def _load_qt_modules() -> tuple[Any, Any, Any]:
    """实际打开窗口时才导入 PySide6。"""

    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            "缺少 PySide6；请执行 python -m pip install "
            "-r requirements_hcx_follower_gui.txt"
        ) from exc
    return QtCore, QtGui, QtWidgets


def _load_table(path: Path) -> Any:
    """读取一个 Parquet 文件并拒绝空路径。"""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"找不到 Parquet 文件: {resolved}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("查看 Parquet 需要安装 pyarrow") from exc
    return pq.read_table(resolved)


def _python_value(table: Any, row: int, column: int) -> Any:
    """将一个 Arrow scalar 转为便于展示的 Python 值。"""

    return table.column(column)[row].as_py()


def _full_text(value: Any) -> str:
    """将数组、结构体和空值格式化为可复制的完整文本。"""

    if value is None:
        return "null"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True)
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def _cell_preview(value: Any, limit: int = CELL_PREVIEW_LENGTH) -> str:
    """生成不会撑大表格列宽的单行摘要。"""

    if limit < 4:
        raise ValueError("单元格摘要长度至少为 4")
    text = _full_text(value).replace("\n", " ")
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _timestamp_summary(table: Any) -> str:
    """根据 timestamp 列计算时长、平均频率和间隔范围。"""

    if "timestamp" not in table.column_names or table.num_rows < 2:
        return "时间统计: timestamp 有效帧不足 2 条"
    raw_values = table.column("timestamp").to_pylist()
    timestamps = [
        float(value)
        for value in raw_values
        if value is not None and math.isfinite(float(value))
    ]
    intervals = [
        later - earlier
        for earlier, later in zip(timestamps, timestamps[1:])
        if later > earlier
    ]
    if len(timestamps) < 2 or not intervals:
        return "时间统计: timestamp 没有有效正向间隔"
    duration_s = timestamps[-1] - timestamps[0]
    average_hz = (len(timestamps) - 1) / duration_s if duration_s > 0.0 else 0.0
    return (
        f"时长 {duration_s:.3f} s  |  平均 {average_hz:.2f} Hz  |  "
        f"间隔 中位/最小/最大 "
        f"{median(intervals) * 1000:.2f}/{min(intervals) * 1000:.2f}/"
        f"{max(intervals) * 1000:.2f} ms"
    )


def _schema_text(table: Any) -> str:
    """生成带序号、字段名和 Arrow 类型的 schema 文本。"""

    lines = [
        f"{index + 1:>3}. {field.name}\n     {field.type}"
        for index, field in enumerate(table.schema)
    ]
    return "\n\n".join(lines)


def _create_window(application: Any, table: Any, path: Path) -> Any:
    """创建按需读取 Arrow 单元格的 Qt 主窗口。"""

    QtCore, QtGui, QtWidgets = _load_qt_modules()

    class ParquetTableModel(QtCore.QAbstractTableModel):
        def rowCount(self, parent: Any = QtCore.QModelIndex()) -> int:  # noqa: N802
            return 0 if parent.isValid() else table.num_rows

        def columnCount(self, parent: Any = QtCore.QModelIndex()) -> int:  # noqa: N802
            return 0 if parent.isValid() else table.num_columns

        def data(
            self, index: Any, role: int = QtCore.Qt.ItemDataRole.DisplayRole
        ) -> Any:
            if not index.isValid():
                return None
            if role == QtCore.Qt.ItemDataRole.DisplayRole:
                return _cell_preview(_python_value(table, index.row(), index.column()))
            if role == QtCore.Qt.ItemDataRole.ToolTipRole:
                return "单击后在下方查看完整值"
            if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
                return int(
                    QtCore.Qt.AlignmentFlag.AlignLeft
                    | QtCore.Qt.AlignmentFlag.AlignVCenter
                )
            return None

        def headerData(  # noqa: N802
            self,
            section: int,
            orientation: Any,
            role: int = QtCore.Qt.ItemDataRole.DisplayRole,
        ) -> Any:
            if role != QtCore.Qt.ItemDataRole.DisplayRole:
                return None
            if orientation == QtCore.Qt.Orientation.Horizontal:
                return table.column_names[section]
            return str(section)

    class ParquetWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(f"Parquet 数据查看器 - {path.name}")
            self.setMinimumSize(980, 680)
            self.resize(1440, 900)
            self._model = ParquetTableModel(self)
            self._build()

        def _build(self) -> None:
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            root = QtWidgets.QVBoxLayout(central)
            root.setContentsMargins(12, 12, 12, 12)
            root.setSpacing(8)

            path_label = QtWidgets.QLabel(str(path.expanduser().resolve()))
            path_label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            path_label.setStyleSheet("font-weight: 600;")
            root.addWidget(path_label)

            summary = QtWidgets.QLabel(
                f"{table.num_rows} 帧  |  {table.num_columns} 个字段  |  "
                f"{_timestamp_summary(table)}"
            )
            summary.setWordWrap(True)
            root.addWidget(summary)

            tools = QtWidgets.QHBoxLayout()
            tools.addWidget(QtWidgets.QLabel("跳转到帧"))
            self._row_spin = QtWidgets.QSpinBox()
            self._row_spin.setRange(0, max(0, table.num_rows - 1))
            self._row_spin.setEnabled(table.num_rows > 0)
            self._row_spin.valueChanged.connect(self._jump_to_row)
            tools.addWidget(self._row_spin)
            tools.addWidget(QtWidgets.QLabel("定位字段"))
            self._column_combo = QtWidgets.QComboBox()
            self._column_combo.addItems(table.column_names)
            self._column_combo.currentIndexChanged.connect(self._jump_to_column)
            tools.addWidget(self._column_combo, 1)
            root.addLayout(tools)

            tabs = QtWidgets.QTabWidget()
            root.addWidget(tabs, 1)

            data_page = QtWidgets.QWidget()
            data_layout = QtWidgets.QVBoxLayout(data_page)
            data_layout.setContentsMargins(0, 8, 0, 0)
            splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            data_layout.addWidget(splitter)

            self._table_view = QtWidgets.QTableView()
            self._table_view.setModel(self._model)
            self._table_view.setAlternatingRowColors(True)
            self._table_view.setSelectionBehavior(
                QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems
            )
            self._table_view.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
            )
            self._table_view.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
            self._table_view.verticalHeader().setDefaultSectionSize(28)
            self._table_view.horizontalHeader().setMinimumSectionSize(100)
            self._table_view.horizontalHeader().setDefaultSectionSize(190)
            self._table_view.clicked.connect(self._show_value)
            splitter.addWidget(self._table_view)

            details = QtWidgets.QGroupBox("当前单元格完整值")
            details_layout = QtWidgets.QVBoxLayout(details)
            self._selection_label = QtWidgets.QLabel("请选择一个单元格")
            self._selection_label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            details_layout.addWidget(self._selection_label)
            self._value_text = QtWidgets.QPlainTextEdit()
            self._value_text.setReadOnly(True)
            self._value_text.setLineWrapMode(
                QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth
            )
            self._value_text.setFont(
                QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
            )
            details_layout.addWidget(self._value_text, 1)
            copy_button = QtWidgets.QPushButton("复制完整值")
            copy_button.clicked.connect(
                lambda: application.clipboard().setText(self._value_text.toPlainText())
            )
            details_layout.addWidget(copy_button, 0, QtCore.Qt.AlignmentFlag.AlignRight)
            splitter.addWidget(details)
            splitter.setSizes([560, 240])
            tabs.addTab(data_page, "逐帧数据")

            schema_text = QtWidgets.QPlainTextEdit()
            schema_text.setReadOnly(True)
            schema_text.setPlainText(_schema_text(table))
            schema_text.setFont(
                QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
            )
            tabs.addTab(schema_text, "字段结构")

            if table.num_rows and table.num_columns:
                first = self._model.index(0, 0)
                self._table_view.setCurrentIndex(first)
                self._show_value(first)

        def _show_value(self, index: Any) -> None:
            if not index.isValid():
                return
            name = table.column_names[index.column()]
            value = _python_value(table, index.row(), index.column())
            self._selection_label.setText(
                f"帧 {index.row()}  |  字段 {name}  |  "
                f"类型 {table.schema.field(index.column()).type}"
            )
            self._value_text.setPlainText(_full_text(value))
            self._row_spin.blockSignals(True)
            self._row_spin.setValue(index.row())
            self._row_spin.blockSignals(False)
            self._column_combo.blockSignals(True)
            self._column_combo.setCurrentIndex(index.column())
            self._column_combo.blockSignals(False)

        def _jump_to_row(self, row: int) -> None:
            self._select(row, self._column_combo.currentIndex())

        def _jump_to_column(self, column: int) -> None:
            if column >= 0:
                self._select(self._row_spin.value(), column)

        def _select(self, row: int, column: int) -> None:
            if not table.num_rows or column < 0:
                return
            index = self._model.index(row, column)
            self._table_view.setCurrentIndex(index)
            self._table_view.scrollTo(
                index,
                QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter,
            )
            self._show_value(index)

    return ParquetWindow()


def main() -> int:
    """加载顶部指定的 Parquet，并显示一个只读查看窗口。"""

    try:
        table = _load_table(PARQUET_PATH)
        _, _, QtWidgets = _load_qt_modules()
    except Exception as exc:
        print(f"[ERROR] 无法打开 Parquet 查看器: {exc}", file=sys.stderr)
        return 2

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = _create_window(application, table, PARQUET_PATH)
    window.show()
    return int(application.exec())


if __name__ == "__main__":
    raise SystemExit(main())
