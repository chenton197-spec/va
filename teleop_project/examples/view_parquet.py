#!/usr/bin/env python3
"""打印一个采集 episode 的 Parquet 内容与采样频率统计。"""

from __future__ import annotations

import json
from pathlib import Path
from pprint import pformat
from statistics import median


# 修改为要查看的 episode Parquet 文件。
PARQUET_PATH = Path("datasets/alicia_fr3/data/chunk-000/episode_000000.parquet")
# None 表示打印全部行；较大的 episode 建议先保留一个较小值。
MAX_ROWS: int | None = 20


def main() -> None:
    """读取并打印 Parquet 的 schema、行数据及标称/实际采集频率。"""

    if MAX_ROWS is not None and MAX_ROWS <= 0:
        raise ValueError("MAX_ROWS 必须为正整数或 None")
    if not PARQUET_PATH.is_file():
        raise FileNotFoundError(f"找不到 Parquet 文件: {PARQUET_PATH}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("查看 Parquet 需要安装 pyarrow") from exc

    table = pq.read_table(PARQUET_PATH)
    print(f"[INFO] 文件: {PARQUET_PATH}")
    print(f"[INFO] 行数: {table.num_rows}，列数: {table.num_columns}")
    print(f"[INFO] 列: {', '.join(table.column_names)}")
    print("[INFO] Schema:")
    print(table.schema)
    _print_sampling_rate(table)
    _print_audit_sampling_rate(PARQUET_PATH)

    row_count = table.num_rows if MAX_ROWS is None else min(MAX_ROWS, table.num_rows)
    rows = table.to_pylist() if MAX_ROWS is None else table.slice(0, row_count).to_pylist()
    for row_index, row in enumerate(rows):
        print(f"\n[ROW {row_index}]")
        print(pformat(row, sort_dicts=False, width=120))

    if row_count < table.num_rows:
        print(f"\n[INFO] 仅显示前 {row_count} 行；将 MAX_ROWS 设为 None 可查看全部。")


def _print_sampling_rate(table: object) -> None:
    """Print the nominal rate stored in the synthetic Parquet time axis."""

    column_names = getattr(table, "column_names")
    if "timestamp" not in column_names:
        print("[RATE] Parquet 中没有 timestamp 列，无法统计标称频率")
        return
    timestamps = [float(value) for value in getattr(table, "column")("timestamp").to_pylist()]
    _print_rate("Parquet timestamp（标称）", timestamps, unit_to_seconds=1.0)
    print("[RATE] 注：Parquet timestamp 由 frame_index / recording.fps 生成，不代表真实落盘间隔。")


def _print_audit_sampling_rate(parquet_path: Path) -> None:
    """Use host monotonic timestamps in the audit sidecar for the real row cadence."""

    audit_path = _audit_path_for_parquet(parquet_path)
    if audit_path is None or not audit_path.is_file():
        print("[RATE] 未找到审计 JSONL，无法统计实际已写入行频率")
        return
    timestamps: list[float] = []
    with audit_path.open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            if not item.get("emitted"):
                continue
            value = item.get("sample_started_monotonic_ns")
            if isinstance(value, int) and not isinstance(value, bool):
                timestamps.append(float(value))
    print(f"[RATE] 审计文件: {audit_path}")
    _print_rate("审计 sample_started_monotonic_ns（实际写入行）", timestamps, unit_to_seconds=1e-9)


def _audit_path_for_parquet(parquet_path: Path) -> Path | None:
    """Resolve the sibling audit file from the standard dataset layout."""

    if parquet_path.parent.name.startswith("chunk-") and parquet_path.parent.parent.name == "data":
        return parquet_path.parent.parent.parent / "meta" / "recording_audit" / f"{parquet_path.stem}.jsonl"
    return None


def _print_rate(label: str, timestamps: list[float], *, unit_to_seconds: float) -> None:
    """Print robust interval and average frequency statistics for one time axis."""

    if len(timestamps) < 2:
        print(f"[RATE] {label}: 有效时间戳不足 2 个")
        return
    intervals_s = [
        (later - earlier) * unit_to_seconds
        for earlier, later in zip(timestamps, timestamps[1:])
        if later > earlier
    ]
    if not intervals_s:
        print(f"[RATE] {label}: 时间戳没有正向间隔")
        return
    elapsed_s = (timestamps[-1] - timestamps[0]) * unit_to_seconds
    average_hz = (len(timestamps) - 1) / elapsed_s if elapsed_s > 0.0 else 0.0
    print(
        f"[RATE] {label}: {average_hz:.2f} Hz "
        f"({len(timestamps)} 行，跨度 {elapsed_s:.3f} s，"
        f"间隔中位数 {median(intervals_s) * 1_000:.2f} ms，"
        f"最小/最大 {min(intervals_s) * 1_000:.2f}/{max(intervals_s) * 1_000:.2f} ms)"
    )


if __name__ == "__main__":
    main()
