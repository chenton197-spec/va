#!/usr/bin/env python3
"""持续检查 HCX 正式采集使用的三路 Orbbec 相机是否断流。

脚本只复刻正式采集的相机选帧路径：每张 head 新帧触发一次记录候选，
left_hand/right_hand 分别选择不晚于该 head 帧的最新缓存帧。它不会连接或控制
OpenArm、HCX 和 Gloria-M。完整的三相机候选使用正式数据写入器编码 JPEG 并落盘。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from leobot_scripts import (
    CameraFrame,
    OrbbecRGBDSource,
    OrbbecRGBSource,
    RGBDFrame,
    load_recording_config,
)
from orbbec_sdk import (
    CameraMode,
    CameraStatus,
    OrbbecCameraConfig,
    OrbbecManager,
    load_orbbec_camera_configs,
)
from leobot_scripts.v21_writer import (
    CameraSpec,
    RecordedFrame,
    V21DatasetWriter,
    WriterConfig,
)


# 与 openarm_hcx_dual_arm_record.py 完全相同的三路相机名称和主相机。
CAMERA_NAMES: tuple[str, str, str] = ("head", "left_hand", "right_hand")
MASTER_CAMERA_NAME = "head"
# None 表示持续运行到 Ctrl+C；也可填写正数进行定时测试，单位为秒。
TEST_DURATION_S: float | None = None
# 每隔多少秒输出一次窗口统计。
REPORT_INTERVAL_S = 5.0
# 没有新 head 帧时的轮询间隔；相机仍由 SDK 后台线程持续读取。
POLL_INTERVAL_S = 0.001
# head 最后一帧超过该时间即判定断流，与正式采集程序当前阈值一致。
MASTER_STALE_TIMEOUT_S = 3.0
# 单独的测试数据集目录，禁止指向正式 hcx_recording.root。
WRITE_DATASET_ROOT = Path("datasets/hcx_three_camera_write_test")
# 与当前正式采集一致，使用三路并行 JPEG 编码和图像序列落盘。
WRITE_IMAGE_STORAGE = "jpg"
# JPEG 编码质量，范围为 1-100。
WRITE_JPEG_QUALITY = 75


Frame = CameraFrame | RGBDFrame


@dataclass
class CameraStats:
    """一台相机在一个统计窗口内的选帧质量。"""

    selected: int = 0
    missing: int = 0
    invalid: int = 0
    reused: int = 0
    skipped_source_frames: int = 0
    source_index_reversals: int = 0
    first_source_index: int | None = None
    last_source_index: int | None = None
    first_capture_ns: int | None = None
    last_capture_ns: int | None = None
    max_capture_gap_ns: int = 0
    pair_delta_sum_ns: int = 0
    pair_delta_count: int = 0
    pair_delta_max_ns: int = 0

    def observe(self, frame: Frame | None, master_capture_ns: int) -> bool:
        """记录一次主相机驱动选帧；返回该帧是否完整有效。"""

        if frame is None:
            self.missing += 1
            return False
        self.selected += 1

        valid = _valid_rgb(frame)
        capture_ns = _capture_monotonic_ns(frame)
        if capture_ns is None or capture_ns > master_capture_ns:
            valid = False
        if not valid:
            self.invalid += 1

        source_index = _source_frame_index(frame)
        if source_index is not None:
            if self.first_source_index is None:
                self.first_source_index = source_index
            if self.last_source_index is not None:
                if source_index == self.last_source_index:
                    self.reused += 1
                elif source_index > self.last_source_index + 1:
                    self.skipped_source_frames += source_index - self.last_source_index - 1
                elif source_index < self.last_source_index:
                    self.source_index_reversals += 1
            self.last_source_index = source_index

        if capture_ns is not None:
            if self.first_capture_ns is None:
                self.first_capture_ns = capture_ns
            if self.last_capture_ns is not None and capture_ns > self.last_capture_ns:
                self.max_capture_gap_ns = max(
                    self.max_capture_gap_ns,
                    capture_ns - self.last_capture_ns,
                )
            self.last_capture_ns = capture_ns
            delta_ns = master_capture_ns - capture_ns
            if delta_ns >= 0:
                self.pair_delta_sum_ns += delta_ns
                self.pair_delta_count += 1
                self.pair_delta_max_ns = max(self.pair_delta_max_ns, delta_ns)
        return valid

    def producer_rate_hz(self) -> float | None:
        if (
            self.first_source_index is None
            or self.last_source_index is None
            or self.first_capture_ns is None
            or self.last_capture_ns is None
            or self.last_capture_ns <= self.first_capture_ns
        ):
            return None
        return (self.last_source_index - self.first_source_index) / (
            (self.last_capture_ns - self.first_capture_ns) / 1_000_000_000
        )


@dataclass
class WriteStats:
    calls: int = 0
    elapsed_sum_ns: int = 0
    elapsed_max_ns: int = 0

    def observe(self, elapsed_ns: int) -> None:
        self.calls += 1
        self.elapsed_sum_ns += elapsed_ns
        self.elapsed_max_ns = max(self.elapsed_max_ns, elapsed_ns)

    def text(self) -> str:
        if not self.calls:
            return "append=0"
        average_ms = self.elapsed_sum_ns / self.calls / 1_000_000
        maximum_ms = self.elapsed_max_ns / 1_000_000
        return f"append={self.calls} call(avg/max)={average_ms:.2f}/{maximum_ms:.2f}ms"


def _capture_monotonic_ns(frame: Frame) -> int | None:
    if isinstance(frame, CameraFrame):
        return frame.capture_monotonic_ns
    if frame.rgb_capture_monotonic_ns is not None:
        return frame.rgb_capture_monotonic_ns
    return frame.depth.capture_monotonic_ns


def _source_frame_index(frame: Frame) -> int | None:
    if isinstance(frame, CameraFrame):
        return frame.source_frame_index
    if frame.rgb_source_frame_index is not None:
        return frame.rgb_source_frame_index
    return frame.depth.source_frame_index


def _valid_rgb(frame: Frame) -> bool:
    rgb = frame.rgb
    return (
        isinstance(rgb, np.ndarray)
        and rgb.dtype == np.uint8
        and rgb.ndim == 3
        and rgb.shape[2] == 3
        and rgb.size > 0
    )


def _recorded_camera_frame(
    frame: Frame,
    origin_ns: int,
) -> CameraFrame:
    capture_ns = _capture_monotonic_ns(frame)
    timestamp_s = (
        0.0 if capture_ns is None else (capture_ns - origin_ns) / 1_000_000_000
    )
    return CameraFrame(
        rgb=frame.rgb,
        timestamp_s=timestamp_s,
        capture_monotonic_ns=capture_ns,
        source_frame_index=_source_frame_index(frame),
    )


def _selected_configs() -> tuple[OrbbecCameraConfig, ...]:
    if TEST_DURATION_S is not None and (
        not math.isfinite(TEST_DURATION_S) or TEST_DURATION_S <= 0.0
    ):
        raise ValueError("TEST_DURATION_S 必须为正的有限秒数或 None")
    for name, value in (
        ("REPORT_INTERVAL_S", REPORT_INTERVAL_S),
        ("POLL_INTERVAL_S", POLL_INTERVAL_S),
        ("MASTER_STALE_TIMEOUT_S", MASTER_STALE_TIMEOUT_S),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} 必须为正的有限数")
    if WRITE_IMAGE_STORAGE != "jpg":
        raise ValueError("WRITE_IMAGE_STORAGE 当前必须为 'jpg'")
    if (
        isinstance(WRITE_JPEG_QUALITY, bool)
        or not isinstance(WRITE_JPEG_QUALITY, int)
        or not 1 <= WRITE_JPEG_QUALITY <= 100
    ):
        raise ValueError("WRITE_JPEG_QUALITY 必须是 1-100 的整数")

    recording = load_recording_config(section_name="hcx_recording")
    if WRITE_DATASET_ROOT.resolve() == recording.root.resolve():
        raise ValueError("WRITE_DATASET_ROOT 不能与正式 hcx_recording.root 相同")
    declared = load_orbbec_camera_configs(section_name="hcx_orbbec")
    by_name = {config.name: config for config in declared}
    missing = [name for name in CAMERA_NAMES if name not in by_name]
    if missing:
        raise ValueError(f"hcx_orbbec.cameras 缺少相机: {missing}")
    configs = tuple(by_name[name] for name in CAMERA_NAMES)
    if any(config.mode not in {CameraMode.RGB, CameraMode.RGBD} for config in configs):
        raise ValueError("三路诊断相机必须提供 RGB 流")
    if any(config.fps != recording.fps for config in configs):
        raise ValueError("三台 hcx_orbbec 相机 fps 必须与 hcx_recording.fps 一致")
    return configs


def _new_stats() -> dict[str, CameraStats]:
    return {name: CameraStats() for name in CAMERA_NAMES}


def _camera_state_text(
    manager: OrbbecManager,
    name: str,
    stats: CameraStats,
    now_ns: int,
) -> str:
    camera = manager.camera(name)
    latest = camera.get_frame()
    latest_capture_ns = (
        latest.capture_monotonic_ns
        if latest is not None
        else stats.last_capture_ns
    )
    age_text = "--"
    if latest_capture_ns is not None:
        age_text = f"{max(0, now_ns - latest_capture_ns) / 1_000_000:.1f}ms"
    error = camera.last_error or "--"
    return f"status={camera.status.value} latest-age={age_text} error={error!r}"


def _stats_text(name: str, stats: CameraStats, elapsed_s: float) -> str:
    selected_rate = stats.selected / elapsed_s if elapsed_s > 0.0 else 0.0
    producer_rate = stats.producer_rate_hz()
    producer_text = "--" if producer_rate is None else f"{producer_rate:.2f}Hz"
    max_gap_ms = stats.max_capture_gap_ns / 1_000_000
    pair_text = "--"
    if stats.pair_delta_count:
        average_ms = stats.pair_delta_sum_ns / stats.pair_delta_count / 1_000_000
        maximum_ms = stats.pair_delta_max_ns / 1_000_000
        pair_text = f"{average_ms:.1f}/{maximum_ms:.1f}ms"
    return (
        f"{name}: selected={stats.selected} ({selected_rate:.2f}Hz) "
        f"producer={producer_text} missing={stats.missing} invalid={stats.invalid} "
        f"reused={stats.reused} source-skip={stats.skipped_source_frames} "
        f"index-reverse={stats.source_index_reversals} gap-max={max_gap_ms:.1f}ms "
        f"pair-delta(avg/max)={pair_text}"
    )


def _print_report(
    manager: OrbbecManager,
    stats: dict[str, CameraStats],
    elapsed_s: float,
    complete_rows: int,
    incomplete_rows: int,
    write_stats: WriteStats,
) -> None:
    now_ns = time.perf_counter_ns()
    print("-" * 88)
    print(
        f"[CAMERA TEST] window={elapsed_s:.2f}s "
        f"complete={complete_rows} incomplete={incomplete_rows} {write_stats.text()}"
    )
    for name in CAMERA_NAMES:
        print(f"  {_stats_text(name, stats[name], elapsed_s)}")
        print(f"    {_camera_state_text(manager, name, stats[name], now_ns)}")


def _failed_camera_message(manager: OrbbecManager) -> str | None:
    failures = []
    for name in CAMERA_NAMES:
        camera = manager.camera(name)
        if camera.status is CameraStatus.FAILED:
            failures.append(f"{name}: {camera.last_error or 'unknown camera failure'}")
    return "; ".join(failures) if failures else None


def main() -> int:
    try:
        configs = _selected_configs()
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[ERROR] 三相机诊断配置无效: {exc}")
        return 2

    manager = OrbbecManager(configs)
    total_stats = _new_stats()
    window_stats = _new_stats()
    total_complete = 0
    total_incomplete = 0
    window_complete = 0
    window_incomplete = 0
    total_write_stats = WriteStats()
    window_write_stats = WriteStats()
    started_ns: int | None = None
    report_started_ns = 0
    last_master_capture_ns: int | None = None
    exit_code = 0
    writer: V21DatasetWriter | None = None
    writer_frame_count = 0
    episode_origin_ns: int | None = None

    try:
        manager.start()
        sources = {
            config.name: (
                OrbbecRGBSource(manager.camera(config.name))
                if config.mode is CameraMode.RGB
                else OrbbecRGBDSource(manager.camera(config.name))
            )
            for config in configs
        }
        master_source = sources[MASTER_CAMERA_NAME]
        writer = V21DatasetWriter(
            WriterConfig(
                root=WRITE_DATASET_ROOT,
                robot_type="hcx_three_camera_write_test",
                fps=int(configs[0].fps or 1),
                joint_count=14,
                cameras=tuple(
                    CameraSpec(
                        f"observation.images.{name}",
                        sources[name].shape,
                    )
                    for name in CAMERA_NAMES
                ),
                image_storage=WRITE_IMAGE_STORAGE,
                quality=WRITE_JPEG_QUALITY,
            )
        )
        episode_index = writer.begin_episode("hcx_three_camera_stream_continuity")
        master_sequence = master_source.latest_sequence()
        started_ns = time.perf_counter_ns()
        report_started_ns = started_ns
        latest_master = manager.camera(MASTER_CAMERA_NAME).get_frame()
        if latest_master is not None:
            last_master_capture_ns = latest_master.capture_monotonic_ns

        fps = configs[0].fps
        duration = "持续到 Ctrl+C" if TEST_DURATION_S is None else f"{TEST_DURATION_S:.1f}s"
        print("=" * 88)
        print("    HCX 三路 Orbbec 主相机驱动连续性测试")
        print("=" * 88)
        print(f"  相机: {', '.join(CAMERA_NAMES)}；主相机: {MASTER_CAMERA_NAME}")
        print(f"  配置帧率: {fps} FPS；测试时长: {duration}")
        print("  每张 head 新帧触发一次左右手因果配对；不连接机器人。")
        print(
            f"  JPEG 写盘: quality={WRITE_JPEG_QUALITY}；"
            f"测试 episode={episode_index:06d}；目录={WRITE_DATASET_ROOT}"
        )
        print("  selected 是参与记录候选的帧；producer 是相机源帧率；pair-delta 是手部帧早于 head 的时间。")

        while True:
            now_ns = time.perf_counter_ns()
            assert started_ns is not None
            if (
                TEST_DURATION_S is not None
                and now_ns - started_ns >= int(TEST_DURATION_S * 1_000_000_000)
            ):
                break

            failure = _failed_camera_message(manager)
            if failure is not None:
                raise RuntimeError(f"相机进入 FAILED: {failure}")
            if (
                last_master_capture_ns is not None
                and now_ns - last_master_capture_ns
                > int(MASTER_STALE_TIMEOUT_S * 1_000_000_000)
            ):
                raise RuntimeError(
                    f"head 已 {((now_ns - last_master_capture_ns) / 1_000_000_000):.3f}s "
                    "没有新帧"
                )

            item = master_source.next_frame_after(master_sequence)
            if item is None:
                time.sleep(POLL_INTERVAL_S)
            else:
                master_sequence, master_frame = item
                master_capture_ns = _capture_monotonic_ns(master_frame)
                if master_capture_ns is None:
                    raise RuntimeError("head 帧缺少主机捕获时间戳")
                last_master_capture_ns = master_capture_ns
                master_ok = total_stats[MASTER_CAMERA_NAME].observe(
                    master_frame, master_capture_ns
                )
                window_stats[MASTER_CAMERA_NAME].observe(master_frame, master_capture_ns)
                row_complete = master_ok
                row_frames: dict[str, Frame] = {MASTER_CAMERA_NAME: master_frame}
                for name in CAMERA_NAMES:
                    if name == MASTER_CAMERA_NAME:
                        continue
                    paired = sources[name].frame_at_or_before(master_capture_ns)
                    total_ok = total_stats[name].observe(paired, master_capture_ns)
                    window_stats[name].observe(paired, master_capture_ns)
                    row_complete = row_complete and total_ok
                    if paired is not None:
                        row_frames[name] = paired
                if row_complete:
                    if episode_origin_ns is None:
                        episode_origin_ns = master_capture_ns
                    assert writer is not None
                    write_started_ns = time.perf_counter_ns()
                    writer.append_frame(
                        RecordedFrame(
                            state=np.zeros(14, dtype=float),
                            action=np.zeros(14, dtype=float),
                            timestamp_s=(
                                master_capture_ns - episode_origin_ns
                            )
                            / 1_000_000_000,
                            cameras={
                                f"observation.images.{name}": _recorded_camera_frame(
                                    row_frames[name], episode_origin_ns
                                )
                                for name in CAMERA_NAMES
                            },
                            audit={
                                "diagnostic_only": True,
                                "master_capture_monotonic_ns": master_capture_ns,
                            },
                        )
                    )
                    write_elapsed_ns = time.perf_counter_ns() - write_started_ns
                    total_write_stats.observe(write_elapsed_ns)
                    window_write_stats.observe(write_elapsed_ns)
                    writer_frame_count += 1
                    total_complete += 1
                    window_complete += 1
                else:
                    total_incomplete += 1
                    window_incomplete += 1

            now_ns = time.perf_counter_ns()
            if now_ns - report_started_ns >= int(REPORT_INTERVAL_S * 1_000_000_000):
                elapsed_s = (now_ns - report_started_ns) / 1_000_000_000
                _print_report(
                    manager,
                    window_stats,
                    elapsed_s,
                    window_complete,
                    window_incomplete,
                    window_write_stats,
                )
                window_stats = _new_stats()
                window_write_stats = WriteStats()
                window_complete = 0
                window_incomplete = 0
                report_started_ns = now_ns
    except KeyboardInterrupt:
        print("\n[STOP] 用户停止三相机连续性测试")
    except BaseException as exc:
        exit_code = 1
        print(f"[FAULT] 三相机连续性测试失败: {exc}")
        now_ns = time.perf_counter_ns()
        for name in CAMERA_NAMES:
            print(f"  {name}: {_camera_state_text(manager, name, total_stats[name], now_ns)}")
    finally:
        if started_ns is not None:
            elapsed_s = max(
                (time.perf_counter_ns() - started_ns) / 1_000_000_000,
                1e-9,
            )
            print("=" * 88)
            print(f"    最终结果：运行 {elapsed_s:.2f}s")
            print(f"  完整记录候选={total_complete}；不完整记录候选={total_incomplete}")
            for name in CAMERA_NAMES:
                print(f"  {_stats_text(name, total_stats[name], elapsed_s)}")
            print(f"  JPEG 写盘调用: {total_write_stats.text()}")
            if exit_code == 0 and total_incomplete == 0:
                print("  [PASS] 测试期间未发现相机断流、无效图像或主相机配对缺失。")
            elif exit_code == 0:
                print("  [WARN] 相机未进入 FAILED，但测试期间存在不完整的主相机驱动记录。")
        if writer is not None and writer.active:
            if writer_frame_count:
                finalize_started_ns = time.perf_counter_ns()
                try:
                    episode_index = writer.finish_episode()
                    finalize_s = (
                        time.perf_counter_ns() - finalize_started_ns
                    ) / 1_000_000_000
                    print(
                        f"  [INFO] JPEG 测试 episode {episode_index:06d} 已提交，"
                        f"收尾耗时={finalize_s:.2f}s；目录={WRITE_DATASET_ROOT}"
                    )
                except Exception as exc:
                    exit_code = 1
                    print(f"  [FAULT] JPEG 测试 episode 提交失败: {exc}")
            else:
                writer.discard_episode()
        manager.stop()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
