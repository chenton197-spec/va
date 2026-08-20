#!/usr/bin/env python3
"""按 hand 新帧驱动方式测量奥比中光采集帧率。"""

from __future__ import annotations

import time
from dataclasses import dataclass

from leobot_scripts import CameraFrame, OrbbecRGBDSource, OrbbecRGBSource, RGBDFrame, load_recording_config
from orbbec_sdk import (
    CameraMode,
    CameraStatus,
    OrbbecCameraConfig,
    OrbbecManager,
    load_orbbec_camera_configs,
)


# 通过变量选择测试相机；hand 是采集的主相机，head 与 hand 时间戳配对。
CAMERA_NAMES: tuple[str, ...] = ("hand", "head")
MASTER_CAMERA_NAME = "hand"
# None 表示持续运行到 Ctrl+C；建议先测试 20 秒。
MEASUREMENT_DURATION_S: float | None = 20.0
# 每隔多少秒打印一次窗口统计。
REPORT_INTERVAL_S = 5.0
# 没有新 hand 帧时的轮询间隔；相机帧本身仍由 SDK 后台线程读取。
POLL_INTERVAL_S = 0.001


@dataclass
class _FrameRateStats:
    selected_frames: int = 0
    empty_selections: int = 0
    first_source_index: int | None = None
    last_source_index: int | None = None
    first_source_timestamp_ns: int | None = None
    last_source_timestamp_ns: int | None = None
    skipped_source_frames: int = 0
    reused_source_frames: int = 0
    capture_age_sum_ns: int = 0
    capture_age_count: int = 0
    capture_age_max_ns: int = 0

    def observe(self, frame: CameraFrame | RGBDFrame | None, target_monotonic_ns: int) -> None:
        if frame is None:
            self.empty_selections += 1
            return
        self.selected_frames += 1
        index = _source_frame_index(frame)
        timestamp_ns = _source_timestamp_ns(frame)
        if index is not None:
            if self.last_source_index is not None:
                if index == self.last_source_index:
                    self.reused_source_frames += 1
                elif index > self.last_source_index + 1:
                    self.skipped_source_frames += index - self.last_source_index - 1
            if self.first_source_index is None:
                self.first_source_index = index
            self.last_source_index = index
        if timestamp_ns is not None:
            if self.first_source_timestamp_ns is None:
                self.first_source_timestamp_ns = timestamp_ns
            self.last_source_timestamp_ns = timestamp_ns
        capture_monotonic_ns = _capture_monotonic_ns(frame)
        if capture_monotonic_ns is not None:
            age_ns = target_monotonic_ns - capture_monotonic_ns
            if age_ns >= 0:
                self.capture_age_sum_ns += age_ns
                self.capture_age_count += 1
                self.capture_age_max_ns = max(self.capture_age_max_ns, age_ns)

    def report(self, camera_name: str, elapsed_s: float, *, role: str) -> None:
        collection_hz = self.selected_frames / elapsed_s if elapsed_s > 0.0 else 0.0
        message = (
            f"[FPS] {camera_name} ({role}): collection={collection_hz:.2f} Hz, "
            f"frames={self.selected_frames}, empty={self.empty_selections}, "
            f"reused={self.reused_source_frames}, source_skipped={self.skipped_source_frames}"
        )
        if (
            self.first_source_index is not None
            and self.last_source_index is not None
            and self.first_source_timestamp_ns is not None
            and self.last_source_timestamp_ns is not None
            and self.last_source_timestamp_ns > self.first_source_timestamp_ns
        ):
            producer_hz = (self.last_source_index - self.first_source_index) / (
                (self.last_source_timestamp_ns - self.first_source_timestamp_ns) / 1_000_000_000
            )
            message += f", producer={producer_hz:.2f} Hz"
        else:
            message += ", producer=insufficient frames"
        if self.capture_age_count:
            average_age_ms = self.capture_age_sum_ns / self.capture_age_count / 1_000_000
            max_age_ms = self.capture_age_max_ns / 1_000_000
            message += f", capture_age(avg/max)={average_age_ms:.2f}/{max_age_ms:.2f} ms"
        else:
            message += ", capture_age=unavailable"
        print(message)


def _capture_monotonic_ns(frame: CameraFrame | RGBDFrame) -> int | None:
    if isinstance(frame, CameraFrame):
        return frame.capture_monotonic_ns
    if frame.rgb_capture_monotonic_ns is not None:
        return frame.rgb_capture_monotonic_ns
    return frame.depth.capture_monotonic_ns


def _source_timestamp_ns(frame: CameraFrame | RGBDFrame) -> int | None:
    if isinstance(frame, CameraFrame):
        return frame.source_timestamp_ns
    if frame.rgb_source_timestamp_ns is not None:
        return frame.rgb_source_timestamp_ns
    return frame.depth.source_timestamp_ns


def _source_frame_index(frame: CameraFrame | RGBDFrame) -> int | None:
    if isinstance(frame, CameraFrame):
        return frame.source_frame_index
    if frame.rgb_source_frame_index is not None:
        return frame.rgb_source_frame_index
    return frame.depth.source_frame_index


def _selected_configs(recording_fps: int) -> tuple[OrbbecCameraConfig, ...]:
    if not CAMERA_NAMES:
        raise ValueError("CAMERA_NAMES 至少指定一台相机")
    if MASTER_CAMERA_NAME not in CAMERA_NAMES:
        raise ValueError("MASTER_CAMERA_NAME 必须在 CAMERA_NAMES 中")
    if MEASUREMENT_DURATION_S is not None and MEASUREMENT_DURATION_S <= 0.0:
        raise ValueError("MEASUREMENT_DURATION_S 必须为正数或 None")
    if REPORT_INTERVAL_S <= 0.0 or POLL_INTERVAL_S <= 0.0:
        raise ValueError("统计和轮询间隔必须为正数")

    by_name = {config.name: config for config in load_orbbec_camera_configs()}
    unknown = set(CAMERA_NAMES) - set(by_name)
    if unknown:
        raise ValueError(f"CAMERA_NAMES 包含未配置相机: {sorted(unknown)}")
    configs = tuple(by_name[name] for name in CAMERA_NAMES)
    if any(config.mode not in {CameraMode.RGB, CameraMode.RGBD} for config in configs):
        raise ValueError("此测试要求所选相机都提供 RGB 流")
    if any(config.fps != recording_fps for config in configs):
        raise ValueError("所选相机的 fps 必须与 recording.fps 一致")
    return configs


def _report(
    stats: dict[str, _FrameRateStats],
    elapsed_s: float,
) -> None:
    for name, value in stats.items():
        role = "master" if name == MASTER_CAMERA_NAME else "paired"
        value.report(name, elapsed_s, role=role)


def main() -> None:
    """统计与 ``CameraProcessDatasetRecorder`` 相同的 hand 驱动选帧路径。"""

    recording = load_recording_config()
    configs = _selected_configs(recording.fps)
    manager = OrbbecManager(configs)
    total_stats = {config.name: _FrameRateStats() for config in configs}
    window_stats = {config.name: _FrameRateStats() for config in configs}
    start_ns: int | None = None
    report_start_ns = 0
    master_sequence = 0

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
        # 忽略启动前已缓冲的帧，和开始 episode 时的采集器保持一致。
        master_sequence = master_source.latest_sequence()
        start_ns = time.perf_counter_ns()
        report_start_ns = start_ns
        names = ", ".join(config.name for config in configs)
        print(
            f"[INFO] hand 驱动采集帧率测试: master={MASTER_CAMERA_NAME}; cameras={names}; "
            f"configured_fps={recording.fps}; "
            f"duration={MEASUREMENT_DURATION_S if MEASUREMENT_DURATION_S is not None else 'until Ctrl+C'} s"
        )
        print(
            "[INFO] 每张新 hand RGB 帧触发一条记录；其他相机按该 hand 捕获时间取不晚于它的最新帧"
        )

        while True:
            now_ns = time.perf_counter_ns()
            assert start_ns is not None
            if (
                MEASUREMENT_DURATION_S is not None
                and now_ns - start_ns >= int(MEASUREMENT_DURATION_S * 1_000_000_000)
            ):
                break
            for config in configs:
                camera = manager.camera(config.name)
                if camera.status is CameraStatus.FAILED:
                    raise RuntimeError(f"{config.name}: {camera.last_error or 'Camera capture failed'}")

            item = master_source.next_frame_after(master_sequence)
            if item is None:
                time.sleep(POLL_INTERVAL_S)
            else:
                master_sequence, master_frame = item
                target_monotonic_ns = _capture_monotonic_ns(master_frame)
                if target_monotonic_ns is None:
                    raise RuntimeError("hand RGB 帧没有主机捕获时间")
                total_stats[MASTER_CAMERA_NAME].observe(master_frame, target_monotonic_ns)
                window_stats[MASTER_CAMERA_NAME].observe(master_frame, target_monotonic_ns)
                for config in configs:
                    if config.name == MASTER_CAMERA_NAME:
                        continue
                    frame = sources[config.name].frame_at_or_before(target_monotonic_ns)
                    total_stats[config.name].observe(frame, target_monotonic_ns)
                    window_stats[config.name].observe(frame, target_monotonic_ns)

            now_ns = time.perf_counter_ns()
            if now_ns - report_start_ns >= int(REPORT_INTERVAL_S * 1_000_000_000):
                elapsed_s = (now_ns - report_start_ns) / 1_000_000_000
                print(f"[INFO] 最近 {elapsed_s:.2f} s")
                _report(window_stats, elapsed_s)
                window_stats = {config.name: _FrameRateStats() for config in configs}
                report_start_ns = now_ns
    except KeyboardInterrupt:
        print("\n[STOP] 用户停止相机帧率测试")
    finally:
        elapsed_s = (
            0.0
            if start_ns is None
            else (time.perf_counter_ns() - start_ns) / 1_000_000_000
        )
        if elapsed_s > 0.0:
            print(f"[INFO] 总计 {elapsed_s:.2f} s")
            _report(total_stats, elapsed_s)
        manager.stop()


if __name__ == "__main__":
    main()
