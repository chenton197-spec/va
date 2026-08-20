#!/usr/bin/env python3
"""独立读取两台 Orbbec 的采集器 RGB-D 帧。"""

from __future__ import annotations

import time

import cv2

from leobot_scripts import OrbbecRGBDSource, load_recording_config
from orbbec_sdk import (
    CameraMode,
    CameraStatus,
    OrbbecCameraConfig,
    OrbbecManager,
    load_orbbec_camera_configs,
)


SHOW_WINDOWS = True
PRINT_EVERY_N_FRAMES = 30


def _validate_configuration(sample_fps: int) -> tuple[OrbbecCameraConfig, ...]:
    configs = load_orbbec_camera_configs()
    if len(configs) != 2:
        raise ValueError("在 teleop.yaml 的 orbbec.cameras 中配置两台 RGB-D 相机")
    if any(config.mode is not CameraMode.RGBD for config in configs):
        raise ValueError("此测试要求两台相机都使用 RGB-D 模式")
    if any(config.fps != sample_fps for config in configs):
        raise ValueError("orbbec.cameras 的 fps 必须与 recording.fps 一致")
    if PRINT_EVERY_N_FRAMES <= 0:
        raise ValueError("PRINT_EVERY_N_FRAMES 必须为正数")
    return configs


def main() -> None:
    """显示并打印直接交给采集器的 RGB 与原始 Y16 深度帧。"""

    recording = load_recording_config()
    configs = _validate_configuration(recording.fps)
    manager = OrbbecManager(configs)
    frame_counts = {config.name: 0 for config in configs}
    period_s = 1.0 / recording.fps
    next_deadline = time.perf_counter()

    try:
        manager.start()
        sources = {
            config.name: OrbbecRGBDSource(manager.camera(config.name)) for config in configs
        }
        if SHOW_WINDOWS:
            for config in configs:
                cv2.namedWindow(f"{config.name} RGB", cv2.WINDOW_AUTOSIZE)
                cv2.namedWindow(f"{config.name} Depth", cv2.WINDOW_AUTOSIZE)
        print(f"[INFO] 开始读取 RGB-D 帧，频率 {recording.fps} Hz；按 q、Esc 或 Ctrl+C 退出")

        while True:
            for config in configs:
                camera = manager.camera(config.name)
                if camera.status is CameraStatus.FAILED:
                    raise RuntimeError(f"{config.name}: {camera.last_error or 'Camera capture failed'}")

                frame = sources[config.name].latest_frame()
                if frame is None:
                    continue
                frame_index = frame_counts[config.name]
                frame_counts[config.name] += 1
                if frame_index % PRINT_EVERY_N_FRAMES == 0:
                    print(
                        f"{config.name}: rgb={frame.rgb.shape}/{frame.rgb.dtype}, "
                        f"depth={frame.depth.raw.shape}/{frame.depth.raw.dtype}, "
                        f"rgb_source_frame_index={frame.rgb_source_frame_index}, "
                        f"depth_source_frame_index={frame.depth.source_frame_index}"
                    )
                if SHOW_WINDOWS:
                    cv2.imshow(
                        f"{config.name} RGB",
                        cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR),
                    )
                    cv2.imshow(f"{config.name} Depth", frame.depth.raw)

            if SHOW_WINDOWS:
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    break

            next_deadline += period_s
            delay_s = next_deadline - time.perf_counter()
            if delay_s > 0.0:
                time.sleep(delay_s)
            else:
                next_deadline = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[STOP] 停止读取 RGB-D 帧")
    finally:
        manager.stop()
        if SHOW_WINDOWS:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
