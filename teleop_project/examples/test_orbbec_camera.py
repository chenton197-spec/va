#!/usr/bin/env python3
"""直接预览两台 Orbbec 相机的 RGB 和原始深度图。"""

from __future__ import annotations

import cv2

from orbbec_sdk import (
    CameraMode,
    CameraStatus,
    OrbbecCamera,
    OrbbecCameraConfig,
    OrbbecManager,
    load_orbbec_camera_configs,
)


def _rgb_window_name(camera: OrbbecCamera) -> str:
    return f"{camera.config.name} RGB"


def _depth_window_name(camera: OrbbecCamera) -> str:
    return f"{camera.config.name} Depth"


def _show_current_frames(camera: OrbbecCamera) -> None:
    """Show the current raw images without color mapping or depth normalization."""

    frame = camera.get_frame()
    if frame is None:
        return
    if frame.rgb is not None:
        cv2.imshow(_rgb_window_name(camera), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
    if frame.depth is not None:
        cv2.imshow(_depth_window_name(camera), frame.depth)


def _validate_configuration(camera_configs: tuple[OrbbecCameraConfig, ...]) -> None:
    if len(camera_configs) != 2:
        raise ValueError("在 teleop.yaml 的 orbbec.cameras 中配置两台相机")


def main() -> None:
    """Start two cameras and show their current RGB and depth images."""

    camera_configs = load_orbbec_camera_configs()
    _validate_configuration(camera_configs)
    manager = OrbbecManager(camera_configs)

    try:
        manager.start()
        cameras = [manager.camera(config.name) for config in camera_configs]
        for camera in cameras:
            if camera.config.mode in {CameraMode.RGB, CameraMode.RGBD}:
                cv2.namedWindow(_rgb_window_name(camera), cv2.WINDOW_AUTOSIZE)
            if camera.config.mode in {CameraMode.DEPTH, CameraMode.RGBD}:
                cv2.namedWindow(_depth_window_name(camera), cv2.WINDOW_AUTOSIZE)
        print("按 q 或 Esc 关闭预览窗口。")

        while True:
            for camera in cameras:
                _show_current_frames(camera)
                if camera.status is CameraStatus.FAILED:
                    raise RuntimeError(
                        f"{camera.config.name}: {camera.last_error or 'Camera capture failed'}"
                    )

            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                break
    finally:
        manager.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
