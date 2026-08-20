#!/usr/bin/env python3
"""按固定帧率回放一个目录中的 PNG 或 JPG 图像序列。"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

# 修改为要回放的 PNG 或 JPG 图像目录。
IMAGE_DIRECTORY = Path(
    "datasets/alicia_fr3/images/chunk-000/observation.images.head/episode_000003"
)
# 以采集时的目标帧率播放。
PLAYBACK_FPS = 30.0
# 播放到最后一张时是否从第一张重新开始。
LOOP_PLAYBACK = True
WINDOW_NAME = "Image Sequence Playback"


def _image_paths(directory: Path) -> list[Path]:
    if PLAYBACK_FPS <= 0.0:
        raise ValueError("PLAYBACK_FPS 必须为正数")
    if not directory.is_dir():
        raise NotADirectoryError(f"找不到图像目录: {directory}")
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise FileNotFoundError(f"目录中没有 PNG 或 JPG 图像: {directory}")
    return paths


def _next_index(index: int, count: int) -> int:
    if index + 1 < count:
        return index + 1
    return 0 if LOOP_PLAYBACK else count - 1


def main() -> None:
    """Display the image sequence as a timed media stream."""

    paths = _image_paths(IMAGE_DIRECTORY)
    frame_period_s = 1.0 / PLAYBACK_FPS
    index = 0
    playing = True
    deadline_s = time.perf_counter()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    print(f"[INFO] 回放 {len(paths)} 张图像: {IMAGE_DIRECTORY}")
    print("[INFO] 空格暂停/继续；a/d 后退/前进一帧；r 重播；q、Esc 退出")

    try:
        while True:
            image = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"无法读取图像: {paths[index]}")
            cv2.imshow(WINDOW_NAME, image)

            if playing:
                deadline_s += frame_period_s
                delay_ms = max(1, round((deadline_s - time.perf_counter()) * 1_000))
            else:
                delay_ms = 30
            key = cv2.waitKey(delay_ms) & 0xFF

            if key in {27, ord("q")}:
                return
            if key == ord(" "):
                playing = not playing
                deadline_s = time.perf_counter()
                continue
            if key == ord("a"):
                index = (index - 1) % len(paths)
                playing = False
                continue
            if key == ord("d"):
                index = _next_index(index, len(paths))
                playing = False
                continue
            if key == ord("r"):
                index = 0
                deadline_s = time.perf_counter()
                continue
            if playing:
                index = _next_index(index, len(paths))
                if index == len(paths) - 1 and not LOOP_PLAYBACK:
                    playing = False
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
