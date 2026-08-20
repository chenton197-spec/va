#!/usr/bin/env python3
"""在 MuJoCo 中复现采集的双臂轨迹，并同步播放三路相机图像。"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from teleop_sdk.adapters import MujocoFollower, MujocoSimulation

# 修改为需要回放的 episode Parquet 文件，不使用命令行参数。
PARQUET_PATH = Path(
    "datasets/openarm_hcx_dual_arm/data/chunk-000/episode_000000.parquet"
)
# MuJoCo 使用的双臂机器人模型。
URDF_FILENAME = "CASBOTWL12_WL12P1.urdf"
# observation.state 是采集到的真实从臂反馈；action 是当时发送的目标角度。
JOINT_TRAJECTORY_COLUMN = "observation.state"
# True 表示播放结束后从第一帧重新播放；False 表示播放一次后退出。
LOOP_PLAYBACK = False
# 1.0 为原速，2.0 为两倍速，0.5 为半速。
PLAYBACK_SPEED = 1.0
# 统一控制头部、左手和右手三个 OpenCV 窗口；False 时只回放 MuJoCo。
DISPLAY_CAMERA_WINDOWS = False
# 与 MuJoCo 遥操示例相同：启用棋盘格地板、渐变天空和定向光。
ENABLE_CHECKERBOARD_FLOOR = True
# 地板高度；机器人模型最低点会自动对齐到该高度。
CHECKERBOARD_FLOOR_Z_M = -0.5

CAMERA_COLUMNS = (
    "observation.images.head",
    "observation.images.left_hand",
    "observation.images.right_hand",
)
CAMERA_WINDOWS = {
    "observation.images.head": "Head Camera",
    "observation.images.left_hand": "Left Hand Camera",
    "observation.images.right_hand": "Right Hand Camera",
}

LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
)
RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)


@dataclass(frozen=True)
class PlaybackFrame:
    """一条同步的关节状态和三路图像记录。"""

    timestamp_s: float
    left_angles_deg: np.ndarray
    right_angles_deg: np.ndarray
    image_paths: dict[str, Path]


def _dataset_root(parquet_path: Path) -> Path:
    """从标准 data/chunk-xxx/episode_xxx.parquet 布局解析数据集根目录。"""

    resolved = parquet_path.expanduser().resolve()
    if resolved.parent.name.startswith("chunk-") and resolved.parent.parent.name == "data":
        return resolved.parent.parent.parent
    raise ValueError(
        "Parquet 路径必须采用 <dataset>/data/chunk-xxx/episode_xxxxxx.parquet 布局"
    )


def _image_path(value: Any, dataset_root: Path, column_name: str) -> Path:
    """解析 Parquet 图像结构中的 path 字段。"""

    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError(f"{column_name} 必须包含字符串 path 字段")
    path = Path(value["path"]).expanduser()
    return path if path.is_absolute() else dataset_root / path


def _validate_timestamps(timestamps: Sequence[Any]) -> list[float]:
    """校验用于原速调度的 episode 相对时间戳。"""

    values = [float(value) for value in timestamps]
    if not values or not np.isfinite(values).all():
        raise ValueError("Parquet timestamp 为空或包含无效数值")
    if any(later < earlier for earlier, later in zip(values, values[1:])):
        raise ValueError("Parquet timestamp 必须单调不减")
    origin = values[0]
    return [value - origin for value in values]


def load_playback_frames(parquet_path: Path) -> list[PlaybackFrame]:
    """一次读取 episode 的轻量元数据，图像在播放到对应帧时才从磁盘读取。"""

    path = parquet_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到 Parquet 文件: {path}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("回放 Parquet 需要安装 pyarrow") from exc

    required = ("timestamp", JOINT_TRAJECTORY_COLUMN, *CAMERA_COLUMNS)
    schema_names = set(pq.read_schema(path).names)
    missing = [name for name in required if name not in schema_names]
    if missing:
        raise ValueError(f"Parquet 缺少回放字段: {', '.join(missing)}")

    table = pq.read_table(path, columns=list(required))
    if table.num_rows == 0:
        raise ValueError("Parquet 中没有可回放帧")
    timestamps = _validate_timestamps(table.column("timestamp").to_pylist())
    joint_rows = table.column(JOINT_TRAJECTORY_COLUMN).to_pylist()
    camera_rows = {
        name: table.column(name).to_pylist()
        for name in CAMERA_COLUMNS
    }
    dataset_root = _dataset_root(path)
    frames: list[PlaybackFrame] = []
    missing_images: list[Path] = []
    for index, (timestamp_s, joint_row) in enumerate(zip(timestamps, joint_rows)):
        angles = np.asarray(joint_row, dtype=float)
        if angles.shape != (14,) or not np.isfinite(angles).all():
            raise ValueError(
                f"第 {index} 帧 {JOINT_TRAJECTORY_COLUMN} 必须是 14 个有效角度"
            )
        image_paths = {
            name: _image_path(camera_rows[name][index], dataset_root, name)
            for name in CAMERA_COLUMNS
        }
        missing_images.extend(path for path in image_paths.values() if not path.is_file())
        frames.append(
            PlaybackFrame(
                timestamp_s=timestamp_s,
                left_angles_deg=angles[:7].copy(),
                right_angles_deg=angles[7:].copy(),
                image_paths=image_paths,
            )
        )
    if missing_images:
        preview = ", ".join(str(path) for path in missing_images[:3])
        suffix = "" if len(missing_images) <= 3 else f" 等 {len(missing_images)} 个文件"
        raise FileNotFoundError(f"Parquet 引用的图像不存在: {preview}{suffix}")
    return frames


def _show_images(cv2: Any, frame: PlaybackFrame) -> None:
    """从磁盘读取并显示当前记录对应的三路图像。"""

    for column_name in CAMERA_COLUMNS:
        path = frame.image_paths[column_name]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV 无法读取图像: {path}")
        cv2.imshow(CAMERA_WINDOWS[column_name], image)


def _send_frame(
    left_follower: MujocoFollower,
    right_follower: MujocoFollower,
    frame: PlaybackFrame,
) -> None:
    """将一条采集反馈同时写入共享 MuJoCo 双臂场景。"""

    if not left_follower.send_joint_angles_deg(frame.left_angles_deg, 0.001):
        raise RuntimeError("MuJoCo 左臂拒绝轨迹帧")
    if not right_follower.send_joint_angles_deg(frame.right_angles_deg, 0.001):
        raise RuntimeError("MuJoCo 右臂拒绝轨迹帧")


def main() -> int:
    """按采集时间轴同步回放双臂状态和三路 JPG 图像。"""

    if not np.isfinite(PLAYBACK_SPEED) or PLAYBACK_SPEED <= 0.0:
        print("[ERROR] PLAYBACK_SPEED 必须为正数")
        return 2
    cv2: Any | None = None
    if DISPLAY_CAMERA_WINDOWS:
        try:
            import cv2 as imported_cv2
        except ImportError:
            print("[ERROR] 三路图像回放需要安装 opencv-python")
            return 2
        cv2 = imported_cv2

    try:
        frames = load_playback_frames(PARQUET_PATH)
    except Exception as exc:
        print(f"[ERROR] 无法加载回放数据: {exc}")
        return 2

    project_root = Path(__file__).resolve().parents[1]
    simulation = MujocoSimulation(
        project_root / "simulation" / "urdfs" / URDF_FILENAME
    )
    if ENABLE_CHECKERBOARD_FLOOR:
        simulation.set_mjcf_environment(
            floor_z_m=CHECKERBOARD_FLOOR_Z_M,
            align_model_lowest_point_to_floor=True,
        )
    left_follower = MujocoFollower(simulation, LEFT_ARM_JOINT_NAMES)
    right_follower = MujocoFollower(simulation, RIGHT_ARM_JOINT_NAMES)
    connected: list[MujocoFollower] = []

    try:
        left_follower.connect()
        connected.append(left_follower)
        right_follower.connect()
        connected.append(right_follower)
        if not left_follower.start_servo() or not right_follower.start_servo():
            raise RuntimeError("MuJoCo 双臂无法进入伺服状态")

        simulation.open_viewer()
        if cv2 is not None:
            for window_name in CAMERA_WINDOWS.values():
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        duration_s = frames[-1].timestamp_s
        average_hz = (len(frames) - 1) / duration_s if duration_s > 0.0 else 0.0
        print("=" * 72)
        print("    OpenArm/HCX 数据集 -> MuJoCo 双臂与三相机同步回放")
        print("=" * 72)
        print(f"  Parquet: {PARQUET_PATH}")
        print(f"  轨迹字段: {JOINT_TRAJECTORY_COLUMN}")
        print(
            f"  天空、光影与棋盘格地板: "
            f"{'已启用' if ENABLE_CHECKERBOARD_FLOOR else '已关闭'}"
        )
        print(
            f"  三路 OpenCV 相机窗口: "
            f"{'已启用' if DISPLAY_CAMERA_WINDOWS else '已关闭'}"
        )
        print(
            f"  帧数: {len(frames)}；时长: {duration_s:.3f} s；"
            f"平均帧率: {average_hz:.2f} Hz；速度: {PLAYBACK_SPEED:.2f}x"
        )
        if DISPLAY_CAMERA_WINDOWS:
            print("  空格暂停/继续；q 或 Esc 退出。")
        else:
            print("  关闭 MuJoCo viewer 或按 Ctrl+C 退出。")
        print("-" * 72)

        while simulation.viewer_is_running:
            playback_started_s = time.perf_counter()
            paused_total_s = 0.0
            for frame_index, frame in enumerate(frames):
                target_s = playback_started_s + paused_total_s + frame.timestamp_s / PLAYBACK_SPEED
                while simulation.viewer_is_running:
                    remaining_s = target_s - time.perf_counter()
                    if remaining_s <= 0.0:
                        break
                    if cv2 is None:
                        simulation.sync_viewer()
                        time.sleep(min(remaining_s, 0.01))
                        continue
                    key = cv2.waitKey(
                        max(1, min(10, round(remaining_s * 1_000)))
                    ) & 0xFF
                    if key in {27, ord("q")}:
                        return 0
                    if key == ord(" "):
                        pause_started_s = time.perf_counter()
                        while simulation.viewer_is_running:
                            simulation.sync_viewer()
                            pause_key = cv2.waitKey(20) & 0xFF
                            if pause_key in {27, ord("q")}:
                                return 0
                            if pause_key == ord(" "):
                                break
                        pause_duration_s = time.perf_counter() - pause_started_s
                        paused_total_s += pause_duration_s
                        target_s += pause_duration_s
                if not simulation.viewer_is_running:
                    return 0

                _send_frame(left_follower, right_follower, frame)
                if cv2 is not None:
                    _show_images(cv2, frame)
                simulation.sync_viewer()
                if cv2 is not None:
                    key = cv2.waitKey(1) & 0xFF
                    if key in {27, ord("q")}:
                        return 0
                if frame_index + 1 == len(frames):
                    print("[INFO] episode 回放完成")

            if not LOOP_PLAYBACK:
                return 0
    except KeyboardInterrupt:
        print("\n[STOP] 收到退出请求")
        return 0
    except Exception as exc:
        print(f"[ERROR] 回放失败: {exc}")
        return 1
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()
        for follower in reversed(connected):
            follower.stop_servo()
            follower.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
