#!/usr/bin/env python3
"""回放 openarm_hcx_dual_arm_record.py 采集的 HCX 双臂轨迹（move_joints）。

默认读取 ``teleop_project/datasets/to_init`` 中的 Parquet，按帧顺序用 ``RobotClient.arm.move_joints``
下发 ``action``（14 关节角）与左右夹爪目标，到位确认方式与 ``run_w2/run.py`` 一致。
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

VA_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPLAY_YAML = SCRIPT_DIR / "replay.yaml"
TELEOP_ROOT = VA_ROOT / "teleop_project"

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

if not TELEOP_ROOT.is_dir():
    raise FileNotFoundError(f"找不到 in-repo teleop_project: {TELEOP_ROOT}")
if str(TELEOP_ROOT) not in sys.path:
    sys.path.insert(0, str(TELEOP_ROOT))

from teleop_sdk.adapters.gloria_m import GloriaMGripperFollower  # noqa: E402
from teleop_sdk.config import load_runtime_config  # noqa: E402


def _import_robot_client() -> Any:
    from hcx_sdk import RobotClient

    return RobotClient


@dataclass(frozen=True)
class PlaybackFrame:
    timestamp_s: float
    left_angles_deg: np.ndarray
    right_angles_deg: np.ndarray
    left_gripper: float
    right_gripper: float
    image_paths: dict[str, Path]


@dataclass
class HardwareBundle:
    hcx_client: Any | None = None
    left_arm: Any | None = None
    right_arm: Any | None = None
    left_gripper: GloriaMGripperFollower | None = None
    right_gripper: GloriaMGripperFollower | None = None
    left_gripper_loop: "BackgroundGripperLoop | None" = None
    right_gripper_loop: "BackgroundGripperLoop | None" = None


class BackgroundGripperLoop:
    def __init__(self, gripper: Any, *, rate_hz: float = 30.0):
        if rate_hz <= 0.0:
            raise ValueError("gripper rate_hz 必须 > 0")
        self._gripper = gripper
        self._period_s = 1.0 / float(rate_hz)
        self._lock = threading.Lock()
        self._target = 1.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set_opening(self, opening: float) -> None:
        with self._lock:
            self._target = float(np.clip(opening, 0.0, 1.0))

    def start(self, initial_opening: float | None = None) -> None:
        if self._thread is not None:
            raise RuntimeError("夹爪后台循环已在运行")
        if initial_opening is not None:
            self.set_opening(initial_opening)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, *, join_timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=join_timeout_s)
        self._thread = None

    def _run(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            with self._lock:
                target = self._target
            try:
                _ = bool(self._gripper.send_normalized(target))
            except Exception:
                pass
            next_t += self._period_s
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0.0:
                if self._stop.wait(timeout=sleep_s):
                    break
            else:
                next_t = time.perf_counter()


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.expanduser().resolve()


def _optional_float(value: Any, default: float | None) -> float | None:
    if value is None:
        return default
    return float(value)


def _load_replay_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到回放配置: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"回放配置根节点必须是映射: {path}")

    dataset_root = _resolve_path(
        data.get("dataset_root", TELEOP_ROOT / "datasets" / "to_init"), base=VA_ROOT
    )
    teleop_yaml = _resolve_path(
        data.get("teleop_yaml", TELEOP_ROOT / "teleop.yaml"),
        base=VA_ROOT,
    )
    playback_speed = float(data.get("playback_speed", 1.0))
    if not math.isfinite(playback_speed) or playback_speed <= 0.0:
        raise ValueError("playback_speed 必须为正数")

    joint_column = str(data.get("joint_column", "action"))
    if joint_column not in {"action", "observation.state"}:
        raise ValueError("joint_column 只能是 action 或 observation.state")

    move = data.get("move_joints", {}) or {}
    if not isinstance(move, dict):
        raise ValueError("move_joints 必须是映射")

    return {
        "dataset_root": dataset_root,
        "episode_index": int(data.get("episode_index", 0)),
        "teleop_yaml": teleop_yaml,
        "joint_column": joint_column,
        "playback_speed": playback_speed,
        "loop": bool(data.get("loop", False)),
        "display_cameras": bool(data.get("display_cameras", False)),
        "dry_run": bool(data.get("dry_run", False)),
        "connect_grippers": bool(data.get("connect_grippers", True)),
        "pace_by_timestamp": bool(data.get("pace_by_timestamp", False)),
        "move_speed_ratio": _optional_float(move.get("speed_ratio", 0.1), 0.1),
        "move_acceleration_seconds": _optional_float(
            move.get("acceleration_seconds", 0.5), 0.5
        ),
        "move_deceleration_seconds": _optional_float(
            move.get("deceleration_seconds", 0.5), 0.5
        ),
        "move_feedback_confirm_timeout_s": float(
            move.get("feedback_confirm_timeout_s", 30.0)
        ),
        "move_feedback_confirm_poll_interval_s": float(
            move.get("feedback_confirm_poll_interval_s", 0.05)
        ),
        "move_angle_tolerance_deg": float(move.get("angle_tolerance_deg", 0.5)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="回放 openarm_hcx_dual_arm_record 采集的 HCX 双臂轨迹（move_joints）"
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_REPLAY_YAML))
    parser.add_argument("--dataset", type=str, default=None, help="覆盖 dataset_root")
    parser.add_argument("--episode", type=int, default=None, help="覆盖 episode_index")
    parser.add_argument("--speed", type=float, default=None, help="覆盖 playback_speed")
    parser.add_argument(
        "--joint-column",
        choices=("action", "observation.state"),
        default=None,
        help="覆盖轨迹字段",
    )
    parser.add_argument("--loop", action="store_true", default=None)
    parser.add_argument("--display-cameras", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument(
        "--no-grippers",
        action="store_true",
        default=None,
        help="不连接 Gloria-M 夹爪，只回放双臂",
    )
    return parser.parse_args()


def _episode_parquet(dataset_root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    path = (
        dataset_root
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(f"找不到 episode Parquet: {path}")
    return path


def _dataset_root_from_parquet(parquet_path: Path) -> Path:
    resolved = parquet_path.expanduser().resolve()
    if resolved.parent.name.startswith("chunk-") and resolved.parent.parent.name == "data":
        return resolved.parent.parent.parent
    raise ValueError(
        "Parquet 路径必须采用 <dataset>/data/chunk-xxx/episode_xxxxxx.parquet 布局"
    )


def _image_path(value: Any, dataset_root: Path, column_name: str) -> Path:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError(f"{column_name} 必须包含字符串 path 字段")
    path = Path(value["path"]).expanduser()
    return path if path.is_absolute() else dataset_root / path


def _validate_timestamps(timestamps: Sequence[Any]) -> list[float]:
    values = [float(value) for value in timestamps]
    if not values or not np.isfinite(values).all():
        raise ValueError("Parquet timestamp 为空或包含无效数值")
    if any(later < earlier for earlier, later in zip(values, values[1:])):
        raise ValueError("Parquet timestamp 必须单调不减")
    origin = values[0]
    return [value - origin for value in values]


def _scalar_series(column: Any, *, name: str, rows: int) -> list[float]:
    values = column.to_pylist()
    if len(values) != rows:
        raise ValueError(f"{name} 行数与 timestamp 不一致")
    out: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"第 {index} 帧 {name} 必须是标量")
            value = value[0]
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"第 {index} 帧 {name} 无效: {value}")
        out.append(float(np.clip(numeric, 0.0, 1.0)))
    return out


def load_playback_frames(
    parquet_path: Path,
    *,
    joint_column: str,
    require_images: bool,
) -> list[PlaybackFrame]:
    path = parquet_path.expanduser().resolve()
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("回放 Parquet 需要安装 pyarrow") from exc

    required = [
        "timestamp",
        joint_column,
        "action.left_gripper",
        "action.right_gripper",
    ]
    if require_images:
        required.extend(CAMERA_COLUMNS)

    schema_names = set(pq.read_schema(path).names)
    missing = [name for name in required if name not in schema_names]
    if missing:
        raise ValueError(f"Parquet 缺少回放字段: {', '.join(missing)}")

    table = pq.read_table(path, columns=required)
    if table.num_rows == 0:
        raise ValueError("Parquet 中没有可回放帧")

    timestamps = _validate_timestamps(table.column("timestamp").to_pylist())
    joint_rows = table.column(joint_column).to_pylist()
    left_grippers = _scalar_series(
        table.column("action.left_gripper"),
        name="action.left_gripper",
        rows=table.num_rows,
    )
    right_grippers = _scalar_series(
        table.column("action.right_gripper"),
        name="action.right_gripper",
        rows=table.num_rows,
    )
    camera_rows = (
        {name: table.column(name).to_pylist() for name in CAMERA_COLUMNS}
        if require_images
        else {}
    )
    dataset_root = _dataset_root_from_parquet(path)

    frames: list[PlaybackFrame] = []
    missing_images: list[Path] = []
    for index, (timestamp_s, joint_row) in enumerate(zip(timestamps, joint_rows)):
        angles = np.asarray(joint_row, dtype=float)
        if angles.shape != (14,) or not np.isfinite(angles).all():
            raise ValueError(f"第 {index} 帧 {joint_column} 必须是 14 个有效角度")
        image_paths: dict[str, Path] = {}
        if require_images:
            image_paths = {
                name: _image_path(camera_rows[name][index], dataset_root, name)
                for name in CAMERA_COLUMNS
            }
            missing_images.extend(
                path for path in image_paths.values() if not path.is_file()
            )
        frames.append(
            PlaybackFrame(
                timestamp_s=timestamp_s,
                left_angles_deg=angles[:7].copy(),
                right_angles_deg=angles[7:].copy(),
                left_gripper=left_grippers[index],
                right_gripper=right_grippers[index],
                image_paths=image_paths,
            )
        )
    if missing_images:
        preview = ", ".join(str(path) for path in missing_images[:3])
        suffix = "" if len(missing_images) <= 3 else f" 等 {len(missing_images)} 个文件"
        raise FileNotFoundError(f"Parquet 引用的图像不存在: {preview}{suffix}")
    return frames


def _connect_hcx_arms(teleop_yaml: Path) -> tuple[Any, Any, Any]:
    RobotClient = _import_robot_client()
    runtime = load_runtime_config(teleop_yaml)
    h = runtime.hcx
    client = RobotClient(h.local_ip, h.remote_ip, h.port)
    client.connect(timeout_s=h.connect_timeout_s)
    time.sleep(float(h.controller_initialization_wait_s))

    if h.auto_detach_hmi and not client.hmi_detached:
        client.detach_hmi()
    if h.auto_clear_alarms:
        for _ in range(int(h.alarm_clear_retry_count) + 1):
            if not client.active_alarms:
                break
            client.clear_alarms()
            time.sleep(float(h.alarm_clear_retry_interval_s))
    if client.active_alarms:
        raise RuntimeError(f"HCX 活动报警未清除: {client.active_alarms}")
    if not client.soft_emergency_stop_normal:
        raise RuntimeError("HCX soft emergency stop 非正常状态")

    for master in h.ethercat_master_indices:
        if not client.ethercat_master_operational(master):
            raise RuntimeError(f"HCX EtherCAT 主站未 OP: {master}")

    if h.auto_enable and not client.global_enabled:
        client.set_global_enable(True)
    if not client.global_enabled:
        raise RuntimeError("HCX global enable=false，请现场使能或开启 hcx.auto_enable")

    left_arm = client.arm(h.left_robot_id)
    right_arm = client.arm(h.right_robot_id)
    if h.auto_enable:
        if not left_arm.enabled:
            left_arm.set_enabled(True)
        if not right_arm.enabled:
            right_arm.set_enabled(True)
    if not left_arm.enabled or not right_arm.enabled:
        raise RuntimeError("HCX 左右臂未使能，请现场使能或开启 hcx.auto_enable")
    return client, left_arm, right_arm


def _connect_grippers(
    teleop_yaml: Path,
) -> tuple[GloriaMGripperFollower | None, GloriaMGripperFollower | None, float]:
    runtime = load_runtime_config(teleop_yaml)
    cfg = runtime.gloria_m_dual
    rate_hz = float(cfg.rate_hz)
    left = None
    right = None
    if bool(cfg.left.enabled):
        left = GloriaMGripperFollower(cfg.side_config("left"))
        left.connect()
    if bool(cfg.right.enabled):
        right = GloriaMGripperFollower(cfg.side_config("right"))
        right.connect()
    return left, right, rate_hz


def _confirm_targets_by_feedback(
    hw: HardwareBundle,
    left_target: list[float],
    right_target: list[float],
    *,
    timeout_s: float,
    poll_interval_s: float,
    angle_tolerance_deg: float,
) -> None:
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("move_joints.feedback_confirm_timeout_s 必须是正的有限秒数")
    if not math.isfinite(poll_interval_s) or poll_interval_s <= 0.0:
        raise ValueError(
            "move_joints.feedback_confirm_poll_interval_s 必须是正的有限秒数"
        )
    if not math.isfinite(angle_tolerance_deg) or angle_tolerance_deg <= 0.0:
        raise ValueError("move_joints.angle_tolerance_deg 必须是正的有限数")
    if hw.left_arm is None or hw.right_arm is None:
        raise RuntimeError("HCX 双臂未连接，无法确认反馈到位")

    deadline = time.monotonic() + timeout_s
    left_target_np = np.asarray(left_target, dtype=np.float64)
    right_target_np = np.asarray(right_target, dtype=np.float64)
    while True:
        left_fb = np.asarray(hw.left_arm.joint_angles(), dtype=np.float64)
        right_fb = np.asarray(hw.right_arm.joint_angles(), dtype=np.float64)
        left_ok = np.all(np.abs(left_fb - left_target_np) <= angle_tolerance_deg)
        right_ok = np.all(np.abs(right_fb - right_target_np) <= angle_tolerance_deg)
        if left_ok and right_ok:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise RuntimeError("move_joints 反馈确认超时：关节未在容差内到位")
        time.sleep(min(poll_interval_s, remaining))


def _send_frame_move_joints(
    hw: HardwareBundle,
    frame: PlaybackFrame,
    *,
    speed_ratio: float | None,
    acceleration_seconds: float | None,
    deceleration_seconds: float | None,
    feedback_confirm_timeout_s: float,
    feedback_confirm_poll_interval_s: float,
    angle_tolerance_deg: float,
) -> None:
    left = frame.left_angles_deg.astype(float).tolist()
    right = frame.right_angles_deg.astype(float).tolist()
    if hw.left_arm is None or hw.right_arm is None:
        raise RuntimeError("HCX 双臂未连接")

    if hw.left_gripper_loop is not None:
        hw.left_gripper_loop.set_opening(frame.left_gripper)
    elif hw.left_gripper is not None:
        _ = hw.left_gripper.send_normalized(frame.left_gripper)
    if hw.right_gripper_loop is not None:
        hw.right_gripper_loop.set_opening(frame.right_gripper)
    elif hw.right_gripper is not None:
        _ = hw.right_gripper.send_normalized(frame.right_gripper)

    _ = hw.left_arm.move_joints(
        left,
        interrupt=False,
        acceleration_seconds=acceleration_seconds,
        deceleration_seconds=deceleration_seconds,
        speed_ratio=speed_ratio,
        smooth=1,
        wait=False,
    )
    _ = hw.right_arm.move_joints(
        right,
        interrupt=False,
        acceleration_seconds=acceleration_seconds,
        deceleration_seconds=deceleration_seconds,
        speed_ratio=speed_ratio,
        smooth=1,
        wait=False,
    )
    _confirm_targets_by_feedback(
        hw,
        left,
        right,
        timeout_s=feedback_confirm_timeout_s,
        poll_interval_s=feedback_confirm_poll_interval_s,
        angle_tolerance_deg=angle_tolerance_deg,
    )


def _show_images(cv2: Any, frame: PlaybackFrame) -> None:
    for column_name in CAMERA_COLUMNS:
        path = frame.image_paths[column_name]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV 无法读取图像: {path}")
        cv2.imshow(CAMERA_WINDOWS[column_name], image)


def _play_frames(
    frames: list[PlaybackFrame],
    *,
    hw: HardwareBundle,
    cfg: dict[str, Any],
    cv2: Any | None,
    label: str,
) -> bool:
    """逐点 move_joints；返回 False 表示用户请求退出。"""

    playback_started_s = time.perf_counter()
    paused_total_s = 0.0
    for frame_index, frame in enumerate(frames):
        if cfg["pace_by_timestamp"]:
            target_s = (
                playback_started_s
                + paused_total_s
                + frame.timestamp_s / cfg["playback_speed"]
            )
            while True:
                remaining_s = target_s - time.perf_counter()
                if remaining_s <= 0.0:
                    break
                if cv2 is None:
                    time.sleep(min(remaining_s, 0.01))
                    continue
                key = cv2.waitKey(max(1, min(10, round(remaining_s * 1_000)))) & 0xFF
                if key in {27, ord("q")}:
                    return False
                if key == ord(" "):
                    pause_started_s = time.perf_counter()
                    while True:
                        pause_key = cv2.waitKey(20) & 0xFF
                        if pause_key in {27, ord("q")}:
                            return False
                        if pause_key == ord(" "):
                            break
                    pause_duration_s = time.perf_counter() - pause_started_s
                    paused_total_s += pause_duration_s
                    target_s += pause_duration_s

        print(
            f"[INFO] {label} frame={frame_index}/{len(frames) - 1} "
            f"t={frame.timestamp_s:.3f}s "
            f"left={[round(v, 2) for v in frame.left_angles_deg.tolist()]} "
            f"right={[round(v, 2) for v in frame.right_angles_deg.tolist()]} "
            f"grip=({frame.left_gripper:.3f},{frame.right_gripper:.3f})",
            flush=True,
        )
        _send_frame_move_joints(
            hw,
            frame,
            speed_ratio=cfg["move_speed_ratio"],
            acceleration_seconds=cfg["move_acceleration_seconds"],
            deceleration_seconds=cfg["move_deceleration_seconds"],
            feedback_confirm_timeout_s=cfg["move_feedback_confirm_timeout_s"],
            feedback_confirm_poll_interval_s=cfg[
                "move_feedback_confirm_poll_interval_s"
            ],
            angle_tolerance_deg=cfg["move_angle_tolerance_deg"],
        )
        if cv2 is not None and frame.image_paths:
            _show_images(cv2, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                return False
        if frame_index + 1 == len(frames):
            print(f"[INFO] {label} 回放完成（{len(frames)} 帧）")
    return True


def _shutdown(hw: HardwareBundle, cv2: Any | None) -> None:
    if hw.left_gripper_loop is not None:
        try:
            hw.left_gripper_loop.stop()
        except Exception as exc:
            print(f"[WARN] 停止左夹爪后台失败: {exc}")
    if hw.right_gripper_loop is not None:
        try:
            hw.right_gripper_loop.stop()
        except Exception as exc:
            print(f"[WARN] 停止右夹爪后台失败: {exc}")
    if hw.left_gripper is not None:
        try:
            hw.left_gripper.disable()
        except Exception:
            pass
        try:
            hw.left_gripper.disconnect()
        except Exception as exc:
            print(f"[WARN] 断开左夹爪失败: {exc}")
    if hw.right_gripper is not None:
        try:
            hw.right_gripper.disable()
        except Exception:
            pass
        try:
            hw.right_gripper.disconnect()
        except Exception as exc:
            print(f"[WARN] 断开右夹爪失败: {exc}")
    if hw.hcx_client is not None:
        try:
            hw.hcx_client.close()
        except Exception as exc:
            print(f"[WARN] 关闭 HCX 客户端失败: {exc}")
    if cv2 is not None:
        cv2.destroyAllWindows()


def main() -> int:
    args = _parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _resolve_path(config_path, base=VA_ROOT)
    cfg = _load_replay_config(config_path)

    if args.dataset is not None:
        cfg["dataset_root"] = _resolve_path(args.dataset, base=VA_ROOT)
    if args.episode is not None:
        cfg["episode_index"] = int(args.episode)
    if args.speed is not None:
        if not math.isfinite(args.speed) or args.speed <= 0.0:
            print("[ERROR] --speed 必须为正数")
            return 2
        cfg["playback_speed"] = float(args.speed)
    if args.joint_column is not None:
        cfg["joint_column"] = args.joint_column
    if args.loop:
        cfg["loop"] = True
    if args.display_cameras:
        cfg["display_cameras"] = True
    if args.dry_run:
        cfg["dry_run"] = True
    if args.no_grippers:
        cfg["connect_grippers"] = False

    parquet_path = _episode_parquet(cfg["dataset_root"], cfg["episode_index"])
    try:
        frames = load_playback_frames(
            parquet_path,
            joint_column=cfg["joint_column"],
            require_images=cfg["display_cameras"],
        )
    except Exception as exc:
        print(f"[ERROR] 无法加载回放数据: {exc}")
        return 2

    duration_s = frames[-1].timestamp_s
    average_hz = (len(frames) - 1) / duration_s if duration_s > 0.0 else 0.0
    print("=" * 72)
    print("    OpenArm/HCX 数据集 -> 实机双臂轨迹回放（move_joints）")
    print("=" * 72)
    print(f"  配置: {config_path}")
    print(f"  数据集: {cfg['dataset_root']}")
    print(f"  Parquet: {parquet_path}")
    print(f"  轨迹字段: {cfg['joint_column']}")
    print(
        f"  帧数: {len(frames)}；时长: {duration_s:.3f} s；"
        f"平均帧率: {average_hz:.2f} Hz；速度: {cfg['playback_speed']:.2f}x"
    )
    print(
        f"  move_joints: speed_ratio={cfg['move_speed_ratio']} "
        f"acc={cfg['move_acceleration_seconds']}s "
        f"dec={cfg['move_deceleration_seconds']}s "
        f"tol={cfg['move_angle_tolerance_deg']}deg"
    )
    print(f"  循环: {'是' if cfg['loop'] else '否'}；干跑: {'是' if cfg['dry_run'] else '否'}；夹爪: {'是' if cfg['connect_grippers'] else '否'}")
    print("-" * 72)

    if cfg["dry_run"]:
        first = frames[0]
        print(
            "[INFO] dry-run 首帧 "
            f"left={np.round(first.left_angles_deg, 2).tolist()} "
            f"right={np.round(first.right_angles_deg, 2).tolist()} "
            f"grippers=({first.left_gripper:.3f}, {first.right_gripper:.3f})"
        )
        print("[INFO] dry-run 完成，未连接硬件")
        return 0

    if not cfg["teleop_yaml"].is_file():
        print(f"[ERROR] 找不到 teleop.yaml: {cfg['teleop_yaml']}")
        return 2

    cv2: Any | None = None
    if cfg["display_cameras"]:
        try:
            import cv2 as imported_cv2
        except ImportError:
            print("[ERROR] 图像回放需要安装 opencv-python")
            return 2
        cv2 = imported_cv2
        for window_name in CAMERA_WINDOWS.values():
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        print("  空格暂停/继续；q 或 Esc 退出。")

    hw = HardwareBundle()
    try:
        print(f"[INFO] 连接 HCX（teleop={cfg['teleop_yaml']}）")
        hw.hcx_client, hw.left_arm, hw.right_arm = _connect_hcx_arms(cfg["teleop_yaml"])
        if cfg["connect_grippers"]:
            hw.left_gripper, hw.right_gripper, gripper_rate_hz = _connect_grippers(
                cfg["teleop_yaml"]
            )
            if hw.left_gripper is not None:
                hw.left_gripper_loop = BackgroundGripperLoop(
                    hw.left_gripper, rate_hz=gripper_rate_hz
                )
                hw.left_gripper_loop.start(initial_opening=frames[0].left_gripper)
            if hw.right_gripper is not None:
                hw.right_gripper_loop = BackgroundGripperLoop(
                    hw.right_gripper, rate_hz=gripper_rate_hz
                )
                hw.right_gripper_loop.start(initial_opening=frames[0].right_gripper)
        else:
            print("[INFO] 已跳过 Gloria-M 夹爪连接")

        print("[INFO] 使用 move_joints 逐点回放")
        while True:
            ok = _play_frames(
                frames,
                hw=hw,
                cfg=cfg,
                cv2=cv2,
                label="episode",
            )
            if not ok or not cfg["loop"]:
                break
            print("[INFO] 循环回放下一轮")
        return 0
    except KeyboardInterrupt:
        print("\n[STOP] 收到退出请求")
        return 0
    except Exception as exc:
        print(f"[ERROR] 回放失败: {exc}")
        return 1
    finally:
        _shutdown(hw, cv2)


if __name__ == "__main__":
    raise SystemExit(main())
