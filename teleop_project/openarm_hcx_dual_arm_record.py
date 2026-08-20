#!/usr/bin/env python3
"""OpenArm Mini -> HCX 双臂遥操作与头部相机驱动数据采集。

本入口独立于 ``openarm_hcx_dual_arm_teleop.py``。它复用该入口的双臂
控制链、Gloria-M 工作线程和 HCX 直伺服配置，但不修改其 500 Hz 输出
线程。每张新的头部相机帧只触发一次后台真实反馈读取；左右手图像选择各自在
该头部帧之前的最新一帧。
"""

from __future__ import annotations

import json
import math
import os
import select
import shutil
import signal
import sys
import termios
import threading
import time
import tty
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

import openarm_hcx_dual_arm_teleop as dual_teleop
from leobot_scripts import (
    AsyncEpisodeFinalizer,
    CameraRecorderHealth,
    MasterFrameRequest,
    MasterFrameSkip,
    MasterFrameSnapshot,
    MasterTriggeredCameraProcessDatasetRecorder,
    OrbbecCameraAdapterConfig,
    RecorderConfig,
    RecordingDeploymentConfig,
    load_recording_config,
)
from orbbec_sdk import CameraMode, OrbbecCameraConfig, load_orbbec_camera_configs
from teleop_sdk import TeleopController
from teleop_sdk.adapters import (
    GloriaMGripperFollower,
    HcxConnection,
    HcxConnectionConfig,
    HcxDirectServoConfig,
    HcxFollower,
)
from teleop_sdk.config import (
    GloriaMDualGripperConfig,
    RuntimeConfig,
    load_runtime_config,
)

_AXIS_COUNT = 7
_CAMERA_READY_TIMEOUT_S = 10.0
_CAMERA_STALE_TIMEOUT_S = 3.0
_REQUIRED_CAMERA_NAMES = ("head", "left_hand", "right_hand")


def _age_text(now_ns: int, timestamp_ns: int | None) -> str:
    if timestamp_ns is None:
        return "unavailable"
    return f"{max(0, now_ns - timestamp_ns) / 1_000_000_000:.3f}s"


def _camera_health_context(health: CameraRecorderHealth, now_ns: int) -> str:
    worker = (
        f"worker(state={health.state}, alive={health.worker_alive}, "
        f"active={health.active}, "
        f"heartbeat-age={_age_text(now_ns, health.last_heartbeat_monotonic_ns)})"
    )
    source_parts = []
    ordered_names = tuple(
        dict.fromkeys((*_REQUIRED_CAMERA_NAMES, *health.source_health.keys()))
    )
    for name in ordered_names:
        source = health.source_health.get(name)
        if source is None:
            source_parts.append(f"{name}(status=unavailable)")
            continue
        source_parts.append(
            f"{name}(status={source.status}, "
            f"latest-age={_age_text(now_ns, source.latest_capture_monotonic_ns)}, "
            f"error={source.error or '--'!r})"
        )
    sources = "; ".join(source_parts) or "camera diagnostics unavailable"
    return f"{worker}; cameras: {sources}"


def _camera_source_failure_message(
    health: CameraRecorderHealth,
    *,
    now_ns: int | None = None,
) -> str | None:
    failed = [
        (name, source)
        for name, source in health.source_health.items()
        if source.status in {"failed", "error", "stopped"}
    ]
    if not failed:
        return None
    checked_ns = time.perf_counter_ns() if now_ns is None else now_ns
    summary = "; ".join(
        f"{name} status={source.status} error={source.error or '--'!r}"
        for name, source in failed
    )
    return (
        f"相机源故障: {summary}; "
        f"{_camera_health_context(health, checked_ns)}"
    )


def _camera_worker_failure_message(
    health: CameraRecorderHealth,
    fallback: str,
    *,
    now_ns: int | None = None,
) -> str:
    checked_ns = time.perf_counter_ns() if now_ns is None else now_ns
    return f"{health.error or fallback}; {_camera_health_context(health, checked_ns)}"


def _camera_stale_message(
    health: CameraRecorderHealth,
    *,
    now_ns: int | None = None,
) -> str | None:
    """Explain whether stale capture comes from a camera or worker stall."""

    checked_ns = time.perf_counter_ns() if now_ns is None else now_ns
    capture_ns = health.last_master_capture_monotonic_ns
    if capture_ns is not None:
        age_s = (checked_ns - capture_ns) / 1_000_000_000
        if age_s <= _CAMERA_STALE_TIMEOUT_S:
            return None
        headline = f"头部相机已 {age_s:.3f} 秒未更新"
    else:
        headline = "头部相机尚未产生可用 RGB 帧"

    heartbeat_age_s = (
        None
        if health.last_heartbeat_monotonic_ns is None
        else (checked_ns - health.last_heartbeat_monotonic_ns) / 1_000_000_000
    )
    head_health = health.source_health.get("head")
    if head_health is not None and head_health.status == "failed":
        likely_cause = (
            "head camera FAILED"
            f" ({head_health.error or 'camera adapter did not provide an error'})"
        )
    elif heartbeat_age_s is not None and heartbeat_age_s > _CAMERA_STALE_TIMEOUT_S:
        likely_cause = "camera worker heartbeat stalled (encoding/write or worker loop blocked)"
    elif (
        head_health is not None
        and head_health.latest_capture_monotonic_ns is not None
        and checked_ns - head_health.latest_capture_monotonic_ns
        > int(_CAMERA_STALE_TIMEOUT_S * 1_000_000_000)
    ):
        likely_cause = "head source stopped producing complete frames"
    elif head_health is not None and head_health.status == "streaming":
        likely_cause = "head is streaming but the recorder master cursor did not advance"
    else:
        likely_cause = "camera diagnostics unavailable"
    return (
        f"{headline}; likely-cause={likely_cause}; "
        f"{_camera_health_context(health, checked_ns)}"
    )


class SessionState(str, Enum):
    PREFLIGHT = "preflight"
    TELEOP = "teleop"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    FAULT = "fault"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class _TimedJointAction:
    target_deg: np.ndarray
    accepted_monotonic_ns: int


@dataclass(frozen=True)
class _TimedGripperAction:
    opening: float
    accepted_monotonic_ns: int


@dataclass(frozen=True)
class _DualActionSnapshot:
    action_deg: np.ndarray
    left_gripper: float
    right_gripper: float
    audit: dict[str, Any]


class _DualActionTracker:
    """Record only the latest targets accepted by the existing low-rate paths."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._joint_actions: dict[str, _TimedJointAction] = {}
        self._gripper_actions: dict[str, _TimedGripperAction] = {}

    def note_joint_action(self, side: str, target_deg: np.ndarray) -> None:
        target = np.asarray(target_deg, dtype=float)
        if target.shape != (_AXIS_COUNT,) or not np.isfinite(target).all():
            raise ValueError(f"{side} accepted an invalid HCX joint target")
        with self._lock:
            self._joint_actions[side] = _TimedJointAction(
                target_deg=target.copy(),
                accepted_monotonic_ns=time.perf_counter_ns(),
            )

    def note_gripper_action(self, side: str, opening: float) -> None:
        value = float(opening)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{side} accepted an invalid gripper opening")
        with self._lock:
            self._gripper_actions[side] = _TimedGripperAction(
                opening=value,
                accepted_monotonic_ns=time.perf_counter_ns(),
            )

    def ready(self) -> bool:
        with self._lock:
            return all(
                side in self._joint_actions and side in self._gripper_actions
                for side in ("left", "right")
            )

    def snapshot(self) -> _DualActionSnapshot | None:
        with self._lock:
            if not self.ready_locked():
                return None
            left_joint = self._joint_actions["left"]
            right_joint = self._joint_actions["right"]
            left_gripper = self._gripper_actions["left"]
            right_gripper = self._gripper_actions["right"]
            return _DualActionSnapshot(
                action_deg=np.concatenate(
                    (left_joint.target_deg, right_joint.target_deg)
                ).copy(),
                left_gripper=left_gripper.opening,
                right_gripper=right_gripper.opening,
                audit={
                    "left_joint_action_accepted_monotonic_ns": left_joint.accepted_monotonic_ns,
                    "right_joint_action_accepted_monotonic_ns": right_joint.accepted_monotonic_ns,
                    "left_gripper_action_accepted_monotonic_ns": left_gripper.accepted_monotonic_ns,
                    "right_gripper_action_accepted_monotonic_ns": right_gripper.accepted_monotonic_ns,
                },
            )

    def ready_locked(self) -> bool:
        return all(
            side in self._joint_actions and side in self._gripper_actions
            for side in ("left", "right")
        )


class _TrackedHcxFollower:
    """Pass through HCX control while observing accepted 100 Hz input targets.

    It does not attach an observer to the HCX direct-servo output session and
    does not call the native direct-servo interface itself.
    """

    def __init__(self, inner: HcxFollower, side: str, tracker: _DualActionTracker):
        self._inner = inner
        self._side = side
        self._tracker = tracker

    @property
    def joint_count(self) -> int:
        return self._inner.joint_count

    @property
    def joint_limits_deg(self) -> tuple[np.ndarray, np.ndarray]:
        return self._inner.joint_limits_deg

    @property
    def requires_per_cycle_target_updates(self) -> bool:
        return self._inner.requires_per_cycle_target_updates

    def connect(self) -> None:
        self._inner.connect()

    def read_joint_angles_deg(self) -> np.ndarray:
        return self._inner.read_joint_angles_deg()

    def start_servo(self) -> bool:
        return self._inner.start_servo()

    def refresh_servo_target(self) -> bool:
        return self._inner.refresh_servo_target()

    def send_joint_angles_deg(
        self, angles_deg: np.ndarray, command_time_s: float
    ) -> bool:
        target = np.asarray(angles_deg, dtype=float).copy()
        accepted = self._inner.send_joint_angles_deg(target, command_time_s)
        if accepted:
            self._tracker.note_joint_action(self._side, target)
        return accepted

    def recover(self) -> bool:
        return self._inner.recover()

    def stop_servo(self) -> None:
        self._inner.stop_servo()

    def disconnect(self) -> None:
        self._inner.disconnect()


class _TrackedGloriaGripper:
    """Pass through one existing Gloria-M worker while recording accepted action."""

    def __init__(
        self,
        inner: GloriaMGripperFollower,
        side: str,
        tracker: _DualActionTracker,
    ) -> None:
        self._inner = inner
        self._side = side
        self._tracker = tracker

    def send_normalized(self, opening: float) -> bool:
        value = float(opening)
        accepted = self._inner.send_normalized(value)
        if accepted:
            self._tracker.note_gripper_action(self._side, value)
        return accepted

    def disable(self) -> None:
        self._inner.disable()

    def disconnect(self) -> None:
        self._inner.disconnect()


@contextmanager
def _single_key_terminal() -> Generator[int, None, None]:
    if not sys.stdin.isatty():
        raise RuntimeError("采集程序需要交互式终端，以读取 s、c 和 q")
    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    tty.setcbreak(descriptor)
    try:
        yield descriptor
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def _status(message: str) -> None:
    print(message, flush=True)


def _select_camera_configs(
    recording: RecordingDeploymentConfig,
) -> tuple[OrbbecCameraConfig, ...]:
    if tuple(recording.enabled_cameras) != _REQUIRED_CAMERA_NAMES:
        raise ValueError(
            "hcx_recording.enabled_cameras 必须按顺序为 "
            "['head', 'left_hand', 'right_hand']"
        )
    if recording.master_camera != "head":
        raise ValueError("hcx_recording.master_camera 必须为 'head'")
    declared = load_orbbec_camera_configs(section_name="hcx_orbbec")
    by_name = {camera.name: camera for camera in declared}
    missing = [name for name in _REQUIRED_CAMERA_NAMES if name not in by_name]
    if missing:
        raise ValueError(f"hcx_orbbec.cameras 缺少相机: {missing}")
    configs = tuple(by_name[name] for name in _REQUIRED_CAMERA_NAMES)
    if any(camera.mode not in {CameraMode.RGB, CameraMode.RGBD} for camera in configs):
        raise ValueError("HCX 采集要求头部和左右手相机都提供 RGB 流")
    if any(camera.fps != recording.fps for camera in configs):
        raise ValueError("hcx_orbbec 相机 fps 必须与 hcx_recording.fps 一致")
    return configs


def _resolve_dataset_root(recording: RecordingDeploymentConfig) -> Path:
    root = recording.root.resolve()
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"hcx_recording.root 不是目录: {root}")
    return root


def _check_disk_space(root: Path, min_free_disk_gb: float) -> None:
    probe = root
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            raise FileNotFoundError(f"找不到用于检查磁盘空间的目录: {root}")
        probe = parent
    free_bytes = shutil.disk_usage(probe).free
    required_bytes = int(min_free_disk_gb * 1024**3)
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"磁盘剩余空间不足: {free_bytes / 1024**3:.1f} GiB，"
            f"要求至少 {min_free_disk_gb:.1f} GiB"
        )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def _write_run_manifest(
    root: Path,
    runtime: RuntimeConfig,
    recording: RecordingDeploymentConfig,
    camera_configs: tuple[OrbbecCameraConfig, ...],
) -> Path:
    value = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolved_dataset_root": str(root),
        "runtime": _jsonable(runtime),
        "recording": _jsonable(recording),
        "orbbec_cameras": _jsonable(camera_configs),
        "collection_semantics": {
            "master_camera": "head",
            "row_trigger": "each new head camera frame",
            "hand_pairing": "latest capture at or before head capture",
            "feedback": "one post-capture actual HCX and Gloria-M read per head event",
            "fixed_rate_feedback_cache": False,
            "hcx_direct_servo_modified": False,
        },
    }
    meta_dir = root / "meta"
    first = meta_dir / "run_config.json"
    if not first.exists():
        _write_json_atomic(first, value)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = meta_dir / "runs" / f"run_{timestamp}.json"
    counter = 1
    while path.exists():
        path = meta_dir / "runs" / f"run_{timestamp}_{counter}.json"
        counter += 1
    _write_json_atomic(path, value)
    return path


def _run_dual_control_loop(
    left_controller: TeleopController,
    right_controller: TeleopController,
    rate_hz: float,
    stop_event: threading.Event,
) -> None:
    """Match the established parallel 100 Hz controller scheduling.

    HCX high-rate output remains in each follower's existing direct-servo
    thread.  This loop only produces its normal upstream 100 Hz targets.
    """

    period_s = 1.0 / rate_hz
    next_deadline = time.perf_counter()
    next_overrun_report_at = 0.0
    with ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="openarm-hcx-record-control"
    ) as executor:
        while not stop_event.is_set():
            timestamp = time.perf_counter()
            left_step = executor.submit(left_controller.step, timestamp)
            right_step = executor.submit(right_controller.step, timestamp)
            left_step.result()
            right_step.result()

            next_deadline += period_s
            finished_at = time.perf_counter()
            delay_s = next_deadline - finished_at
            if delay_s > 0.0:
                stop_event.wait(delay_s)
                continue
            if timestamp >= next_overrun_report_at:
                _status(
                    "[WARN] 双臂主控周期超期: "
                    f"耗时 {-delay_s * 1000.0:.1f} ms，目标周期 "
                    f"{period_s * 1000.0:.1f} ms"
                )
                next_overrun_report_at = timestamp + 1.0
            next_deadline = finished_at


class OpenArmHcxRecordingSession:
    """Coordinate unchanged dual-arm teleoperation and head-driven collection."""

    def __init__(
        self,
        runtime: RuntimeConfig,
        recording: RecordingDeploymentConfig,
        camera_configs: tuple[OrbbecCameraConfig, ...],
        dataset_root: Path,
    ) -> None:
        self._runtime = runtime
        self._recording = recording
        self._camera_configs = camera_configs
        self._dataset_root = dataset_root
        self._state = SessionState.PREFLIGHT
        self._stop_event = threading.Event()
        self._controller_done = threading.Event()
        self._controller_error: BaseException | None = None
        self._controller_thread: threading.Thread | None = None
        self._controllers: list[TeleopController] = []
        self._gripper_workers: list[dual_teleop.GloriaGripperWorker] = []
        self._followers: dict[str, HcxFollower] = {}
        self._grippers: dict[str, GloriaMGripperFollower] = {}
        self._tracker = _DualActionTracker()
        self._recorder: MasterTriggeredCameraProcessDatasetRecorder | None = None
        self._finalizer: AsyncEpisodeFinalizer | None = None
        self._shutdown_started = False

    @property
    def state(self) -> SessionState:
        return self._state

    def start(self) -> None:
        if self._state is not SessionState.PREFLIGHT:
            raise RuntimeError("采集会话已经启动")

        runtime = self._runtime
        dual_teleop._validate_openarm_config(runtime.openarm_mini)
        dual_teleop._validate_hcx_config(runtime.hcx)
        gloria_config = runtime.gloria_m_dual
        if not isinstance(gloria_config, GloriaMDualGripperConfig):
            raise ValueError("gloria_m_dual 配置必须是 GloriaMDualGripperConfig")
        enabled_sides = dual_teleop._validate_gloria_config(
            runtime.openarm_mini, gloria_config
        )
        if enabled_sides != ("left", "right"):
            raise ValueError(
                "正式双臂采集需要同时启用 gloria_m_dual.left 和 right，"
                "以写入左右夹爪的 action/observation 字段"
            )
        control_rate_hz = dual_teleop._validated_rate_hz(runtime.teleop)
        direct_config = dual_teleop._direct_servo_config(runtime.hcx, control_rate_hz)
        connection = HcxConnection(HcxConnectionConfig.from_runtime_config(runtime.hcx))

        left_follower = HcxFollower(
            connection,
            robot_id=runtime.hcx.left_robot_id,
            side="left",
            direct_servo_config=direct_config,
        )
        right_follower = HcxFollower(
            connection,
            robot_id=runtime.hcx.right_robot_id,
            side="right",
            direct_servo_config=direct_config,
        )
        self._followers = {"left": left_follower, "right": right_follower}
        left_tracked = _TrackedHcxFollower(left_follower, "left", self._tracker)
        right_tracked = _TrackedHcxFollower(right_follower, "right", self._tracker)

        left_controller = dual_teleop._create_controller(
            port=runtime.openarm_mini.port_left,
            side="left",
            openarm_config=runtime.openarm_mini,
            follower=left_tracked,  # type: ignore[arg-type]
            teleop_config=runtime.teleop,
            axis_sign=runtime.hcx.left_axis_sign,
        )
        right_controller = dual_teleop._create_controller(
            port=runtime.openarm_mini.port_right,
            side="right",
            openarm_config=runtime.openarm_mini,
            follower=right_tracked,  # type: ignore[arg-type]
            teleop_config=runtime.teleop,
            axis_sign=runtime.hcx.right_axis_sign,
        )
        self._controllers = [left_controller, right_controller]

        self._recorder = MasterTriggeredCameraProcessDatasetRecorder(
            RecorderConfig(
                root=self._dataset_root,
                robot_type="hcx_dual_arm",
                fps=self._recording.fps,
                image_storage=self._recording.image_storage,
                quality=self._recording.quality,
            ),
            OrbbecCameraAdapterConfig(self._camera_configs),
            joint_count=_AXIS_COUNT * 2,
            master_camera_name="head",
            snapshot_provider=self._snapshot_for_master_frame,
            scalar_actuator_names=("left_gripper", "right_gripper"),
            ready=self._tracker.ready,
        )
        self._finalizer = AsyncEpisodeFinalizer(self._recorder)

        for controller in self._controllers:
            controller.connect()
        self._create_tracked_gloria_workers(gloria_config)

        # Camera preflight occurs before direct servo starts.  The child owns
        # camera I/O and does not call any HCX or Gloria-M APIs by itself.
        health = self._recorder.wait_until_camera_ready(_CAMERA_READY_TIMEOUT_S)
        self._assert_camera_fresh(health)

        if not left_controller.start_servo():
            raise RuntimeError("HCX 左臂未通过启动前置检查")
        if not right_controller.start_servo():
            raise RuntimeError("HCX 右臂未通过启动前置检查")
        for worker in self._gripper_workers:
            worker.start()

        self._controller_thread = threading.Thread(
            target=self._run_control_thread,
            name="openarm-hcx-record-control",
            daemon=False,
        )
        self._controller_thread.start()
        self._state = SessionState.TELEOP
        self._print_ready(control_rate_hz, direct_config)

    def run_interactive(self, stop_requested: threading.Event) -> None:
        if self._state is not SessionState.TELEOP:
            raise RuntimeError("会话尚未通过启动预检")
        _status("[INFO] 按 s 开始/结束 episode；按 c 丢弃；按 q 或 Ctrl+C 安全退出")
        with _single_key_terminal() as descriptor:
            while (
                not stop_requested.is_set() and self._state is not SessionState.SHUTDOWN
            ):
                self._poll()
                if self._state is SessionState.FAULT:
                    return
                readable, _, _ = select.select([descriptor], [], [], 0.1)
                if not readable:
                    continue
                key = os.read(descriptor, 1)
                if key == b"s":
                    self._toggle_episode()
                elif key == b"c":
                    self._discard_episode()
                elif key in {b"q", b"Q"}:
                    _status("[INFO] 收到退出请求，正在安全关闭遥操作和采集资源")
                    self._stop_event.set()
                    stop_requested.set()

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._state = SessionState.SHUTDOWN
        self._stop_event.set()

        controller_thread = self._controller_thread
        if controller_thread is not None:
            controller_thread.join(timeout=5.0)
            if controller_thread.is_alive():
                _status("[WARN] 双臂控制线程仍在退出，继续等待")
                controller_thread.join()

        # Prevent low-rate gripper writes first.  Direct servo is stopped next,
        # before the recording parent can be closed and hardware disconnected.
        for worker in self._gripper_workers:
            worker.request_stop()
        for controller in reversed(self._controllers):
            try:
                controller.follower.stop_servo()
            except Exception as exc:
                _status(f"[WARN] 停止一侧 HCX 直伺服失败: {exc}")

        finalizer = self._finalizer
        if finalizer is not None and finalizer.busy:
            _status("[INFO] 正在等待当前 episode 收尾，HCX 直伺服已停止")
            result = finalizer.wait()
            if result is not None and not result.succeeded:
                _status(f"[WARN] episode 收尾失败: {result.error}")

        recorder = self._recorder
        if recorder is not None:
            try:
                recorder.close()
                _status("[INFO] 头部相机驱动采集器和相机流已关闭")
            except Exception as exc:
                _status(f"[WARN] 关闭采集器失败: {exc}")
        if finalizer is not None:
            try:
                finalizer.close()
            except Exception as exc:
                _status(f"[WARN] 关闭 episode 收尾线程失败: {exc}")

        for controller in reversed(self._controllers):
            try:
                controller.shutdown()
            except Exception as exc:
                _status(f"[WARN] 关闭一侧 OpenArm Mini -> HCX 控制链失败: {exc}")
        for worker in reversed(self._gripper_workers):
            if not worker.close():
                _status(
                    f"[WARN] Gloria-M {worker.side} 夹爪线程未在超时内退出；"
                    "未在主线程强制断开该夹爪"
                )

    def _create_tracked_gloria_workers(self, config: GloriaMDualGripperConfig) -> None:
        leaders = {
            "left": self._controllers[0].leader,
            "right": self._controllers[1].leader,
        }
        workers: list[dual_teleop.GloriaGripperWorker] = []
        created: dict[str, GloriaMGripperFollower] = {}
        try:
            for side in ("left", "right"):
                gripper = GloriaMGripperFollower(config.side_config(side))
                gripper.connect()
                tracked = _TrackedGloriaGripper(gripper, side, self._tracker)
                workers.append(
                    dual_teleop.GloriaGripperWorker(
                        side,  # type: ignore[arg-type]
                        leaders[side],
                        tracked,  # type: ignore[arg-type]
                        rate_hz=float(config.rate_hz),
                        status_print_interval_s=float(config.status_print_interval_s),
                    )
                )
                created[side] = gripper
        except BaseException:
            for gripper in created.values():
                try:
                    gripper.disable()
                except Exception:
                    pass
                try:
                    gripper.disconnect()
                except Exception:
                    pass
            raise
        self._grippers = created
        self._gripper_workers = workers

    def _snapshot_for_master_frame(
        self, request: MasterFrameRequest
    ) -> MasterFrameSnapshot | MasterFrameSkip:
        """Read real follower feedback once, after one head-frame event."""

        left_started = time.perf_counter_ns()
        left_state = np.asarray(
            self._followers["left"].read_joint_angles_deg(), dtype=float
        )
        left_finished = time.perf_counter_ns()
        right_started = time.perf_counter_ns()
        right_state = np.asarray(
            self._followers["right"].read_joint_angles_deg(), dtype=float
        )
        right_finished = time.perf_counter_ns()
        if (
            left_state.shape != (_AXIS_COUNT,)
            or right_state.shape != (_AXIS_COUNT,)
            or not np.isfinite(left_state).all()
            or not np.isfinite(right_state).all()
        ):
            raise RuntimeError("HCX returned invalid joint feedback during recording")

        left_gripper_started = time.perf_counter_ns()
        left_gripper = self._grippers["left"].read_normalized_opening()
        left_gripper_finished = time.perf_counter_ns()
        right_gripper_started = time.perf_counter_ns()
        right_gripper = self._grippers["right"].read_normalized_opening()
        right_gripper_finished = time.perf_counter_ns()
        audit = {
            "master_event_capture_monotonic_ns": request.capture_monotonic_ns,
            "left_joint_feedback_started_monotonic_ns": left_started,
            "left_joint_feedback_finished_monotonic_ns": left_finished,
            "right_joint_feedback_started_monotonic_ns": right_started,
            "right_joint_feedback_finished_monotonic_ns": right_finished,
            "left_gripper_feedback_started_monotonic_ns": left_gripper_started,
            "left_gripper_feedback_finished_monotonic_ns": left_gripper_finished,
            "right_gripper_feedback_started_monotonic_ns": right_gripper_started,
            "right_gripper_feedback_finished_monotonic_ns": right_gripper_finished,
        }
        if left_gripper is None or right_gripper is None:
            return MasterFrameSkip("missing_gloria_m_feedback", audit)

        actions = self._tracker.snapshot()
        if actions is None:
            return MasterFrameSkip("missing_successful_upstream_action", audit)
        return MasterFrameSnapshot(
            state=np.concatenate((left_state, right_state)),
            action=actions.action_deg,
            actuator_states={
                "left_gripper": float(left_gripper),
                "right_gripper": float(right_gripper),
            },
            actuator_actions={
                "left_gripper": actions.left_gripper,
                "right_gripper": actions.right_gripper,
            },
            audit={**audit, **actions.audit},
        )

    def _run_control_thread(self) -> None:
        try:
            _run_dual_control_loop(
                self._controllers[0],
                self._controllers[1],
                dual_teleop._validated_rate_hz(self._runtime.teleop),
                self._stop_event,
            )
        except BaseException as exc:
            self._controller_error = exc
        finally:
            self._controller_done.set()

    def _poll(self) -> None:
        if self._controller_done.is_set() and self._state is not SessionState.SHUTDOWN:
            if self._controller_error is not None:
                self._fault(f"双臂遥操作控制线程异常退出: {self._controller_error!r}")
            else:
                self._fault("双臂遥操作控制线程意外停止")
            return

        recorder = self._recorder
        if recorder is not None:
            health = recorder.check_health()
            if not health.healthy:
                self._fault(
                    _camera_worker_failure_message(
                        health,
                        "头部相机采集进程故障",
                    )
                )
                return
            source_failure = _camera_source_failure_message(health)
            if source_failure is not None:
                self._fault(source_failure)
                return
            if self._state in {SessionState.TELEOP, SessionState.RECORDING}:
                try:
                    self._assert_camera_fresh(health)
                except RuntimeError as exc:
                    self._fault(str(exc))
                    return

        finalizer = self._finalizer
        if finalizer is None:
            return
        result = finalizer.poll()
        if result is None:
            return
        if not result.succeeded:
            detail = f"episode {result.operation.value} 失败: {result.error}"
            if recorder is not None:
                detail = _camera_worker_failure_message(
                    recorder.check_health(),
                    detail,
                )
            self._fault(detail)
            return
        self._state = SessionState.TELEOP
        if result.episode_index is not None:
            _status(f"[INFO] episode {result.episode_index:06d} 已写入数据集")
        else:
            _status("[INFO] 当前 episode 已丢弃")

    def _toggle_episode(self) -> None:
        assert self._recorder is not None and self._finalizer is not None
        if self._state is SessionState.TELEOP:
            try:
                episode = self._recorder.start_episode(self._recording.task)
            except Exception as exc:
                _status(f"[WARN] 尚不能开始采集: {exc}")
                return
            self._state = SessionState.RECORDING
            _status(
                f"[INFO] 开始采集 episode {episode:06d}；每张头部相机帧触发一条记录"
            )
            return
        if self._state is SessionState.RECORDING:
            try:
                self._finalizer.finish()
            except Exception as exc:
                self._fault(f"无法结束 episode: {exc}")
                return
            self._state = SessionState.FINALIZING
            _status("[INFO] 已停止接收新头部帧，正在后台提交 episode")
            return
        if self._state is SessionState.FINALIZING:
            _status("[INFO] 当前 episode 正在收尾，请等待")

    def _discard_episode(self) -> None:
        assert self._recorder is not None and self._finalizer is not None
        if self._state is not SessionState.RECORDING:
            _status("[INFO] 当前没有可丢弃的 episode")
            return
        try:
            self._finalizer.discard()
        except Exception as exc:
            self._fault(f"无法丢弃 episode: {exc}")
            return
        self._state = SessionState.FINALIZING
        _status("[INFO] 正在后台丢弃当前 episode")

    def _assert_camera_fresh(self, health: CameraRecorderHealth) -> None:
        source_failure = _camera_source_failure_message(health)
        if source_failure is not None:
            raise RuntimeError(source_failure)
        message = _camera_stale_message(health)
        if message is not None:
            raise RuntimeError(message)

    def _fault(self, message: str) -> None:
        if self._state is SessionState.FAULT:
            return
        self._state = SessionState.FAULT
        self._stop_event.set()
        _status(f"[FAULT] {message}")

    def _print_ready(
        self, control_rate_hz: float, direct_config: HcxDirectServoConfig
    ) -> None:
        _status("=" * 68)
        _status(" OpenArm Mini -> HCX 双臂遥操作与头部相机驱动采集")
        _status("=" * 68)
        _status(f"[INFO] 数据集目录: {self._dataset_root}")
        _status(f"[INFO] OpenArm 主控频率: {control_rate_hz:.1f} Hz")
        _status(f"[INFO] HCX 直伺服: {direct_config.rate_hz} Hz（沿用现有输出线程）")
        _status(f"[INFO] 数据集标称帧率: {self._recording.fps} Hz")
        _status("[INFO] 头部相机每张新帧触发一次真实双臂/双夹爪反馈读取")
        _status("[INFO] 左右手图像取各自在头部帧时间之前的最新帧")
        _status("[INFO] 无固定频率 HCX/Gloria-M 反馈缓存；慢反馈帧只会审计丢弃")


def main() -> int:
    try:
        runtime = load_runtime_config()
        recording = load_recording_config(section_name="hcx_recording")
        camera_configs = _select_camera_configs(recording)
        dataset_root = _resolve_dataset_root(recording)
        _check_disk_space(dataset_root, recording.min_free_disk_gb)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        _status(f"[ERROR] OpenArm Mini -> HCX 采集配置无效: {exc}")
        return 2

    session = OpenArmHcxRecordingSession(
        runtime, recording, camera_configs, dataset_root
    )
    stop_requested = threading.Event()
    previous_handler = signal.getsignal(signal.SIGINT)

    def on_sigint(signum: int, frame: Any) -> None:
        del signum, frame
        stop_requested.set()

    signal.signal(signal.SIGINT, on_sigint)
    try:
        session.start()
        manifest = _write_run_manifest(dataset_root, runtime, recording, camera_configs)
        _status(f"[INFO] 本次运行配置已写入: {manifest}")
        session.run_interactive(stop_requested)
    except KeyboardInterrupt:
        _status("[STOP] 收到中断信号")
    except BaseException as exc:
        _status(f"[FAULT] 启动或运行失败: {exc}")
    finally:
        stop_requested.set()
        session.shutdown()
        signal.signal(signal.SIGINT, previous_handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
