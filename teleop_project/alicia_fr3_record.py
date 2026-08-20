#!/usr/bin/env python3
"""正式的 Alicia-D -> FR3 双相机遥操作与数据采集入口。

本程序不使用 CLI 参数。部署、相机选择、数据集策略和采样频率全部位于根目录
``teleop.yaml``。现有 ``demo_alicia_fr3_record.py`` 保留为调试和快速验证入口。
"""

from __future__ import annotations

import json
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
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from leobot_scripts import (
    AsyncEpisodeFinalizer,
    CallableGripperFeedbackSource,
    CameraProcessDatasetRecorder,
    CameraRecorderHealth,
    DatasetRecorder,
    EpisodeOperation,
    EpisodeRecorder,
    OrbbecCameraAdapterConfig,
    RecorderConfig,
    RecordingDeploymentConfig,
    RecordingFollower,
    RecordingGripper,
    load_recording_config,
)
from orbbec_sdk import CameraMode, OrbbecCameraConfig, load_orbbec_camera_configs
from teleop_sdk import TeleopController
from teleop_sdk.adapters import (
    AliciaLeaderArm,
    FairinoFR3Follower,
    GloriaMGripperFollower,
)
from teleop_sdk.config import RuntimeConfig, load_runtime_config

CAMERA_READY_TIMEOUT_S = 10.0
CAMERA_STALE_TIMEOUT_S = 3.0
SERVO_START_TIMEOUT_S = 5.0


class SessionState(str, Enum):
    PREFLIGHT = "preflight"
    READY = "ready"
    TELEOP = "teleop"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    FAULT = "fault"
    SHUTDOWN = "shutdown"


@contextmanager
def _single_key_terminal() -> Generator[int, None, None]:
    """Read single keys while preserving terminal-generated Ctrl+C signals."""

    if not sys.stdin.isatty():
        raise RuntimeError("正式采集程序需要交互式终端，以读取 s、c 和 q")
    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    # cbreak keeps ISIG enabled, so Ctrl+C remains a real SIGINT even while
    # episode finalization is running in a background worker.
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
    if not recording.enabled_cameras:
        if recording.master_camera is not None:
            raise ValueError("未启用相机时不能设置 recording.master_camera")
        return ()
    declared = load_orbbec_camera_configs()
    by_name = {config.name: config for config in declared}
    unknown = set(recording.enabled_cameras) - set(by_name)
    if unknown:
        raise ValueError(f"recording.enabled_cameras 包含未声明相机: {sorted(unknown)}")
    configs = tuple(by_name[name] for name in recording.enabled_cameras)
    if any(config.mode not in {CameraMode.RGB, CameraMode.RGBD} for config in configs):
        raise ValueError("正式采集要求所有启用相机都提供 RGB 流")
    if any(config.fps != recording.fps for config in configs):
        raise ValueError("启用相机的 fps 必须与 recording.fps 一致")
    if len(configs) > 1 and recording.master_camera is None:
        raise ValueError("启用多台相机时必须设置 recording.master_camera")
    if (
        recording.master_camera is not None
        and recording.master_camera not in recording.enabled_cameras
    ):
        raise ValueError("recording.master_camera 必须位于 recording.enabled_cameras")
    return configs


def _resolve_dataset_root(recording: RecordingDeploymentConfig) -> Path:
    """Use one persistent dataset root across collection program restarts."""

    root = recording.root.resolve()
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"recording.root 不是目录: {root}")
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
        free_gb = free_bytes / 1024**3
        raise RuntimeError(
            f"磁盘剩余空间不足: {free_gb:.1f} GiB，要求至少 {min_free_disk_gb:.1f} GiB"
        )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        if isinstance(value, type):
            return value.__name__
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
    }
    meta_dir = root / "meta"
    initial_manifest = meta_dir / "run_config.json"
    if not initial_manifest.exists():
        _write_json_atomic(initial_manifest, value)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    history_dir = meta_dir / "runs"
    path = history_dir / f"run_{timestamp}.json"
    suffix = 1
    while path.exists():
        path = history_dir / f"run_{timestamp}_{suffix}.json"
        suffix += 1
    _write_json_atomic(path, value)
    return path


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


class AliciaFR3RecordingSession:
    """Coordinate control, camera collection, finalization, and shutdown."""

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
        self._controller: TeleopController | None = None
        self._recorder: EpisodeRecorder | None = None
        self._camera_recorder: CameraProcessDatasetRecorder | None = None
        self._finalizer: AsyncEpisodeFinalizer | None = None
        self._fault_message: str | None = None
        self._shutdown_started = False

    @property
    def state(self) -> SessionState:
        return self._state

    def start(self) -> None:
        if self._state is not SessionState.PREFLIGHT:
            raise RuntimeError("采集会话已经启动")

        physical_gripper = GloriaMGripperFollower(self._runtime.gloria_m)
        recorder_config = RecorderConfig(
            root=self._dataset_root,
            robot_type="fairino_fr3",
            fps=self._recording.fps,
            image_storage=self._recording.image_storage,
            quality=self._recording.quality,
            gripper_source=CallableGripperFeedbackSource(
                physical_gripper.read_cached_normalized_opening
            ),
        )
        if self._camera_configs:
            camera_recorder = CameraProcessDatasetRecorder(
                recorder_config,
                OrbbecCameraAdapterConfig(self._camera_configs),
                master_camera_name=self._recording.master_camera,
                numeric_sample_fps=self._recording.numeric_sample_fps,
                diagnostics_enabled=True,
            )
            recorder: EpisodeRecorder = camera_recorder
            self._camera_recorder = camera_recorder
        else:
            recorder = DatasetRecorder(recorder_config)
        self._recorder = recorder
        self._finalizer = AsyncEpisodeFinalizer(recorder)

        follower = RecordingFollower(
            FairinoFR3Follower(self._runtime.fr3.robot_ip), recorder
        )
        gripper = RecordingGripper(physical_gripper, recorder)
        leader = AliciaLeaderArm(
            port=self._runtime.alicia.port,
            gripper_type=self._runtime.alicia.gripper_type,
            connect_retries=self._runtime.alicia.connect_retries,
            connect_retry_delay_s=self._runtime.alicia.connect_retry_delay_s,
        )
        controller = TeleopController(
            leader,
            follower,
            replace(
                self._runtime.teleop,
                axis_sign=self._runtime.fr3.axis_sign,
            ),
            gripper=gripper,
        )
        self._controller = controller

        self._state = SessionState.READY
        controller.connect()
        if controller.gripper is None:
            raise RuntimeError("Gloria-M 连接失败；正式采集拒绝在没有夹爪反馈时启动")
        if self._camera_recorder is not None:
            health = self._camera_recorder.wait_until_camera_ready(
                CAMERA_READY_TIMEOUT_S
            )
            self._assert_camera_fresh(health)

        self._controller_thread = threading.Thread(
            target=self._run_controller,
            name="alicia-fr3-control",
            daemon=False,
        )
        self._controller_thread.start()
        if not controller.wait_for_servo_start(SERVO_START_TIMEOUT_S):
            raise RuntimeError("FR3 ServoMoveStart 失败；拒绝进入正式遥操作")
        self._state = SessionState.TELEOP
        self._print_ready()

    def run_interactive(self, stop_requested: threading.Event) -> None:
        if self._state is not SessionState.TELEOP:
            raise RuntimeError("会话尚未通过启动预检")
        _status(
            "[INFO] 按 s 开始/结束 episode；按 c 丢弃 episode；按 q 或 Ctrl+C 安全退出"
        )
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
        """Stop sampling before disabling and disconnecting robot hardware."""

        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._state = SessionState.SHUTDOWN
        self._stop_event.set()

        controller_thread = self._controller_thread
        if controller_thread is not None:
            controller_thread.join(timeout=5.0)
            if controller_thread.is_alive():
                _status("[WARN] 控制线程仍在退出，等待其结束")
                controller_thread.join()

        finalizer = self._finalizer
        if finalizer is not None and finalizer.busy:
            _status("[INFO] 正在等待当前 episode 收尾，控制已停止")
            result = finalizer.wait()
            if result is not None and not result.succeeded:
                _status(f"[WARN] episode 收尾失败: {result.error}")

        recorder = self._recorder
        if recorder is not None and recorder.active:
            _status("[INFO] 正在丢弃未完成 episode 并关闭采集器")
        if recorder is not None:
            try:
                recorder.close()
                if self._camera_recorder is not None:
                    _status("[INFO] 相机采集进程和相机流已关闭")
            except Exception as exc:
                _status(f"[WARN] 关闭采集器失败: {exc}")
        if finalizer is not None:
            try:
                finalizer.close()
            except Exception as exc:
                _status(f"[WARN] 关闭 episode 收尾线程失败: {exc}")

        controller = self._controller
        if controller is not None:
            try:
                controller.shutdown()
            except Exception as exc:
                _status(f"[WARN] 硬件清理失败: {exc}")

    def _run_controller(self) -> None:
        assert self._controller is not None
        try:
            self._controller.run(self._stop_event, cleanup_on_exit=False)
        except BaseException as exc:
            self._controller_error = exc
        finally:
            self._controller_done.set()

    def _poll(self) -> None:
        if self._controller_done.is_set() and self._state is not SessionState.SHUTDOWN:
            if self._controller_error is not None:
                self._fault(f"遥操作控制线程异常退出: {self._controller_error!r}")
            else:
                self._fault("遥操作控制线程意外停止")
            return

        if self._camera_recorder is not None:
            health = self._camera_recorder.check_health()
            if not health.healthy:
                self._fault(health.error or "相机采集进程故障")
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
            self._fault(f"episode {result.operation.value} 失败: {result.error}")
            return
        self._state = SessionState.TELEOP
        if result.operation is EpisodeOperation.FINISH:
            assert result.episode_index is not None
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
            _status(f"[INFO] 开始采集 episode {episode:06d}")
            return
        if self._state is SessionState.RECORDING:
            try:
                self._finalizer.finish()
            except Exception as exc:
                self._fault(f"无法结束 episode: {exc}")
                return
            self._state = SessionState.FINALIZING
            _status("[INFO] 已停止采样，正在后台提交 episode")
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
        capture_ns = health.last_master_capture_monotonic_ns
        if capture_ns is None:
            raise RuntimeError("主相机尚未产生可用 RGB 帧")
        age_s = (time.perf_counter_ns() - capture_ns) / 1_000_000_000
        if age_s > CAMERA_STALE_TIMEOUT_S:
            raise RuntimeError(f"主相机已 {age_s:.1f} 秒未更新")

    def _fault(self, message: str) -> None:
        if self._state is SessionState.FAULT:
            return
        self._fault_message = message
        self._state = SessionState.FAULT
        self._stop_event.set()
        _status(f"[FAULT] {message}")

    def _print_ready(self) -> None:
        names = [config.name for config in self._camera_configs]
        _status("=" * 60)
        _status(" Alicia-D -> FR3 正式遥操作与采集")
        _status("=" * 60)
        _status(f"[INFO] 数据集目录: {self._dataset_root}")
        _status(f"[INFO] ServoJ 控制频率: {self._runtime.teleop.rate_hz} Hz")
        _status(f"[INFO] 数据集标称帧率: {self._recording.fps} Hz")
        _status(f"[INFO] FR3/夹爪数值采样频率: {self._recording.numeric_sample_fps} Hz")
        _status(f"[INFO] RGB 相机: {names if names else '未启用'}")
        if names:
            master = self._recording.master_camera or names[0]
            _status(f"[INFO] 主相机: {master}")
        storage = {
            "png": "无损 PNG",
            "video": "AV1 视频",
        }.get(
            self._recording.image_storage,
            f"JPG 图像序列（quality={self._recording.quality}）",
        )
        _status(f"[INFO] RGB 存储: {storage}")


def main() -> None:
    runtime = load_runtime_config()
    recording = load_recording_config()
    camera_configs = _select_camera_configs(recording)
    dataset_root = _resolve_dataset_root(recording)
    _check_disk_space(dataset_root, recording.min_free_disk_gb)

    session = AliciaFR3RecordingSession(
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
        _write_run_manifest(dataset_root, runtime, recording, camera_configs)
        session.run_interactive(stop_requested)
    except KeyboardInterrupt:
        _status("[STOP] 收到中断信号")
    except BaseException as exc:
        _status(f"[FAULT] 启动或运行失败: {exc}")
    finally:
        stop_requested.set()
        session.shutdown()
        signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    main()
