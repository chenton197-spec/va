"""Camera-driven dataset recording independent of a camera vendor SDK."""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .camera import (
    CameraAdapterConfig,
    CameraAdapterSession,
    CameraFrameSource,
    CameraSourceHealth,
    RGBDFrameSource,
)
from .recorder import RecorderConfig, RecordingFollower
from .sources import CameraFrame, DepthFrame, RGBDFrame
from .v21_writer import CameraSpec, DepthCameraSpec, RecordedFrame, V21DatasetWriter, WriterConfig


_NUMERIC_HISTORY_CAPACITY = 120
_WORKER_HEARTBEAT_INTERVAL_S = 0.5
_DIAGNOSTIC_INTERVAL_S = 1.0
_DIAGNOSTIC_PROBE_INTERVAL_S = 0.1
_WORKER_CLOSE_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class _WorkerControl:
    request_id: int
    kind: str
    task: str | None = None


@dataclass(frozen=True)
class _WorkerResponse:
    request_id: int
    ok: bool
    value: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class _WorkerStatus:
    """Small child-to-parent liveness and capture-progress message."""

    state: str
    active: bool
    heartbeat_monotonic_ns: int
    last_master_capture_monotonic_ns: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class CameraRecorderHealth:
    """Observable health of one camera-driven recording worker."""

    state: str
    worker_alive: bool
    active: bool
    last_heartbeat_monotonic_ns: int | None
    last_master_capture_monotonic_ns: int | None
    error: str | None
    source_health: dict[str, CameraSourceHealth] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return self.error is None and self.state not in {"fault", "closed"}


@dataclass(frozen=True)
class _WorkerSample:
    """One numeric sample read by the parent process.

    ``capture_monotonic_ns`` is when the follower snapshot became available,
    rather than its requested schedule time. This keeps causal camera pairing
    from accidentally selecting a state read after a camera frame.
    """

    state: np.ndarray
    action: np.ndarray
    gripper_state: float | None
    gripper_action: float | None
    capture_monotonic_ns: int
    audit: dict[str, Any]


@dataclass(frozen=True)
class _WorkerAudit:
    audit: dict[str, Any]


@dataclass(frozen=True)
class _WorkerBarrier:
    """A same-queue marker that makes episode stop/drain ordering explicit."""

    request_id: int


@dataclass
class _EpisodeTimeline:
    """Convert one host monotonic clock into episode-relative timestamps."""

    origin_monotonic_ns: int | None = None

    def timestamp_s(self, capture_monotonic_ns: int) -> float:
        if self.origin_monotonic_ns is None:
            self.origin_monotonic_ns = capture_monotonic_ns
        return (capture_monotonic_ns - self.origin_monotonic_ns) / 1_000_000_000

    def camera_timestamp_s(self, capture_monotonic_ns: int | None, row_timestamp_s: float) -> float:
        if capture_monotonic_ns is None:
            return row_timestamp_s
        if self.origin_monotonic_ns is None:
            self.origin_monotonic_ns = capture_monotonic_ns
        return (capture_monotonic_ns - self.origin_monotonic_ns) / 1_000_000_000


class _WorkerDiagnostics:
    """Low-rate, opt-in progress counters for one camera worker process."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._lock = threading.Lock()
        now_ns = time.perf_counter_ns()
        self._active = False
        self._stage = "starting"
        self._stage_started_ns = now_ns
        self._last_loop_progress_ns = now_ns
        self._worker_master_capture_ns: int | None = None
        self._monitor_master_capture_ns: int | None = None
        self._worker_master_updates = 0
        self._monitor_master_updates = 0
        self._numeric_received = 0
        self._numeric_audits = 0
        self._numeric_processed = 0
        self._monitor_error: str | None = None

    def set_active(self, active: bool) -> None:
        if not self.enabled:
            return
        with self._lock:
            now_ns = time.perf_counter_ns()
            self._active = active
            self._stage = "idle" if active else "ready"
            self._stage_started_ns = now_ns
            self._last_loop_progress_ns = now_ns
            if active:
                self._worker_master_capture_ns = None
                self._monitor_master_capture_ns = None
                self._worker_master_updates = 0
                self._monitor_master_updates = 0
                self._numeric_received = 0
                self._numeric_audits = 0
                self._numeric_processed = 0
                self._monitor_error = None

    def set_stage(self, stage: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._stage = stage
            self._stage_started_ns = time.perf_counter_ns()

    def note_loop_progress(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._last_loop_progress_ns = time.perf_counter_ns()

    def note_worker_master_frame(self, capture_monotonic_ns: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._worker_master_capture_ns != capture_monotonic_ns:
                self._worker_master_updates += 1
            self._worker_master_capture_ns = capture_monotonic_ns

    def note_monitor_master_frame(self, capture_monotonic_ns: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._monitor_master_capture_ns != capture_monotonic_ns:
                self._monitor_master_updates += 1
            self._monitor_master_capture_ns = capture_monotonic_ns

    def note_numeric_sample(self, sample: _WorkerSample | _WorkerAudit) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._numeric_received += 1
            if isinstance(sample, _WorkerAudit):
                self._numeric_audits += 1

    def note_processed_sample(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._numeric_processed += 1

    def note_monitor_error(self, error: BaseException) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._monitor_error = repr(error)

    def snapshot(self) -> dict[str, Any]:
        now_ns = time.perf_counter_ns()
        with self._lock:
            return {
                "active": self._active,
                "stage": self._stage,
                "stage_age_ns": now_ns - self._stage_started_ns,
                "loop_age_ns": now_ns - self._last_loop_progress_ns,
                "worker_master_capture_ns": self._worker_master_capture_ns,
                "monitor_master_capture_ns": self._monitor_master_capture_ns,
                "worker_master_updates": self._worker_master_updates,
                "monitor_master_updates": self._monitor_master_updates,
                "numeric_received": self._numeric_received,
                "numeric_audits": self._numeric_audits,
                "numeric_processed": self._numeric_processed,
                "monitor_error": self._monitor_error,
            }


def _run_worker_diagnostic_monitor(
    diagnostics: _WorkerDiagnostics,
    master_source: CameraFrameSource,
    sample_queue: Any,
    stop_event: threading.Event,
) -> None:
    """Report whether the camera reader advances while the worker is busy."""

    if not diagnostics.enabled:
        return
    sequence = master_source.latest_sequence()
    report_started_s = time.monotonic()
    previous = diagnostics.snapshot()
    while not stop_event.wait(_DIAGNOSTIC_PROBE_INTERVAL_S):
        try:
            while True:
                item = master_source.next_frame_after(sequence)
                if item is None:
                    break
                next_sequence, frame = item
                if next_sequence <= sequence:
                    raise RuntimeError("Diagnostic master source returned a non-increasing sequence")
                sequence = next_sequence
                capture_monotonic_ns = _frame_capture_monotonic_ns(frame)
                if capture_monotonic_ns is not None:
                    diagnostics.note_monitor_master_frame(capture_monotonic_ns)
        except BaseException as exc:
            # Diagnostics must never stop recording. Preserve the error for the
            # next report and keep trying in case this was a transient SDK read.
            diagnostics.note_monitor_error(exc)

        now_s = time.monotonic()
        if now_s - report_started_s < _DIAGNOSTIC_INTERVAL_S:
            continue
        current = diagnostics.snapshot()
        elapsed_s = max(now_s - report_started_s, 1e-6)
        report_started_s = now_s
        if current["active"]:
            try:
                queue_depth: int | str = int(sample_queue.qsize())
            except (AttributeError, NotImplementedError, OSError):
                queue_depth = "?"
            now_ns = time.perf_counter_ns()
            monitor_capture_ns = current["monitor_master_capture_ns"]
            worker_capture_ns = current["worker_master_capture_ns"]
            monitor_age_ms = (
                "-"
                if monitor_capture_ns is None
                else f"{(now_ns - monitor_capture_ns) / 1_000_000:.0f}"
            )
            worker_age_ms = (
                "-"
                if worker_capture_ns is None
                else f"{(now_ns - worker_capture_ns) / 1_000_000:.0f}"
            )
            monitor_hz = max(
                0, current["monitor_master_updates"] - previous["monitor_master_updates"]
            ) / elapsed_s
            worker_hz = max(
                0, current["worker_master_updates"] - previous["worker_master_updates"]
            ) / elapsed_s
            numeric_in = max(0, current["numeric_received"] - previous["numeric_received"])
            numeric_processed = max(
                0, current["numeric_processed"] - previous["numeric_processed"])
            numeric_audits = max(0, current["numeric_audits"] - previous["numeric_audits"])
            message = (
                "[CAMERA-DIAG] "
                f"阶段={current['stage']} 阶段耗时={current['stage_age_ns'] / 1_000_000_000:.2f}s "
                f"主循环未推进={current['loop_age_ns'] / 1_000_000_000:.2f}s | "
                f"主相机独立探针={monitor_hz:.1f}Hz age={monitor_age_ms}ms "
                f"工作循环探针={worker_hz:.1f}Hz age={worker_age_ms}ms | "
                f"数值样本 输入={numeric_in} 处理完成={numeric_processed} 审计={numeric_audits} "
                f"队列={queue_depth}"
            )
            if current["monitor_error"] is not None:
                message += f" | 独立探针错误={current['monitor_error']}"
            print(message, flush=True)
        previous = current


class CameraProcessDatasetRecorder:
    """Record camera-driven episodes while keeping vendor I/O out of ServoJ.

    The parent process periodically samples small robot state/action/gripper
    values. A spawned child process owns the camera adapter, selects a causal
    numeric sample for each new master-camera frame, and writes the dataset.
    ``camera_adapter`` must be pickle-safe because it is opened only in that
    child process.
    """

    def __init__(
        self,
        config: RecorderConfig,
        camera_adapter: CameraAdapterConfig,
        *,
        master_camera_name: str | None = None,
        numeric_sample_fps: int | None = None,
        startup_timeout_s: float = 20.0,
        diagnostics_enabled: bool = False,
    ) -> None:
        if startup_timeout_s <= 0.0:
            raise ValueError("startup_timeout_s must be positive")
        if not isinstance(diagnostics_enabled, bool):
            raise ValueError("diagnostics_enabled must be a boolean")
        if numeric_sample_fps is not None and (
            isinstance(numeric_sample_fps, bool) or not isinstance(numeric_sample_fps, int)
        ):
            raise ValueError("numeric_sample_fps must be a positive integer")
        selected_numeric_fps = config.fps if numeric_sample_fps is None else numeric_sample_fps
        if selected_numeric_fps <= 0:
            raise ValueError("numeric_sample_fps must be a positive integer")

        names = tuple(camera_adapter.camera_names)
        if not names:
            raise ValueError("CameraProcessDatasetRecorder requires at least one camera")
        if len(names) != len(set(names)):
            raise ValueError("Camera adapter camera names must be unique")
        if len(names) == 1:
            selected_master = master_camera_name or names[0]
        else:
            if master_camera_name is None:
                available = ", ".join(names)
                raise ValueError(
                    "master_camera_name is required when multiple cameras are configured "
                    f"(available: {available})"
                )
            selected_master = master_camera_name
        if selected_master not in names:
            available = ", ".join(names)
            raise ValueError(f"Unknown master camera: {selected_master} (available: {available})")

        self.config = config
        self._camera_adapter = camera_adapter
        self._camera_names = names
        self._master_camera_name = selected_master
        self._numeric_sample_fps = selected_numeric_fps
        self._startup_timeout_s = startup_timeout_s
        self._diagnostics_enabled = diagnostics_enabled
        self._context = mp.get_context("spawn")
        self._control_queue: Any = self._context.Queue()
        # This queue carries only small numeric samples. Its short burst buffer
        # absorbs dataset I/O jitter without allowing stale state to drift far
        # from the camera frame selected in the worker.
        self._sample_queue: Any = self._context.Queue(maxsize=config.queue_maxsize)
        self._response_queue: Any = self._context.Queue()
        self._error_queue: Any = self._context.Queue()
        self._status_queue: Any = self._context.Queue()
        self._process: mp.Process | None = None
        self._follower: RecordingFollower | None = None
        self._action_lock = threading.Lock()
        self._latest_action: tuple[np.ndarray, int] | None = None
        self._latest_gripper_action: tuple[float, int] | None = None
        self._state_lock = threading.Lock()
        self._active = False
        self._stop_event: threading.Event | None = None
        self._sampler: threading.Thread | None = None
        self._request_lock = threading.Lock()
        self._next_request_id = 1
        self._pending_responses: dict[int, _WorkerResponse] = {}
        self._failure: RuntimeError | None = None
        self._pending_sample_queue_drops = 0
        self._health_lock = threading.Lock()
        self._worker_state = "unprepared"
        self._last_heartbeat_monotonic_ns: int | None = None
        self._last_master_capture_monotonic_ns: int | None = None
        self._closed = False

    @property
    def active(self) -> bool:
        with self._state_lock:
            return self._active

    @property
    def ready(self) -> bool:
        """Whether at least one follower joint command has been accepted."""

        return self._current_action() is not None

    @property
    def master_camera_name(self) -> str:
        """Return the declared camera that drives dataset-row emission."""

        return self._master_camera_name

    @property
    def numeric_sample_fps(self) -> int:
        """Return the parent-side robot and gripper sampling frequency."""

        return self._numeric_sample_fps

    def prepare(self) -> CameraRecorderHealth:
        """Open the worker-owned cameras after external preflight succeeds."""

        with self._state_lock:
            if self._follower is None:
                raise RuntimeError("Wrap a FollowerArm with RecordingFollower before preparing cameras")
            joint_count = self._follower.joint_count
        self._start_worker(joint_count)
        return self.check_health()

    def check_health(self) -> CameraRecorderHealth:
        """Drain worker status without blocking the teleoperation control path."""

        self._drain_worker_status()
        self._drain_worker_errors()
        process = self._process
        worker_alive = process is not None and process.is_alive()
        active = self.active
        with self._health_lock:
            if process is not None and not worker_alive and not self._closed and self._failure is None:
                self._failure = RuntimeError("Camera recording worker exited unexpectedly")
                self._worker_state = "fault"
            error = None if self._failure is None else str(self._failure)
            return CameraRecorderHealth(
                state=self._worker_state,
                worker_alive=worker_alive,
                active=active,
                last_heartbeat_monotonic_ns=self._last_heartbeat_monotonic_ns,
                last_master_capture_monotonic_ns=self._last_master_capture_monotonic_ns,
                error=error,
            )

    def wait_until_camera_ready(self, timeout_s: float = 10.0) -> CameraRecorderHealth:
        """Wait for one fresh master-camera frame during preflight."""

        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        self.prepare()
        deadline = time.monotonic() + timeout_s
        while True:
            health = self.check_health()
            if not health.healthy:
                raise RuntimeError(health.error or "Camera recording worker is unhealthy")
            if health.last_master_capture_monotonic_ns is not None:
                return health
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for a fresh master camera frame")
            time.sleep(0.02)

    def attach_follower(self, follower: RecordingFollower) -> None:
        with self._state_lock:
            if self._follower is not None and self._follower is not follower:
                raise RuntimeError("A recorder can only be attached to one follower")
            if self._active:
                raise RuntimeError("Cannot replace the follower during an active episode")
            self._follower = follower

    def start_episode(self, task: str) -> int:
        self.prepare()
        self._raise_if_failed()
        with self._state_lock:
            if self._active:
                raise RuntimeError("An episode is already active")
            if self._follower is None:
                raise RuntimeError("Wrap a FollowerArm with RecordingFollower before starting")
            if self._current_action() is None:
                raise RuntimeError("Cannot record before the follower accepts at least one joint command")
        response = self._request("start", task=task)
        assert response.value is not None
        with self._state_lock:
            self._stop_event = threading.Event()
            self._active = True
            self._sampler = threading.Thread(
                target=self._sample_loop,
                name="leobot-camera-numeric-sampler",
                daemon=True,
            )
            self._sampler.start()
        return response.value

    def stop_episode(self) -> int:
        self._stop_sampler()
        try:
            response = self._request("finish", drain_sample_queue=True)
        finally:
            with self._state_lock:
                self._active = False
        self._raise_if_failed()
        assert response.value is not None
        return response.value

    def discard_episode(self) -> None:
        self._stop_sampler()
        try:
            self._request("discard", drain_sample_queue=True)
        finally:
            with self._state_lock:
                self._active = False

    def close(self) -> None:
        """Release the worker-owned camera session before returning.

        ``close`` is the final ownership boundary for the child process. It
        first asks the worker to discard an incomplete episode and run its
        normal ``session.close()`` path. A wedged worker is then terminated so
        an abandoned SDK pipeline cannot keep the camera device open.
        """

        if self._closed:
            return
        if self.active:
            try:
                self._discard_for_close()
            except Exception:
                # Continue to process shutdown below. It will terminate a
                # worker that could not acknowledge the discard request.
                pass
        process = self._process
        if process is None:
            with self._health_lock:
                self._closed = True
                self._worker_state = "closed"
            self._close_queues()
            return
        try:
            self._request("shutdown", timeout_s=_WORKER_CLOSE_TIMEOUT_S)
        except Exception:
            process.terminate()
        process.join(timeout=_WORKER_CLOSE_TIMEOUT_S)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=2.0)
        self._process = None
        try:
            process.close()
        except ValueError:
            pass
        with self._health_lock:
            self._closed = True
            self._worker_state = "closed"
        self._close_queues()

    def _discard_for_close(self) -> None:
        """Bound active-episode shutdown so application exit cannot hang forever."""

        self._stop_sampler()
        try:
            self._request(
                "discard",
                timeout_s=_WORKER_CLOSE_TIMEOUT_S,
                drain_sample_queue=True,
            )
        finally:
            with self._state_lock:
                self._active = False

    def _close_queues(self) -> None:
        for worker_queue in (
            self._control_queue,
            self._sample_queue,
            self._response_queue,
            self._error_queue,
            self._status_queue,
        ):
            try:
                worker_queue.close()
                worker_queue.cancel_join_thread()
            except (AttributeError, OSError, ValueError):
                pass

    def note_joint_action(self, angles_deg: np.ndarray, sent_monotonic_ns: int) -> None:
        with self._action_lock:
            self._latest_action = (
                np.asarray(angles_deg, dtype=float).copy(),
                int(sent_monotonic_ns),
            )

    def note_gripper_action(self, opening: float, sent_monotonic_ns: int) -> None:
        with self._action_lock:
            self._latest_gripper_action = (float(opening), int(sent_monotonic_ns))

    def _start_worker(self, joint_count: int) -> None:
        if self._process is not None:
            return
        process = self._context.Process(
            target=_run_camera_recording_worker,
            args=(
                self._camera_adapter,
                self._camera_names,
                self._master_camera_name,
                self.config.root,
                self.config.robot_type,
                self.config.fps,
                joint_count,
                self.config.gripper_source is not None,
                self.config.image_storage,
                self.config.quality,
                self._control_queue,
                self._sample_queue,
                self._response_queue,
                self._error_queue,
                self._status_queue,
                self._diagnostics_enabled,
            ),
            name="leobot-camera-recorder",
            daemon=False,
        )
        self._process = process
        with self._health_lock:
            self._closed = False
            self._worker_state = "starting"
            self._last_heartbeat_monotonic_ns = None
            self._last_master_capture_monotonic_ns = None
        process.start()
        response = self._wait_response(0, self._startup_timeout_s)
        if response.ok:
            self._drain_worker_status()
            return
        self.close()
        raise RuntimeError(f"Camera recording worker failed to start: {response.error}")

    def _request(
        self,
        kind: str,
        *,
        task: str | None = None,
        timeout_s: float = 30.0,
        drain_sample_queue: bool = False,
    ) -> _WorkerResponse:
        self._raise_if_failed()
        process = self._process
        if process is None or not process.is_alive():
            raise RuntimeError("Camera recording worker is not running")
        with self._request_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            if drain_sample_queue:
                try:
                    self._sample_queue.put(_WorkerBarrier(request_id), timeout=timeout_s)
                except queue.Full as exc:
                    raise TimeoutError("Timed out placing the camera recording queue drain barrier") from exc
            self._control_queue.put(_WorkerControl(request_id=request_id, kind=kind, task=task))
            response = self._wait_response(request_id, timeout_s)
        if not response.ok:
            raise RuntimeError(response.error or f"Camera recording worker rejected {kind}")
        return response

    def _wait_response(self, request_id: int, timeout_s: float) -> _WorkerResponse:
        pending = self._pending_responses.pop(request_id, None)
        if pending is not None:
            return pending
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"Timed out waiting for camera worker request {request_id}")
            try:
                response = self._response_queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(f"Timed out waiting for camera worker request {request_id}") from exc
            if response.request_id == request_id:
                return response
            self._pending_responses[response.request_id] = response

    def _raise_if_failed(self) -> None:
        self._drain_worker_errors()
        if self._failure is not None:
            raise self._failure

    def _drain_worker_errors(self) -> None:
        while True:
            try:
                message = self._error_queue.get_nowait()
            except queue.Empty:
                break
            with self._health_lock:
                self._failure = RuntimeError(f"Camera recording worker failed: {message}")
                self._worker_state = "fault"

    def _drain_worker_status(self) -> None:
        while True:
            try:
                status = self._status_queue.get_nowait()
            except queue.Empty:
                return
            if not isinstance(status, _WorkerStatus):
                continue
            with self._health_lock:
                self._worker_state = status.state
                self._last_heartbeat_monotonic_ns = status.heartbeat_monotonic_ns
                if status.last_master_capture_monotonic_ns is not None:
                    self._last_master_capture_monotonic_ns = status.last_master_capture_monotonic_ns
                if status.error is not None:
                    self._failure = RuntimeError(f"Camera recording worker failed: {status.error}")
                    self._worker_state = "fault"

    def _current_action(self) -> tuple[np.ndarray, int] | None:
        with self._action_lock:
            if self._latest_action is None:
                return None
            angles, timestamp = self._latest_action
            return angles.copy(), timestamp

    def _current_gripper_action(self) -> tuple[float, int] | None:
        with self._action_lock:
            return self._latest_gripper_action

    def _sample_loop(self) -> None:
        assert self._follower is not None
        assert self._stop_event is not None
        period_ns = int(1_000_000_000 / self._numeric_sample_fps)
        scheduled_ns = time.perf_counter_ns()
        tick_index = 0
        while not self._stop_event.is_set():
            delay_ns = scheduled_ns - time.perf_counter_ns()
            if delay_ns > 0:
                self._stop_event.wait(delay_ns / 1_000_000_000)
                if self._stop_event.is_set():
                    break
            audit: dict[str, Any] = {
                "tick_index": tick_index,
                "scheduled_monotonic_ns": scheduled_ns,
                "numeric_sample_fps": self._numeric_sample_fps,
                "sample_started_monotonic_ns": time.perf_counter_ns(),
            }
            try:
                snapshot = self._follower.recording_snapshot()
                if snapshot is None:
                    self._enqueue_sample(_WorkerAudit({**audit, "skip_reason": "missing_cached_follower_state"}))
                else:
                    state, state_read_started_ns, state_read_finished_ns = snapshot
                    action = self._current_action()
                    audit["state_read_started_monotonic_ns"] = state_read_started_ns
                    audit["state_read_finished_monotonic_ns"] = state_read_finished_ns
                    if action is None:
                        self._enqueue_sample(_WorkerAudit({**audit, "skip_reason": "no_successful_joint_action"}))
                    else:
                        gripper_state, gripper_action = self._sample_gripper(audit)
                        if self.config.gripper_source is not None and (
                            gripper_state is None or gripper_action is None
                        ):
                            self._enqueue_sample(
                                _WorkerAudit({**audit, "skip_reason": "missing_gripper_state_or_action"})
                            )
                        else:
                            capture_monotonic_ns = max(state_read_finished_ns, time.perf_counter_ns())
                            audit["numeric_sample_capture_monotonic_ns"] = capture_monotonic_ns
                            audit["joint_action_sent_monotonic_ns"] = action[1]
                            self._enqueue_sample(
                                _WorkerSample(
                                    state=np.asarray(state, dtype=float).copy(),
                                    action=action[0],
                                    gripper_state=gripper_state,
                                    gripper_action=gripper_action,
                                    capture_monotonic_ns=capture_monotonic_ns,
                                    audit=audit,
                                )
                            )
            except BaseException as exc:
                self._enqueue_sample(_WorkerAudit({**audit, "skip_reason": "sampling_error", "error": repr(exc)}))
            tick_index += 1
            scheduled_ns += period_ns

    def _sample_gripper(self, audit: dict[str, Any]) -> tuple[float | None, float | None]:
        source = self.config.gripper_source
        if source is None:
            return None, None
        feedback = source.read_gripper_opening()
        action = self._current_gripper_action()
        if feedback is None or action is None:
            return None, None
        audit["gripper_capture_monotonic_ns"] = feedback.capture_monotonic_ns
        audit["gripper_action_sent_monotonic_ns"] = action[1]
        value = float(np.asarray(feedback.value, dtype=float).reshape(-1)[0])
        return value, action[0]

    def _enqueue_sample(self, item: _WorkerSample | _WorkerAudit) -> None:
        while True:
            queued_item = item
            if self._pending_sample_queue_drops:
                queued_item = replace(
                    item,
                    audit={
                        **item.audit,
                        "parent_sample_queue_dropped_samples": self._pending_sample_queue_drops,
                    },
                )
            try:
                self._sample_queue.put_nowait(queued_item)
            except queue.Full:
                try:
                    self._sample_queue.get_nowait()
                except queue.Empty:
                    # multiprocessing.Queue can briefly report full before its
                    # feeder exposes an item to get_nowait(). Do not block ServoJ.
                    self._pending_sample_queue_drops += 1
                    return
                self._pending_sample_queue_drops += 1
                continue
            self._pending_sample_queue_drops = 0
            return

    def _stop_sampler(self) -> None:
        with self._state_lock:
            if not self._active:
                raise RuntimeError("No active episode")
            stop_event = self._stop_event
            sampler = self._sampler
        assert stop_event is not None and sampler is not None
        stop_event.set()
        sampler.join()
        with self._state_lock:
            self._stop_event = None
            self._sampler = None


def _run_camera_recording_worker(
    camera_adapter: CameraAdapterConfig,
    camera_names: tuple[str, ...],
    master_camera_name: str,
    root: Any,
    robot_type: str,
    fps: int,
    joint_count: int,
    include_gripper: bool,
    image_storage: str,
    quality: int,
    control_queue: Any,
    sample_queue: Any,
    response_queue: Any,
    error_queue: Any,
    status_queue: Any,
    diagnostics_enabled: bool,
) -> None:
    """Own camera capture and dataset I/O outside the teleoperation process."""

    session: CameraAdapterSession | None = None
    writer: V21DatasetWriter | None = None
    active = False
    worker_state = "starting"
    last_master_capture_monotonic_ns: int | None = None
    last_heartbeat_s = 0.0
    diagnostics = _WorkerDiagnostics(diagnostics_enabled)
    diagnostic_stop_event = threading.Event()
    diagnostic_thread: threading.Thread | None = None

    def publish_status(*, force: bool = False, error: str | None = None) -> None:
        nonlocal last_heartbeat_s
        now_s = time.monotonic()
        if not force and now_s - last_heartbeat_s < _WORKER_HEARTBEAT_INTERVAL_S:
            return
        status_queue.put(
            _WorkerStatus(
                state="fault" if error is not None else worker_state,
                active=active,
                heartbeat_monotonic_ns=time.perf_counter_ns(),
                last_master_capture_monotonic_ns=last_master_capture_monotonic_ns,
                error=error,
            )
        )
        last_heartbeat_s = now_s

    def note_master_frame(capture_monotonic_ns: int) -> None:
        nonlocal last_master_capture_monotonic_ns
        last_master_capture_monotonic_ns = capture_monotonic_ns
        diagnostics.note_worker_master_frame(capture_monotonic_ns)
    try:
        session = camera_adapter.open()
        raw_sources = dict(session.sources)
        missing = [name for name in camera_names if name not in raw_sources]
        extras = [name for name in raw_sources if name not in camera_names]
        if missing or extras:
            raise ValueError(
                "Camera adapter sources differ from its declaration "
                f"(missing: {missing}, unexpected: {extras})"
            )
        sources = {
            _rgb_feature_name(name): raw_sources[name]
            for name in camera_names
        }
        master_feature_name = _rgb_feature_name(master_camera_name)
        if master_feature_name not in sources:
            raise ValueError(f"Configured master camera is unavailable: {master_camera_name}")
        cameras = tuple(CameraSpec(name, source.shape) for name, source in sources.items())
        depth_cameras = tuple(
            DepthCameraSpec(
                feature_name=_depth_feature_name(name),
                rgb_feature_name=name,
                shape=source.depth_shape,
                metadata=source.metadata,
            )
            for name, source in sources.items()
            if isinstance(source, RGBDFrameSource)
        )
        writer = V21DatasetWriter(
            WriterConfig(
                root=root,
                robot_type=robot_type,
                fps=fps,
                joint_count=joint_count,
                include_gripper=include_gripper,
                cameras=cameras,
                depth_cameras=depth_cameras,
                image_storage=image_storage,  # type: ignore[arg-type]
                quality=quality,
            )
        )
        diagnostic_thread = threading.Thread(
            target=_run_worker_diagnostic_monitor,
            args=(diagnostics, sources[master_feature_name], sample_queue, diagnostic_stop_event),
            name="leobot-camera-diagnostics",
            daemon=True,
        )
        diagnostic_thread.start()
        worker_state = "ready"
        publish_status(force=True)
        response_queue.put(_WorkerResponse(request_id=0, ok=True))
    except BaseException as exc:
        publish_status(force=True, error=repr(exc))
        response_queue.put(_WorkerResponse(request_id=0, ok=False, error=repr(exc)))
        if session is not None:
            session.close()
        return

    try:
        seen_barriers: set[int] = set()
        numeric_history: deque[_WorkerSample] = deque(maxlen=_NUMERIC_HISTORY_CAPACITY)
        master_sequence = 0
        probe_sequence = sources[master_feature_name].latest_sequence()
        timeline = _EpisodeTimeline()
        while True:
            diagnostics.set_stage("master_health_probe")
            try:
                probe_sequence = _probe_master_camera(
                    sources[master_feature_name],
                    probe_sequence,
                    note_master_frame,
                )
            except BaseException as exc:
                if active:
                    try:
                        assert writer is not None
                        writer.discard_episode()
                    except BaseException:
                        pass
                    active = False
                    diagnostics.set_active(False)
                worker_state = "fault"
                error_queue.put(repr(exc))
                publish_status(force=True, error=repr(exc))
                return
            diagnostics.note_loop_progress()
            try:
                diagnostics.set_stage("control_queue.get_nowait")
                control = control_queue.get_nowait()
            except queue.Empty:
                control = None
            if control is not None:
                try:
                    if control.kind == "start":
                        if active:
                            raise RuntimeError("An episode is already active")
                        if control.task is None:
                            raise ValueError("start requires a task")
                        assert writer is not None
                        diagnostics.set_stage("writer.begin_episode")
                        episode = writer.begin_episode(control.task)
                        numeric_history.clear()
                        timeline = _EpisodeTimeline()
                        master_sequence = sources[master_feature_name].latest_sequence()
                        probe_sequence = master_sequence
                        active = True
                        diagnostics.set_active(True)
                        worker_state = "recording"
                        publish_status(force=True)
                        response_queue.put(_WorkerResponse(control.request_id, ok=True, value=episode))
                    elif control.kind == "finish":
                        if not active:
                            raise RuntimeError("No active episode")
                        worker_state = "finalizing"
                        publish_status(force=True)
                        diagnostics.set_stage("drain_worker_samples")
                        master_sequence = _drain_worker_samples(
                            sample_queue,
                            writer,
                            sources,
                            master_feature_name,
                            numeric_history,
                            master_sequence,
                            timeline=timeline,
                            barrier_request_id=control.request_id,
                            seen_barriers=seen_barriers,
                            append=True,
                            on_master_frame=note_master_frame,
                        )
                        master_sequence = _append_available_master_frames(
                            writer,
                            sources,
                            master_feature_name,
                            numeric_history,
                            master_sequence,
                            timeline=timeline,
                            maximum_capture_monotonic_ns=_latest_sample_capture(numeric_history),
                            on_master_frame=note_master_frame,
                        )
                        diagnostics.set_stage("writer.finish_episode")
                        episode = writer.finish_episode()
                        active = False
                        diagnostics.set_active(False)
                        worker_state = "ready"
                        publish_status(force=True)
                        response_queue.put(_WorkerResponse(control.request_id, ok=True, value=episode))
                    elif control.kind == "discard":
                        if active:
                            assert writer is not None
                            worker_state = "finalizing"
                            publish_status(force=True)
                            diagnostics.set_stage("drain_worker_samples")
                            _drain_worker_samples(
                                sample_queue,
                                writer,
                                sources,
                                master_feature_name,
                                numeric_history,
                                master_sequence,
                                timeline=timeline,
                                barrier_request_id=control.request_id,
                                seen_barriers=seen_barriers,
                                append=False,
                            )
                            diagnostics.set_stage("writer.discard_episode")
                            writer.discard_episode()
                            active = False
                            diagnostics.set_active(False)
                            worker_state = "ready"
                            publish_status(force=True)
                        response_queue.put(_WorkerResponse(control.request_id, ok=True))
                    elif control.kind == "shutdown":
                        if active:
                            assert writer is not None
                            diagnostics.set_stage("writer.discard_episode")
                            writer.discard_episode()
                            active = False
                            diagnostics.set_active(False)
                        worker_state = "closed"
                        publish_status(force=True)
                        response_queue.put(_WorkerResponse(control.request_id, ok=True))
                        return
                    else:
                        raise ValueError(f"Unknown worker control command: {control.kind}")
                except BaseException as exc:
                    # A failed finalization must not leave an active staging
                    # episode behind before the parent reports the failure.
                    if control.kind in {"finish", "discard", "shutdown"} and active:
                        try:
                            assert writer is not None
                            writer.discard_episode()
                        except BaseException:
                            pass
                        active = False
                        diagnostics.set_active(False)
                    worker_state = "fault"
                    publish_status(force=True, error=repr(exc))
                    response_queue.put(_WorkerResponse(control.request_id, ok=False, error=repr(exc)))
                continue

            try:
                diagnostics.set_stage("sample_queue.get")
                sample = sample_queue.get(timeout=0.05)
            except queue.Empty:
                diagnostics.note_loop_progress()
                publish_status()
                continue
            if isinstance(sample, _WorkerBarrier):
                diagnostics.note_loop_progress()
                seen_barriers.add(sample.request_id)
                continue
            if not active:
                diagnostics.note_loop_progress()
                publish_status()
                continue
            diagnostics.note_numeric_sample(sample)
            try:
                diagnostics.set_stage("consume_worker_sample")
                master_sequence = _consume_worker_sample(
                    writer,
                    sources,
                    master_feature_name,
                    numeric_history,
                    master_sequence,
                    sample,
                    timeline=timeline,
                    on_master_frame=note_master_frame,
                )
                diagnostics.note_processed_sample()
                diagnostics.note_loop_progress()
                publish_status()
            except BaseException as exc:
                error_queue.put(repr(exc))
                assert writer is not None
                diagnostics.set_stage("writer.discard_episode")
                writer.discard_episode()
                active = False
                diagnostics.set_active(False)
                worker_state = "fault"
                publish_status(force=True, error=repr(exc))
    finally:
        diagnostic_stop_event.set()
        if diagnostic_thread is not None:
            diagnostic_thread.join(timeout=1.0)
        if session is not None:
            session.close()


def _probe_master_camera(
    source: CameraFrameSource,
    sequence: int,
    on_master_frame: Callable[[int], None],
) -> int:
    """Advance a health-only cursor without changing the recording cursor."""

    while True:
        item = source.next_frame_after(sequence)
        if item is None:
            return sequence
        next_sequence, frame = item
        if next_sequence <= sequence:
            raise RuntimeError("Master camera source returned a non-increasing sequence")
        sequence = next_sequence
        capture_monotonic_ns = _frame_capture_monotonic_ns(frame)
        if capture_monotonic_ns is not None:
            on_master_frame(capture_monotonic_ns)


def _drain_worker_samples(
    sample_queue: Any,
    writer: V21DatasetWriter | None,
    sources: Mapping[str, CameraFrameSource],
    master_feature_name: str,
    numeric_history: deque[_WorkerSample],
    master_sequence: int,
    *,
    barrier_request_id: int,
    seen_barriers: set[int],
    append: bool,
    timeline: _EpisodeTimeline,
    on_master_frame: Callable[[int], None] | None = None,
) -> int:
    while True:
        if barrier_request_id in seen_barriers:
            seen_barriers.remove(barrier_request_id)
            return master_sequence
        try:
            sample = sample_queue.get(timeout=30.0)
        except queue.Empty as exc:
            raise TimeoutError(
                f"Timed out waiting for camera recording queue drain barrier {barrier_request_id}"
            ) from exc
        if isinstance(sample, _WorkerBarrier):
            seen_barriers.add(sample.request_id)
            continue
        if append:
            master_sequence = _consume_worker_sample(
                writer,
                sources,
                master_feature_name,
                numeric_history,
                master_sequence,
                sample,
                timeline=timeline,
                on_master_frame=on_master_frame,
            )


def _consume_worker_sample(
    writer: V21DatasetWriter | None,
    sources: Mapping[str, CameraFrameSource],
    master_feature_name: str,
    numeric_history: deque[_WorkerSample],
    master_sequence: int,
    sample: _WorkerSample | _WorkerAudit,
    *,
    timeline: _EpisodeTimeline | None = None,
    on_master_frame: Callable[[int], None] | None = None,
) -> int:
    """Store a numeric sample, then emit each available new master frame once."""

    assert writer is not None
    active_timeline = _EpisodeTimeline() if timeline is None else timeline
    if isinstance(sample, _WorkerAudit):
        _append_worker_sample(writer, sources, sample, timeline=active_timeline)
        return master_sequence

    prepared_sample = replace(sample, audit=_record_queue_overflow(writer, sample.audit))
    numeric_history.append(prepared_sample)
    return _append_available_master_frames(
        writer,
        sources,
        master_feature_name,
        numeric_history,
        master_sequence,
        timeline=active_timeline,
        on_master_frame=on_master_frame,
    )


def _latest_sample_capture(samples: deque[_WorkerSample]) -> int | None:
    if not samples:
        return None
    return samples[-1].capture_monotonic_ns


def _latest_sample_at_or_before(
    samples: deque[_WorkerSample], target_monotonic_ns: int
) -> _WorkerSample | None:
    for sample in reversed(samples):
        if sample.capture_monotonic_ns <= target_monotonic_ns:
            return sample
    return None


def _frame_capture_monotonic_ns(frame: CameraFrame | RGBDFrame) -> int | None:
    if isinstance(frame, CameraFrame):
        return frame.capture_monotonic_ns
    if frame.rgb_capture_monotonic_ns is not None:
        return frame.rgb_capture_monotonic_ns
    return frame.depth.capture_monotonic_ns


def _frame_source_timestamp_ns(frame: CameraFrame | RGBDFrame) -> int | None:
    if isinstance(frame, CameraFrame):
        return frame.source_timestamp_ns
    if frame.rgb_source_timestamp_ns is not None:
        return frame.rgb_source_timestamp_ns
    return frame.depth.source_timestamp_ns


def _frame_source_frame_index(frame: CameraFrame | RGBDFrame) -> int | None:
    if isinstance(frame, CameraFrame):
        return frame.source_frame_index
    if frame.rgb_source_frame_index is not None:
        return frame.rgb_source_frame_index
    return frame.depth.source_frame_index


def _recorded_camera_frame(
    frame: CameraFrame | RGBDFrame,
    timeline: _EpisodeTimeline,
    row_timestamp_s: float,
) -> CameraFrame:
    capture_monotonic_ns = _frame_capture_monotonic_ns(frame)
    return CameraFrame(
        rgb=frame.rgb,
        timestamp_s=timeline.camera_timestamp_s(capture_monotonic_ns, row_timestamp_s),
        capture_monotonic_ns=capture_monotonic_ns,
        source_timestamp_ns=_frame_source_timestamp_ns(frame),
        source_frame_index=_frame_source_frame_index(frame),
    )


def _append_available_master_frames(
    writer: V21DatasetWriter | None,
    sources: Mapping[str, CameraFrameSource],
    master_feature_name: str,
    numeric_history: deque[_WorkerSample],
    master_sequence: int,
    *,
    maximum_capture_monotonic_ns: int | None = None,
    timeline: _EpisodeTimeline | None = None,
    on_master_frame: Callable[[int], None] | None = None,
) -> int:
    """Write one row per retained new master frame, in capture order."""

    assert writer is not None
    active_timeline = _EpisodeTimeline() if timeline is None else timeline
    master_source = sources[master_feature_name]
    while True:
        item = master_source.next_frame_after(master_sequence)
        if item is None:
            return master_sequence
        next_sequence, master_frame = item
        target_monotonic_ns = _frame_capture_monotonic_ns(master_frame)
        if target_monotonic_ns is None:
            master_sequence = next_sequence
            writer.append_skipped_tick(
                {
                    "skip_reason": "master_camera_frame_has_no_capture_timestamp",
                    f"{master_feature_name}.source_frame_index": _frame_source_frame_index(master_frame),
                }
            )
            continue
        if on_master_frame is not None:
            on_master_frame(target_monotonic_ns)
        if (
            maximum_capture_monotonic_ns is not None
            and target_monotonic_ns > maximum_capture_monotonic_ns
        ):
            return master_sequence
        master_sequence = next_sequence
        sample = _latest_sample_at_or_before(numeric_history, target_monotonic_ns)
        if sample is None:
            writer.append_skipped_tick(
                {
                    "skip_reason": "missing_numeric_sample_for_master_frame",
                    "target_monotonic_ns": target_monotonic_ns,
                    f"{master_feature_name}.capture_monotonic_ns": target_monotonic_ns,
                    f"{master_feature_name}.source_timestamp_ns": _frame_source_timestamp_ns(master_frame),
                    f"{master_feature_name}.source_frame_index": _frame_source_frame_index(master_frame),
                }
            )
            continue
        _append_worker_sample(
            writer,
            sources,
            sample,
            target_monotonic_ns=target_monotonic_ns,
            master_feature_name=master_feature_name,
            master_frame=master_frame,
            timeline=active_timeline,
        )


def _record_queue_overflow(writer: V21DatasetWriter, audit: dict[str, Any]) -> dict[str, Any]:
    """Persist parent-queue drops once before a numeric sample enters history."""

    prepared = dict(audit)
    dropped_samples = prepared.pop("parent_sample_queue_dropped_samples", 0)
    if isinstance(dropped_samples, int) and not isinstance(dropped_samples, bool) and dropped_samples > 0:
        writer.append_skipped_tick(
            {
                "skip_reason": "parent_sample_queue_overflow",
                "dropped_sample_count": dropped_samples,
                "before_tick_index": prepared.get("tick_index"),
            }
        )
    return prepared


def _append_worker_sample(
    writer: V21DatasetWriter | None,
    sources: Mapping[str, CameraFrameSource],
    sample: _WorkerSample | _WorkerAudit,
    *,
    target_monotonic_ns: int | None = None,
    master_feature_name: str | None = None,
    master_frame: CameraFrame | RGBDFrame | None = None,
    timeline: _EpisodeTimeline | None = None,
) -> None:
    assert writer is not None
    if isinstance(sample, _WorkerAudit):
        writer.append_skipped_tick(_record_queue_overflow(writer, sample.audit))
        return
    audit = _record_queue_overflow(writer, sample.audit)
    selected_target_monotonic_ns = (
        sample.capture_monotonic_ns if target_monotonic_ns is None else target_monotonic_ns
    )
    active_timeline = _EpisodeTimeline() if timeline is None else timeline
    row_timestamp_s = active_timeline.timestamp_s(selected_target_monotonic_ns)
    # Keep the former audit key for existing analysis scripts while making the
    # physical meaning explicit for new recordings.
    audit["sample_target_monotonic_ns"] = sample.capture_monotonic_ns
    audit["numeric_sample_capture_monotonic_ns"] = sample.capture_monotonic_ns
    audit["target_monotonic_ns"] = selected_target_monotonic_ns
    audit["episode_origin_monotonic_ns"] = active_timeline.origin_monotonic_ns
    audit["timestamp_s"] = row_timestamp_s
    audit["state_sample_age_ns"] = selected_target_monotonic_ns - sample.capture_monotonic_ns
    action_timestamp = audit.get("joint_action_sent_monotonic_ns")
    if isinstance(action_timestamp, int) and not isinstance(action_timestamp, bool):
        audit["joint_action_age_ns"] = selected_target_monotonic_ns - action_timestamp
    gripper_timestamp = audit.get("gripper_capture_monotonic_ns")
    if isinstance(gripper_timestamp, int) and not isinstance(gripper_timestamp, bool):
        audit["gripper_capture_age_ns"] = selected_target_monotonic_ns - gripper_timestamp
    cameras: dict[str, CameraFrame] = {}
    depths: dict[str, DepthFrame] = {}
    for name, source in sources.items():
        frame = master_frame if name == master_feature_name else source.frame_at_or_before(selected_target_monotonic_ns)
        if frame is None:
            writer.append_skipped_tick(
                {
                    **audit,
                    "skip_reason": f"missing_camera_for_target:{name}",
                }
            )
            return
        rgb = _recorded_camera_frame(frame, active_timeline, row_timestamp_s)
        cameras[name] = rgb
        audit[f"{name}.capture_monotonic_ns"] = rgb.capture_monotonic_ns
        audit[f"{name}.source_timestamp_ns"] = rgb.source_timestamp_ns
        audit[f"{name}.source_frame_index"] = rgb.source_frame_index
        audit[f"{name}.capture_age_ns"] = (
            None
            if rgb.capture_monotonic_ns is None
            else selected_target_monotonic_ns - rgb.capture_monotonic_ns
        )
        if master_feature_name is not None and rgb.capture_monotonic_ns is not None:
            audit[f"{name}.capture_delta_from_master_ns"] = (
                rgb.capture_monotonic_ns - selected_target_monotonic_ns
            )
        if isinstance(frame, RGBDFrame):
            depth_name = _depth_feature_name(name)
            depths[depth_name] = frame.depth
            audit[f"{depth_name}.capture_monotonic_ns"] = frame.depth.capture_monotonic_ns
            audit[f"{depth_name}.source_timestamp_ns"] = frame.depth.source_timestamp_ns
            audit[f"{depth_name}.source_frame_index"] = frame.depth.source_frame_index
    writer.append_frame(
        RecordedFrame(
            state=sample.state,
            action=sample.action,
            timestamp_s=row_timestamp_s,
            cameras=cameras,
            depths=depths,
            gripper_state=sample.gripper_state,
            gripper_action=sample.gripper_action,
            audit=audit,
        )
    )


def _depth_feature_name(rgb_feature_name: str) -> str:
    suffix = rgb_feature_name.removeprefix("observation.images.")
    return f"observation.depth.{suffix}"


def _rgb_feature_name(camera_name: str) -> str:
    return f"observation.images.{camera_name}"
