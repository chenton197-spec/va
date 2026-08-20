"""Head-camera-triggered dataset recording for a live teleoperation session.

The camera child process owns all camera SDK and dataset writer resources.  A
new master-camera frame asks the parent process for one numeric snapshot; it
does not start a periodic robot-feedback loop.  This keeps collection work out
of a follower's high-rate command path while retaining the capture timestamp
needed to audit that feedback was read after the image was captured.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .camera import (
    CameraAdapterConfig,
    CameraAdapterSession,
    CameraFrameSource,
    CameraSourceHealth,
    RGBDFrameSource,
)
from .camera_recorder import CameraRecorderHealth
from .recorder import RecorderConfig
from .sources import CameraFrame, DepthFrame, RGBDFrame
from .v21_writer import CameraSpec, DepthCameraSpec, RecordedFrame, V21DatasetWriter, WriterConfig


_HEARTBEAT_INTERVAL_S = 0.5
_WORKER_CLOSE_TIMEOUT_S = 5.0
_PARENT_REQUEST_POLL_S = 0.05
_CHILD_IDLE_SLEEP_S = 0.002


@dataclass(frozen=True)
class MasterFrameRequest:
    """A small child-to-parent request for one captured master image.

    The image itself remains in the camera child.  The parent only receives
    timing metadata and therefore never owns camera buffers or image writes.
    """

    sequence: int
    capture_monotonic_ns: int
    emitted_monotonic_ns: int


@dataclass(frozen=True)
class MasterFrameSnapshot:
    """One parent-side state/action snapshot returned for a master frame."""

    state: np.ndarray
    action: np.ndarray
    actuator_states: dict[str, float] = field(default_factory=dict)
    actuator_actions: dict[str, float] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MasterFrameSkip:
    """A normal, audited per-frame skip returned by the snapshot provider."""

    reason: str
    audit: dict[str, Any] = field(default_factory=dict)


MasterFrameSnapshotProvider = Callable[
    [MasterFrameRequest], MasterFrameSnapshot | MasterFrameSkip
]


@dataclass(frozen=True)
class _MasterFrameResponse:
    sequence: int
    snapshot: MasterFrameSnapshot | None = None
    skip_reason: str | None = None
    audit: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


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
    state: str
    active: bool
    heartbeat_monotonic_ns: int
    last_master_capture_monotonic_ns: int | None = None
    source_health: dict[str, CameraSourceHealth] = field(default_factory=dict)
    error: str | None = None


@dataclass
class _PendingMasterFrame:
    request: MasterFrameRequest
    master_feature_name: str
    frame: CameraFrame | RGBDFrame
    request_sent_monotonic_ns: int


@dataclass
class _EpisodeTimeline:
    """Convert host monotonic timestamps into one episode-relative timeline."""

    origin_monotonic_ns: int | None = None

    def timestamp_s(self, capture_monotonic_ns: int) -> float:
        if self.origin_monotonic_ns is None:
            self.origin_monotonic_ns = capture_monotonic_ns
        return (capture_monotonic_ns - self.origin_monotonic_ns) / 1_000_000_000

    def camera_timestamp_s(
        self, capture_monotonic_ns: int | None, row_timestamp_s: float
    ) -> float:
        if capture_monotonic_ns is None:
            return row_timestamp_s
        if self.origin_monotonic_ns is None:
            self.origin_monotonic_ns = capture_monotonic_ns
        return (capture_monotonic_ns - self.origin_monotonic_ns) / 1_000_000_000


class MasterTriggeredCameraProcessDatasetRecorder:
    """Record one row for each head-camera frame without periodic feedback polling.

    The spawned child owns cameras and the :class:`V21DatasetWriter`.  A
    parent-owned snapshot worker invokes ``snapshot_provider`` at most once for
    an in-flight master frame.  If that read takes too long, the child records
    an audit skip and advances its master cursor rather than queueing stale
    historical frames.
    """

    def __init__(
        self,
        config: RecorderConfig,
        camera_adapter: CameraAdapterConfig,
        *,
        joint_count: int,
        master_camera_name: str,
        snapshot_provider: MasterFrameSnapshotProvider,
        scalar_actuator_names: tuple[str, ...] = (),
        ready: Callable[[], bool] | None = None,
        snapshot_timeout_s: float = 0.25,
        startup_timeout_s: float = 20.0,
    ) -> None:
        if not isinstance(joint_count, int) or isinstance(joint_count, bool) or joint_count <= 0:
            raise ValueError("joint_count must be a positive integer")
        if not callable(snapshot_provider):
            raise TypeError("snapshot_provider must be callable")
        if ready is not None and not callable(ready):
            raise TypeError("ready must be callable or None")
        if snapshot_timeout_s <= 0.0:
            raise ValueError("snapshot_timeout_s must be positive")
        if startup_timeout_s <= 0.0:
            raise ValueError("startup_timeout_s must be positive")
        if config.gripper_source is not None:
            raise ValueError(
                "MasterTriggeredCameraProcessDatasetRecorder uses named scalar actuators; "
                "do not set RecorderConfig.gripper_source"
            )

        camera_names = tuple(camera_adapter.camera_names)
        if not camera_names:
            raise ValueError("MasterTriggeredCameraProcessDatasetRecorder requires at least one camera")
        if len(camera_names) != len(set(camera_names)):
            raise ValueError("Camera adapter camera names must be unique")
        if master_camera_name not in camera_names:
            available = ", ".join(camera_names)
            raise ValueError(
                f"Unknown master camera: {master_camera_name} (available: {available})"
            )

        self.config = config
        self._camera_adapter = camera_adapter
        self._camera_names = camera_names
        self._master_camera_name = master_camera_name
        self._joint_count = joint_count
        self._snapshot_provider = snapshot_provider
        self._ready_callback = ready
        self._scalar_actuator_names = tuple(scalar_actuator_names)
        self._snapshot_timeout_s = float(snapshot_timeout_s)
        self._startup_timeout_s = float(startup_timeout_s)

        self._context = mp.get_context("spawn")
        self._control_queue: Any = self._context.Queue()
        # Only one master frame can wait for numeric feedback.  A full queue is
        # an audited drop, never a reason to block a servo-adjacent thread.
        self._master_request_queue: Any = self._context.Queue(maxsize=1)
        self._master_response_queue: Any = self._context.Queue(maxsize=2)
        self._response_queue: Any = self._context.Queue()
        self._error_queue: Any = self._context.Queue()
        self._status_queue: Any = self._context.Queue()

        self._process: mp.Process | None = None
        self._snapshot_stop_event = threading.Event()
        self._snapshot_thread: threading.Thread | None = None
        self._request_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._health_lock = threading.Lock()
        self._next_request_id = 1
        self._pending_responses: dict[int, _WorkerResponse] = {}
        self._active = False
        self._closed = False
        self._failure: RuntimeError | None = None
        self._worker_state = "unprepared"
        self._last_heartbeat_monotonic_ns: int | None = None
        self._last_master_capture_monotonic_ns: int | None = None
        self._source_health: dict[str, CameraSourceHealth] = {}

    @property
    def active(self) -> bool:
        with self._state_lock:
            return self._active

    @property
    def ready(self) -> bool:
        """Whether the caller has enough accepted upstream action to record."""

        callback = self._ready_callback
        return callback is None or bool(callback())

    @property
    def master_camera_name(self) -> str:
        return self._master_camera_name

    def prepare(self) -> CameraRecorderHealth:
        """Start the camera child and its event-driven parent snapshot worker."""

        if self._closed:
            raise RuntimeError("Camera recorder is closed")
        self._start_worker()
        self._start_snapshot_worker()
        return self.check_health()

    def wait_until_camera_ready(self, timeout_s: float = 10.0) -> CameraRecorderHealth:
        """Wait until the master camera has produced at least one fresh frame."""

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

    def check_health(self) -> CameraRecorderHealth:
        """Read child health messages without invoking any robot SDK methods."""

        self._drain_worker_status()
        self._drain_worker_errors()
        process = self._process
        worker_alive = process is not None and process.is_alive()
        active = self.active
        with self._health_lock:
            if process is not None and not worker_alive and not self._closed and self._failure is None:
                self._failure = RuntimeError("Master-triggered camera worker exited unexpectedly")
                self._worker_state = "fault"
            error = None if self._failure is None else str(self._failure)
            return CameraRecorderHealth(
                state=self._worker_state,
                worker_alive=worker_alive,
                active=active,
                last_heartbeat_monotonic_ns=self._last_heartbeat_monotonic_ns,
                last_master_capture_monotonic_ns=self._last_master_capture_monotonic_ns,
                error=error,
                source_health=dict(self._source_health),
            )

    def start_episode(self, task: str) -> int:
        """Start accepting new master-camera events into one episode."""

        self.prepare()
        self._raise_if_failed()
        if not self.ready:
            raise RuntimeError("Cannot record before both sides accept an upstream joint target")
        with self._state_lock:
            if self._active:
                raise RuntimeError("An episode is already active")
            # Set this before the child can emit its first event, so the parent
            # snapshot worker never rejects a valid first frame as inactive.
            self._active = True
        try:
            response = self._request("start", task=task)
        except BaseException:
            with self._state_lock:
                self._active = False
            raise
        assert response.value is not None
        return response.value

    def stop_episode(self) -> int:
        """Finish the active episode without accepting more feedback requests."""

        with self._state_lock:
            if not self._active:
                raise RuntimeError("No active episode")
            self._active = False
        try:
            response = self._request("finish")
        finally:
            with self._state_lock:
                self._active = False
        self._raise_if_failed()
        assert response.value is not None
        return response.value

    def discard_episode(self) -> None:
        """Discard the active staging episode and stop taking snapshots."""

        with self._state_lock:
            if not self._active:
                raise RuntimeError("No active episode")
            self._active = False
        try:
            self._request("discard")
        finally:
            with self._state_lock:
                self._active = False

    def close(self) -> None:
        """Stop collection workers and release the camera child process."""

        if self._closed:
            return
        if self.active:
            try:
                self.discard_episode()
            except Exception:
                pass

        self._snapshot_stop_event.set()
        process = self._process
        if process is not None:
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

        snapshot_thread = self._snapshot_thread
        if snapshot_thread is not None:
            snapshot_thread.join(timeout=1.0)
        self._snapshot_thread = None
        with self._health_lock:
            self._closed = True
            self._worker_state = "closed"
        self._close_queues()

    def _start_worker(self) -> None:
        if self._process is not None:
            return
        process = self._context.Process(
            target=_run_master_triggered_camera_worker,
            args=(
                self._camera_adapter,
                self._camera_names,
                self._master_camera_name,
                self.config.root,
                self.config.robot_type,
                self.config.fps,
                self._joint_count,
                self._scalar_actuator_names,
                self.config.image_storage,
                self.config.quality,
                self._snapshot_timeout_s,
                self._control_queue,
                self._master_request_queue,
                self._master_response_queue,
                self._response_queue,
                self._error_queue,
                self._status_queue,
            ),
            name="leobot-master-triggered-camera-recorder",
            daemon=False,
        )
        self._process = process
        with self._health_lock:
            self._worker_state = "starting"
            self._last_heartbeat_monotonic_ns = None
            self._last_master_capture_monotonic_ns = None
            self._failure = None
        process.start()
        response = self._wait_response(0, self._startup_timeout_s)
        if response.ok:
            self._drain_worker_status()
            return
        self.close()
        raise RuntimeError(
            "Master-triggered camera worker failed to start: "
            f"{response.error or 'unknown error'}"
        )

    def _start_snapshot_worker(self) -> None:
        if self._snapshot_thread is not None:
            return
        self._snapshot_stop_event.clear()
        thread = threading.Thread(
            target=self._snapshot_loop,
            name="leobot-master-frame-snapshot",
            daemon=True,
        )
        self._snapshot_thread = thread
        thread.start()

    def _snapshot_loop(self) -> None:
        """Run only in response to a master-frame event, never on a timer."""

        while not self._snapshot_stop_event.is_set():
            try:
                request = self._master_request_queue.get(timeout=_PARENT_REQUEST_POLL_S)
            except queue.Empty:
                continue
            if not isinstance(request, MasterFrameRequest):
                continue

            if not self.active:
                response = _MasterFrameResponse(
                    sequence=request.sequence,
                    skip_reason="episode_not_active",
                    audit={"parent_response_monotonic_ns": time.perf_counter_ns()},
                )
            else:
                started_ns = time.perf_counter_ns()
                try:
                    outcome = self._snapshot_provider(request)
                    finished_ns = time.perf_counter_ns()
                    audit = {
                        "feedback_read_started_monotonic_ns": started_ns,
                        "feedback_read_finished_monotonic_ns": finished_ns,
                        "feedback_read_after_master_ns": started_ns - request.capture_monotonic_ns,
                        "parent_response_monotonic_ns": finished_ns,
                    }
                    if isinstance(outcome, MasterFrameSnapshot):
                        response = _MasterFrameResponse(
                            sequence=request.sequence,
                            snapshot=outcome,
                            audit=audit,
                        )
                    elif isinstance(outcome, MasterFrameSkip):
                        response = _MasterFrameResponse(
                            sequence=request.sequence,
                            skip_reason=outcome.reason,
                            audit={**audit, **outcome.audit},
                        )
                    else:
                        raise TypeError(
                            "snapshot_provider must return MasterFrameSnapshot or MasterFrameSkip"
                        )
                except BaseException as exc:
                    finished_ns = time.perf_counter_ns()
                    error = f"{type(exc).__name__}: {exc}"
                    response = _MasterFrameResponse(
                        sequence=request.sequence,
                        audit={
                            "feedback_read_started_monotonic_ns": started_ns,
                            "feedback_read_finished_monotonic_ns": finished_ns,
                            "feedback_read_after_master_ns": started_ns - request.capture_monotonic_ns,
                        },
                        error=error,
                    )
                    with self._health_lock:
                        self._failure = RuntimeError(
                            f"Master-frame feedback snapshot failed: {error}"
                        )
                        self._worker_state = "fault"
            try:
                self._master_response_queue.put_nowait(response)
            except queue.Full:
                # The camera child has already timed out or exited.  Do not
                # block this worker and do not let stale feedback pile up.
                continue

    def _request(
        self,
        kind: str,
        *,
        task: str | None = None,
        timeout_s: float = 30.0,
    ) -> _WorkerResponse:
        self._raise_if_failed()
        process = self._process
        if process is None or not process.is_alive():
            raise RuntimeError("Master-triggered camera worker is not running")
        with self._request_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            self._control_queue.put(_WorkerControl(request_id=request_id, kind=kind, task=task))
            response = self._wait_response(request_id, timeout_s)
        if not response.ok:
            raise RuntimeError(response.error or f"Camera worker rejected {kind}")
        return response

    def _wait_response(self, request_id: int, timeout_s: float) -> _WorkerResponse:
        pending = self._pending_responses.pop(request_id, None)
        if pending is not None:
            return pending
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    f"Timed out waiting for master-triggered camera request {request_id}"
                )
            try:
                response = self._response_queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(
                    f"Timed out waiting for master-triggered camera request {request_id}"
                ) from exc
            if response.request_id == request_id:
                return response
            self._pending_responses[response.request_id] = response

    def _raise_if_failed(self) -> None:
        self._drain_worker_errors()
        with self._health_lock:
            failure = self._failure
        if failure is not None:
            raise failure

    def _drain_worker_errors(self) -> None:
        while True:
            try:
                message = self._error_queue.get_nowait()
            except queue.Empty:
                return
            with self._health_lock:
                self._failure = RuntimeError(
                    f"Master-triggered camera worker failed: {message}"
                )
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
                    self._last_master_capture_monotonic_ns = (
                        status.last_master_capture_monotonic_ns
                    )
                self._source_health = dict(status.source_health)
                if status.error is not None:
                    self._failure = RuntimeError(
                        f"Master-triggered camera worker failed: {status.error}"
                    )
                    self._worker_state = "fault"

    def _close_queues(self) -> None:
        for worker_queue in (
            self._control_queue,
            self._master_request_queue,
            self._master_response_queue,
            self._response_queue,
            self._error_queue,
            self._status_queue,
        ):
            try:
                worker_queue.close()
                worker_queue.cancel_join_thread()
            except (AttributeError, OSError, ValueError):
                pass


def _run_master_triggered_camera_worker(
    camera_adapter: CameraAdapterConfig,
    camera_names: tuple[str, ...],
    master_camera_name: str,
    root: Any,
    robot_type: str,
    fps: int,
    joint_count: int,
    scalar_actuator_names: tuple[str, ...],
    image_storage: str,
    quality: int,
    snapshot_timeout_s: float,
    control_queue: Any,
    master_request_queue: Any,
    master_response_queue: Any,
    response_queue: Any,
    error_queue: Any,
    status_queue: Any,
) -> None:
    """Own cameras and writes; communicate only numeric master-frame events."""

    session: CameraAdapterSession | None = None
    writer: V21DatasetWriter | None = None
    active = False
    worker_state = "starting"
    last_master_capture_monotonic_ns: int | None = None
    last_heartbeat_s = 0.0

    def publish_status(*, force: bool = False, error: str | None = None) -> None:
        nonlocal last_heartbeat_s
        now_s = time.monotonic()
        if not force and now_s - last_heartbeat_s < _HEARTBEAT_INTERVAL_S:
            return
        source_health = _read_session_source_health(session)
        status_queue.put(
            _WorkerStatus(
                state="fault" if error is not None else worker_state,
                active=active,
                heartbeat_monotonic_ns=time.perf_counter_ns(),
                last_master_capture_monotonic_ns=last_master_capture_monotonic_ns,
                source_health=source_health,
                error=error,
            )
        )
        last_heartbeat_s = now_s

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
        master_source = sources.get(master_feature_name)
        if master_source is None:
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
                scalar_actuator_names=scalar_actuator_names,
                cameras=cameras,
                depth_cameras=depth_cameras,
                image_storage=image_storage,  # type: ignore[arg-type]
                quality=quality,
            )
        )
        worker_state = "ready"
        publish_status(force=True)
        response_queue.put(_WorkerResponse(request_id=0, ok=True))
    except BaseException as exc:
        message = repr(exc)
        publish_status(force=True, error=message)
        response_queue.put(_WorkerResponse(request_id=0, ok=False, error=message))
        if session is not None:
            session.close()
        return

    health_sequence = master_source.latest_sequence()
    recording_sequence = health_sequence
    next_master_request_sequence = 1
    pending: _PendingMasterFrame | None = None
    timeline = _EpisodeTimeline()

    try:
        while True:
            control = _drain_one_control(control_queue)
            if control is not None:
                try:
                    if control.kind == "start":
                        if active:
                            raise RuntimeError("An episode is already active")
                        if control.task is None:
                            raise ValueError("start requires a task")
                        assert writer is not None
                        episode = writer.begin_episode(control.task)
                        recording_sequence = master_source.latest_sequence()
                        health_sequence = recording_sequence
                        pending = None
                        timeline = _EpisodeTimeline()
                        active = True
                        worker_state = "recording"
                        publish_status(force=True)
                        response_queue.put(
                            _WorkerResponse(control.request_id, ok=True, value=episode)
                        )
                    elif control.kind == "finish":
                        if not active:
                            raise RuntimeError("No active episode")
                        assert writer is not None
                        if pending is not None:
                            _append_pending_skip(
                                writer,
                                pending,
                                "episode_finished_before_feedback_response",
                            )
                            pending = None
                        worker_state = "finalizing"
                        publish_status(force=True)
                        episode = writer.finish_episode()
                        active = False
                        worker_state = "ready"
                        publish_status(force=True)
                        response_queue.put(
                            _WorkerResponse(control.request_id, ok=True, value=episode)
                        )
                    elif control.kind == "discard":
                        if active:
                            assert writer is not None
                            writer.discard_episode()
                            pending = None
                            active = False
                            worker_state = "ready"
                            publish_status(force=True)
                        response_queue.put(_WorkerResponse(control.request_id, ok=True))
                    elif control.kind == "shutdown":
                        if active:
                            assert writer is not None
                            writer.discard_episode()
                            pending = None
                            active = False
                        worker_state = "closed"
                        publish_status(force=True)
                        response_queue.put(_WorkerResponse(control.request_id, ok=True))
                        return
                    else:
                        raise ValueError(f"Unknown worker control command: {control.kind}")
                except BaseException as exc:
                    if active:
                        try:
                            assert writer is not None
                            writer.discard_episode()
                        except BaseException:
                            pass
                        active = False
                        pending = None
                    worker_state = "fault"
                    message = repr(exc)
                    publish_status(force=True, error=message)
                    response_queue.put(
                        _WorkerResponse(control.request_id, ok=False, error=message)
                    )
                continue

            if active:
                assert writer is not None
                try:
                    pending = _consume_snapshot_responses(
                        writer,
                        sources,
                        master_feature_name,
                        master_response_queue,
                        pending,
                        timeline,
                        _read_session_source_health(session),
                    )
                    if pending is not None:
                        elapsed_s = (
                            time.perf_counter_ns() - pending.request_sent_monotonic_ns
                        ) / 1_000_000_000
                        if elapsed_s >= snapshot_timeout_s:
                            _append_pending_skip(
                                writer,
                                pending,
                                "feedback_snapshot_timeout",
                                extra_audit={
                                    "feedback_snapshot_timeout_s": snapshot_timeout_s,
                                    "feedback_wait_elapsed_s": elapsed_s,
                                },
                            )
                            pending = None

                    while True:
                        item = master_source.next_frame_after(recording_sequence)
                        if item is None:
                            break
                        sequence, frame = item
                        if sequence <= recording_sequence:
                            raise RuntimeError(
                                "Master camera source returned a non-increasing sequence"
                            )
                        recording_sequence = sequence
                        capture_ns = _frame_capture_monotonic_ns(frame)
                        if capture_ns is not None:
                            last_master_capture_monotonic_ns = capture_ns
                        if capture_ns is None:
                            writer.append_skipped_tick(
                                {
                                    "skip_reason": "master_camera_frame_has_no_capture_timestamp",
                                    "master_camera": master_feature_name,
                                    f"{master_feature_name}.source_frame_index": _frame_source_frame_index(frame),
                                }
                            )
                            continue
                        if pending is not None:
                            _append_master_skip(
                                writer,
                                master_feature_name,
                                frame,
                                capture_ns,
                                "feedback_snapshot_pending",
                            )
                            continue
                        request = MasterFrameRequest(
                            sequence=next_master_request_sequence,
                            capture_monotonic_ns=capture_ns,
                            emitted_monotonic_ns=time.perf_counter_ns(),
                        )
                        next_master_request_sequence += 1
                        try:
                            master_request_queue.put_nowait(request)
                        except queue.Full:
                            _append_master_skip(
                                writer,
                                master_feature_name,
                                frame,
                                capture_ns,
                                "parent_snapshot_request_queue_full",
                            )
                            continue
                        pending = _PendingMasterFrame(
                            request=request,
                            master_feature_name=master_feature_name,
                            frame=frame,
                            request_sent_monotonic_ns=time.perf_counter_ns(),
                        )
                except BaseException as exc:
                    try:
                        writer.discard_episode()
                    except BaseException:
                        pass
                    active = False
                    pending = None
                    worker_state = "fault"
                    message = repr(exc)
                    error_queue.put(message)
                    publish_status(force=True, error=message)
                    return
            else:
                health_sequence, last_master_capture_monotonic_ns = _advance_health_cursor(
                    master_source,
                    health_sequence,
                    last_master_capture_monotonic_ns,
                )

            publish_status()
            time.sleep(_CHILD_IDLE_SLEEP_S)
    finally:
        if session is not None:
            session.close()


def _drain_one_control(control_queue: Any) -> _WorkerControl | None:
    try:
        value = control_queue.get_nowait()
    except queue.Empty:
        return None
    if not isinstance(value, _WorkerControl):
        raise TypeError("Camera worker received an invalid control message")
    return value


def _read_session_source_health(
    session: CameraAdapterSession | None,
) -> dict[str, CameraSourceHealth]:
    if session is None:
        return {}
    provider = getattr(session, "source_health", None)
    if not callable(provider):
        return {}
    try:
        return dict(provider())
    except BaseException as exc:
        return {
            "diagnostics": CameraSourceHealth(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
        }


def _advance_health_cursor(
    source: CameraFrameSource,
    sequence: int,
    latest_capture_ns: int | None,
) -> tuple[int, int | None]:
    while True:
        item = source.next_frame_after(sequence)
        if item is None:
            return sequence, latest_capture_ns
        next_sequence, frame = item
        if next_sequence <= sequence:
            raise RuntimeError("Master camera source returned a non-increasing sequence")
        sequence = next_sequence
        capture_ns = _frame_capture_monotonic_ns(frame)
        if capture_ns is not None:
            latest_capture_ns = capture_ns


def _consume_snapshot_responses(
    writer: V21DatasetWriter,
    sources: Mapping[str, CameraFrameSource],
    master_feature_name: str,
    response_queue: Any,
    pending: _PendingMasterFrame | None,
    timeline: _EpisodeTimeline,
    source_health: Mapping[str, CameraSourceHealth] | None = None,
) -> _PendingMasterFrame | None:
    """Consume ready numeric feedback without blocking camera capture."""

    while True:
        try:
            response = response_queue.get_nowait()
        except queue.Empty:
            return pending
        if not isinstance(response, _MasterFrameResponse):
            continue
        if pending is None or response.sequence != pending.request.sequence:
            # A timed-out/finished frame may complete later.  It is already
            # audited and must not be paired with a newer master image.
            continue
        if response.error is not None:
            raise RuntimeError(
                f"parent feedback snapshot failed for master frame {response.sequence}: "
                f"{response.error}"
            )
        if response.skip_reason is not None:
            _append_pending_skip(
                writer,
                pending,
                response.skip_reason,
                extra_audit=response.audit,
            )
            pending = None
            continue
        if response.snapshot is None:
            raise RuntimeError("parent returned a master-frame response without a snapshot")
        _append_snapshot_row(
            writer,
            sources,
            master_feature_name,
            pending,
            response.snapshot,
            timeline,
            response.audit,
            source_health,
        )
        pending = None


def _append_snapshot_row(
    writer: V21DatasetWriter,
    sources: Mapping[str, CameraFrameSource],
    master_feature_name: str,
    pending: _PendingMasterFrame,
    snapshot: MasterFrameSnapshot,
    timeline: _EpisodeTimeline,
    response_audit: Mapping[str, Any],
    source_health: Mapping[str, CameraSourceHealth] | None = None,
) -> None:
    master_capture_ns = pending.request.capture_monotonic_ns
    row_timestamp_s = timeline.timestamp_s(master_capture_ns)
    audit: dict[str, Any] = {
        **snapshot.audit,
        **response_audit,
        "master_request_sequence": pending.request.sequence,
        "master_request_emitted_monotonic_ns": pending.request.emitted_monotonic_ns,
        "master_capture_monotonic_ns": master_capture_ns,
        "episode_origin_monotonic_ns": timeline.origin_monotonic_ns,
        "timestamp_s": row_timestamp_s,
        # The snapshot is intentionally post-capture.  This flag prevents
        # downstream consumers from treating it as a causal-before-image state.
        "feedback_is_post_master_capture": True,
    }
    cameras: dict[str, CameraFrame] = {}
    depths: dict[str, DepthFrame] = {}
    for feature_name, source in sources.items():
        frame = pending.frame if feature_name == master_feature_name else source.frame_at_or_before(master_capture_ns)
        if frame is None:
            logical_name = feature_name.removeprefix("observation.images.")
            camera_health = (source_health or {}).get(logical_name)
            health_audit: dict[str, Any] = {}
            if camera_health is not None:
                health_audit = {
                    f"{feature_name}.status": camera_health.status,
                    f"{feature_name}.last_error": camera_health.error,
                    f"{feature_name}.latest_capture_monotonic_ns": (
                        camera_health.latest_capture_monotonic_ns
                    ),
                }
            _append_pending_skip(
                writer,
                pending,
                f"missing_camera_for_master_frame:{feature_name}",
                extra_audit={**audit, **health_audit},
            )
            return
        capture_ns = _frame_capture_monotonic_ns(frame)
        if capture_ns is None:
            _append_pending_skip(
                writer,
                pending,
                f"camera_frame_has_no_capture_timestamp:{feature_name}",
                extra_audit=audit,
            )
            return
        if capture_ns > master_capture_ns:
            _append_pending_skip(
                writer,
                pending,
                f"future_camera_frame_rejected:{feature_name}",
                extra_audit={
                    **audit,
                    f"{feature_name}.capture_monotonic_ns": capture_ns,
                },
            )
            return
        recorded = _recorded_camera_frame(frame, timeline, row_timestamp_s)
        cameras[feature_name] = recorded
        audit[f"{feature_name}.capture_monotonic_ns"] = recorded.capture_monotonic_ns
        audit[f"{feature_name}.source_timestamp_ns"] = recorded.source_timestamp_ns
        audit[f"{feature_name}.source_frame_index"] = recorded.source_frame_index
        audit[f"{feature_name}.capture_delta_from_master_ns"] = (
            capture_ns - master_capture_ns
        )
        if isinstance(frame, RGBDFrame):
            depth_name = _depth_feature_name(feature_name)
            depths[depth_name] = frame.depth
            audit[f"{depth_name}.capture_monotonic_ns"] = frame.depth.capture_monotonic_ns
            audit[f"{depth_name}.source_timestamp_ns"] = frame.depth.source_timestamp_ns
            audit[f"{depth_name}.source_frame_index"] = frame.depth.source_frame_index

    writer.append_frame(
        RecordedFrame(
            state=np.asarray(snapshot.state, dtype=float),
            action=np.asarray(snapshot.action, dtype=float),
            timestamp_s=row_timestamp_s,
            cameras=cameras,
            depths=depths,
            actuator_states=dict(snapshot.actuator_states),
            actuator_actions=dict(snapshot.actuator_actions),
            audit=audit,
        )
    )


def _append_pending_skip(
    writer: V21DatasetWriter,
    pending: _PendingMasterFrame,
    reason: str,
    *,
    extra_audit: Mapping[str, Any] | None = None,
) -> None:
    _append_master_skip(
        writer,
        pending.master_feature_name,
        pending.frame,
        pending.request.capture_monotonic_ns,
        reason,
        extra_audit={
            "master_request_sequence": pending.request.sequence,
            "master_request_emitted_monotonic_ns": pending.request.emitted_monotonic_ns,
            **({} if extra_audit is None else dict(extra_audit)),
        },
    )


def _append_master_skip(
    writer: V21DatasetWriter,
    master_feature_name: str,
    frame: CameraFrame | RGBDFrame,
    capture_monotonic_ns: int,
    reason: str,
    *,
    extra_audit: Mapping[str, Any] | None = None,
) -> None:
    writer.append_skipped_tick(
        {
            "skip_reason": reason,
            "master_camera": master_feature_name,
            "master_capture_monotonic_ns": capture_monotonic_ns,
            f"{master_feature_name}.source_timestamp_ns": _frame_source_timestamp_ns(frame),
            f"{master_feature_name}.source_frame_index": _frame_source_frame_index(frame),
            **({} if extra_audit is None else dict(extra_audit)),
        }
    )


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
    capture_ns = _frame_capture_monotonic_ns(frame)
    return CameraFrame(
        rgb=frame.rgb,
        timestamp_s=timeline.camera_timestamp_s(capture_ns, row_timestamp_s),
        capture_monotonic_ns=capture_ns,
        source_timestamp_ns=_frame_source_timestamp_ns(frame),
        source_frame_index=_frame_source_frame_index(frame),
    )


def _depth_feature_name(rgb_feature_name: str) -> str:
    suffix = rgb_feature_name.removeprefix("observation.images.")
    return f"observation.depth.{suffix}"


def _rgb_feature_name(camera_name: str) -> str:
    return f"observation.images.{camera_name}"
