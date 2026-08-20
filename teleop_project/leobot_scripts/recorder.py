"""Fixed-rate robot observation recording without camera ownership."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from teleop_sdk.interfaces import FollowerArm, GripperActuator

from .sources import GripperFeedbackSource
from .v21_writer import ImageStorage, RecordedFrame, V21DatasetWriter, WriterConfig


@dataclass(frozen=True)
class RecorderConfig:
    """Dataset schema configuration independent of robot and camera vendors.

    ``DatasetRecorder`` uses ``fps`` as its fixed robot sampling frequency.
    Camera-driven recorders keep it as the dataset's output frame rate and
    receive their independent numeric sampling frequency separately.
    """

    root: Path
    robot_type: str
    fps: int = 30
    gripper_source: GripperFeedbackSource | None = None
    queue_maxsize: int = 60
    image_storage: ImageStorage = "video"
    quality: int = 75

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.queue_maxsize <= 0:
            raise ValueError("queue_maxsize must be positive")
        if self.image_storage not in {"video", "png", "jpg"}:
            raise ValueError("image_storage must be 'video', 'png', or 'jpg'")
        if isinstance(self.quality, bool) or not isinstance(self.quality, int) or not 1 <= self.quality <= 100:
            raise ValueError("quality must be an integer from 1 to 100")


@dataclass(frozen=True)
class _ActionSample:
    angles_deg: np.ndarray
    sent_monotonic_ns: int


@dataclass
class _SampleWorkItem:
    state: np.ndarray
    action: np.ndarray
    gripper_state: float | None
    gripper_action: float | None
    audit: dict[str, Any]


@dataclass
class _AuditWorkItem:
    audit: dict[str, Any]


class RecorderCallbacks(Protocol):
    """Minimal recorder callbacks used by the control-path decorators."""

    def attach_follower(self, follower: "RecordingFollower") -> None:
        ...

    def note_joint_action(self, angles_deg: np.ndarray, sent_monotonic_ns: int) -> None:
        ...

    def note_gripper_action(self, opening: float, sent_monotonic_ns: int) -> None:
        ...


class EpisodeRecorder(RecorderCallbacks, Protocol):
    """Lifecycle surface shared by local and isolated recorders."""

    config: RecorderConfig

    @property
    def active(self) -> bool:
        ...

    @property
    def ready(self) -> bool:
        ...

    def start_episode(self, task: str) -> int:
        ...

    def stop_episode(self) -> int:
        ...

    def discard_episode(self) -> None:
        ...

    def close(self) -> None:
        ...


class DatasetRecorder:
    """Collect fixed-rate robot state, action, and optional gripper observations."""

    def __init__(self, config: RecorderConfig):
        self.config = config
        self._follower: RecordingFollower | None = None
        self._writer: V21DatasetWriter | None = None
        self._action_lock = threading.Lock()
        self._latest_action: _ActionSample | None = None
        self._latest_gripper_action: tuple[float, int] | None = None
        self._queue: queue.Queue[_SampleWorkItem | _AuditWorkItem | None] | None = None
        self._stop_event: threading.Event | None = None
        self._sampler: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None
        self._failure: BaseException | None = None
        self._state_lock = threading.Lock()
        self._active = False

    @property
    def active(self) -> bool:
        with self._state_lock:
            return self._active

    @property
    def ready(self) -> bool:
        """Whether at least one follower joint command has been accepted."""

        return self._current_action() is not None

    def attach_follower(self, follower: RecordingFollower) -> None:
        """Bind one wrapped follower before recording starts."""

        with self._state_lock:
            if self._follower is not None and self._follower is not follower:
                raise RuntimeError("A recorder can only be attached to one follower")
            if self._active:
                raise RuntimeError("Cannot replace the follower during an active episode")
            self._follower = follower
            self._writer = V21DatasetWriter(
                WriterConfig(
                    root=self.config.root,
                    robot_type=self.config.robot_type,
                    fps=self.config.fps,
                    joint_count=follower.joint_count,
                    include_gripper=self.config.gripper_source is not None,
                    image_storage=self.config.image_storage,
                    quality=self.config.quality,
                )
            )

    def start_episode(self, task: str) -> int:
        """Start a new SDK-controlled episode with a natural-language task label."""

        with self._state_lock:
            if self._active:
                raise RuntimeError("An episode is already active")
            if self._follower is None or self._writer is None:
                raise RuntimeError("Wrap a FollowerArm with RecordingFollower before starting")
            if self._current_action() is None:
                raise RuntimeError("Cannot record before the follower accepts at least one joint command")
            episode_index = self._writer.begin_episode(task)
            self._failure = None
            self._queue = queue.Queue(maxsize=self.config.queue_maxsize)
            self._stop_event = threading.Event()
            self._active = True
            self._writer_thread = threading.Thread(target=self._writer_loop, name="leobot-writer", daemon=True)
            self._sampler = threading.Thread(target=self._sample_loop, name="leobot-sampler", daemon=True)
            self._writer_thread.start()
            self._sampler.start()
            return episode_index

    def stop_episode(self) -> int:
        """Drain collection work and commit one complete episode."""

        self._stop_workers()
        with self._state_lock:
            writer = self._writer
            failure = self._failure
            self._active = False
        if writer is None:
            raise RuntimeError("No writer is attached")
        if failure is not None:
            writer.discard_episode()
            raise RuntimeError("Episode recording failed; the staged data was discarded") from failure
        return writer.finish_episode()

    def discard_episode(self) -> None:
        """Stop collection and remove all staged files for the current episode."""

        self._stop_workers()
        with self._state_lock:
            writer = self._writer
            was_active = self._active
            self._active = False
        if writer is not None and was_active:
            writer.discard_episode()

    def close(self) -> None:
        """Discard an active episode; callers still own robot disconnect lifecycle."""

        if self.active:
            self.discard_episode()

    def note_joint_action(self, angles_deg: np.ndarray, sent_monotonic_ns: int) -> None:
        """Store only targets accepted by the wrapped follower."""

        with self._action_lock:
            self._latest_action = _ActionSample(np.asarray(angles_deg, dtype=float).copy(), sent_monotonic_ns)

    def note_gripper_action(self, opening: float, sent_monotonic_ns: int) -> None:
        with self._action_lock:
            self._latest_gripper_action = (float(opening), sent_monotonic_ns)

    def _current_action(self) -> _ActionSample | None:
        with self._action_lock:
            if self._latest_action is None:
                return None
            return _ActionSample(self._latest_action.angles_deg.copy(), self._latest_action.sent_monotonic_ns)

    def _current_gripper_action(self) -> tuple[float, int] | None:
        with self._action_lock:
            return self._latest_gripper_action

    def _sample_loop(self) -> None:
        assert self._follower is not None
        assert self._queue is not None
        assert self._stop_event is not None
        period_ns = int(1_000_000_000 / self.config.fps)
        scheduled_ns = time.perf_counter_ns()
        tick_index = 0
        while not self._stop_event.is_set():
            now_ns = time.perf_counter_ns()
            delay_ns = scheduled_ns - now_ns
            if delay_ns > 0:
                self._stop_event.wait(delay_ns / 1_000_000_000)
                if self._stop_event.is_set():
                    break
            audit: dict[str, Any] = {
                "tick_index": tick_index,
                "scheduled_monotonic_ns": scheduled_ns,
                "target_monotonic_ns": scheduled_ns,
                "sample_started_monotonic_ns": time.perf_counter_ns(),
            }
            try:
                snapshot = self._follower.recording_snapshot()
                if snapshot is None:
                    self._enqueue_audit({**audit, "skip_reason": "missing_cached_follower_state"})
                    tick_index += 1
                    scheduled_ns += period_ns
                    continue
                state, state_read_started_ns, state_read_finished_ns = snapshot
                action = self._current_action()
                audit["state_read_started_monotonic_ns"] = state_read_started_ns
                audit["state_read_finished_monotonic_ns"] = state_read_finished_ns
                if action is None:
                    self._enqueue_audit({**audit, "skip_reason": "no_successful_joint_action"})
                else:
                    audit["joint_action_sent_monotonic_ns"] = action.sent_monotonic_ns
                    item = self._build_sample_item(state, action.angles_deg, audit)
                    if item is None:
                        pass
                    else:
                        self._enqueue_sample(item)
            except BaseException as exc:
                self._enqueue_audit({**audit, "skip_reason": "sampling_error", "error": repr(exc)})
            tick_index += 1
            scheduled_ns += period_ns

    def _build_sample_item(
        self,
        state: np.ndarray,
        action: np.ndarray,
        audit: dict[str, Any],
    ) -> _SampleWorkItem | None:
        gripper_state: float | None = None
        gripper_action: float | None = None
        if self.config.gripper_source is not None:
            feedback = self.config.gripper_source.read_gripper_opening()
            action_sample = self._current_gripper_action()
            if feedback is None or action_sample is None:
                self._enqueue_audit({**audit, "skip_reason": "missing_gripper_state_or_action"})
                return None
            gripper_state = float(np.asarray(feedback.value, dtype=float).reshape(-1)[0])
            gripper_action = action_sample[0]
            audit["gripper_capture_monotonic_ns"] = feedback.capture_monotonic_ns
            audit["gripper_action_sent_monotonic_ns"] = action_sample[1]
        return _SampleWorkItem(
            state=state,
            action=action,
            gripper_state=gripper_state,
            gripper_action=gripper_action,
            audit=audit,
        )

    def _enqueue_sample(self, item: _SampleWorkItem) -> None:
        assert self._queue is not None
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self._enqueue_audit({**item.audit, "skip_reason": "writer_queue_full"})

    def _enqueue_audit(self, audit: dict[str, Any]) -> None:
        assert self._queue is not None
        try:
            self._queue.put_nowait(_AuditWorkItem(audit))
        except queue.Full:
            # This event cannot be persisted without blocking the controller-adjacent sampler.
            pass

    def _writer_loop(self) -> None:
        assert self._queue is not None
        assert self._writer is not None
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                if isinstance(item, _AuditWorkItem):
                    self._writer.append_skipped_tick(item.audit)
                else:
                    self._writer.append_frame(
                        RecordedFrame(
                            state=item.state,
                            action=item.action,
                            cameras={},
                            depths={},
                            gripper_state=item.gripper_state,
                            gripper_action=item.gripper_action,
                            audit=item.audit,
                        )
                    )
            except BaseException as exc:
                self._failure = exc

    def _stop_workers(self) -> None:
        with self._state_lock:
            if not self._active:
                raise RuntimeError("No active episode")
            stop_event = self._stop_event
            sampler = self._sampler
            writer_thread = self._writer_thread
            work_queue = self._queue
        assert stop_event is not None and sampler is not None and writer_thread is not None and work_queue is not None
        stop_event.set()
        sampler.join()
        work_queue.put(None)
        writer_thread.join()
        with self._state_lock:
            self._stop_event = None
            self._sampler = None
            self._writer_thread = None
            self._queue = None


class RecordingFollower(FollowerArm):
    """A thread-safe ``FollowerArm`` decorator that records accepted commands."""

    def __init__(self, inner: FollowerArm, recorder: RecorderCallbacks):
        self._inner = inner
        self._recorder = recorder
        self._lock = threading.Lock()
        recorder.attach_follower(self)

    @property
    def joint_count(self) -> int:
        return self._inner.joint_count

    @property
    def joint_limits_deg(self) -> tuple[np.ndarray, np.ndarray]:
        return self._inner.joint_limits_deg

    def connect(self) -> None:
        with self._lock:
            self._inner.connect()

    def read_joint_angles_deg(self) -> np.ndarray:
        with self._lock:
            return self._inner.read_joint_angles_deg()

    def start_servo(self) -> bool:
        with self._lock:
            return self._inner.start_servo()

    def send_joint_angles_deg(self, angles_deg: np.ndarray, command_time_s: float) -> bool:
        target = np.asarray(angles_deg, dtype=float).copy()
        with self._lock:
            accepted = self._inner.send_joint_angles_deg(target, command_time_s)
            if accepted:
                self._recorder.note_joint_action(target, time.perf_counter_ns())
            return accepted

    def recover(self) -> bool:
        with self._lock:
            return self._inner.recover()

    def stop_servo(self) -> None:
        with self._lock:
            self._inner.stop_servo()

    def disconnect(self) -> None:
        with self._lock:
            self._inner.disconnect()

    def recording_snapshot(self) -> tuple[np.ndarray, int, int] | None:
        """Return cached feedback without delaying ServoJ when the adapter supports it."""

        cached_reader = getattr(self._inner, "read_cached_joint_angles_deg", None)
        if callable(cached_reader):
            started = time.perf_counter_ns()
            value = cached_reader()
            finished = time.perf_counter_ns()
            if value is None:
                return None
            return np.asarray(value, dtype=float).copy(), started, finished

        with self._lock:
            started = time.perf_counter_ns()
            value = self._inner.read_joint_angles_deg()
            finished = time.perf_counter_ns()
        return np.asarray(value, dtype=float).copy(), started, finished


class RecordingGripper(GripperActuator):
    """Optional command decorator; feedback remains a separate source protocol."""

    def __init__(self, inner: GripperActuator, recorder: RecorderCallbacks):
        self._inner = inner
        self._recorder = recorder
        self._lock = threading.Lock()

    def connect(self) -> None:
        with self._lock:
            self._inner.connect()

    def send_normalized(self, opening: float) -> bool:
        value = float(opening)
        with self._lock:
            accepted = self._inner.send_normalized(value)
            if accepted:
                self._recorder.note_gripper_action(value, time.perf_counter_ns())
            return accepted

    def disable(self) -> None:
        with self._lock:
            self._inner.disable()

    def disconnect(self) -> None:
        with self._lock:
            self._inner.disconnect()
