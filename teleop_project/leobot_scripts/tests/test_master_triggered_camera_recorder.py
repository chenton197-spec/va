from __future__ import annotations

import json
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from leobot_scripts.master_triggered_camera_recorder import (
    MasterFrameRequest,
    MasterFrameSkip,
    MasterFrameSnapshot,
    MasterTriggeredCameraProcessDatasetRecorder,
    _EpisodeTimeline,
    _PendingMasterFrame,
    _append_snapshot_row,
)
from leobot_scripts.camera import CameraSourceHealth
from leobot_scripts.recorder import RecorderConfig
from leobot_scripts.sources import CameraFrame


def _frame(capture_monotonic_ns: int, value: int) -> CameraFrame:
    return CameraFrame(
        rgb=np.full((2, 3, 3), value, dtype=np.uint8),
        capture_monotonic_ns=capture_monotonic_ns,
        source_frame_index=value,
    )


class _StaticSource:
    def __init__(self, frame: CameraFrame | None) -> None:
        self.frame = frame
        self.targets: list[int] = []

    @property
    def shape(self) -> tuple[int, int, int]:
        return (2, 3, 3)

    def latest_sequence(self) -> int:
        return 0

    def next_frame_after(self, sequence: int) -> tuple[int, CameraFrame] | None:
        return None

    def frame_at_or_before(self, target_monotonic_ns: int) -> CameraFrame | None:
        self.targets.append(target_monotonic_ns)
        return self.frame


class _RecordingWriter:
    def __init__(self) -> None:
        self.frames: list[object] = []
        self.skipped: list[dict[str, object]] = []

    def append_frame(self, frame: object) -> int:
        self.frames.append(frame)
        return len(self.frames) - 1

    def append_skipped_tick(self, audit: dict[str, object]) -> None:
        self.skipped.append(audit)


class _UnusedCameraAdapter:
    @property
    def camera_names(self) -> tuple[str, ...]:
        return ("head",)

    def open(self) -> object:
        raise AssertionError("The snapshot-thread tests do not open cameras")


class _FastCameraSource:
    """Pickle-safe source for a spawned child-process recorder test."""

    def __init__(self) -> None:
        self._sequence = 0
        self._last_emitted_ns = time.perf_counter_ns()

    @property
    def shape(self) -> tuple[int, int, int]:
        return (2, 3, 3)

    def latest_sequence(self) -> int:
        return self._sequence

    def next_frame_after(self, sequence: int) -> tuple[int, CameraFrame] | None:
        now_ns = time.perf_counter_ns()
        if now_ns - self._last_emitted_ns < 5_000_000:
            return None
        self._sequence = max(self._sequence, sequence) + 1
        self._last_emitted_ns = now_ns
        return self._sequence, _frame(now_ns, self._sequence % 255)

    def frame_at_or_before(self, target_monotonic_ns: int) -> CameraFrame:
        return _frame(target_monotonic_ns - 1, self._sequence % 255)


class _FastCameraSession:
    def __init__(self) -> None:
        self._sources = {
            "head": _FastCameraSource(),
            "left_hand": _FastCameraSource(),
            "right_hand": _FastCameraSource(),
        }

    @property
    def sources(self) -> dict[str, _FastCameraSource]:
        return self._sources

    def close(self) -> None:
        return None

    def source_health(self) -> dict[str, CameraSourceHealth]:
        return {
            name: CameraSourceHealth(
                status="streaming",
                latest_capture_monotonic_ns=time.perf_counter_ns(),
            )
            for name in self._sources
        }


class _FastCameraAdapter:
    @property
    def camera_names(self) -> tuple[str, ...]:
        return ("head", "left_hand", "right_hand")

    def open(self) -> _FastCameraSession:
        return _FastCameraSession()


class MasterTriggeredCameraRecorderTest(unittest.TestCase):
    def _snapshot(self) -> MasterFrameSnapshot:
        return MasterFrameSnapshot(
            state=np.arange(14, dtype=float),
            action=np.arange(14, dtype=float) + 0.5,
            actuator_states={"left_gripper": 0.2, "right_gripper": 0.8},
            actuator_actions={"left_gripper": 0.3, "right_gripper": 0.9},
        )

    def _new_pending(self, master: CameraFrame) -> _PendingMasterFrame:
        return _PendingMasterFrame(
            request=MasterFrameRequest(1, 100, 110),
            master_feature_name="observation.images.head",
            frame=master,
            request_sent_monotonic_ns=time.perf_counter_ns(),
        )

    def test_pairs_hand_frames_at_or_before_master_capture(self) -> None:
        writer = _RecordingWriter()
        master = _frame(100, 1)
        left = _StaticSource(_frame(98, 2))
        right = _StaticSource(_frame(99, 3))
        sources = {
            "observation.images.head": _StaticSource(master),
            "observation.images.left_hand": left,
            "observation.images.right_hand": right,
        }

        _append_snapshot_row(
            writer,  # type: ignore[arg-type]
            sources,  # type: ignore[arg-type]
            "observation.images.head",
            self._new_pending(master),
            self._snapshot(),
            _EpisodeTimeline(),
            {"feedback_read_started_monotonic_ns": 120},
        )

        self.assertEqual(left.targets, [100])
        self.assertEqual(right.targets, [100])
        self.assertEqual(len(writer.frames), 1)
        frame = writer.frames[0]
        cameras = getattr(frame, "cameras")
        self.assertEqual(cameras["observation.images.left_hand"].capture_monotonic_ns, 98)
        self.assertEqual(cameras["observation.images.right_hand"].capture_monotonic_ns, 99)
        self.assertTrue(getattr(frame, "audit")["feedback_is_post_master_capture"])

    def test_worker_health_includes_per_camera_source_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = MasterTriggeredCameraProcessDatasetRecorder(
                RecorderConfig(
                    root=Path(directory) / "dataset",
                    robot_type="fake",
                    fps=30,
                    image_storage="png",
                ),
                _FastCameraAdapter(),  # type: ignore[arg-type]
                joint_count=14,
                master_camera_name="head",
                snapshot_provider=lambda _request: self._snapshot(),
                scalar_actuator_names=("left_gripper", "right_gripper"),
                ready=lambda: True,
            )
            try:
                health = recorder.wait_until_camera_ready(timeout_s=3.0)
                self.assertEqual(
                    set(health.source_health),
                    {"head", "left_hand", "right_hand"},
                )
                self.assertEqual(health.source_health["head"].status, "streaming")
            finally:
                recorder.close()

    def test_missing_pair_audit_includes_camera_status_and_sdk_error(self) -> None:
        writer = _RecordingWriter()
        master = _frame(100, 1)
        sources = {
            "observation.images.head": _StaticSource(master),
            "observation.images.left_hand": _StaticSource(_frame(98, 2)),
            "observation.images.right_hand": _StaticSource(None),
        }

        _append_snapshot_row(
            writer,  # type: ignore[arg-type]
            sources,  # type: ignore[arg-type]
            "observation.images.head",
            self._new_pending(master),
            self._snapshot(),
            _EpisodeTimeline(),
            {},
            {
                "right_hand": CameraSourceHealth(
                    status="failed",
                    latest_capture_monotonic_ns=90,
                    error="Device disconnected",
                )
            },
        )

        self.assertEqual(len(writer.skipped), 1)
        audit = writer.skipped[0]
        self.assertEqual(
            audit["skip_reason"],
            "missing_camera_for_master_frame:observation.images.right_hand",
        )
        self.assertEqual(
            audit["observation.images.right_hand.status"],
            "failed",
        )
        self.assertEqual(
            audit["observation.images.right_hand.last_error"],
            "Device disconnected",
        )

    def test_rejects_future_hand_frame_instead_of_pairing_it(self) -> None:
        writer = _RecordingWriter()
        master = _frame(100, 1)
        sources = {
            "observation.images.head": _StaticSource(master),
            "observation.images.left_hand": _StaticSource(_frame(101, 2)),
            "observation.images.right_hand": _StaticSource(_frame(99, 3)),
        }

        _append_snapshot_row(
            writer,  # type: ignore[arg-type]
            sources,  # type: ignore[arg-type]
            "observation.images.head",
            self._new_pending(master),
            self._snapshot(),
            _EpisodeTimeline(),
            {},
        )

        self.assertEqual(writer.frames, [])
        self.assertEqual(len(writer.skipped), 1)
        self.assertEqual(
            writer.skipped[0]["skip_reason"],
            "future_camera_frame_rejected:observation.images.left_hand",
        )

    def test_parent_snapshot_thread_reads_only_after_master_event(self) -> None:
        calls: list[int] = []

        def provider(request: MasterFrameRequest) -> MasterFrameSnapshot:
            calls.append(request.sequence)
            return self._snapshot()

        recorder = MasterTriggeredCameraProcessDatasetRecorder(
            RecorderConfig(root=Path("dataset"), robot_type="fake", fps=30),
            _UnusedCameraAdapter(),  # type: ignore[arg-type]
            joint_count=14,
            master_camera_name="head",
            snapshot_provider=provider,
            scalar_actuator_names=("left_gripper", "right_gripper"),
        )
        try:
            with recorder._state_lock:
                recorder._active = True
            recorder._start_snapshot_worker()
            time.sleep(0.08)
            self.assertEqual(calls, [])
            recorder._master_request_queue.put(MasterFrameRequest(3, 100, 100))
            response = recorder._master_response_queue.get(timeout=1.0)
            self.assertEqual(calls, [3])
            self.assertEqual(response.sequence, 3)
            self.assertIsNotNone(response.snapshot)
        finally:
            with recorder._state_lock:
                recorder._active = False
            recorder.close()

    def test_slow_snapshot_audits_dropped_master_frames_without_backlog(self) -> None:
        calls = 0
        calls_lock = threading.Lock()

        def slow_provider(request: MasterFrameRequest) -> MasterFrameSnapshot:
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.06)
            return self._snapshot()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            recorder = MasterTriggeredCameraProcessDatasetRecorder(
                RecorderConfig(
                    root=root,
                    robot_type="fake",
                    fps=30,
                    image_storage="png",
                ),
                _FastCameraAdapter(),  # type: ignore[arg-type]
                joint_count=14,
                master_camera_name="head",
                snapshot_provider=slow_provider,
                scalar_actuator_names=("left_gripper", "right_gripper"),
                ready=lambda: True,
                snapshot_timeout_s=0.2,
            )
            try:
                recorder.wait_until_camera_ready(timeout_s=3.0)
                recorder.start_episode("slow feedback")
                time.sleep(0.32)
                recorder.stop_episode()
            finally:
                recorder.close()

            audit_path = root / "meta" / "recording_audit" / "episode_000000.jsonl"
            audits = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
            emitted = [audit for audit in audits if audit["emitted"]]
            skipped = [audit for audit in audits if not audit["emitted"]]
            self.assertGreater(len(emitted), 0)
            self.assertGreater(len(skipped), 0)
            self.assertTrue(
                any(audit["skip_reason"] == "feedback_snapshot_pending" for audit in skipped)
            )
            with calls_lock:
                self.assertLess(calls, len(audits))

    def test_all_skipped_frames_report_reason_and_preserve_failed_audit(self) -> None:
        def skip_provider(request: MasterFrameRequest) -> MasterFrameSkip:
            return MasterFrameSkip(
                "missing_gloria_m_feedback",
                {"request_sequence": request.sequence},
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            recorder = MasterTriggeredCameraProcessDatasetRecorder(
                RecorderConfig(
                    root=root,
                    robot_type="fake",
                    fps=30,
                    image_storage="png",
                ),
                _FastCameraAdapter(),  # type: ignore[arg-type]
                joint_count=14,
                master_camera_name="head",
                snapshot_provider=skip_provider,
                scalar_actuator_names=("left_gripper", "right_gripper"),
                ready=lambda: True,
            )
            try:
                recorder.wait_until_camera_ready(timeout_s=3.0)
                recorder.start_episode("missing gripper feedback")
                time.sleep(0.12)
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"missing_gloria_m_feedback=\d+",
                ):
                    recorder.stop_episode()
            finally:
                recorder.close()

            failed_audits = tuple(
                (root / "meta" / "failed_recording_audit").glob("*.jsonl")
            )
            self.assertEqual(len(failed_audits), 1)
            audits = [
                json.loads(line)
                for line in failed_audits[0].read_text(encoding="utf-8").splitlines()
            ]
            self.assertGreater(len(audits), 0)
            self.assertTrue(
                all(
                    audit["skip_reason"] == "missing_gloria_m_feedback"
                    for audit in audits
                )
            )

    def test_recorder_never_uses_direct_servo_output_api(self) -> None:
        source = Path(
            __file__
        ).parents[1].joinpath("master_triggered_camera_recorder.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("set_target(", source)
        self.assertNotIn("on_direct_servo_target_submitted", source)


if __name__ == "__main__":
    unittest.main()
