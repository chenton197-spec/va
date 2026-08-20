from __future__ import annotations

import json
import tempfile
import time
import unittest
from collections import deque
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from leobot_scripts import (
    CameraFrame,
    CameraProcessDatasetRecorder,
    DepthFrame,
    RGBDFrame,
    RGBDMetadata,
    RecorderConfig,
    RecordingFollower,
)
from leobot_scripts.camera_recorder import (
    _WorkerSample,
    _append_available_master_frames,
    _append_worker_sample,
    _consume_worker_sample,
    _probe_master_camera,
)
from teleop_sdk.interfaces import FollowerArm


class _CachedFakeFollower(FollowerArm):
    @property
    def joint_count(self) -> int:
        return 2

    @property
    def joint_limits_deg(self) -> tuple[np.ndarray, np.ndarray]:
        return np.full(2, -180.0), np.full(2, 180.0)

    def connect(self) -> None:
        return None

    def read_joint_angles_deg(self) -> np.ndarray:
        raise AssertionError("The isolated recorder must use the cached state path")

    def read_cached_joint_angles_deg(self) -> np.ndarray:
        return np.array([1.0, -2.0])

    def start_servo(self) -> bool:
        return True

    def send_joint_angles_deg(self, angles_deg: np.ndarray, command_time_s: float) -> bool:
        return True

    def recover(self) -> bool:
        return True

    def stop_servo(self) -> None:
        return None

    def disconnect(self) -> None:
        return None


class _UnavailableCachedFakeFollower(_CachedFakeFollower):
    """Force numeric audit records while the child camera source still runs."""

    def read_cached_joint_angles_deg(self) -> np.ndarray | None:
        return None


class _RecordingWriter:
    def __init__(self) -> None:
        self.frames: list[object] = []
        self.skipped_ticks: list[dict[str, object]] = []

    def append_frame(self, frame: object) -> None:
        self.frames.append(frame)

    def append_skipped_tick(self, audit: dict[str, object]) -> None:
        self.skipped_ticks.append(audit)


class _TargetSelectingRGBDSource:
    def __init__(self) -> None:
        self.targets: list[int] = []

    @property
    def shape(self) -> tuple[int, int, int]:
        return (2, 3, 3)

    @property
    def depth_shape(self) -> tuple[int, int]:
        return (2, 3)

    @property
    def metadata(self) -> RGBDMetadata:
        return RGBDMetadata()

    def latest_sequence(self) -> int:
        return 0

    def next_frame_after(self, sequence: int) -> tuple[int, RGBDFrame] | None:
        return None

    def frame_at_or_before(self, target_monotonic_ns: int) -> RGBDFrame:
        self.targets.append(target_monotonic_ns)
        return _rgbd_frame(990, 7)


def _rgbd_frame(capture_monotonic_ns: int, frame_index: int) -> RGBDFrame:
    return RGBDFrame(
        rgb=np.full((2, 3, 3), frame_index, dtype=np.uint8),
        depth=DepthFrame(
            raw=np.full((2, 3), 1000 + frame_index, dtype=np.uint16),
            meters_per_raw_unit=0.001,
            capture_monotonic_ns=capture_monotonic_ns,
            source_timestamp_ns=10_000 + frame_index,
            source_frame_index=frame_index,
        ),
        rgb_capture_monotonic_ns=capture_monotonic_ns,
        rgb_source_timestamp_ns=10_000 + frame_index,
        rgb_source_frame_index=frame_index,
    )


def _rgb_frame(capture_monotonic_ns: int, frame_index: int) -> CameraFrame:
    return CameraFrame(
        rgb=np.full((2, 3, 3), frame_index, dtype=np.uint8),
        capture_monotonic_ns=capture_monotonic_ns,
        source_timestamp_ns=20_000 + frame_index,
        source_frame_index=frame_index,
    )


class _TargetSelectingRGBSource:
    def __init__(self) -> None:
        self.targets: list[int] = []

    @property
    def shape(self) -> tuple[int, int, int]:
        return (2, 3, 3)

    def latest_sequence(self) -> int:
        return 0

    def next_frame_after(self, sequence: int) -> tuple[int, CameraFrame] | None:
        return None

    def frame_at_or_before(self, target_monotonic_ns: int) -> CameraFrame:
        self.targets.append(target_monotonic_ns)
        return _rgb_frame(990, 7)


class _MasterFrameSource:
    @property
    def shape(self) -> tuple[int, int, int]:
        return (2, 3, 3)

    @property
    def depth_shape(self) -> tuple[int, int]:
        return (2, 3)

    @property
    def metadata(self) -> RGBDMetadata:
        return RGBDMetadata()

    def __init__(self) -> None:
        self.frames: list[tuple[int, RGBDFrame]] = []

    def add(self, sequence: int, frame: RGBDFrame) -> None:
        self.frames.append((sequence, frame))

    def latest_sequence(self) -> int:
        return self.frames[-1][0] if self.frames else 0

    def next_frame_after(self, sequence: int) -> tuple[int, RGBDFrame] | None:
        for frame_sequence, frame in self.frames:
            if frame_sequence > sequence:
                return frame_sequence, frame
        return None

    def frame_at_or_before(self, target_monotonic_ns: int) -> RGBDFrame:
        raise AssertionError("The master frame must be passed directly to the writer")


class _PairedFrameSource(_TargetSelectingRGBDSource):
    def frame_at_or_before(self, target_monotonic_ns: int) -> RGBDFrame:
        self.targets.append(target_monotonic_ns)
        return _rgbd_frame(target_monotonic_ns - 5, target_monotonic_ns // 10)


class _StaticCameraAdapterConfig:
    def __init__(self, camera_names: tuple[str, ...]) -> None:
        self._camera_names = camera_names

    @property
    def camera_names(self) -> tuple[str, ...]:
        return self._camera_names

    def open(self) -> object:
        raise AssertionError("This adapter is only used for constructor validation")


class _ProcessFakeRGBDSource:
    """A child-process source that emits one retained frame about every 30 Hz."""

    def __init__(self) -> None:
        self._sequence = 0
        self._last_emitted_ns = time.perf_counter_ns()

    @property
    def shape(self) -> tuple[int, int, int]:
        return (2, 3, 3)

    @property
    def depth_shape(self) -> tuple[int, int]:
        return (2, 3)

    @property
    def metadata(self) -> RGBDMetadata:
        return RGBDMetadata()

    def latest_sequence(self) -> int:
        return self._sequence

    def next_frame_after(self, sequence: int) -> tuple[int, RGBDFrame] | None:
        now_ns = time.perf_counter_ns()
        if now_ns - self._last_emitted_ns < 25_000_000:
            return None
        self._sequence = max(self._sequence, sequence) + 1
        self._last_emitted_ns = now_ns
        return self._sequence, _rgbd_frame(now_ns - 1_000_000, self._sequence)

    def frame_at_or_before(self, target_monotonic_ns: int) -> RGBDFrame:
        return _rgbd_frame(target_monotonic_ns, self._sequence)


class _ProcessFakeRGBSource:
    """A child-process RGB-only source that emits one retained frame about every 30 Hz."""

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
        if now_ns - self._last_emitted_ns < 25_000_000:
            return None
        self._sequence = max(self._sequence, sequence) + 1
        self._last_emitted_ns = now_ns
        return self._sequence, _rgb_frame(now_ns - 1_000_000, self._sequence)

    def frame_at_or_before(self, target_monotonic_ns: int) -> CameraFrame:
        return _rgb_frame(target_monotonic_ns, self._sequence)


class _ProcessFakeCameraSession:
    def __init__(self) -> None:
        self._sources = {"hand": _ProcessFakeRGBDSource()}

    @property
    def sources(self) -> Mapping[str, _ProcessFakeRGBDSource]:
        return self._sources

    def close(self) -> None:
        return None


class _ProcessFakeCameraAdapterConfig:
    @property
    def camera_names(self) -> tuple[str, ...]:
        return ("hand",)

    def open(self) -> _ProcessFakeCameraSession:
        return _ProcessFakeCameraSession()


class _ProcessFakeRGBCameraSession:
    def __init__(self) -> None:
        self._sources = {"hand": _ProcessFakeRGBSource()}

    @property
    def sources(self) -> Mapping[str, _ProcessFakeRGBSource]:
        return self._sources

    def close(self) -> None:
        return None


class _ProcessFakeRGBCameraAdapterConfig:
    @property
    def camera_names(self) -> tuple[str, ...]:
        return ("hand",)

    def open(self) -> _ProcessFakeRGBCameraSession:
        return _ProcessFakeRGBCameraSession()


class CameraProcessDatasetRecorderTest(unittest.TestCase):
    def test_multiple_cameras_require_an_explicit_master(self) -> None:
        config = RecorderConfig(root=Path("dataset"), robot_type="fake", fps=30)
        adapter = _StaticCameraAdapterConfig(("hand", "head"))

        with self.assertRaisesRegex(ValueError, "master_camera_name is required"):
            CameraProcessDatasetRecorder(config, adapter)  # type: ignore[arg-type]

        recorder = CameraProcessDatasetRecorder(
            config,
            adapter,  # type: ignore[arg-type]
            master_camera_name="hand",
            numeric_sample_fps=60,
        )
        self.assertEqual(recorder.master_camera_name, "hand")
        self.assertEqual(recorder.numeric_sample_fps, 60)

    def test_master_driven_worker_writes_each_new_master_frame_once(self) -> None:
        writer = _RecordingWriter()
        hand = _MasterFrameSource()
        head = _PairedFrameSource()
        history: deque[_WorkerSample] = deque(maxlen=8)
        sources = {
            "observation.images.hand": hand,
            "observation.images.head": head,
        }
        first_sample = _WorkerSample(
            state=np.array([1.0, 2.0]),
            action=np.array([3.0, 4.0]),
            gripper_state=None,
            gripper_action=None,
            capture_monotonic_ns=100,
            audit={"tick_index": 1},
        )
        hand.add(1, _rgbd_frame(150, 1))
        sequence = _consume_worker_sample(
            writer,  # type: ignore[arg-type]
            sources,
            "observation.images.hand",
            history,
            0,
            first_sample,
        )

        second_sample = _WorkerSample(
            state=np.array([5.0, 6.0]),
            action=np.array([7.0, 8.0]),
            gripper_state=None,
            gripper_action=None,
            capture_monotonic_ns=200,
            audit={"tick_index": 2},
        )
        hand.add(2, _rgbd_frame(250, 2))
        sequence = _consume_worker_sample(
            writer,  # type: ignore[arg-type]
            sources,
            "observation.images.hand",
            history,
            sequence,
            second_sample,
        )

        self.assertEqual(sequence, 2)
        self.assertEqual(head.targets, [150, 250])
        self.assertEqual(len(writer.frames), 2)
        first_frame, second_frame = writer.frames
        self.assertEqual(
            getattr(first_frame, "cameras")["observation.images.hand"].source_frame_index,
            1,
        )
        self.assertEqual(
            getattr(second_frame, "cameras")["observation.images.hand"].source_frame_index,
            2,
        )
        np.testing.assert_array_equal(getattr(first_frame, "state"), [1.0, 2.0])
        np.testing.assert_array_equal(getattr(second_frame, "state"), [5.0, 6.0])
        self.assertEqual(getattr(first_frame, "audit")["target_monotonic_ns"], 150)
        self.assertEqual(getattr(second_frame, "audit")["target_monotonic_ns"], 250)

    def test_health_probe_uses_an_independent_master_cursor(self) -> None:
        hand = _MasterFrameSource()
        hand.add(1, _rgbd_frame(100, 1))
        hand.add(2, _rgbd_frame(200, 2))
        captures: list[int] = []

        sequence = _probe_master_camera(hand, 0, captures.append)

        self.assertEqual(sequence, 2)
        self.assertEqual(captures, [100, 200])

    def test_master_frame_never_selects_a_future_numeric_sample(self) -> None:
        writer = _RecordingWriter()
        hand = _MasterFrameSource()
        history: deque[_WorkerSample] = deque(
            [
                _WorkerSample(
                    state=np.array([1.0, 2.0]),
                    action=np.array([3.0, 4.0]),
                    gripper_state=None,
                    gripper_action=None,
                    capture_monotonic_ns=100,
                    audit={"tick_index": 1},
                ),
                _WorkerSample(
                    state=np.array([9.0, 10.0]),
                    action=np.array([11.0, 12.0]),
                    gripper_state=None,
                    gripper_action=None,
                    capture_monotonic_ns=120,
                    audit={"tick_index": 2},
                ),
            ],
            maxlen=8,
        )
        hand.add(1, _rgbd_frame(115, 1))

        _append_available_master_frames(
            writer,  # type: ignore[arg-type]
            {"observation.images.hand": hand},
            "observation.images.hand",
            history,
            0,
        )

        self.assertEqual(len(writer.frames), 1)
        frame = writer.frames[0]
        np.testing.assert_array_equal(getattr(frame, "state"), [1.0, 2.0])
        self.assertEqual(getattr(frame, "audit")["state_sample_age_ns"], 15)

    def test_worker_selects_rgbd_frame_by_parent_target_timestamp(self) -> None:
        writer = _RecordingWriter()
        source = _TargetSelectingRGBDSource()
        sample = _WorkerSample(
            state=np.array([1.0, 2.0]),
            action=np.array([3.0, 4.0]),
            gripper_state=None,
            gripper_action=None,
            capture_monotonic_ns=1_000,
            audit={"tick_index": 4},
        )

        _append_worker_sample(writer, {"observation.images.hand": source}, sample)  # type: ignore[arg-type]

        self.assertEqual(source.targets, [1_000])
        self.assertEqual(len(writer.frames), 1)
        self.assertEqual(writer.skipped_ticks, [])
        frame = writer.frames[0]
        audit = getattr(frame, "audit")
        self.assertEqual(audit["observation.images.hand.capture_age_ns"], 10)
        self.assertEqual(audit["observation.images.hand.source_frame_index"], 7)

    def test_worker_selects_rgb_frame_without_depth_sidecar(self) -> None:
        writer = _RecordingWriter()
        source = _TargetSelectingRGBSource()
        sample = _WorkerSample(
            state=np.array([1.0, 2.0]),
            action=np.array([3.0, 4.0]),
            gripper_state=None,
            gripper_action=None,
            capture_monotonic_ns=1_000,
            audit={"tick_index": 4},
        )

        _append_worker_sample(writer, {"observation.images.hand": source}, sample)  # type: ignore[arg-type]

        self.assertEqual(source.targets, [1_000])
        self.assertEqual(len(writer.frames), 1)
        frame = writer.frames[0]
        self.assertEqual(getattr(frame, "depths"), {})
        audit = getattr(frame, "audit")
        self.assertNotIn("observation.depth.hand.capture_monotonic_ns", audit)

    def test_records_camera_episode_in_a_separate_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            recorder = CameraProcessDatasetRecorder(
                RecorderConfig(
                    root=root,
                    robot_type="fake",
                    fps=30,
                    image_storage="png",
                ),
                _ProcessFakeCameraAdapterConfig(),
                numeric_sample_fps=60,
            )
            try:
                follower = RecordingFollower(_CachedFakeFollower(), recorder)
                self.assertEqual(recorder.check_health().state, "unprepared")
                health = recorder.wait_until_camera_ready(timeout_s=2.0)
                self.assertTrue(health.healthy)
                self.assertIsNotNone(health.last_master_capture_monotonic_ns)
                self.assertTrue(follower.send_joint_angles_deg(np.array([3.0, -4.0]), 0.008))
                self.assertEqual(recorder.start_episode("record with fake camera"), 0)
                time.sleep(0.16)
                self.assertEqual(recorder.stop_episode(), 0)
            finally:
                recorder.close()

            self.assertFalse(recorder.active)
            self.assertTrue((root / "data" / "chunk-000" / "episode_000000.parquet").is_file())
            info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["codebase_version"], "leobot_image_sequence_v1")
            self.assertEqual(info["fps"], 30)
            self.assertGreater(info["total_frames"], 0)

    def test_records_rgb_only_episode_without_depth_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            recorder = CameraProcessDatasetRecorder(
                RecorderConfig(root=root, robot_type="fake", fps=30, image_storage="png"),
                _ProcessFakeRGBCameraAdapterConfig(),
                numeric_sample_fps=60,
            )
            try:
                follower = RecordingFollower(_CachedFakeFollower(), recorder)
                recorder.wait_until_camera_ready(timeout_s=2.0)
                self.assertTrue(follower.send_joint_angles_deg(np.array([3.0, -4.0]), 0.008))
                self.assertEqual(recorder.start_episode("record rgb only"), 0)
                time.sleep(0.16)
                self.assertEqual(recorder.stop_episode(), 0)
            finally:
                recorder.close()

            self.assertTrue((root / "data" / "chunk-000" / "episode_000000.parquet").is_file())
            self.assertFalse((root / "depth").exists())
            self.assertFalse((root / "meta" / "depth_sources.json").exists())

    def test_master_health_advances_when_numeric_samples_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = CameraProcessDatasetRecorder(
                RecorderConfig(
                    root=Path(directory) / "dataset",
                    robot_type="fake",
                    fps=30,
                    image_storage="png",
                ),
                _ProcessFakeCameraAdapterConfig(),
                numeric_sample_fps=60,
            )
            try:
                follower = RecordingFollower(_UnavailableCachedFakeFollower(), recorder)
                before = recorder.wait_until_camera_ready(timeout_s=2.0).last_master_capture_monotonic_ns
                assert before is not None
                self.assertTrue(follower.send_joint_angles_deg(np.array([3.0, -4.0]), 0.008))
                self.assertEqual(recorder.start_episode("missing numeric feedback"), 0)

                deadline = time.monotonic() + 2.0
                after = before
                while time.monotonic() < deadline:
                    health = recorder.check_health()
                    after = health.last_master_capture_monotonic_ns
                    if after is not None and after > before:
                        break
                    time.sleep(0.05)
                self.assertIsNotNone(after)
                assert after is not None
                self.assertGreater(after, before)
                recorder.discard_episode()
            finally:
                recorder.close()


if __name__ == "__main__":
    unittest.main()
