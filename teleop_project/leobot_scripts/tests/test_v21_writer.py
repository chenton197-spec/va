from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from leobot_scripts.sources import CameraFrame
from leobot_scripts.v21_writer import CameraSpec, RecordedFrame, V21DatasetWriter, WriterConfig


class V21DatasetWriterTest(unittest.TestCase):
    def test_writes_numeric_episode_and_v21_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            writer = V21DatasetWriter(
                WriterConfig(root=root, robot_type="fake", fps=30, joint_count=2)
            )
            self.assertEqual(writer.begin_episode("move cube"), 0)
            writer.append_frame(
                RecordedFrame(
                    state=np.array([1.0, 2.0]),
                    action=np.array([3.0, 4.0]),
                    audit={"tick_index": 0},
                )
            )
            writer.append_frame(
                RecordedFrame(
                    state=np.array([2.0, 3.0]),
                    action=np.array([4.0, 5.0]),
                    audit={"tick_index": 1},
                )
            )
            self.assertEqual(writer.finish_episode(), 0)

            info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["codebase_version"], "v2.1")
            self.assertEqual(info["total_episodes"], 1)
            self.assertEqual(info["total_frames"], 2)
            self.assertTrue((root / "data" / "chunk-000" / "episode_000000.parquet").is_file())
            audit = root / "meta" / "recording_audit" / "episode_000000.jsonl"
            self.assertEqual(len(audit.read_text(encoding="utf-8").splitlines()), 2)

    def test_reopens_the_same_dataset_and_appends_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            config = WriterConfig(root=root, robot_type="fake", fps=30, joint_count=2)

            for episode_index in range(2):
                writer = V21DatasetWriter(config)
                self.assertEqual(writer.begin_episode("move cube"), episode_index)
                writer.append_frame(
                    RecordedFrame(
                        state=np.array([float(episode_index), 0.0]),
                        action=np.array([float(episode_index), 1.0]),
                    )
                )
                self.assertEqual(writer.finish_episode(), episode_index)

            info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["total_episodes"], 2)
            self.assertEqual(info["total_frames"], 2)
            self.assertTrue((root / "data" / "chunk-000" / "episode_000000.parquet").is_file())
            self.assertTrue((root / "data" / "chunk-000" / "episode_000001.parquet").is_file())

    def test_writes_named_scalar_actuators(self) -> None:
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            writer = V21DatasetWriter(
                WriterConfig(
                    root=root,
                    robot_type="hcx_dual_arm",
                    fps=30,
                    joint_count=14,
                    scalar_actuator_names=("left_gripper", "right_gripper"),
                )
            )
            writer.begin_episode("dual arm grasp")
            writer.append_frame(
                RecordedFrame(
                    state=np.arange(14, dtype=float),
                    action=np.arange(14, dtype=float) + 0.5,
                    actuator_states={"left_gripper": 0.25, "right_gripper": 0.75},
                    actuator_actions={"left_gripper": 0.5, "right_gripper": 1.0},
                )
            )
            writer.finish_episode()

            info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertIn("action.left_gripper", info["features"])
            self.assertIn("observation.right_gripper", info["features"])
            table = pq.read_table(root / "data" / "chunk-000" / "episode_000000.parquet")
            self.assertEqual(table.column("action.left_gripper").to_pylist(), [0.5])
            self.assertEqual(table.column("observation.left_gripper").to_pylist(), [0.25])
            self.assertEqual(table.column("action.right_gripper").to_pylist(), [1.0])
            self.assertEqual(table.column("observation.right_gripper").to_pylist(), [0.75])

    def test_named_scalar_actuators_require_all_configured_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = V21DatasetWriter(
                WriterConfig(
                    root=Path(directory) / "dataset",
                    robot_type="fake",
                    fps=30,
                    joint_count=2,
                    scalar_actuator_names=("left_gripper", "right_gripper"),
                )
            )
            writer.begin_episode("invalid actuator fields")
            with self.assertRaisesRegex(ValueError, "actuator_states names"):
                writer.append_frame(
                    RecordedFrame(
                        state=np.zeros(2),
                        action=np.zeros(2),
                        actuator_states={"left_gripper": 0.2},
                        actuator_actions={"left_gripper": 0.2, "right_gripper": 0.2},
                    )
                )
            writer.discard_episode()

    def test_discard_does_not_create_dataset_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            writer = V21DatasetWriter(
                WriterConfig(root=root, robot_type="fake", fps=30, joint_count=2)
            )
            writer.begin_episode("discard me")
            writer.append_frame(
                RecordedFrame(state=np.zeros(2), action=np.zeros(2), audit={"tick_index": 0})
            )
            writer.discard_episode()

            info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["total_episodes"], 0)
            self.assertFalse((root / "data").exists())

    def test_empty_episode_preserves_audit_and_reports_skip_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            writer = V21DatasetWriter(
                WriterConfig(root=root, robot_type="fake", fps=30, joint_count=2)
            )
            writer.begin_episode("missing camera")
            for _ in range(2):
                writer.append_skipped_tick(
                    {
                        "skip_reason": (
                            "missing_camera_for_master_frame:"
                            "observation.images.left_hand"
                        )
                    }
                )
            writer.append_skipped_tick(
                {"skip_reason": "feedback_snapshot_timeout"}
            )

            with self.assertRaises(RuntimeError) as raised:
                writer.finish_episode()

            message = str(raised.exception)
            self.assertIn("episode 000000 没有完整帧", message)
            self.assertIn("候选帧=3", message)
            self.assertIn(
                "missing_camera_for_master_frame:"
                "observation.images.left_hand=2",
                message,
            )
            self.assertIn("feedback_snapshot_timeout=1", message)
            failed_audits = tuple(
                (root / "meta" / "failed_recording_audit").glob("*.jsonl")
            )
            self.assertEqual(len(failed_audits), 1)
            audits = [
                json.loads(line)
                for line in failed_audits[0].read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(audits), 3)
            info = json.loads(
                (root / "meta" / "info.json").read_text(encoding="utf-8")
            )
            self.assertEqual(info["total_episodes"], 0)
            writer.discard_episode()

    def test_repeated_empty_episodes_keep_distinct_audits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            writer = V21DatasetWriter(
                WriterConfig(root=root, robot_type="fake", fps=30, joint_count=2)
            )

            for _ in range(2):
                self.assertEqual(writer.begin_episode("no head frames"), 0)
                with self.assertRaises(RuntimeError) as raised:
                    writer.finish_episode()
                self.assertIn("候选帧=0", str(raised.exception))
                self.assertIn("未收到头部相机候选帧", str(raised.exception))
                writer.discard_episode()

            failed_audits = tuple(
                (root / "meta" / "failed_recording_audit").glob("*.jsonl")
            )
            self.assertEqual(len(failed_audits), 2)

    @unittest.skipUnless(importlib.util.find_spec("av"), "PyAV is an optional video dependency")
    def test_encodes_v21_video_feature(self) -> None:
        import av

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            camera = CameraSpec("observation.images.wrist", (32, 32, 3))
            writer = V21DatasetWriter(
                WriterConfig(root=root, robot_type="fake", fps=30, joint_count=2, cameras=(camera,))
            )
            writer.begin_episode("inspect image")
            writer.append_frame(
                RecordedFrame(
                    state=np.zeros(2),
                    action=np.zeros(2),
                    cameras={
                        "observation.images.wrist": CameraFrame(
                            rgb=np.full((32, 32, 3), 127, dtype=np.uint8),
                            capture_monotonic_ns=123,
                        )
                    },
                    audit={"tick_index": 0},
                )
            )
            writer.finish_episode()

            video = root / "videos" / "chunk-000" / "observation.images.wrist" / "episode_000000.mp4"
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                # PyAV reports the installed AV1 decoder (for example libdav1d),
                # rather than the libsvtav1 encoder used when writing.
                self.assertIn(stream.codec_context.name, {"av1", "libdav1d"})
                self.assertEqual(stream.pix_fmt, "yuv420p")
                self.assertEqual(int(stream.base_rate), 30)
                self.assertEqual(sum(1 for _ in container.decode(stream)), 1)

    def test_commits_lossless_png_sequences_without_video_encoding(self) -> None:
        from PIL import Image
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            wrist = CameraSpec("observation.images.wrist", (4, 5, 3))
            head = CameraSpec("observation.images.head", (4, 5, 3))
            writer = V21DatasetWriter(
                WriterConfig(
                    root=root,
                    robot_type="fake",
                    fps=30,
                    joint_count=2,
                    cameras=(wrist, head),
                    image_storage="png",
                )
            )
            first_wrist = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
            second_wrist = np.full((4, 5, 3), 127, dtype=np.uint8)
            first_head = np.full((4, 5, 3), 9, dtype=np.uint8)
            second_head = np.full((4, 5, 3), 211, dtype=np.uint8)

            writer.begin_episode("inspect png")
            writer.append_frame(
                RecordedFrame(
                    state=np.zeros(2),
                    action=np.zeros(2),
                    cameras={
                        wrist.feature_name: CameraFrame(rgb=first_wrist),
                        head.feature_name: CameraFrame(rgb=first_head),
                    },
                )
            )
            writer.append_frame(
                RecordedFrame(
                    state=np.ones(2),
                    action=np.ones(2),
                    cameras={
                        wrist.feature_name: CameraFrame(rgb=second_wrist),
                        head.feature_name: CameraFrame(rgb=second_head),
                    },
                )
            )
            self.assertEqual(writer.finish_episode(), 0)

            info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["codebase_version"], "leobot_image_sequence_v1")
            self.assertEqual(info["image_storage"], "image_sequence")
            self.assertEqual(info["total_images"], 4)
            self.assertNotIn("video_path", info)
            self.assertNotIn("image_path", info)
            self.assertEqual(info["features"][wrist.feature_name]["dtype"], "image_sequence")

            image_dir = root / "images" / "chunk-000" / wrist.feature_name / "episode_000000"
            first_path = image_dir / "frame_000000.png"
            second_path = image_dir / "frame_000001.png"
            self.assertTrue(first_path.is_file())
            self.assertTrue(second_path.is_file())
            with Image.open(first_path) as image:
                np.testing.assert_array_equal(np.asarray(image.convert("RGB")), first_wrist)
            with Image.open(second_path) as image:
                np.testing.assert_array_equal(np.asarray(image.convert("RGB")), second_wrist)
            self.assertFalse((root / "videos").exists())

            table = pq.read_table(root / "data" / "chunk-000" / "episode_000000.parquet")
            samples = table.column(wrist.feature_name).to_pylist()
            self.assertEqual(
                [sample["path"] for sample in samples],
                [
                    "images/chunk-000/observation.images.wrist/episode_000000/frame_000000.png",
                    "images/chunk-000/observation.images.wrist/episode_000000/frame_000001.png",
                ],
            )
            self.assertAlmostEqual(samples[0]["timestamp"], 0.0, places=7)
            self.assertAlmostEqual(samples[1]["timestamp"], 1.0 / 30.0, places=7)

    def test_commits_jpg_sequences_without_video_encoding(self) -> None:
        from PIL import Image
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            camera = CameraSpec("observation.images.wrist", (4, 5, 3))
            writer = V21DatasetWriter(
                WriterConfig(
                    root=root,
                    robot_type="fake",
                    fps=30,
                    joint_count=2,
                    cameras=(camera,),
                    image_storage="jpg",
                    quality=91,
                )
            )
            image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)

            writer.begin_episode("inspect jpg")
            writer.append_frame(
                RecordedFrame(
                    state=np.zeros(2),
                    action=np.zeros(2),
                    cameras={camera.feature_name: CameraFrame(rgb=image)},
                )
            )
            self.assertEqual(writer.finish_episode(), 0)

            info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["codebase_version"], "leobot_image_sequence_v1")
            self.assertEqual(info["image_storage"], "image_sequence")
            self.assertEqual(info["features"][camera.feature_name]["dtype"], "image_sequence")

            image_path = (
                root
                / "images"
                / "chunk-000"
                / camera.feature_name
                / "episode_000000"
                / "frame_000000.jpg"
            )
            self.assertTrue(image_path.is_file())
            reference_path = Path(directory) / "reference.jpg"
            Image.fromarray(image, mode="RGB").save(reference_path, format="JPEG", quality=91)
            with Image.open(image_path) as decoded, Image.open(reference_path) as reference:
                self.assertEqual(decoded.format, "JPEG")
                self.assertEqual(decoded.size, (5, 4))
                self.assertEqual(decoded.quantization, reference.quantization)

            table = pq.read_table(root / "data" / "chunk-000" / "episode_000000.parquet")
            sample = table.column(camera.feature_name).to_pylist()[0]
            self.assertEqual(
                sample["path"],
                "images/chunk-000/observation.images.wrist/episode_000000/frame_000000.jpg",
            )

    def test_appends_jpg_episode_to_legacy_png_chunk(self) -> None:
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            camera = CameraSpec("observation.images.wrist", (2, 2, 3))
            png_writer = V21DatasetWriter(
                WriterConfig(
                    root=root,
                    robot_type="fake",
                    fps=30,
                    joint_count=2,
                    cameras=(camera,),
                    image_storage="png",
                )
            )
            png_writer.begin_episode("png episode")
            png_writer.append_frame(
                RecordedFrame(
                    state=np.zeros(2),
                    action=np.zeros(2),
                    cameras={camera.feature_name: CameraFrame(rgb=np.zeros((2, 2, 3), dtype=np.uint8))},
                )
            )
            png_writer.finish_episode()

            info_path = root / "meta" / "info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["codebase_version"] = "leobot_png_v1"
            info["image_storage"] = "png"
            info["image_path"] = (
                "images/chunk-{episode_chunk:03d}/{image_key}/episode_{episode_index:06d}/"
                "frame_{frame_index:06d}.png"
            )
            info["features"][camera.feature_name]["dtype"] = "png_sequence"
            info_path.write_text(json.dumps(info), encoding="utf-8")

            jpg_writer = V21DatasetWriter(
                WriterConfig(
                    root=root,
                    robot_type="fake",
                    fps=30,
                    joint_count=2,
                    cameras=(camera,),
                    image_storage="jpg",
                )
            )
            self.assertEqual(jpg_writer.begin_episode("jpg episode"), 1)
            jpg_writer.append_frame(
                RecordedFrame(
                    state=np.ones(2),
                    action=np.ones(2),
                    cameras={camera.feature_name: CameraFrame(rgb=np.full((2, 2, 3), 127, dtype=np.uint8))},
                )
            )
            jpg_writer.finish_episode()

            migrated = json.loads(info_path.read_text(encoding="utf-8"))
            self.assertEqual(migrated["codebase_version"], "leobot_image_sequence_v1")
            self.assertEqual(migrated["image_storage"], "image_sequence")
            self.assertNotIn("image_path", migrated)
            self.assertEqual(migrated["total_episodes"], 2)
            self.assertTrue(
                (
                    root
                    / "images"
                    / "chunk-000"
                    / camera.feature_name
                    / "episode_000000"
                    / "frame_000000.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    root
                    / "images"
                    / "chunk-000"
                    / camera.feature_name
                    / "episode_000001"
                    / "frame_000000.jpg"
                ).is_file()
            )
            row = pq.read_table(
                root / "data" / "chunk-000" / "episode_000001.parquet"
            ).column(camera.feature_name).to_pylist()[0]
            self.assertEqual(
                row["path"],
                "images/chunk-000/observation.images.wrist/episode_000001/frame_000000.jpg",
            )

    def test_preserves_explicit_row_and_per_camera_timestamps(self) -> None:
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            hand = CameraSpec("observation.images.hand", (2, 2, 3))
            head = CameraSpec("observation.images.head", (2, 2, 3))
            writer = V21DatasetWriter(
                WriterConfig(
                    root=root,
                    robot_type="fake",
                    fps=30,
                    joint_count=2,
                    cameras=(hand, head),
                    image_storage="png",
                )
            )
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            writer.begin_episode("timing")
            writer.append_frame(
                RecordedFrame(
                    state=np.zeros(2),
                    action=np.zeros(2),
                    timestamp_s=0.0,
                    cameras={
                        hand.feature_name: CameraFrame(rgb=image, timestamp_s=0.0),
                        head.feature_name: CameraFrame(rgb=image, timestamp_s=-0.021),
                    },
                )
            )
            writer.append_frame(
                RecordedFrame(
                    state=np.ones(2),
                    action=np.ones(2),
                    timestamp_s=0.041,
                    cameras={
                        hand.feature_name: CameraFrame(rgb=image, timestamp_s=0.041),
                        head.feature_name: CameraFrame(rgb=image, timestamp_s=0.019),
                    },
                )
            )
            writer.finish_episode()

            table = pq.read_table(root / "data" / "chunk-000" / "episode_000000.parquet")
            row_times = table.column("timestamp").to_pylist()
            self.assertAlmostEqual(row_times[0], 0.0, places=7)
            self.assertAlmostEqual(row_times[1], 0.041, places=7)
            hand_times = [value["timestamp"] for value in table.column(hand.feature_name).to_pylist()]
            head_times = [value["timestamp"] for value in table.column(head.feature_name).to_pylist()]
            self.assertAlmostEqual(hand_times[0], 0.0, places=7)
            self.assertAlmostEqual(hand_times[1], 0.041, places=7)
            self.assertAlmostEqual(head_times[0], -0.021, places=7)
            self.assertAlmostEqual(head_times[1], 0.019, places=7)

    def test_discards_uncommitted_png_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            camera = CameraSpec("observation.images.wrist", (2, 2, 3))
            writer = V21DatasetWriter(
                WriterConfig(
                    root=root,
                    robot_type="fake",
                    fps=30,
                    joint_count=2,
                    cameras=(camera,),
                    image_storage="png",
                )
            )
            writer.begin_episode("discard png")
            writer.append_frame(
                RecordedFrame(
                    state=np.zeros(2),
                    action=np.zeros(2),
                    cameras={camera.feature_name: CameraFrame(rgb=np.zeros((2, 2, 3), dtype=np.uint8))},
                )
            )
            writer.discard_episode()

            self.assertFalse((root / "images").exists())

    def test_rejects_mixing_png_and_video_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            V21DatasetWriter(WriterConfig(root=root, robot_type="fake", fps=30, joint_count=2))

            with self.assertRaisesRegex(ValueError, "cannot be mixed"):
                V21DatasetWriter(
                    WriterConfig(
                        root=root,
                        robot_type="fake",
                        fps=30,
                        joint_count=2,
                        image_storage="png",
                    )
                )
