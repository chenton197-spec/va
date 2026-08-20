from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from leobot_scripts.sources import CameraFrame, DepthFrame, DepthMetadata, RGBDMetadata
from leobot_scripts.v21_writer import (
    CameraSpec,
    DepthCameraSpec,
    RecordedFrame,
    V21DatasetWriter,
    WriterConfig,
)


HAS_AV = importlib.util.find_spec("av") is not None
HAS_ZARR = importlib.util.find_spec("zarr") is not None


def _writer(root: Path) -> V21DatasetWriter:
    rgb_feature = "observation.images.wrist"
    return V21DatasetWriter(
        WriterConfig(
            root=root,
            robot_type="fake",
            fps=30,
            joint_count=2,
            cameras=(CameraSpec(rgb_feature, (32, 32, 3)),),
            depth_cameras=(
                DepthCameraSpec(
                    feature_name="observation.depth.wrist",
                    rgb_feature_name=rgb_feature,
                    shape=(32, 32),
                    metadata=RGBDMetadata(
                        raw_format="Y16",
                        invalid_value=0,
                        aligned_to_rgb=True,
                        color_intrinsics=np.eye(3),
                        camera_model="test-camera",
                    ),
                ),
            ),
        )
    )


def _frame(
    depth: np.ndarray,
    timestamp: int | None = 123,
    meters_per_raw_unit: float = 0.001,
    source_timestamp_ns: int | None = 456,
    source_frame_index: int | None = 7,
) -> RecordedFrame:
    return RecordedFrame(
        state=np.array([1.0, 2.0]),
        action=np.array([3.0, 4.0]),
        cameras={
            "observation.images.wrist": CameraFrame(
                rgb=np.full((32, 32, 3), 127, dtype=np.uint8),
                capture_monotonic_ns=timestamp,
                source_timestamp_ns=source_timestamp_ns,
                source_frame_index=source_frame_index,
            )
        },
        depths={
            "observation.depth.wrist": DepthFrame(
                raw=depth,
                meters_per_raw_unit=meters_per_raw_unit,
                capture_monotonic_ns=timestamp,
                source_timestamp_ns=source_timestamp_ns,
                source_frame_index=source_frame_index,
            )
        },
        audit={"tick_index": 0},
    )


@unittest.skipUnless(HAS_ZARR, "zarr is a depth collection dependency")
class DepthSidecarTest(unittest.TestCase):
    def test_writes_depth_only_episode_and_source_timestamps(self) -> None:
        import zarr

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            writer = V21DatasetWriter(
                WriterConfig(
                    root=root,
                    robot_type="fake",
                    fps=30,
                    joint_count=2,
                    depth_cameras=(
                        DepthCameraSpec(
                            feature_name="observation.depth.top",
                            rgb_feature_name=None,
                            shape=(16, 24),
                            metadata=DepthMetadata(
                                raw_format="Y16",
                                invalid_value=0,
                                source_timestamp_clock="orbbec_global_timestamp_ns",
                            ),
                        ),
                    ),
                )
            )
            writer.begin_episode("depth only")
            raw = np.full((16, 24), 333, dtype=np.uint16)
            writer.append_frame(
                RecordedFrame(
                    state=np.array([1.0, 2.0]),
                    action=np.array([3.0, 4.0]),
                    depths={
                        "observation.depth.top": DepthFrame(
                            raw=raw,
                            meters_per_raw_unit=0.001,
                            capture_monotonic_ns=10,
                            source_timestamp_ns=20,
                            source_frame_index=30,
                        )
                    },
                )
            )
            self.assertEqual(writer.finish_episode(), 0)

            group = zarr.open_group(
                str(root / "depth" / "chunk-000" / "observation.depth.top" / "episode_000000.zarr"),
                mode="r",
            )
            np.testing.assert_array_equal(group["depth_raw"][:], raw[None, ...])
            np.testing.assert_array_equal(group["source_timestamp_ns"][:], np.array([20]))
            np.testing.assert_array_equal(group["source_frame_index"][:], np.array([30]))
            self.assertIsNone(group.attrs["rgb_feature"])
            self.assertFalse(group.attrs["aligned_to_rgb"])

    def test_discard_does_not_commit_depth_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            writer = _writer(root)
            writer.begin_episode("discard depth")
            writer.append_frame(_frame(np.full((32, 32), 1000, dtype=np.uint16)))
            writer.discard_episode()

            self.assertFalse((root / "depth").exists())
            self.assertFalse((root / "meta" / "depth_sources.json").exists())

    def test_rejects_missing_or_invalid_depth_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = _writer(Path(directory) / "dataset")
            writer.begin_episode("validate depth")
            with self.assertRaisesRegex(ValueError, "depth frame"):
                writer.append_frame(
                    RecordedFrame(
                        state=np.zeros(2),
                        action=np.zeros(2),
                        cameras={
                            "observation.images.wrist": CameraFrame(
                                rgb=np.zeros((32, 32, 3), dtype=np.uint8)
                            )
                        },
                    )
                )
            writer.discard_episode()

    @unittest.skipUnless(HAS_AV, "PyAV is a video collection dependency")
    def test_writes_raw_depth_with_frame_mapping_and_metadata(self) -> None:
        import zarr

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            writer = _writer(root)
            writer.begin_episode("record depth")
            first = np.full((32, 32), 1000, dtype=np.uint16)
            second = np.full((32, 32), 2000, dtype=np.uint16)
            writer.append_frame(_frame(first, timestamp=111))
            writer.append_frame(_frame(second, timestamp=None, meters_per_raw_unit=0.002))
            self.assertEqual(writer.finish_episode(), 0)

            path = root / "depth" / "chunk-000" / "observation.depth.wrist" / "episode_000000.zarr"
            group = zarr.open_group(str(path), mode="r")
            np.testing.assert_array_equal(group["depth_raw"][:], np.stack([first, second]))
            np.testing.assert_array_equal(group["frame_index"][:], np.array([0, 1], dtype=np.int64))
            np.testing.assert_array_equal(
                group["capture_monotonic_ns"][:], np.array([111, -1], dtype=np.int64)
            )
            np.testing.assert_array_equal(
                group["source_timestamp_ns"][:], np.array([456, 456], dtype=np.int64)
            )
            np.testing.assert_array_equal(
                group["source_frame_index"][:], np.array([7, 7], dtype=np.int64)
            )
            np.testing.assert_allclose(group["meters_per_raw_unit"][:], [0.001, 0.002])
            self.assertEqual(group.attrs["rgb_feature"], "observation.images.wrist")
            self.assertEqual(group.attrs["invalid_value"], 0)

            metadata = json.loads((root / "meta" / "depth_sources.json").read_text(encoding="utf-8"))
            source = metadata["sources"]["observation.depth.wrist"]
            self.assertEqual(source["raw_format"], "Y16")
            self.assertEqual(source["shape"], [32, 32])
            self.assertEqual(source["camera_model"], "test-camera")

            info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertNotIn("observation.depth.wrist", info["features"])


if __name__ == "__main__":
    unittest.main()
