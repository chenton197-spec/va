"""Parquet 到 MuJoCo 同步回放示例的硬件无关测试。"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from examples import replay_openarm_hcx_parquet_mujoco as example


class ParquetMujocoReplayExampleTest(unittest.TestCase):
    def test_dataset_root_accepts_standard_episode_layout(self) -> None:
        path = Path("/tmp/dataset/data/chunk-003/episode_003001.parquet")
        self.assertEqual(example._dataset_root(path), Path("/tmp/dataset"))

    def test_dataset_root_rejects_nonstandard_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "Parquet 路径"):
            example._dataset_root(Path("/tmp/episode.parquet"))

    def test_validate_timestamps_makes_episode_relative(self) -> None:
        np.testing.assert_allclose(
            example._validate_timestamps([4.0, 4.1, 4.25]),
            [0.0, 0.1, 0.25],
        )

    def test_validate_timestamps_rejects_reverse_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "单调"):
            example._validate_timestamps([0.0, 0.2, 0.1])

    def test_playback_frame_splits_left_and_right_joint_vectors(self) -> None:
        angles = np.arange(14, dtype=float)
        frame = example.PlaybackFrame(
            timestamp_s=0.0,
            left_angles_deg=angles[:7],
            right_angles_deg=angles[7:],
            image_paths={},
        )
        np.testing.assert_array_equal(frame.left_angles_deg, np.arange(7, dtype=float))
        np.testing.assert_array_equal(frame.right_angles_deg, np.arange(7, 14, dtype=float))

    def test_load_playback_frames_resolves_three_camera_paths(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with TemporaryDirectory() as directory:
            root = Path(directory)
            parquet_path = root / "data" / "chunk-000" / "episode_000000.parquet"
            parquet_path.parent.mkdir(parents=True)
            image_paths: dict[str, str] = {}
            for camera_name in example.CAMERA_COLUMNS:
                relative = (
                    f"images/chunk-000/{camera_name}/"
                    "episode_000000/frame_000000.jpg"
                )
                image_path = root / relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.touch()
                image_paths[camera_name] = relative

            arrays = [
                pa.array([2.5], type=pa.float32()),
                pa.FixedSizeListArray.from_arrays(
                    pa.array(np.arange(14, dtype=np.float32)),
                    14,
                ),
            ]
            names = ["timestamp", example.JOINT_TRAJECTORY_COLUMN]
            for camera_name in example.CAMERA_COLUMNS:
                arrays.append(
                    pa.StructArray.from_arrays(
                        [
                            pa.array([image_paths[camera_name]]),
                            pa.array([2.5], type=pa.float32()),
                        ],
                        names=["path", "timestamp"],
                    )
                )
                names.append(camera_name)
            pq.write_table(pa.Table.from_arrays(arrays, names=names), parquet_path)

            frames = example.load_playback_frames(parquet_path)

            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].timestamp_s, 0.0)
            np.testing.assert_array_equal(
                frames[0].left_angles_deg,
                np.arange(7, dtype=float),
            )
            np.testing.assert_array_equal(
                frames[0].right_angles_deg,
                np.arange(7, 14, dtype=float),
            )
            self.assertEqual(
                set(frames[0].image_paths),
                set(example.CAMERA_COLUMNS),
            )


if __name__ == "__main__":
    unittest.main()
