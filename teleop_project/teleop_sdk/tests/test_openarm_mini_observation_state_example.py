"""OpenArm Mini 连续状态读取示例的无硬件测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from examples import test_openarm_mini_observation_state as observation_example


class FakeLeader:
    """替代只读主臂，返回预设状态且不访问串口。"""

    instances: list[FakeLeader] = []
    frames: dict[str, tuple[np.ndarray, float] | None] = {}
    connect_errors: dict[str, Exception] = {}

    def __init__(
        self,
        *,
        port: str,
        calibration_path: str,
        side: str,
        baudrate: int,
        read_only: bool,
    ):
        self.port = port
        self.calibration_path = calibration_path
        self.side = side
        self.baudrate = baudrate
        self.read_only = read_only
        self.connect_calls = 0
        self.read_calls = 0
        self.disconnect_calls = 0
        type(self).instances.append(self)

    def connect(self) -> None:
        self.connect_calls += 1
        error = type(self).connect_errors.get(self.side)
        if error is not None:
            raise error

    def read_joint_angles_and_gripper_opening(self, _timeout_s: float):
        self.read_calls += 1
        return type(self).frames[self.side]

    def disconnect(self) -> None:
        self.disconnect_calls += 1


class OpenArmMiniObservationStateExampleTest(unittest.TestCase):
    """验证状态示例持续读取左右侧、只读构造与清理行为。"""

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        calibration_path = Path(self._temporary_directory.name) / "openarm_mini.json"
        calibration_path.write_text("{}", encoding="utf-8")
        self.runtime = SimpleNamespace(
            openarm_mini=SimpleNamespace(
                port_left="/dev/ttyACM1",
                port_right="/dev/ttyACM0",
                calibration_path=str(calibration_path),
                baudrate=1_000_000,
            )
        )
        FakeLeader.instances = []
        FakeLeader.connect_errors = {}
        FakeLeader.frames = {
            "left": (np.arange(7, dtype=float), 0.25),
            "right": (np.arange(7, dtype=float) + 10.0, 0.75),
        }

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _run_one_sample(self) -> int:
        with (
            patch.object(observation_example, "load_runtime_config", return_value=self.runtime),
            patch.object(observation_example, "OpenArmMiniLeaderArm", FakeLeader),
            patch.object(observation_example, "READ_RATE_HZ", 1.0),
            patch.object(observation_example.time, "sleep", side_effect=KeyboardInterrupt),
        ):
            return observation_example.main()

    def test_reads_both_sides_with_read_only_leaders(self) -> None:
        result = self._run_one_sample()

        self.assertEqual(result, 130)
        self.assertEqual(len(FakeLeader.instances), 2)
        left, right = FakeLeader.instances
        self.assertEqual((left.port, left.side), ("/dev/ttyACM1", "left"))
        self.assertEqual((right.port, right.side), ("/dev/ttyACM0", "right"))
        self.assertTrue(left.read_only)
        self.assertTrue(right.read_only)
        self.assertEqual([item.connect_calls for item in FakeLeader.instances], [1, 1])
        self.assertEqual([item.read_calls for item in FakeLeader.instances], [1, 1])
        self.assertEqual([item.disconnect_calls for item in FakeLeader.instances], [1, 1])

    def test_continues_when_one_side_has_no_frame(self) -> None:
        FakeLeader.frames["left"] = None

        result = self._run_one_sample()

        self.assertEqual(result, 130)
        self.assertEqual([item.read_calls for item in FakeLeader.instances], [1, 1])
        self.assertEqual([item.disconnect_calls for item in FakeLeader.instances], [1, 1])

    def test_rejects_missing_calibration_file_before_creating_leaders(self) -> None:
        self.runtime.openarm_mini.calibration_path = str(
            Path(self._temporary_directory.name) / "missing.json"
        )

        result = self._run_one_sample()

        self.assertEqual(result, 2)
        self.assertEqual(FakeLeader.instances, [])

    def test_connection_failure_disconnects_all_created_leaders(self) -> None:
        FakeLeader.connect_errors["right"] = ConnectionError("serial unavailable")

        result = self._run_one_sample()

        self.assertEqual(result, 1)
        self.assertEqual([item.connect_calls for item in FakeLeader.instances], [1, 1])
        self.assertEqual([item.disconnect_calls for item in FakeLeader.instances], [1, 1])


if __name__ == "__main__":
    unittest.main()
