"""OpenArm Mini 独立标定示例的无硬件测试。"""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples import test_openarm_mini_calibration as calibration_example


class FakeCalibrator:
    """记录示例的标定生命周期，不访问真实串口或修改电机。"""

    instances: list[FakeCalibrator] = []
    calibration_errors: dict[str, Exception] = {}

    def __init__(self, *, port: str, calibration_path: str, side: str, baudrate: int):
        self.port = port
        self.calibration_path = calibration_path
        self.side = side
        self.baudrate = baudrate
        self.connect_calls = 0
        self.calibrate_calls = 0
        self.disconnect_calls = 0
        type(self).instances.append(self)

    def connect(self) -> None:
        self.connect_calls += 1

    def calibrate(self) -> dict[str, dict[str, int]]:
        self.calibrate_calls += 1
        error = type(self).calibration_errors.get(self.side)
        if error is not None:
            raise error
        return {"joint_1": {"id": 1}}

    def disconnect(self) -> None:
        self.disconnect_calls += 1


class OpenArmMiniCalibrationExampleTest(unittest.TestCase):
    """验证示例按左右顺序标定，并始终释放已连接的串口。"""

    def setUp(self) -> None:
        FakeCalibrator.instances = []
        FakeCalibrator.calibration_errors = {}
        self.runtime = SimpleNamespace(
            openarm_mini=SimpleNamespace(
                port_left="/dev/ttyACM1",
                port_right="/dev/ttyACM0",
                calibration_path="./my_openarm_mini.json",
                baudrate=1_000_000,
            )
        )

    def _run(self) -> int:
        with (
            patch.object(calibration_example, "load_runtime_config", return_value=self.runtime),
            patch.object(calibration_example, "OpenArmMiniLeaderCalibrator", FakeCalibrator),
            patch("builtins.input", return_value=""),
        ):
            return calibration_example.main()

    def test_calibrates_both_sides_with_runtime_ports(self) -> None:
        result = self._run()

        self.assertEqual(result, 0)
        self.assertEqual(len(FakeCalibrator.instances), 2)
        left, right = FakeCalibrator.instances
        self.assertEqual((left.port, left.side), ("/dev/ttyACM1", "left"))
        self.assertEqual((right.port, right.side), ("/dev/ttyACM0", "right"))
        self.assertEqual(left.calibration_path, "./my_openarm_mini.json")
        self.assertEqual(right.baudrate, 1_000_000)
        self.assertEqual([item.connect_calls for item in FakeCalibrator.instances], [1, 1])
        self.assertEqual([item.calibrate_calls for item in FakeCalibrator.instances], [1, 1])
        self.assertEqual([item.disconnect_calls for item in FakeCalibrator.instances], [1, 1])

    def test_reports_failure_and_cleans_up_when_one_side_calibration_fails(self) -> None:
        FakeCalibrator.calibration_errors["right"] = RuntimeError("gripper range failed")

        result = self._run()

        self.assertEqual(result, 1)
        self.assertEqual([item.calibrate_calls for item in FakeCalibrator.instances], [1, 1])
        self.assertEqual([item.disconnect_calls for item in FakeCalibrator.instances], [1, 1])

    def test_rejects_missing_calibration_path_before_opening_ports(self) -> None:
        self.runtime.openarm_mini.calibration_path = ""

        result = self._run()

        self.assertEqual(result, 2)
        self.assertEqual(FakeCalibrator.instances, [])

    def test_cancels_before_opening_ports(self) -> None:
        with (
            patch.object(calibration_example, "load_runtime_config", return_value=self.runtime),
            patch.object(calibration_example, "OpenArmMiniLeaderCalibrator", FakeCalibrator),
            patch("builtins.input", side_effect=KeyboardInterrupt),
        ):
            result = calibration_example.main()

        self.assertEqual(result, 130)
        self.assertEqual(FakeCalibrator.instances, [])

    def test_cancels_and_disconnects_both_sides_during_calibration(self) -> None:
        FakeCalibrator.calibration_errors["left"] = KeyboardInterrupt()

        result = self._run()

        self.assertEqual(result, 130)
        self.assertEqual([item.disconnect_calls for item in FakeCalibrator.instances], [1, 1])


if __name__ == "__main__":
    unittest.main()
