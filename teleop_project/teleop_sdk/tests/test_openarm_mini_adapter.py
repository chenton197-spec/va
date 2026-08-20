"""OpenArm Mini 示教臂适配器的无硬件测试。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from teleop_sdk.adapters.openarm_mini import (
    _FeetechSts3215Transport,
    OpenArmMiniLeaderArm,
    OpenArmMiniLeaderCalibrator,
)
from teleop_sdk.interfaces import LeaderArm, LeaderArmWithGripper


class FakeFeetechTransport:
    """替代真实串口，记录适配器传入的配置和读取请求。"""

    def __init__(self):
        self.connected = False
        self.connect_calls = 0
        self.read_only_connect_calls = 0
        self.disconnect_calls = 0
        self.read_calls = 0
        self.calibration = {}
        self.positions: dict[int, int] = {}
        self.last_motor_ids: tuple[int, ...] | None = None
        self.last_timeout_s: float | None = None
        self.connect_error: Exception | None = None
        self.read_error: Exception | None = None

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self, calibration) -> None:
        self.connect_calls += 1
        if self.connect_error is not None:
            raise self.connect_error
        self.calibration = dict(calibration)
        self.connected = True

    def connect_read_only(self, calibration) -> None:
        self.read_only_connect_calls += 1
        if self.connect_error is not None:
            raise self.connect_error
        self.calibration = dict(calibration)
        self.connected = True

    def read_positions(self, motor_ids: tuple[int, ...], timeout_s: float) -> dict[int, int]:
        self.read_calls += 1
        self.last_motor_ids = motor_ids
        self.last_timeout_s = timeout_s
        if self.read_error is not None:
            raise self.read_error
        return dict(self.positions)

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False


class FakeFeetechCalibrationTransport:
    """替代标定串口，记录写入并返回预设的夹爪位置。"""

    def __init__(
        self,
        *,
        homing_offsets: dict[int, int],
        position_frames: list[dict[int, int]],
    ):
        self.connected = False
        self.connect_calls = 0
        self.begin_calls = 0
        self.set_homing_calls = 0
        self.read_calls = 0
        self.apply_calls = 0
        self.disconnect_calls = 0
        self.motor_ids: tuple[int, ...] | None = None
        self.homing_offsets = dict(homing_offsets)
        self.position_frames = list(position_frames)
        self.applied_calibration = None

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect_for_calibration(self, motor_ids: tuple[int, ...]) -> None:
        self.connect_calls += 1
        self.motor_ids = motor_ids
        self.connected = True

    def begin_calibration(self) -> None:
        self.begin_calls += 1

    def set_half_turn_homings(self) -> dict[int, int]:
        self.set_homing_calls += 1
        return dict(self.homing_offsets)

    def read_positions(self, motor_ids: tuple[int, ...], _timeout_s: float) -> dict[int, int]:
        self.read_calls += 1
        if tuple(motor_ids) != self.motor_ids:
            raise AssertionError("unexpected motor IDs")
        if not self.position_frames:
            raise AssertionError("unexpected position read")
        return dict(self.position_frames.pop(0))

    def apply_calibration(self, calibration) -> None:
        self.apply_calls += 1
        self.applied_calibration = dict(calibration)

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False


class FakeScservoPortHandler:
    """最小 scservo_sdk PortHandler 替身，不访问真实串口。"""

    def __init__(self, port: str):
        self.port = port
        self.is_open = False
        self.tx_time_per_byte = 0.01
        self.packet_start_time = 0.0
        self.packet_timeout = 0.0

    def openPort(self) -> bool:
        self.is_open = True
        return True

    def closePort(self) -> None:
        self.is_open = False

    def setBaudRate(self, baudrate: int) -> bool:
        self.baudrate = baudrate
        return True

    def setPacketTimeoutMillis(self, timeout_ms: float) -> None:
        self.packet_timeout = timeout_ms

    def getCurrentTime(self) -> float:
        return 1.0


class FakeScservoPacketHandler:
    """返回已同步的寄存器值，避免测试依赖厂商 SDK 或设备。"""

    def __init__(self) -> None:
        self.write_calls: list[tuple[int, int, int, list[int]]] = []

    def ping(self, _port, _motor_id: int) -> tuple[int, int, int]:
        return 777, 0, 0

    def read1ByteTxRx(self, _port, _motor_id: int, address: int) -> tuple[int, int, int]:
        values = {7: 0, 18: 0, 33: 0, 40: 0, 41: 254, 55: 0, 85: 254}
        return values[address], 0, 0

    def read2ByteTxRx(self, _port, motor_id: int, address: int) -> tuple[int, int, int]:
        if address == 31:
            return 0, 0, 0
        if address == 9:
            return (100 if motor_id == 18 else 0), 0, 0
        if address == 11:
            return (900 if motor_id == 18 else 4095), 0, 0
        raise AssertionError(f"unexpected 2-byte read at address {address}")

    def writeTxRx(self, _port, motor_id: int, address: int, length: int, data: list[int]):
        self.write_calls.append((motor_id, address, length, data))
        return 0, 0

    @staticmethod
    def getTxRxResult(_comm: int) -> str:
        return "communication failure"

    @staticmethod
    def getRxPacketError(_error: int) -> str:
        return "servo error"


class FakeScservoSyncReader:
    """模拟 GroupSyncRead，将每个 ID 映射到可预测的原始位置。"""

    def __init__(self, _port, _packet_handler, start_address: int, data_length: int):
        self.start_address = start_address
        self.data_length = data_length
        self.motor_ids: list[int] = []

    def clearParam(self) -> None:
        self.motor_ids = []

    def addParam(self, motor_id: int) -> bool:
        self.motor_ids.append(motor_id)
        return True

    @staticmethod
    def txRxPacket() -> int:
        return 0

    def isAvailable(self, motor_id: int, _address: int, _length: int) -> bool:
        return motor_id in self.motor_ids

    @staticmethod
    def getData(motor_id: int, _address: int, _length: int) -> int:
        return motor_id * 100


def fake_scservo_sdk(packet_handler: FakeScservoPacketHandler | None = None) -> SimpleNamespace:
    """构造传输层所需的 scservo_sdk 表面。"""

    handler = packet_handler or FakeScservoPacketHandler()

    return SimpleNamespace(
        COMM_SUCCESS=0,
        PortHandler=FakeScservoPortHandler,
        PacketHandler=lambda _protocol: handler,
        GroupSyncRead=FakeScservoSyncReader,
    )


class OpenArmMiniLeaderArmTest(unittest.TestCase):
    """验证 LeRobot 标定兼容、转换和示教臂生命周期。"""

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.calibration_path = Path(self._temporary_directory.name) / "openarm_mini.json"
        with self.calibration_path.open("w", encoding="utf-8") as file:
            json.dump(self._combined_calibration(), file)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    @staticmethod
    def _motor(motor_id: int, *, drive_mode: int = 0) -> dict[str, int]:
        return {
            "id": motor_id,
            "drive_mode": drive_mode,
            "homing_offset": 0,
            "range_min": 0,
            "range_max": 4095,
        }

    @classmethod
    def _side(cls, first_id: int, *, gripper_drive_mode: int) -> dict[str, dict[str, int]]:
        side = {f"joint_{index}": cls._motor(first_id + index - 1) for index in range(1, 8)}
        side["gripper"] = {
            "id": first_id + 7,
            "drive_mode": gripper_drive_mode,
            "homing_offset": 0,
            "range_min": 100,
            "range_max": 900,
        }
        return side

    @classmethod
    def _combined_calibration(cls) -> dict[str, object]:
        return {
            "left": cls._side(11, gripper_drive_mode=1),
            "right": cls._side(21, gripper_drive_mode=0),
        }

    def _make_leader(
        self, side: str, transport: FakeFeetechTransport
    ) -> OpenArmMiniLeaderArm:
        return OpenArmMiniLeaderArm(
            port=f"/dev/{side}",
            calibration_path=self.calibration_path,
            side=side,
            _transport_factory=lambda _port, _baudrate: transport,
        )

    def test_connect_selects_side_and_exposes_leader_contract(self) -> None:
        transport = FakeFeetechTransport()
        leader = self._make_leader("left", transport)

        self.assertIsInstance(leader, LeaderArm)
        self.assertIsInstance(leader, LeaderArmWithGripper)
        self.assertEqual(leader.joint_count, 7)
        leader.connect()
        leader.connect()

        self.assertTrue(leader.is_connected)
        self.assertEqual(transport.connect_calls, 1)
        self.assertEqual(transport.calibration["joint_1"].id, 11)
        self.assertEqual(transport.calibration["gripper"].id, 18)

    def test_read_only_connect_does_not_sync_calibration(self) -> None:
        transport = FakeFeetechTransport()
        leader = OpenArmMiniLeaderArm(
            port="/dev/left",
            calibration_path=self.calibration_path,
            side="left",
            read_only=True,
            _transport_factory=lambda _port, _baudrate: transport,
        )

        leader.connect()
        leader.disconnect()

        self.assertEqual(transport.connect_calls, 0)
        self.assertEqual(transport.read_only_connect_calls, 1)
        self.assertEqual(transport.disconnect_calls, 1)

    def test_combined_read_preserves_raw_joint_order_and_normalizes_gripper(self) -> None:
        transport = FakeFeetechTransport()
        leader = self._make_leader("left", transport)
        leader.connect()
        joint_positions = [0, 4095, 1024, 3071, 2048, 512, 3583]
        for index, position in enumerate(joint_positions, start=11):
            transport.positions[index] = position
        transport.positions[18] = 100

        frame = leader.read_joint_angles_and_gripper_opening(timeout_s=0.025)

        assert frame is not None
        joint_angles, opening = frame
        expected = np.array(
            [(position - 2047.5) * 360.0 / 4095.0 for position in joint_positions],
            dtype=float,
        )
        np.testing.assert_allclose(joint_angles, expected)
        self.assertEqual(opening, 1.0)
        self.assertEqual(transport.read_calls, 1)
        self.assertEqual(transport.last_motor_ids, tuple(range(11, 19)))
        self.assertEqual(transport.last_timeout_s, 0.025)

    def test_cached_gripper_opening_reuses_the_latest_joint_read_without_serial_io(self) -> None:
        transport = FakeFeetechTransport()
        leader = self._make_leader("left", transport)
        leader.connect()
        for motor_id in range(11, 18):
            transport.positions[motor_id] = 2048
        transport.positions[18] = 100

        self.assertIsNone(leader.read_cached_gripper_opening())
        self.assertIsNotNone(leader.read_joint_angles_deg(0.1))
        self.assertEqual(leader.read_cached_gripper_opening(), 1.0)
        self.assertEqual(transport.read_calls, 1)

        # 低频夹爪消费者读取缓存时不能额外占用 OpenArm 串口。
        self.assertEqual(leader.read_cached_gripper_opening(), 1.0)
        self.assertEqual(transport.read_calls, 1)

        leader.disconnect()
        self.assertIsNone(leader.read_cached_gripper_opening())

    def test_side_does_not_apply_lerobot_axis_flips_or_remapping(self) -> None:
        left_transport = FakeFeetechTransport()
        right_transport = FakeFeetechTransport()
        left = self._make_leader("left", left_transport)
        right = self._make_leader("right", right_transport)
        left.connect()
        right.connect()
        raw_positions = [0, 400, 800, 1200, 1600, 2000, 2400]
        for index, position in enumerate(raw_positions, start=11):
            left_transport.positions[index] = position
        for index, position in enumerate(raw_positions, start=21):
            right_transport.positions[index] = position
        left_transport.positions[18] = 900
        right_transport.positions[28] = 900

        left_angles = left.read_joint_angles_deg(0.1)
        right_angles = right.read_joint_angles_deg(0.1)

        assert left_angles is not None
        assert right_angles is not None
        np.testing.assert_allclose(left_angles, right_angles)
        self.assertEqual(left.read_gripper_opening(0.1), 0.0)
        self.assertEqual(right.read_gripper_opening(0.1), 1.0)

    def test_read_failures_and_invalid_timeout_return_none(self) -> None:
        transport = FakeFeetechTransport()
        leader = self._make_leader("left", transport)

        self.assertIsNone(leader.read_joint_angles_deg(0.1))
        leader.connect()
        transport.read_error = ConnectionError("serial timeout")

        self.assertIsNone(leader.read_joint_angles_deg(0.1))
        self.assertEqual(transport.read_calls, 1)
        self.assertIsNone(leader.read_joint_angles_deg(0.0))
        self.assertEqual(transport.read_calls, 1)

    def test_disconnect_is_idempotent_and_connect_failure_cleans_up(self) -> None:
        transport = FakeFeetechTransport()
        leader = self._make_leader("left", transport)
        leader.connect()

        leader.disconnect()
        leader.disconnect()

        self.assertFalse(leader.is_connected)
        self.assertEqual(transport.disconnect_calls, 1)

        failing_transport = FakeFeetechTransport()
        failing_transport.connect_error = RuntimeError("handshake failed")
        failing_leader = self._make_leader("right", failing_transport)
        with self.assertRaisesRegex(RuntimeError, "handshake failed"):
            failing_leader.connect()
        self.assertFalse(failing_leader.is_connected)
        self.assertEqual(failing_transport.disconnect_calls, 1)

    def test_invalid_combined_calibration_is_rejected(self) -> None:
        invalid_path = Path(self._temporary_directory.name) / "invalid.json"
        invalid = self._combined_calibration()
        assert isinstance(invalid["left"], dict)
        del invalid["left"]["joint_7"]
        with invalid_path.open("w", encoding="utf-8") as file:
            json.dump(invalid, file)

        with self.assertRaisesRegex(ValueError, "joint_7"):
            OpenArmMiniLeaderArm("/dev/left", invalid_path, "left")

    def test_sts3215_transport_uses_the_vendor_sdk_surface_without_a_serial_port(self) -> None:
        calibration_source = self._combined_calibration()["left"]
        assert isinstance(calibration_source, dict)
        calibration = {
            name: type("Calibration", (), value)()
            for name, value in calibration_source.items()
        }
        transport = _FeetechSts3215Transport("/dev/fake-openarm", 1_000_000)

        with patch.dict(sys.modules, {"scservo_sdk": fake_scservo_sdk()}):
            transport.connect(calibration)
            positions = transport.read_positions(tuple(range(11, 19)), 0.05)
            transport.disconnect()

        self.assertEqual(positions, {motor_id: motor_id * 100 for motor_id in range(11, 19)})
        self.assertFalse(transport.is_connected)

    def test_sts3215_read_only_connection_never_writes_a_register(self) -> None:
        calibration_source = self._combined_calibration()["left"]
        assert isinstance(calibration_source, dict)
        calibration = {
            name: type("Calibration", (), value)()
            for name, value in calibration_source.items()
        }
        transport = _FeetechSts3215Transport("/dev/fake-openarm", 1_000_000)
        packet_handler = FakeScservoPacketHandler()

        with patch.dict(sys.modules, {"scservo_sdk": fake_scservo_sdk(packet_handler)}):
            transport.connect_read_only(calibration)
            positions = transport.read_positions(tuple(range(11, 19)), 0.05)
            transport.disconnect()

        self.assertEqual(positions, {motor_id: motor_id * 100 for motor_id in range(11, 19)})
        self.assertEqual(packet_handler.write_calls, [])


class OpenArmMiniLeaderCalibratorTest(unittest.TestCase):
    """验证独立标定会创建兼容 JSON，且不需要真实串口。"""

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.calibration_path = Path(self._temporary_directory.name) / "nested" / "openarm_mini.json"

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    @staticmethod
    def _positions(gripper_position: int) -> dict[int, int]:
        return {motor_id: (gripper_position if motor_id == 8 else motor_id * 100) for motor_id in range(1, 9)}

    @staticmethod
    def _homing_offsets() -> dict[int, int]:
        return {motor_id: motor_id - 4 for motor_id in range(1, 9)}

    def _make_calibrator(
        self, side: str, transport: FakeFeetechCalibrationTransport
    ) -> OpenArmMiniLeaderCalibrator:
        return OpenArmMiniLeaderCalibrator(
            port=f"/dev/{side}",
            calibration_path=self.calibration_path,
            side=side,
            _transport_factory=lambda _port, _baudrate: transport,
        )

    def test_calibration_creates_missing_json_and_sets_forward_gripper_range(self) -> None:
        transport = FakeFeetechCalibrationTransport(
            homing_offsets=self._homing_offsets(),
            position_frames=[self._positions(120), self._positions(900)],
        )
        calibrator = self._make_calibrator("left", transport)

        calibrator.connect()
        result = calibrator.calibrate(prompt=lambda _message: None)
        calibrator.disconnect()

        with self.calibration_path.open("r", encoding="utf-8") as file:
            document = json.load(file)
        left = document["left"]
        self.assertEqual(set(left), {*(f"joint_{index}" for index in range(1, 8)), "gripper"})
        self.assertEqual(left["joint_1"], result["joint_1"])
        self.assertEqual(left["joint_1"]["id"], 1)
        self.assertEqual(left["joint_1"]["homing_offset"], -3)
        self.assertEqual(left["joint_1"]["range_min"], 0)
        self.assertEqual(left["joint_1"]["range_max"], 4095)
        self.assertEqual(left["gripper"]["drive_mode"], 0)
        self.assertEqual(left["gripper"]["range_min"], 120)
        self.assertEqual(left["gripper"]["range_max"], 900)
        self.assertEqual(transport.apply_calls, 1)
        self.assertEqual(transport.applied_calibration["gripper"].drive_mode, 0)
        self.assertEqual(transport.disconnect_calls, 1)

    def test_two_calibrators_created_before_file_exists_preserve_both_sides(self) -> None:
        left_transport = FakeFeetechCalibrationTransport(
            homing_offsets=self._homing_offsets(),
            position_frames=[self._positions(100), self._positions(800)],
        )
        right_transport = FakeFeetechCalibrationTransport(
            homing_offsets=self._homing_offsets(),
            position_frames=[self._positions(900), self._positions(100)],
        )
        left = self._make_calibrator("left", left_transport)
        right = self._make_calibrator("right", right_transport)

        left.connect()
        left.calibrate(prompt=lambda _message: None)
        right.connect()
        right.calibrate(prompt=lambda _message: None)
        left.disconnect()
        right.disconnect()

        with self.calibration_path.open("r", encoding="utf-8") as file:
            document = json.load(file)
        self.assertEqual(set(document), {"left", "right"})
        self.assertEqual(document["left"]["gripper"]["drive_mode"], 0)
        self.assertEqual(document["right"]["gripper"]["drive_mode"], 1)
        self.assertEqual(document["right"]["gripper"]["range_min"], 100)
        self.assertEqual(document["right"]["gripper"]["range_max"], 900)

    def test_calibration_preserves_an_existing_opposite_side(self) -> None:
        existing_right = OpenArmMiniLeaderArmTest._side(21, gripper_drive_mode=1)
        self.calibration_path.parent.mkdir(parents=True)
        with self.calibration_path.open("w", encoding="utf-8") as file:
            json.dump({"right": existing_right}, file)
        transport = FakeFeetechCalibrationTransport(
            homing_offsets=self._homing_offsets(),
            position_frames=[self._positions(100), self._positions(800)],
        )
        calibrator = self._make_calibrator("left", transport)

        calibrator.connect()
        calibrator.calibrate(prompt=lambda _message: None)
        calibrator.disconnect()

        with self.calibration_path.open("r", encoding="utf-8") as file:
            document = json.load(file)
        self.assertEqual(document["right"], existing_right)

    def test_equal_gripper_positions_fail_without_creating_file(self) -> None:
        transport = FakeFeetechCalibrationTransport(
            homing_offsets=self._homing_offsets(),
            position_frames=[self._positions(500), self._positions(500)],
        )
        calibrator = self._make_calibrator("left", transport)
        calibrator.connect()

        with self.assertRaisesRegex(ValueError, "位置相同"):
            calibrator.calibrate(prompt=lambda _message: None)
        calibrator.disconnect()

        self.assertFalse(self.calibration_path.exists())
        self.assertEqual(transport.apply_calls, 0)


if __name__ == "__main__":
    unittest.main()
