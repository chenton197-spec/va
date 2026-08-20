"""OpenArm Mini Feetech 示教臂适配器。

该模块直接使用 ``feetech-servo-sdk``，并生成与 LeRobot MotorCalibration
字段兼容的标定 JSON。运行时不需要安装或导入 LeRobot。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
import tempfile
from threading import RLock
from typing import Any, Protocol

import numpy as np

from ..interfaces import LeaderArmWithGripper


logger = logging.getLogger(__name__)

_JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 8))
_GRIPPER_NAME = "gripper"
_MOTOR_NAMES = (*_JOINT_NAMES, _GRIPPER_NAME)
_DEFAULT_MOTOR_IDS = {name: index for index, name in enumerate(_MOTOR_NAMES, start=1)}

_STS3215_RESOLUTION = 4096
_STS3215_MAX_POSITION = _STS3215_RESOLUTION - 1
_STS3215_MODEL_NUMBER = 777
_STS3215_PROTOCOL_VERSION = 0

# STS3215 control table addresses used by the read-only leader adapter.
_RETURN_DELAY_TIME = (7, 1)
_MIN_POSITION_LIMIT = (9, 2)
_MAX_POSITION_LIMIT = (11, 2)
_PHASE = (18, 1)
_HOMING_OFFSET = (31, 2)
_OPERATING_MODE = (33, 1)
_TORQUE_ENABLE = (40, 1)
_ACCELERATION = (41, 1)
_LOCK = (55, 1)
_PRESENT_POSITION = (56, 2)
_MAXIMUM_ACCELERATION = (85, 1)

_POSITION_MODE = 0
_DEFAULT_BAUDRATE = 1_000_000
_DEFAULT_PACKET_TIMEOUT_S = 0.1


@dataclass(frozen=True)
class _MotorCalibration:
    """LeRobot ``MotorCalibration`` JSON 条目的本地表示。"""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int

    @classmethod
    def from_mapping(cls, motor_name: str, value: object) -> _MotorCalibration:
        if not isinstance(value, Mapping):
            raise ValueError(f"标定文件中的 {motor_name} 必须是对象")

        fields = ("id", "drive_mode", "homing_offset", "range_min", "range_max")
        parsed: dict[str, int] = {}
        for field in fields:
            raw = value.get(field)
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ValueError(f"标定文件中的 {motor_name}.{field} 必须是整数")
            parsed[field] = raw

        calibration = cls(**parsed)
        if not 0 <= calibration.id <= 253:
            raise ValueError(f"标定文件中的 {motor_name}.id 超出 Feetech ID 范围")
        if calibration.drive_mode not in (0, 1):
            raise ValueError(f"标定文件中的 {motor_name}.drive_mode 必须为 0 或 1")
        if abs(calibration.homing_offset) > 0x7FF:
            raise ValueError(f"标定文件中的 {motor_name}.homing_offset 超出 STS3215 范围")
        if not 0 <= calibration.range_min < calibration.range_max <= _STS3215_MAX_POSITION:
            raise ValueError(f"标定文件中的 {motor_name} 位置范围无效")
        return calibration


def _load_calibration_document(path: Path, *, allow_missing: bool) -> dict[str, object]:
    """加载组合标定 JSON；首次标定时允许目标文件尚不存在。"""

    if not path.exists():
        if allow_missing:
            return {}
        raise ValueError(f"找不到 OpenArm Mini 标定文件: {path}")
    if not path.is_file():
        raise ValueError(f"OpenArm Mini 标定路径不是文件: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            root = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenArm Mini 标定文件不是有效 JSON: {path}") from exc
    if not isinstance(root, dict):
        raise ValueError("OpenArm Mini 标定文件根节点必须是对象")
    return root


def _parse_side_calibration(
    document: Mapping[str, object], side: str
) -> dict[str, _MotorCalibration]:
    """从组合 JSON 解析并校验一侧的全部八个电机。"""

    side_calibration = document.get(side)
    if not isinstance(side_calibration, Mapping):
        raise ValueError(f"OpenArm Mini 标定文件缺少 {side!r} 侧标定")
    missing = [name for name in _MOTOR_NAMES if name not in side_calibration]
    if missing:
        raise ValueError(f"OpenArm Mini {side} 标定缺少电机: {', '.join(missing)}")
    calibration = {
        name: _MotorCalibration.from_mapping(name, side_calibration[name])
        for name in _MOTOR_NAMES
    }
    motor_ids = [entry.id for entry in calibration.values()]
    if len(set(motor_ids)) != len(motor_ids):
        raise ValueError(f"OpenArm Mini {side} 标定包含重复的电机 ID")
    return calibration


def _default_calibration() -> dict[str, _MotorCalibration]:
    """返回标准 OpenArm Mini 1-8 ID 的未标定电机定义。"""

    return {
        name: _MotorCalibration(
            id=_DEFAULT_MOTOR_IDS[name],
            drive_mode=0,
            homing_offset=0,
            range_min=0,
            range_max=_STS3215_MAX_POSITION,
        )
        for name in _MOTOR_NAMES
    }


class _FeetechTransport(Protocol):
    """供适配器测试替换的最小 Feetech 传输边界。"""

    @property
    def is_connected(self) -> bool:
        """串口是否已连接。"""

    def connect(self, calibration: Mapping[str, _MotorCalibration]) -> None:
        """连接、校验并配置主臂电机。"""

    def connect_read_only(self, calibration: Mapping[str, _MotorCalibration]) -> None:
        """只连接、校验并读取主臂，不写入电机寄存器。"""

    def read_positions(self, motor_ids: tuple[int, ...], timeout_s: float) -> dict[int, int]:
        """同步读取每个电机的原始当前位置。"""

    def disconnect(self) -> None:
        """安全关闭串口。"""


class _FeetechCalibrationTransport(Protocol):
    """独立标定流程所需的 Feetech 传输边界。"""

    @property
    def is_connected(self) -> bool:
        """串口是否已连接。"""

    def connect_for_calibration(self, motor_ids: tuple[int, ...]) -> None:
        """连接标准电机 ID，但不要求已有标定。"""

    def begin_calibration(self) -> None:
        """使主臂无扭矩并进入位置标定模式。"""

    def set_half_turn_homings(self) -> dict[int, int]:
        """将当前姿态写为各电机的半圈零位。"""

    def read_positions(self, motor_ids: tuple[int, ...], timeout_s: float) -> dict[int, int]:
        """同步读取原始电机位置。"""

    def apply_calibration(self, calibration: Mapping[str, _MotorCalibration]) -> None:
        """写入完成的零位和范围，并恢复主臂读取配置。"""

    def disconnect(self) -> None:
        """安全关闭串口。"""


def _encode_sign_magnitude(value: int, sign_bit_index: int) -> int:
    """编码 Feetech STS 系列使用的符号-幅值整数。"""

    max_magnitude = (1 << sign_bit_index) - 1
    if abs(value) > max_magnitude:
        raise ValueError(f"值 {value} 无法用 {sign_bit_index} 位符号-幅值编码")
    return ((1 if value < 0 else 0) << sign_bit_index) | abs(value)


def _decode_sign_magnitude(value: int, sign_bit_index: int) -> int:
    """解码 Feetech STS 系列使用的符号-幅值整数。"""

    magnitude_mask = (1 << sign_bit_index) - 1
    return -(value & magnitude_mask) if (value >> sign_bit_index) & 1 else value & magnitude_mask


def _patch_set_packet_timeout(port_handler: Any, packet_length: int) -> None:
    """修复 PyPI Feetech SDK 的分组读取超时计算。"""

    port_handler.packet_start_time = port_handler.getCurrentTime()
    calculated_timeout_ms = (
        (port_handler.tx_time_per_byte * packet_length)
        + (port_handler.tx_time_per_byte * 3.0)
        + 50.0
    )
    timeout_limit_ms = getattr(port_handler, "_teleop_sdk_timeout_limit_ms", calculated_timeout_ms)
    port_handler.packet_timeout = min(calculated_timeout_ms, timeout_limit_ms)


class _FeetechSts3215Transport:
    """仅覆盖 OpenArm Mini 所需寄存器的 Feetech STS3215 传输层。"""

    def __init__(self, port: str, baudrate: int):
        self.port = port
        self.baudrate = baudrate
        self._sdk: Any | None = None
        self._port_handler: Any | None = None
        self._packet_handler: Any | None = None
        self._sync_reader: Any | None = None
        self._motor_ids: tuple[int, ...] = ()
        self._connected = False
        self._read_only = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibration: Mapping[str, _MotorCalibration]) -> None:
        motor_ids = tuple(calibration[name].id for name in _MOTOR_NAMES)
        self._read_only = False
        self._connect_base(motor_ids)
        try:
            self._disable_torque(motor_ids)
            self._sync_calibration(calibration)
            self._configure_leader(calibration)
        except Exception:
            self.disconnect()
            raise

    def connect_read_only(self, calibration: Mapping[str, _MotorCalibration]) -> None:
        """只打开串口、握手和读取位置，完全不修改电机寄存器。"""

        motor_ids = tuple(calibration[name].id for name in _MOTOR_NAMES)
        self._read_only = True
        try:
            self._connect_base(motor_ids)
        except Exception:
            self.disconnect()
            raise

    def connect_for_calibration(self, motor_ids: tuple[int, ...]) -> None:
        """连接未标定的标准 OpenArm Mini 电机。"""

        if len(motor_ids) != len(_MOTOR_NAMES) or len(set(motor_ids)) != len(motor_ids):
            raise ValueError("OpenArm Mini 标定需要八个不重复的电机 ID")
        self._read_only = False
        self._connect_base(motor_ids)
        try:
            self._disable_torque(motor_ids)
        except Exception:
            self.disconnect()
            raise

    def begin_calibration(self) -> None:
        """复位扭矩并设置 LeRobot OpenArm Mini 使用的标定寄存器。"""

        if not self._connected:
            raise RuntimeError("OpenArm Mini 尚未连接")
        self._disable_torque(self._motor_ids)
        for motor_id in self._motor_ids:
            self._write_if_different(motor_id, *_PHASE, 12)
            self._write_if_different(motor_id, *_OPERATING_MODE, _POSITION_MODE)

    def set_half_turn_homings(self) -> dict[int, int]:
        """将当前姿态设为零位，并返回写入的各电机偏移。"""

        if not self._connected:
            raise RuntimeError("OpenArm Mini 尚未连接")
        self._reset_calibration()
        positions = self.read_positions(self._motor_ids, _DEFAULT_PACKET_TIMEOUT_S)
        offsets = {
            motor_id: positions[motor_id] - int(_STS3215_MAX_POSITION / 2)
            for motor_id in self._motor_ids
        }
        for motor_id, offset in offsets.items():
            self._write(motor_id, *_HOMING_OFFSET, _encode_sign_magnitude(offset, 11))
        return offsets

    def apply_calibration(self, calibration: Mapping[str, _MotorCalibration]) -> None:
        """持久化全部标定寄存器并恢复安全的主臂读取配置。"""

        if not self._connected:
            raise RuntimeError("OpenArm Mini 尚未连接")
        self._sync_calibration(calibration)
        self._configure_leader(calibration)

    def _connect_base(self, motor_ids: tuple[int, ...]) -> None:
        self._load_sdk()
        assert self._port_handler is not None

        if self._connected:
            if self._motor_ids != motor_ids:
                raise RuntimeError("同一 Feetech 连接不能切换电机 ID 集合")
            return
        if not self._port_handler.openPort():
            raise ConnectionError(f"无法打开 OpenArm Mini 串口: {self.port}")

        self._connected = True
        self._motor_ids = motor_ids
        try:
            baudrate_result = self._port_handler.setBaudRate(self.baudrate)
            if baudrate_result is False:
                raise ConnectionError(f"无法将 OpenArm Mini 串口设为 {self.baudrate} baud")
            self._set_timeout(_DEFAULT_PACKET_TIMEOUT_S)
            self._verify_motor_ids(motor_ids)
        except Exception:
            self.disconnect()
            raise

    def read_positions(self, motor_ids: tuple[int, ...], timeout_s: float) -> dict[int, int]:
        if not self._connected:
            raise RuntimeError("OpenArm Mini 尚未连接")
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s 必须为正的有限秒数")
        if set(motor_ids) != set(self._motor_ids):
            raise ValueError("读取电机集合与已连接的 OpenArm Mini 不一致")

        assert self._sync_reader is not None
        self._set_timeout(timeout_s)
        address, length = _PRESENT_POSITION
        self._sync_reader.clearParam()
        self._sync_reader.start_address = address
        self._sync_reader.data_length = length
        for motor_id in motor_ids:
            added = self._sync_reader.addParam(motor_id)
            if added is False:
                raise RuntimeError(f"无法将电机 {motor_id} 加入 Feetech 同步读取")

        comm = self._sync_reader.txRxPacket()
        self._raise_for_response(comm, 0, "同步读取 OpenArm Mini 关节位置失败")

        positions: dict[int, int] = {}
        for motor_id in motor_ids:
            if not self._sync_reader.isAvailable(motor_id, address, length):
                raise ConnectionError(f"OpenArm Mini 电机 {motor_id} 未返回位置数据")
            encoded = int(self._sync_reader.getData(motor_id, address, length))
            positions[motor_id] = _decode_sign_magnitude(encoded, 15)
        return positions

    def disconnect(self) -> None:
        if not self._connected:
            self._read_only = False
            return
        try:
            if not self._read_only:
                self._disable_torque(self._motor_ids)
        except Exception as exc:  # Best effort: the port must still be closed.
            logger.warning("关闭 OpenArm Mini 前无法再次关闭扭矩: %s", exc)
        finally:
            assert self._port_handler is not None
            self._port_handler.closePort()
            self._connected = False
            self._motor_ids = ()
            self._read_only = False

    def _load_sdk(self) -> None:
        if self._sdk is not None:
            return
        try:
            import scservo_sdk as scs
        except ImportError as exc:
            raise RuntimeError(
                "OpenArm Mini 需要 feetech-servo-sdk；请安装 requirements-openarm.txt"
            ) from exc

        self._sdk = scs
        self._port_handler = scs.PortHandler(self.port)
        self._port_handler.setPacketTimeout = _patch_set_packet_timeout.__get__(
            self._port_handler, type(self._port_handler)
        )
        self._packet_handler = scs.PacketHandler(_STS3215_PROTOCOL_VERSION)
        self._sync_reader = scs.GroupSyncRead(self._port_handler, self._packet_handler, 0, 0)

    def _set_timeout(self, timeout_s: float) -> None:
        assert self._port_handler is not None
        timeout_ms = max(1.0, math.ceil(timeout_s * 1000.0))
        self._port_handler._teleop_sdk_timeout_limit_ms = timeout_ms
        set_timeout_ms = getattr(self._port_handler, "setPacketTimeoutMillis", None)
        if callable(set_timeout_ms):
            set_timeout_ms(timeout_ms)

    def _verify_motor_ids(self, motor_ids: tuple[int, ...]) -> None:
        assert self._packet_handler is not None
        assert self._port_handler is not None
        for name, motor_id in zip(_MOTOR_NAMES, motor_ids, strict=True):
            model_number, comm, error = self._packet_handler.ping(self._port_handler, motor_id)
            self._raise_for_response(comm, error, f"无法握手 OpenArm Mini 电机 {name} (ID {motor_id})")
            if int(model_number) != _STS3215_MODEL_NUMBER:
                raise RuntimeError(
                    f"OpenArm Mini 电机 {name} (ID {motor_id}) 型号不匹配："
                    f"期望 STS3215 ({_STS3215_MODEL_NUMBER})，实际 {model_number}"
                )

    def _reset_calibration(self) -> None:
        """按 LeRobot 流程清除旧零位和位置范围，再采集新的零位。"""

        for motor_id in self._motor_ids:
            self._write(motor_id, *_HOMING_OFFSET, 0)
            self._write(motor_id, *_MIN_POSITION_LIMIT, 0)
            self._write(motor_id, *_MAX_POSITION_LIMIT, _STS3215_MAX_POSITION)

    def _sync_calibration(self, calibration: Mapping[str, _MotorCalibration]) -> None:
        for name in _MOTOR_NAMES:
            target = calibration[name]
            current_homing = _decode_sign_magnitude(
                self._read(target.id, *_HOMING_OFFSET), 11
            )
            current_min = self._read(target.id, *_MIN_POSITION_LIMIT)
            current_max = self._read(target.id, *_MAX_POSITION_LIMIT)

            if current_homing != target.homing_offset:
                self._write(
                    target.id,
                    *_HOMING_OFFSET,
                    _encode_sign_magnitude(target.homing_offset, 11),
                )
            if current_min != target.range_min:
                self._write(target.id, *_MIN_POSITION_LIMIT, target.range_min)
            if current_max != target.range_max:
                self._write(target.id, *_MAX_POSITION_LIMIT, target.range_max)

    def _configure_leader(self, calibration: Mapping[str, _MotorCalibration]) -> None:
        for name in _MOTOR_NAMES:
            motor_id = calibration[name].id
            self._write_if_different(motor_id, *_RETURN_DELAY_TIME, 0)
            self._write_if_different(motor_id, *_MAXIMUM_ACCELERATION, 254)
            self._write_if_different(motor_id, *_ACCELERATION, 254)
            phase = self._read(motor_id, *_PHASE)
            if phase & 0x10:
                self._write(motor_id, *_PHASE, phase & ~0x10)
            self._write_if_different(motor_id, *_OPERATING_MODE, _POSITION_MODE)

    def _disable_torque(self, motor_ids: tuple[int, ...]) -> None:
        for motor_id in motor_ids:
            self._write_if_different(motor_id, *_TORQUE_ENABLE, 0)
            self._write_if_different(motor_id, *_LOCK, 0)

    def _write_if_different(self, motor_id: int, address: int, length: int, value: int) -> None:
        if self._read(motor_id, address, length) != value:
            self._write(motor_id, address, length, value)

    def _read(self, motor_id: int, address: int, length: int) -> int:
        assert self._packet_handler is not None
        assert self._port_handler is not None
        if length == 1:
            value, comm, error = self._packet_handler.read1ByteTxRx(
                self._port_handler, motor_id, address
            )
        elif length == 2:
            value, comm, error = self._packet_handler.read2ByteTxRx(
                self._port_handler, motor_id, address
            )
        else:
            raise ValueError(f"不支持读取 {length} 字节的 Feetech 寄存器")
        self._raise_for_response(comm, error, f"读取 Feetech 寄存器 {address} 失败")
        return int(value)

    def _write(self, motor_id: int, address: int, length: int, value: int) -> None:
        assert self._packet_handler is not None
        assert self._port_handler is not None
        if not 0 <= value < 1 << (length * 8):
            raise ValueError(f"Feetech 寄存器值 {value} 不能用 {length} 字节表示")
        data = [(value >> (8 * byte_index)) & 0xFF for byte_index in range(length)]
        comm, error = self._packet_handler.writeTxRx(
            self._port_handler, motor_id, address, length, data
        )
        self._raise_for_response(comm, error, f"写入 Feetech 寄存器 {address} 失败")

    def _raise_for_response(self, comm: int, error: int, context: str) -> None:
        assert self._sdk is not None
        assert self._packet_handler is not None
        if comm != self._sdk.COMM_SUCCESS:
            raise ConnectionError(f"{context}: {self._packet_handler.getTxRxResult(comm)}")
        if error:
            raise RuntimeError(f"{context}: {self._packet_handler.getRxPacketError(error)}")


class OpenArmMiniLeaderCalibrator:
    """独立执行单侧 OpenArm Mini 零位和夹爪行程标定。

    标定会永久写入该侧 STS3215 的 homing offset 与位置范围，并把结果保存为
    ``calibration_path`` 下的 ``left`` / ``right`` 组合 JSON。首次标定使用标准
    OpenArm Mini 电机 ID 1-8；已有同侧标定时保留其中记录的 ID。
    """

    def __init__(
        self,
        port: str,
        calibration_path: str | Path,
        side: str,
        *,
        baudrate: int = _DEFAULT_BAUDRATE,
        _transport_factory: Callable[[str, int], _FeetechCalibrationTransport] | None = None,
    ):
        if side not in {"left", "right"}:
            raise ValueError("OpenArm Mini side 必须为 'left' 或 'right'")
        if not port.strip():
            raise ValueError("OpenArm Mini 标定串口不能为空")
        if not isinstance(baudrate, int) or baudrate <= 0:
            raise ValueError("OpenArm Mini baudrate 必须为正整数")
        if not str(calibration_path).strip():
            raise ValueError("OpenArm Mini 标定文件路径不能为空")

        self.port = port
        self.calibration_path = Path(calibration_path).expanduser()
        self.side = side
        self.baudrate = baudrate
        self._document = _load_calibration_document(self.calibration_path, allow_missing=True)
        self._calibration = (
            _parse_side_calibration(self._document, side)
            if side in self._document
            else _default_calibration()
        )
        self._motor_ids = tuple(self._calibration[name].id for name in _MOTOR_NAMES)
        self._transport_factory = _transport_factory or _FeetechSts3215Transport
        self._transport: _FeetechCalibrationTransport | None = None
        self._lock = RLock()

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._transport is not None and self._transport.is_connected

    def connect(self) -> None:
        """连接该侧主臂并验证八个标准 STS3215 电机。"""

        with self._lock:
            if self._transport is not None and self._transport.is_connected:
                return
            transport = self._transport_factory(self.port, self.baudrate)
            try:
                transport.connect_for_calibration(self._motor_ids)
            except Exception:
                try:
                    transport.disconnect()
                except Exception:
                    pass
                raise
            self._transport = transport

    def calibrate(self, prompt: Callable[[str], object] = input) -> dict[str, dict[str, int]]:
        """执行交互式零位和夹爪行程标定，并立即保存该侧 JSON 条目。"""

        with self._lock:
            transport = self._transport
            if transport is None or not transport.is_connected:
                raise RuntimeError("OpenArm Mini 标定前必须先连接")

            transport.begin_calibration()
            prompt(
                f"\n[{self.side}] 零位标定\n"
                "  关闭夹爪，将主臂自然垂直下垂并保持静止。\n"
                "  就绪后按 Enter 写入关节零位..."
            )
            homing_offsets = transport.set_half_turn_homings()

            gripper_id = self._calibration[_GRIPPER_NAME].id
            prompt(f"\n[{self.side}] 夹爪标定第 1 步：完全闭合夹爪后按 Enter 记录...")
            closed_position = transport.read_positions(self._motor_ids, _DEFAULT_PACKET_TIMEOUT_S)[gripper_id]
            prompt(f"\n[{self.side}] 夹爪标定第 2 步：完全张开夹爪后按 Enter 记录...")
            open_position = transport.read_positions(self._motor_ids, _DEFAULT_PACKET_TIMEOUT_S)[gripper_id]
            if closed_position == open_position:
                raise ValueError("夹爪闭合和张开位置相同，无法建立行程标定")

            calibration = self._build_calibration(homing_offsets, closed_position, open_position)
            transport.apply_calibration(calibration)
            self._save_side_calibration(calibration)
            self._calibration = calibration
            return {name: asdict(entry) for name, entry in calibration.items()}

    def disconnect(self) -> None:
        """停止该侧标定会话并安全释放串口。"""

        with self._lock:
            transport = self._transport
            self._transport = None
            if transport is None:
                return
            try:
                transport.disconnect()
            except Exception as exc:
                logger.warning("断开 OpenArm Mini %s 标定串口失败: %s", self.side, exc)

    def _build_calibration(
        self, homing_offsets: Mapping[int, int], closed_position: int, open_position: int
    ) -> dict[str, _MotorCalibration]:
        calibration: dict[str, _MotorCalibration] = {}
        for name in _JOINT_NAMES:
            motor_id = self._calibration[name].id
            calibration[name] = _MotorCalibration.from_mapping(
                name,
                {
                    "id": motor_id,
                    "drive_mode": 0,
                    "homing_offset": int(homing_offsets[motor_id]),
                    "range_min": 0,
                    "range_max": _STS3215_MAX_POSITION,
                },
            )

        gripper_id = self._calibration[_GRIPPER_NAME].id
        if closed_position < open_position:
            range_min, range_max, drive_mode = closed_position, open_position, 0
        else:
            range_min, range_max, drive_mode = open_position, closed_position, 1
        calibration[_GRIPPER_NAME] = _MotorCalibration.from_mapping(
            _GRIPPER_NAME,
            {
                "id": gripper_id,
                "drive_mode": drive_mode,
                "homing_offset": int(homing_offsets[gripper_id]),
                "range_min": int(range_min),
                "range_max": int(range_max),
            },
        )
        return calibration

    def _save_side_calibration(self, calibration: Mapping[str, _MotorCalibration]) -> None:
        # 两个侧别的标定器会在示例启动时一起创建。保存前重新读取磁盘文件，
        # 以免后保存的一侧使用旧缓存覆盖先保存的一侧。
        document = _load_calibration_document(self.calibration_path, allow_missing=True)
        document[self.side] = {name: asdict(entry) for name, entry in calibration.items()}
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.calibration_path.parent,
                prefix=f".{self.calibration_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                json.dump(document, file, indent=2)
                file.write("\n")
                temporary_path = Path(file.name)
            temporary_path.replace(self.calibration_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        self._document = document


class OpenArmMiniLeaderArm(LeaderArmWithGripper):
    """读取独立标定后的单侧 OpenArm Mini 主臂。

    ``side`` 只选择组合标定 JSON 中的 ``left`` 或 ``right`` 条目。返回的七个
    关节保持电机原始 ``joint_1`` 至 ``joint_7`` 顺序，不做 LeRobot 专用的轴翻转
    或 joint_6/joint_7 重排。
    """

    def __init__(
        self,
        port: str,
        calibration_path: str | Path,
        side: str,
        *,
        baudrate: int = _DEFAULT_BAUDRATE,
        read_only: bool = False,
        _transport_factory: Callable[[str, int], _FeetechTransport] | None = None,
    ):
        if side not in {"left", "right"}:
            raise ValueError("OpenArm Mini side 必须为 'left' 或 'right'")
        if not isinstance(baudrate, int) or baudrate <= 0:
            raise ValueError("OpenArm Mini baudrate 必须为正整数")
        if not isinstance(read_only, bool):
            raise ValueError("OpenArm Mini read_only 必须为布尔值")

        self.port = port
        self.calibration_path = Path(calibration_path).expanduser()
        self.side = side
        self.baudrate = baudrate
        self.read_only = read_only
        self._calibration = self._load_calibration(self.calibration_path, side)
        self._motor_ids = tuple(self._calibration[name].id for name in _MOTOR_NAMES)
        self._transport_factory = _transport_factory or _FeetechSts3215Transport
        self._transport: _FeetechTransport | None = None
        self._lock = RLock()
        # 关节读取和夹爪开合量来自同一帧。缓存让附加的低频消费者无需再次
        # 访问 OpenArm 串口，从而不干扰关节遥操作的采样节奏。
        self._latest_gripper_opening: float | None = None

    @property
    def joint_count(self) -> int:
        return len(_JOINT_NAMES)

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._transport is not None and self._transport.is_connected

    def connect(self) -> None:
        """连接当前侧主臂；只读模式不会写入任何电机寄存器。"""

        with self._lock:
            if self._transport is not None and self._transport.is_connected:
                return
            transport = self._transport_factory(self.port, self.baudrate)
            try:
                if self.read_only:
                    transport.connect_read_only(self._calibration)
                else:
                    transport.connect(self._calibration)
            except Exception:
                try:
                    transport.disconnect()
                except Exception:
                    pass
                raise
            self._transport = transport
            self._latest_gripper_opening = None

    def read_joint_angles_deg(self, timeout_s: float) -> np.ndarray | None:
        """读取经标定的七个原始关节角度，单位为度。"""

        frame = self._read_frame(timeout_s)
        return None if frame is None else frame[0]

    def read_gripper_opening(self, timeout_s: float) -> float | None:
        """读取经标定的夹爪开合量，0 为闭合、1 为张开。"""

        frame = self._read_frame(timeout_s)
        return None if frame is None else frame[1]

    def read_cached_gripper_opening(self) -> float | None:
        """返回最近一次成功关节帧中的夹爪开合量，不访问串口。"""

        with self._lock:
            return self._latest_gripper_opening

    def read_joint_angles_and_gripper_opening(
        self, timeout_s: float
    ) -> tuple[np.ndarray, float] | None:
        """从同一同步读取帧返回关节角度和夹爪开合量。"""

        return self._read_frame(timeout_s)

    def disconnect(self) -> None:
        """关闭扭矩并释放串口；重复调用安全。"""

        with self._lock:
            transport = self._transport
            self._transport = None
            self._latest_gripper_opening = None
            if transport is None:
                return
            try:
                transport.disconnect()
            except Exception as exc:
                logger.warning("断开 OpenArm Mini %s 失败: %s", self.side, exc)

    def _read_frame(self, timeout_s: float) -> tuple[np.ndarray, float] | None:
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            return None
        with self._lock:
            transport = self._transport
            if transport is None or not transport.is_connected:
                return None
            try:
                positions = transport.read_positions(self._motor_ids, timeout_s)
                if set(positions) != set(self._motor_ids):
                    raise ValueError("OpenArm Mini 返回的电机位置集合不完整")
                joint_angles = np.array(
                    [
                        self._position_to_degrees(positions[self._calibration[name].id], self._calibration[name])
                        for name in _JOINT_NAMES
                    ],
                    dtype=float,
                )
                gripper_opening = self._position_to_opening(
                    positions[self._calibration[_GRIPPER_NAME].id],
                    self._calibration[_GRIPPER_NAME],
                )
            except Exception as exc:
                logger.debug("读取 OpenArm Mini %s 状态失败: %s", self.side, exc)
                return None

        if not np.all(np.isfinite(joint_angles)) or not math.isfinite(gripper_opening):
            return None
        with self._lock:
            # 读取完成前可能刚好发生 disconnect/reconnect；只有这一帧仍属于
            # 当前活动传输时才发布，避免附加消费者看到旧连接的缓存值。
            if self._transport is not transport or not transport.is_connected:
                return None
            self._latest_gripper_opening = gripper_opening
        return joint_angles, gripper_opening

    @staticmethod
    def _position_to_degrees(position: int, calibration: _MotorCalibration) -> float:
        bounded = min(calibration.range_max, max(calibration.range_min, int(position)))
        midpoint = (calibration.range_min + calibration.range_max) / 2.0
        return (bounded - midpoint) * 360.0 / _STS3215_MAX_POSITION

    @staticmethod
    def _position_to_opening(position: int, calibration: _MotorCalibration) -> float:
        bounded = min(calibration.range_max, max(calibration.range_min, int(position)))
        opening = (bounded - calibration.range_min) / (calibration.range_max - calibration.range_min)
        return 1.0 - opening if calibration.drive_mode else opening

    @staticmethod
    def _load_calibration(path: Path, side: str) -> dict[str, _MotorCalibration]:
        return _parse_side_calibration(_load_calibration_document(path, allow_missing=False), side)
