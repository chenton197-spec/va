"""Gloria-M 电机的 CAN 控制层。

本模块位于串口/CAN 传输层和 ``Actuator`` 状态对象之间：负责组装控制帧、解析
反馈帧，并维护参数读取缓存。它不决定夹爪的抓取策略，只执行上层调用方给出的
使能、模式、参数和 MIT/PV 控制命令。
"""

from __future__ import annotations

import logging
import struct
import time
from typing import Dict, Optional

from .actuator import Actuator
from .protocol_mit import pack_f32, pack_mit_command, unpack_f32, unpack_mit_feedback
from .serial_can_adapter import CanPacket, SerialCanAdapter
from .types import ControlMode, Limits

if False:  # 仅供类型检查使用，避免运行时循环导入
    from .transport import ICanTransport

_log = logging.getLogger(__name__)


def _u32_to_bytes_le(value: int) -> bytes:
    """将无符号 32 位参数编码为固件要求的小端四字节载荷。"""

    if not (0 <= int(value) <= 0xFFFFFFFF):
        raise ValueError("u32 out of range")
    return struct.pack("<I", int(value))


def _is_u32_param(rid: int) -> bool:
    # 固件的参数 RID 并非全部是 32 位浮点数；这些区间使用无符号 32 位整数解码。
    return (7 <= rid <= 10) or (13 <= rid <= 16) or (35 <= rid <= 36)


class CanController:
    """管理执行器注册、CAN 命令分发、反馈解析及参数读写。"""

    def __init__(self, adapter: "SerialCanAdapter | ICanTransport"):
        self._adapter = adapter
        self._by_can_id: Dict[int, Actuator] = {}

    def register(self, actuator: Actuator) -> None:
        # 某些固件从 command_id 返回反馈，另一些使用独立 feedback_id；两者都映射
        # 到同一个 Actuator，接收路径无需依赖具体固件版本。
        self._by_can_id[actuator.command_id] = actuator
        self._by_can_id[actuator.feedback_id] = actuator
        _log.debug("Registered actuator %r (cmd_id=0x%03X fb_id=0x%03X)",
                   actuator.name, actuator.command_id, actuator.feedback_id)

    def poll(self) -> None:
        """取尽当前 RX 缓冲区中的 CAN 帧，并同步更新执行器状态/参数缓存。"""

        for pkt in self._adapter.read_packets():
            self._handle_packet(pkt)

    def _handle_packet(self, pkt: CanPacket) -> None:
        # 本协议只有 cmd=0x11 的帧属于电机应答；其他帧交给其它使用者或直接忽略。
        if pkt.cmd != 0x11:
            return

        # 参数读取/写入应答：data[2] 为操作码（0x33/0x55），data[3] 为 RID，
        # data[4:8] 是四字节返回值。读取结果写入 Actuator.params 供同步等待逻辑消费。
        if len(pkt.data) == 8 and pkt.data[2] in (0x33, 0x55):
            rid = pkt.data[3]
            if _is_u32_param(rid):
                value = struct.unpack("<I", pkt.data[4:8])[0]
            else:
                value = unpack_f32(pkt.data[4:8])
            act = self._by_can_id.get(pkt.can_id)
            if act is not None:
                act.params[int(rid)] = value
            return

        # 其余 8 字节应答视为 MIT 布局的状态反馈。即使当前使用 PV 模式，固件也会
        # 用同一位置/速度/扭矩打包格式返回状态。
        act = self._by_can_id.get(pkt.can_id)
        if act is None and pkt.can_id == 0x00 and len(pkt.data) == 8:
            # 兼容 CAN ID 为 0 的固件：实际电机 ID 位于 data[0] 低 4 位。
            derived_id = pkt.data[0] & 0x0F
            act = self._by_can_id.get(int(derived_id))
        if act is None:
            return
        fb = unpack_mit_feedback(pkt.data, limits=act.limits)
        act.update_state(position=fb.position, velocity=fb.velocity, torque=fb.torque)

    # ---------------------------
    # 基础控制命令
    def enable(self, act: Actuator) -> None:
        _log.info("Enabling actuator %r (cmd_id=0x%03X)", act.name, act.command_id)
        self._control_cmd(act, 0xFC)
        time.sleep(0.1)
        self.poll()

    def disable(self, act: Actuator) -> None:
        _log.info("Disabling actuator %r (cmd_id=0x%03X)", act.name, act.command_id)
        self._control_cmd(act, 0xFD)
        time.sleep(0.01)

    def set_zero(self, act: Actuator) -> None:
        # 这是设备侧永久操作；不要把它当作本次遥操的临时起点。
        _log.warning("set_zero called on %r — this permanently resets the angle origin.",
                     act.name)
        self._control_cmd(act, 0xFE)
        time.sleep(0.1)
        self.poll()

    def _control_cmd(self, act: Actuator, cmd: int) -> None:
        # 使能/失能/置零都是 7 个 0xFF 加末字节命令码的专用 8 字节控制帧。
        data = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, cmd & 0xFF])
        self._adapter.send(act.command_id, data)

    # ---------------------------
    # 控制模式与参数读写
    def set_control_mode(self, act: Actuator, mode: ControlMode, *, retries: int = 10, retry_s: float = 0.05) -> bool:
        from .registers import Variable
        rid = int(Variable.CTRL_MODE)
        # 先清除旧缓存，再写模式并轮询应答；否则可能把上一次的同值缓存误判为本次成功。
        act.params.pop(rid, None)
        self.write_param_u32(act, rid, int(mode))

        deadline = time.time() + retries * retry_s
        while time.time() < deadline:
            time.sleep(retry_s)
            self.poll()
            if rid in act.params and int(act.params[rid]) == int(mode):
                _log.info("Control mode confirmed: %s on %r", mode.name, act.name)
                return True
        _log.warning("Mode switch to %s timed out on %r (retries=%d, retry_s=%.3f)",
                     mode.name, act.name, retries, retry_s)
        return False

    def request_state(self, act: Actuator) -> None:
        """请求一帧状态反馈，不等待应答返回。"""

        # 通过 0x7FF 广播封装发送状态请求。载荷前两个字节是目标 command_id 的小端
        # 表示，0xCC 是固件定义的“请求一帧状态”操作码。
        can_id_l = act.command_id & 0xFF
        can_id_h = (act.command_id >> 8) & 0xFF
        data = bytes([can_id_l, can_id_h, 0xCC, 0x00, 0x00, 0x00, 0x00, 0x00])
        self._adapter.send(0x7FF, data)

    def refresh_state(self, act: Actuator) -> None:
        """请求状态，并立即处理当前接收缓冲区中已有的反馈帧。"""

        self.request_state(act)
        self.poll()

    def write_param_u32(self, act: Actuator, rid: int, value: int) -> None:
        # 参数写入同样经 0x7FF 广播封装：目标 ID、0x55 写操作码、RID、四字节值。
        can_id_l = act.command_id & 0xFF
        can_id_h = (act.command_id >> 8) & 0xFF
        data = bytes([can_id_l, can_id_h, 0x55, rid & 0xFF]) + _u32_to_bytes_le(value)
        self._adapter.send(0x7FF, data)

    def write_param_f32(self, act: Actuator, rid: int, value: float) -> None:
        """向电机写入一个 32 位浮点参数。"""
        can_id_l = act.command_id & 0xFF
        can_id_h = (act.command_id >> 8) & 0xFF
        data = bytes([can_id_l, can_id_h, 0x55, rid & 0xFF]) + pack_f32(value)
        self._adapter.send(0x7FF, data)

    def save_params(self, act: Actuator) -> None:
        """将已写参数持久化到电机非易失存储（0xAA 命令）。"""

        # 仅在确实需要跨重启保留参数时调用；频繁写闪存会增加磨损。
        can_id_l = act.command_id & 0xFF
        can_id_h = (act.command_id >> 8) & 0xFF
        self._adapter.send(0x7FF, bytes([can_id_l, can_id_h, 0xAA, 0x00, 0, 0, 0, 0]))

    def apply_limits_and_save(self, act: Actuator, limits: Limits) -> None:
        """写入 MIT 编码范围（PMAX/VMAX/TMAX）并保存到电机闪存。"""
        # 这些范围决定 MIT 报文的整数编码比例，必须与发送端和反馈解析使用的 limits 一致。
        from .registers import Variable
        self.write_param_f32(act, int(Variable.PMAX), limits.pmax)
        self.write_param_f32(act, int(Variable.VMAX), limits.vmax)
        self.write_param_f32(act, int(Variable.TMAX), limits.tmax)
        self.save_params(act)

    def read_param(self, act: Actuator, rid: int, *, timeout_s: float = 0.05) -> Optional[float]:
        """发送参数读取请求；超时返回 ``None``，否则返回应答值。

        发送请求前清除 ``act.params[rid]`` 的旧缓存，避免把历史数据误当作本次应答。
        """
        # 0x33 为参数读取操作码。poll() 收到应答后会将值放入 act.params[rid]。
        can_id_l = act.command_id & 0xFF
        can_id_h = (act.command_id >> 8) & 0xFF
        act.params.pop(int(rid), None)
        self._adapter.send(0x7FF, bytes([can_id_l, can_id_h, 0x33, rid & 0xFF, 0, 0, 0, 0]))
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self.poll()
            if int(rid) in act.params:
                value = act.params[int(rid)]
                _log.debug("read_param rid=%d → %s on %r", rid, value, act.name)
                return value
            time.sleep(0.002)
        _log.warning(
            "read_param timed out (rid=%d, timeout=%.3fs) on %r — "
            "check CAN wiring and motor power",
            rid, timeout_s, act.name,
        )
        return None


    def send_mit(
        self,
        act: Actuator,
        *,
        kp: float,
        kd: float,
        q: float,
        dq: float,
        tau: float,
        poll: bool = True,
    ) -> None:
        # 先在 SDK 边界限制目标位置，避免将超出电机安全范围的 q 编码进 MIT 报文。
        q = act.clamp_position(q)
        payload = pack_mit_command(kp=kp, kd=kd, q=q, dq=dq, tau=tau, limits=act.limits)
        self._adapter.send(act.command_id, payload)
        if poll:
            self.poll()

    def send_pos_vel(self, act: Actuator, *, position: float, velocity: float, poll: bool = True) -> None:
        """PV 模式：发送位置和速度两个 32 位浮点数。

        为兼容原厂固件示例，报文使用 ``CAN ID = 0x100 + slave_id`` 发送。
        """
        position = act.clamp_position(position)
        # PV 模式不使用 command_id 直接发送，而是遵循原厂示例的 0x100 + slave_id 路由。
        can_id = 0x100 + (act.command_id & 0x7FF)
        payload = pack_f32(position) + pack_f32(velocity)
        self._adapter.send(can_id, payload)
        if poll:
            self.poll()
