"""Gloria-M 从端夹爪适配器。"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any

from ..config import GloriaMGripperConfig
from ..interfaces import GripperActuator


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class _ContactState:
    """接触检测的适配器内部状态。"""

    active: bool = False
    candidate_since_s: float | None = None
    candidate_position_rad: float | None = None
    last_feedback_updated_at_s: float | None = None


class GloriaMGripperFollower(GripperActuator):
    """以 MIT 力矩模式驱动 Gloria-M，并在接触后切换到低扭矩保压。"""

    def __init__(self, config: GloriaMGripperConfig):
        if not math.isfinite(config.open_q_rad) or not math.isfinite(
            config.close_q_rad
        ):
            raise ValueError("Gloria-M 开合标定必须是有限弧度值")
        if math.isclose(config.open_q_rad, config.close_q_rad, abs_tol=1e-9):
            raise ValueError("Gloria-M 的 open_q_rad 和 close_q_rad 必须不同")
        self._validate_contact_config(config)
        self.config = config
        self._gripper: Any | None = None
        self._lock = threading.RLock()
        self._contact = _ContactState()

    @staticmethod
    def _validate_contact_config(config: GloriaMGripperConfig) -> None:
        values = (
            config.max_torque_nm,
            config.contact_torque_nm,
            config.contact_stall_duration_s,
            config.contact_position_tolerance_rad,
            config.hold_torque_nm,
            config.contact_release_hysteresis_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Gloria-M 接触控制参数必须是有限数值")
        if config.max_torque_nm <= 0.0:
            raise ValueError("Gloria-M 的 max_torque_nm 必须为正数")
        if not 0.0 < config.contact_torque_nm <= config.max_torque_nm:
            raise ValueError("contact_torque_nm 必须在 0 到 max_torque_nm 之间")
        if not 0.0 <= config.hold_torque_nm <= config.contact_torque_nm:
            raise ValueError("hold_torque_nm 必须在 0 到 contact_torque_nm 之间")
        if config.contact_stall_duration_s <= 0.0:
            raise ValueError("contact_stall_duration_s 必须为正数")
        if config.contact_position_tolerance_rad < 0.0:
            raise ValueError("contact_position_tolerance_rad 不能为负数")
        if config.contact_release_hysteresis_rad < 0.0:
            raise ValueError("contact_release_hysteresis_rad 不能为负数")

    def connect(self) -> None:
        """延迟加载厂商 SDK，连接、检查 KT_Value 并使能 MIT 模式。"""
        with self._lock:
            try:
                from serial.tools import list_ports

                from gloria_m_sdk import (
                    ControlMode,
                    GloriaGripper,
                    Limits,
                    PositionRange,
                    Variable,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Gloria-M 夹爪需要 gloria_m_sdk 与 pyserial"
                ) from exc

            if self.config.port.lower() == "auto":
                ports = list(list_ports.comports())
                if len(ports) != 1:
                    raise RuntimeError(
                        f"夹爪串口自动检测失败（找到 {len(ports)} 个），请在 config.py 指定 port"
                    )
                port = ports[0].device
            else:
                port = self.config.port
            limits = Limits(
                pmax=self.config.position_limit_rad,
                vmax=self.config.velocity_limit_rad_s,
                tmax=self.config.torque_limit_nm,
            )
            safe_position = PositionRange(
                min=min(self.config.open_q_rad, self.config.close_q_rad),
                max=max(self.config.open_q_rad, self.config.close_q_rad),
            )
            gripper = GloriaGripper(
                port,
                baudrate=self.config.baudrate,
                command_id=self.config.command_id,
                feedback_id=self.config.feedback_id,
                limits=limits,
                safe_position=safe_position,
            )
            try:
                gripper.connect()
                self._ensure_kt_value(gripper, Variable)
                gripper.motor.set_mode(ControlMode.MIT)
                gripper.motor.enable()
                gripper.motor.refresh()
                self._gripper = gripper
                self._reset_contact_state()
                print(
                    f"[INFO] Gloria-M 从端夹爪已连接并上使能，当前位置: {gripper.state.position:.3f} rad"
                )
            except Exception:
                try:
                    gripper.disconnect()
                except Exception:
                    pass
                raise

    @staticmethod
    def _ensure_kt_value(gripper: Any, variable: Any) -> None:
        """在未标定 KT_Value 时临时估算，避免 MIT 模式被固件拒绝。"""
        kt_register = int(variable.KT_Value)
        current = gripper.params.read(kt_register, timeout_s=0.2)
        if current is not None and abs(current) > 1e-9:
            return
        npp = gripper.params.read(int(variable.NPP), timeout_s=0.2)
        flux = gripper.params.read(int(variable.Flux), timeout_s=0.2)
        if npp is None or flux is None:
            print("[WARN] 无法读取 NPP/Flux，跳过 KT_Value 估算；MIT 模式可能失败")
            return
        candidate = 1.5 * float(npp) * float(flux)
        gripper.params.write_f32(kt_register, candidate)
        print(
            f"[WARN] 检测到 KT_Value=0，已临时写入估算值 {candidate:.6f} Nm/A（未写入 Flash）"
        )

    def send_normalized(self, opening: float) -> bool:
        """下发开合目标；接触后忽略继续闭合输入并以低扭矩保压。"""
        with self._lock:
            if self._gripper is None:
                return False
            position = self._finite_float(
                getattr(self._gripper.state, "position", None)
            )
            if position is None:
                print("[WARN] Gloria-M 当前位置反馈无效，本帧跳过")
                return False
            fraction = _clamp(float(opening), 0.0, 1.0)
            target = self.config.close_q_rad + fraction * (
                self.config.open_q_rad - self.config.close_q_rad
            )
            close_direction = self._close_direction()

            if self._contact.active:
                if self._opening_requested(target, position, close_direction):
                    self._reset_contact_state()
                    print("[INFO] Gloria-M 收到张开输入，退出接触保压")
                else:
                    return self._send_torque(
                        close_direction * self.config.hold_torque_nm
                    )

            torque = _clamp(
                self.config.stiffness_nm_per_rad * (target - position),
                -self.config.max_torque_nm,
                self.config.max_torque_nm,
            )
            # 接触前也不能以 max_torque_nm 持续顶住硬物；闭合方向单独限幅。
            if close_direction * torque > 0.0:
                torque = close_direction * min(
                    abs(torque), self.config.contact_torque_nm
                )
            if self._detect_contact(position, torque, target, close_direction):
                torque = close_direction * self.config.hold_torque_nm
                print(
                    "[INFO] Gloria-M 检测到接触，忽略继续闭合输入，"
                    f"切换至 {self.config.hold_torque_nm:.3f} Nm 保压"
                )
            return self._send_torque(torque)

    @property
    def contact_active(self) -> bool:
        """当前是否已检测到接触并处于保压状态。"""

        with self._lock:
            return self._contact.active

    @staticmethod
    def _finite_float(value: object) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    def _close_direction(self) -> float:
        return 1.0 if self.config.close_q_rad > self.config.open_q_rad else -1.0

    def _opening_requested(
        self, target: float, position: float, close_direction: float
    ) -> bool:
        return (
            close_direction * (target - position)
            < -self.config.contact_release_hysteresis_rad
        )

    def _detect_contact(
        self,
        position: float,
        torque: float,
        target: float,
        close_direction: float,
    ) -> bool:
        """在高闭合扭矩且新的位置反馈持续停滞时锁定接触。"""

        if not self.config.contact_detection_enabled:
            return False
        closing_requested = close_direction * (target - position) > 0.0
        high_closing_torque = close_direction * torque >= self.config.contact_torque_nm
        if not closing_requested or not high_closing_torque:
            self._contact.candidate_since_s = None
            self._contact.candidate_position_rad = None
            return False

        sample_time_s = self._new_feedback_time()
        if sample_time_s is None:
            return False
        if self._contact.candidate_position_rad is None:
            self._contact.candidate_since_s = sample_time_s
            self._contact.candidate_position_rad = position
            return False

        movement_rad = abs(position - self._contact.candidate_position_rad)
        if movement_rad > self.config.contact_position_tolerance_rad:
            self._contact.candidate_since_s = sample_time_s
            self._contact.candidate_position_rad = position
            return False
        if (
            self._contact.candidate_since_s is None
            or sample_time_s < self._contact.candidate_since_s
        ):
            self._contact.candidate_since_s = sample_time_s
            self._contact.candidate_position_rad = position
            return False
        if (
            sample_time_s - self._contact.candidate_since_s
            < self.config.contact_stall_duration_s
        ):
            return False

        self._contact.active = True
        self._contact.candidate_since_s = None
        self._contact.candidate_position_rad = None
        return True

    def _new_feedback_time(self) -> float | None:
        """返回一帧新反馈的时间；旧反馈不会重复推进接触计时。"""

        if self._gripper is None:
            return None
        updated_at_s = self._finite_float(
            getattr(self._gripper.state, "updated_at", None)
        )
        if updated_at_s is None or updated_at_s <= 0.0:
            return time.monotonic()
        if self._contact.last_feedback_updated_at_s == updated_at_s:
            return None
        self._contact.last_feedback_updated_at_s = updated_at_s
        return updated_at_s

    def _send_torque(self, torque: float) -> bool:
        """通过既有 SDK 的 MIT 接口发送一帧扭矩命令。"""

        try:
            self._gripper.motion.send_mit(
                kp=0.0,
                kd=self.config.damping_nm_s_per_rad,
                q=0.0,
                dq=0.0,
                tau=torque,
                poll=True,
            )
            return True
        except Exception as exc:
            print(f"[WARN] Gloria-M 夹爪控制异常，本帧跳过: {exc}")
            return False

    def _reset_contact_state(self) -> None:
        self._contact = _ContactState()

    def read_normalized_opening(self) -> float | None:
        """读取归一化开合量，并为下一次采样请求一帧新的非阻塞反馈。"""

        with self._lock:
            if self._gripper is None:
                return None
            try:
                # 先消费上一采样周期请求的回包，再请求下一帧；不阻塞遥操控制线程。
                self._gripper.motor.poll()
                position = float(self._gripper.state.position)
                self._gripper.motor.request_state()
            except Exception:
                return None

        if not math.isfinite(position):
            return None
        opening = (position - self.config.close_q_rad) / (
            self.config.open_q_rad - self.config.close_q_rad
        )
        return _clamp(opening, 0.0, 1.0)

    def read_cached_normalized_opening(self) -> float | None:
        """读取已到达的归一化反馈，不执行串口轮询或状态请求。"""

        with self._lock:
            if self._gripper is None:
                return None
            position = self._finite_float(
                getattr(self._gripper.state, "position", None)
            )
        if position is None:
            return None
        opening = (position - self.config.close_q_rad) / (
            self.config.open_q_rad - self.config.close_q_rad
        )
        return _clamp(opening, 0.0, 1.0)

    def disable(self) -> None:
        with self._lock:
            if self._gripper is not None:
                try:
                    self._gripper.motor.disable()
                    print("[INFO] Gloria-M 从端夹爪已失能")
                except Exception as exc:
                    print(f"[WARN] 夹爪失能时出错: {exc}")
            self._reset_contact_state()

    def disconnect(self) -> None:
        with self._lock:
            if self._gripper is not None:
                try:
                    self._gripper.disconnect()
                    print("[INFO] Gloria-M 从端夹爪已断开")
                finally:
                    self._gripper = None
                    self._reset_contact_state()
