"""主臂与从臂的通用接口定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LeaderArm(ABC):
    """示教臂接口。

    所有实现均应从厂商原始数据转换后返回关节角度，单位固定为度；
    读取失败或超时时返回 ``None``，由控制器在下一控制周期重试。
    """

    @property
    @abstractmethod
    def joint_count(self) -> int:
        """示教臂固定关节数量。"""

    @abstractmethod
    def connect(self) -> None:
        """建立与示教臂的连接。"""

    @abstractmethod
    def read_joint_angles_deg(self, timeout_s: float) -> np.ndarray | None:
        """读取当前关节角度（度）。"""

    @abstractmethod
    def disconnect(self) -> None:
        """断开与示教臂的连接。"""


class LeaderGripperInput(ABC):
    """可选示教端夹爪输入接口，开合值统一为 0（闭合）到 1（张开）。"""

    @abstractmethod
    def read_gripper_opening(self, timeout_s: float) -> float | None:
        """读取归一化夹爪开合量；读取失败或超时时返回 ``None``。"""


class LeaderArmWithGripper(LeaderArm, LeaderGripperInput):
    """支持从同一状态帧读取关节角度和夹爪输入的示教臂接口。"""

    @abstractmethod
    def read_joint_angles_and_gripper_opening(
        self, timeout_s: float
    ) -> tuple[np.ndarray, float] | None:
        """返回同一时刻的关节角度（度）和归一化夹爪开合量。"""


class FollowerArm(ABC):
    """遥操从臂接口。

    从臂实现负责厂商 SDK 的单位、伺服模式和错误码处理；控制器只传递
    角度制关节目标及单周期命令时长。
    """

    @property
    @abstractmethod
    def joint_count(self) -> int:
        """从臂固定关节数量。"""

    @property
    @abstractmethod
    def joint_limits_deg(self) -> tuple[np.ndarray, np.ndarray]:
        """从臂默认安全关节限位，返回 ``(min_angles, max_angles)``（度）。"""

    @abstractmethod
    def connect(self) -> None:
        """建立与从臂的连接。"""

    @abstractmethod
    def read_joint_angles_deg(self) -> np.ndarray:
        """读取当前从臂关节角度（度）。"""

    @abstractmethod
    def start_servo(self) -> bool:
        """进入厂商伺服模式，成功时返回 ``True``。"""

    @abstractmethod
    def send_joint_angles_deg(self, angles_deg: np.ndarray, command_time_s: float) -> bool:
        """发送一帧关节目标（度），成功时返回 ``True``。"""

    def refresh_servo_target(self) -> bool:
        """检查或刷新已接受的伺服目标，供有独立会话的从臂实现覆盖。

        普通规划或逐帧伺服从臂不需要额外动作，默认成功返回。高频 Python
        输出线程可在此方法中仅报告会话健康状态，也可按其自身规则刷新保存的
        目标；控制器不会传入新的关节值，也不会识别具体硬件类型。
        """

        return True

    @property
    def requires_per_cycle_target_updates(self) -> bool:
        """是否需要控制器在死区内也提交每个控制周期的目标。

        默认从臂继续使用死区抑制。需要每个低频输入样本的从臂可覆盖此属性，
        而带独立 Python 输出线程的从臂可以保持默认值。
        """

        return False

    @abstractmethod
    def recover(self) -> bool:
        """执行厂商定义的伺服恢复流程，成功时返回 ``True``。"""

    @abstractmethod
    def stop_servo(self) -> None:
        """停止伺服及当前运动。"""

    @abstractmethod
    def disconnect(self) -> None:
        """断开与从臂的连接。"""


class GripperActuator(ABC):
    """可选从端夹爪接口，开合目标统一为 0（闭合）到 1（张开）。"""

    @abstractmethod
    def connect(self) -> None:
        """连接并使能夹爪。"""

    @abstractmethod
    def send_normalized(self, opening: float) -> bool:
        """发送归一化开合目标，成功时返回 ``True``。"""

    @abstractmethod
    def disable(self) -> None:
        """使夹爪失能。"""

    @abstractmethod
    def disconnect(self) -> None:
        """断开夹爪连接。"""
