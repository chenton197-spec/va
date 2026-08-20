"""HCX 关节控制 SDK 抛出的异常。"""


class HcxSdkError(RuntimeError):
    """SDK 异常的基类。"""


class ConnectionStateError(HcxSdkError):
    """控制器连接不可用时抛出。"""


class AlarmActiveError(HcxSdkError):
    """控制器存在活动报警而阻止命令执行时抛出。"""


class MotionRejectedError(HcxSdkError):
    """控制器拒绝规划关节运动时抛出。"""


class MotionTimeoutError(HcxSdkError):
    """未在规定时间内收到运动完成回调时抛出。"""


class JointLimitError(HcxSdkError):
    """请求的关节角度超出配置限位时抛出。"""


class SafetyConfirmationError(HcxSdkError):
    """未显式确认危险操作时抛出。"""


class DirectServoFault(HcxSdkError):
    """Python 直伺服会话或单次厂商调用进入故障状态时抛出。"""
