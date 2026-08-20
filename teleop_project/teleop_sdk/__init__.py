"""通用机械臂主从遥操作 SDK。

公共关节角度单位统一为度。设备相关实现位于 ``teleop_sdk.adapters``，
控制算法位于 ``teleop_sdk.controller``。
"""

from .algorithms import LinearInterpolator, SpringDamper
from .config import (
    GloriaMDualGripperConfig,
    GloriaMGripperConfig,
    HcxConfig,
    RuntimeConfig,
    TeleopConfig,
    load_runtime_config,
)
from .controller import TeleopController
from .interfaces import (
    FollowerArm,
    GripperActuator,
    LeaderArm,
    LeaderArmWithGripper,
    LeaderGripperInput,
)
from .visualization import start_visualization

__all__ = [
    "FollowerArm",
    "GloriaMDualGripperConfig",
    "GloriaMGripperConfig",
    "GripperActuator",
    "HcxConfig",
    "LeaderArm",
    "LeaderArmWithGripper",
    "LeaderGripperInput",
    "LinearInterpolator",
    "RuntimeConfig",
    "SpringDamper",
    "TeleopConfig",
    "TeleopController",
    "load_runtime_config",
    "start_visualization",
]
