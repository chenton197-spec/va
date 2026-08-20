"""遥操作控制参数。

默认值与项目原有 ``alicia_teleop_fr3.py`` 保持一致，单位均为角度制。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class TeleopConfig:
    """控制器配置；所有关节位置、速度和加速度均以度为单位。"""

    rate_hz: float = 125.0  # 控制频率（Hz），FR3 ServoJ 最高 125 Hz
    max_step_deg: float = 30.0  # 每周期最大关节变化量（度），30 度约等于不限速
    jump_threshold_deg: float = 180.0  # 跳变检测阈值（度），超过视为传感器毛刺并丢帧
    dead_zone_deg: float = 0.005  # 死区（度），变化小于此值不下发从臂命令

    filter_enabled: bool = True  # 是否启用关节 OneEuro 与固定低通两级滤波
    filter_mincutoff_hz: float = 3.0  # OneEuro 最低截止频率（Hz），越小越稳但越慢，建议 1-5
    filter_beta: float = 0.05  # OneEuro 速度自适应系数，越大运动时越跟手，建议 0.02-0.1
    tremor_cutoff_hz: float = 5.0  # 第二级固定低通截止频率（Hz），截断手部震颤，建议 2-5

    spring_enabled: bool = True  # 是否启用关节弹簧阻尼及前瞻预测
    spring_omega: float = 20.0  # 弹簧阻尼固有频率（rad/s），越大越跟手，建议 8-20
    max_accel_deg_s2: float = 800.0  # 关节最大加速度（度/s^2），抑制突发加速，建议 200-800
    max_vel_deg_s: float = 150.0  # 关节最大速度（度/s），安全兜底，建议 60-150
    predict_lookahead_ms: float = 10.0  # 前瞻预测时长（毫秒），过大可能导致超调

    # 以下字段留空时由传入的从臂自动提供：限位来自从臂，映射为同轴恒等映射。
    # 部署入口可用 dataclasses.replace 注入其所属硬件映射。
    min_angles_deg: tuple[float, ...] | None = None  # YAML 可覆盖的从臂最小安全角度（度）
    max_angles_deg: tuple[float, ...] | None = None  # YAML 可覆盖的从臂最大安全角度（度）
    axis_order: tuple[int, ...] | None = None  # YAML 可覆盖的主臂到从臂轴映射
    axis_sign: tuple[float, ...] | None = None  # 入口可覆盖的轴方向，+1 同向、-1 反向

    relative_mode: bool = True  # True 为相对模式，False 为绝对模式（需主臂已完成零位校准）
    # 仅用于离线或短时诊断。它观测的是目标提交，不读取从臂实际反馈；实时遥操
    # 默认关闭，避免高频终端输出扰动控制周期。
    latency_probe_enabled: bool = False
    latency_probe_threshold_deg: float = 2.0  # 延迟埋点触发和响应阈值（度）
    latency_probe_quiescent_deg: float = 0.3  # 延迟埋点静止判定阈值（度）


@dataclass(frozen=True)
class AliciaLeaderConfig:
    """Alicia-D 示教臂连接配置。"""

    port: str = ""  # Alicia-D 串口端口，留空时由 SDK 自动查找
    gripper_type: str = "50mm"  # Alicia-D 夹爪型号
    connect_retries: int = 5  # 示教臂连接失败时的最大重试次数
    connect_retry_delay_s: float = 3.0  # 每次连接重试之间的等待时间（秒）


@dataclass(frozen=True)
class OpenArmMiniLeaderConfig:
    """双侧 OpenArm Mini 示教臂的连接和独立标定配置。"""

    port_left: str = "/dev/ttyACM1"
    port_right: str = "/dev/ttyACM0"
    calibration_path: str = ""  # 创建或更新的 left/right 组合标定 JSON
    baudrate: int = 1_000_000


@dataclass(frozen=True)
class FR3FollowerConfig:
    """真实 FAIRINO FR3 从臂连接配置。"""

    robot_ip: str = "192.168.57.3"  # 当前 FR3 机械臂 IP 地址
    # Alicia-D -> FR3 的安装方向。该映射由 FR3 参考入口注入通用控制器。
    axis_sign: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, -1.0, -1.0)


@dataclass(frozen=True)
class HcxConfig:
    """HCX 双臂控制器、机器人 ID 与安全启动配置。"""

    local_ip: str = ""  # 本机与 HCX 控制器通信使用的网卡地址
    remote_ip: str = ""  # HCX 控制器地址
    port: int = 12345
    connect_timeout_s: float | None = 10.0
    left_robot_id: int = 2
    right_robot_id: int = 1
    # OpenArm Mini -> HCX 双臂遥操的每侧关节方向；+1 同向，-1 反向。
    left_axis_sign: tuple[float, ...] = (
        1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
    )
    right_axis_sign: tuple[float, ...] = (
        -1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
    )
    # HCX PluseToServo 的 Python 输出参数。危险确认默认关闭，必须由现场人员显式授权。
    direct_servo_rate_hz: int = 125
    direct_servo_watchdog_s: float = 0.2
    direct_servo_confirm_unsafe: bool = False
    # direct: Python 输出线程重发当前目标；linear: Python 线性重采样；limited:
    # HCX 适配器按 direct_servo_rate_hz 独立低通、限速、限加速度并逐点输出。
    direct_servo_interpolation: Literal["direct", "linear", "limited"] = "direct"
    # 仅在 limited 模式下使用，单位分别为度/秒、度/秒^2；低通系数范围为 [0, 1]。
    direct_servo_limited_max_vel_deg_s: float = 20.0
    direct_servo_limited_max_accel_deg_s2: float = 80.0
    direct_servo_limited_lowpass_alpha: float = 0.25

    # 所有会改变控制器状态的动作默认关闭，必须由现场人员在 YAML 中显式授权。
    auto_detach_hmi: bool = False
    auto_clear_alarms: bool = False
    auto_enable: bool = False

    controller_initialization_wait_s: float = 2.0
    ethercat_master_indices: tuple[int, ...] = ()
    ethercat_op_timeout_s: float = 15.0
    alarm_clear_retry_count: int = 5
    alarm_clear_retry_interval_s: float = 1.0
    global_enable_retry_count: int = 5
    global_enable_retry_interval_s: float = 1.0
    single_arm_enable_timeout_s: float = 5.0
    enable_status_poll_interval_s: float = 0.1


@dataclass(frozen=True)
class GloriaMGripperConfig:
    """Gloria-M 从端夹爪配置；夹爪不可用不影响六轴遥操。"""

    enabled: bool = True  # 真实遥操入口是否连接 Gloria-M 夹爪
    port: str = "auto"  # 串口；auto 时要求仅发现一个可用串口
    baudrate: int = 921_600  # CAN 串口波特率
    command_id: int = 0x01  # 发送 CAN ID
    feedback_id: int = 0x101  # 接收 CAN ID
    open_q_rad: float = 2.5  # 夹爪全开位置（弧度，需按实际机构标定）
    close_q_rad: float = 0.0  # 夹爪全合位置（弧度，需按实际机构标定）
    max_torque_nm: float = 0.75  # MIT 模式最大输出扭矩（Nm）
    stiffness_nm_per_rad: float = 6.0  # 虚拟弹簧刚度（Nm/rad）
    damping_nm_s_per_rad: float = 0.15  # MIT 阻尼参数（Nm*s/rad）
    position_limit_rad: float = 3.14  # MIT 协议位置编码范围
    velocity_limit_rad_s: float = 10.0  # MIT 协议速度编码范围
    torque_limit_nm: float = 6.0  # MIT 协议扭矩编码范围

    contact_detection_enabled: bool = True  # 接触后锁定闭合目标并进入保压
    contact_torque_nm: float = 0.50  # 接触前的最大闭合扭矩，也是接触判定门限（Nm）
    contact_stall_duration_s: float = 0.12  # 高扭矩且位置不动的最短持续时间（秒）
    contact_position_tolerance_rad: float = 0.01  # 接触判定中的位置停滞容差（弧度）
    hold_torque_nm: float = 0.20  # 接触后的固定闭合保压扭矩（Nm）
    contact_release_hysteresis_rad: float = 0.05  # 张开多少才退出接触保压（弧度）


@dataclass(frozen=True)
class GloriaMDualGripperConfig:
    """OpenArm Mini 双侧示教夹爪到 Gloria-M 双侧从端夹爪的部署配置。"""

    # 两侧各自独立读取和下发，避免一侧串口操作阻塞另一侧。
    rate_hz: float = 30.0
    leader_read_timeout_s: float = 0.05
    status_print_interval_s: float = 1.0
    # 两只夹爪共用的目标跟随、接触检测与闭合保压参数。
    stiffness_nm_per_rad: float = 1.0
    damping_nm_s_per_rad: float = 0.1
    contact_detection_enabled: bool = True
    contact_torque_nm: float = 0.50
    contact_stall_duration_s: float = 0.12
    contact_position_tolerance_rad: float = 0.01
    hold_torque_nm: float = 0.20
    contact_release_hysteresis_rad: float = 0.05
    # 默认不连接任何真实夹爪。双侧端口和 enabled 需在部署 YAML 中显式确认。
    left: GloriaMGripperConfig = field(
        default_factory=lambda: GloriaMGripperConfig(enabled=False, port="")
    )
    right: GloriaMGripperConfig = field(
        default_factory=lambda: GloriaMGripperConfig(enabled=False, port="")
    )

    def side_config(self, side: Literal["left", "right"]) -> GloriaMGripperConfig:
        """返回一侧独立硬件参数和公共夹爪控制参数合并后的适配器配置。"""

        if side == "left":
            config = self.left
        elif side == "right":
            config = self.right
        else:
            raise ValueError("Gloria-M side 必须为 'left' 或 'right'")
        return replace(
            config,
            stiffness_nm_per_rad=self.stiffness_nm_per_rad,
            damping_nm_s_per_rad=self.damping_nm_s_per_rad,
            contact_detection_enabled=self.contact_detection_enabled,
            contact_torque_nm=self.contact_torque_nm,
            contact_stall_duration_s=self.contact_stall_duration_s,
            contact_position_tolerance_rad=self.contact_position_tolerance_rad,
            hold_torque_nm=self.hold_torque_nm,
            contact_release_hysteresis_rad=self.contact_release_hysteresis_rad,
        )


# 三个测试入口共用的集中配置。需要修改参数时，只修改本文件。
TELEOP_CONFIG = TeleopConfig()
ALICIA_LEADER_CONFIG = AliciaLeaderConfig()
OPENARM_MINI_LEADER_CONFIG = OpenArmMiniLeaderConfig()
FR3_FOLLOWER_CONFIG = FR3FollowerConfig()
HCX_CONFIG = HcxConfig()
GLORIA_M_GRIPPER_CONFIG = GloriaMGripperConfig()
GLORIA_M_DUAL_GRIPPER_CONFIG = GloriaMDualGripperConfig()


@dataclass(frozen=True)
class RuntimeConfig:
    """代码默认值与 YAML 覆盖合并后的运行时配置。"""

    teleop: TeleopConfig = TELEOP_CONFIG
    alicia: AliciaLeaderConfig = ALICIA_LEADER_CONFIG
    openarm_mini: OpenArmMiniLeaderConfig = OPENARM_MINI_LEADER_CONFIG
    fr3: FR3FollowerConfig = FR3_FOLLOWER_CONFIG
    hcx: HcxConfig = HCX_CONFIG
    gloria_m: GloriaMGripperConfig = GLORIA_M_GRIPPER_CONFIG
    gloria_m_dual: GloriaMDualGripperConfig = GLORIA_M_DUAL_GRIPPER_CONFIG


DEFAULT_YAML_PATH = Path(__file__).resolve().parents[1] / "teleop.yaml"
# Reference-deployment sections parsed by their owning hardware packages.
EXTERNAL_DEPLOYMENT_SECTIONS = frozenset(
    {"orbbec", "recording", "hcx_orbbec", "hcx_recording"}
)


def _merge_dataclass(instance: Any, overrides: dict[str, Any], section: str) -> Any:
    """合并受白名单约束的 YAML 字段，并将关节列表转换为不可变元组。"""
    allowed = {field.name for field in fields(instance)}
    unknown = set(overrides) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"YAML 的 {section} 包含未知或禁止覆盖字段: {names}")
    values = dict(overrides)
    for name in (
        "min_angles_deg",
        "max_angles_deg",
        "axis_order",
        "axis_sign",
        "ethercat_master_indices",
        "left_axis_sign",
        "right_axis_sign",
    ):
        if name in values and values[name] is not None:
            if not isinstance(values[name], list):
                raise ValueError(f"YAML 的 {section}.{name} 必须是列表")
            values[name] = tuple(values[name])
    return replace(instance, **values)


def _merge_gloria_m_dual_config(
    instance: GloriaMDualGripperConfig,
    overrides: dict[str, Any],
    section: str,
) -> GloriaMDualGripperConfig:
    """合并双侧夹爪配置，并继续对白名单字段做严格校验。"""

    allowed = {field.name for field in fields(instance)}
    unknown = set(overrides) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"YAML 的 {section} 包含未知或禁止覆盖字段: {names}")

    values = dict(overrides)
    shared_control_fields = {
        "stiffness_nm_per_rad",
        "damping_nm_s_per_rad",
        "contact_detection_enabled",
        "contact_torque_nm",
        "contact_stall_duration_s",
        "contact_position_tolerance_rad",
        "hold_torque_nm",
        "contact_release_hysteresis_rad",
    }
    for side in ("left", "right"):
        if side not in values:
            continue
        side_overrides = values[side]
        if not isinstance(side_overrides, dict):
            raise ValueError(f"YAML 的 {section}.{side} 必须是映射对象")
        misplaced = set(side_overrides) & shared_control_fields
        if misplaced:
            names = ", ".join(sorted(misplaced))
            raise ValueError(
                f"YAML 的 {section}.{side} 不可单独设置公共夹爪控制参数: {names}；"
                f"请改在 {section} 下配置"
            )
        values[side] = _merge_dataclass(
            getattr(instance, side), side_overrides, f"{section}.{side}"
        )
    return replace(instance, **values)


def load_runtime_config(path: str | Path = DEFAULT_YAML_PATH) -> RuntimeConfig:
    """加载 YAML 部署覆盖；文件不存在时完整使用代码默认值。"""
    config_path = Path(path)
    defaults = RuntimeConfig()
    if not config_path.exists():
        return defaults
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("读取 teleop.yaml 需要安装 PyYAML") from exc
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("teleop.yaml 根节点必须是映射对象")
    sections = {
        "teleop": defaults.teleop,
        "alicia": defaults.alicia,
        "openarm_mini": defaults.openarm_mini,
        "fr3": defaults.fr3,
        "hcx": defaults.hcx,
        "gloria_m": defaults.gloria_m,
        "gloria_m_dual": defaults.gloria_m_dual,
    }
    unknown_sections = set(data) - set(sections) - EXTERNAL_DEPLOYMENT_SECTIONS
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ValueError(f"teleop.yaml 包含未知配置段: {names}")
    raw_overrides: dict[str, dict[str, Any]] = {}
    for name, default in sections.items():
        overrides = data.get(name, {})
        if not isinstance(overrides, dict):
            raise ValueError(f"teleop.yaml 的 {name} 必须是映射对象")
        raw_overrides[name] = dict(overrides)

    # ``teleop.axis_sign`` 曾只服务 Alicia-D -> FR3 参考部署。保留一次
    # 明确迁移，避免旧配置静默影响其他从臂；控制器的程序接口仍支持 axis_sign。
    missing = object()
    legacy_axis_sign = raw_overrides["teleop"].pop("axis_sign", missing)
    if legacy_axis_sign is not missing:
        if "axis_sign" in raw_overrides["fr3"]:
            raise ValueError("teleop.axis_sign 与 fr3.axis_sign 不能同时配置")
        raw_overrides["fr3"]["axis_sign"] = legacy_axis_sign

    merged: dict[str, Any] = {}
    for name, default in sections.items():
        if name == "gloria_m_dual":
            merged[name] = _merge_gloria_m_dual_config(
                default, raw_overrides[name], name
            )
        else:
            merged[name] = _merge_dataclass(default, raw_overrides[name], name)
    return RuntimeConfig(**merged)
