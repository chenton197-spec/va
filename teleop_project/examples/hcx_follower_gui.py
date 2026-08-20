#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HCX 双七轴从臂简易 Qt 上位机。

连接参数和左右机器人 ID 来自根目录 teleop.yaml 的 hcx 段。界面仅使用 HCX SDK
的规划关节运动接口，不启动直伺服，也不执行 hcx.auto_* 自动状态操作。

先安装界面专用依赖：

    python -m pip install -r requirements_hcx_follower_gui.txt

运行：

    python -m examples.hcx_follower_gui
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
import time
from typing import Any, Callable, Iterable, Literal

from teleop_sdk.config import HcxConfig, load_runtime_config


ArmSide = Literal["left", "right"]
_ARM_SIDES: tuple[ArmSide, ArmSide] = ("left", "right")
_AXIS_COUNT = 7
_STATUS_REFRESH_INTERVAL_MS = 200
# 单次读取失败可能是瞬时调度或控制器忙；连续失败才视为链路丢失。
_STATUS_FAILURE_THRESHOLD = 3
# 规划式点动不会把目标设在控制器软限位边界，给厂商侧限位检查留出余量。
_JOG_LIMIT_MARGIN_DEG = 1.0


def _load_qt_modules() -> tuple[Any, Any, Any]:
    """仅在实际启动窗口时导入 PySide6，保持后端测试无 GUI 依赖。"""

    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            "缺少 PySide6；请使用当前 Python 执行 "
            "python -m pip install -r requirements_hcx_follower_gui.txt"
        ) from exc
    return QtCore, QtGui, QtWidgets


def _load_robot_client() -> Any:
    """仅在实际连接时导入 HCX 原生 SDK。"""

    from hcx_sdk import RobotClient

    return RobotClient


def _validate_hcx_config(config: HcxConfig) -> None:
    """在加载原生 SDK 前验证上位机需要的连接配置。"""

    for name, value in (
        ("hcx.local_ip", config.local_ip),
        ("hcx.remote_ip", config.remote_ip),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} 不能为空")
    if (
        not isinstance(config.port, int)
        or isinstance(config.port, bool)
        or not 1 <= config.port <= 65535
    ):
        raise ValueError("hcx.port 必须是 1 到 65535 的整数")
    if config.connect_timeout_s is not None:
        if isinstance(config.connect_timeout_s, bool):
            raise ValueError("hcx.connect_timeout_s 必须是正的有限秒数或 null")
        try:
            timeout_s = float(config.connect_timeout_s)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "hcx.connect_timeout_s 必须是正的有限秒数或 null"
            ) from exc
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("hcx.connect_timeout_s 必须是正的有限秒数或 null")
    for name, robot_id in (
        ("hcx.left_robot_id", config.left_robot_id),
        ("hcx.right_robot_id", config.right_robot_id),
    ):
        if not isinstance(robot_id, int) or isinstance(robot_id, bool) or robot_id < 0:
            raise ValueError(f"{name} 必须是非负整数")
    if config.left_robot_id == config.right_robot_id:
        raise ValueError("hcx.left_robot_id 与 hcx.right_robot_id 必须不同")
    if not isinstance(config.ethercat_master_indices, tuple):
        raise ValueError("hcx.ethercat_master_indices 必须是列表")
    if len(set(config.ethercat_master_indices)) != len(config.ethercat_master_indices):
        raise ValueError("hcx.ethercat_master_indices 不能包含重复索引")
    for master_index in config.ethercat_master_indices:
        if (
            not isinstance(master_index, int)
            or isinstance(master_index, bool)
            or master_index not in (0, 1)
        ):
            raise ValueError("hcx.ethercat_master_indices 只能包含 0 或 1")


@dataclass(frozen=True)
class MotionOptions:
    """一次 HCX 规划关节运动的用户可编辑参数。"""

    speed_ratio: float
    acceleration_seconds: float
    deceleration_seconds: float
    smooth: int


@dataclass(frozen=True)
class ArmSnapshot:
    """一侧七轴从臂的只读状态快照。"""

    side: ArmSide
    robot_id: int
    enabled: bool
    protection_enabled: bool
    angles_deg: tuple[float, ...]
    torque_feedback: tuple[int, ...]
    joint_limits_deg: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ControllerSnapshot:
    """上位机展示的控制器和双臂只读状态。"""

    connected: bool
    global_enabled: bool
    active_alarms: tuple[str, ...]
    hmi_detached: bool
    soft_emergency_stop_normal: bool
    ethercat_operational: tuple[tuple[int, bool], ...]
    arms: tuple[ArmSnapshot, ArmSnapshot]


def _controller_enable_block_reason(snapshot: ControllerSnapshot) -> str | None:
    """返回按 HCX 推荐流程执行全局使能时尚未满足的条件。"""

    reasons: list[str] = []
    if not snapshot.connected:
        reasons.append("HCX 未连接")
    if not snapshot.hmi_detached:
        reasons.append("示教器未脱离")
    if snapshot.active_alarms:
        reasons.append("存在报警")
    if not snapshot.soft_emergency_stop_normal:
        reasons.append("软急停未恢复正常")
    if any(
        not operational for _, operational in snapshot.ethercat_operational
    ):
        reasons.append("EtherCAT 未进入 OP")
    return "；".join(reasons) or None


def _arm_motion_block_reason(
    snapshot: ControllerSnapshot, arm: ArmSnapshot
) -> str | None:
    """返回指定单臂开始或恢复规划运动时尚未满足的条件。"""

    reasons: list[str] = []
    controller_reason = _controller_enable_block_reason(snapshot)
    if controller_reason is not None:
        reasons.append(controller_reason)
    if not snapshot.global_enabled:
        reasons.append("全局未使能")
    if not arm.enabled:
        reasons.append(f"{'左' if arm.side == 'left' else '右'}臂未使能")
    return "；".join(reasons) or None


def _parse_target_angles(values: Iterable[object]) -> tuple[float, ...]:
    """解析七个角度输入框，并拒绝空值、非数值和无穷值。"""

    try:
        angles = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("七个目标关节角度均必须是有限数值") from exc
    if len(angles) != _AXIS_COUNT:
        raise ValueError("必须填写七个目标关节角度")
    if not all(math.isfinite(value) for value in angles):
        raise ValueError("七个目标关节角度均必须是有限数值")
    return angles


def _parse_motion_options(
    speed_ratio: object,
    acceleration_seconds: object,
    deceleration_seconds: object,
    smooth: object,
) -> MotionOptions:
    """解析并验证 HCX SDK 支持的规划运动参数范围。"""

    try:
        speed = float(speed_ratio)
        acceleration = float(acceleration_seconds)
        deceleration = float(deceleration_seconds)
        smooth_value = int(str(smooth).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("速度比例、加减速时间和平滑等级必须是有效数值") from exc
    if not math.isfinite(speed) or not 0.0 < speed <= 1.0:
        raise ValueError("速度比例必须在 (0, 1] 范围内")
    if not math.isfinite(acceleration) or not 0.1 <= acceleration <= 1.0:
        raise ValueError("加速时间必须在 0.1 到 1.0 秒之间")
    if not math.isfinite(deceleration) or not 0.1 <= deceleration <= 1.0:
        raise ValueError("减速时间必须在 0.1 到 1.0 秒之间")
    if not 0 <= smooth_value <= 9:
        raise ValueError("平滑等级必须是 0 到 9 的整数")
    return MotionOptions(speed, acceleration, deceleration, smooth_value)


def _build_jog_target(
    current_angles_deg: Iterable[object],
    joint_limits_deg: Iterable[tuple[object, object]],
    axis_index: int,
    direction: int,
    *,
    limit_margin_deg: float = _JOG_LIMIT_MARGIN_DEG,
) -> tuple[float, ...]:
    """生成仅改变一个关节、朝软限位内侧移动的规划式点动目标。"""

    current = _parse_target_angles(current_angles_deg)
    limits = tuple(
        (float(negative), float(positive))
        for negative, positive in joint_limits_deg
    )
    if len(limits) != _AXIS_COUNT:
        raise ValueError(f"预期 {_AXIS_COUNT} 个关节限位，实际为 {len(limits)}")
    if not isinstance(axis_index, int) or not 0 <= axis_index < _AXIS_COUNT:
        raise ValueError(f"关节索引必须是 0 到 {_AXIS_COUNT - 1} 的整数")
    if direction not in (-1, 1):
        raise ValueError("点动方向必须为 -1 或 1")
    if not math.isfinite(limit_margin_deg) or limit_margin_deg < 0.0:
        raise ValueError("点动限位余量必须是非负有限数")

    negative, positive = limits[axis_index]
    if (
        not math.isfinite(negative)
        or not math.isfinite(positive)
        or negative >= positive
    ):
        raise ValueError(f"J{axis_index + 1} 的关节限位无效")
    lower_target = negative + limit_margin_deg
    upper_target = positive - limit_margin_deg
    if lower_target > upper_target:
        raise ValueError(f"J{axis_index + 1} 的点动限位余量过大")

    target_value = upper_target if direction > 0 else lower_target
    current_value = current[axis_index]
    if direction > 0 and current_value >= target_value:
        raise ValueError(f"J{axis_index + 1} 已到达正向点动边界")
    if direction < 0 and current_value <= target_value:
        raise ValueError(f"J{axis_index + 1} 已到达反向点动边界")

    target = list(current)
    target[axis_index] = target_value
    return tuple(target)


class HcxFollowerBackend:
    """将上位机动作限制为 HCX SDK 的公开状态和规划运动接口。"""

    def __init__(
        self,
        config: HcxConfig,
        *,
        robot_client_factory: Callable[[str, str, int], Any] | None = None,
    ) -> None:
        _validate_hcx_config(config)
        self.config = config
        self._robot_client_factory = robot_client_factory
        self._robot: Any | None = None
        self._arms: dict[ArmSide, Any] = {}

    @property
    def connected(self) -> bool:
        return self._robot is not None and bool(self._robot.connected)

    def connect(self) -> ControllerSnapshot:
        """建立唯一 HCX SDK 连接并读取首帧双臂状态。"""

        if self.connected:
            return self.snapshot()
        robot_client_factory = self._robot_client_factory or _load_robot_client()
        robot = robot_client_factory(
            self.config.local_ip, self.config.remote_ip, self.config.port
        )
        try:
            robot.connect(timeout_s=self.config.connect_timeout_s)
            if not robot.connected:
                raise RuntimeError("HCX SDK connect 返回后仍未处于连接状态")
            self._robot = robot
            self._arms = {
                "left": robot.arm(self.config.left_robot_id),
                "right": robot.arm(self.config.right_robot_id),
            }
            return self.snapshot()
        except Exception:
            try:
                robot.close()
            except Exception:
                pass
            self._robot = None
            self._arms = {}
            raise

    def close(self) -> None:
        """关闭 SDK 连接；不将其视为物理急停。"""

        robot, self._robot = self._robot, None
        self._arms = {}
        if robot is not None:
            robot.close()

    def _require_robot(self) -> Any:
        if not self.connected or self._robot is None:
            raise RuntimeError("尚未连接 HCX 控制器")
        return self._robot

    def _arm(self, side: ArmSide) -> Any:
        self._require_robot()
        try:
            return self._arms[side]
        except KeyError as exc:
            raise ValueError(f"未知机械臂侧别: {side}") from exc

    def link_healthy(self) -> bool:
        """读取 HCX 控制器链路；区别于本地 SDK 会话是否已初始化。"""

        return bool(self._require_robot().link_status)

    def _read_arm_snapshot(self, side: ArmSide) -> ArmSnapshot:
        arm = self._arm(side)
        angles = tuple(float(value) for value in arm.joint_angles())
        torque = tuple(int(value) for value in arm.joint_torque_feedback())
        limits = tuple(
            (float(negative), float(positive))
            for negative, positive in arm.joint_limits_deg
        )
        if len(angles) != _AXIS_COUNT or len(torque) != _AXIS_COUNT:
            raise RuntimeError(
                f"HCX {side} 预期 {_AXIS_COUNT} 轴反馈，实际角度/力矩为 "
                f"{len(angles)}/{len(torque)} 轴"
            )
        if len(limits) != _AXIS_COUNT:
            raise RuntimeError(
                f"HCX {side} 预期 {_AXIS_COUNT} 轴限位，实际为 {len(limits)} 轴"
            )
        return ArmSnapshot(
            side=side,
            robot_id=arm.robot_id,
            enabled=bool(arm.enabled),
            protection_enabled=bool(arm.protection_enabled),
            angles_deg=angles,
            torque_feedback=torque,
            joint_limits_deg=limits,
        )

    def snapshot(self) -> ControllerSnapshot:
        """读取控制器与左右七轴状态；不改变任何状态。"""

        robot = self._require_robot()
        if not self.link_healthy():
            raise RuntimeError("HCX 控制器链路状态为 false")
        return ControllerSnapshot(
            connected=bool(robot.connected),
            global_enabled=bool(robot.global_enabled),
            active_alarms=tuple(str(value) for value in robot.active_alarms),
            hmi_detached=bool(robot.hmi_detached),
            soft_emergency_stop_normal=bool(robot.soft_emergency_stop_normal),
            ethercat_operational=tuple(
                (master_index, bool(robot.ethercat_master_operational(master_index)))
                for master_index in self.config.ethercat_master_indices
            ),
            arms=(
                self._read_arm_snapshot("left"),
                self._read_arm_snapshot("right"),
            ),
        )

    def clear_alarms(self) -> None:
        self._require_robot().clear_alarms()

    def detach_hmi(self) -> None:
        self._require_robot().detach_hmi()

    def set_global_enabled(self, enabled: bool) -> None:
        self._require_robot().set_global_enable(enabled)

    def set_arm_enabled(self, side: ArmSide, enabled: bool) -> None:
        self._arm(side).set_enabled(enabled)

    def pause_arm(self, side: ArmSide) -> None:
        self._arm(side).pause()

    def resume_arm(self, side: ArmSide) -> None:
        self._arm(side).resume()

    def clear_arm_route(self, side: ArmSide) -> None:
        self._arm(side).clear_route(emergency_stop=True)

    def move_arm(
        self, side: ArmSide, target_angles_deg: Iterable[object], options: MotionOptions
    ) -> Any:
        """提交单侧规划运动；SDK 再执行连接、报警、使能和限位检查。"""

        target = _parse_target_angles(target_angles_deg)
        arm = self._arm(side)
        limits = tuple(arm.joint_limits_deg)
        for axis_index, (target_angle, limit) in enumerate(zip(target, limits)):
            negative, positive = float(limit[0]), float(limit[1])
            if target_angle < negative or target_angle > positive:
                raise ValueError(
                    f"J{axis_index + 1} 目标 {target_angle:g} 度超出 "
                    f"[{negative:g}, {positive:g}]"
                )
        return arm.move_joints(
            target,
            interrupt=False,
            acceleration_seconds=options.acceleration_seconds,
            deceleration_seconds=options.deceleration_seconds,
            speed_ratio=options.speed_ratio,
            smooth=options.smooth,
            wait=False,
        )

    def jog_axis(
        self,
        side: ArmSide,
        axis_index: int,
        direction: int,
        options: MotionOptions,
    ) -> tuple[Any, tuple[float, ...]]:
        """开始一条单轴规划式点动；调用方须在松开时清路停止。"""

        arm = self._arm(side)
        target = _build_jog_target(
            arm.joint_angles(),
            arm.joint_limits_deg,
            axis_index,
            direction,
        )
        return self.move_arm(side, target, options), target


@dataclass
class _ArmView:
    """一侧 Qt 控件引用。"""

    side: ArmSide
    state_label: Any
    table: Any


class HcxFollowerGui:
    """PySide6/Qt 视图；所有状态修改均由用户按钮触发。"""

    def __init__(self, application: Any, backend: HcxFollowerBackend) -> None:
        self._qt_core, self._qt_gui, self._qt_widgets = _load_qt_modules()
        self._application = application
        self.backend = backend
        self._closed = False
        self._connection_active = False
        self._status_failure_count = 0
        self._arm_views: dict[ArmSide, _ArmView] = {}
        self._active_jogs: dict[ArmSide, tuple[int, int] | None] = {
            side: None for side in _ARM_SIDES
        }
        self._toolbar_buttons: dict[str, Any] = {}
        self._arm_action_buttons: dict[ArmSide, dict[str, Any]] = {
            side: {} for side in _ARM_SIDES
        }
        self._jog_buttons: dict[ArmSide, list[Any]] = {
            side: [] for side in _ARM_SIDES
        }

        self._window = self._qt_widgets.QMainWindow()
        self._window.setWindowTitle("HCX 从臂上位机")
        self._window.setMinimumSize(1040, 680)
        self._window.resize(1280, 820)
        self._build_layout()
        self._set_disconnected_controls()

        self._refresh_timer = self._qt_core.QTimer(self._window)
        self._refresh_timer.timeout.connect(self._poll_status)
        self._refresh_timer.start(_STATUS_REFRESH_INTERVAL_MS)
        self._application.aboutToQuit.connect(self.close)
        self._application.applicationStateChanged.connect(
            self._on_application_state_changed
        )

    def show(self) -> None:
        self._window.show()

    def close(self) -> None:
        """停止状态轮询并关闭 SDK 连接；不发送物理急停。"""

        if self._closed:
            return
        self._stop_all_jogs()
        self._closed = True
        self._refresh_timer.stop()
        try:
            self.backend.close()
        except Exception as exc:
            print(f"[WARN] 关闭 HCX 连接失败: {exc}", file=sys.stderr)

    def _build_layout(self) -> None:
        widgets = self._qt_widgets
        central = widgets.QWidget()
        self._window.setCentralWidget(central)
        root_layout = widgets.QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        toolbar = widgets.QHBoxLayout()
        for action, label, handler in (
            ("connect", "连接", self._connect),
            ("disconnect", "断开", self._disconnect),
            ("refresh", "刷新", self._refresh_now),
            ("clear_alarms", "清除报警", self._clear_alarms),
            ("detach_hmi", "脱离示教器", self._detach_hmi),
            ("global_enable", "全局使能", lambda: self._set_global_enabled(True)),
            ("global_disable", "全局失能", lambda: self._set_global_enabled(False)),
        ):
            button = widgets.QPushButton(label)
            button.clicked.connect(handler)
            self._toolbar_buttons[action] = button
            toolbar.addWidget(button)
        toolbar.addStretch(1)
        root_layout.addLayout(toolbar)

        self._controller_label = widgets.QLabel("未连接")
        self._controller_label.setWordWrap(True)
        self._alarm_label = widgets.QLabel("报警: --")
        self._alarm_label.setWordWrap(True)
        root_layout.addWidget(self._controller_label)
        root_layout.addWidget(self._alarm_label)
        root_layout.addWidget(self._build_motion_options())

        arms_layout = widgets.QHBoxLayout()
        arms_layout.addWidget(self._build_arm_view("left"))
        arms_layout.addWidget(self._build_arm_view("right"))
        root_layout.addLayout(arms_layout, 1)

        self._log_text = widgets.QPlainTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumBlockCount(400)
        self._log_text.setMinimumHeight(130)
        root_layout.addWidget(self._log_text)

    def _build_motion_options(self) -> Any:
        widgets = self._qt_widgets
        group = widgets.QGroupBox("规划运动参数")
        layout = widgets.QHBoxLayout(group)

        self._speed_input = widgets.QDoubleSpinBox()
        self._speed_input.setRange(0.01, 1.0)
        self._speed_input.setDecimals(2)
        self._speed_input.setSingleStep(0.01)
        self._speed_input.setValue(0.10)

        self._acceleration_input = widgets.QDoubleSpinBox()
        self._acceleration_input.setRange(0.1, 1.0)
        self._acceleration_input.setDecimals(2)
        self._acceleration_input.setSingleStep(0.1)
        self._acceleration_input.setValue(0.50)

        self._deceleration_input = widgets.QDoubleSpinBox()
        self._deceleration_input.setRange(0.1, 1.0)
        self._deceleration_input.setDecimals(2)
        self._deceleration_input.setSingleStep(0.1)
        self._deceleration_input.setValue(0.50)

        self._smooth_input = widgets.QSpinBox()
        self._smooth_input.setRange(0, 9)
        self._smooth_input.setValue(1)

        for label, input_widget in (
            ("速度比例", self._speed_input),
            ("加速时间 (s)", self._acceleration_input),
            ("减速时间 (s)", self._deceleration_input),
            ("平滑等级", self._smooth_input),
        ):
            layout.addWidget(widgets.QLabel(label))
            layout.addWidget(input_widget)
        layout.addStretch(1)
        return group

    def _build_arm_view(self, side: ArmSide) -> Any:
        widgets = self._qt_widgets
        robot_id = (
            self.backend.config.left_robot_id
            if side == "left"
            else self.backend.config.right_robot_id
        )
        title = f"{'左' if side == 'left' else '右'}臂 (robot_id={robot_id})"
        group = widgets.QGroupBox(title)
        layout = widgets.QVBoxLayout(group)

        state_label = widgets.QLabel("状态: --")
        layout.addWidget(state_label)

        table = widgets.QTableWidget(_AXIS_COUNT, 7)
        table.setHorizontalHeaderLabels(
            ("轴", "角度 (度)", "原始力矩", "限位 (度)", "目标 (度)", "反转", "正转")
        )
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(
            widgets.QHeaderView.ResizeMode.Stretch
        )
        for axis_index in range(_AXIS_COUNT):
            self._set_table_item(table, axis_index, 0, f"J{axis_index + 1}", False)
            self._set_table_item(table, axis_index, 1, "--", False)
            self._set_table_item(table, axis_index, 2, "--", False)
            self._set_table_item(table, axis_index, 3, "--", False)
            self._set_table_item(table, axis_index, 4, "", True)
            table.setCellWidget(
                axis_index, 5, self._build_jog_button(side, axis_index, -1)
            )
            table.setCellWidget(
                axis_index, 6, self._build_jog_button(side, axis_index, 1)
            )
        layout.addWidget(table)

        controls = widgets.QGridLayout()
        for column, (action, label, handler) in enumerate(
            (
                ("fill_current", "当前位置填入", lambda: self._fill_current(side)),
                ("move", "规划运动", lambda: self._move_arm(side)),
                ("enable", "单臂使能", lambda: self._set_arm_enabled(side, True)),
                ("disable", "单臂失能", lambda: self._set_arm_enabled(side, False)),
                ("pause", "暂停", lambda: self._pause_arm(side)),
                ("resume", "恢复", lambda: self._resume_arm(side)),
                ("clear_route", "清路", lambda: self._clear_arm_route(side)),
            )
        ):
            button = widgets.QPushButton(label)
            button.clicked.connect(handler)
            if action == "enable":
                button.setToolTip("使用单机器人使能期间，请勿切换示教器三挡开关")
            self._arm_action_buttons[side][action] = button
            controls.addWidget(button, 0, column)
        layout.addLayout(controls)

        self._arm_views[side] = _ArmView(side, state_label, table)
        return group

    def _build_jog_button(
        self, side: ArmSide, axis_index: int, direction: int
    ) -> Any:
        button = self._qt_widgets.QPushButton("-" if direction < 0 else "+")
        button.setToolTip(
            f"按住使 J{axis_index + 1}{'反转' if direction < 0 else '正转'}；"
            "松开立即停止"
        )
        button.setMinimumWidth(34)
        button.pressed.connect(
            lambda: self._start_jog(side, axis_index, direction)
        )
        button.released.connect(
            lambda: self._stop_jog(side, axis_index, direction)
        )
        self._jog_buttons[side].append(button)
        return button

    def _set_button_available(
        self, button: Any, available: bool, blocked_reason: str = ""
    ) -> None:
        """更新按钮状态，并保留其启用时的原始提示文本。"""

        normal_tooltip = button.property("hcx_normal_tooltip")
        if normal_tooltip is None:
            normal_tooltip = button.toolTip()
            button.setProperty("hcx_normal_tooltip", normal_tooltip)
        button.setEnabled(available)
        button.setToolTip(
            str(normal_tooltip)
            if available
            else f"不可用：{blocked_reason or '当前状态不满足要求'}"
        )

    def _set_disconnected_controls(self) -> None:
        for action, button in self._toolbar_buttons.items():
            self._set_button_available(
                button,
                action == "connect",
                "HCX 未连接" if action != "connect" else "",
            )
        for side in _ARM_SIDES:
            for button in self._arm_action_buttons[side].values():
                self._set_button_available(button, False, "HCX 未连接")
            for button in self._jog_buttons[side]:
                self._set_button_available(button, False, "HCX 未连接")

    def _apply_control_availability(self, snapshot: ControllerSnapshot) -> None:
        """按 SDK 推荐启动顺序更新每个控制按钮的可用性。"""

        if not snapshot.connected:
            self._set_disconnected_controls()
            return

        enable_reason = _controller_enable_block_reason(snapshot)
        self._set_button_available(
            self._toolbar_buttons["connect"], False, "HCX 已连接"
        )
        self._set_button_available(self._toolbar_buttons["disconnect"], True)
        self._set_button_available(self._toolbar_buttons["refresh"], True)
        self._set_button_available(
            self._toolbar_buttons["clear_alarms"],
            bool(snapshot.active_alarms),
            "当前无报警",
        )
        detach_reason = (
            "已脱离示教器"
            if snapshot.hmi_detached
            else "请先全局失能"
            if snapshot.global_enabled
            else ""
        )
        self._set_button_available(
            self._toolbar_buttons["detach_hmi"],
            not snapshot.hmi_detached and not snapshot.global_enabled,
            detach_reason,
        )
        self._set_button_available(
            self._toolbar_buttons["global_enable"],
            not snapshot.global_enabled and enable_reason is None,
            "全局已使能" if snapshot.global_enabled else enable_reason or "",
        )
        self._set_button_available(
            self._toolbar_buttons["global_disable"],
            snapshot.global_enabled,
            "全局已失能",
        )

        for arm in snapshot.arms:
            controls = self._arm_action_buttons[arm.side]
            motion_reason = _arm_motion_block_reason(snapshot, arm)
            arm_enable_reason = (
                "单臂已使能"
                if arm.enabled
                else "全局未使能"
                if not snapshot.global_enabled
                else enable_reason or ""
            )
            self._set_button_available(controls["fill_current"], True)
            self._set_button_available(
                controls["move"], motion_reason is None, motion_reason or ""
            )
            self._set_button_available(
                controls["enable"],
                not arm.enabled
                and snapshot.global_enabled
                and enable_reason is None,
                arm_enable_reason,
            )
            self._set_button_available(
                controls["disable"], arm.enabled, "单臂已失能"
            )
            self._set_button_available(
                controls["pause"], motion_reason is None, motion_reason or ""
            )
            self._set_button_available(
                controls["resume"], motion_reason is None, motion_reason or ""
            )
            # 清路是撤销当前规划路径的操作，在链路仍正常时保持可用。
            self._set_button_available(controls["clear_route"], True)
            for button in self._jog_buttons[arm.side]:
                self._set_button_available(
                    button, motion_reason is None, motion_reason or ""
                )

    def _set_disconnected_ui(self, reason: str) -> None:
        self._connection_active = False
        self._status_failure_count = 0
        self._set_disconnected_controls()
        self._controller_label.setText(f"连接=否  原因={reason}")
        self._alarm_label.setText("报警: --")
        for view in self._arm_views.values():
            view.state_label.setText("状态: 连接已断开")
            for axis_index in range(_AXIS_COUNT):
                for column in (1, 2, 3):
                    self._set_table_item(view.table, axis_index, column, "--", False)

    def _handle_connection_lost(self, reason: str) -> None:
        """失去链路后撤销本地控制权限，不尝试在失效链路上继续下发命令。"""

        if not self._connection_active and not self.backend.connected:
            return
        for side in _ARM_SIDES:
            self._active_jogs[side] = None
        try:
            self.backend.close()
        except Exception as exc:
            self._log(f"关闭已失效 HCX 会话失败: {exc}")
        self._set_disconnected_ui(reason)
        self._log(f"HCX 连接已断开: {reason}")

    def _record_status_failure(self, exc: Exception) -> None:
        self._status_failure_count += 1
        if self._status_failure_count >= _STATUS_FAILURE_THRESHOLD:
            self._handle_connection_lost(
                f"连续 {_STATUS_FAILURE_THRESHOLD} 次状态读取失败: {exc}"
            )
            return
        self._log(
            "状态轮询失败 "
            f"({self._status_failure_count}/{_STATUS_FAILURE_THRESHOLD}): {exc}"
        )

    def _set_table_item(
        self, table: Any, row: int, column: int, text: str, editable: bool
    ) -> None:
        item = table.item(row, column)
        if item is None:
            item = self._qt_widgets.QTableWidgetItem()
            table.setItem(row, column, item)
        if not editable:
            item.setFlags(
                item.flags() & ~self._qt_core.Qt.ItemFlag.ItemIsEditable
            )
        item.setText(text)

    def _target_angles(self, side: ArmSide) -> tuple[str, ...]:
        table = self._arm_views[side].table
        return tuple(
            table.item(axis_index, 4).text()
            for axis_index in range(_AXIS_COUNT)
        )

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self._log_text.appendPlainText(f"[{timestamp}] {message}")
        scrollbar = self._log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _show_error(self, action: str, exc: Exception) -> None:
        message = f"{action}失败: {exc}"
        self._log(message)
        self._qt_widgets.QMessageBox.critical(
            self._window, "HCX 上位机", message
        )

    def _require_confirmation(self, title: str, message: str) -> bool:
        buttons = (
            self._qt_widgets.QMessageBox.StandardButton.Yes
            | self._qt_widgets.QMessageBox.StandardButton.No
        )
        selected = self._qt_widgets.QMessageBox.question(
            self._window,
            title,
            message,
            buttons,
            self._qt_widgets.QMessageBox.StandardButton.No,
        )
        return selected == self._qt_widgets.QMessageBox.StandardButton.Yes

    def _connect(self) -> None:
        self._set_button_available(
            self._toolbar_buttons["connect"], False, "正在连接"
        )
        try:
            snapshot = self.backend.connect()
        except Exception as exc:
            self._set_disconnected_controls()
            self._show_error("连接 HCX", exc)
            return
        self._connection_active = True
        self._status_failure_count = 0
        self._apply_snapshot(snapshot)
        self._log("HCX 已连接")

    def _disconnect(self) -> None:
        self._stop_all_jogs()
        try:
            self.backend.close()
        except Exception as exc:
            self._show_error("断开 HCX", exc)
            return
        self._set_disconnected_ui("用户主动断开")
        self._log("HCX 已断开")

    def _refresh_now(self) -> None:
        if not self.backend.connected:
            if self._connection_active:
                self._handle_connection_lost("HCX SDK 会话已关闭")
            self._log("尚未连接 HCX")
            return
        try:
            if not self.backend.link_healthy():
                self._handle_connection_lost("HCX 控制器链路状态为 false")
                return
            self._apply_snapshot(self.backend.snapshot())
        except Exception as exc:
            self._record_status_failure(exc)

    def _poll_status(self) -> None:
        if not self._connection_active:
            return
        if not self.backend.connected:
            self._handle_connection_lost("HCX SDK 会话已关闭")
            return
        try:
            if not self.backend.link_healthy():
                self._handle_connection_lost("HCX 控制器链路状态为 false")
                return
            self._apply_snapshot(self.backend.snapshot())
        except Exception as exc:
            self._record_status_failure(exc)
            return
        self._status_failure_count = 0

    def _on_application_state_changed(self, state: Any) -> None:
        if state != self._qt_core.Qt.ApplicationState.ApplicationActive:
            self._stop_all_jogs()

    def _apply_snapshot(self, snapshot: ControllerSnapshot) -> None:
        ethercat_text = (
            "，".join(
                f"ECAT{master_index}={'OP' if operational else '非 OP'}"
                for master_index, operational in snapshot.ethercat_operational
            )
            or "ECAT: 未配置"
        )
        self._controller_label.setText(
            "连接={}  全局使能={}  软急停正常={}  示教器脱离={}  {}".format(
                "是" if snapshot.connected else "否",
                "是" if snapshot.global_enabled else "否",
                "是" if snapshot.soft_emergency_stop_normal else "否",
                "是" if snapshot.hmi_detached else "否",
                ethercat_text,
            )
        )
        self._alarm_label.setText(
            "报警: " + ("；".join(snapshot.active_alarms) or "无")
        )
        for arm_snapshot in snapshot.arms:
            view = self._arm_views[arm_snapshot.side]
            view.state_label.setText(
                "单臂使能={}  防护={}".format(
                    "是" if arm_snapshot.enabled else "否",
                    "是" if arm_snapshot.protection_enabled else "否",
                )
            )
            for axis_index in range(_AXIS_COUNT):
                self._set_table_item(
                    view.table,
                    axis_index,
                    1,
                    f"{arm_snapshot.angles_deg[axis_index]:.3f}",
                    False,
                )
                self._set_table_item(
                    view.table,
                    axis_index,
                    2,
                    str(arm_snapshot.torque_feedback[axis_index]),
                    False,
                )
                negative, positive = arm_snapshot.joint_limits_deg[axis_index]
                self._set_table_item(
                    view.table,
                    axis_index,
                    3,
                    f"{negative:g} .. {positive:g}",
                    False,
                )
        self._apply_control_availability(snapshot)

    def _fill_current(self, side: ArmSide) -> None:
        try:
            snapshot = self.backend.snapshot()
            arm_snapshot = next(arm for arm in snapshot.arms if arm.side == side)
        except Exception as exc:
            self._show_error("读取当前位置", exc)
            return
        self._apply_snapshot(snapshot)
        for axis_index, value in enumerate(arm_snapshot.angles_deg):
            self._set_table_item(
                self._arm_views[side].table,
                axis_index,
                4,
                f"{value:.3f}",
                True,
            )
        self._log(f"{side.upper()} 当前角度已填入目标")

    def _motion_options(self) -> MotionOptions:
        return _parse_motion_options(
            self._speed_input.value(),
            self._acceleration_input.value(),
            self._deceleration_input.value(),
            self._smooth_input.value(),
        )

    def _move_arm(self, side: ArmSide) -> None:
        try:
            target = _parse_target_angles(self._target_angles(side))
            options = self._motion_options()
        except ValueError as exc:
            self._show_error("解析规划运动参数", exc)
            return
        target_text = ", ".join(f"{value:.2f}" for value in target)
        if not self._require_confirmation(
            "确认规划运动",
            f"{side.upper()} 将以速度比例 {options.speed_ratio:g} 规划运动到：\n"
            f"[{target_text}]",
        ):
            return
        try:
            motion = self.backend.move_arm(side, target, options)
        except Exception as exc:
            self._show_error(f"{side.upper()} 规划运动", exc)
            return
        self._log(
            f"{side.upper()} 已提交规划运动，sequence={getattr(motion, 'sequence', '--')}"
        )

    def _start_jog(self, side: ArmSide, axis_index: int, direction: int) -> None:
        active = self._active_jogs[side]
        if active is not None:
            self._log(f"{side.upper()} 已有点动正在执行，请先松开当前按钮")
            return
        try:
            motion, target = self.backend.jog_axis(
                side, axis_index, direction, self._motion_options()
            )
        except ValueError as exc:
            self._log(f"{side.upper()} J{axis_index + 1} 无法开始点动: {exc}")
            return
        except Exception as exc:
            self._show_error(f"{side.upper()} J{axis_index + 1} 点动", exc)
            return
        self._active_jogs[side] = (axis_index, direction)
        direction_text = "正转" if direction > 0 else "反转"
        self._log(
            f"{side.upper()} J{axis_index + 1} {direction_text}已开始，"
            f"目标={target[axis_index]:.2f}，"
            f"sequence={getattr(motion, 'sequence', '--')}"
        )

    def _stop_jog(self, side: ArmSide, axis_index: int, direction: int) -> None:
        if self._active_jogs[side] != (axis_index, direction):
            return
        self._active_jogs[side] = None
        try:
            self.backend.clear_arm_route(side)
        except Exception as exc:
            self._log(
                f"{side.upper()} J{axis_index + 1} 松开后清路停止失败: {exc}"
            )
            return
        self._log(f"{side.upper()} J{axis_index + 1} 已松开并清路停止")

    def _stop_all_jogs(self) -> None:
        for side, active in tuple(self._active_jogs.items()):
            if active is None:
                continue
            self._active_jogs[side] = None
            try:
                self.backend.clear_arm_route(side)
            except Exception as exc:
                print(
                    f"[WARN] {side.upper()} 点动停止清路失败: {exc}",
                    file=sys.stderr,
                )

    def _clear_alarms(self) -> None:
        if not self._require_confirmation(
            "确认清除报警", "确认已排除报警原因后再请求清除报警？"
        ):
            return
        try:
            self.backend.clear_alarms()
            self._apply_snapshot(self.backend.snapshot())
        except Exception as exc:
            self._show_error("清除报警", exc)
            return
        self._log("已请求清除报警")

    def _detach_hmi(self) -> None:
        if not self._require_confirmation(
            "确认脱离示教器", "确认现场未连接示教器且已按规程物理拔除？"
        ):
            return
        try:
            self.backend.detach_hmi()
            self._apply_snapshot(self.backend.snapshot())
        except Exception as exc:
            self._show_error("脱离示教器", exc)
            return
        self._log("已请求脱离示教器")

    def _set_global_enabled(self, enabled: bool) -> None:
        action = "全局使能" if enabled else "全局失能"
        if not self._require_confirmation(f"确认{action}", f"确认执行{action}？"):
            return
        try:
            self.backend.set_global_enabled(enabled)
            self._apply_snapshot(self.backend.snapshot())
        except Exception as exc:
            self._show_error(action, exc)
            return
        self._log(f"已请求{action}")

    def _set_arm_enabled(self, side: ArmSide, enabled: bool) -> None:
        action = "单臂使能" if enabled else "单臂失能"
        if not self._require_confirmation(
            f"确认{action}", f"确认对 {side.upper()} 执行{action}？"
        ):
            return
        try:
            self.backend.set_arm_enabled(side, enabled)
            self._apply_snapshot(self.backend.snapshot())
        except Exception as exc:
            self._show_error(f"{side.upper()} {action}", exc)
            return
        self._log(f"{side.upper()} 已请求{action}")

    def _pause_arm(self, side: ArmSide) -> None:
        if not self._require_confirmation(
            "确认暂停", f"确认对 {side.upper()} 请求减速暂停？"
        ):
            return
        try:
            self.backend.pause_arm(side)
            self._apply_snapshot(self.backend.snapshot())
        except Exception as exc:
            self._show_error(f"{side.upper()} 暂停", exc)
            return
        self._log(f"{side.upper()} 已请求暂停")

    def _resume_arm(self, side: ArmSide) -> None:
        if not self._require_confirmation(
            "确认恢复", f"确认恢复 {side.upper()} 当前已暂停的运动？"
        ):
            return
        try:
            self.backend.resume_arm(side)
            self._apply_snapshot(self.backend.snapshot())
        except Exception as exc:
            self._show_error(f"{side.upper()} 恢复", exc)
            return
        self._log(f"{side.upper()} 已请求恢复")

    def _clear_arm_route(self, side: ArmSide) -> None:
        self._active_jogs[side] = None
        try:
            self.backend.clear_arm_route(side)
            self._apply_snapshot(self.backend.snapshot())
        except Exception as exc:
            self._show_error(f"{side.upper()} 清路", exc)
            return
        self._log(f"{side.upper()} 已请求清路")


def _configure_application_font(application: Any, qt_gui: Any) -> None:
    """请求常见中文字体；Qt 找不到时会自动使用系统回退字体。"""

    application.setFont(qt_gui.QFont("Noto Sans CJK SC", 10))


def main() -> int:
    """启动 HCX Qt 上位机窗口。"""

    try:
        backend = HcxFollowerBackend(load_runtime_config().hcx)
    except (TypeError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] HCX 上位机配置无效: {exc}", file=sys.stderr)
        return 2
    try:
        _, qt_gui, qt_widgets = _load_qt_modules()
        application = qt_widgets.QApplication.instance()
        if application is None:
            application = qt_widgets.QApplication(sys.argv)
        _configure_application_font(application, qt_gui)
        gui = HcxFollowerGui(application, backend)
    except RuntimeError as exc:
        print(f"[ERROR] 无法启动 HCX Qt 上位机: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ERROR] 无法创建 HCX Qt 窗口: {exc}", file=sys.stderr)
        return 1
    gui.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
