"""按左右臂目标关节角度数组分两段提交规划运动。

第一段固定 J1 的当前角度，只移动其余关节；读取左右臂关节反馈确认第一段目标
到位后，第二段再提交完整目标，使 J1 到达目标角度。
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from hcx_sdk import Arm, HcxSdkError, MotionHandle, MotionRejectedError, RobotClient

# 本机用于接收控制器通信数据的 IP 地址。
LOCAL_IP = "172.16.0.111"
# 华成控制器的 IP 地址。
REMOTE_IP = "172.16.0.89"
# 华成控制器通信端口，范围为 1 到 65535。
PORT = 12345

# 左臂在控制器项目中配置的机器人 ID，不是固定的左右臂编号。
LEFT_ARM_ID: int = 1
# 右臂在控制器项目中配置的机器人 ID，必须与 LEFT_ARM_ID 不同。
RIGHT_ARM_ID: int = 2

# 左臂目标关节角度数组，单位为度，索引从 0 开始；数值为绝对目标，None 保持当前角度。[90, -90, 0.0, 0.0, 0.0, 0.0, 0.0]
LEFT_TARGET_ANGLES_DEG: list[float | None] = [90, -90, -90.0, 0.0, 0.0, 0.0, 0.0]
# 右臂目标关节角度数组，单位为度，索引从 0 开始；数值为绝对目标，None 保持当前角度。[-90, -90, 0.0, 0.0, 0.0, 0.0, 0.0]
RIGHT_TARGET_ANGLES_DEG: list[float | None] = [-90, -90, 90.0, 0.0, 0.0, 0.0, 0.0]
# 当前反馈与目标角度相差不超过该值时视为已到位，不发送该轴的运动请求，单位为度。
ANGLE_TOLERANCE_DEG = 0.01
# J1 在七轴目标数组中的索引。第一段保持该轴，第二段才移动该轴。
J1_AXIS_INDEX = 0

# 规划运动的加速时间，单位为秒，允许范围为 0.1 到 1.0。
ACCELERATION_SECONDS = 0.5
# 规划运动的减速时间，单位为秒，允许范围为 0.1 到 1.0。
DECELERATION_SECONDS = 0.5
# 唯一的运动速度比例，范围为 (0, 1]；0.5 表示控制器允许速度的 50%。
MOTION_SPEED_RATIO = 0.1
# 控制器关节规划平滑等级，整数范围为 0 到 9。
SMOOTH = 1
# 每段运动等待实际关节反馈到达目标的最长时间，单位为秒。
# 不使用 moveJoints2 的完成回调作为阶段切换条件。
FEEDBACK_CONFIRM_TIMEOUT_S = 30.0
# 读取实际关节反馈的轮询间隔，单位为秒。
FEEDBACK_CONFIRM_POLL_INTERVAL_S = 0.05

# 仅在现场安全确认完成后设为 True；True 时脚本会改变使能状态并发送运动。
CONFIRM_MOTION = True

# 为 True 时自动开启控制器全局使能和左右单臂使能；不会修改防护状态。
AUTO_ENABLE = True

# 初始化通信后、执行使能前等待控制器稳定的时长，单位为秒。
CONTROLLER_INITIALIZATION_WAIT_S = 2.0

# 仅在现场未连接示教器且会物理拔除时设为 True；True 时调用 detachHMI()。
AUTO_DETACH_HMI_IF_NO_TEACH_PENDANT = True

# 仅在已排除报警原因、确认允许复位时设为 True；True 时调用 clearAlarm()。
AUTO_CLEAR_ALARMS = True
# 自动清报警的最大重试次数；总请求次数为初始请求加该重试次数。
ALARM_CLEAR_RETRY_COUNT = 5
# 自动清报警重试之间的等待时长，单位为秒。
ALARM_CLEAR_RETRY_INTERVAL_S = 1.0

# EtherCAT 主站索引元组，仅 EtherCAT 伺服填写 0 或 1；非 EtherCAT 保持为空元组。
ETHERCAT_MASTER_INDICES: tuple[int, ...] = ()
# 等待每个已配置 EtherCAT 主站进入 OP 状态的最长时间，单位为秒。
ETHERCAT_OP_TIMEOUT_S = 15.0

# 全局使能的最大重试次数；总请求次数为初始请求加该重试次数。
GLOBAL_ENABLE_RETRY_COUNT = 5
# 全局使能重试之间的等待时长，单位为秒。
GLOBAL_ENABLE_RETRY_INTERVAL_S = 1.0
# 单臂使能请求后等待使能反馈变为真的最长时间，单位为秒。
SINGLE_ARM_ENABLE_TIMEOUT_S = 5.0
# 轮询单臂使能反馈状态的时间间隔，单位为秒。
ENABLE_STATUS_POLL_INTERVAL_S = 0.1


@dataclass(frozen=True)
class JointTargetPlan:
    """单条机械臂的当前关节反馈和完整目标关节角度。"""

    arm_name: str
    arm: Arm
    current_angles_deg: tuple[float, ...]
    target_angles_deg: tuple[float, ...]
    changed_axis_indices: tuple[int, ...]


def _ethercat_master_indices() -> tuple[int, ...]:
    """验证并返回配置的 EtherCAT 主站索引。"""

    if not isinstance(ETHERCAT_MASTER_INDICES, tuple):
        raise ValueError("ETHERCAT_MASTER_INDICES 必须是元组")
    if len(set(ETHERCAT_MASTER_INDICES)) != len(ETHERCAT_MASTER_INDICES):
        raise ValueError("ETHERCAT_MASTER_INDICES 不能包含重复索引")
    for master_index in ETHERCAT_MASTER_INDICES:
        if (
            not isinstance(master_index, int)
            or isinstance(master_index, bool)
            or not 0 <= master_index <= 1
        ):
            raise ValueError("ETHERCAT_MASTER_INDICES 中的索引必须是 0 或 1")
    return ETHERCAT_MASTER_INDICES


def _read_enable_preflight(robot: RobotClient) -> dict[str, object]:
    """读取可能阻止全局使能的控制器状态，不发送任何控制命令。"""

    master_status = {
        f"主站 {master_index}": robot.ethercat_master_operational(master_index)
        for master_index in _ethercat_master_indices()
    }
    return {
        "活动报警": list(robot.active_alarms),
        "软急停正常": robot.soft_emergency_stop_normal,
        "示教器已脱离": robot.hmi_detached,
        "EtherCAT 主站 OP 状态": master_status
        or "未配置检测（非 EtherCAT 时保持为空）",
    }


def _wait_for_ethercat_operational(robot: RobotClient, master_index: int) -> None:
    """等待一个已配置的 EtherCAT 主站进入 OP，超时则阻止使能。"""

    if not math.isfinite(ETHERCAT_OP_TIMEOUT_S) or ETHERCAT_OP_TIMEOUT_S <= 0:
        raise ValueError("ETHERCAT_OP_TIMEOUT_S 必须为正的有限秒数")
    deadline = time.monotonic() + ETHERCAT_OP_TIMEOUT_S
    while not robot.ethercat_master_operational(master_index):
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise MotionRejectedError(
                f"EtherCAT 主站 {master_index} 在 {ETHERCAT_OP_TIMEOUT_S} 秒内未进入 OP 状态"
            )
        time.sleep(min(GLOBAL_ENABLE_RETRY_INTERVAL_S, remaining_s))


def _clear_active_alarms_if_configured(robot: RobotClient) -> bool:
    """在显式开启后按有限次数请求清除报警，并确认报警反馈已经消失。"""

    if not isinstance(AUTO_CLEAR_ALARMS, bool):
        raise ValueError("AUTO_CLEAR_ALARMS 必须为布尔值")
    if not AUTO_CLEAR_ALARMS:
        return False
    if not isinstance(ALARM_CLEAR_RETRY_COUNT, int) or ALARM_CLEAR_RETRY_COUNT < 0:
        raise ValueError("ALARM_CLEAR_RETRY_COUNT 必须为非负整数")
    if (
        not math.isfinite(ALARM_CLEAR_RETRY_INTERVAL_S)
        or ALARM_CLEAR_RETRY_INTERVAL_S <= 0
    ):
        raise ValueError("ALARM_CLEAR_RETRY_INTERVAL_S 必须为正的有限秒数")

    alarms = tuple(robot.active_alarms)
    if not alarms:
        return False

    for attempt in range(ALARM_CLEAR_RETRY_COUNT + 1):
        try:
            robot.clear_alarms()
        except MotionRejectedError as exc:
            raise MotionRejectedError("控制器拒绝自动清除报警请求") from exc

        alarms = tuple(robot.active_alarms)
        if not alarms:
            print("自动清除报警成功。")
            return True
        if attempt < ALARM_CLEAR_RETRY_COUNT:
            print(f"报警仍存在，{ALARM_CLEAR_RETRY_INTERVAL_S} 秒后再次请求清除。")
            time.sleep(ALARM_CLEAR_RETRY_INTERVAL_S)

    raise MotionRejectedError(
        "控制器活动报警在初始清除请求后重试 "
        f"{ALARM_CLEAR_RETRY_COUNT} 次仍未消失：" + "；".join(alarms)
    )


def _prepare_controller_for_enable(robot: RobotClient) -> None:
    """按说明书执行只读前置检查，并在明确配置时请求脱离示教器。"""

    if (
        not math.isfinite(CONTROLLER_INITIALIZATION_WAIT_S)
        or CONTROLLER_INITIALIZATION_WAIT_S < 0
    ):
        raise ValueError("CONTROLLER_INITIALIZATION_WAIT_S 必须是非负有限秒数")
    time.sleep(CONTROLLER_INITIALIZATION_WAIT_S)

    if AUTO_DETACH_HMI_IF_NO_TEACH_PENDANT:
        robot.detach_hmi()

    preflight = _read_enable_preflight(robot)
    print("使能前诊断：")
    print(json.dumps(preflight, ensure_ascii=False, indent=2))

    if _clear_active_alarms_if_configured(robot):
        print("报警清除后诊断：")
        print(json.dumps(_read_enable_preflight(robot), ensure_ascii=False, indent=2))

    alarms = tuple(robot.active_alarms)
    if alarms:
        raise MotionRejectedError(
            "控制器存在活动报警；请先排除报警原因，并在现场确认后将 "
            "AUTO_CLEAR_ALARMS 设为 True：" + "；".join(alarms)
        )
    if not robot.soft_emergency_stop_normal:
        raise MotionRejectedError("控制器处于软急停状态；请先在现场恢复急停状态")
    if AUTO_DETACH_HMI_IF_NO_TEACH_PENDANT and not robot.hmi_detached:
        raise MotionRejectedError(
            "已请求脱离示教器但状态仍未生效；请确认示教器已物理拔除后重试"
        )
    for master_index in _ethercat_master_indices():
        _wait_for_ethercat_operational(robot, master_index)


def _global_enable_failure_detail(robot: RobotClient) -> str:
    """汇总全局使能失败后的只读状态，给出下一步现场排查方向。"""

    details: list[str] = []
    alarms = tuple(robot.active_alarms)
    if alarms:
        details.append("活动报警：" + "；".join(alarms))
    if not robot.soft_emergency_stop_normal:
        details.append("软急停未恢复")
    if not robot.hmi_detached:
        details.append(
            "示教器未脱离（若现场未接示教器，请将 "
            "AUTO_DETACH_HMI_IF_NO_TEACH_PENDANT 设为 True，并物理拔除示教器）"
        )
    for master_index in _ethercat_master_indices():
        if not robot.ethercat_master_operational(master_index):
            details.append(f"EtherCAT 主站 {master_index} 未进入 OP")
    if not details:
        details.append("前置状态未报告异常；请检查控制柜物理急停、安全回路和使能条件")
    return "；".join(details)


def _wait_for_enabled_state(state: Callable[[], bool], description: str) -> None:
    """在单臂使能请求后等待状态反馈，超时则阻止后续运动。"""

    if (
        not math.isfinite(SINGLE_ARM_ENABLE_TIMEOUT_S)
        or SINGLE_ARM_ENABLE_TIMEOUT_S <= 0
    ):
        raise ValueError("SINGLE_ARM_ENABLE_TIMEOUT_S 必须为正的有限秒数")
    if (
        not math.isfinite(ENABLE_STATUS_POLL_INTERVAL_S)
        or ENABLE_STATUS_POLL_INTERVAL_S <= 0
    ):
        raise ValueError("ENABLE_STATUS_POLL_INTERVAL_S 必须为正的有限秒数")

    deadline = time.monotonic() + SINGLE_ARM_ENABLE_TIMEOUT_S
    while not state():
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise MotionRejectedError(
                f"{description} 在 {SINGLE_ARM_ENABLE_TIMEOUT_S} 秒内未生效"
            )
        time.sleep(min(ENABLE_STATUS_POLL_INTERVAL_S, remaining_s))


def _enable_global_robot(robot: RobotClient) -> bool:
    """按华成说明书的重试节奏请求全局使能并等待反馈变为真。"""

    if not isinstance(GLOBAL_ENABLE_RETRY_COUNT, int) or GLOBAL_ENABLE_RETRY_COUNT < 0:
        raise ValueError("GLOBAL_ENABLE_RETRY_COUNT 必须为非负整数")
    if (
        not math.isfinite(GLOBAL_ENABLE_RETRY_INTERVAL_S)
        or GLOBAL_ENABLE_RETRY_INTERVAL_S <= 0
    ):
        raise ValueError("GLOBAL_ENABLE_RETRY_INTERVAL_S 必须为正的有限秒数")
    if robot.global_enabled:
        return False

    last_error: MotionRejectedError | None = None
    for attempt in range(GLOBAL_ENABLE_RETRY_COUNT + 1):
        try:
            robot.set_global_enable(True)
        except MotionRejectedError as exc:
            last_error = exc
        if robot.global_enabled:
            return True
        if attempt < GLOBAL_ENABLE_RETRY_COUNT:
            time.sleep(GLOBAL_ENABLE_RETRY_INTERVAL_S)

    message = (
        f"控制器全局使能在初始请求后重试 {GLOBAL_ENABLE_RETRY_COUNT} 次仍未报告已使能"
    )
    message += "；诊断：" + _global_enable_failure_detail(robot)
    if last_error is not None:
        raise MotionRejectedError(message) from last_error
    raise MotionRejectedError(message)


def _ensure_motion_enabled(
    robot: RobotClient, arms: tuple[tuple[str, Arm], ...]
) -> list[str]:
    """仅对未使能的全局状态和单臂请求使能，并等待状态反馈。"""

    changes: list[str] = []
    if _enable_global_robot(robot):
        changes.append("控制器全局使能")
    for arm_name, arm in arms:
        if not arm.enabled:
            arm.set_enabled(True)
            _wait_for_enabled_state(lambda arm=arm: arm.enabled, f"{arm_name} 单臂使能")
            changes.append(f"{arm_name} 单臂使能")
    return changes


def _require_motion_enabled(
    robot: RobotClient, arms: tuple[tuple[str, Arm], ...]
) -> None:
    """确认全局和每条单臂均已使能，否则阻止后续运动。"""

    if not robot.global_enabled:
        raise MotionRejectedError("控制器全局使能未开启")
    for arm_name, arm in arms:
        if not arm.enabled:
            raise MotionRejectedError(f"{arm_name} 未使能")


def _resolve_target_angles(
    current_angles_deg: tuple[float, ...],
    configured_angles_deg: list[float | None],
    configuration_name: str,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """合并目标数组和当前反馈，并验证数组长度及数值有效性。"""

    if not isinstance(configured_angles_deg, list):
        raise TypeError(f"{configuration_name} 必须是列表")
    if len(configured_angles_deg) != len(current_angles_deg):
        raise ValueError(
            f"{configuration_name} 需要 {len(current_angles_deg)} 个元素，"
            f"当前为 {len(configured_angles_deg)} 个"
        )

    target_angles_deg: list[float] = []
    changed_axis_indices: list[int] = []
    for axis_index, (current_angle, configured_angle) in enumerate(
        zip(current_angles_deg, configured_angles_deg)
    ):
        if configured_angle is None:
            target_angles_deg.append(current_angle)
            continue
        if isinstance(configured_angle, bool):
            raise TypeError(f"{configuration_name}[{axis_index}] 必须是数值或 None")
        try:
            target_angle = float(configured_angle)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{configuration_name}[{axis_index}] 必须是数值或 None"
            ) from exc
        if not math.isfinite(target_angle):
            raise ValueError(f"{configuration_name}[{axis_index}] 必须是有限数值")
        target_angles_deg.append(target_angle)
        if not math.isclose(
            current_angle, target_angle, abs_tol=ANGLE_TOLERANCE_DEG
        ):
            changed_axis_indices.append(axis_index)

    return tuple(target_angles_deg), tuple(changed_axis_indices)


def _build_plan(
    arm_name: str,
    arm: Arm,
    configured_angles_deg: list[float | None],
    configuration_name: str,
) -> JointTargetPlan:
    """读取关节反馈并依据该机械臂的目标数组生成完整运动计划。"""

    current_angles_deg = arm.joint_angles()
    target_angles_deg, changed_axis_indices = _resolve_target_angles(
        current_angles_deg, configured_angles_deg, configuration_name
    )
    return JointTargetPlan(
        arm_name=arm_name,
        arm=arm,
        current_angles_deg=current_angles_deg,
        target_angles_deg=target_angles_deg,
        changed_axis_indices=changed_axis_indices,
    )


def _build_non_j1_plan(plan: JointTargetPlan) -> JointTargetPlan:
    """构造第一段计划：保持当前 J1，只移动目标中其余需要变化的关节。"""

    if not 0 <= J1_AXIS_INDEX < len(plan.current_angles_deg):
        raise ValueError("J1_AXIS_INDEX 超出当前机械臂关节数组范围")
    first_stage_target = list(plan.target_angles_deg)
    first_stage_target[J1_AXIS_INDEX] = plan.current_angles_deg[J1_AXIS_INDEX]
    return JointTargetPlan(
        arm_name=plan.arm_name,
        arm=plan.arm,
        current_angles_deg=plan.current_angles_deg,
        target_angles_deg=tuple(first_stage_target),
        changed_axis_indices=tuple(
            axis_index
            for axis_index in plan.changed_axis_indices
            if axis_index != J1_AXIS_INDEX
        ),
    )


def _build_j1_only_plan(
    full_plan: JointTargetPlan, first_stage_plan: JointTargetPlan
) -> JointTargetPlan:
    """构造第二段计划：第一段完成后只让 J1 与完整目标不同。"""

    if full_plan.arm is not first_stage_plan.arm:
        raise ValueError("两段计划必须属于同一条机械臂")
    if full_plan.arm_name != first_stage_plan.arm_name:
        raise ValueError("两段计划的机械臂名称必须一致")
    if len(full_plan.target_angles_deg) != len(first_stage_plan.target_angles_deg):
        raise ValueError("两段计划的关节数量必须一致")
    if not 0 <= J1_AXIS_INDEX < len(full_plan.target_angles_deg):
        raise ValueError("J1_AXIS_INDEX 超出当前机械臂关节数组范围")

    return JointTargetPlan(
        arm_name=full_plan.arm_name,
        arm=full_plan.arm,
        # 第一段的目标就是第二段开始时其他六轴应保持的位置。
        current_angles_deg=first_stage_plan.target_angles_deg,
        target_angles_deg=full_plan.target_angles_deg,
        changed_axis_indices=(
            (J1_AXIS_INDEX,)
            if J1_AXIS_INDEX in full_plan.changed_axis_indices
            else ()
        ),
    )


def _describe_plan(plan: JointTargetPlan) -> dict[str, object]:
    """将内部运动计划转换为可打印的字典。"""

    return {
        "机械臂": plan.arm_name,
        "机器人 ID": plan.arm.robot_id,
        "将运动的轴索引": list(plan.changed_axis_indices),
        "当前关节角度（度）": list(plan.current_angles_deg),
        "目标关节角度（度）": list(plan.target_angles_deg),
    }


def _submit_plan(plan: JointTargetPlan) -> MotionHandle | None:
    """提交单条计划但不等待完成；全部目标轴已到位时不发送命令。"""

    if not plan.changed_axis_indices:
        return None

    return plan.arm.move_joints(
        plan.target_angles_deg,
        interrupt=False,
        acceleration_seconds=ACCELERATION_SECONDS,
        deceleration_seconds=DECELERATION_SECONDS,
        speed_ratio=MOTION_SPEED_RATIO,
        smooth=SMOOTH,
        wait=False,
    )


def _target_feedback_errors(
    plan: JointTargetPlan, feedback_angles_deg: tuple[float, ...]
) -> list[dict[str, float | int]]:
    """返回计划中尚未由实际反馈确认到位的目标轴。"""

    if len(feedback_angles_deg) != len(plan.target_angles_deg):
        raise ValueError(
            f"{plan.arm_name} 反馈关节数为 {len(feedback_angles_deg)}，"
            f"但目标关节数为 {len(plan.target_angles_deg)}"
        )

    errors: list[dict[str, float | int]] = []
    for axis_index in plan.changed_axis_indices:
        feedback_angle = feedback_angles_deg[axis_index]
        target_angle = plan.target_angles_deg[axis_index]
        error_deg = feedback_angle - target_angle
        if not math.isclose(feedback_angle, target_angle, abs_tol=ANGLE_TOLERANCE_DEG):
            errors.append(
                {
                    "轴索引": axis_index,
                    "反馈角度（度）": feedback_angle,
                    "目标角度（度）": target_angle,
                    "误差（度）": error_deg,
                }
            )
    return errors


def _confirm_submitted_plans_by_feedback(
    submitted_plans: list[tuple[JointTargetPlan, MotionHandle | None]],
) -> list[dict[str, object]]:
    """以实际关节反馈确认每条已提交计划的目标轴到位。

    ``moveJoints2`` 回调仅代表厂商回调已发生，不作为两段运动之间的阶段闸门。
    这里始终读取 ``getJoints(..., fb=True)`` 返回的反馈，确认对应目标轴到位后才返回。
    """

    if (
        not math.isfinite(FEEDBACK_CONFIRM_TIMEOUT_S)
        or FEEDBACK_CONFIRM_TIMEOUT_S <= 0.0
    ):
        raise ValueError("FEEDBACK_CONFIRM_TIMEOUT_S 必须为正的有限秒数")
    if (
        not math.isfinite(FEEDBACK_CONFIRM_POLL_INTERVAL_S)
        or FEEDBACK_CONFIRM_POLL_INTERVAL_S <= 0.0
    ):
        raise ValueError("FEEDBACK_CONFIRM_POLL_INTERVAL_S 必须为正的有限秒数")

    reports: list[dict[str, object] | None] = [None] * len(submitted_plans)
    pending_indices: set[int] = set(range(len(submitted_plans)))
    last_errors: dict[int, list[dict[str, float | int]]] = {}
    deadline = time.monotonic() + FEEDBACK_CONFIRM_TIMEOUT_S

    while pending_indices:
        for index in tuple(pending_indices):
            plan, motion = submitted_plans[index]
            if motion is None:
                reports[index] = {
                    "机械臂": plan.arm_name,
                    "机器人 ID": plan.arm.robot_id,
                    "状态": "所有目标轴已在目标角度附近，未发送运动命令",
                }
                pending_indices.remove(index)
                continue

            feedback_angles_deg = plan.arm.joint_angles()
            errors = _target_feedback_errors(plan, feedback_angles_deg)
            if errors:
                last_errors[index] = errors
                continue

            reports[index] = {
                "机械臂": plan.arm_name,
                "机器人 ID": plan.arm.robot_id,
                "已运动的轴索引": list(plan.changed_axis_indices),
                "状态": "实际关节反馈已确认到位",
                "运动序号": motion.sequence,
                "反馈关节角度（度）": list(feedback_angles_deg),
            }
            pending_indices.remove(index)

        if not pending_indices:
            break

        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            details = [
                {
                    "机械臂": submitted_plans[index][0].arm_name,
                    "机器人 ID": submitted_plans[index][0].arm.robot_id,
                    "未到位轴": last_errors.get(index, []),
                }
                for index in sorted(pending_indices)
            ]
            raise MotionRejectedError(
                "在 "
                f"{FEEDBACK_CONFIRM_TIMEOUT_S} 秒内未从实际关节反馈确认目标到位："
                + json.dumps(details, ensure_ascii=False)
            )
        time.sleep(min(FEEDBACK_CONFIRM_POLL_INTERVAL_S, remaining_s))

    return [report for report in reports if report is not None]


def _submit_plans_then_confirm_feedback(
    plans: tuple[JointTargetPlan, ...],
) -> list[dict[str, object]]:
    """先连续提交所有计划，再以实际反馈确认到位，不提供原子或严格同步起动。"""

    submitted_plans: list[tuple[JointTargetPlan, MotionHandle | None]] = []
    for plan in plans:
        try:
            submitted_plans.append((plan, _submit_plan(plan)))
        except (HcxSdkError, ValueError, TypeError) as exc:
            moving_arms = [
                submitted_plan.arm_name
                for submitted_plan, motion in submitted_plans
                if motion is not None
            ]
            if moving_arms:
                raise MotionRejectedError(
                    f"{plan.arm_name} 运动命令提交失败；"
                    f"已提交的{'、'.join(moving_arms)}可能仍在运动，"
                    "请立即在现场确认状态并按安全流程处理"
                ) from exc
            raise

    return _confirm_submitted_plans_by_feedback(submitted_plans)


def main() -> int:
    """在安全确认后连接、自动使能并按反馈确认分两段运动。"""

    if not CONFIRM_MOTION:
        print("未执行运动：请在现场确认安全后将 CONFIRM_MOTION 改为 True。")
        return 2
    if LEFT_ARM_ID == RIGHT_ARM_ID:
        print("配置错误：LEFT_ARM_ID 和 RIGHT_ARM_ID 必须不同。", file=sys.stderr)
        return 2

    robot: RobotClient | None = None
    operation_error: HcxSdkError | ValueError | TypeError | None = None
    reports: list[dict[str, object]] | None = None
    try:
        robot = RobotClient(LOCAL_IP, REMOTE_IP, PORT)
        robot.connect()
        _prepare_controller_for_enable(robot)
        left_arm = robot.arm(LEFT_ARM_ID)
        right_arm = robot.arm(RIGHT_ARM_ID)
        arms = (
            ("左臂", left_arm),
            ("右臂", right_arm),
        )
        if AUTO_ENABLE:
            changes = _ensure_motion_enabled(robot, arms)
            if changes:
                print("已自动开启：" + "、".join(changes))
        _require_motion_enabled(robot, arms)

        full_plans = (
            _build_plan(
                "左臂",
                left_arm,
                LEFT_TARGET_ANGLES_DEG,
                "LEFT_TARGET_ANGLES_DEG",
            ),
            _build_plan(
                "右臂",
                right_arm,
                RIGHT_TARGET_ANGLES_DEG,
                "RIGHT_TARGET_ANGLES_DEG",
            ),
        )
        first_stage_plans = tuple(_build_non_j1_plan(plan) for plan in full_plans)

        print("第一段运动计划：保持 J1，仅移动其余关节。")
        print(
            json.dumps(
                [_describe_plan(plan) for plan in first_stage_plans],
                ensure_ascii=False,
                indent=2,
            )
        )
        print("正在连续提交左右臂第一段命令，并读取反馈确认其余关节到位。")
        first_stage_reports = _submit_plans_then_confirm_feedback(first_stage_plans)

        second_stage_plans = tuple(
            _build_j1_only_plan(full_plan, first_stage_plan)
            for full_plan, first_stage_plan in zip(full_plans, first_stage_plans)
        )
        print("第二段运动计划：其余关节保持第一段目标，仅移动 J1。")
        print(
            json.dumps(
                [_describe_plan(plan) for plan in second_stage_plans],
                ensure_ascii=False,
                indent=2,
            )
        )
        print("正在连续提交左右臂第二段 J1 命令，并读取反馈确认到位。")
        second_stage_reports = _submit_plans_then_confirm_feedback(second_stage_plans)
        reports = [
            {"阶段": "第一段：J1 保持", **report}
            for report in first_stage_reports
        ] + [
            {"阶段": "第二段：J1 到目标", **report}
            for report in second_stage_reports
        ]
    except (HcxSdkError, ValueError, TypeError) as exc:
        operation_error = exc
    finally:
        if robot is not None:
            try:
                robot.close()
            except HcxSdkError as exc:
                if operation_error is None:
                    operation_error = exc
                else:
                    print(f"关闭控制器连接失败: {exc}", file=sys.stderr)

    if operation_error is not None:
        print(f"关节目标运动失败: {operation_error}", file=sys.stderr)
        return 1

    assert reports is not None
    print("运动结果：")
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
