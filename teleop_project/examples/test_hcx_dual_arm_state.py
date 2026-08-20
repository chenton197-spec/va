#!/usr/bin/env python3
"""只读打印 HCX 左右七轴机械臂的角度与原始力矩反馈。"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

from teleop_sdk.config import HcxConfig, load_runtime_config


def _load_robot_client() -> Any:
    """在实际运行示例时加载 HCX 原生 SDK。"""

    from hcx_sdk import RobotClient

    return RobotClient


def _validate_config(config: HcxConfig) -> None:
    """在建立网络连接前验证部署配置。"""

    if not isinstance(config.local_ip, str) or not config.local_ip.strip():
        raise ValueError("hcx.local_ip 不能为空")
    if not isinstance(config.remote_ip, str) or not config.remote_ip.strip():
        raise ValueError("hcx.remote_ip 不能为空")
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
            raise ValueError("hcx.connect_timeout_s 必须是正的有限秒数或 null") from exc
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
            or not 0 <= master_index <= 1
        ):
            raise ValueError("hcx.ethercat_master_indices 只能包含 0 或 1")


def _read_ethercat_master_status(
    robot: Any, master_indices: tuple[int, ...]
) -> dict[str, bool]:
    """读取显式配置的 EtherCAT 主站 OP 状态。"""

    return {
        f"主站 {master_index}": robot.ethercat_master_operational(master_index)
        for master_index in master_indices
    }


def _read_arm_information(robot: Any, robot_id: int, arm_name: str) -> dict[str, object]:
    """读取一条七轴机械臂的只读状态，不改变控制器状态。"""

    arm = robot.arm(robot_id)
    joint_angles_deg = arm.joint_angles()
    joint_torque_feedback = arm.joint_torque_feedback()
    if arm.axis_count != 7:
        raise ValueError(f"{arm_name} 预期为 7 轴，控制器返回 {arm.axis_count} 轴")
    return {
        "机器人 ID": robot_id,
        "单臂使能": arm.enabled,
        "防护已启用": arm.protection_enabled,
        "轴数": arm.axis_count,
        "关节角度（度）": list(joint_angles_deg),
        "关节力矩反馈（原始值）": list(joint_torque_feedback),
    }


def main() -> int:
    """连接一次并输出双臂诊断快照；不执行任何控制命令。"""

    try:
        config = load_runtime_config().hcx
        _validate_config(config)
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f"[ERROR] HCX 配置无效: {exc}", file=sys.stderr)
        return 2

    robot: Any | None = None
    operation_error: Exception | None = None
    information: dict[str, object] | None = None
    try:
        robot_client_class = _load_robot_client()
        robot = robot_client_class(config.local_ip, config.remote_ip, config.port)
        robot.connect(timeout_s=config.connect_timeout_s)
        information = {
            "连接状态": robot.connected,
            "本机 IP": config.local_ip,
            "控制器 IP": config.remote_ip,
            "端口": config.port,
            "全局使能": robot.global_enabled,
            "活动报警": list(robot.active_alarms),
            "软急停正常": robot.soft_emergency_stop_normal,
            "示教器已脱离": robot.hmi_detached,
            "EtherCAT 主站 OP 状态": _read_ethercat_master_status(
                robot, config.ethercat_master_indices
            ),
            "机械臂": {
                "左臂": _read_arm_information(
                    robot, config.left_robot_id, "左臂"
                ),
                "右臂": _read_arm_information(
                    robot, config.right_robot_id, "右臂"
                ),
            },
        }
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        operation_error = exc
    finally:
        if robot is not None:
            try:
                robot.close()
            except (RuntimeError, TypeError, ValueError) as exc:
                if operation_error is None:
                    operation_error = exc
                else:
                    print(f"关闭 HCX 控制器连接失败: {exc}", file=sys.stderr)

    if operation_error is not None:
        print(f"读取 HCX 双臂状态失败: {operation_error}", file=sys.stderr)
        return 1

    assert information is not None
    print("只读模式：未调用使能、运动、暂停、清路、直伺服或防护状态修改接口。")
    print(json.dumps(information, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
