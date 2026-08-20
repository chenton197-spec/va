"""FAIRINO FR3 从臂适配器。"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..interfaces import FollowerArm


class FairinoFR3Follower(FollowerArm):
    """封装 FAIRINO SDK 的连接、ServoJ 与伺服恢复流程。"""

    def __init__(self, robot_ip: str = "192.168.57.3"):
        self.robot_ip = robot_ip
        self._robot: Any | None = None
        self._send_frame_count = 0

    @property
    def joint_count(self) -> int:
        """FR3 的固定六轴关节数量。"""
        return 6

    @property
    def joint_limits_deg(self) -> tuple[np.ndarray, np.ndarray]:
        """FR3 默认安全关节限位（度）。"""
        return (
            np.array([-170, -265, -145, -265, -170, -355], dtype=float),
            np.array([170, 85, 145, 85, 170, 355], dtype=float),
        )

    @staticmethod
    def _load_robot_class() -> Any:
        """在真实连接时加载仓库内或环境中已安装的 FAIRINO SDK。"""
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from fairino390.linux.fairino import Robot

        return Robot

    def connect(self) -> None:
        """连接 FR3、诊断状态，并按原脚本执行自动模式和上使能请求。"""
        try:
            robot_class = self._load_robot_class()
            # 局部变量在本次连接流程内确定非空，便于静态类型检查。
            robot = robot_class.RPC(self.robot_ip)
            self._robot = robot
            print(f"[INFO] 已连接 FR3 机械臂: {self.robot_ip}")
            _, version = robot.GetSDKVersion()
            print(f"[INFO] FR3 SDK 版本: {version}")
            time.sleep(0.5)

            self._diagnose()
            try:
                already_auto = robot.robot_state_pkg.robot_mode == 0
            except AttributeError:
                already_auto = False
            if already_auto:
                print("[INFO] FR3 已处于自动模式（跳过 Mode(0)）")
            else:
                ret = robot.Mode(0)
                if ret == 0:
                    print("[INFO] FR3 已切换到自动模式")
                else:
                    print(f"[WARN] Mode(0) 返回 {ret}")

            ret = robot.RobotEnable(1)
            if ret == 0:
                print("[INFO] FR3 已上使能")
            else:
                print(f"[INFO] RobotEnable(1) 返回 {ret}（机器人已处于使能状态）")
            time.sleep(0.3)
            self._print_initial_angles()
        except Exception as exc:
            self._robot = None
            raise RuntimeError(f"连接 FR3 失败: {exc}") from exc

    def _diagnose(self) -> bool:
        """迁移原脚本的 FR3 状态包诊断和报警清除逻辑。"""
        if self._robot is None:
            return False
        try:
            package = self._robot.robot_state_pkg
            mode_str = "自动" if package.robot_mode == 0 else "手动"
            state_map = {1: "停止", 2: "运行中", 3: "暂停", 4: "拖动示教"}
            state_str = state_map.get(
                package.robot_state, f"未知({package.robot_state})"
            )
            print(f"[INFO] FR3 运动模式: {mode_str}  运动状态: {state_str}")

            has_fault = package.main_code != 0
            if has_fault:
                print(
                    f"[WARN] FR3 存在故障码: 主码={package.main_code}  子码={package.sub_code}"
                )
                print("[INFO] 尝试自动清除错误...")
                ret = self._robot.ResetAllError()
                if ret == 0:
                    print("[INFO] FR3 错误已清除")
                    has_fault = False
                else:
                    print(
                        f"[WARN] 自动清除失败 (ret={ret})，请在示教器手动清除故障后重试"
                    )

            if package.robot_mode == 1:
                print("[ERROR] FR3 当前处于手动模式，ServoJ 无法执行")
                return False
            return not has_fault
        except AttributeError:
            print("[WARN] 状态包尚未就绪，跳过诊断")
            return True

    def _print_initial_angles(self) -> None:
        """读取并打印当前关节角度，保持原脚本的连接期诊断行为。"""
        if self._robot is None:
            return
        ret, angles = self._robot.GetActualJointPosDegree(0)
        if ret == 0 and isinstance(angles, (list, np.ndarray)) and len(angles) == 6:
            values = np.array(angles, dtype=float)
            print(f"[INFO] FR3 当前关节角度: {np.round(values, 2).tolist()}")
            min_angles = np.array([-170, -265, -145, -265, -170, -355], dtype=float)
            max_angles = np.array([170, 85, 145, 85, 170, 355], dtype=float)
            near_limit = np.where(np.minimum(values - min_angles, max_angles - values) < 5.0)[0]
            if len(near_limit) > 0:
                joints = ", ".join(f"J{i + 1}({values[i]:.1f}°)" for i in near_limit)
                print(f"[WARN] 以下关节距限位不足 5°: {joints}")
        else:
            print(f"[WARN] 获取 FR3 初始角度失败 ret={ret}，以零位为基准")

    def read_joint_angles_deg(self) -> np.ndarray:
        """读取 FR3 当前角度；失败时遵循原脚本的零位回退语义。"""
        if self._robot is None:
            return np.zeros(6, dtype=float)
        ret, angles = self._robot.GetActualJointPosDegree(0)
        if ret == 0 and isinstance(angles, (list, np.ndarray)) and len(angles) == 6:
            return np.array(angles, dtype=float)
        print(f"[WARN] 获取 FR3 初始角度失败 ret={ret}，以零位为基准")
        return np.zeros(6, dtype=float)

    def read_cached_joint_angles_deg(self) -> np.ndarray | None:
        """从 SDK 的实时状态包读取关节反馈，不发起同步 RPC。

        ``robot_state_pkg`` 由 FAIRINO SDK 的状态接收线程更新。采集侧应使用
        此方法，避免 30 Hz 观测查询与 125 Hz ServoJ 共用命令通道。
        """

        if self._robot is None:
            return None
        try:
            values = np.asarray(self._robot.robot_state_pkg.jt_cur_pos, dtype=float)
        except (AttributeError, TypeError, ValueError):
            return None
        if values.shape != (self.joint_count,) or not np.isfinite(values).all():
            return None
        return values.copy()

    def start_servo(self) -> bool:
        """启动 FR3 ServoMove 模式。"""
        if self._robot is None:
            return False
        ret = self._robot.ServoMoveStart()
        if ret == 0:
            print("[INFO] FR3 伺服模式已开启 (ServoMoveStart OK)")
            return True
        print(f"[ERROR] ServoMoveStart 失败 (ret={ret})")
        return False

    def send_joint_angles_deg(
        self, angles_deg: np.ndarray, command_time_s: float
    ) -> bool:
        """通过 ServoJ 下发一帧角度制目标，并保留原有队列积压提示。"""
        if self._robot is None:
            return False
        try:
            ret = self._robot.ServoJ(
                np.asarray(angles_deg, dtype=float).tolist(),
                [0.0, 0.0, 0.0, 0.0],
                0.0,
                0.0,
                command_time_s,
            )
            if ret != 0:
                print(f"[WARN] ServoJ 错误码: {ret}")
                return False
            self._send_frame_count += 1
            if self._send_frame_count % 25 == 0:
                _, queue_len = self._robot.GetMotionQueueLength()
                if isinstance(queue_len, int) and queue_len >= 2:
                    print(f"[WARN] 运动队列积压: {queue_len}，可能有延迟")
            return True
        except Exception as exc:
            print(f"[ERROR] FR3 控制异常: {exc}")
            return False

    def recover(self) -> bool:
        """迁移原脚本的报警清除和 ServoMove 重启流程。"""
        if self._robot is None:
            return False
        try:
            self._robot.ServoMoveEnd()
            ret = self._robot.ResetAllError()
            if ret == 0:
                print("[INFO] FR3 报警已清除，正在重新初始化伺服模式...")
            else:
                print(f"[WARN] ResetAllError 返回 {ret}，继续尝试重启伺服")
            time.sleep(0.1)
            ret = self._robot.ServoMoveStart()
            if ret == 0:
                print("[INFO] ServoMoveStart 重新成功，继续遥操作")
                return True
            print(f"[ERROR] ServoMoveStart 重新失败 (ret={ret})，请检查示教器状态")
        except Exception as exc:
            print(f"[ERROR] 伺服恢复异常: {exc}")
        return False

    def stop_servo(self) -> None:
        """结束 ServoMove 并停止 FR3 运动。"""
        if self._robot is None:
            return
        try:
            self._robot.ServoMoveEnd()
            self._robot.StopMotion()
            print("[INFO] FR3 伺服模式已关闭，运动已停止")
        except Exception as exc:
            print(f"[WARN] 停止 FR3 时出错: {exc}")

    def disconnect(self) -> None:
        """释放适配器持有的 FR3 引用。"""
        self._robot = None
