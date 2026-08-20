"""硬件无关的主从遥操作控制器。"""

from __future__ import annotations

from collections.abc import Callable
import threading
import time

import numpy as np

from .algorithms import AngleUnwrapper, LatencyProbe, SpringDamper
from .config import TeleopConfig
from .filters import LowPassFilter, OneEuroFilter
from .interfaces import (
    FollowerArm,
    GripperActuator,
    LeaderArm,
    LeaderArmWithGripper,
    LeaderGripperInput,
)
from .timing import precise_sleep


class TeleopController:
    """迁移自原脚本的遥操作算法。

    控制器中的计算、默认调用顺序和安全阈值保持原有实现；不同机械臂
    仅通过 ``LeaderArm`` 与 ``FollowerArm`` 接口接入。
    """

    def __init__(
        self,
        leader: LeaderArm,
        follower: FollowerArm,
        config: TeleopConfig,
        gripper: GripperActuator | None = None,
        leader_gripper: LeaderGripperInput | None = None,
        on_joint_target_submitted: Callable[[np.ndarray, float], None] | None = None,
        on_joint_target_generated: Callable[[np.ndarray, float], None] | None = None,
    ):
        self.leader = leader
        self.follower = follower
        self.config = config
        self.gripper = gripper
        self.leader_gripper = leader_gripper
        if on_joint_target_submitted is not None and not callable(
            on_joint_target_submitted
        ):
            raise TypeError("on_joint_target_submitted must be callable or None")
        if on_joint_target_generated is not None and not callable(
            on_joint_target_generated
        ):
            raise TypeError("on_joint_target_generated must be callable or None")
        self._on_joint_target_submitted = on_joint_target_submitted
        # 该旁路观察点位于映射、限位和可选平滑之后，死区抑制之前。它用于
        # 诊断上游目标的连续性，不参与从臂发送或硬件状态读取。
        self._on_joint_target_generated = on_joint_target_generated

        if leader.joint_count != follower.joint_count:
            raise ValueError(
                "示教臂与从臂关节数量不一致："
                f"{leader.joint_count} 轴与 {follower.joint_count} 轴不能直接配对"
            )
        n_joints = follower.joint_count
        # YAML 未显式指定限位、轴顺序或方向时，使用从臂的安全限位和同轴同向映射。
        default_min, default_max = follower.joint_limits_deg
        self.min_angles = np.array(
            config.min_angles_deg if config.min_angles_deg is not None else default_min,
            dtype=float,
        )
        self.max_angles = np.array(
            config.max_angles_deg if config.max_angles_deg is not None else default_max,
            dtype=float,
        )
        self.axis_order = (
            config.axis_order if config.axis_order is not None else tuple(range(n_joints))
        )
        self.axis_sign = np.array(
            config.axis_sign if config.axis_sign is not None else (1.0,) * n_joints,
            dtype=float,
        )
        # 相对模式以首帧主臂角度为零点；绝对模式直接将主臂角度映射到从臂目标。
        self._relative = config.relative_mode
        self._leader_start: np.ndarray | None = None

        # 关节一级滤波：One Euro 根据关节运动速度动态提高截止频率，静止时抑制噪声，
        # 快速运动时尽量降低跟手延迟。
        self._leader_filter: OneEuroFilter | None = (
            OneEuroFilter(
                n_joints=n_joints,
                mincutoff=config.filter_mincutoff_hz,
                beta=config.filter_beta,
            )
            if config.filter_enabled
            else None
        )
        # 关节二级滤波：固定截止频率低通，进一步削弱手部高频抖动。
        self._tremor_filter: LowPassFilter | None = (
            LowPassFilter(n_joints=n_joints, cutoff_hz=config.tremor_cutoff_hz)
            if config.filter_enabled
            else None
        )
        # 连接完成后会以真实从臂反馈覆盖这两个数组，作为相对模式基准和命令历史。
        self.follower_init_angles = np.zeros(n_joints, dtype=float)
        self.last_follower_angles = self.follower_init_angles.copy()
        self._last_velocity: np.ndarray | None = None
        # 弹簧阻尼只处理从臂关节角度数组；夹爪在 _step_gripper 中独立处理。
        # 即使运行时关闭弹簧路径，也保留实例以便重新启用时不改变控制器结构。
        self._spring_damper = SpringDamper(
            rate_hz=config.rate_hz,
            omega=config.spring_omega,
            jump_threshold_deg=config.jump_threshold_deg,
            max_accel_deg_s2=config.max_accel_deg_s2,
            max_vel_deg_s=config.max_vel_deg_s,
            min_angles_deg=self.min_angles,
            max_angles_deg=self.max_angles,
        )
        self._follower_ready = False
        self._servo_start_event = threading.Event()
        self._last_recovery_time = 0.0
        # 消除主臂编码器跨越 +/-180 度时的角度跳变，避免误触发安全跳变保护。
        self._unwrapper = AngleUnwrapper(leader.joint_count)
        # 延迟探针只观测关节输入到从臂命令的时序，不参与控制结果计算。
        self._latency_probe = (
            LatencyProbe(
                config.rate_hz,
                config.latency_probe_threshold_deg,
                config.latency_probe_quiescent_deg,
            )
            if config.latency_probe_enabled
            else None
        )

    def connect(self) -> None:
        """按原脚本顺序连接主臂、从臂并记录从臂当前角度。"""

        # 每次新会话都清除上次的伺服启动状态；避免录制线程读取到旧会话结果。
        self._validate_config()
        self._servo_start_event.clear()
        self._follower_ready = False
        self.leader.connect()
        self.follower.connect()
        if self.gripper is not None:
            try:
                self.gripper.connect()
            except Exception as exc:
                # 夹爪是可选执行器，连接失败不能阻断六轴遥操。
                print(f"[WARN] 从端夹爪连接失败，本次运行不控制夹爪: {exc}")
                self.gripper = None
        # 相对模式以该姿态为基准；绝对模式仍保留它作为命令历史的初始值。
        self.follower_init_angles = self.follower.read_joint_angles_deg().copy()
        self.last_follower_angles = self.follower_init_angles.copy()

    def _validate_config(self) -> None:
        """在开始控制前检查映射与限位的基本维度。"""
        n_joints = self.follower.joint_count
        if len(self.axis_order) != n_joints:
            raise ValueError("轴映射数量必须与从臂关节数量一致")
        if len(self.axis_sign) != n_joints:
            raise ValueError("轴方向数量必须与轴映射数量一致")
        if len(self.min_angles) != n_joints or len(self.max_angles) != n_joints:
            raise ValueError("从臂关节限位数量必须与从臂关节数量一致")
        if np.any(self.min_angles >= self.max_angles):
            raise ValueError("每个从臂关节的最小限位必须小于最大限位")
        if any(sign not in (-1.0, 1.0) for sign in self.axis_sign):
            raise ValueError("轴方向只能为 +1 或 -1")
        if any(index < 0 or index >= self.leader.joint_count for index in self.axis_order):
            raise ValueError("轴映射索引超出关节范围")

    def _map_to_follower(self, leader_deg: np.ndarray) -> np.ndarray:
        """按配置执行主臂到从臂的相对或绝对轴映射。"""
        if self._relative and self._leader_start is not None:
            delta = leader_deg - self._leader_start
        else:
            delta = leader_deg

        # 相对模式叠加从臂启动姿态；绝对模式以双方完成标定后的零位直接对应。
        target = (
            self.follower_init_angles.copy()
            if self._relative
            else np.zeros_like(self.follower_init_angles)
        )
        for index in range(len(self.axis_order)):
            source = self.axis_order[index]
            target[index] += self.axis_sign[index] * delta[source]
        # 相对模式的主臂基准在本次会话内保持不变。调用方的 _limit() 会逐轴
        # 钳制越界目标，因此主臂必须回到有效映射范围后，从臂才会从边界向内移动。
        return target

    def _limit(self, angles: np.ndarray) -> np.ndarray:
        """将目标钳制在配置的各关节安全范围内。"""
        return np.clip(angles, self.min_angles, self.max_angles)

    def _smooth(self, target: np.ndarray) -> np.ndarray | None:
        """迁移原脚本保留的三级平滑实现。

        当前主循环与原脚本一致，使用临界阻尼弹簧路径而不调用本方法；
        该方法保留以确保原有算法完整迁移。
        """
        # 该旧路径未被主循环调用，保留用于兼容旧算法和离线验证。
        delta = target - self.last_follower_angles
        if np.any(np.abs(delta) > self.config.jump_threshold_deg):
            print(
                f"[WARN] 关节跳变，跳过本次。最大变化: {np.round(np.abs(delta).max(), 2)}°"
            )
            self._last_velocity = None
            return None

        velocity = np.clip(delta, -self.config.max_step_deg, self.config.max_step_deg)
        if self._last_velocity is not None:
            max_accel = self.config.max_step_deg / 3.0
            delta_velocity = velocity - self._last_velocity
            velocity = self._last_velocity + np.clip(
                delta_velocity, -max_accel, max_accel
            )

        self._last_velocity = velocity.copy()
        return self.last_follower_angles + velocity

    def _recover_follower(self) -> bool:
        """保留原脚本 0.5 秒恢复节流和弹簧状态重置。"""

        # 连续通信失败时限制恢复频率，避免每个控制周期都重复切换伺服状态。
        now = time.perf_counter()
        if now - self._last_recovery_time < 0.5:
            return False
        self._last_recovery_time = now
        self._follower_ready = False
        self._last_velocity = None
        self._spring_damper.reset()
        self._follower_ready = self.follower.recover()
        return self._follower_ready

    def _refresh_follower_servo_target(self) -> bool:
        """维持需要最新目标心跳的从臂伺服会话。"""

        if not self._follower_ready:
            return False
        if self.follower.refresh_servo_target():
            return True
        print("[WARN] 从臂伺服目标保活失败，尝试自动恢复...")
        self._recover_follower()
        return False

    def _send(self, angles: np.ndarray) -> None:
        """原样迁移死区处理、伺服发送和失败恢复流程。"""

        # 普通从臂在死区内不发送命令；高频轨迹从臂需要每个输入采样点，
        # 因此由通用能力声明绕过该抑制逻辑。
        delta = np.abs(angles - self.last_follower_angles)
        continuous_targets = self.follower.requires_per_cycle_target_updates
        if np.all(delta < self.config.dead_zone_deg) and not continuous_targets:
            return
        if not continuous_targets:
            angles = np.where(
                delta < self.config.dead_zone_deg, self.last_follower_angles, angles
            )

        command_accepted = False
        if self._follower_ready:
            command_time = max(0.008, 1.0 / self.config.rate_hz)
            if not self.follower.send_joint_angles_deg(angles, command_time):
                print("[WARN] 从臂伺服命令失败，尝试自动恢复...")
                self._recover_follower()
                return
            command_accepted = True

        self.last_follower_angles = angles.copy()
        if command_accepted:
            self._notify_joint_target_submitted(angles)

    def _notify_joint_target_submitted(self, angles: np.ndarray) -> None:
        """通知可选观察者本次目标已被从臂接受，不读取实际关节反馈。"""

        observer = self._on_joint_target_submitted
        if observer is None:
            return
        try:
            observer(angles.copy(), time.perf_counter())
        except Exception as exc:
            # 可视化和记录属于旁路功能，不能让其异常中断实时控制。
            self._on_joint_target_submitted = None
            print(f"[WARN] 关节目标观察回调已禁用: {exc}")

    def _notify_joint_target_generated(self, angles: np.ndarray) -> None:
        """通知可选观察者映射和可选平滑后的完整控制器目标。"""

        observer = self._on_joint_target_generated
        if observer is None:
            return
        try:
            observer(angles.copy(), time.perf_counter())
        except Exception as exc:
            # 诊断或可视化必须是旁路；观察者失败不能阻断实时控制。
            self._on_joint_target_generated = None
            print(f"[WARN] 关节生成目标观察回调已禁用: {exc}")

    def step(self, now: float | None = None) -> bool:
        """执行一个控制周期；便于测试和外部调度器复用。"""
        timestamp = time.perf_counter() if now is None else now
        # 在可能阻塞的主臂读取前先刷新一次。默认从臂实现为无操作，只有需要
        # 上游目标心跳的适配器会发送其保存的最新目标。
        self._refresh_follower_servo_target()
        gripper_opening: float | None = None
        # 支持主臂一次性返回关节和夹爪，保证同一控制周期内两者来自同一采样帧。
        if self.gripper is not None and isinstance(self.leader, LeaderArmWithGripper):
            combined = self.leader.read_joint_angles_and_gripper_opening(timeout_s=0.1)
            if combined is None:
                self._refresh_follower_servo_target()
                return False
            leader_deg, gripper_opening = combined
        else:
            leader_deg = self.leader.read_joint_angles_deg(timeout_s=0.1)
            if self.gripper is not None and self.leader_gripper is not None:
                gripper_opening = self.leader_gripper.read_gripper_opening(0.1)
        if leader_deg is None:
            # 双臂串行读取的一侧超时时，仍为本侧直伺服刷新目标，避免把短暂
            # 读取抖动误判为上游控制链已停止。
            self._refresh_follower_servo_target()
            return False
        # 先解包角度连续性，再对关节执行两级滤波；夹爪不使用这一条关节路径。
        leader_deg = self._unwrapper.step(leader_deg)
        leader_raw = leader_deg.copy()

        if self._leader_filter is not None:
            leader_deg = self._leader_filter.step(leader_deg, timestamp)
        if self._tremor_filter is not None:
            leader_deg = self._tremor_filter.step(leader_deg, timestamp)

        # 夹爪命令独立下发，不等待关节映射、限位或弹簧阻尼计算完成。
        self._step_gripper(gripper_opening)

        if self._relative and self._leader_start is None:
            self._leader_start = leader_deg.copy()
            print(
                f"[INFO] 示教臂起始位置已记录: {np.round(self._leader_start, 1).tolist()}"
            )
            print("[INFO] 现在移动示教臂，从臂将跟随相对变化量")

        # 关节控制路径：主臂映射 -> 安全限位 -> 可选弹簧阻尼/前瞻预测 -> 伺服下发。
        target = self._map_to_follower(leader_deg)
        target = self._limit(target)
        if self.config.spring_enabled:
            smoothed = self._spring_damper.step(target, self.last_follower_angles)
            smoothed = self._spring_damper.predict(
                self.config.predict_lookahead_ms / 1000.0
            )
        else:
            # 关闭弹簧路径时直接下发映射后的目标，便于与平滑控制做同条件对比。
            smoothed = target
        self._notify_joint_target_generated(smoothed)
        self._send(smoothed)
        # 每个完整控制周期结束前再刷新一次，缩短双臂串行读取时任一侧的最长
        # 心跳间隔；真正卡住超过看门狗的循环仍会由从臂安全停止。
        self._refresh_follower_servo_target()
        if self._latency_probe is not None:
            for axis, latency_ms in self._latency_probe.step(
                leader_raw, self.last_follower_angles, self.axis_order, timestamp
            ):
                print(
                    f"[目标提交诊断] J{axis + 1} 主臂变化到目标提交: "
                    f"{latency_ms:.1f} ms"
                )
        return True

    def _step_gripper(self, opening: float | None) -> None:
        """将示教端归一化夹爪输入直接限幅后下发给可选从端夹爪。"""
        if self.gripper is None or opening is None:
            return
        # 夹爪不参与关节的 One Euro、固定低通或弹簧阻尼；仅保留执行器边界限幅。
        opening = float(np.clip(float(opening), 0.0, 1.0))
        if not self.gripper.send_normalized(opening):
            print("[WARN] 从端夹爪命令失败，本帧跳过")

    def shutdown(self) -> None:
        """按固定顺序停止运动并释放设备资源。

        独立调用时 ``run()`` 默认退出即清理；录制会话可延迟到采样器和相机工作线程
        完全停止后再调用，避免资源关闭早于数据收尾。
        """

        # 先停止从臂伺服，再断开夹爪和主臂，防止断开主臂后从臂继续保持最后一帧命令。
        self.follower.stop_servo()
        self.follower.disconnect()
        if self.gripper is not None:
            self.gripper.disable()
            self.gripper.disconnect()
        self.leader.disconnect()
        print("[INFO] 设备已断开，程序退出")

    @property
    def follower_ready(self) -> bool:
        """最近一次从臂伺服启动请求是否成功。"""

        return self._follower_ready

    def start_servo(self) -> bool:
        """启动从臂伺服，供外部控制循环在 ``connect`` 后调用。

        单臂 ``run`` 与需要统一调度多条控制链路的调用方共用此入口，确保
        ``follower_ready`` 和 ``wait_for_servo_start`` 的状态语义保持一致。
        """

        self._follower_ready = self.follower.start_servo()
        self._servo_start_event.set()
        return self._follower_ready

    def wait_for_servo_start(self, timeout_s: float) -> bool:
        """等待 ``run`` 或 ``start_servo`` 完成一次从臂伺服启动尝试。"""

        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if not self._servo_start_event.wait(timeout_s):
            raise TimeoutError("Timed out waiting for follower servo startup")
        return self._follower_ready

    def run(
        self,
        stop_event: threading.Event | None = None,
        *,
        cleanup_on_exit: bool = True,
    ) -> None:
        """按控制频率运行，直到 Ctrl+C 或可选的停止事件触发。"""
        mode_str = "相对（推荐）" if self._relative else "绝对（需零位校准）"
        print("=" * 60)
        print("    通用主从遥操作系统已启动")
        print("=" * 60)
        print(
            f"  控制频率: {self.config.rate_hz} Hz  最大步长: {self.config.max_step_deg}°"
        )
        print(f"  运动模式: {mode_str}")
        print(f"  轴顺序:   {list(self.axis_order)}")
        print(f"  轴方向:   {self.axis_sign.tolist()}")
        print("  按 Ctrl+C 安全退出")
        print("-" * 60)

        interval = 1.0 / self.config.rate_hz
        spin_threshold = 0.002 if interval <= 0.010 else 0.010
        if not self.start_servo():
            print("[ERROR] 从臂伺服模式启动失败")

        if self._relative:
            print("[INFO] 相对模式：保持示教臂静止，从臂将不动。移动示教臂后从臂跟随。")

        next_deadline = time.perf_counter()
        try:
            while stop_event is None or not stop_event.is_set():
                start = time.perf_counter()
                self.step(start)
                if stop_event is not None and stop_event.is_set():
                    break
                next_deadline += interval
                remaining = max(0.0, next_deadline - time.perf_counter())
                if stop_event is None:
                    precise_sleep(remaining, spin_threshold_s=spin_threshold)
                else:
                    # 外部停止事件使用可中断等待，末尾仍保留短暂忙等以维持既有调度精度。
                    blocking = max(0.0, remaining - spin_threshold)
                    if blocking > 0.0 and stop_event.wait(blocking):
                        break
                    if stop_event.is_set():
                        break
                    precise_sleep(
                        max(0.0, next_deadline - time.perf_counter()),
                        spin_threshold_s=spin_threshold,
                    )
        except KeyboardInterrupt:
            print("\n[STOP] 收到中断信号，正在安全停止...")
        finally:
            if cleanup_on_exit:
                self.shutdown()
