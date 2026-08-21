#!/usr/bin/env python3
"""W2 HCX 左臂 A2A 伺服闭环部署（15Hz 推理 + 100Hz 直伺服源点 → 500Hz 输出）。

与 run.py（move_joints 规划模式）的本质区别：
- 模型按训练 fps（15Hz）推理，输出未来 n_action_steps 步**绝对关节目标**；
- 应用侧 100Hz 下发线程（teleop.rate_hz）对相邻目标点线性插值，调用
  ``HcxFollower.send_joint_angles_deg``；
- SDK ``HcxDirectServoConfig``（linear/limited）在独立输出线程把每个
  100Hz 源点预生成一批高频点，按 ``hcx.direct_servo_rate_hz``（必须 500）
  调用薄原生 PluseToServo。500Hz 由 SDK 输出线程保证，不在 Python 里硬打。

对齐 ``run_left_arm_depth_direct_threaded.py`` / ``run_direct_servo.py``：
- 连接走 ``HcxConnection`` + ``HcxFollower``，不直接 ``arm.pluse_to_servo``；
- 只启动 left 直伺服；right follower 保持连接但不进伺服；
- 队列/插值不足时保持最后目标，避免 SDK watchdog 断流；
- 每秒读 ``direct_servo_output_stats().observed_rate_hz``，低于阈值则停机。

安全设计：
- 限位 clamp（deploy joint_limits + follower 限位）；
- 100Hz 可选一阶平滑（tau=0 关闭）+ 独立速度/单拍 delta 限幅；
- 推理看门狗：主线程超时未提交新 chunk 时，下发线程保持最后目标并告警；
- 首帧插值从当前物理关节角过渡到 chunk[0]，避免跳变。

用法：
    python run_w2/servo_deploy.py --deploy run_w2/servo_deploy_a2a_depth_left.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

VA_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DEPLOY_YAML = SCRIPT_DIR / "servo_deploy_a2a_depth_left.yaml"
TELEOP_ROOT = VA_ROOT / "teleop_project"
if not TELEOP_ROOT.is_dir():
    raise FileNotFoundError(f"找不到 in-repo teleop_project: {TELEOP_ROOT}")

for p in (TELEOP_ROOT, VA_ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# 推理日志降频：每步 4 行 × 8 帧/步 × 8 值的大日志（Python 逐值格式化 + flush）
# 在主线程持有 GIL 期间产生长 IO，会拖累 100Hz 源点下发（进而让 SDK 500Hz
# 批次变空、watchdog 重发旧点）。前 3 步全量打印便于确认初始状态，
# 之后按下列间隔抽样打印；调大间隔可进一步降低 GIL 竞争。
_LOG_STATE_INPUT_EVERY = 25
_LOG_RESULT_EVERY = 5

ARM_SERVO_HZ = 500
ARM_RATE_FAIL_HZ = 450.0

# ---- 复用 run.py 的连接/观测/推理链路 ----
from run import (  # noqa: E402
    BackgroundGripperLoop,
    CameraPreviewLoop,
    PREVIEW_FPS,
    FpsObservationSampler,
    HardwareBundle,
    _build_obs_batch,
    _clamp_joints_by_max_delta,
    _confirm_targets_by_feedback,
    _connect_cameras,
    _connect_dual_grippers,
    _load_deploy_config,
    _log_inference_result,
    _log_inference_state_input,
    _pace_step,
    _ramp_start_grippers,
    _read_gripper,
    _resolve_train_config,
    _shutdown,
    _validate_runtime_contract,
    denormalize_predicted_action,
    joint_mask_from_names,
)
from robotfm.config import _normalize_rtc_config, load_config
from robotfm.train import build_policy
from teleop_sdk.adapters.hcx import (  # noqa: E402
    HcxConnection,
    HcxConnectionConfig,
    HcxDirectServoConfig,
    HcxFollower,
)
from teleop_sdk.config import load_runtime_config  # noqa: E402

LEFT_ARM_DIM = 8  # L0-L6 + left_gripper


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W2 HCX 左臂 A2A 伺服部署")
    parser.add_argument("--deploy", type=str, default=str(DEFAULT_DEPLOY_YAML))
    return parser.parse_args()


def _resolve_deploy_path(value: str) -> Path:
    """相对路径按仓库根解析（与 run.py 一致），并兼容只写文件名。"""
    path = Path(value)
    if path.is_file():
        return path.resolve()
    if path.is_absolute():
        return path
    for cand in (VA_ROOT / path, SCRIPT_DIR / path, SCRIPT_DIR / path.name):
        if cand.is_file():
            return cand.resolve()
    return (VA_ROOT / path).resolve()


def _direct_servo_config_from_teleop(
    teleop_yaml: Path,
) -> tuple[Any, HcxDirectServoConfig, int]:
    """与 openarm_hcx_dual_arm_teleop / run_direct_servo 相同：100Hz 源 → 500Hz 出。"""
    runtime = load_runtime_config(teleop_yaml)
    h = runtime.hcx
    rate_hz = float(runtime.teleop.rate_hz)
    if not rate_hz.is_integer():
        raise RuntimeError("limited/linear 直伺服要求 teleop.rate_hz 为整数")
    source_hz = int(rate_hz)
    if int(h.direct_servo_rate_hz) != ARM_SERVO_HZ:
        raise RuntimeError(
            f"teleop.yaml hcx.direct_servo_rate_hz 必须是 {ARM_SERVO_HZ}，"
            f"实际为 {h.direct_servo_rate_hz}"
        )
    if ARM_SERVO_HZ % source_hz != 0:
        raise RuntimeError(
            f"{ARM_SERVO_HZ} Hz 必须是 teleop.rate_hz={source_hz} 的整数倍"
        )
    if not bool(h.direct_servo_confirm_unsafe):
        raise RuntimeError(
            "直伺服要求 teleop.yaml 中 hcx.direct_servo_confirm_unsafe: true"
        )
    source_rate_hz = (
        source_hz if h.direct_servo_interpolation in ("linear", "limited") else None
    )
    direct_cfg = HcxDirectServoConfig.from_runtime_config(
        h, source_rate_hz=source_rate_hz
    )
    if direct_cfg.watchdog_s <= 1.0 / direct_cfg.rate_hz:
        raise RuntimeError("hcx.direct_servo_watchdog_s 必须大于一个 500 Hz 周期")
    return runtime, direct_cfg, source_hz


def _connect_hcx_direct(
    teleop_yaml: Path,
) -> tuple[HcxConnection, HcxFollower, HcxFollower, Any, Any, int, HcxDirectServoConfig]:
    runtime, direct_cfg, source_hz = _direct_servo_config_from_teleop(teleop_yaml)
    h = runtime.hcx
    connection = HcxConnection(HcxConnectionConfig.from_runtime_config(h))
    left = HcxFollower(
        connection,
        robot_id=int(h.left_robot_id),
        side="left",
        direct_servo_config=direct_cfg,
    )
    right = HcxFollower(
        connection,
        robot_id=int(h.right_robot_id),
        side="right",
        direct_servo_config=direct_cfg,
    )
    left.connect()
    right.connect()
    if not connection.prepare_for_motion(int(h.left_robot_id)):
        raise RuntimeError("HCX 左臂 prepare_for_motion 失败")
    if not connection.motion_ready(int(h.left_robot_id)):
        raise RuntimeError("HCX 未处于可运动状态")
    client = connection.client
    if client is None:
        raise RuntimeError("HCX 连接未建立")
    print(
        f"[INFO] HCX 直伺服已连接: out={direct_cfg.rate_hz} Hz "
        f"source={direct_cfg.source_rate_hz} Hz "
        f"interpolation={direct_cfg.interpolation} "
        f"watchdog={direct_cfg.watchdog_s:g}s",
        flush=True,
    )
    return (
        connection,
        left,
        right,
        client.arm(int(h.left_robot_id)),
        client.arm(int(h.right_robot_id)),
        source_hz,
        direct_cfg,
    )


def _format_servo_stats(follower: HcxFollower) -> tuple[str, bool, float | None]:
    stats = follower.direct_servo_output_stats()
    if stats is None:
        return "left=no-stats", False, None
    running = bool(stats.running)
    hz = stats.observed_rate_hz
    hz_s = "empty-window" if hz is None else f"{float(hz):.1f}"
    late = stats.max_start_lateness_s
    dur = stats.max_set_target_duration_s
    late_s = f"{late * 1e3:.1f}ms" if late is not None else "n/a"
    dur_s = f"{dur * 1e3:.1f}ms" if dur is not None else "n/a"
    run_s = "run" if running else "STOP"
    line = (
        f"left={hz_s}Hz {run_s} n={stats.recent_successful_command_count} "
        f"late={late_s} set_target={dur_s}"
    )
    return line, running, None if hz is None else float(hz)


class ServoInterpolator:
    """线程安全：保存最近一次推理的绝对关节目标 chunk，按时间线性插值。

    主线程 15Hz 调用 ``submit``；100Hz 下发线程调用 ``sample``。
    返回 (n_joints,) 绝对关节目标（度），或 None（尚未提交）。
    """

    def __init__(self, source_fps: float, n_joints: int) -> None:
        self._fps = float(source_fps)
        self._n_joints = int(n_joints)
        self._lock = threading.Lock()
        self._points: np.ndarray | None = None  # (N, n_joints) 绝对目标
        self._t0 = 0.0  # chunk[0] 对应时刻（perf_counter）
        self._last: np.ndarray | None = None

    def submit(self, points: np.ndarray, t0: float) -> None:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, self._n_joints)
        with self._lock:
            self._points = pts
            self._t0 = float(t0)

    def sample(self, t: float) -> np.ndarray | None:
        with self._lock:
            if self._points is None:
                return None
            x = (t - self._t0) * self._fps  # 步序号（0 = chunk[0]）
            n = len(self._points)
            if x <= 0.0:
                target = self._points[0]
            elif x >= n - 1:
                target = self._points[-1]
            else:
                i = int(math.floor(x))
                alpha = x - i
                target = self._points[i] * (1.0 - alpha) + self._points[i + 1] * alpha
            self._last = target
            return target.copy()


class ServoOutputThread:
    """100Hz 源点下发：插值取目标 → 限位/限速 → ``send_joint_angles_deg``。

    SDK 独立 500Hz 线程消费每个源点预生成的批次。本线程必须每拍都发，
    否则 linear/limited 插值器无法填满 500Hz 队列。

    ``watchdog_s``：距上次成功 ``submit`` 超过该值时保持最后目标并告警。
    """

    def __init__(
        self,
        follower: HcxFollower,
        interp: ServoInterpolator,
        *,
        source_hz: int,
        watchdog_s: float = 0.5,
        joint_lo: np.ndarray | None = None,
        joint_hi: np.ndarray | None = None,
        command_filter_tau_s: float = 0.05,
        max_vel_deg_s: float = 90.0,
        max_delta_deg: float = 12.0,
        servo_fault_hz_warn: float = ARM_RATE_FAIL_HZ,
    ) -> None:
        if source_hz <= 0:
            raise ValueError("servo source_hz 必须 > 0")
        self._follower = follower
        self._interp = interp
        self._source_hz = int(source_hz)
        self._period_s = 1.0 / float(self._source_hz)
        self._command_time_s = self._period_s
        self._watchdog_s = float(watchdog_s)
        self._lo = None if joint_lo is None else np.asarray(joint_lo, dtype=np.float64)
        self._hi = None if joint_hi is None else np.asarray(joint_hi, dtype=np.float64)
        self._tau_s = float(command_filter_tau_s)
        self._max_vel_deg_s = float(max_vel_deg_s)
        self._max_delta_deg = float(max_delta_deg)
        self._fault_hz_warn = float(servo_fault_hz_warn)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_submit_t = time.perf_counter()
        self._sent = 0
        self._warned_watchdog = False
        self._fault: str | None = None
        follow_lo, follow_hi = follower.joint_limits_deg
        self._follow_lo = np.asarray(follow_lo, dtype=np.float64)
        self._follow_hi = np.asarray(follow_hi, dtype=np.float64)
        self._cur_cmd: np.ndarray | None = None
        self._last_sent: np.ndarray | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("伺服线程已在运行")
        name = f"servo-{self._source_hz}hz-source"
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def check_fault(self) -> None:
        if self._fault:
            raise RuntimeError(self._fault)

    def _shape_command(self, desired: np.ndarray, dt: float) -> np.ndarray:
        desired = np.asarray(desired, dtype=np.float64).reshape(7)
        if self._last_sent is None:
            cmd = desired.copy()
        else:
            clamped, _ = _clamp_joints_by_max_delta(
                desired, self._last_sent, self._max_delta_deg
            )
            desired = np.asarray(clamped, dtype=np.float64)
            # 滤波可选；限速始终相对上一拍已发指令，不随 tau=0 关掉
            if self._tau_s > 0.0 and self._cur_cmd is not None:
                alpha = dt / (self._tau_s + dt)
                cmd = self._cur_cmd + alpha * (desired - self._cur_cmd)
            else:
                cmd = desired
            max_step = self._max_vel_deg_s * dt
            cmd = self._last_sent + np.clip(
                cmd - self._last_sent, -max_step, max_step
            )
        if self._lo is not None:
            cmd = np.clip(cmd, self._lo, self._hi)
        cmd = np.clip(cmd, self._follow_lo, self._follow_hi)
        self._cur_cmd = cmd.astype(np.float64, copy=False)
        self._last_sent = self._cur_cmd.copy()
        return self._cur_cmd

    def _run(self) -> None:
        try:
            next_t = time.perf_counter()
            last_stat_check_t = 0.0
            last_tick = next_t
            while not self._stop.is_set():
                t = time.perf_counter()
                dt = max(t - last_tick, 1e-4)
                last_tick = t
                target = self._interp.sample(t)
                if target is None:
                    self._stop.wait(min(self._period_s, 0.005))
                    next_t = time.perf_counter()
                    last_tick = next_t
                    continue
                cmd = self._shape_command(target, dt)
                self._follower.send_joint_angles_deg(cmd, self._command_time_s)
                self._sent += 1
                if self._watchdog_s > 0.0 and (t - self._last_submit_t) > self._watchdog_s:
                    if not self._warned_watchdog:
                        print(
                            f"[WARN] 推理看门狗：距上次提交 {t - self._last_submit_t:.2f}s "
                            "> 阈值，伺服保持最后目标",
                            flush=True,
                        )
                        self._warned_watchdog = True
                else:
                    self._warned_watchdog = False

                next_t += self._period_s
                sleep_s = next_t - time.perf_counter()
                if sleep_s > 0.0:
                    if self._stop.wait(timeout=sleep_s):
                        break
                else:
                    next_t = time.perf_counter()

                now = time.perf_counter()
                if now - last_stat_check_t > 1.0:
                    last_stat_check_t = now
                    line, running, hz = _format_servo_stats(self._follower)
                    print(f"[SERVO] {line} 源点={self._source_hz}Hz 目标={ARM_SERVO_HZ}Hz", flush=True)
                    if not running:
                        raise RuntimeError("HCX left direct-servo 输出线程已停")
                    if hz is not None and float(hz) < self._fault_hz_warn:
                        raise RuntimeError(
                            f"HCX left direct-servo 过慢: observed_rate_hz={hz:.1f} "
                            f"< {self._fault_hz_warn:.0f}"
                        )
        except BaseException as exc:
            self._fault = f"伺服下发线程异常: {exc}"
            self._stop.set()

    def mark_submit(self) -> None:
        self._last_submit_t = time.perf_counter()


def _load_servo_section(deploy_path: Path) -> dict[str, Any]:
    """``_load_deploy_config`` 只返回固定字段集，会丢弃 yaml 里的 ``servo`` 段，
    这里直接从原始 yaml 读取伺服参数。
    """
    with deploy_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    servo = raw.get("servo", {}) or {}
    if not isinstance(servo, dict):
        raise ValueError(f"deploy.yaml 的 servo 必须是映射: {deploy_path}")

    watchdog_s = float(servo.get("watchdog_s", 0.5))
    if not math.isfinite(watchdog_s) or watchdog_s < 0.0:
        raise ValueError("servo.watchdog_s 必须是 >= 0 的有限数")

    command_filter_tau_s = float(servo.get("command_filter_tau_s", 0.05))
    if not math.isfinite(command_filter_tau_s) or command_filter_tau_s < 0.0:
        raise ValueError("servo.command_filter_tau_s 必须是 >= 0 的有限数")

    max_joint_vel_deg_s = float(servo.get("max_joint_vel_deg_s", 90.0))
    if not math.isfinite(max_joint_vel_deg_s) or max_joint_vel_deg_s <= 0.0:
        raise ValueError("servo.max_joint_vel_deg_s 必须 > 0")

    servo_fault_hz_warn = float(servo.get("servo_fault_hz_warn", ARM_RATE_FAIL_HZ))
    if not math.isfinite(servo_fault_hz_warn) or servo_fault_hz_warn <= 0.0:
        raise ValueError("servo.servo_fault_hz_warn 必须 > 0")

    expected_out_hz = servo.get("rate_hz", ARM_SERVO_HZ)
    if int(expected_out_hz) != ARM_SERVO_HZ:
        raise ValueError(
            f"servo.rate_hz 必须是 {ARM_SERVO_HZ}（SDK 输出频率），实际为 {expected_out_hz}"
        )

    return {
        "watchdog_s": watchdog_s,
        "command_filter_tau_s": command_filter_tau_s,
        "max_joint_vel_deg_s": max_joint_vel_deg_s,
        "servo_fault_hz_warn": servo_fault_hz_warn,
    }


def _resolve_teleop_yaml(deploy: dict[str, Any]) -> Path:
    """优先用 deploy.yaml 的 teleop_yaml；缺失时回退到 in-repo teleop_project。"""
    p = Path(deploy["teleop_yaml"])
    if p.is_file():
        return p
    cand = TELEOP_ROOT / "teleop.yaml"
    if cand.is_file():
        print(f"[INFO] deploy teleop_yaml 不存在 {p}，回退到 {cand}", flush=True)
        return cand
    raise FileNotFoundError(f"找不到 teleop.yaml: {p}")


def main() -> None:
    args = _parse_args()
    deploy_path = _resolve_deploy_path(args.deploy)
    deploy = _load_deploy_config(deploy_path)
    teleop_yaml = _resolve_teleop_yaml(deploy)

    ckpt_path = deploy["checkpoint"]
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")
    train_cfg_path = _resolve_train_config(ckpt_path, deploy["config"])
    if train_cfg_path is not None and not train_cfg_path.is_file():
        raise FileNotFoundError(f"找不到训练配置: {train_cfg_path}")

    print(f"[INFO] 加载 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = load_config(train_cfg_path) if train_cfg_path is not None else ckpt["config"]
    stats = ckpt["stats"]
    cameras = list(cfg.cameras)
    layout = _validate_runtime_contract(cfg, cameras, stats)

    # RTC：伺服模式每步重规划，与 RTC leftover 冲突，强制关掉
    from robotfm.config import RTCConfig

    cfg.policy.rtc = _normalize_rtc_config(RTCConfig(enabled=False))

    train_history_noise = float(getattr(cfg.policy, "history_noise_std", 0.0) or 0.0)
    cfg.policy.history_noise_std = 0.0
    norm_mode = cfg.dataset.norm_mode
    n_obs = int(cfg.dataset.n_obs_steps)
    n_action_steps = int(cfg.policy.n_action_steps)
    train_fps = int(cfg.fps)
    max_steps = int(deploy["max_steps"])
    # ``_load_deploy_config`` 会丢弃 yaml 中未知的 ``servo`` 段，需从原始文件读取
    servo_cfg = _load_servo_section(deploy_path)
    watchdog_s = float(servo_cfg["watchdog_s"])
    state_dim = int(cfg.state_dim)
    if state_dim not in (LEFT_ARM_DIM, 16):
        raise ValueError(f"state_dim 应为 {LEFT_ARM_DIM} 或 16，实际 {state_dim}")

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    policy = build_policy(cfg, stats)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()
    policy_cfg = getattr(policy, "cfg", None)
    if hasattr(policy_cfg, "history_noise_std"):
        policy_cfg.history_noise_std = 0.0
    print(
        f"[INFO] 策略就绪 layout={layout} device={device} cameras={cameras} "
        f"norm={norm_mode} n_obs={n_obs} n_action_steps={n_action_steps} "
        f"train_fps={train_fps} state_dim={state_dim} "
        f"watchdog_s={watchdog_s:g}",
        flush=True,
    )

    joint_lo = np.asarray(deploy["joint_limits_min_deg"], dtype=np.float64)
    joint_hi = np.asarray(deploy["joint_limits_max_deg"], dtype=np.float64)

    hw = HardwareBundle(
        left_start_joints_deg=deploy["left_start_joints_deg"],
        right_start_joints_deg=deploy["right_start_joints_deg"],
    )
    servo = None
    left_follower: HcxFollower | None = None
    right_follower: HcxFollower | None = None
    try:
        (
            _hcx_connection,
            left_follower,
            right_follower,
            left_arm,
            right_arm,
            source_hz,
            direct_cfg,
        ) = _connect_hcx_direct(teleop_yaml)
        hw.hcx_client = _hcx_connection.client
        hw.left_arm = left_arm
        hw.right_arm = right_arm
        hw.left_gripper, hw.right_gripper, gripper_rate_hz = _connect_dual_grippers(
            teleop_yaml
        )
        if hw.left_gripper is not None:
            hold_left = _read_gripper(hw.left_gripper, fallback=1.0)
            hw.left_gripper_loop = BackgroundGripperLoop(hw.left_gripper, rate_hz=gripper_rate_hz)
            hw.left_gripper_loop.start(initial_opening=hold_left)
        if hw.right_gripper is not None:
            hold_right = _read_gripper(hw.right_gripper, fallback=1.0)
            hw.right_gripper_loop = BackgroundGripperLoop(hw.right_gripper, rate_hz=gripper_rate_hz)
            hw.right_gripper_loop.start(initial_opening=hold_right)

        hw.camera_manager = _connect_cameras(teleop_yaml, cameras)
        if deploy.get("display_cameras", True):
            hw.camera_preview = CameraPreviewLoop(hw.camera_manager, cameras, fps=PREVIEW_FPS)
            hw.camera_preview.start()
            print(f"[INFO] 相机预览已启动 cameras={cameras}", flush=True)

        # 起始位：move_joints 到位（规划模式），再进入直伺服
        if deploy["left_start_joints_deg"] is not None:
            hw.left_arm.move_joints(
                deploy["left_start_joints_deg"],
                interrupt=False,
                wait=False,
                speed_ratio=deploy["move_speed_ratio"],
                acceleration_seconds=deploy["move_acceleration_seconds"],
                deceleration_seconds=deploy["move_deceleration_seconds"],
            )
            _confirm_targets_by_feedback(
                hw,
                deploy["left_start_joints_deg"],
                None,
                timeout_s=deploy["move_feedback_confirm_timeout_s"],
                poll_interval_s=deploy["move_feedback_confirm_poll_interval_s"],
                angle_tolerance_deg=deploy["move_angle_tolerance_deg"],
            )
        if deploy["right_start_joints_deg"] is not None:
            hw.right_arm.move_joints(
                deploy["right_start_joints_deg"],
                interrupt=False,
                wait=False,
                speed_ratio=deploy["move_speed_ratio"],
                acceleration_seconds=deploy["move_acceleration_seconds"],
                deceleration_seconds=deploy["move_deceleration_seconds"],
            )
        _ramp_start_grippers(
            hw,
            left_target=deploy["left_start_gripper"],
            right_target=deploy["right_start_gripper"],
            duration_s=deploy["start_gripper_ramp_s"],
            rate_hz=gripper_rate_hz,
        )

        pre_crop_size = cfg.dataset.pre_crop_size
        resize_size = cfg.dataset.resize_size
        crop_size = cfg.dataset.crop_size
        eval_fixed_crop = bool(cfg.dataset.eval_fixed_crop)
        obs_sampler = FpsObservationSampler(
            hw,
            cameras,
            n_obs_steps=n_obs,
            fps=train_fps,
            state_dim=state_dim,
            pre_crop_size=pre_crop_size,
            resize_size=resize_size,
            crop_size=crop_size,
            eval_fixed_crop=eval_fixed_crop,
        )
        obs_sampler.start()
        hw.obs_sampler = obs_sampler
        fill_timeout_s = max(5.0, float(n_obs) / float(train_fps) + 3.0)
        print(
            f"[INFO] 观测采样已启动 fps={train_fps} n_obs={n_obs} "
            f"pre_crop={pre_crop_size} resize={resize_size} crop={crop_size} "
            f"fill_timeout_s={fill_timeout_s:.1f}",
            flush=True,
        )
        obs_sampler.wait_until_filled(timeout_s=fill_timeout_s)
        print(f"[INFO] 观测缓冲已就绪 frames={n_obs} fps={train_fps}", flush=True)

        if not left_follower.start_servo():
            raise RuntimeError("HCX left direct-servo 启动失败")
        print(
            f"[INFO] 左臂 direct-servo 已启动（SDK {direct_cfg.rate_hz}Hz 输出，"
            f"插值={direct_cfg.interpolation}）",
            flush=True,
        )

        interp = ServoInterpolator(source_fps=train_fps, n_joints=7)
        servo = ServoOutputThread(
            left_follower,
            interp,
            source_hz=source_hz,
            watchdog_s=watchdog_s,
            joint_lo=joint_lo,
            joint_hi=joint_hi,
            command_filter_tau_s=float(servo_cfg["command_filter_tau_s"]),
            max_vel_deg_s=float(servo_cfg["max_joint_vel_deg_s"]),
            max_delta_deg=float(deploy["move_max_delta_deg"]),
            servo_fault_hz_warn=float(servo_cfg["servo_fault_hz_warn"]),
        )
        servo.start()
        print(
            f"[INFO] 伺服下发线程已启动 source_hz={source_hz} → {ARM_SERVO_HZ}Hz，"
            f"开始 {train_fps}Hz 推理 + {source_hz}Hz 源点下发",
            flush=True,
        )

        joint_mask = joint_mask_from_names(cfg.action_names, cfg.action_dim)
        # 推理耗时统计：推理必须远小于 chunk 覆盖时间（n_action_steps / fps），
        # 否则伺服会周期性停在 chunk 末尾等新数据（走走停停）
        chunk_cover_s = n_action_steps / train_fps
        infer_times: list[float] = []
        infer_win_start = time.perf_counter()
        infer_win_n = 0
        for step_i in range(max_steps):
            servo.check_fault()
            t0 = time.perf_counter()
            obs_history = obs_sampler.snapshot()
            batch = _build_obs_batch(
                obs_history,
                cameras=cameras,
                n_obs_steps=n_obs,
                stats=stats,
                norm_mode=norm_mode,
                device=device,
            )
            if step_i < 3 or step_i % _LOG_STATE_INPUT_EVERY == 0:
                _log_inference_state_input(
                    step_i=step_i,
                    obs_history=obs_history,
                    n_obs_steps=n_obs,
                    batch=batch,
                )
            infer_t0 = time.perf_counter()
            with torch.no_grad():
                pred = policy.sample_actions(batch)[0].cpu()
            infer_s = time.perf_counter() - infer_t0
            infer_times.append(infer_s)
            if len(infer_times) > 100:
                infer_times.pop(0)
            infer_win_n += 1
            if infer_s > chunk_cover_s * 0.8:
                print(
                    f"[WARN] 推理耗时 {infer_s * 1e3:.0f}ms 超过 chunk 覆盖时间 "
                    f"{chunk_cover_s * 1e3:.0f}ms 的 80%，伺服将走走停停",
                    flush=True,
                )
            if time.perf_counter() - infer_win_start >= 5.0:
                arr = np.asarray(infer_times, dtype=np.float64)
                print(
                    f"[INFER] 推理p50={float(np.median(arr)) * 1e3:.0f}ms "
                    f"p95={float(np.percentile(arr, 95)) * 1e3:.0f}ms "
                    f"max={float(arr.max()) * 1e3:.0f}ms chunk覆盖={chunk_cover_s * 1e3:.0f}ms "
                    f"steps={infer_win_n}",
                    flush=True,
                )
                infer_win_start = time.perf_counter()
                infer_win_n = 0
            pred_phys = np.asarray(
                denormalize_predicted_action(
                    pred,
                    stats,
                    norm_mode,
                    q_now_phys=np.asarray(obs_history[-1].state, dtype=np.float32),
                    predict_joint_delta=bool(cfg.policy.predict_joint_delta),
                    joint_mask=joint_mask,
                )
            )
            if step_i < 3 or step_i % _LOG_RESULT_EVERY == 0:
                _log_inference_result(
                    step_i=step_i, pred_norm=pred, pred_phys=pred_phys
                )
            # chunk = 未来 n_action_steps 步绝对关节目标（前 7 维关节）
            chunk_joints = np.asarray(pred_phys[:, :7], dtype=np.float64)
            # 首帧从当前物理关节角过渡到 chunk[0]，避免跳变。
            # q_now 直接复用 obs-sampler 后台线程采到的最新关节角（obs_history[-1].state），
            # 不再每步同步读关节（网络往返）——该值只在 step 0 用到。
            if step_i == 0:
                q_now = np.asarray(obs_history[-1].state, dtype=np.float64)[:7]
                chunk_joints = np.vstack([q_now[None, :], chunk_joints])
            interp.submit(chunk_joints, t0)
            servo.mark_submit()
            # 夹爪：取 chunk[0] 的 gripper 维（若存在）
            if pred_phys.shape[1] >= 8:
                left_g = float(np.clip(pred_phys[0, 7], 0.0, 1.0))
                if hw.left_gripper_loop is not None:
                    hw.left_gripper_loop.set_opening(left_g)
            _pace_step(t0, train_fps)
        servo.check_fault()
    except KeyboardInterrupt:
        print("\n[INFO] 收到 Ctrl+C，停止伺服", flush=True)
    finally:
        if servo is not None:
            servo.stop()
        if left_follower is not None:
            try:
                left_follower.stop_servo()
            except Exception:
                pass
            try:
                left_follower.disconnect()
            except Exception as exc:
                print(f"[WARN] 断开左臂 follower 出错: {exc}", flush=True)
        if right_follower is not None:
            try:
                right_follower.disconnect()
            except Exception as exc:
                print(f"[WARN] 断开右臂 follower 出错: {exc}", flush=True)
        _shutdown(hw)


if __name__ == "__main__":
    main()
