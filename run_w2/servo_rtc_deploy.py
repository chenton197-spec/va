#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

VA_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DEPLOY_YAML = SCRIPT_DIR / "servo_rtc_deploy.yaml"
TELEOP_ROOT = VA_ROOT.parent / "teleop_project"
if not TELEOP_ROOT.is_dir():
    raise FileNotFoundError(f"找不到 teleop_project: {TELEOP_ROOT}")

for p in (TELEOP_ROOT, VA_ROOT, SCRIPT_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from robotfm.config import _normalize_rtc_config, load_config
from robotfm.data.action_delta import (
    denormalize_predicted_action,
    joint_mask_from_names,
    subtract_joint_pose,
)
from robotfm.data.stats import normalize
from robotfm.policies.rtc import ActionQueue
from robotfm.train import build_policy
from run import (
    BackgroundGripperLoop,
    CameraPreviewLoop,
    DUAL_ARM_CAMERAS,
    HardwareBundle,
    PREVIEW_FPS,
    _apply_rtc_overrides,
    _clamp_joints_by_limits,
    _clamp_joints_by_max_delta,
    _confirm_targets_by_feedback,
    _connect_dual_grippers,
    _load_deploy_config,
    _log_inference_result,
    _log_inference_state_input,
    _pump_camera_preview,
    _ramp_start_grippers,
    _read_gripper,
    _read_hcx_joints,
    _resolve_train_config,
    _shutdown,
    _validate_runtime_contract,
)
from run_dual_arm_depth import (
    HeadTriggeredCapture,
    _build_obs_batch,
    _connect_record_cameras,
    _prepare_rgbd_observation,
    _read_dual_observation,
)
from teleop_sdk.adapters.hcx import (
    HcxConnection,
    HcxConnectionConfig,
    HcxDirectServoConfig,
    HcxFollower,
)
from teleop_sdk.config import load_runtime_config

ARM_SERVO_HZ = 500
ARM_RATE_FAIL_HZ = 450.0
_LOG_STATE_INPUT_EVERY = 25
_LOG_RESULT_EVERY = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W2 双臂 RTC 直伺服部署")
    parser.add_argument("--deploy", type=str, default=str(DEFAULT_DEPLOY_YAML))
    return parser.parse_args()


def _resolve_deploy_path(value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path.resolve()
    if path.is_absolute():
        return path
    for cand in (VA_ROOT / path, SCRIPT_DIR / path, SCRIPT_DIR / path.name):
        if cand.is_file():
            return cand.resolve()
    return (VA_ROOT / path).resolve()


def _load_extra_deploy(deploy_path: Path) -> dict[str, Any]:
    with deploy_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    servo = raw.get("servo", {}) or {}
    if not isinstance(servo, dict):
        raise ValueError(f"deploy.yaml 的 servo 必须是映射: {deploy_path}")
    watchdog_s = float(servo.get("watchdog_s", 0.5))
    if not math.isfinite(watchdog_s) or watchdog_s < 0.0:
        raise ValueError("servo.watchdog_s 必须是 >= 0 的有限数")
    command_filter_tau_s = float(servo.get("command_filter_tau_s", 0.0))
    if not math.isfinite(command_filter_tau_s) or command_filter_tau_s < 0.0:
        raise ValueError("servo.command_filter_tau_s 必须是 >= 0 的有限数")
    max_joint_vel_deg_s = float(servo.get("max_joint_vel_deg_s", 90.0))
    if not math.isfinite(max_joint_vel_deg_s) or max_joint_vel_deg_s <= 0.0:
        raise ValueError("servo.max_joint_vel_deg_s 必须 > 0")
    servo_fault_hz_warn = float(servo.get("servo_fault_hz_warn", ARM_RATE_FAIL_HZ))
    if not math.isfinite(servo_fault_hz_warn) or servo_fault_hz_warn <= 0.0:
        raise ValueError("servo.servo_fault_hz_warn 必须 > 0")
    if int(servo.get("rate_hz", ARM_SERVO_HZ)) != ARM_SERVO_HZ:
        raise ValueError(
            f"servo.rate_hz 必须是 {ARM_SERVO_HZ}，实际为 {servo.get('rate_hz')}"
        )
    threshold = raw.get("action_queue_size_to_get_new_actions")
    if threshold in (None, ""):
        raise ValueError("deploy.yaml 需要 action_queue_size_to_get_new_actions")
    threshold_i = int(threshold)
    if threshold_i <= 0:
        raise ValueError("action_queue_size_to_get_new_actions 必须 > 0")
    return {
        "watchdog_s": watchdog_s,
        "command_filter_tau_s": command_filter_tau_s,
        "max_joint_vel_deg_s": max_joint_vel_deg_s,
        "servo_fault_hz_warn": servo_fault_hz_warn,
        "action_queue_size_to_get_new_actions": threshold_i,
    }


def _direct_servo_config(teleop_yaml: Path) -> tuple[Any, HcxDirectServoConfig]:
    runtime = load_runtime_config(teleop_yaml)
    h = runtime.hcx
    if int(h.direct_servo_rate_hz) != ARM_SERVO_HZ:
        raise RuntimeError(
            f"teleop.yaml hcx.direct_servo_rate_hz 必须是 {ARM_SERVO_HZ}，"
            f"实际为 {h.direct_servo_rate_hz}"
        )
    if not bool(h.direct_servo_confirm_unsafe):
        raise RuntimeError("直伺服要求 hcx.direct_servo_confirm_unsafe: true")
    direct_cfg = HcxDirectServoConfig(
        rate_hz=ARM_SERVO_HZ,
        watchdog_s=float(h.direct_servo_watchdog_s),
        confirm_unsafe=True,
        interpolation="direct",
        source_rate_hz=None,
    )
    if direct_cfg.watchdog_s <= 1.0 / float(ARM_SERVO_HZ):
        raise RuntimeError("hcx.direct_servo_watchdog_s 必须大于一个 500 Hz 周期")
    return runtime, direct_cfg


def _connect_hcx_direct(
    teleop_yaml: Path,
) -> tuple[HcxConnection, HcxFollower, HcxFollower, Any, Any, HcxDirectServoConfig]:
    runtime, direct_cfg = _direct_servo_config(teleop_yaml)
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
    for rid, label in (
        (int(h.left_robot_id), "左臂"),
        (int(h.right_robot_id), "右臂"),
    ):
        if not connection.prepare_for_motion(rid):
            raise RuntimeError(f"HCX {label} prepare_for_motion 失败")
        if not connection.motion_ready(rid):
            raise RuntimeError(f"HCX {label} 未处于可运动状态")
    client = connection.client
    if client is None:
        raise RuntimeError("HCX 连接未建立")
    print(
        f"[INFO] HCX 直伺服已连接: out={direct_cfg.rate_hz} Hz "
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
        direct_cfg,
    )


def _follower_stats_line(name: str, follower: HcxFollower) -> tuple[str, bool, float | None]:
    stats = follower.direct_servo_output_stats()
    if stats is None:
        return f"{name}=no-stats", False, None
    running = bool(stats.running)
    hz = stats.observed_rate_hz
    hz_s = "empty-window" if hz is None else f"{float(hz):.1f}"
    run_s = "run" if running else "STOP"
    return f"{name}={hz_s}Hz {run_s}", running, None if hz is None else float(hz)


class LatencyTracker:
    def __init__(self, maxlen: int = 32) -> None:
        self._vals: deque[float] = deque(maxlen=int(maxlen))

    def add(self, latency_s: float) -> None:
        self._vals.append(float(latency_s))

    def max(self) -> float:
        if not self._vals:
            return 0.0
        return float(max(self._vals))


class ServoSendQueue:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._q: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self._last: tuple[np.ndarray, np.ndarray] | None = None
        self._popped = 0

    def replace(self, points: list[tuple[np.ndarray, np.ndarray]]) -> None:
        with self._lock:
            self._q = deque(points)
            self._popped = 0

    def pop(self) -> tuple[np.ndarray, np.ndarray] | None:
        with self._lock:
            if self._q:
                item = self._q.popleft()
                self._last = item
                self._popped += 1
                return item
            return self._last

    def qsize(self) -> int:
        with self._lock:
            return len(self._q)

    def popped(self) -> int:
        with self._lock:
            return self._popped

    def last(self) -> tuple[np.ndarray, np.ndarray] | None:
        with self._lock:
            return self._last


class ServoSendThread:
    def __init__(
        self,
        left: HcxFollower,
        right: HcxFollower,
        send_q: ServoSendQueue,
        *,
        rate_hz: int,
    ) -> None:
        self._left = left
        self._right = right
        self._q = send_q
        self._dt = 1.0 / float(rate_hz)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fault: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("伺服下发线程已在运行")
        self._thread = threading.Thread(target=self._run, name="servo-500hz", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def check_fault(self) -> None:
        if self._fault:
            raise RuntimeError(self._fault)

    def _run(self) -> None:
        try:
            next_t = time.perf_counter()
            while not self._stop.is_set():
                item = self._q.pop()
                if item is not None:
                    left, right = item
                    self._left.send_joint_angles_deg(left, self._dt)
                    self._right.send_joint_angles_deg(right, self._dt)
                next_t += self._dt
                sleep_s = next_t - time.perf_counter()
                if sleep_s > 0.0:
                    if self._stop.wait(timeout=sleep_s):
                        break
                else:
                    next_t = time.perf_counter()
        except BaseException as exc:
            self._fault = f"伺服下发线程异常: {exc}"
            self._stop.set()


class ServoWatchdogThread:
    def __init__(
        self,
        left: HcxFollower,
        right: HcxFollower,
        *,
        fault_hz: float,
    ) -> None:
        self._left = left
        self._right = right
        self._fault_hz = float(fault_hz)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fault: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="servo-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def check_fault(self) -> None:
        if self._fault:
            raise RuntimeError(self._fault)

    def _run(self) -> None:
        try:
            while not self._stop.wait(timeout=1.0):
                parts = []
                for name, fol in (("left", self._left), ("right", self._right)):
                    line, running, hz = _follower_stats_line(name, fol)
                    parts.append(line)
                    if not running:
                        raise RuntimeError(f"HCX {name} direct-servo 输出线程已停")
                    if hz is not None and float(hz) < self._fault_hz:
                        raise RuntimeError(
                            f"HCX {name} direct-servo 过慢: observed_rate_hz={hz:.1f} "
                            f"< {self._fault_hz:.0f}"
                        )
                print(f"[SERVO] {' '.join(parts)} 目标={ARM_SERVO_HZ}Hz", flush=True)
        except BaseException as exc:
            if self._stop.is_set():
                return
            self._fault = f"伺服看门狗: {exc}"
            self._stop.set()


class FpsRgbDObservationSampler:
    def __init__(
        self,
        hw: HardwareBundle,
        cameras: list[str],
        *,
        n_obs_steps: int,
        fps: int,
        image_size: int | list[int] | None,
        depth_cameras: list[str],
        depth_min_mm: float,
        depth_max_mm: float,
        expected_scale_mm: dict[str, float],
    ) -> None:
        if n_obs_steps <= 0:
            raise ValueError("n_obs_steps 必须 > 0")
        if fps <= 0:
            raise ValueError("fps 必须 > 0")
        self._hw = hw
        self._cameras = list(cameras)
        self._n_obs_steps = int(n_obs_steps)
        self._fps = int(fps)
        self._period_s = 1.0 / float(fps)
        self._image_size = image_size
        self._depth_cameras = list(depth_cameras)
        self._depth_min_mm = float(depth_min_mm)
        self._depth_max_mm = float(depth_max_mm)
        self._expected_scale_mm = dict(expected_scale_mm)
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=self._n_obs_steps)
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("观测采样已在运行")
        self._thread = threading.Thread(target=self._run, name="obs-rgbd-fps", daemon=True)
        self._thread.start()

    def stop(self, *, join_timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=join_timeout_s)
        self._thread = None

    def snapshot(self) -> list:
        with self._lock:
            self._raise_if_locked_unhealthy()
            return list(self._history)

    def raise_if_unhealthy(self) -> None:
        with self._lock:
            self._raise_if_locked_unhealthy()

    def _raise_if_locked_unhealthy(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"{self._fps}fps RGB-D 观测采样失败") from self._error
        if not self._history:
            raise RuntimeError(f"{self._fps}fps RGB-D 观测缓冲为空")

    def wait_until_filled(self, *, timeout_s: float) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("观测缓冲等待超时必须是正的有限秒数")
        deadline = time.perf_counter() + timeout_s
        size = 0
        while time.perf_counter() < deadline:
            with self._lock:
                err = self._error
                size = len(self._history)
            if err is not None:
                raise RuntimeError(f"{self._fps}fps RGB-D 观测采样失败") from err
            if size >= self._n_obs_steps:
                return
            if self._stop.wait(timeout=0.01):
                raise RuntimeError("观测采样在缓冲填满前已停止")
        raise TimeoutError(
            f"等待 {self._n_obs_steps} 帧 {self._fps}fps 观测超时 ({timeout_s:.1f}s)，"
            f"当前 {size} 帧"
        )

    def _run(self) -> None:
        next_t = time.perf_counter()
        last_state: np.ndarray | None = None
        while not self._stop.is_set():
            try:
                raw = _read_dual_observation(
                    self._hw, self._cameras, last_state=last_state
                )
                raw.validate(self._cameras, int(raw.state.shape[0]))
                obs = _prepare_rgbd_observation(
                    raw,
                    image_size=self._image_size,
                    depth_cameras=self._depth_cameras,
                    depth_min_mm=self._depth_min_mm,
                    depth_max_mm=self._depth_max_mm,
                    expected_scale_mm=self._expected_scale_mm,
                )
                last_state = np.asarray(obs.state, dtype=np.float32)
                with self._lock:
                    self._history.append(obs)
                    self._error = None
            except Exception as exc:
                with self._lock:
                    self._error = exc
            next_t += self._period_s
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0.0:
                if self._stop.wait(timeout=sleep_s):
                    break
            else:
                next_t = time.perf_counter()


def _policy_index_from_pops(popped: int, train_fps: float, servo_hz: int) -> int:
    return int(float(popped) * float(train_fps) / float(servo_hz))


def _sync_action_queue_index(
    action_queue: ActionQueue, popped: int, train_fps: float, servo_hz: int
) -> int:
    target = _policy_index_from_pops(popped, train_fps, servo_hz)
    with action_queue.lock:
        if action_queue.original_queue is None:
            action_queue.last_index = 0
            return 0
        n = int(action_queue.original_queue.shape[0])
        action_queue.last_index = max(0, min(target, n))
        return action_queue.last_index


def _interpolate_policy_to_servo(
    phys: np.ndarray,
    *,
    train_fps: float,
    servo_hz: int,
    skip_policy_steps: int,
) -> np.ndarray:
    pts = np.asarray(phys, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 1:
        raise ValueError(f"策略动作形状异常: {pts.shape}")
    skip = max(0, min(int(skip_policy_steps), int(pts.shape[0]) - 1))
    remain = pts[skip:]
    n_pol = int(remain.shape[0])
    n_hz = max(1, int(round(float(n_pol) * float(servo_hz) / float(train_fps))))
    if n_pol == 1:
        return np.repeat(remain, n_hz, axis=0)
    x = np.linspace(0.0, float(n_pol - 1), n_hz, dtype=np.float64)
    i0 = np.floor(x).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_pol - 1)
    a = (x - i0.astype(np.float64))[:, None]
    return remain[i0] * (1.0 - a) + remain[i1] * a


def _shape_arm_traj(
    desired: np.ndarray,
    *,
    last: np.ndarray | None,
    lo: np.ndarray,
    hi: np.ndarray,
    follow_lo: np.ndarray,
    follow_hi: np.ndarray,
    tau_s: float,
    max_vel_deg_s: float,
    max_delta_deg: float,
    dt: float,
) -> np.ndarray:
    n = int(desired.shape[0])
    out = np.empty((n, 7), dtype=np.float64)
    cur = None if last is None else np.asarray(last, dtype=np.float64).reshape(7)
    filt = None if cur is None else cur.copy()
    max_step = float(max_vel_deg_s) * float(dt)
    for i in range(n):
        des = np.asarray(desired[i], dtype=np.float64).reshape(7)
        if cur is None:
            cmd = np.clip(des, lo, hi)
            cmd = np.clip(cmd, follow_lo, follow_hi)
            cur = cmd
            filt = cmd.copy()
            out[i] = cmd
            continue
        clamped, _ = _clamp_joints_by_max_delta(des, cur, max_delta_deg)
        des = np.asarray(clamped, dtype=np.float64)
        if tau_s > 0.0 and filt is not None:
            alpha = dt / (tau_s + dt)
            des = filt + alpha * (des - filt)
            filt = des
        cmd = cur + np.clip(des - cur, -max_step, max_step)
        cmd = np.clip(cmd, lo, hi)
        cmd = np.clip(cmd, follow_lo, follow_hi)
        cur = cmd
        out[i] = cmd
    return out


def _reanchor_leftover(
    leftover_abs: torch.Tensor,
    q_now: np.ndarray,
    *,
    stats: dict,
    norm_mode: str,
    joint_mask: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    abs_np = leftover_abs.detach().cpu().numpy().astype(np.float32)
    delta = subtract_joint_pose(abs_np, q_now.astype(np.float32), joint_mask)
    normed = normalize(delta, stats, prefix="action", mode=norm_mode)
    return torch.from_numpy(np.asarray(normed, dtype=np.float32)).to(device)


def main() -> None:
    args = _parse_args()
    deploy_path = _resolve_deploy_path(args.deploy)
    deploy = _load_deploy_config(deploy_path)
    extra = _load_extra_deploy(deploy_path)
    teleop_yaml = deploy["teleop_yaml"]
    if not Path(teleop_yaml).is_file():
        cand = TELEOP_ROOT / "teleop.yaml"
        if cand.is_file():
            teleop_yaml = cand
        else:
            raise FileNotFoundError(f"找不到 teleop.yaml: {teleop_yaml}")

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
    if layout != "dual" or tuple(cameras) != DUAL_ARM_CAMERAS:
        raise ValueError(
            f"本脚本仅支持双臂 cameras={list(DUAL_ARM_CAMERAS)}，"
            f"实际 cameras={cameras} layout={layout}"
        )

    _apply_rtc_overrides(cfg, deploy.get("rtc") or {})
    rtc_cfg = _normalize_rtc_config(cfg.policy.rtc)
    if not bool(rtc_cfg.enabled):
        raise ValueError("本脚本要求 rtc.enabled=true")
    if str(deploy.get("obs_mode") or "fps").strip().lower() != "fps":
        raise ValueError("本脚本要求 obs_mode=fps")

    train_history_noise = float(getattr(cfg.policy, "history_noise_std", 0.0) or 0.0)
    cfg.policy.history_noise_std = 0.0
    norm_mode = cfg.dataset.norm_mode
    n_obs = int(cfg.dataset.n_obs_steps)
    n_action_steps = int(cfg.policy.n_action_steps)
    horizon = int(cfg.dataset.horizon)
    train_fps = float(cfg.fps)
    max_steps = int(deploy["max_steps"])
    threshold = int(extra["action_queue_size_to_get_new_actions"])
    exec_h = int(rtc_cfg.execution_horizon)
    if exec_h >= n_action_steps:
        raise ValueError(
            f"RTC execution_horizon 必须小于 n_action_steps "
            f"(execution_horizon={exec_h}, n_action_steps={n_action_steps})"
        )
    if not (exec_h < threshold < n_action_steps):
        raise ValueError(
            "action_queue_size_to_get_new_actions 须满足 "
            f"execution_horizon < threshold < n_action_steps "
            f"(got execution_horizon={exec_h}, threshold={threshold}, "
            f"n_action_steps={n_action_steps})"
        )
    policy_type = str(cfg.policy.type).lower().replace("-", "_")
    if policy_type == "a2au":
        policy_type = "a2a_u"
    if policy_type in {"a2a", "n_a2a", "a2a_u"} and n_action_steps != horizon:
        raise ValueError(
            "A2A RTC 要求 n_action_steps == horizon "
            f"(got n_action_steps={n_action_steps}, horizon={horizon})"
        )

    depth_cameras = list(getattr(cfg.dataset, "depth_cameras", ()) or ())
    expected_scale_mm = dict(getattr(cfg.dataset, "scale_mm_per_raw_unit", None) or {})
    if not depth_cameras:
        raise ValueError("本脚本需要 dataset.depth_cameras")
    for cam in depth_cameras:
        if cam not in expected_scale_mm:
            raise ValueError(f"训练配置缺少 dataset.scale_mm_per_raw_unit[{cam}]")

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    policy = build_policy(cfg, stats)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()
    policy_cfg = getattr(policy, "cfg", None)
    if hasattr(policy_cfg, "history_noise_std"):
        policy_cfg.history_noise_std = 0.0

    predict_joint_delta = bool(cfg.policy.predict_joint_delta)
    joint_mask = joint_mask_from_names(cfg.action_names, cfg.action_dim)
    joint_idx = np.where(joint_mask)[0]
    if joint_idx.size != 14:
        raise ValueError(f"双臂关节维数应为 14，实际 {joint_idx.size}")
    left_j = joint_idx[:7]
    right_j = joint_idx[7:14]
    grip_idx = np.where(~joint_mask)[0]
    left_g_i = int(grip_idx[0]) if grip_idx.size >= 1 else 14
    right_g_i = int(grip_idx[1]) if grip_idx.size >= 2 else 15

    print(
        f"[INFO] 策略就绪 layout=dual device={device} cameras={cameras} "
        f"norm={norm_mode} n_obs={n_obs} n_action_steps={n_action_steps} "
        f"train_fps={train_fps:g} rtc.guidance={rtc_cfg.guidance_enabled} "
        f"exec_h={exec_h} threshold={threshold} "
        f"history_noise_std=0 (train={train_history_noise:g})",
        flush=True,
    )

    joint_lo = np.asarray(deploy["joint_limits_min_deg"], dtype=np.float64)
    joint_hi = np.asarray(deploy["joint_limits_max_deg"], dtype=np.float64)
    hw = HardwareBundle(
        left_start_joints_deg=deploy["left_start_joints_deg"],
        right_start_joints_deg=deploy["right_start_joints_deg"],
    )
    send_thread: ServoSendThread | None = None
    watchdog: ServoWatchdogThread | None = None
    left_follower: HcxFollower | None = None
    right_follower: HcxFollower | None = None
    obs_sampler: FpsRgbDObservationSampler | None = None
    try:
        (
            _hcx_connection,
            left_follower,
            right_follower,
            left_arm,
            right_arm,
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
            hw.left_gripper_loop = BackgroundGripperLoop(
                hw.left_gripper, rate_hz=gripper_rate_hz
            )
            hw.left_gripper_loop.start(initial_opening=hold_left)
        if hw.right_gripper is not None:
            hold_right = _read_gripper(hw.right_gripper, fallback=1.0)
            hw.right_gripper_loop = BackgroundGripperLoop(
                hw.right_gripper, rate_hz=gripper_rate_hz
            )
            hw.right_gripper_loop.start(initial_opening=hold_right)

        hw.camera_manager = _connect_record_cameras(teleop_yaml)
        hw.camera_capture = HeadTriggeredCapture(
            hw.camera_manager,
            depth_cameras=depth_cameras,
            expected_scale_mm=expected_scale_mm,
        )
        if deploy["display_cameras"]:
            hw.camera_preview = CameraPreviewLoop(
                hw.camera_manager, list(cameras), fps=PREVIEW_FPS
            )
            hw.camera_preview.start()

        if deploy["left_start_joints_deg"] is not None:
            left_start = _clamp_joints_by_limits(
                deploy["left_start_joints_deg"],
                joint_lo,
                joint_hi,
                side="start_pose.left",
            )
            hw.left_arm.move_joints(
                left_start,
                interrupt=False,
                wait=False,
                speed_ratio=deploy["move_speed_ratio"],
                acceleration_seconds=deploy["move_acceleration_seconds"],
                deceleration_seconds=deploy["move_deceleration_seconds"],
            )
            _confirm_targets_by_feedback(
                hw,
                left_start,
                _read_hcx_joints(hw.right_arm, hw.right_start_joints_deg)
                .astype(float)
                .tolist(),
                timeout_s=deploy["move_feedback_confirm_timeout_s"],
                poll_interval_s=deploy["move_feedback_confirm_poll_interval_s"],
                angle_tolerance_deg=deploy["move_angle_tolerance_deg"],
            )
        if deploy["right_start_joints_deg"] is not None:
            right_start = _clamp_joints_by_limits(
                deploy["right_start_joints_deg"],
                joint_lo,
                joint_hi,
                side="start_pose.right",
            )
            hw.right_arm.move_joints(
                right_start,
                interrupt=False,
                wait=False,
                speed_ratio=deploy["move_speed_ratio"],
                acceleration_seconds=deploy["move_acceleration_seconds"],
                deceleration_seconds=deploy["move_deceleration_seconds"],
            )
            _confirm_targets_by_feedback(
                hw,
                _read_hcx_joints(hw.left_arm, hw.left_start_joints_deg)
                .astype(float)
                .tolist(),
                right_start,
                timeout_s=deploy["move_feedback_confirm_timeout_s"],
                poll_interval_s=deploy["move_feedback_confirm_poll_interval_s"],
                angle_tolerance_deg=deploy["move_angle_tolerance_deg"],
            )

        _ramp_start_grippers(
            hw,
            left_target=deploy["left_start_gripper"],
            right_target=deploy["right_start_gripper"],
            duration_s=deploy["start_gripper_ramp_s"],
            rate_hz=gripper_rate_hz,
        )

        obs_sampler = FpsRgbDObservationSampler(
            hw,
            list(cameras),
            n_obs_steps=n_obs,
            fps=int(train_fps),
            image_size=cfg.dataset.image_size,
            depth_cameras=depth_cameras,
            depth_min_mm=float(cfg.dataset.depth_min_mm),
            depth_max_mm=float(cfg.dataset.depth_max_mm),
            expected_scale_mm=expected_scale_mm,
        )
        obs_sampler.start()
        hw.obs_sampler = obs_sampler
        fill_timeout_s = max(5.0, float(n_obs) / float(train_fps) + 3.0)
        print(
            f"[INFO] 观测采样已启动 fps={int(train_fps)} n_obs={n_obs} "
            f"fill_timeout_s={fill_timeout_s:.1f}",
            flush=True,
        )
        obs_sampler.wait_until_filled(timeout_s=fill_timeout_s)
        print("[INFO] 观测缓冲已就绪", flush=True)

        if not left_follower.start_servo():
            raise RuntimeError("HCX left direct-servo 启动失败")
        if not right_follower.start_servo():
            raise RuntimeError("HCX right direct-servo 启动失败")
        print(
            f"[INFO] 双臂 direct-servo 已启动 interpolation={direct_cfg.interpolation} "
            f"out={direct_cfg.rate_hz}Hz",
            flush=True,
        )

        follow_l_lo, follow_l_hi = left_follower.joint_limits_deg
        follow_r_lo, follow_r_hi = right_follower.joint_limits_deg
        follow_l_lo = np.asarray(follow_l_lo, dtype=np.float64)
        follow_l_hi = np.asarray(follow_l_hi, dtype=np.float64)
        follow_r_lo = np.asarray(follow_r_lo, dtype=np.float64)
        follow_r_hi = np.asarray(follow_r_hi, dtype=np.float64)
        dt = 1.0 / float(ARM_SERVO_HZ)
        tau_s = float(extra["command_filter_tau_s"])
        max_vel = float(extra["max_joint_vel_deg_s"])
        max_delta = float(deploy["move_max_delta_deg"])

        send_q = ServoSendQueue()
        send_thread = ServoSendThread(
            left_follower, right_follower, send_q, rate_hz=ARM_SERVO_HZ
        )
        send_thread.start()
        watchdog = ServoWatchdogThread(
            left_follower,
            right_follower,
            fault_hz=float(extra["servo_fault_hz_warn"]),
        )
        watchdog.start()
        print("[INFO] 500Hz 下发线程已启动（只 pop+send）", flush=True)

        action_queue = ActionQueue(rtc_cfg)
        latency_tracker = LatencyTracker()
        infer_i = 0
        while infer_i < max_steps:
            send_thread.check_fault()
            watchdog.check_fault()
            obs_sampler.raise_if_unhealthy()
            _pump_camera_preview(hw)
            _sync_action_queue_index(
                action_queue, send_q.popped(), train_fps, ARM_SERVO_HZ
            )
            if infer_i > 0 and action_queue.qsize() > threshold:
                time.sleep(0.002)
                continue

            idx_before = action_queue.get_action_index()
            leftover = action_queue.get_left_over()
            if leftover is not None and leftover.shape[0] == 0:
                leftover = None
            infer_delay = int(math.ceil(latency_tracker.max() * train_fps))
            obs_history = obs_sampler.snapshot()
            q_now = np.asarray(obs_history[-1].state, dtype=np.float32)
            batch = _build_obs_batch(
                obs_history,
                cameras=list(cameras),
                n_obs_steps=n_obs,
                stats=stats,
                norm_mode=norm_mode,
                device=device,
                depth_cameras=depth_cameras,
                predict_joint_delta=predict_joint_delta,
                joint_mask=joint_mask,
            )
            if leftover is not None:
                leftover = leftover.to(device)
                if predict_joint_delta:
                    abs_left = action_queue.get_processed_left_over()
                    if abs_left is not None and abs_left.shape[0] > 0:
                        leftover = _reanchor_leftover(
                            abs_left,
                            q_now,
                            stats=stats,
                            norm_mode=norm_mode,
                            joint_mask=joint_mask,
                            device=device,
                        )
            if infer_i < 3 or infer_i % _LOG_STATE_INPUT_EVERY == 0:
                _log_inference_state_input(
                    step_i=infer_i,
                    obs_history=obs_history,
                    n_obs_steps=n_obs,
                    batch=batch,
                )
            t_infer = time.perf_counter()
            with torch.no_grad():
                pred = policy.sample_actions(
                    batch,
                    prev_chunk_left_over=leftover,
                    inference_delay=infer_delay,
                    execution_horizon=exec_h,
                )[0].cpu()
            infer_s = time.perf_counter() - t_infer
            latency_tracker.add(infer_s)
            idx_after = _sync_action_queue_index(
                action_queue, send_q.popped(), train_fps, ARM_SERVO_HZ
            )
            new_delay = max(
                int(math.ceil(infer_s * train_fps)),
                max(0, idx_after - idx_before),
            )
            processed = denormalize_predicted_action(
                pred,
                stats,
                norm_mode,
                q_now_phys=q_now,
                predict_joint_delta=predict_joint_delta,
                joint_mask=joint_mask,
            )
            processed_t = torch.as_tensor(np.asarray(processed), dtype=torch.float32)
            if infer_i < 3 or infer_i % _LOG_RESULT_EVERY == 0:
                _log_inference_result(
                    step_i=infer_i, pred_norm=pred, pred_phys=np.asarray(processed)
                )
            action_queue.merge(pred, processed_t, new_delay, idx_before)
            phys = np.asarray(processed, dtype=np.float64)
            if infer_i == 0:
                q7l = q_now[left_j].astype(np.float64)
                q7r = q_now[right_j].astype(np.float64)
                row0 = phys[0].copy()
                row0[left_j] = q7l
                row0[right_j] = q7r
                phys = np.vstack([row0[None, :], phys])
                skip = 0
            else:
                skip = new_delay
            hz_phys = _interpolate_policy_to_servo(
                phys,
                train_fps=train_fps,
                servo_hz=ARM_SERVO_HZ,
                skip_policy_steps=skip,
            )
            left_des = hz_phys[:, left_j]
            right_des = hz_phys[:, right_j]
            last_item = send_q.last()
            last_left = None if last_item is None else last_item[0]
            last_right = None if last_item is None else last_item[1]
            left_cmd = _shape_arm_traj(
                left_des,
                last=last_left,
                lo=joint_lo,
                hi=joint_hi,
                follow_lo=follow_l_lo,
                follow_hi=follow_l_hi,
                tau_s=tau_s,
                max_vel_deg_s=max_vel,
                max_delta_deg=max_delta,
                dt=dt,
            )
            right_cmd = _shape_arm_traj(
                right_des,
                last=last_right,
                lo=joint_lo,
                hi=joint_hi,
                follow_lo=follow_r_lo,
                follow_hi=follow_r_hi,
                tau_s=tau_s,
                max_vel_deg_s=max_vel,
                max_delta_deg=max_delta,
                dt=dt,
            )
            points = [
                (left_cmd[i].copy(), right_cmd[i].copy())
                for i in range(int(left_cmd.shape[0]))
            ]
            send_q.replace(points)
            g_row = phys[min(skip, phys.shape[0] - 1)]
            if hw.left_gripper_loop is not None:
                hw.left_gripper_loop.set_opening(float(np.clip(g_row[left_g_i], 0.0, 1.0)))
            if hw.right_gripper_loop is not None:
                hw.right_gripper_loop.set_opening(
                    float(np.clip(g_row[right_g_i], 0.0, 1.0))
                )
            leftover_len = 0 if leftover is None else int(leftover.shape[0])
            print(
                f"[INFO] infer={infer_i} leftover={leftover_len} delay={new_delay} "
                f"qsize={action_queue.qsize()} send={send_q.qsize()} "
                f"infer_ms={infer_s * 1e3:.0f}",
                flush=True,
            )
            infer_i += 1
    except KeyboardInterrupt:
        print("\n[INFO] 收到 Ctrl+C，停止伺服", flush=True)
    finally:
        if send_thread is not None:
            send_thread.stop()
        if watchdog is not None:
            watchdog.stop()
        if obs_sampler is not None:
            obs_sampler.stop()
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
                right_follower.stop_servo()
            except Exception:
                pass
            try:
                right_follower.disconnect()
            except Exception as exc:
                print(f"[WARN] 断开右臂 follower 出错: {exc}", flush=True)
        _shutdown(hw)


if __name__ == "__main__":
    main()
