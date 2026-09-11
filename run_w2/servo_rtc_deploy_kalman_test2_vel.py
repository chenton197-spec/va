#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import math
import multiprocessing as mp
import os
import select
import signal
import sys
import termios
import threading
import time
import tty
from collections import deque
from pathlib import Path
from queue import Empty, Full
from typing import Any

import numpy as np
import torch
import yaml

VA_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DEPLOY_YAML = SCRIPT_DIR / "servo_rtc_deploy_kalman.yaml"
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
from robotfm.types import Observation
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
)
from teleop_sdk.adapters.hcx import (
    HcxConnection,
    HcxConnectionConfig,
    HcxDirectServoConfig,
    HcxFollower,
)
from teleop_sdk.config import load_runtime_config
from teleop_sdk.filters import OneEuroFilter

ARM_SERVO_HZ = 500
ARM_RATE_FAIL_HZ = 450.0
_LOG_STATE_INPUT_EVERY = 25
_LOG_RESULT_EVERY = 5
GRIP_CLOSED = 0.2
GRIP_OPEN = 0.45
SKIP_PHASE_A = 20
SKIP_PHASE_B = 10
RAMP_ACTIONS = 15
SPEED_MAX_A = 2.0
SPEED_MAX_B = 1.5
GL_STILL_OPEN = 0.25
GL_WILL_CLOSE = 0.15
GR_IS_OPEN = 0.6
R0_LIFTED = -100.0
R0_DOWN = -90.0
PATH_IDLE = 2.0
PATH_LIFT = 8.0
DR0_LIFT = -0.4
DR0_FWD = 0.4
LEFT_HOLD_S = 1.0
LEFT_DOWN_S = 2.0
LEFT_CLOSE_HALF_S = 1.0
RIGHT_OPEN_MOVE = 2.0
RIGHT_NEAR_N = 30


class QuitKeyThread:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._quit = threading.Event()
        self._fd: int | None = None
        self._old = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            self._fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except Exception:
                    pass
            self._fd = None
            self._old = None
            return
        self._thread = threading.Thread(
            target=self._run, name="quit-key", daemon=True
        )
        self._thread.start()

    def pressed(self) -> bool:
        return self._quit.is_set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._fd is None:
            return
        if self._old is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except Exception:
                pass
        try:
            os.close(self._fd)
        except Exception:
            pass
        self._fd = None

    def _run(self) -> None:
        fd = self._fd
        if fd is None:
            return
        try:
            while not self._stop.is_set() and not self._quit.is_set():
                try:
                    tty.setcbreak(fd)
                except Exception:
                    pass
                ready, _, _ = select.select([fd], [], [], 0.1)
                if not ready:
                    continue
                data = os.read(fd, 8)
                if b"q" in data.lower():
                    self._quit.set()
                    return
        except Exception:
            return


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
    command_deadband_deg = float(servo.get("command_deadband_deg", 0.08))
    if not math.isfinite(command_deadband_deg) or command_deadband_deg < 0.0:
        raise ValueError("servo.command_deadband_deg 必须是 >= 0 的有限数")
    command_filter_tau_s = float(servo.get("command_filter_tau_s", 0.025))
    if not math.isfinite(command_filter_tau_s) or command_filter_tau_s < 0.0:
        raise ValueError("servo.command_filter_tau_s 必须是 >= 0 的有限数")
    command_filter_tau2_s = float(servo.get("command_filter_tau2_s", 0.06))
    if not math.isfinite(command_filter_tau2_s) or command_filter_tau2_s < 0.0:
        raise ValueError("servo.command_filter_tau2_s 必须是 >= 0 的有限数")
    kalman_enabled = bool(servo.get("kalman_enabled", True))
    kalman_q = float(servo.get("kalman_q", 400.0))
    if not math.isfinite(kalman_q) or kalman_q < 0.0:
        raise ValueError("servo.kalman_q 必须是 >= 0 的有限数")
    kalman_r = float(servo.get("kalman_r", 0.3))
    if not math.isfinite(kalman_r) or kalman_r <= 0.0:
        raise ValueError("servo.kalman_r 必须 > 0")
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
    chunk = raw.get("chunk")
    if chunk in (None, ""):
        raise ValueError("deploy.yaml 需要 chunk")
    chunk_i = int(chunk)
    if chunk_i < 4:
        raise ValueError("chunk 必须 >= 4")
    fps_raw = raw.get("fps")
    if fps_raw in (None, ""):
        fps_override = None
    else:
        fps_override = float(fps_raw)
        if not math.isfinite(fps_override) or fps_override < 0.0:
            raise ValueError("fps 必须是 >= 0 的有限数")
        if fps_override == 0.0:
            fps_override = None
    one_euro = raw.get("one_euro", {}) or {}
    if not isinstance(one_euro, dict):
        raise ValueError(f"deploy.yaml 的 one_euro 必须是映射: {deploy_path}")
    one_euro_enabled = bool(one_euro.get("enabled", False))
    mincutoff_hz = float(one_euro.get("mincutoff_hz", 3.0))
    if not math.isfinite(mincutoff_hz) or mincutoff_hz <= 0.0:
        raise ValueError("one_euro.mincutoff_hz 必须 > 0")
    beta = float(one_euro.get("beta", 0.05))
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("one_euro.beta 必须是 >= 0 的有限数")
    dcutoff_hz = float(one_euro.get("dcutoff_hz", 1.0))
    if not math.isfinite(dcutoff_hz) or dcutoff_hz <= 0.0:
        raise ValueError("one_euro.dcutoff_hz 必须 > 0")
    return {
        "watchdog_s": watchdog_s,
        "command_deadband_deg": command_deadband_deg,
        "command_filter_tau_s": command_filter_tau_s,
        "command_filter_tau2_s": command_filter_tau2_s,
        "kalman_enabled": kalman_enabled,
        "kalman_q": kalman_q,
        "kalman_r": kalman_r,
        "max_joint_vel_deg_s": max_joint_vel_deg_s,
        "servo_fault_hz_warn": servo_fault_hz_warn,
        "chunk": chunk_i,
        "fps": fps_override,
        "one_euro_enabled": one_euro_enabled,
        "one_euro_mincutoff_hz": mincutoff_hz,
        "one_euro_beta": beta,
        "one_euro_dcutoff_hz": dcutoff_hz,
    }


def _direct_servo_config(teleop_yaml: Path) -> tuple[Any, HcxDirectServoConfig, int]:
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
        raise RuntimeError("直伺服要求 hcx.direct_servo_confirm_unsafe: true")
    source_rate_hz = (
        source_hz if h.direct_servo_interpolation in ("linear", "limited") else None
    )
    direct_cfg = HcxDirectServoConfig.from_runtime_config(
        h, source_rate_hz=source_rate_hz
    )
    if direct_cfg.watchdog_s <= 1.0 / float(ARM_SERVO_HZ):
        raise RuntimeError("hcx.direct_servo_watchdog_s 必须大于一个 500 Hz 周期")
    return runtime, direct_cfg, source_hz


def _connect_hcx_direct(
    teleop_yaml: Path,
) -> tuple[HcxConnection, HcxFollower, HcxFollower, Any, Any, HcxDirectServoConfig, int]:
    runtime, direct_cfg, source_hz = _direct_servo_config(teleop_yaml)
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
        f"[INFO] HCX 直伺服已连接: source={source_hz} Hz out={direct_cfg.rate_hz} Hz "
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
        source_hz,
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

    def last(self) -> float:
        if not self._vals:
            return 0.0
        return float(self._vals[-1])


class ServoInterpolator:
    def __init__(self, source_fps: float, n_joints: int, servo_hz: int) -> None:
        self._fps = float(source_fps)
        self._servo_hz = float(servo_hz)
        self._n_joints = int(n_joints)
        self._lock = threading.Lock()
        self._points: np.ndarray | None = None
        self._t0 = 0.0
        self._popped = 0
        self._speed_scale = 1.0
        self._x = 0.0

    def submit(self, points: np.ndarray, t0: float) -> None:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, self._n_joints)
        with self._lock:
            self._points = pts
            self._t0 = float(t0)
            self._popped = 0
            self._x = 0.0

    def set_speed_scale(self, scale: float, t: float | None = None) -> None:
        scale = max(1e-6, float(scale))
        now = time.perf_counter() if t is None else float(t)
        with self._lock:
            if abs(scale - self._speed_scale) < 1e-9:
                return
            x = (now - self._t0) * self._fps * self._speed_scale
            self._speed_scale = scale
            self._t0 = now - x / (self._fps * self._speed_scale)
            self._x = x

    def speed_scale(self) -> float:
        with self._lock:
            return float(self._speed_scale)

    def progress(self) -> float:
        with self._lock:
            if self._points is None:
                return 0.0
            t = time.perf_counter()
            x = (t - self._t0) * self._fps * self._speed_scale
            n = len(self._points)
            return float(max(0.0, min(x, float(max(n - 1, 0)))))

    def sample(self, t: float) -> np.ndarray | None:
        with self._lock:
            if self._points is None:
                return None
            x = (t - self._t0) * self._fps * self._speed_scale
            n = len(self._points)
            self._x = x
            if x <= 0.0:
                target = self._points[0]
            elif x >= n - 1:
                target = self._points[-1]
            else:
                i = int(math.floor(x))
                alpha = x - i
                target = self._points[i] * (1.0 - alpha) + self._points[i + 1] * alpha
            return target.copy()

    def mark_sent(self) -> None:
        with self._lock:
            self._popped += 1

    def popped(self) -> int:
        with self._lock:
            if self._points is None:
                return 0
            t = time.perf_counter()
            x = (t - self._t0) * self._fps * self._speed_scale
            n = len(self._points)
            x = max(0.0, min(x, float(max(n - 1, 0))))
            return int(x * self._servo_hz / self._fps)


class _JointCVKalman:
    def __init__(self, n: int, q: float, r: float) -> None:
        self._q = float(q)
        self._r = float(r)
        self._n = int(n)
        self._p: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._P00: np.ndarray | None = None
        self._P01: np.ndarray | None = None
        self._P11: np.ndarray | None = None

    def step(self, z: np.ndarray, dt: float) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).reshape(self._n)
        if self._p is None:
            self._p = z.copy()
            self._v = np.zeros(self._n, dtype=np.float64)
            self._P00 = np.full(self._n, self._r, dtype=np.float64)
            self._P01 = np.zeros(self._n, dtype=np.float64)
            self._P11 = np.ones(self._n, dtype=np.float64)
            return self._p.copy()
        dt = float(dt)
        p = self._p + self._v * dt
        q00 = self._q * (dt ** 4) / 4.0
        q01 = self._q * (dt ** 3) / 2.0
        q11 = self._q * (dt ** 2)
        P00 = self._P00 + dt * (self._P01 + self._P01) + dt * dt * self._P11 + q00
        P01 = self._P01 + dt * self._P11 + q01
        P11 = self._P11 + q11
        s = P00 + self._r
        k0 = P00 / s
        k1 = P01 / s
        y = z - p
        self._p = p + k0 * y
        self._v = self._v + k1 * y
        self._P00 = (1.0 - k0) * P00
        self._P01 = (1.0 - k0) * P01
        self._P11 = P11 - k1 * P01
        return self._p.copy()


class ServoSendThread:
    def __init__(
        self,
        left: HcxFollower,
        right: HcxFollower,
        interp: ServoInterpolator,
        *,
        rate_hz: int,
        joint_lo: np.ndarray,
        joint_hi: np.ndarray,
        follow_l_lo: np.ndarray,
        follow_l_hi: np.ndarray,
        follow_r_lo: np.ndarray,
        follow_r_hi: np.ndarray,
        tau_s: float,
        tau2_s: float,
        deadband_deg: float,
        max_vel_deg_s: float,
        max_delta_deg: float,
        kalman_enabled: bool = True,
        kalman_q: float = 0.0,
        kalman_r: float = 0.3,
    ) -> None:
        self._left = left
        self._right = right
        self._interp = interp
        self._rate_hz = int(rate_hz)
        self._dt = 1.0 / float(self._rate_hz)
        self._lo = np.asarray(joint_lo, dtype=np.float64)
        self._hi = np.asarray(joint_hi, dtype=np.float64)
        self._follow_l_lo = np.asarray(follow_l_lo, dtype=np.float64)
        self._follow_l_hi = np.asarray(follow_l_hi, dtype=np.float64)
        self._follow_r_lo = np.asarray(follow_r_lo, dtype=np.float64)
        self._follow_r_hi = np.asarray(follow_r_hi, dtype=np.float64)
        self._tau_s = float(tau_s)
        self._tau2_s = float(tau2_s)
        self._deadband_deg = float(deadband_deg)
        self._max_vel_deg_s = float(max_vel_deg_s)
        self._max_delta_deg = float(max_delta_deg)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fault: str | None = None
        self._f1_l: np.ndarray | None = None
        self._f2_l: np.ndarray | None = None
        self._f1_r: np.ndarray | None = None
        self._f2_r: np.ndarray | None = None
        self._last_l: np.ndarray | None = None
        self._last_r: np.ndarray | None = None
        self._cmd_lock = threading.Lock()
        q = float(kalman_q)
        r = float(kalman_r)
        use_kf = bool(kalman_enabled) and q > 0.0
        self._kf_l = _JointCVKalman(7, q, r) if use_kf else None
        self._kf_r = _JointCVKalman(7, q, r) if use_kf else None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("伺服下发线程已在运行")
        self._thread = threading.Thread(
            target=self._run, name=f"servo-{self._rate_hz}hz-source", daemon=True
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

    def popped(self) -> int:
        return self._interp.popped()

    def set_max_vel(self, max_vel_deg_s: float) -> None:
        self._max_vel_deg_s = float(max_vel_deg_s)

    def latest_joints(self) -> np.ndarray | None:
        with self._cmd_lock:
            if self._last_l is None or self._last_r is None:
                return None
            return np.concatenate([self._last_l, self._last_r]).astype(np.float32)

    def _shape_arm(
        self,
        desired: np.ndarray,
        *,
        last: np.ndarray | None,
        f1: np.ndarray | None,
        f2: np.ndarray | None,
        follow_lo: np.ndarray,
        follow_hi: np.ndarray,
        dt: float,
        kf: _JointCVKalman | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        desired = np.asarray(desired, dtype=np.float64).reshape(7)
        if last is None:
            cmd = np.clip(desired, self._lo, self._hi)
            cmd = np.clip(cmd, follow_lo, follow_hi)
            if kf is not None:
                kf.step(cmd, dt)
            return cmd, cmd.copy(), cmd.copy()
        clamped, _ = _clamp_joints_by_max_delta(desired, last, self._max_delta_deg)
        desired = np.asarray(clamped, dtype=np.float64)
        if self._deadband_deg > 0.0:
            desired = np.where(
                np.abs(desired - last) < self._deadband_deg, last, desired
            )
        if self._tau_s > 0.0 and f1 is not None:
            alpha = dt / (self._tau_s + dt)
            f1 = f1 + alpha * (desired - f1)
        else:
            f1 = desired
        if self._tau2_s > 0.0 and f2 is not None:
            alpha2 = dt / (self._tau2_s + dt)
            f2 = f2 + alpha2 * (f1 - f2)
        else:
            f2 = f1
        filt = f2 if kf is None else kf.step(f2, dt)
        max_step = self._max_vel_deg_s * dt
        cmd = last + np.clip(filt - last, -max_step, max_step)
        cmd = np.clip(cmd, self._lo, self._hi)
        cmd = np.clip(cmd, follow_lo, follow_hi)
        return (
            cmd,
            np.array(f1, dtype=np.float64, copy=True),
            np.array(f2, dtype=np.float64, copy=True),
        )

    def _run(self) -> None:
        try:
            next_t = time.perf_counter()
            last_tick = next_t
            while not self._stop.is_set():
                t = time.perf_counter()
                dt = max(t - last_tick, 1e-4)
                last_tick = t
                target = self._interp.sample(t)
                if target is not None:
                    left_des = target[:7]
                    right_des = target[7:14]
                    left_cmd, self._f1_l, self._f2_l = self._shape_arm(
                        left_des,
                        last=self._last_l,
                        f1=self._f1_l,
                        f2=self._f2_l,
                        follow_lo=self._follow_l_lo,
                        follow_hi=self._follow_l_hi,
                        dt=dt,
                        kf=self._kf_l,
                    )
                    right_cmd, self._f1_r, self._f2_r = self._shape_arm(
                        right_des,
                        last=self._last_r,
                        f1=self._f1_r,
                        f2=self._f2_r,
                        follow_lo=self._follow_r_lo,
                        follow_hi=self._follow_r_hi,
                        dt=dt,
                        kf=self._kf_r,
                    )
                    with self._cmd_lock:
                        self._last_l = left_cmd.copy()
                        self._last_r = right_cmd.copy()
                    self._left.send_joint_angles_deg(left_cmd, self._dt)
                    self._right.send_joint_angles_deg(right_cmd, self._dt)
                    self._interp.mark_sent()
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


def _capture_infer_obs(
    hw: HardwareBundle,
    cameras: list[str],
    joint_arr: Any,
    *,
    image_size: int | list[int] | None,
    depth_cameras: list[str],
    depth_min_mm: float,
    depth_max_mm: float,
    expected_scale_mm: dict[str, float],
) -> Observation:
    capture = hw.camera_capture
    if capture is None:
        raise RuntimeError("相机采集器未初始化")
    images, depths = capture.capture()
    missing = [n for n in cameras if n not in images]
    if missing:
        raise RuntimeError(f"采图缺少相机: {missing}")
    images = {n: images[n] for n in cameras}
    with joint_arr.get_lock():
        arr = np.frombuffer(joint_arr.get_obj(), dtype=np.float64).copy()
    raw = Observation(
        images=images,
        state=arr.astype(np.float32),
        timestamp=time.time(),
        depths=depths,
    )
    raw.validate(cameras, int(raw.state.shape[0]))
    return _prepare_rgbd_observation(
        raw,
        image_size=image_size,
        depth_cameras=depth_cameras,
        depth_min_mm=depth_min_mm,
        depth_max_mm=depth_max_mm,
        expected_scale_mm=expected_scale_mm,
    )


def _rtc_delay(
    latency_tracker: LatencyTracker, train_fps: float, chunk: int, infer_i: int
) -> int:
    if infer_i == 0:
        return 0
    d = int(math.ceil(latency_tracker.last() * float(train_fps)))
    return max(0, min(d, int(chunk) - 3))


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


def _deploy_rtc(deploy_rtc: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(deploy_rtc or {})
    out.pop("execution_horizon", None)
    return out


def _submit_interp_chunk(interp: ServoInterpolator, chunk: np.ndarray) -> None:
    interp.submit(np.asarray(chunk, dtype=np.float64), time.perf_counter())


def _phase_speed_scale(phase_actions: float, skip: int, speed_max: float) -> float:
    if phase_actions < float(skip):
        return 1.0
    ramp_t = (float(phase_actions) - float(skip)) / float(RAMP_ACTIONS)
    if ramp_t >= 1.0:
        return float(speed_max)
    return 1.0 + (float(speed_max) - 1.0) * ramp_t


class _VelPhase:
    def __init__(self) -> None:
        self.lifted = False
        self.prev_path = 0.0
        self.prev_gr = 1.0
        self.right_x2 = False
        self.right_moved = False
        self.left_on = False
        self.left_i = 0
        self.hold_i = 0
        self.down_i = 0
        self.close_half_i = 0
        self.right_lat_i = 0


def _reset_left_sm(phase: _VelPhase) -> None:
    phase.left_on = False
    phase.left_i = 0
    phase.hold_i = 0
    phase.down_i = 0
    phase.close_half_i = 0


def _close_scale(steps_to_k: int) -> float:
    if steps_to_k > 8:
        return 1.0
    if steps_to_k > 4:
        return 0.6
    return 0.5


def _path_scale(path_i: float) -> float:
    if path_i > 15.0:
        return 1.0
    return 0.5


def _lifted_move_scale(dist: int | None, lat_i: int) -> float:
    if dist is not None and dist <= RIGHT_NEAR_N:
        if dist <= 0:
            return 0.5
        return 0.5 + 0.5 * (float(dist) / float(RIGHT_NEAR_N))
    if lat_i >= RAMP_ACTIONS:
        return 1.0
    return 2.0 - float(lat_i) / float(RAMP_ACTIONS)


def _scale_joint_deltas(
    seq: np.ndarray, idx: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    out = np.array(seq, dtype=np.float32, copy=True)
    cols = np.asarray(idx)
    t = int(seq.shape[0])
    for i in range(1, t):
        d = seq[i, cols] - seq[i - 1, cols]
        out[i, cols] = out[i - 1, cols] + np.float32(scales[i - 1]) * d
    return out


def _right_path(remain: np.ndarray, right_j: np.ndarray) -> tuple[float, np.ndarray]:
    right = remain[:, right_j]
    if right.shape[0] < 2:
        return 0.0, np.zeros(0, dtype=np.float64)
    step = np.linalg.norm(np.diff(right, axis=0), axis=1)
    path_from = np.cumsum(step[::-1])[::-1]
    return float(step.sum()), path_from.astype(np.float64, copy=False)


def _apply_phase_slowdown(
    processed: np.ndarray,
    skip: int,
    left_j: np.ndarray,
    right_j: np.ndarray,
    left_g_i: int,
    right_g_i: int,
    phase: _VelPhase,
    fps: float,
) -> np.ndarray:
    n = int(processed.shape[0])
    skip_i = max(0, min(int(skip), n - 1))
    remain = np.array(processed[skip_i:], dtype=np.float32, copy=True)
    t = int(remain.shape[0])
    if t < 2:
        if t >= 1:
            if float(remain[0, left_g_i]) < GRIP_CLOSED:
                _reset_left_sm(phase)
            phase.prev_gr = float(remain[0, right_g_i])
        phase.prev_path = 0.0
        return processed
    path, path_from = _right_path(remain, right_j)
    r0 = remain[:, int(right_j[0])]
    d_r0 = np.diff(r0.astype(np.float64))
    d_r0_mean = float(d_r0.mean()) if d_r0.size else 0.0
    g_l = remain[:, left_g_i]
    g_r = remain[:, right_g_i]
    g_r0 = float(g_r[0])
    open_at: int | None = None
    if phase.prev_gr < GRIP_CLOSED and g_r0 > GR_IS_OPEN:
        open_at = 0
    else:
        for i in range(t):
            prev = phase.prev_gr if i == 0 else float(g_r[i - 1])
            if prev < GRIP_CLOSED and float(g_r[i]) > GR_IS_OPEN:
                open_at = i
                break
    if open_at is not None:
        phase.right_x2 = True
        phase.right_moved = False
    if float(g_l[0]) < GRIP_CLOSED:
        _reset_left_sm(phase)
    elif float(g_l[0]) > GRIP_OPEN and g_r0 < GRIP_CLOSED:
        phase.left_on = True
    close_k: int | None = None
    if float(g_l[0]) > GL_STILL_OPEN:
        below = np.where(g_l < GL_WILL_CLOSE)[0]
        if below.size:
            close_k = int(below[0])
    if close_k is not None:
        half_n = max(1, int(round(float(fps) * LEFT_CLOSE_HALF_S)))
        s_l = np.empty(t - 1, dtype=np.float64)
        for i in range(t - 1):
            s = _close_scale(close_k - i)
            if s <= 0.5:
                if phase.close_half_i >= half_n:
                    s = 1.0
                else:
                    phase.close_half_i += 1
            s_l[i] = s
        remain = _scale_joint_deltas(remain, left_j, s_l)
    elif phase.left_on:
        skip_n = int(SKIP_PHASE_A)
        ramp_n = int(RAMP_ACTIONS)
        at_2x = skip_n + ramp_n
        hold_n = max(1, int(round(float(fps) * LEFT_HOLD_S)))
        down_n = max(1, int(round(float(fps) * LEFT_DOWN_S)))
        s_l = np.ones(t - 1, dtype=np.float64)
        for i in range(t - 1):
            right_closed_i = float(g_r[i]) < GRIP_CLOSED
            s = 1.0
            if phase.left_i >= skip_n and phase.left_i < at_2x:
                rt = float(phase.left_i - skip_n) / float(ramp_n)
                desired = 1.0 + rt
                s = 1.0 if right_closed_i else desired
            elif phase.left_i >= at_2x:
                if phase.hold_i < hold_n:
                    s = 1.0 if right_closed_i else 2.0
                elif not right_closed_i:
                    s = float(
                        np.clip(2.0 - float(phase.down_i) / float(down_n), 1.0, 2.0)
                    )
                    phase.down_i += 1
                if phase.hold_i < hold_n:
                    phase.hold_i += 1
            s_l[i] = s
            phase.left_i += 1
        if float(s_l.max()) > 1.0:
            remain = _scale_joint_deltas(remain, left_j, s_l)
    if float(r0.min()) < R0_LIFTED:
        phase.lifted = True
    if float(r0[0]) > R0_DOWN and path < PATH_IDLE:
        phase.lifted = False
    if not phase.lifted:
        phase.right_lat_i = 0
    if phase.right_x2 and path > PATH_IDLE:
        phase.right_moved = True
    r_close_k: int | None = None
    if g_r0 > GR_IS_OPEN:
        below_r = np.where(g_r < GRIP_CLOSED)[0]
        if below_r.size:
            r_close_k = int(below_r[0])
    s_r = np.ones(t - 1, dtype=np.float64)
    apply_r = False
    lifting = g_r0 > GR_IS_OPEN and d_r0_mean < DR0_LIFT
    from_rest = phase.prev_path < PATH_IDLE
    if lifting and (not from_rest or path > PATH_LIFT):
        apply_r = True
        for i in range(t - 1):
            s_r[i] = _path_scale(float(path_from[i]))
    elif phase.lifted:
        apply_r = True
        for i in range(t - 1):
            dist = (r_close_k - i) if r_close_k is not None else None
            s_r[i] = _lifted_move_scale(dist, phase.right_lat_i)
            phase.right_lat_i += 1
    elif phase.right_x2:
        apply_r = True
        off = int(open_at) if open_at is not None else 0
        for i in range(t - 1):
            if i >= off:
                s_r[i] = RIGHT_OPEN_MOVE
    if phase.right_x2 and phase.right_moved and path < PATH_IDLE:
        phase.right_x2 = False
        phase.right_moved = False
    if apply_r:
        remain = _scale_joint_deltas(remain, right_j, s_r)
    phase.prev_path = path
    phase.prev_gr = float(g_r[-1])
    out = np.array(processed, dtype=np.float32, copy=True)
    out[skip_i:] = remain
    return out


def _euro_snap(
    euro: OneEuroFilter,
) -> tuple[np.ndarray | None, np.ndarray, float | None]:
    x = None if euro._x is None else np.array(euro._x, dtype=np.float64, copy=True)
    return (
        x,
        np.array(euro._dx, dtype=np.float64, copy=True),
        euro._last_t,
    )


def _euro_restore(
    euro: OneEuroFilter,
    snap: tuple[np.ndarray | None, np.ndarray, float | None],
) -> None:
    x, dx, last_t = snap
    euro._x = None if x is None else np.array(x, dtype=np.float64, copy=True)
    euro._dx = np.array(dx, dtype=np.float64, copy=True)
    euro._last_t = last_t


def _filter_chunk_joints(
    processed: np.ndarray,
    *,
    skip: int,
    restore_i: int,
    joint_idx: np.ndarray,
    euro: OneEuroFilter,
    dt: float,
    snaps_prev: list[tuple[np.ndarray | None, np.ndarray, float | None]],
) -> tuple[np.ndarray, list[tuple[np.ndarray | None, np.ndarray, float | None]]]:
    out = np.array(processed, dtype=np.float32, copy=True)
    n = int(out.shape[0])
    skip = max(0, min(int(skip), n - 1))
    if snaps_prev and restore_i >= 0:
        _euro_restore(euro, snaps_prev[min(restore_i, len(snaps_prev) - 1)])
    snaps: list[tuple[np.ndarray | None, np.ndarray, float | None]] = []
    dt_f = float(dt)
    for i in range(skip, n):
        if euro._last_t is None:
            t = 0.0
        else:
            t = float(euro._last_t) + dt_f
        des = euro.step(np.asarray(out[i, joint_idx], dtype=np.float64), t)
        out[i, joint_idx] = np.asarray(des, dtype=np.float32)
        snaps.append(_euro_snap(euro))
    return out, snaps


def _cpu_affinity_halves() -> tuple[set[int], set[int]] | None:
    n = os.cpu_count() or 0
    if n < 4:
        return None
    mid = n // 2
    return set(range(0, mid)), set(range(mid, n))


def _bind_cpus(cores: set[int]) -> None:
    try:
        os.sched_setaffinity(0, cores)
    except (AttributeError, OSError):
        pass


def _gpu_keepalive(keep: torch.Tensor | None) -> None:
    if keep is not None:
        keep.add_(0.0)


def _sample_actions_timed(
    policy: Any,
    batch: dict[str, torch.Tensor],
    *,
    leftover: torch.Tensor | None,
    inference_delay: int,
    execution_horizon: int,
    device: torch.device,
    chunk_n: int,
) -> tuple[torch.Tensor, float]:
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        pred_g = policy.sample_actions(
            batch,
            prev_chunk_left_over=leftover,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )[0][:chunk_n].contiguous()
        if use_cuda:
            torch.cuda.synchronize()
        pred = pred_g.cpu()
    return pred, time.perf_counter() - t0


def _infer_worker(
    spec: dict[str, Any],
    joint_arr: Any,
    popped_val: Any,
    ack_val: Any,
    out_q: Any,
    ready_evt: Any,
    stop_evt: Any,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        devnull = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull, 0)
        os.close(devnull)
    except OSError:
        pass
    hw: HardwareBundle | None = None
    try:
        halves = _cpu_affinity_halves()
        if halves is not None:
            _bind_cpus(halves[1])
        gc.disable()
        torch.set_num_threads(1)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        ckpt_path = Path(spec["ckpt"])
        train_cfg_path = spec["train_cfg"]
        train_cfg_path = None if train_cfg_path is None else Path(train_cfg_path)
        teleop_yaml = Path(spec["teleop_yaml"])
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = load_config(train_cfg_path) if train_cfg_path is not None else ckpt["config"]
        stats = ckpt["stats"]
        cameras = list(cfg.cameras)
        _apply_rtc_overrides(cfg, _deploy_rtc(spec.get("deploy_rtc")))
        rtc_cfg = _normalize_rtc_config(cfg.policy.rtc)
        cfg.policy.history_noise_std = 0.0
        n_obs = int(cfg.dataset.n_obs_steps)
        train_fps = float(spec["fps"])
        chunk_n = int(spec["chunk"])
        max_steps = int(spec["max_steps"])
        source_hz = int(spec["source_hz"])
        depth_cameras = list(getattr(cfg.dataset, "depth_cameras", ()) or ())
        expected_scale_mm = dict(getattr(cfg.dataset, "scale_mm_per_raw_unit", None) or {})
        depth_min_mm = float(cfg.dataset.depth_min_mm)
        depth_max_mm = float(cfg.dataset.depth_max_mm)
        image_size = cfg.dataset.image_size
        device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
        policy = build_policy(cfg, stats)
        policy.load_state_dict(ckpt["policy_state_dict"])
        policy.to(device)
        policy.eval()
        policy_cfg = getattr(policy, "cfg", None)
        if hasattr(policy_cfg, "history_noise_std"):
            policy_cfg.history_noise_std = 0.0
        if device.type == "cuda":
            policy = torch.compile(policy, mode="default")
        keep = (
            torch.zeros(1, device=device, dtype=torch.float32)
            if device.type == "cuda"
            else None
        )
        predict_joint_delta = bool(cfg.policy.predict_joint_delta)
        joint_mask = joint_mask_from_names(cfg.action_names, cfg.action_dim)
        joint_idx = np.where(joint_mask)[0]
        left_j = joint_idx[:7]
        right_j = joint_idx[7:14]
        grip_idx = np.where(~joint_mask)[0]
        left_g_i = int(grip_idx[0]) if grip_idx.size >= 1 else 14
        right_g_i = int(grip_idx[1]) if grip_idx.size >= 2 else 15
        hw = HardwareBundle(
            left_start_joints_deg=spec["left_start"],
            right_start_joints_deg=spec["right_start"],
        )
        hw.camera_manager = _connect_record_cameras(teleop_yaml)
        hw.camera_capture = HeadTriggeredCapture(
            hw.camera_manager,
            depth_cameras=depth_cameras,
            expected_scale_mm=expected_scale_mm,
        )
        obs_kw = dict(
            image_size=image_size,
            depth_cameras=depth_cameras,
            depth_min_mm=depth_min_mm,
            depth_max_mm=depth_max_mm,
            expected_scale_mm=expected_scale_mm,
        )
        action_queue = ActionQueue(rtc_cfg)
        latency_tracker = LatencyTracker()
        one_euro = OneEuroFilter(
            n_joints=int(joint_idx.size),
            mincutoff=float(spec["one_euro_mincutoff_hz"]),
            beta=float(spec["one_euro_beta"]),
            dcutoff=float(spec["one_euro_dcutoff_hz"]),
        )
        euro_snaps: list[tuple[np.ndarray | None, np.ndarray, float | None]] = []
        filter_enabled = bool(spec["one_euro_enabled"])
        warmup_obs = _capture_infer_obs(hw, list(cameras), joint_arr, **obs_kw)
        warmup_batch = _build_obs_batch(
            [warmup_obs],
            cameras=list(cameras),
            n_obs_steps=n_obs,
            stats=stats,
            norm_mode=cfg.dataset.norm_mode,
            device=device,
            depth_cameras=depth_cameras,
            predict_joint_delta=predict_joint_delta,
            joint_mask=joint_mask,
        )
        _, _ = _sample_actions_timed(
            policy,
            warmup_batch,
            leftover=None,
            inference_delay=0,
            execution_horizon=2,
            device=device,
            chunk_n=chunk_n,
        )
        infer_i = 0
        vel_phase = _VelPhase()
        while infer_i < max_steps and not stop_evt.is_set():
            infer_delay = _rtc_delay(latency_tracker, train_fps, chunk_n, infer_i)
            if infer_i > 0:
                while not stop_evt.is_set() and int(ack_val.value) != infer_i - 1:
                    _gpu_keepalive(keep)
                    time.sleep(0.002)
                if stop_evt.is_set():
                    break
                _sync_action_queue_index(
                    action_queue, int(popped_val.value), train_fps, source_hz
                )
            idx_before = action_queue.get_action_index()
            leftover = action_queue.get_left_over()
            if leftover is not None and leftover.shape[0] == 0:
                leftover = None
            leftover_len = 0 if leftover is None else int(leftover.shape[0])
            exec_h = infer_delay + 2
            if leftover_len > 0:
                exec_h = min(exec_h, leftover_len)
            obs = _capture_infer_obs(hw, list(cameras), joint_arr, **obs_kw)
            obs_history = [obs]
            q_now = np.asarray(obs_history[-1].state, dtype=np.float32)
            batch = _build_obs_batch(
                obs_history,
                cameras=list(cameras),
                n_obs_steps=n_obs,
                stats=stats,
                norm_mode=cfg.dataset.norm_mode,
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
                            norm_mode=cfg.dataset.norm_mode,
                            joint_mask=joint_mask,
                            device=device,
                        )
                n_left = int(leftover.shape[0])
                if n_left < chunk_n:
                    leftover = torch.cat(
                        [leftover, leftover[-1:].expand(chunk_n - n_left, -1)],
                        dim=0,
                    )
                elif n_left > chunk_n:
                    leftover = leftover[:chunk_n]
            pred, infer_s = _sample_actions_timed(
                policy,
                batch,
                leftover=leftover,
                inference_delay=infer_delay,
                execution_horizon=exec_h,
                device=device,
                chunk_n=chunk_n,
            )
            latency_tracker.add(infer_s)
            popped = 0 if infer_i == 0 else int(popped_val.value)
            idx_after = _sync_action_queue_index(
                action_queue, popped, train_fps, source_hz
            )
            new_delay = 0
            if infer_i > 0:
                new_delay = max(0, idx_after - idx_before)
                new_delay = min(new_delay, leftover_len, chunk_n - 3)
            processed = denormalize_predicted_action(
                pred,
                stats,
                cfg.dataset.norm_mode,
                q_now_phys=q_now,
                predict_joint_delta=predict_joint_delta,
                joint_mask=joint_mask,
            )
            processed = np.asarray(processed, dtype=np.float32)[:chunk_n]
            if infer_i == 0:
                processed[0, left_j] = q_now[left_j]
                processed[0, right_j] = q_now[right_j]
                skip = 0
                restore_i = -1
            else:
                skip = min(new_delay, int(processed.shape[0]) - 1)
                restore_i = int(idx_after) - 1
            if filter_enabled:
                processed, euro_snaps = _filter_chunk_joints(
                    processed,
                    skip=skip,
                    restore_i=restore_i,
                    joint_idx=joint_idx,
                    euro=one_euro,
                    dt=1.0 / float(train_fps),
                    snaps_prev=euro_snaps,
                )
            processed = _apply_phase_slowdown(
                processed,
                skip,
                left_j,
                right_j,
                left_g_i,
                right_g_i,
                vel_phase,
                train_fps,
            )
            processed_t = torch.as_tensor(processed, dtype=torch.float32)
            action_queue.merge(pred, processed_t, new_delay, idx_before)
            phys = np.asarray(processed, dtype=np.float64)
            remain = phys[skip:]
            out_chunk = np.concatenate([remain[:, left_j], remain[:, right_j]], axis=1)
            g_row = remain[0]
            payload = {
                "infer_i": infer_i,
                "chunk": out_chunk,
                "grip_l": float(np.clip(g_row[left_g_i], 0.0, 1.0)),
                "grip_r": float(np.clip(g_row[right_g_i], 0.0, 1.0)),
                "leftover": leftover_len,
                "delay": int(new_delay),
                "qsize": int(action_queue.qsize()),
                "infer_ms": float(infer_s * 1e3),
            }
            while not stop_evt.is_set():
                try:
                    out_q.put(payload, timeout=0.2)
                    break
                except Full:
                    continue
            if stop_evt.is_set():
                break
            if infer_i == 0:
                ready_evt.set()
            infer_i += 1
    except BaseException as exc:
        try:
            out_q.put({"error": f"{type(exc).__name__}: {exc}"}, timeout=1.0)
        except Exception:
            pass
        ready_evt.set()
    finally:
        if hw is not None:
            try:
                _shutdown(hw)
            except Exception:
                pass


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

    _apply_rtc_overrides(cfg, _deploy_rtc(deploy.get("rtc")))
    rtc_cfg = _normalize_rtc_config(cfg.policy.rtc)
    if not bool(rtc_cfg.enabled):
        raise ValueError("本脚本要求 rtc.enabled=true")

    train_history_noise = float(getattr(cfg.policy, "history_noise_std", 0.0) or 0.0)
    cfg.policy.history_noise_std = 0.0
    norm_mode = cfg.dataset.norm_mode
    n_obs = int(cfg.dataset.n_obs_steps)
    n_action_steps = int(cfg.policy.n_action_steps)
    horizon = int(cfg.dataset.horizon)
    train_fps = float(cfg.fps)
    if extra.get("fps"):
        train_fps = float(extra["fps"])
    max_steps = int(deploy["max_steps"])
    chunk_n = int(extra["chunk"])
    if chunk_n >= horizon:
        raise ValueError(
            f"chunk 必须小于模型输出长度 (chunk={chunk_n}, horizon={horizon})"
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

    halves = _cpu_affinity_halves()
    if halves is not None:
        _bind_cpus(halves[0])

    print(
        f"[INFO] 策略就绪 layout=dual device={device} cameras={cameras} "
        f"norm={norm_mode} n_obs={n_obs} "
        f"train_fps={train_fps:g} rtc.guidance={rtc_cfg.guidance_enabled} "
        f"chunk={chunk_n} delay_exec_h=delay+2 n_action_steps={n_action_steps} "
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
    infer_proc: mp.Process | None = None
    stop_evt = None
    watchdog: ServoWatchdogThread | None = None
    left_follower: HcxFollower | None = None
    right_follower: HcxFollower | None = None
    quit_keys: QuitKeyThread | None = None
    try:
        (
            _hcx_connection,
            left_follower,
            right_follower,
            left_arm,
            right_arm,
            direct_cfg,
            source_hz,
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

        follow_l_lo, follow_l_hi = left_follower.joint_limits_deg
        follow_r_lo, follow_r_hi = right_follower.joint_limits_deg
        follow_l_lo = np.asarray(follow_l_lo, dtype=np.float64)
        follow_l_hi = np.asarray(follow_l_hi, dtype=np.float64)
        follow_r_lo = np.asarray(follow_r_lo, dtype=np.float64)
        follow_r_hi = np.asarray(follow_r_hi, dtype=np.float64)
        tau_s = float(extra["command_filter_tau_s"])
        tau2_s = float(extra["command_filter_tau2_s"])
        deadband_deg = float(extra["command_deadband_deg"])
        max_vel = float(extra["max_joint_vel_deg_s"])
        max_delta = float(deploy["move_max_delta_deg"])
        interp = ServoInterpolator(
            source_fps=train_fps, n_joints=14, servo_hz=source_hz
        )
        send_thread = ServoSendThread(
            left_follower,
            right_follower,
            interp,
            rate_hz=source_hz,
            joint_lo=joint_lo,
            joint_hi=joint_hi,
            follow_l_lo=follow_l_lo,
            follow_l_hi=follow_l_hi,
            follow_r_lo=follow_r_lo,
            follow_r_hi=follow_r_hi,
            tau_s=tau_s,
            tau2_s=tau2_s,
            deadband_deg=deadband_deg,
            max_vel_deg_s=max_vel,
            max_delta_deg=max_delta,
            kalman_enabled=bool(extra["kalman_enabled"]),
            kalman_q=float(extra["kalman_q"]),
            kalman_r=float(extra["kalman_r"]),
        )
        seed_l = _read_hcx_joints(hw.left_arm, hw.left_start_joints_deg)
        seed_r = _read_hcx_joints(hw.right_arm, hw.right_start_joints_deg)
        seed_lg = _read_gripper(hw.left_gripper, fallback=1.0)
        seed_rg = _read_gripper(hw.right_gripper, fallback=1.0)
        seed = np.concatenate(
            [seed_l, seed_r, np.asarray([seed_lg, seed_rg], dtype=np.float32)]
        ).astype(np.float64)
        ctx = mp.get_context("spawn")
        joint_arr = ctx.Array("d", 16)
        with joint_arr.get_lock():
            for i, v in enumerate(seed.tolist()):
                joint_arr[i] = float(v)
        popped_val = ctx.Value("i", 0)
        ack_val = ctx.Value("i", -1)
        out_q = ctx.Queue(maxsize=4)
        ready_evt = ctx.Event()
        stop_evt = ctx.Event()
        spec = {
            "ckpt": str(ckpt_path),
            "train_cfg": None if train_cfg_path is None else str(train_cfg_path),
            "teleop_yaml": str(teleop_yaml),
            "deploy_rtc": _deploy_rtc(deploy.get("rtc")),
            "chunk": int(chunk_n),
            "max_steps": int(max_steps),
            "source_hz": int(source_hz),
            "fps": float(train_fps),
            "left_start": deploy["left_start_joints_deg"],
            "right_start": deploy["right_start_joints_deg"],
            "one_euro_enabled": bool(extra["one_euro_enabled"]),
            "one_euro_mincutoff_hz": float(extra["one_euro_mincutoff_hz"]),
            "one_euro_beta": float(extra["one_euro_beta"]),
            "one_euro_dcutoff_hz": float(extra["one_euro_dcutoff_hz"]),
        }
        infer_proc = ctx.Process(
            target=_infer_worker,
            args=(spec, joint_arr, popped_val, ack_val, out_q, ready_evt, stop_evt),
            name="infer-worker",
            daemon=True,
        )
        infer_proc.start()
        print("[INFO] 推理进程已启动", flush=True)
        if not ready_evt.wait(timeout=180.0):
            raise RuntimeError("推理进程启动超时")
        item = out_q.get(timeout=30.0)
        if "error" in item:
            raise RuntimeError(item["error"])
        _submit_interp_chunk(interp, np.asarray(item["chunk"], dtype=np.float64))
        popped_val.value = 0
        ack_val.value = int(item["infer_i"])
        if hw.left_gripper_loop is not None:
            hw.left_gripper_loop.set_opening(float(item["grip_l"]))
        if hw.right_gripper_loop is not None:
            hw.right_gripper_loop.set_opening(float(item["grip_r"]))
        print(
            f"[INFO] infer={item['infer_i']} leftover={item['leftover']} "
            f"delay={item['delay']} qsize={item['qsize']} "
            f"infer_ms={item['infer_ms']:.0f}",
            flush=True,
        )
        if not left_follower.start_servo():
            raise RuntimeError("HCX left direct-servo 启动失败")
        if not right_follower.start_servo():
            raise RuntimeError("HCX right direct-servo 启动失败")
        print(
            f"[INFO] 双臂 direct-servo 已启动 interpolation={direct_cfg.interpolation} "
            f"source={source_hz}Hz out={direct_cfg.rate_hz}Hz",
            flush=True,
        )
        send_thread.start()
        watchdog = ServoWatchdogThread(
            left_follower,
            right_follower,
            fault_hz=float(extra["servo_fault_hz_warn"]),
        )
        watchdog.start()
        print(
            f"[INFO] {source_hz}Hz 源点下发已启动 → {ARM_SERVO_HZ}Hz，按 q 安全退出",
            flush=True,
        )
        quit_keys = QuitKeyThread()
        quit_keys.start()
        base_max_vel = float(max_vel)
        phase: str | None = None
        phase_actions = 0.0
        last_prog = 0.0
        cmd_gl = float(item["grip_l"])
        cmd_gr = float(item["grip_r"])
        while True:
            if quit_keys.pressed():
                print("\n[INFO] 收到 q，停止伺服", flush=True)
                break
            send_thread.check_fault()
            watchdog.check_fault()
            joints = send_thread.latest_joints()
            with joint_arr.get_lock():
                if joints is not None:
                    for i, v in enumerate(np.asarray(joints, dtype=np.float64).reshape(-1)[:14]):
                        joint_arr[i] = float(v)
                joint_arr[14] = float(
                    _read_gripper(hw.left_gripper, fallback=float(joint_arr[14]))
                )
                joint_arr[15] = float(
                    _read_gripper(hw.right_gripper, fallback=float(joint_arr[15]))
                )
            prog = float(interp.progress())
            if prog + 1e-6 < last_prog:
                last_prog = 0.0
            dprog = max(0.0, prog - last_prog)
            last_prog = prog
            left_open = cmd_gl > GRIP_OPEN
            left_closed = cmd_gl < GRIP_CLOSED
            right_open = cmd_gr > GRIP_OPEN
            right_closed = cmd_gr < GRIP_CLOSED
            new_phase: str | None = None
            if right_closed and left_open:
                new_phase = "A"
            elif left_closed and right_open:
                new_phase = "B"
            if new_phase != phase:
                phase = new_phase
                phase_actions = 0.0
            elif phase is not None:
                phase_actions += dprog
            if phase == "A":
                scale = _phase_speed_scale(phase_actions, SKIP_PHASE_A, SPEED_MAX_A)
            elif phase == "B":
                scale = _phase_speed_scale(phase_actions, SKIP_PHASE_B, SPEED_MAX_B)
            else:
                scale = 1.0
            interp.set_speed_scale(scale)
            send_thread.set_max_vel(base_max_vel * scale)
            popped_val.value = int(send_thread.popped())
            if infer_proc is not None and not infer_proc.is_alive():
                try:
                    item = out_q.get_nowait()
                except Empty:
                    break
            else:
                try:
                    item = out_q.get(timeout=0.002)
                except Empty:
                    continue
            if "error" in item:
                raise RuntimeError(item["error"])
            _submit_interp_chunk(interp, np.asarray(item["chunk"], dtype=np.float64))
            last_prog = 0.0
            popped_val.value = 0
            ack_val.value = int(item["infer_i"])
            cmd_gl = float(item["grip_l"])
            cmd_gr = float(item["grip_r"])
            if hw.left_gripper_loop is not None:
                hw.left_gripper_loop.set_opening(cmd_gl)
            if hw.right_gripper_loop is not None:
                hw.right_gripper_loop.set_opening(cmd_gr)
            print(
                f"[INFO] infer={item['infer_i']} leftover={item['leftover']} "
                f"delay={item['delay']} qsize={item['qsize']} "
                f"infer_ms={item['infer_ms']:.0f}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\n[INFO] 收到 Ctrl+C，停止伺服", flush=True)
    finally:
        if quit_keys is not None:
            quit_keys.stop()
        if send_thread is not None:
            send_thread.stop()
        if watchdog is not None:
            watchdog.stop()
        if left_follower is not None:
            try:
                left_follower.stop_servo()
            except Exception:
                pass
        if right_follower is not None:
            try:
                right_follower.stop_servo()
            except Exception:
                pass
        if stop_evt is not None:
            stop_evt.set()
        if infer_proc is not None:
            infer_proc.join(timeout=5.0)
            if infer_proc.is_alive():
                infer_proc.terminate()
                infer_proc.join(timeout=2.0)
        if left_follower is not None:
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
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
