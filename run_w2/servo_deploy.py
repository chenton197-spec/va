#!/usr/bin/env python3
"""W2 HCX 左臂 A2A 伺服闭环部署（15Hz 推理 + 500Hz PluseToServo 插值下发）。

与 run.py（move_joints 规划模式）的本质区别：
- 模型按训练 fps（15Hz）推理，输出未来 n_action_steps 步**绝对关节目标**；
- 伺服线程以 500Hz 在相邻目标点之间线性插值，直接调用 hcx_sdk
  ``Arm.pluse_to_servo()``（厂商 PluseToServo 直伺服，不经 move_joints 规划）；
- 机械臂以 500Hz 收到平滑连续的绝对目标，天然满足"高频控制 + 插值"。

为什么不自用 HcxDirectServoConfig 的插值：
- SDK linear/limited 插值要求 ``rate_hz % source_rate_hz == 0``
  （500 % 15 != 0），SDK 现成插值不适用于 15Hz 源；
- 原生 ``pluse_to_servo`` 一对一透传、无插值/无限位/无看门狗，
  发送调度必须在应用侧实现。因此本脚本自建 500Hz 插值伺服线程。

安全设计：
- 限位 clamp（复用 run.py 的 joint_limits）；
- 推理看门狗：主线程超时未提交新 chunk 时，伺服线程保持最后目标并告警；
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

# ---- 路径：SDK 实际在 /home/casbot/ct/teleop_project（run.py 里是旧路径）----
TELEOP_ROOT = Path("/home/casbot/ct/teleop_project")
VA_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DEPLOY_YAML = SCRIPT_DIR / "servo_deploy_a2a_depth_left.yaml"

for p in (TELEOP_ROOT, VA_ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# ---- 复用 run.py 的连接/观测/推理链路 ----
from run import (  # noqa: E402
    BackgroundGripperLoop,
    CameraPreviewLoop,
    PREVIEW_FPS,
    FpsObservationSampler,
    HardwareBundle,
    _build_obs_batch,
    _confirm_targets_by_feedback,
    _connect_cameras,
    _connect_dual_grippers,
    _connect_hcx_arms,
    _load_deploy_config,
    _log_inference_result,
    _log_inference_state_input,
    _pace_step,
    _ramp_start_grippers,
    _read_gripper,
    _read_hcx_joints,
    _resolve_train_config,
    _shutdown,
    _validate_runtime_contract,
    denormalize_predicted_action,
    joint_mask_from_names,
)
from robotfm.config import _normalize_rtc_config, load_config
from robotfm.train import build_policy

LEFT_ARM_DIM = 8  # L0-L6 + left_gripper


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W2 HCX 左臂 A2A 伺服部署")
    parser.add_argument("--deploy", type=str, default=str(DEFAULT_DEPLOY_YAML))
    return parser.parse_args()


class ServoInterpolator:
    """线程安全：保存最近一次推理的绝对关节目标 chunk，按时间线性插值。

    主线程 15Hz 调用 ``submit``；伺服线程 500Hz 调用 ``sample``。
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
    """500Hz 伺服输出线程：插值取目标 → 限位 → ``arm.pluse_to_servo``。

    ``watchdog_s``：距上次成功 ``submit`` 超过该值且插值已到 chunk 末尾时，
    保持最后目标持续发送（PluseToServo 必须持续刷新），并打印告警。
    """

    def __init__(
        self,
        arm: Any,
        interp: ServoInterpolator,
        *,
        rate_hz: float = 500.0,
        watchdog_s: float = 0.5,
        joint_lo: np.ndarray | None = None,
        joint_hi: np.ndarray | None = None,
    ) -> None:
        if rate_hz <= 0.0:
            raise ValueError("servo rate_hz 必须 > 0")
        self._arm = arm
        self._interp = interp
        self._period_s = 1.0 / float(rate_hz)
        self._watchdog_s = float(watchdog_s)
        self._lo = None if joint_lo is None else np.asarray(joint_lo, dtype=np.float64)
        self._hi = None if joint_hi is None else np.asarray(joint_hi, dtype=np.float64)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_submit_t = time.perf_counter()
        self._sent = 0
        self._warned_watchdog = False

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("伺服线程已在运行")
        self._thread = threading.Thread(target=self._run, name="servo-500hz", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        next_t = time.perf_counter()
        cycles: list[float] = []  # 最近窗口的实测周期（s）
        win_start = time.perf_counter()
        win_sent = 0
        while not self._stop.is_set():
            t = time.perf_counter()
            target = self._interp.sample(t)
            if target is None:
                # 尚未有首个 chunk：等待
                self._stop.wait(min(self._period_s, 0.005))
                continue
            target = np.clip(target, self._lo, self._hi) if self._lo is not None else target
            ok = bool(self._arm.pluse_to_servo(target.tolist()))
            if not ok and self._sent < 20:
                print("[WARN] pluse_to_servo 返回 false", flush=True)
            self._sent += 1
            win_sent += 1
            # 看门狗：推理超时且已到 chunk 末尾 → 保持最后目标并告警
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
            # 实测周期统计（含 pluse_to_servo 调用耗时）
            now = time.perf_counter()
            cycles.append(now - t)
            if len(cycles) > 200:
                cycles.pop(0)
            if now - win_start >= 1.0:
                arr = np.asarray(cycles, dtype=np.float64)
                actual_hz = win_sent / (now - win_start)
                over_ratio = float(np.mean(arr > self._period_s * 1.05)) * 100.0
                print(
                    f"[SERVO] 实际={actual_hz:.1f}Hz 目标={1.0 / self._period_s:.0f}Hz "
                    f"周期p50={float(np.median(arr)) * 1e3:.2f}ms "
                    f"p95={float(np.percentile(arr, 95)) * 1e3:.2f}ms "
                    f"max={float(arr.max()) * 1e3:.2f}ms 超周期5%={over_ratio:.1f}%",
                    flush=True,
                )
                if actual_hz < (1.0 / self._period_s) * 0.9:
                    print(
                        "[WARN] 实际伺服频率明显低于目标（<90%），"
                        "大概率是 pluse_to_servo 单次调用耗时过长，建议降低 servo.rate_hz",
                        flush=True,
                    )
                win_start = now
                win_sent = 0
            # 精确节拍
            next_t += self._period_s
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0.0:
                if self._stop.wait(timeout=sleep_s):
                    break
            else:
                next_t = time.perf_counter()

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
    return servo


def _resolve_teleop_yaml(deploy: dict[str, Any]) -> Path:
    """deploy.yaml 里的 teleop_yaml 可能是旧机器路径；缺失时回退到 SDK 根。"""
    p = Path(deploy["teleop_yaml"])
    if p.is_file():
        return p
    cand = TELEOP_ROOT / "teleop.yaml"
    if cand.is_file():
        print(f"[INFO] deploy teleop_yaml 不存在 {p}，回退到 {cand}", flush=True)
        return cand
    raise FileNotFoundError(f"找不到 teleop.yaml: {p}（候选 {cand} 也不存在）")


def main() -> None:
    args = _parse_args()
    deploy_path = Path(args.deploy)
    if not deploy_path.is_absolute():
        deploy_path = SCRIPT_DIR / deploy_path
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
    servo_rate_hz = float(servo_cfg.get("rate_hz", 500.0))
    watchdog_s = float(servo_cfg.get("watchdog_s", 0.5))
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
        f"servo_rate_hz={servo_rate_hz:g} watchdog_s={watchdog_s:g}",
        flush=True,
    )

    joint_lo = np.asarray(deploy["joint_limits_min_deg"], dtype=np.float64)
    joint_hi = np.asarray(deploy["joint_limits_max_deg"], dtype=np.float64)

    hw = HardwareBundle(
        left_start_joints_deg=deploy["left_start_joints_deg"],
        right_start_joints_deg=deploy["right_start_joints_deg"],
    )
    servo = None
    try:
        hw.hcx_client, hw.left_arm, hw.right_arm = _connect_hcx_arms(teleop_yaml)
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

        # 起始位：move_joints 到位（规划模式），再进入伺服
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
        print(
            f"[INFO] 观测采样已启动 fps={train_fps} n_obs={n_obs} "
            f"pre_crop={pre_crop_size} resize={resize_size} crop={crop_size}",
            flush=True,
        )

        interp = ServoInterpolator(source_fps=train_fps, n_joints=7)
        servo = ServoOutputThread(
            hw.left_arm,
            interp,
            rate_hz=servo_rate_hz,
            watchdog_s=watchdog_s,
            joint_lo=joint_lo,
            joint_hi=joint_hi,
        )
        servo.start()
        print(
            f"[INFO] 伺服线程已启动 rate_hz={servo_rate_hz:g}，"
            "开始 15Hz 推理 + 500Hz 插值下发",
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
            _log_inference_result(step_i=step_i, pred_norm=pred, pred_phys=pred_phys)
            # chunk = 未来 n_action_steps 步绝对关节目标（前 7 维关节）
            chunk_joints = np.asarray(pred_phys[:, :7], dtype=np.float64)
            # 首帧从当前物理关节角过渡到 chunk[0]，避免跳变
            q_now = _read_hcx_joints(hw.left_arm, None).astype(np.float64)
            if step_i == 0:
                chunk_joints = np.vstack([q_now[None, :], chunk_joints])
            interp.submit(chunk_joints, t0)
            servo.mark_submit()
            # 夹爪：取 chunk[0] 的 gripper 维（若存在）
            if pred_phys.shape[1] >= 8:
                left_g = float(np.clip(pred_phys[0, 7], 0.0, 1.0))
                if hw.left_gripper_loop is not None:
                    hw.left_gripper_loop.set_opening(left_g)
            _pace_step(t0, train_fps)
    except KeyboardInterrupt:
        print("\n[INFO] 收到 Ctrl+C，停止伺服", flush=True)
    finally:
        if servo is not None:
            servo.stop()
        _shutdown(hw)


if __name__ == "__main__":
    main()
