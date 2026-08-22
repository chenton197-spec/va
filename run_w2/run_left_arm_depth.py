#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

VA_ROOT = Path(__file__).resolve().parents[1]
TELEOP_ROOT = VA_ROOT / "teleop_project"
SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_YAML = SCRIPT_DIR / "deploy_left_arm_depth.yaml"
RECORD_CAMERA_ORDER = ("head", "left_hand")

if not TELEOP_ROOT.is_dir():
    raise FileNotFoundError(f"找不到 in-repo teleop_project: {TELEOP_ROOT}")

for p in (TELEOP_ROOT, VA_ROOT, SCRIPT_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from hcx_sdk import RobotClient  # noqa: E402
from orbbec_sdk import (  # noqa: E402
    AlignmentMode,
    CameraMode,
    OrbbecManager,
    OrbbecStartupError,
    load_orbbec_camera_configs,
)
from robotfm.config import _normalize_rtc_config, load_config  # noqa: E402
from robotfm.train import build_policy  # noqa: E402
from robotfm.types import Observation  # noqa: E402
from run import (  # noqa: E402
    LEFT_ARM_CAMERAS,
    LEFT_ARM_DIM,
    BackgroundGripperLoop,
    CameraPreviewLoop,
    HardwareBundle,
    PREVIEW_FPS,
    StepObservationQueue,
    _apply_rtc_overrides,
    _build_obs_batch,
    _clamp_joints_by_limits,
    _clamp_joints_by_max_delta,
    _load_deploy_config,
    _log_inference_result,
    _log_inference_state_input,
    _log_max_delta_clip,
    _pace_step,
    _prepare_observation,
    _pump_camera_preview,
    _ramp_start_grippers,
    _read_gripper,
    _read_hcx_joints,
    _resolve_path,
    _resolve_train_config,
    _shutdown,
    _validate_gloria_dual_config,
    _validate_runtime_contract,
    denormalize_predicted_action,
    joint_mask_from_names,
)
from teleop_sdk.adapters.gloria_m import GloriaMGripperFollower  # noqa: E402
from teleop_sdk.config import load_runtime_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W2 左臂 RGB-D + 头相机 MoveJ 闭环")
    parser.add_argument("--deploy", type=str, default=str(DEPLOY_YAML))
    return parser.parse_args()


def _as_uint8_rgb(rgb: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise RuntimeError(f"相机 {name} RGB 形状异常: {arr.shape}")
    return np.ascontiguousarray(arr)


def _as_uint16_depth(raw: np.ndarray, meters: float, name: str) -> tuple[np.ndarray, float]:
    depth = np.asarray(raw)
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise RuntimeError(f"相机 {name} 深度形状异常: {depth.shape} {depth.dtype}")
    scale_m = float(meters)
    if not math.isfinite(scale_m) or scale_m <= 0.0:
        raise RuntimeError(f"相机 {name} meters_per_raw_unit 无效: {meters}")
    return np.ascontiguousarray(depth), scale_m * 1000.0


def _describe_rgbd_issue(cam: Any, name: str, *, need_depth: bool) -> str:
    serial = getattr(getattr(cam, "config", None), "serial_number", None) or "?"
    status = getattr(cam, "status", None)
    status_s = getattr(status, "value", status)
    last_error = getattr(cam, "last_error", None)
    try:
        frame = cam.get_frame()
    except Exception as exc:
        return (
            f"{name}(sn={serial}): get_frame 异常 {type(exc).__name__}: {exc}; "
            f"status={status_s}; last_error={last_error}"
        )
    if frame is None:
        reason = "get_frame()=None"
    else:
        parts = []
        if getattr(frame, "rgb", None) is None:
            parts.append("rgb=None")
        else:
            rgb = np.asarray(frame.rgb)
            parts.append(f"rgb={rgb.shape}/{rgb.dtype}")
        if need_depth:
            depth = getattr(frame, "depth", None)
            scale = getattr(frame, "meters_per_raw_unit", None)
            if depth is None:
                parts.append("depth=None")
            else:
                d = np.asarray(depth)
                parts.append(f"depth={d.shape}/{d.dtype}")
            parts.append(f"meters_per_raw_unit={scale}")
        reason = ", ".join(parts)
    detail = f"{name}(sn={serial}): {reason}; status={status_s}"
    if last_error:
        detail += f"; last_error={last_error}"
    return detail


class HeadTriggeredCapture:
    def __init__(self, camera_manager: Any) -> None:
        self._manager = camera_manager
        self._head = camera_manager.camera("head")
        self._left = camera_manager.camera("left_hand")
        self._head_seq = int(self._head.latest_sequence())
        self._logged_scale = False

    def capture(self, *, timeout_s: float = 2.0) -> dict[str, np.ndarray]:
        self._head_seq = int(self._head.latest_sequence())
        deadline = time.perf_counter() + timeout_s
        head_item = None
        while time.perf_counter() < deadline:
            head_item = self._head.get_next_frame_after(self._head_seq)
            if head_item is not None:
                break
            time.sleep(0.005)
        if head_item is None:
            details = [
                _describe_rgbd_issue(self._head, "head", need_depth=False),
                _describe_rgbd_issue(self._left, "left_hand", need_depth=True),
            ]
            raise TimeoutError(
                f"等待 head 新帧超时 ({timeout_s}s)，对齐采集：head 驱动。\n"
                + "\n".join(f"  - {line}" for line in details)
            )
        seq, head_frame = head_item
        self._head_seq = int(seq)
        if head_frame is None or head_frame.rgb is None:
            raise RuntimeError("head 新帧缺少 RGB")
        master_ns = int(head_frame.capture_monotonic_ns)

        left_frame = None
        while time.perf_counter() < deadline:
            candidate = self._left.get_frame_at_or_before(master_ns)
            if (
                candidate is not None
                and candidate.rgb is not None
                and candidate.depth is not None
                and candidate.meters_per_raw_unit is not None
                and int(candidate.capture_monotonic_ns) <= master_ns
            ):
                left_frame = candidate
                break
            time.sleep(0.005)
        if left_frame is None:
            details = [
                _describe_rgbd_issue(self._head, "head", need_depth=False),
                _describe_rgbd_issue(self._left, "left_hand", need_depth=True),
            ]
            raise TimeoutError(
                f"left_hand 没有不晚于 head 的 RGB-D 帧 (head_ns={master_ns})。\n"
                + "\n".join(f"  - {line}" for line in details)
            )

        depth, scale_mm = _as_uint16_depth(
            left_frame.depth, float(left_frame.meters_per_raw_unit), "left_hand"
        )
        rgb_left = _as_uint8_rgb(left_frame.rgb, "left_hand")
        if rgb_left.shape[:2] != depth.shape:
            raise RuntimeError(
                "left_hand RGB 与 depth 尺寸不一致（采集要求 software 对齐后同尺寸）: "
                f"rgb={rgb_left.shape} depth={depth.shape}"
            )
        if not self._logged_scale:
            print(
                f"[INFO] observation.depth.left_hand scale={scale_mm:g} mm/raw-unit "
                f"（与 openarm_hcx_dual_arm_record / depth_sources.json 相同）",
                flush=True,
            )
            self._logged_scale = True
        return {
            "head": _as_uint8_rgb(head_frame.rgb, "head"),
            "left_hand": rgb_left,
        }


def _connect_hcx_left_arm(teleop_yaml: Path) -> tuple[Any, Any]:
    runtime = load_runtime_config(teleop_yaml)
    h = runtime.hcx
    client = RobotClient(h.local_ip, h.remote_ip, h.port)
    client.connect(timeout_s=h.connect_timeout_s)
    time.sleep(float(h.controller_initialization_wait_s))

    if h.auto_detach_hmi and not client.hmi_detached:
        client.detach_hmi()
    if h.auto_clear_alarms:
        for _ in range(int(h.alarm_clear_retry_count) + 1):
            if not client.active_alarms:
                break
            client.clear_alarms()
            time.sleep(float(h.alarm_clear_retry_interval_s))
    if client.active_alarms:
        raise RuntimeError(f"HCX 活动报警未清除: {client.active_alarms}")
    if not client.soft_emergency_stop_normal:
        raise RuntimeError("HCX soft emergency stop 非正常状态")

    for master in h.ethercat_master_indices:
        if not client.ethercat_master_operational(master):
            raise RuntimeError(f"HCX EtherCAT 主站未 OP: {master}")

    if h.auto_enable and not client.global_enabled:
        client.set_global_enable(True)
    if not client.global_enabled:
        raise RuntimeError("HCX global enable=false，请现场使能或开启 hcx.auto_enable")

    left_arm = client.arm(h.left_robot_id)
    if h.auto_enable and not left_arm.enabled:
        left_arm.set_enabled(True)
    if not left_arm.enabled:
        raise RuntimeError("HCX 左臂未使能，请现场使能或开启 hcx.auto_enable")
    print("[INFO] 仅使能左臂；右臂不运动", flush=True)
    return client, left_arm


def _connect_record_cameras(teleop_yaml: Path) -> Any:
    all_configs = load_orbbec_camera_configs(teleop_yaml, section_name="hcx_orbbec")
    by_name = {c.name: c for c in all_configs}
    missing = [n for n in RECORD_CAMERA_ORDER if n not in by_name]
    if missing:
        raise ValueError(
            f"teleop.yaml 的 hcx_orbbec 缺少采集所需相机: {missing} "
            f"(已声明: {sorted(by_name)})"
        )
    configs = []
    for name in RECORD_CAMERA_ORDER:
        camera = by_name[name]
        depth_on = name != "head"
        configs.append(
            replace(
                camera,
                mode=CameraMode.RGBD if depth_on else CameraMode.RGB,
                alignment=AlignmentMode.SOFTWARE if depth_on else AlignmentMode.NONE,
                first_frame_timeout_s=max(float(camera.first_frame_timeout_s), 15.0),
            )
        )
    print(
        "[INFO] 启动 Orbbec（对齐 openarm_hcx_dual_arm_record）: "
        + ", ".join(
            f"{cfg.name}({cfg.serial_number}) mode={cfg.mode.value} "
            f"align={cfg.alignment.value}"
            for cfg in configs
        ),
        flush=True,
    )
    last_error: Exception | None = None
    for attempt in range(1, 4):
        manager = OrbbecManager(tuple(configs))
        try:
            manager.start()
            return manager
        except OrbbecStartupError as exc:
            last_error = exc
            print(f"[WARN] 相机启动失败 ({attempt}/3): {exc}", flush=True)
            try:
                manager.stop()
            except Exception:
                pass
            time.sleep(1.5 * attempt)
    assert last_error is not None
    raise last_error


def _connect_left_gripper(teleop_yaml: Path) -> tuple[Any | None, float]:
    runtime = load_runtime_config(teleop_yaml)
    left_enabled, _right_enabled, rate_hz = _validate_gloria_dual_config(runtime)
    if not left_enabled:
        return None, rate_hz
    left = GloriaMGripperFollower(runtime.gloria_m_dual.side_config("left"))
    left.connect()
    return left, rate_hz


def _read_left_observation(
    hw: HardwareBundle,
    cameras: list[str],
    *,
    last_state: np.ndarray | None,
) -> Observation:
    capture = getattr(hw, "camera_capture", None)
    if capture is None:
        raise RuntimeError("相机采集器未初始化")
    images = capture.capture()
    missing = [n for n in cameras if n not in images]
    if missing:
        raise RuntimeError(f"采图缺少相机: {missing}")
    images = {n: images[n] for n in cameras}
    left_fb = last_state[:7].tolist() if last_state is not None else hw.left_start_joints_deg
    left = _read_hcx_joints(hw.left_arm, left_fb)
    lg_fb = float(last_state[7]) if last_state is not None else 1.0
    left_g = _read_gripper(hw.left_gripper, fallback=lg_fb)
    state = np.concatenate(
        [left.astype(np.float32), np.asarray([left_g], dtype=np.float32)]
    )
    return Observation(images=images, state=state, timestamp=time.time())


class LeftStepObservationQueue(StepObservationQueue):
    def _capture(self) -> Observation:
        last_state = (
            None
            if not self._history
            else np.asarray(self._history[-1].state, dtype=np.float32)
        )
        obs = _read_left_observation(self._hw, self._cameras, last_state=last_state)
        obs.validate(self._cameras, self._state_dim)
        return _prepare_observation(
            obs,
            pre_crop_size=self._pre_crop_size,
            resize_size=self._resize_size,
            crop_size=self._crop_size,
            eval_fixed_crop=self._eval_fixed_crop,
        )


def _confirm_left_by_feedback(
    hw: HardwareBundle,
    left_target: list[float],
    *,
    timeout_s: float,
    poll_interval_s: float,
    angle_tolerance_deg: float,
) -> None:
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("move_joints.feedback_confirm_timeout_s 必须是正的有限秒数")
    if not math.isfinite(poll_interval_s) or poll_interval_s <= 0.0:
        raise ValueError("move_joints.feedback_confirm_poll_interval_s 必须是正的有限秒数")
    if not math.isfinite(angle_tolerance_deg) or angle_tolerance_deg <= 0.0:
        raise ValueError("move_joints.angle_tolerance_deg 必须是正的有限数")
    if hw.left_arm is None:
        raise RuntimeError("HCX 左臂未连接，无法确认反馈到位")

    deadline = time.monotonic() + timeout_s
    left_target_np = np.asarray(left_target, dtype=np.float64)
    while True:
        left_fb = np.asarray(hw.left_arm.joint_angles(), dtype=np.float64)
        if np.all(np.abs(left_fb - left_target_np) <= angle_tolerance_deg):
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise RuntimeError("move_joints 反馈确认超时：左臂关节未在容差内到位")
        _pump_camera_preview(hw)
        time.sleep(min(poll_interval_s, remaining))


def _send_left_action(
    hw: HardwareBundle,
    action: np.ndarray,
    *,
    speed_ratio: float | None,
    acceleration_seconds: float | None,
    deceleration_seconds: float | None,
    feedback_confirm_timeout_s: float,
    feedback_confirm_poll_interval_s: float,
    angle_tolerance_deg: float,
    max_delta_deg: float,
    joint_limits_min_deg: np.ndarray,
    joint_limits_max_deg: np.ndarray,
    move_interrupt: bool,
) -> None:
    if hw.left_arm is None:
        raise RuntimeError("HCX 左臂未连接")
    if not math.isfinite(max_delta_deg) or max_delta_deg <= 0.0:
        raise ValueError("move_joints.max_delta_deg 必须是正的有限数")
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.shape != (LEFT_ARM_DIM,):
        raise ValueError(f"左臂动作必须是 {LEFT_ARM_DIM} 维 [L7, L_grip]，实际 {a.shape}")

    left_current = np.asarray(hw.left_arm.joint_angles(), dtype=np.float64)
    left, left_delta = _clamp_joints_by_max_delta(a[:7], left_current, max_delta_deg)
    _log_max_delta_clip("left", left_delta, max_delta_deg=max_delta_deg)
    left = _clamp_joints_by_limits(
        left, joint_limits_min_deg, joint_limits_max_deg, side="left"
    )
    left_g = float(np.clip(a[7], 0.0, 1.0))
    print(
        f"  left={[round(v, 3) for v in left]} left_gripper={left_g:.4f}",
        flush=True,
    )

    if hw.left_gripper_loop is not None:
        hw.left_gripper_loop.set_opening(left_g)
    elif hw.left_gripper is not None:
        _ = hw.left_gripper.send_normalized(left_g)

    _ = hw.left_arm.move_joints(
        left,
        interrupt=move_interrupt,
        acceleration_seconds=acceleration_seconds,
        deceleration_seconds=deceleration_seconds,
        speed_ratio=speed_ratio,
        smooth=1,
        wait=False,
    )
    _confirm_left_by_feedback(
        hw,
        left,
        timeout_s=feedback_confirm_timeout_s,
        poll_interval_s=feedback_confirm_poll_interval_s,
        angle_tolerance_deg=angle_tolerance_deg,
    )


def main() -> None:
    args = _parse_args()
    deploy_path = Path(args.deploy)
    if not deploy_path.is_absolute():
        deploy_path = _resolve_path(deploy_path, base=VA_ROOT)
    deploy = _load_deploy_config(deploy_path)

    ckpt_path = deploy["checkpoint"]
    teleop_yaml = deploy["teleop_yaml"]
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")
    if not teleop_yaml.is_file():
        raise FileNotFoundError(f"找不到 teleop.yaml: {teleop_yaml}")

    train_cfg_path = _resolve_train_config(ckpt_path, deploy["config"])
    if train_cfg_path is not None and not train_cfg_path.is_file():
        raise FileNotFoundError(f"找不到训练配置: {train_cfg_path}")

    print(f"[INFO] 部署配置: {deploy_path}")
    print("[INFO] 控制: HCX 左臂 MoveJ（到位后采图）")
    print(f"[INFO] teleop SDK: {TELEOP_ROOT}")
    print(f"[INFO] teleop.yaml: {teleop_yaml}")
    print(
        "[INFO] 关节限位 min="
        f"{[round(float(v), 1) for v in deploy['joint_limits_min_deg']]} "
        "max="
        f"{[round(float(v), 1) for v in deploy['joint_limits_max_deg']]}"
    )
    print(f"[INFO] 加载 checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if train_cfg_path is not None:
        cfg = load_config(train_cfg_path)
        print(f"[INFO] 训练配置: {train_cfg_path}")
    else:
        cfg = ckpt["config"]
        print("[INFO] 训练配置: checkpoint 内嵌 config")
    stats = ckpt["stats"]
    cameras = list(cfg.cameras)
    layout = _validate_runtime_contract(cfg, cameras, stats)
    if layout != "left" or tuple(cameras) != LEFT_ARM_CAMERAS:
        raise ValueError(
            f"本脚本仅支持左臂 cameras={list(LEFT_ARM_CAMERAS)} 8D，"
            f"实际 cameras={cameras} layout={layout}"
        )

    _apply_rtc_overrides(cfg, deploy.get("rtc") or {})
    rtc_cfg = _normalize_rtc_config(cfg.policy.rtc)
    if bool(rtc_cfg.enabled):
        raise ValueError("本脚本仅支持到位采图，不能开 RTC")

    train_history_noise = float(getattr(cfg.policy, "history_noise_std", 0.0) or 0.0)
    cfg.policy.history_noise_std = 0.0

    norm_mode = cfg.dataset.norm_mode
    n_obs = int(cfg.dataset.n_obs_steps)
    n_action_steps = int(cfg.policy.n_action_steps)
    exec_action_steps = deploy["exec_action_steps"]
    if exec_action_steps is None:
        exec_action_steps = n_action_steps
    if exec_action_steps > n_action_steps:
        raise ValueError(
            f"deploy.yaml exec_action_steps={exec_action_steps} 不能大于 "
            f"policy.n_action_steps={n_action_steps}"
        )
    max_steps = deploy["max_steps"]
    train_fps = int(cfg.fps)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    policy = build_policy(cfg, stats)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()
    policy_cfg = getattr(policy, "cfg", None)
    if hasattr(policy_cfg, "history_noise_std"):
        policy_cfg.history_noise_std = 0.0
    print(
        f"[INFO] 策略已就绪 layout=left device={device} cameras={cameras} "
        f"norm={norm_mode} n_obs={n_obs} n_action_steps={n_action_steps} "
        f"exec_action_steps={exec_action_steps} train_fps={train_fps} "
        f"obs_mode=after_action history_noise_std=0 (train={train_history_noise:g})",
        flush=True,
    )

    hw = HardwareBundle(
        left_start_joints_deg=deploy["left_start_joints_deg"],
        right_start_joints_deg=None,
    )
    try:
        hw.hcx_client, hw.left_arm = _connect_hcx_left_arm(teleop_yaml)
        hw.left_gripper, gripper_rate_hz = _connect_left_gripper(teleop_yaml)
        if hw.left_gripper is not None:
            hold_left = _read_gripper(hw.left_gripper, fallback=1.0)
            hw.left_gripper_loop = BackgroundGripperLoop(
                hw.left_gripper, rate_hz=gripper_rate_hz
            )
            hw.left_gripper_loop.start(initial_opening=hold_left)
        hw.camera_manager = _connect_record_cameras(teleop_yaml)
        hw.camera_capture = HeadTriggeredCapture(hw.camera_manager)
        print(
            "[INFO] 采图对齐 openarm_hcx_dual_arm_record："
            "head 新帧触发，left_hand 取 at_or_before 的 RGB-D",
            flush=True,
        )
        if deploy["display_cameras"]:
            hw.camera_preview = CameraPreviewLoop(
                hw.camera_manager, list(cameras), fps=PREVIEW_FPS
            )
            hw.camera_preview.start()
            if hw.camera_preview is not None and hw.camera_preview.is_active:
                print(
                    f"[INFO] 相机预览窗口已启动 fps={int(PREVIEW_FPS)} cameras={list(cameras)}"
                )
            else:
                hw.camera_preview = None

        if deploy["left_start_joints_deg"] is not None:
            left_start = _clamp_joints_by_limits(
                deploy["left_start_joints_deg"],
                deploy["joint_limits_min_deg"],
                deploy["joint_limits_max_deg"],
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
            _confirm_left_by_feedback(
                hw,
                left_start,
                timeout_s=deploy["move_feedback_confirm_timeout_s"],
                poll_interval_s=deploy["move_feedback_confirm_poll_interval_s"],
                angle_tolerance_deg=deploy["move_angle_tolerance_deg"],
            )

        _ramp_start_grippers(
            hw,
            left_target=deploy["left_start_gripper"],
            right_target=None,
            duration_s=deploy["start_gripper_ramp_s"],
            rate_hz=gripper_rate_hz,
        )

        pre_crop_size = cfg.dataset.pre_crop_size
        resize_size = cfg.dataset.resize_size
        crop_size = cfg.dataset.crop_size
        eval_fixed_crop = bool(cfg.dataset.eval_fixed_crop)
        obs_queue = LeftStepObservationQueue(
            hw,
            list(cameras),
            n_obs_steps=n_obs,
            state_dim=int(cfg.state_dim),
            pre_crop_size=pre_crop_size,
            resize_size=resize_size,
            crop_size=crop_size,
            eval_fixed_crop=eval_fixed_crop,
        )
        obs_queue.fill_initial()
        print(
            f"[INFO] 到位观测队列已就绪 n_obs={n_obs}："
            "head 新帧驱动，left_hand RGB-D at_or_before",
            flush=True,
        )
        print(
            f"[INFO] 开始闭环：每次推理执行 chunk 前 {exec_action_steps}/{n_action_steps} 步，"
            f"每步到位后等新帧，最多 {max_steps} 步",
            flush=True,
        )

        step_i = 0
        while step_i < max_steps:
            _pump_camera_preview(hw)
            obs_history = obs_queue.snapshot()
            batch = _build_obs_batch(
                obs_history,
                cameras=list(cameras),
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
            with torch.no_grad():
                pred = policy.sample_actions(batch)[0].cpu()
            pred_phys = np.asarray(
                denormalize_predicted_action(
                    pred,
                    stats,
                    norm_mode,
                    q_now_phys=np.asarray(obs_history[-1].state, dtype=np.float32),
                    predict_joint_delta=bool(cfg.policy.predict_joint_delta),
                    joint_mask=joint_mask_from_names(cfg.action_names, cfg.action_dim),
                )
            )
            _log_inference_result(step_i=step_i, pred_norm=pred, pred_phys=pred_phys)
            n_chunk = int(pred_phys.shape[0])
            n_exec = min(exec_action_steps, n_chunk, max_steps - step_i)
            print(
                f"[INFO] step={step_i} 推理 chunk={n_chunk} 执行前 {n_exec} 步"
                + (f"，丢弃其余 {n_chunk - n_exec}" if n_exec < n_chunk else "")
            )
            for k in range(n_exec):
                t0 = time.perf_counter()
                action = np.asarray(pred_phys[k], dtype=np.float32)
                print(
                    f"[INFO] step={step_i} 执行 chunk[{k}/{n_exec}]",
                    end="",
                )
                _send_left_action(
                    hw,
                    action,
                    speed_ratio=deploy["move_speed_ratio"],
                    acceleration_seconds=deploy["move_acceleration_seconds"],
                    deceleration_seconds=deploy["move_deceleration_seconds"],
                    feedback_confirm_timeout_s=deploy["move_feedback_confirm_timeout_s"],
                    feedback_confirm_poll_interval_s=deploy[
                        "move_feedback_confirm_poll_interval_s"
                    ],
                    angle_tolerance_deg=deploy["move_angle_tolerance_deg"],
                    max_delta_deg=deploy["move_max_delta_deg"],
                    joint_limits_min_deg=deploy["joint_limits_min_deg"],
                    joint_limits_max_deg=deploy["joint_limits_max_deg"],
                    move_interrupt=deploy["move_interrupt"],
                )
                obs_queue.push_after_action()
                step_i += 1
                _pump_camera_preview(hw)
                _pace_step(t0, train_fps)
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        _shutdown(hw)


if __name__ == "__main__":
    main()
