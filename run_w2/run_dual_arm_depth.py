#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

VA_ROOT = Path(__file__).resolve().parents[1]
TELEOP_ROOT = VA_ROOT.parent / "teleop_project"
SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_YAML = SCRIPT_DIR / "deploy_fm_statedelta_depth.yaml"
RECORD_CAMERA_ORDER = ("head", "left_hand", "right_hand")
RECORD_RGB_WH = (640, 480)
RECORD_DEPTH_WH = (640, 480)
RECORD_FPS = 30

if not TELEOP_ROOT.is_dir():
    raise FileNotFoundError(f"找不到 teleop_project: {TELEOP_ROOT}")

for p in (TELEOP_ROOT, VA_ROOT, SCRIPT_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from orbbec_sdk import (  # noqa: E402
    AlignmentMode,
    CameraMode,
    OrbbecManager,
    OrbbecStartupError,
    load_orbbec_camera_configs,
)
from robotfm.config import _normalize_rtc_config, load_config  # noqa: E402
from robotfm.data.action_delta import (  # noqa: E402
    denormalize_predicted_action,
    flow_history_from_phys,
    joint_mask_from_names,
)
from robotfm.data.lerobot_dataset import _raw_depth_to_normalized, _resize_hw  # noqa: E402
from robotfm.data.stats import normalize  # noqa: E402
from robotfm.train import build_policy  # noqa: E402
from robotfm.types import Observation  # noqa: E402
from run import (  # noqa: E402
    DUAL_ARM_CAMERAS,
    BackgroundGripperLoop,
    CameraPreviewLoop,
    HardwareBundle,
    PREVIEW_FPS,
    StepObservationQueue,
    _apply_rtc_overrides,
    _clamp_joints_by_limits,
    _confirm_targets_by_feedback,
    _connect_dual_grippers,
    _connect_hcx_arms,
    _load_deploy_config,
    _log_inference_result,
    _log_inference_state_input,
    _pump_camera_preview,
    _ramp_start_grippers,
    _read_gripper,
    _read_hcx_joints,
    _resolve_path,
    _resolve_train_config,
    _send_action,
    _shutdown,
    _validate_runtime_contract,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W2 双臂 RGB-D MoveJ 闭环")
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


def _check_scale_mm_per_raw_unit(cam: str, scale_mm: float, expected: dict[str, float]) -> None:
    if cam not in expected:
        raise RuntimeError(f"训练配置缺少 dataset.scale_mm_per_raw_unit[{cam}]")
    exp = float(expected[cam])
    if not math.isclose(scale_mm, exp, rel_tol=1e-3, abs_tol=1e-6):
        raise RuntimeError(
            f"{cam} scale_mm_per_raw_unit={scale_mm:g} != 训练配置 {exp:g}"
        )


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
    def __init__(
        self,
        camera_manager: Any,
        depth_cameras: tuple[str, ...] | list[str],
        expected_scale_mm: dict[str, float],
    ) -> None:
        self._manager = camera_manager
        self._head = camera_manager.camera("head")
        self._hands = {name: camera_manager.camera(name) for name in depth_cameras}
        self._head_seq = int(self._head.latest_sequence())
        self._expected_scale_mm = expected_scale_mm
        self._logged_scale: set[str] = set()

    def capture(
        self, *, timeout_s: float = 2.0
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        self._head_seq = int(self._head.latest_sequence())
        deadline = time.perf_counter() + timeout_s
        head_item = None
        while time.perf_counter() < deadline:
            head_item = self._head.get_next_frame_after(self._head_seq)
            if head_item is not None:
                break
            time.sleep(0.005)
        if head_item is None:
            details = [_describe_rgbd_issue(self._head, "head", need_depth=False)]
            details.extend(
                _describe_rgbd_issue(cam, name, need_depth=True)
                for name, cam in self._hands.items()
            )
            raise TimeoutError(
                f"等待 head 新帧超时 ({timeout_s}s)。\n"
                + "\n".join(f"  - {line}" for line in details)
            )
        seq, head_frame = head_item
        self._head_seq = int(seq)
        if head_frame is None or head_frame.rgb is None:
            raise RuntimeError("head 新帧缺少 RGB")
        master_ns = int(head_frame.capture_monotonic_ns)

        images = {"head": _as_uint8_rgb(head_frame.rgb, "head")}
        depths: dict[str, np.ndarray] = {}
        for name, cam in self._hands.items():
            hand_frame = None
            while time.perf_counter() < deadline:
                candidate = cam.get_frame_at_or_before(master_ns)
                if (
                    candidate is not None
                    and candidate.rgb is not None
                    and candidate.depth is not None
                    and candidate.meters_per_raw_unit is not None
                    and int(candidate.capture_monotonic_ns) <= master_ns
                ):
                    hand_frame = candidate
                    break
                time.sleep(0.005)
            if hand_frame is None:
                details = [_describe_rgbd_issue(self._head, "head", need_depth=False)]
                details.extend(
                    _describe_rgbd_issue(c, n, need_depth=True)
                    for n, c in self._hands.items()
                )
                raise TimeoutError(
                    f"{name} 没有不晚于 head 的 RGB-D 帧 (head_ns={master_ns})。\n"
                    + "\n".join(f"  - {line}" for line in details)
                )
            depth, scale_mm = _as_uint16_depth(
                hand_frame.depth, float(hand_frame.meters_per_raw_unit), name
            )
            _check_scale_mm_per_raw_unit(name, scale_mm, self._expected_scale_mm)
            rgb_hand = _as_uint8_rgb(hand_frame.rgb, name)
            if rgb_hand.shape[:2] != depth.shape:
                raise RuntimeError(
                    f"{name} RGB 与 depth 尺寸不一致: rgb={rgb_hand.shape} depth={depth.shape}"
                )
            if name not in self._logged_scale:
                print(
                    f"[INFO] observation.depth.{name} scale={scale_mm:g} mm/raw-unit",
                    flush=True,
                )
                self._logged_scale.add(name)
            images[name] = rgb_hand
            depths[name] = depth
        return images, depths


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
        kwargs: dict[str, Any] = dict(
            mode=CameraMode.RGBD if depth_on else CameraMode.RGB,
            alignment=AlignmentMode.SOFTWARE if depth_on else AlignmentMode.NONE,
            rgb_resolution=RECORD_RGB_WH,
            fps=RECORD_FPS,
            first_frame_timeout_s=max(float(camera.first_frame_timeout_s), 15.0),
        )
        if depth_on:
            kwargs["depth_resolution"] = RECORD_DEPTH_WH
        configs.append(replace(camera, **kwargs))
    print(
        "[INFO] 启动 Orbbec: "
        + ", ".join(
            f"{cfg.name}({cfg.serial_number}) mode={cfg.mode.value} "
            f"rgb={cfg.rgb_resolution} depth={cfg.depth_resolution} fps={cfg.fps} "
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


def _prepare_rgbd_observation(
    obs: Observation,
    *,
    image_size: int | list[int] | None,
    depth_cameras: list[str],
    depth_min_mm: float,
    depth_max_mm: float,
    expected_scale_mm: dict[str, float],
) -> Observation:
    images: dict[str, np.ndarray] = {}
    for name, rgb in obs.images.items():
        arr = _as_uint8_rgb(rgb, name)
        arr = _resize_hw(arr, image_size, cv2.INTER_AREA)
        images[name] = (arr.astype(np.float32) / 255.0)
    depths: dict[str, np.ndarray] = {}
    src = obs.depths or {}
    for name in depth_cameras:
        if name not in src:
            raise KeyError(f"Missing depth camera '{name}'")
        raw = np.asarray(src[name])
        raw = _resize_hw(raw, image_size, cv2.INTER_NEAREST)
        norm = _raw_depth_to_normalized(
            raw,
            float(expected_scale_mm[name]),
            float(depth_min_mm),
            float(depth_max_mm),
        )
        depths[name] = np.asarray(norm[0], dtype=np.float32)
    return Observation(
        images=images,
        state=np.asarray(obs.state, dtype=np.float32),
        timestamp=obs.timestamp,
        depths=depths,
    )


def _build_obs_batch(
    obs_history: list[Observation],
    cameras: list[str],
    n_obs_steps: int,
    stats: dict,
    norm_mode: str,
    device: torch.device,
    *,
    depth_cameras: list[str],
    predict_joint_delta: bool,
    joint_mask: np.ndarray,
) -> dict[str, torch.Tensor]:
    history = obs_history[-n_obs_steps:]
    while len(history) < n_obs_steps:
        history.insert(0, history[0])

    camera_histories = []
    for cam in cameras:
        frames = []
        for obs in history:
            img = np.asarray(obs.images[cam], dtype=np.float32)
            frames.append(torch.from_numpy(np.transpose(img, (2, 0, 1))))
        camera_histories.append(torch.stack(frames, dim=0))

    states = [
        torch.from_numpy(
            normalize(obs.state.astype(np.float32), stats, prefix="state", mode=norm_mode)
        )
        for obs in history
    ]
    state_phys = np.stack(
        [np.asarray(obs.state, dtype=np.float32) for obs in history], axis=0
    )
    batch = {
        "obs_images": torch.stack(camera_histories, dim=0).unsqueeze(0).to(device),
        "obs_state": torch.stack(states, dim=0).unsqueeze(0).to(device),
        "obs_history": torch.from_numpy(
            flow_history_from_phys(
                state_phys,
                stats,
                norm_mode,
                predict_joint_delta=bool(predict_joint_delta),
                joint_mask=joint_mask,
            )
        )
        .unsqueeze(0)
        .to(device),
    }
    if depth_cameras:
        depth_histories = []
        for cam in depth_cameras:
            frames = []
            for obs in history:
                if not obs.depths or cam not in obs.depths:
                    raise KeyError(f"Missing depth camera '{cam}' in observation.depths")
                d = np.asarray(obs.depths[cam], dtype=np.float32)
                if d.ndim == 2:
                    d = d[None, ...]
                frames.append(torch.from_numpy(np.ascontiguousarray(d)))
            depth_histories.append(torch.stack(frames, dim=0))
        batch["obs_depth"] = torch.stack(depth_histories, dim=0).unsqueeze(0).to(device)
    return batch


def _read_dual_observation(
    hw: HardwareBundle,
    cameras: list[str],
    *,
    last_state: np.ndarray | None,
) -> Observation:
    capture = getattr(hw, "camera_capture", None)
    if capture is None:
        raise RuntimeError("相机采集器未初始化")
    images, depths = capture.capture()
    missing = [n for n in cameras if n not in images]
    if missing:
        raise RuntimeError(f"采图缺少相机: {missing}")
    images = {n: images[n] for n in cameras}
    left_fb = last_state[:7].tolist() if last_state is not None else hw.left_start_joints_deg
    right_fb = (
        last_state[7:14].tolist() if last_state is not None else hw.right_start_joints_deg
    )
    left = _read_hcx_joints(hw.left_arm, left_fb)
    right = _read_hcx_joints(hw.right_arm, right_fb)
    lg_fb = float(last_state[14]) if last_state is not None else 1.0
    rg_fb = float(last_state[15]) if last_state is not None else 1.0
    left_g = _read_gripper(hw.left_gripper, fallback=lg_fb)
    right_g = _read_gripper(hw.right_gripper, fallback=rg_fb)
    state = np.concatenate(
        [
            left.astype(np.float32),
            right.astype(np.float32),
            np.asarray([left_g, right_g], dtype=np.float32),
        ]
    )
    return Observation(images=images, state=state, timestamp=time.time(), depths=depths)


class DualStepObservationQueue(StepObservationQueue):
    def __init__(
        self,
        hw: HardwareBundle,
        cameras: list[str],
        *,
        n_obs_steps: int,
        state_dim: int,
        image_size: int | list[int] | None,
        depth_cameras: list[str],
        depth_min_mm: float,
        depth_max_mm: float,
        expected_scale_mm: dict[str, float],
    ) -> None:
        super().__init__(
            hw,
            cameras,
            n_obs_steps=n_obs_steps,
            state_dim=state_dim,
            image_size=image_size,
        )
        self._depth_cameras = list(depth_cameras)
        self._depth_min_mm = float(depth_min_mm)
        self._depth_max_mm = float(depth_max_mm)
        self._expected_scale_mm = dict(expected_scale_mm)

    def _capture(self) -> Observation:
        last_state = (
            None
            if not self._history
            else np.asarray(self._history[-1].state, dtype=np.float32)
        )
        obs = _read_dual_observation(self._hw, self._cameras, last_state=last_state)
        obs.validate(self._cameras, self._state_dim)
        return _prepare_rgbd_observation(
            obs,
            image_size=self._image_size,
            depth_cameras=self._depth_cameras,
            depth_min_mm=self._depth_min_mm,
            depth_max_mm=self._depth_max_mm,
            expected_scale_mm=self._expected_scale_mm,
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
    print("[INFO] 控制: HCX 双臂 MoveJ（到位后采图）")
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
    if layout != "dual" or tuple(cameras) != DUAL_ARM_CAMERAS:
        raise ValueError(
            f"本脚本仅支持双臂 cameras={list(DUAL_ARM_CAMERAS)} 16D，"
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
        exec_action_steps = 1
    if exec_action_steps > n_action_steps:
        raise ValueError(
            f"deploy.yaml exec_action_steps={exec_action_steps} 不能大于 "
            f"policy.n_action_steps={n_action_steps}"
        )
    max_steps = deploy["max_steps"]
    predict_joint_delta = bool(cfg.policy.predict_joint_delta)
    joint_mask = joint_mask_from_names(cfg.action_names, cfg.action_dim)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    policy = build_policy(cfg, stats)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()
    policy_cfg = getattr(policy, "cfg", None)
    if hasattr(policy_cfg, "history_noise_std"):
        policy_cfg.history_noise_std = 0.0
    print(
        f"[INFO] 策略已就绪 layout=dual device={device} cameras={cameras} "
        f"norm={norm_mode} n_obs={n_obs} n_action_steps={n_action_steps} "
        f"exec_action_steps={exec_action_steps} "
        f"obs_mode=after_action history_noise_std=0 (train={train_history_noise:g})",
        flush=True,
    )

    depth_cameras = list(getattr(cfg.dataset, "depth_cameras", ()) or ())
    expected_scale_mm = dict(getattr(cfg.dataset, "scale_mm_per_raw_unit", None) or {})
    if not depth_cameras:
        raise ValueError("本脚本需要 dataset.depth_cameras")
    for cam in depth_cameras:
        if cam not in expected_scale_mm:
            raise ValueError(f"训练配置缺少 dataset.scale_mm_per_raw_unit[{cam}]")
    depth_min_mm = float(cfg.dataset.depth_min_mm)
    depth_max_mm = float(cfg.dataset.depth_max_mm)

    hw = HardwareBundle(
        left_start_joints_deg=deploy["left_start_joints_deg"],
        right_start_joints_deg=deploy["right_start_joints_deg"],
    )
    try:
        hw.hcx_client, hw.left_arm, hw.right_arm = _connect_hcx_arms(teleop_yaml)
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
        print(
            "[INFO] 采图: head 新帧驱动，left_hand/right_hand RGB-D at_or_before",
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
            _confirm_targets_by_feedback(
                hw,
                left_start,
                _read_hcx_joints(hw.right_arm, hw.right_start_joints_deg).astype(float).tolist(),
                timeout_s=deploy["move_feedback_confirm_timeout_s"],
                poll_interval_s=deploy["move_feedback_confirm_poll_interval_s"],
                angle_tolerance_deg=deploy["move_angle_tolerance_deg"],
            )
        if deploy["right_start_joints_deg"] is not None:
            right_start = _clamp_joints_by_limits(
                deploy["right_start_joints_deg"],
                deploy["joint_limits_min_deg"],
                deploy["joint_limits_max_deg"],
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
                _read_hcx_joints(hw.left_arm, hw.left_start_joints_deg).astype(float).tolist(),
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

        image_size = cfg.dataset.image_size
        obs_queue = DualStepObservationQueue(
            hw,
            list(cameras),
            n_obs_steps=n_obs,
            state_dim=int(cfg.state_dim),
            image_size=image_size,
            depth_cameras=depth_cameras,
            depth_min_mm=depth_min_mm,
            depth_max_mm=depth_max_mm,
            expected_scale_mm=expected_scale_mm,
        )
        obs_queue.fill_initial()
        print(
            f"[INFO] 到位观测队列已就绪 n_obs={n_obs} image_size={image_size}",
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
                depth_cameras=depth_cameras,
                predict_joint_delta=predict_joint_delta,
                joint_mask=joint_mask,
            )
            _log_inference_state_input(
                step_i=step_i,
                obs_history=obs_history,
                n_obs_steps=n_obs,
                batch=batch,
            )
            with torch.no_grad():
                pred = policy.sample_actions(batch)[0, :n_action_steps].cpu()
            pred_phys = np.asarray(
                denormalize_predicted_action(
                    pred,
                    stats,
                    norm_mode,
                    q_now_phys=np.asarray(obs_history[-1].state, dtype=np.float32),
                    predict_joint_delta=predict_joint_delta,
                    joint_mask=joint_mask,
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
                action = np.asarray(pred_phys[k], dtype=np.float32)
                print(
                    f"[INFO] step={step_i} 执行 chunk[{k}/{n_exec}]",
                    end="",
                )
                _send_action(
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
                    feedback_confirm=deploy["move_feedback_confirm"],
                    move_interrupt=deploy["move_interrupt"],
                )
                obs_queue.push_after_action()
                step_i += 1
                _pump_camera_preview(hw)
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        _shutdown(hw)


if __name__ == "__main__":
    main()
