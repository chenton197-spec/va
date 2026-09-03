#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

VA_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DEPLOY_YAML = SCRIPT_DIR / "quanfu_servo_rtc.yaml"
TELEOP_ROOT = VA_ROOT.parent / "teleop_project"
if not TELEOP_ROOT.is_dir():
    raise FileNotFoundError(f"找不到 teleop_project: {TELEOP_ROOT}")

for p in (TELEOP_ROOT, VA_ROOT, SCRIPT_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from robotfm.config import _normalize_rtc_config
from robotfm.policies.rtc import ActionQueue, RTCConfig, RTCProcessor
from run import (
    BackgroundGripperLoop,
    HardwareBundle,
    _clamp_joints_by_limits,
    _confirm_targets_by_feedback,
    _connect_dual_grippers,
    _load_deploy_config,
    _ramp_start_grippers,
    _read_gripper,
    _read_hcx_joints,
    _shutdown,
)
from run_dual_arm_depth import (
    HeadTriggeredCapture,
    _as_uint16_depth,
    _as_uint8_rgb,
    _connect_record_cameras,
    _describe_rgbd_issue,
)
from servo_rtc_deploy import (
    ARM_SERVO_HZ,
    LatencyTracker,
    ServoInterpolator,
    ServoSendThread,
    ServoWatchdogThread,
    _connect_hcx_direct,
    _load_extra_deploy,
    _resolve_deploy_path,
    _rtc_delay,
    _submit_interp_chunk,
    _sync_action_queue_index,
)
from teleop_sdk.adapters.hcx import HcxFollower

REQUIRED_CAMERAS = ("head", "left_hand", "right_hand")
EXPECTED_DEPTH_CAMERAS = ("left_hand", "right_hand")
SOLE_ACTION_LAYOUT = "left7_grip_right7_grip"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="quanfu v3.1 双臂 RTC 直伺服")
    parser.add_argument("--deploy", type=str, default=str(DEFAULT_DEPLOY_YAML))
    return parser.parse_args()


def _hw_state_to_sole(state: np.ndarray) -> np.ndarray:
    x = np.asarray(state, dtype=np.float32).reshape(-1)
    if x.shape != (16,):
        raise ValueError(f"硬件 state 必须是 16 维，实际为 {x.shape}")
    return np.concatenate([x[:7], x[14:15], x[7:14], x[15:16]])


def _sole_action_to_hw(action: np.ndarray) -> np.ndarray:
    x = np.asarray(action, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[-1] != 16:
        raise ValueError(f"sole action 最后一维必须是 16，实际为 {x.shape}")
    return np.concatenate([x[:, :7], x[:, 8:15], x[:, 7:8], x[:, 15:16]], axis=1).astype(
        np.float32
    )


def _load_quanfu_extra(deploy_path: Path) -> dict[str, Any]:
    extra = _load_extra_deploy(deploy_path)
    with deploy_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    high_dim = raw.get("high_dim_root")
    if high_dim in (None, ""):
        raise ValueError("deploy.yaml 需要 high_dim_root")
    high_dim_path = Path(high_dim)
    if not high_dim_path.is_absolute():
        high_dim_path = (VA_ROOT / high_dim_path).resolve()
    if not high_dim_path.is_dir():
        raise FileNotFoundError(f"找不到 high_dim_root: {high_dim_path}")
    ode_steps = int(raw.get("ode_steps", 10))
    if ode_steps <= 0:
        raise ValueError("ode_steps 必须 > 0")
    fps = float(raw.get("fps", 15))
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps 必须 > 0")
    native = raw.get("camera_native_size", [480, 640])
    camera_native_size = tuple(int(v) for v in native)
    if len(camera_native_size) != 2:
        raise ValueError("camera_native_size 必须是 [H, W]")
    norm_raw = raw.get("norm_params")
    extra.update(
        {
            "high_dim_root": high_dim_path,
            "ode_steps": ode_steps,
            "fps": fps,
            "camera_native_size": camera_native_size,
            "norm_params": None if norm_raw in (None, "") else Path(norm_raw),
        }
    )
    return extra


def _rtc_from_deploy(deploy_rtc: dict[str, Any] | None) -> RTCConfig:
    raw = dict(deploy_rtc or {})
    raw.pop("execution_horizon", None)
    cfg = RTCConfig(
        enabled=bool(raw.get("enabled", True)),
        guidance_enabled=bool(raw.get("guidance_enabled", True)),
        inference_delay=int(raw.get("inference_delay", 0)),
    )
    if "prefix_attention_schedule" in raw and raw["prefix_attention_schedule"] is not None:
        cfg.prefix_attention_schedule = raw["prefix_attention_schedule"]
    if "max_guidance_weight" in raw and raw["max_guidance_weight"] is not None:
        cfg.max_guidance_weight = float(raw["max_guidance_weight"])
    return _normalize_rtc_config(cfg)


class LiveScaleHeadCapture(HeadTriggeredCapture):
    def __init__(self, camera_manager: Any, depth_cameras: list[str] | tuple[str, ...]) -> None:
        super().__init__(camera_manager, depth_cameras, expected_scale_mm={})
        self.last_scale_mm: dict[str, float] = {}

    def capture(
        self, *, timeout_s: float = 2.0
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
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
        scales: dict[str, float] = {}
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
            scales[name] = scale_mm
        self.last_scale_mm = scales
        return images, depths, scales


def _ensure_high_dim_path(high_dim_root: Path) -> None:
    s = str(high_dim_root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _rgb_native_to_model(
    rgb: np.ndarray, image_size: tuple[int, int], native_hw: tuple[int, int], name: str
) -> np.ndarray:
    from high_dim_pushT.dataset import _rgb_to_target_size

    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise RuntimeError(f"相机 {name} RGB 形状异常: {arr.shape}")
    native_h, native_w = native_hw
    if arr.shape[0] != native_h or arr.shape[1] != native_w:
        print(
            f"[WARN] 相机 {name} RGB 尺寸 {arr.shape[0]}x{arr.shape[1]} "
            f"与 camera_native_size={native_h}x{native_w} 不一致，仍尝试缩放",
            flush=True,
        )
    out = np.asarray(_rgb_to_target_size(Image.fromarray(arr), image_size), dtype=np.uint8)
    h, w = image_size
    if out.shape[:2] != (h, w):
        raise RuntimeError(f"相机 {name} 缩放后尺寸异常: {out.shape}，期望 {h}x{w}")
    return np.ascontiguousarray(out)


def _depth_native_to_model_chw(
    raw: np.ndarray,
    scale_mm: float,
    image_size: tuple[int, int],
    depth_min_mm: float,
    depth_max_mm: float,
) -> np.ndarray:
    from high_dim_pushT.dataset import (
        _depth_raw_to_target_size,
        _raw_depth_to_normalized,
        depth_to_rgb3,
    )

    raw_u16 = np.asarray(raw)
    if raw_u16.dtype != np.uint16 or raw_u16.ndim != 2:
        raise ValueError(f"深度必须是 HxW uint16，实际 {raw_u16.shape} {raw_u16.dtype}")
    resized = _depth_raw_to_target_size(raw_u16, image_size)
    norm = _raw_depth_to_normalized(
        resized,
        float(scale_mm),
        depth_min_mm,
        depth_max_mm,
        invalid_raw_value=0,
    )
    return depth_to_rgb3(norm).astype(np.float32)


def _normalize_action(normalizer: Any, action: np.ndarray) -> np.ndarray:
    from high_dim_pushT.dataset import _minmax_normalize, _std2_normalize

    x = np.asarray(action, dtype=np.float32)
    if normalizer.norm_type == "std2":
        return _std2_normalize(x, normalizer.action_mean, normalizer.action_std)
    return _minmax_normalize(x, normalizer.action_mean, normalizer.action_std)


def _reanchor_leftover(
    leftover_abs: torch.Tensor,
    q_now: np.ndarray,
    normalizer: Any,
    device: torch.device,
) -> torch.Tensor:
    abs_np = leftover_abs.detach().cpu().numpy().astype(np.float32)
    delta = abs_np - np.asarray(q_now, dtype=np.float32)
    normed = _normalize_action(normalizer, delta)
    return torch.from_numpy(np.asarray(normed, dtype=np.float32)).to(device)


def _sample_v31_rtc(
    model: Any,
    normalizer: Any,
    packed: dict[str, Any],
    *,
    leftover: torch.Tensor | None,
    infer_delay: int,
    exec_h: int,
    ode_steps: int,
    device: torch.device,
    rtc_processor: RTCProcessor,
) -> tuple[torch.Tensor, np.ndarray]:
    from high_dim_pushT.utils import prepare_obs_batch

    cameras = getattr(model, "cameras", REQUIRED_CAMERAS)
    depth_cameras = getattr(model, "depth_cameras", EXPECTED_DEPTH_CAMERAS)
    obs = prepare_obs_batch(
        packed, cameras=cameras, depth_cameras=depth_cameras, device=str(device)
    )
    b = next(iter(obs.values())).shape[0]
    act_hor = int(model.action_horizon)
    act_dim = int(model.act_dim)
    x = torch.randn((b, act_hor, act_dim), device=device)
    dt = 1.0 / float(ode_steps)
    obs_emb = model.encode_obs(obs)
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    with torch.autocast(device_type="cuda", enabled=use_cuda):
        for i in range(ode_steps):
            t_val = i / ode_steps
            t = torch.full((b,), t_val * 1000.0, device=device, dtype=torch.float32)

            def denoise_step_partial(
                input_x_t: torch.Tensor,
                current_t: torch.Tensor = t,
            ) -> torch.Tensor:
                return model(obs, input_x_t, current_t, obs_emb=obs_emb)

            v = rtc_processor.denoise_step(
                x_t=x,
                prev_chunk_left_over=leftover,
                inference_delay=infer_delay,
                time=t_val,
                original_denoise_step_partial=denoise_step_partial,
                execution_horizon=exec_h,
            )
            x = x + v.float() * dt
    action_norm = x.float()
    proprio_t = None
    if str(getattr(normalizer, "action_mode", "absolute")).strip().lower() == "delta":
        proprio_t = normalizer.unnormalize_proprio(obs["proprio"][:, -1, :])
    abs_action = normalizer.unnormalize_action(action_norm, proprio_t=proprio_t)
    if abs_action.dim() == 3:
        abs_np = abs_action[0].detach().cpu().numpy().astype(np.float32)
        pred_norm = action_norm[0].detach().cpu()
    else:
        abs_np = abs_action.detach().cpu().numpy().astype(np.float32)
        pred_norm = action_norm.detach().cpu()
    return pred_norm, abs_np


def _capture_infer_obs(
    hw: HardwareBundle,
    cameras: list[str],
    joint_arr: Any,
    *,
    normalizer: Any,
    image_size: tuple[int, int],
    native_hw: tuple[int, int],
    depth_cameras: list[str],
    depth_min_mm: float,
    depth_max_mm: float,
    n_obs: int,
) -> tuple[dict[str, Any], np.ndarray]:
    from high_dim_pushT.run_policy import build_obs

    capture = hw.camera_capture
    if capture is None:
        raise RuntimeError("相机采集器未初始化")
    images, depths, scales = capture.capture()
    missing = [n for n in cameras if n not in images]
    if missing:
        raise RuntimeError(f"采图缺少相机: {missing}")
    with joint_arr.get_lock():
        arr = np.frombuffer(joint_arr.get_obj(), dtype=np.float64).copy()
    state_hw = arr.astype(np.float32)
    if state_hw.shape[0] != 16:
        raise RuntimeError(f"joint_arr 必须是 16 维，实际 {state_hw.shape}")
    state_sole = _hw_state_to_sole(state_hw)
    n_obs_i = int(n_obs)
    camera_images = {}
    for cam in cameras:
        frame = _rgb_native_to_model(images[cam], image_size, native_hw, cam)
        camera_images[cam] = [frame] * n_obs_i
    depth_images = {}
    for cam in depth_cameras:
        chw = _depth_native_to_model_chw(
            depths[cam],
            float(scales[cam]),
            image_size,
            depth_min_mm,
            depth_max_mm,
        )
        depth_images[f"{cam}_depth"] = np.stack([chw] * n_obs_i, axis=0)
    proprio = np.stack([state_sole] * n_obs_i, axis=0)
    packed = build_obs(camera_images, proprio, normalizer, depth_images=depth_images)
    return packed, state_sole


def _infer_worker(
    spec: dict[str, Any],
    joint_arr: Any,
    popped_val: Any,
    ack_val: Any,
    out_q: Any,
    ready_evt: Any,
    stop_evt: Any,
) -> None:
    try:
        torch.set_num_threads(1)
        high_dim_root = Path(spec["high_dim_root"])
        _ensure_high_dim_path(high_dim_root)
        from high_dim_pushT.run_policy import load_policy

        ckpt_path = Path(spec["ckpt"])
        teleop_yaml = Path(spec["teleop_yaml"])
        norm_path = spec["norm_params"]
        ode_steps = int(spec["ode_steps"])
        chunk_n = int(spec["chunk"])
        max_steps = int(spec["max_steps"])
        source_hz = int(spec["source_hz"])
        train_fps = float(spec["fps"])
        native_hw = tuple(int(v) for v in spec["camera_native_size"])
        rtc_cfg = _rtc_from_deploy(spec.get("deploy_rtc"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, normalizer = load_policy(
            ckpt_path,
            norm_params_path=norm_path,
            ode_steps=ode_steps,
            device=str(device),
            pretrained_encoder=False,
        )
        cameras = list(normalizer.cameras)
        depth_cameras = list(normalizer.depth_cameras)
        missing = [n for n in REQUIRED_CAMERAS if n not in cameras]
        extra_cam = [n for n in cameras if n not in REQUIRED_CAMERAS]
        if missing or extra_cam:
            raise ValueError(
                f"cameras 必须是 {list(REQUIRED_CAMERAS)} 的排列，"
                f"实际={cameras} missing={missing} extra={extra_cam}"
            )
        if tuple(depth_cameras) != EXPECTED_DEPTH_CAMERAS:
            raise ValueError(
                f"depth_cameras 必须是 {list(EXPECTED_DEPTH_CAMERAS)}，"
                f"实际为 {depth_cameras}"
            )
        if int(normalizer.act_dim) != 16 or int(normalizer.proprio_dim) != 16:
            raise ValueError(
                f"双臂模型要求 act/proprio=16，实际 act={normalizer.act_dim} "
                f"proprio={normalizer.proprio_dim}"
            )
        layout = getattr(normalizer, "action_layout", SOLE_ACTION_LAYOUT)
        if layout not in (None, SOLE_ACTION_LAYOUT):
            raise ValueError(f"仅支持 {SOLE_ACTION_LAYOUT}，实际 layout={layout}")
        n_obs = int(normalizer.obs_horizon)
        action_horizon = int(normalizer.action_horizon)
        if chunk_n >= action_horizon:
            raise ValueError(
                f"chunk 必须小于模型输出长度 (chunk={chunk_n}, horizon={action_horizon})"
            )
        image_size = tuple(int(v) for v in normalizer.image_size)
        action_mode = str(getattr(normalizer, "action_mode", "absolute")).strip().lower()
        rtc_processor = RTCProcessor(rtc_cfg)
        print(
            f"[INFO] quanfu action_mode={action_mode} "
            f"native={native_hw} model={image_size} "
            f"depth={depth_cameras} ode_steps={ode_steps} "
            f"n_obs={n_obs} horizon={action_horizon} chunk={chunk_n} "
            f"rtc.guidance={rtc_cfg.guidance_enabled} device={device}",
            flush=True,
        )
        hw = HardwareBundle(
            left_start_joints_deg=spec["left_start"],
            right_start_joints_deg=spec["right_start"],
        )
        hw.camera_manager = _connect_record_cameras(teleop_yaml)
        hw.camera_capture = LiveScaleHeadCapture(hw.camera_manager, depth_cameras)
        action_queue = ActionQueue(rtc_cfg)
        latency_tracker = LatencyTracker()
        obs_kw = dict(
            normalizer=normalizer,
            image_size=image_size,
            native_hw=native_hw,
            depth_cameras=depth_cameras,
            depth_min_mm=float(normalizer.depth_min_mm),
            depth_max_mm=float(normalizer.depth_max_mm),
            n_obs=n_obs,
        )
        packed, _ = _capture_infer_obs(hw, cameras, joint_arr, **obs_kw)
        with torch.no_grad():
            _sample_v31_rtc(
                model,
                normalizer,
                packed,
                leftover=None,
                infer_delay=0,
                exec_h=2,
                ode_steps=ode_steps,
                device=device,
                rtc_processor=rtc_processor,
            )
        infer_i = 0
        while infer_i < max_steps and not stop_evt.is_set():
            infer_delay = _rtc_delay(latency_tracker, train_fps, chunk_n, infer_i)
            exec_h = infer_delay + 2
            if infer_i > 0:
                while not stop_evt.is_set() and int(ack_val.value) != infer_i - 1:
                    time.sleep(0.002)
                threshold = infer_delay + 2
                while not stop_evt.is_set():
                    _sync_action_queue_index(
                        action_queue, int(popped_val.value), train_fps, source_hz
                    )
                    if action_queue.qsize() <= threshold:
                        break
                    time.sleep(0.002)
                if stop_evt.is_set():
                    break
            idx_before = action_queue.get_action_index()
            leftover = action_queue.get_left_over()
            if leftover is not None and leftover.shape[0] == 0:
                leftover = None
            leftover_len = 0 if leftover is None else int(leftover.shape[0])
            packed, q_now = _capture_infer_obs(hw, cameras, joint_arr, **obs_kw)
            if leftover is not None:
                leftover = leftover.to(device)
                if action_mode == "delta":
                    abs_left = action_queue.get_processed_left_over()
                    if abs_left is not None and abs_left.shape[0] > 0:
                        leftover = _reanchor_leftover(
                            abs_left, q_now, normalizer, device
                        )
            t_infer = time.perf_counter()
            with torch.no_grad():
                pred_norm, processed = _sample_v31_rtc(
                    model,
                    normalizer,
                    packed,
                    leftover=leftover,
                    infer_delay=infer_delay,
                    exec_h=exec_h,
                    ode_steps=ode_steps,
                    device=device,
                    rtc_processor=rtc_processor,
                )
            infer_s = time.perf_counter() - t_infer
            latency_tracker.add(infer_s)
            pred_norm = pred_norm[:chunk_n]
            processed = np.asarray(processed, dtype=np.float32)[:chunk_n]
            popped = 0 if infer_i == 0 else int(popped_val.value)
            idx_after = _sync_action_queue_index(
                action_queue, popped, train_fps, source_hz
            )
            new_delay = 0
            if infer_i > 0:
                new_delay = max(0, idx_after - idx_before)
                new_delay = min(new_delay, leftover_len, chunk_n - 3)
            processed_t = torch.as_tensor(processed, dtype=torch.float32)
            action_queue.merge(pred_norm, processed_t, new_delay, idx_before)
            phys = np.asarray(processed, dtype=np.float64)
            if infer_i == 0:
                phys[0, :7] = q_now[:7].astype(np.float64)
                phys[0, 8:15] = q_now[8:15].astype(np.float64)
                skip = 0
            else:
                skip = min(new_delay, int(phys.shape[0]) - 1)
            remain = phys[skip:]
            out_chunk = np.concatenate([remain[:, :7], remain[:, 8:15]], axis=1)
            g_row = remain[0]
            out_q.put(
                {
                    "infer_i": infer_i,
                    "chunk": out_chunk,
                    "grip_l": float(np.clip(g_row[7], 0.0, 1.0)),
                    "grip_r": float(np.clip(g_row[15], 0.0, 1.0)),
                    "leftover": leftover_len,
                    "delay": int(new_delay),
                    "qsize": int(action_queue.qsize()),
                    "infer_ms": float(infer_s * 1e3),
                }
            )
            if infer_i == 0:
                ready_evt.set()
            infer_i += 1
        _shutdown(hw)
    except BaseException as exc:
        try:
            out_q.put({"error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
        ready_evt.set()


def main() -> None:
    args = _parse_args()
    deploy_path = _resolve_deploy_path(args.deploy)
    deploy = _load_deploy_config(deploy_path)
    extra = _load_quanfu_extra(deploy_path)
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
    rtc_cfg = _rtc_from_deploy(deploy.get("rtc"))
    if not bool(rtc_cfg.enabled):
        raise ValueError("本脚本要求 rtc.enabled=true")

    train_fps = float(extra["fps"])
    max_steps = int(deploy["max_steps"])
    chunk_n = int(extra["chunk"])
    print(
        f"[INFO] quanfu RTC 部署 deploy={deploy_path} ckpt={ckpt_path} "
        f"fps={train_fps:g} chunk={chunk_n} ode_steps={extra['ode_steps']} "
        f"rtc.guidance={rtc_cfg.guidance_enabled}",
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
        interp = ServoInterpolator(source_fps=train_fps, n_joints=14)
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
        norm_path = extra["norm_params"]
        if norm_path is None:
            norm_s = None
        else:
            norm_s = str(norm_path)
        spec = {
            "ckpt": str(ckpt_path),
            "norm_params": norm_s,
            "high_dim_root": str(extra["high_dim_root"]),
            "teleop_yaml": str(teleop_yaml),
            "deploy_rtc": dict(deploy.get("rtc") or {}),
            "chunk": int(chunk_n),
            "max_steps": int(max_steps),
            "source_hz": int(source_hz),
            "fps": float(train_fps),
            "ode_steps": int(extra["ode_steps"]),
            "camera_native_size": list(extra["camera_native_size"]),
            "left_start": deploy["left_start_joints_deg"],
            "right_start": deploy["right_start_joints_deg"],
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
            f"[INFO] {source_hz}Hz 源点下发已启动 → {ARM_SERVO_HZ}Hz",
            flush=True,
        )
        while True:
            send_thread.check_fault()
            watchdog.check_fault()
            joints = send_thread.latest_joints()
            with joint_arr.get_lock():
                if joints is not None:
                    for i, v in enumerate(
                        np.asarray(joints, dtype=np.float64).reshape(-1)[:14]
                    ):
                        joint_arr[i] = float(v)
                joint_arr[14] = float(
                    _read_gripper(hw.left_gripper, fallback=float(joint_arr[14]))
                )
                joint_arr[15] = float(
                    _read_gripper(hw.right_gripper, fallback=float(joint_arr[15]))
                )
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
    except KeyboardInterrupt:
        print("\n[INFO] 收到 Ctrl+C，停止伺服", flush=True)
    finally:
        if stop_evt is not None:
            stop_evt.set()
        if infer_proc is not None:
            infer_proc.join(timeout=5.0)
            if infer_proc.is_alive():
                infer_proc.terminate()
        if send_thread is not None:
            send_thread.stop()
        if watchdog is not None:
            watchdog.stop()
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
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
