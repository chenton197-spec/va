#!/usr/bin/env python3
"""W2 实机闭环运行入口（HCX 双臂，CLI 仅支持 --deploy）。"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

VA_ROOT = Path(__file__).resolve().parents[1]
TELEOP_ROOT = Path("/home/a/Code/teleop_project")
SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_YAML = SCRIPT_DIR / "deploy.yaml"

if str(TELEOP_ROOT) not in sys.path:
    sys.path.insert(0, str(TELEOP_ROOT))
if str(VA_ROOT) not in sys.path:
    sys.path.insert(0, str(VA_ROOT))

from hcx_sdk import RobotClient  # noqa: E402
from robotfm.config import _normalize_rtc_config, load_config  # noqa: E402
from robotfm.data.action_delta import denormalize_predicted_action, flow_history_from_phys, joint_mask_from_names  # noqa: E402
from robotfm.data.dataset import spatial_preprocess_images  # noqa: E402
from robotfm.data.stats import normalize  # noqa: E402
from robotfm.policies.rtc import ActionQueue, RTCConfig  # noqa: E402
from robotfm.train import build_policy  # noqa: E402
from robotfm.types import Observation  # noqa: E402
from teleop_sdk.adapters.gloria_m import GloriaMGripperFollower  # noqa: E402
from teleop_sdk.config import load_runtime_config  # noqa: E402


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def _parse_optional_gripper(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    opening = float(value)
    if not math.isfinite(opening) or opening < 0.0 or opening > 1.0:
        raise ValueError(f"{field} 必须是 [0, 1] 内的有限数")
    return opening


JOINT_LIMITS_MIN_DEG = (-169.0, -100.0, -169.0, -139.0, -169.0, -54.0, -59.0)
JOINT_LIMITS_MAX_DEG = (169.0, 100.0, 169.0, 54.0, 169.0, 54.0, 59.0)
DUAL_ARM_CAMERAS = ("head", "left_hand", "right_hand")
LEFT_ARM_CAMERAS = ("head", "left_hand")
DUAL_ARM_DIM = 16
LEFT_ARM_DIM = 8


def _runtime_layout(
    cameras: list[str], state_dim: int, action_dim: int
) -> str:
    """返回 ``dual``（16-D 双臂）或 ``left``（8-D 左臂）。"""
    cams = tuple(cameras)
    if (
        cams == DUAL_ARM_CAMERAS
        and int(state_dim) == DUAL_ARM_DIM
        and int(action_dim) == DUAL_ARM_DIM
    ):
        return "dual"
    if (
        cams == LEFT_ARM_CAMERAS
        and int(state_dim) == LEFT_ARM_DIM
        and int(action_dim) == LEFT_ARM_DIM
    ):
        return "left"
    raise ValueError(
        "当前 HCX 脚本支持双臂 cameras="
        f"{list(DUAL_ARM_CAMERAS)} state/action_dim={DUAL_ARM_DIM}，"
        f"或左臂 cameras={list(LEFT_ARM_CAMERAS)} state/action_dim={LEFT_ARM_DIM}；"
        f"实际 cameras={list(cameras)} state_dim={state_dim} action_dim={action_dim}"
    )


def _parse_joint_limits_deg(
    raw: Any, source: Path
) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(JOINT_LIMITS_MIN_DEG, dtype=np.float64)
    hi = np.asarray(JOINT_LIMITS_MAX_DEG, dtype=np.float64)
    if raw is None:
        return lo, hi
    if not isinstance(raw, dict):
        raise ValueError(f"joint_limits_deg 必须是映射: {source}")
    if raw.get("min") is not None:
        lo = np.asarray([float(v) for v in raw["min"]], dtype=np.float64)
    if raw.get("max") is not None:
        hi = np.asarray([float(v) for v in raw["max"]], dtype=np.float64)
    if lo.shape != (7,) or hi.shape != (7,):
        raise ValueError("joint_limits_deg.min / max 必须是 7 维")
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("joint_limits_deg 必须是有限数")
    if np.any(lo >= hi):
        raise ValueError("joint_limits_deg 每轴必须 min < max")
    return lo, hi


def _parse_start_pose(
    start: Any, source: Path
) -> tuple[
    list[float] | None,
    list[float] | None,
    float | None,
    float | None,
    float,
]:
    if start is None:
        return None, None, None, None, 1.0
    if not isinstance(start, dict):
        raise ValueError(f"start_pose 必须是映射: {source}")
    left = start.get("left_joints_deg")
    right = start.get("right_joints_deg")
    left_j = [float(v) for v in left] if left is not None else None
    right_j = [float(v) for v in right] if right is not None else None
    if left_j is not None and len(left_j) != 7:
        raise ValueError("start_pose.left_joints_deg 必须是 7 维")
    if right_j is not None and len(right_j) != 7:
        raise ValueError("start_pose.right_joints_deg 必须是 7 维")
    left_g = _parse_optional_gripper(
        start.get("left_gripper"), field="start_pose.left_gripper"
    )
    right_g = _parse_optional_gripper(
        start.get("right_gripper"), field="start_pose.right_gripper"
    )
    ramp_raw = start.get("gripper_ramp_s", 1.0)
    ramp_s = 1.0 if ramp_raw in (None, "") else float(ramp_raw)
    if not math.isfinite(ramp_s) or ramp_s <= 0.0:
        raise ValueError("start_pose.gripper_ramp_s 必须是正的有限秒数")
    return left_j, right_j, left_g, right_g, ramp_s


def _load_deploy_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到部署配置: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"部署配置根节点必须是映射: {path}")

    required = ("checkpoint", "teleop_yaml", "max_steps")
    missing = [k for k in required if k not in data or data[k] in (None, "")]
    if missing:
        raise ValueError(f"deploy.yaml 缺少字段: {missing}")

    left_start, right_start, left_g, right_g, gripper_ramp_s = _parse_start_pose(
        data.get("start_pose"), path
    )
    joint_lo, joint_hi = _parse_joint_limits_deg(data.get("joint_limits_deg"), path)
    if left_start is not None:
        left_start = _clamp_joints_by_limits(
            left_start, joint_lo, joint_hi, side="start_pose.left"
        )
    if right_start is not None:
        right_start = _clamp_joints_by_limits(
            right_start, joint_lo, joint_hi, side="start_pose.right"
        )
    move = data.get("move_joints", {}) or {}
    if not isinstance(move, dict):
        raise ValueError("deploy.yaml 的 move_joints 必须是映射")

    train_cfg = data.get("config")
    train_cfg_path = (
        _resolve_path(train_cfg, base=VA_ROOT) if train_cfg else None
    )

    exec_raw = data.get("exec_action_steps", None)
    if exec_raw in (None, ""):
        exec_action_steps = None
    else:
        exec_action_steps = int(exec_raw)
        if exec_action_steps <= 0:
            raise ValueError("deploy.yaml 的 exec_action_steps 必须 > 0")

    obs_mode_raw = data.get("obs_mode", "fps")
    if obs_mode_raw in (None, ""):
        obs_mode = "fps"
    else:
        obs_mode = str(obs_mode_raw).strip().lower()
    if obs_mode not in {"fps", "after_action"}:
        raise ValueError(
            "deploy.yaml 的 obs_mode 必须是 fps 或 after_action，"
            f"实际为 {obs_mode_raw!r}"
        )

    obs_fps_raw = data.get("obs_fps", None)
    if obs_fps_raw in (None, ""):
        obs_fps = None
    else:
        obs_fps = int(obs_fps_raw)
        if obs_fps <= 0:
            raise ValueError("deploy.yaml 的 obs_fps 必须 > 0")
    if obs_mode == "after_action":
        obs_fps = None

    rtc_raw = data.get("rtc")
    rtc_override: dict[str, Any] = {}
    if rtc_raw is not None:
        if not isinstance(rtc_raw, dict):
            raise ValueError(f"deploy.yaml 的 rtc 必须是映射: {path}")
        if "enabled" in rtc_raw and rtc_raw["enabled"] is not None:
            rtc_override["enabled"] = bool(rtc_raw["enabled"])
        if "guidance_enabled" in rtc_raw and rtc_raw["guidance_enabled"] is not None:
            rtc_override["guidance_enabled"] = bool(rtc_raw["guidance_enabled"])
        if "inference_delay" in rtc_raw and rtc_raw["inference_delay"] is not None:
            rtc_override["inference_delay"] = int(rtc_raw["inference_delay"])
        if "execution_horizon" in rtc_raw and rtc_raw["execution_horizon"] is not None:
            rtc_override["execution_horizon"] = int(rtc_raw["execution_horizon"])

    return {
        "checkpoint": _resolve_path(data["checkpoint"], base=VA_ROOT),
        "config": train_cfg_path,
        "teleop_yaml": _resolve_path(data["teleop_yaml"], base=SCRIPT_DIR),
        "max_steps": int(data["max_steps"]),
        "obs_mode": obs_mode,
        "obs_fps": obs_fps,
        "exec_action_steps": exec_action_steps,
        "display_cameras": bool(data.get("display_cameras", False)),
        "rtc": rtc_override,
        "left_start_joints_deg": left_start,
        "right_start_joints_deg": right_start,
        "left_start_gripper": left_g,
        "right_start_gripper": right_g,
        "start_gripper_ramp_s": gripper_ramp_s,
        "joint_limits_min_deg": joint_lo,
        "joint_limits_max_deg": joint_hi,
        "move_speed_ratio": (
            None if move.get("speed_ratio", 0.1) is None else float(move.get("speed_ratio", 0.1))
        ),
        "move_acceleration_seconds": (
            None
            if move.get("acceleration_seconds", 0.5) is None
            else float(move.get("acceleration_seconds", 0.5))
        ),
        "move_deceleration_seconds": (
            None
            if move.get("deceleration_seconds", 0.5) is None
            else float(move.get("deceleration_seconds", 0.5))
        ),
        "move_feedback_confirm_timeout_s": float(
            move.get("feedback_confirm_timeout_s", 30.0)
        ),
        "move_feedback_confirm_poll_interval_s": float(
            move.get("feedback_confirm_poll_interval_s", 0.05)
        ),
        "move_angle_tolerance_deg": float(move.get("angle_tolerance_deg", 0.01)),
        "move_max_delta_deg": float(move.get("max_delta_deg", 3.0)),
        "move_feedback_confirm": bool(move.get("feedback_confirm", True)),
        # interrupt=false 时厂商 moveJoints2 将新目标排队；连续高频下发必须
        # 用 interrupt=true 中断当前插补、始终追赶最新目标（旧 handle 被取消）
        "move_interrupt": bool(move.get("interrupt", False)),
    }


def _apply_rtc_overrides(cfg: Any, deploy_rtc: dict[str, Any]) -> None:
    """按 deploy.yaml rtc > 训练 YAML 优先级覆盖 policy.rtc。"""
    rtc = _normalize_rtc_config(cfg.policy.rtc)
    enabled = rtc.enabled
    guidance_enabled = rtc.guidance_enabled
    inference_delay = rtc.inference_delay
    execution_horizon = rtc.execution_horizon

    if "enabled" in deploy_rtc:
        enabled = bool(deploy_rtc["enabled"])
    if "guidance_enabled" in deploy_rtc:
        guidance_enabled = bool(deploy_rtc["guidance_enabled"])
    if "inference_delay" in deploy_rtc:
        inference_delay = int(deploy_rtc["inference_delay"])
    if "execution_horizon" in deploy_rtc:
        execution_horizon = int(deploy_rtc["execution_horizon"])

    need_rebuild = (
        bool(deploy_rtc)
        or enabled != rtc.enabled
        or guidance_enabled != rtc.guidance_enabled
        or inference_delay != rtc.inference_delay
        or execution_horizon != rtc.execution_horizon
    )
    if need_rebuild:
        cfg.policy.rtc = RTCConfig(
            enabled=enabled,
            guidance_enabled=guidance_enabled,
            prefix_attention_schedule=rtc.prefix_attention_schedule,
            max_guidance_weight=rtc.max_guidance_weight,
            execution_horizon=execution_horizon,
            inference_delay=inference_delay,
            debug=rtc.debug,
            debug_maxlen=rtc.debug_maxlen,
        )
    else:
        cfg.policy.rtc = rtc
    cfg.policy.rtc = _normalize_rtc_config(cfg.policy.rtc)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W2 HCX 双臂实机闭环部署")
    parser.add_argument("--deploy", type=str, default=str(DEPLOY_YAML))
    return parser.parse_args()


def _pace_step(step_start: float, fps: int) -> None:
    if fps <= 0:
        return
    remain = (1.0 / fps) - (time.perf_counter() - step_start)
    if remain > 0:
        time.sleep(remain)


def _preprocess_images(
    images: dict[str, np.ndarray],
    *,
    pre_crop_size: int | None,
    resize_size: int | None,
    crop_size: int | None,
    eval_fixed_crop: bool,
) -> dict[str, np.ndarray]:
    """相机原图 HWC uint8 → 策略分辨率 HWC float32 [0,1]。

    顺序与训练/开环评估一致：中心 pre_crop → resize → 可选中心 crop。
    """
    out: dict[str, np.ndarray] = {}
    for name, rgb in images.items():
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        t = spatial_preprocess_images(
            t,
            pre_crop_size=pre_crop_size,
            resize_size=resize_size,
            crop_size=crop_size if eval_fixed_crop else None,
            random_crop=False,
        )
        out[name] = t.squeeze(0).permute(1, 2, 0).contiguous().numpy()
    return out


def _prepare_observation(
    obs: Observation,
    *,
    pre_crop_size: int | None,
    resize_size: int | None,
    crop_size: int | None,
    eval_fixed_crop: bool,
) -> Observation:
    return Observation(
        images=_preprocess_images(
            obs.images,
            pre_crop_size=pre_crop_size,
            resize_size=resize_size,
            crop_size=crop_size,
            eval_fixed_crop=eval_fixed_crop,
        ),
        state=np.asarray(obs.state, dtype=np.float32),
        timestamp=obs.timestamp,
    )


def _build_obs_batch(
    obs_history: list[Observation],
    cameras: list[str],
    n_obs_steps: int,
    stats: dict,
    norm_mode: str,
    device: torch.device,
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
    return {
        "obs_images": torch.stack(camera_histories, dim=0).unsqueeze(0).to(device),
        "obs_state": torch.stack(states, dim=0).unsqueeze(0).to(device),
        "obs_history": torch.from_numpy(
            flow_history_from_phys(
                np.stack(
                    [np.asarray(obs.state, dtype=np.float32) for obs in history],
                    axis=0,
                ),
                stats,
                norm_mode,
            )
        )
        .unsqueeze(0)
        .to(device),
    }


def _format_phys_state(state: np.ndarray) -> str:
    s = np.asarray(state, dtype=np.float32).reshape(-1)
    if s.size == LEFT_ARM_DIM:
        return (
            f"left={[round(float(v), 3) for v in s[:7]]} "
            f"left_gripper={float(s[7]):.4f}"
        )
    return (
        f"left={[round(float(v), 3) for v in s[:7]]} "
        f"right={[round(float(v), 3) for v in s[7:14]]} "
        f"grip=({float(s[14]):.4f},{float(s[15]):.4f})"
    )


def _log_inference_state_input(
    *,
    step_i: int,
    obs_history: list[Observation],
    n_obs_steps: int,
    batch: dict[str, torch.Tensor],
) -> None:
    history = obs_history[-n_obs_steps:]
    while len(history) < n_obs_steps:
        history.insert(0, history[0])
    norm = batch["obs_state"][0].detach().cpu().numpy()
    phys_parts = [
        f"[{i}] {_format_phys_state(np.asarray(obs.state))}"
        for i, obs in enumerate(history)
    ]
    norm_parts = [
        f"[{i}] {[round(float(v), 4) for v in row]}"
        for i, row in enumerate(norm)
    ]
    print(
        f"[INFO] step={step_i} 推理输入 state phys ({len(history)} 帧): "
        + " | ".join(phys_parts),
        flush=True,
    )
    print(
        f"[INFO] step={step_i} 推理输入 state norm ({len(norm)} 帧): "
        + " | ".join(norm_parts),
        flush=True,
    )


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _log_inference_result(
    *,
    step_i: int,
    pred_norm: np.ndarray | torch.Tensor,
    pred_phys: np.ndarray | torch.Tensor,
) -> None:
    """打印模型原始归一化输出，以及反归一化后的物理动作。"""
    norm = np.asarray(_as_numpy(pred_norm), dtype=np.float32)
    phys = np.asarray(_as_numpy(pred_phys), dtype=np.float32)
    if norm.ndim == 1:
        norm = norm[None, :]
    if phys.ndim == 1:
        phys = phys[None, :]
    norm_parts = [
        f"[{i}] {[round(float(v), 4) for v in row]}"
        for i, row in enumerate(norm)
    ]
    phys_parts = [
        f"[{i}] {_format_phys_state(row)}"
        for i, row in enumerate(phys)
    ]
    print(
        f"[INFO] step={step_i} 推理输出 action norm ({len(norm)} 步): "
        + " | ".join(norm_parts),
        flush=True,
    )
    print(
        f"[INFO] step={step_i} 推理输出 action phys ({len(phys)} 步): "
        + " | ".join(phys_parts),
        flush=True,
    )


@dataclass
class HardwareBundle:
    hcx_client: Any | None = None
    left_arm: Any | None = None
    right_arm: Any | None = None
    left_gripper: Any | None = None
    right_gripper: Any | None = None
    left_gripper_loop: "BackgroundGripperLoop | None" = None
    right_gripper_loop: "BackgroundGripperLoop | None" = None
    camera_manager: Any | None = None
    camera_preview: "CameraPreviewLoop | None" = None
    obs_sampler: "FpsObservationSampler | None" = None
    left_start_joints_deg: list[float] | None = None
    right_start_joints_deg: list[float] | None = None


class BackgroundGripperLoop:
    def __init__(self, gripper: Any, *, rate_hz: float = 125.0):
        if rate_hz <= 0.0:
            raise ValueError("gripper rate_hz 必须 > 0")
        self._gripper = gripper
        self._period_s = 1.0 / float(rate_hz)
        self._lock = threading.Lock()
        self._target = 1.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set_opening(self, opening: float) -> None:
        with self._lock:
            self._target = float(np.clip(opening, 0.0, 1.0))

    def start(self, initial_opening: float | None = None) -> None:
        if self._thread is not None:
            raise RuntimeError("夹爪后台循环已在运行")
        if initial_opening is not None:
            self.set_opening(initial_opening)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, *, join_timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=join_timeout_s)
        self._thread = None

    def _run(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            with self._lock:
                target = self._target
            try:
                _ = bool(self._gripper.send_normalized(target))
            except Exception:
                pass
            next_t += self._period_s
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0.0:
                if self._stop.wait(timeout=sleep_s):
                    break
            else:
                next_t = time.perf_counter()


PREVIEW_WINDOW_NAME = "W2 cameras"
PREVIEW_FPS = 15.0
PREVIEW_TILE_HEIGHT = 360


def _bgr_preview_tile(rgb: np.ndarray | None, name: str, height: int) -> np.ndarray:
    if rgb is None:
        tile = np.zeros((height, int(round(height * 16 / 9)), 3), dtype=np.uint8)
        label = f"{name}: no frame"
    else:
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        src_h, src_w = bgr.shape[:2]
        new_w = max(1, int(round(src_w * height / max(src_h, 1))))
        tile = cv2.resize(bgr, (new_w, height), interpolation=cv2.INTER_AREA)
        label = name
    cv2.putText(
        tile,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return tile


def _compose_camera_preview(
    camera_manager: Any,
    camera_names: list[str],
    *,
    tile_height: int,
) -> np.ndarray:
    tiles: list[np.ndarray] = []
    for name in camera_names:
        rgb: np.ndarray | None = None
        try:
            frame = camera_manager.camera(name).get_frame()
            if frame is not None and frame.rgb is not None:
                rgb = np.asarray(frame.rgb)
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        except Exception:
            rgb = None
        tiles.append(_bgr_preview_tile(rgb, name, tile_height))
    height = max(tile.shape[0] for tile in tiles)
    gap = np.zeros((height, 4, 3), dtype=np.uint8)
    parts: list[np.ndarray] = []
    for i, tile in enumerate(tiles):
        if i:
            parts.append(gap)
        parts.append(tile)
    return np.concatenate(parts, axis=1)


class CameraPreviewLoop:
    """后台按固定帧率拼接相机画面；OpenCV 窗口必须在主线程 ``pump()``。"""

    def __init__(
        self,
        camera_manager: Any,
        camera_names: list[str],
        *,
        fps: float = PREVIEW_FPS,
        tile_height: int = PREVIEW_TILE_HEIGHT,
        window_name: str = PREVIEW_WINDOW_NAME,
    ) -> None:
        if fps <= 0.0:
            raise ValueError("preview fps 必须 > 0")
        self._camera_manager = camera_manager
        self._camera_names = list(camera_names)
        self._period_s = 1.0 / float(fps)
        self._tile_height = int(tile_height)
        self._window_name = window_name
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._window_ready = False

    @property
    def is_active(self) -> bool:
        return self._window_ready

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("相机预览已在运行")
        try:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            self._window_ready = True
        except Exception as exc:
            print(f"[WARN] 无法创建相机预览窗口: {exc}")
            return
        self._thread = threading.Thread(
            target=self._run, name="camera-preview", daemon=True
        )
        self._thread.start()

    def pump(self) -> None:
        if not self._window_ready:
            return
        with self._lock:
            mosaic = None if self._latest is None else self._latest.copy()
        if mosaic is not None:
            cv2.imshow(self._window_name, mosaic)
        cv2.waitKey(1)

    def stop(self, *, join_timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=join_timeout_s)
        self._thread = None
        if self._window_ready:
            try:
                cv2.destroyWindow(self._window_name)
                cv2.waitKey(1)
            except Exception:
                pass
            self._window_ready = False

    def _run(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            try:
                mosaic = _compose_camera_preview(
                    self._camera_manager,
                    self._camera_names,
                    tile_height=self._tile_height,
                )
                with self._lock:
                    self._latest = mosaic
            except Exception as exc:
                print(f"[WARN] 相机预览采图失败: {exc}")
            next_t += self._period_s
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0.0:
                if self._stop.wait(timeout=sleep_s):
                    break
            else:
                next_t = time.perf_counter()


def _pump_camera_preview(hw: HardwareBundle) -> None:
    if hw.camera_preview is not None:
        hw.camera_preview.pump()


class FpsObservationSampler:
    """按 ``obs_fps``（默认训练 fps）后台采集图像+状态，与动作到位脱钩。

    相机本身持续出帧；本线程每 1/fps 秒抓一次最新 RGB 和关节/夹爪，
    只保留最近 ``n_obs_steps`` 帧。推理时 snapshot，不再在控制步后采图。
    """

    def __init__(
        self,
        hw: HardwareBundle,
        cameras: list[str],
        *,
        n_obs_steps: int,
        fps: int,
        state_dim: int,
        pre_crop_size: int | None,
        resize_size: int | None,
        crop_size: int | None,
        eval_fixed_crop: bool,
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
        self._state_dim = int(state_dim)
        self._pre_crop_size = pre_crop_size
        self._resize_size = resize_size
        self._crop_size = crop_size
        self._eval_fixed_crop = bool(eval_fixed_crop)
        self._lock = threading.Lock()
        self._history: deque[Observation] = deque(maxlen=self._n_obs_steps)
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("观测采样已在运行")
        self._thread = threading.Thread(
            target=self._run, name="obs-sampler", daemon=True
        )
        self._thread.start()

    def stop(self, *, join_timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=join_timeout_s)
        self._thread = None

    def snapshot(self) -> list[Observation]:
        with self._lock:
            self._raise_if_locked_unhealthy()
            return list(self._history)

    def raise_if_unhealthy(self) -> None:
        with self._lock:
            self._raise_if_locked_unhealthy()

    def _raise_if_locked_unhealthy(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"{self._fps}fps 观测采样失败") from self._error
        if not self._history:
            raise RuntimeError(f"{self._fps}fps 观测缓冲为空")

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
                raise RuntimeError(f"{self._fps}fps 观测采样失败") from err
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
                obs = _read_observation(
                    self._hw,
                    self._cameras,
                    last_state=last_state,
                    state_dim=self._state_dim,
                )
                obs.validate(self._cameras, self._state_dim)
                obs = _prepare_observation(
                    obs,
                    pre_crop_size=self._pre_crop_size,
                    resize_size=self._resize_size,
                    crop_size=self._crop_size,
                    eval_fixed_crop=self._eval_fixed_crop,
                )
                last_state = obs.state
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


class StepObservationQueue:
    """到位后再采：每执行完一步动作等各相机新帧，维护 ``n_obs_steps`` 滑动队列。

    不使用 ``obs_fps`` 后台线程。到位瞬间不读缓冲里的旧 RGB（运动中已拍到的
    那张），而是记下当前 ``latest_sequence`` 再等下一张。启动时同样等新帧
    并重复填满，供第一次推理。
    """

    def __init__(
        self,
        hw: HardwareBundle,
        cameras: list[str],
        *,
        n_obs_steps: int,
        state_dim: int,
        pre_crop_size: int | None,
        resize_size: int | None,
        crop_size: int | None,
        eval_fixed_crop: bool,
    ) -> None:
        if n_obs_steps <= 0:
            raise ValueError("n_obs_steps 必须 > 0")
        self._hw = hw
        self._cameras = list(cameras)
        self._n_obs_steps = int(n_obs_steps)
        self._state_dim = int(state_dim)
        self._pre_crop_size = pre_crop_size
        self._resize_size = resize_size
        self._crop_size = crop_size
        self._eval_fixed_crop = bool(eval_fixed_crop)
        self._history: deque[Observation] = deque(maxlen=self._n_obs_steps)

    def fill_initial(self) -> None:
        obs = self._capture()
        for _ in range(self._n_obs_steps):
            self._history.append(obs)

    def push_after_action(self) -> None:
        self._history.append(self._capture())

    def snapshot(self) -> list[Observation]:
        if len(self._history) < self._n_obs_steps:
            raise RuntimeError(
                f"到位观测队列未满: {len(self._history)}/{self._n_obs_steps}"
            )
        return list(self._history)

    def _capture(self) -> Observation:
        last_state = None if not self._history else np.asarray(
            self._history[-1].state, dtype=np.float32
        )
        obs = _read_observation(
            self._hw,
            self._cameras,
            last_state=last_state,
            state_dim=self._state_dim,
            wait_new=True,
        )
        obs.validate(self._cameras, self._state_dim)
        return _prepare_observation(
            obs,
            pre_crop_size=self._pre_crop_size,
            resize_size=self._resize_size,
            crop_size=self._crop_size,
            eval_fixed_crop=self._eval_fixed_crop,
        )


def _connect_cameras(teleop_yaml: Path, required: list[str]) -> Any:
    from orbbec_sdk import OrbbecManager, load_orbbec_camera_configs

    # 与 openarm_hcx_dual_arm 采集一致：读 hcx_orbbec（head/left_hand/right_hand），
    # 不要用默认 orbbec 段（hand/head），否则相机名对不上训练配置。
    all_configs = load_orbbec_camera_configs(teleop_yaml, section_name="hcx_orbbec")
    by_name = {c.name: c for c in all_configs}
    missing = [n for n in required if n not in by_name]
    if missing:
        raise ValueError(
            f"teleop.yaml 的 hcx_orbbec 缺少训练所需相机: {missing} "
            f"(已声明: {sorted(by_name)})"
        )
    configs = tuple(by_name[n] for n in required)
    manager = OrbbecManager(configs)
    manager.start()
    return manager


def _connect_hcx_arms(teleop_yaml: Path) -> tuple[Any, Any, Any]:
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
    right_arm = client.arm(h.right_robot_id)
    if h.auto_enable:
        if not left_arm.enabled:
            left_arm.set_enabled(True)
        if not right_arm.enabled:
            right_arm.set_enabled(True)
    if not left_arm.enabled or not right_arm.enabled:
        raise RuntimeError("HCX 左右臂未使能，请现场使能或开启 hcx.auto_enable")
    return client, left_arm, right_arm


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是正的有限数")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是正的有限数") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} 必须是正的有限数")
    return numeric


def _validate_gloria_dual_config(runtime: Any) -> tuple[bool, bool, float]:
    cfg = runtime.gloria_m_dual
    left_enabled = bool(cfg.left.enabled)
    right_enabled = bool(cfg.right.enabled)
    rate_hz = _positive_finite("gloria_m_dual.rate_hz", cfg.rate_hz)
    _ = _positive_finite("gloria_m_dual.status_print_interval_s", cfg.status_print_interval_s)
    if not (left_enabled or right_enabled):
        return False, False, rate_hz

    used_ports = {
        runtime.openarm_mini.port_left: "openarm_mini.port_left",
        runtime.openarm_mini.port_right: "openarm_mini.port_right",
    }
    for side in ("left", "right"):
        side_cfg = cfg.side_config(side)
        if not bool(side_cfg.enabled):
            continue
        port = side_cfg.port
        if not isinstance(port, str) or not port.strip():
            raise ValueError(f"gloria_m_dual.{side}.port 不能为空")
        if port.lower() == "auto":
            raise ValueError(f"gloria_m_dual.{side}.port 不能为 auto")
        if not isinstance(side_cfg.baudrate, int) or isinstance(side_cfg.baudrate, bool) or side_cfg.baudrate <= 0:
            raise ValueError(f"gloria_m_dual.{side}.baudrate 必须为正整数")
        duplicate = used_ports.get(port)
        if duplicate is not None:
            raise ValueError(f"gloria_m_dual.{side}.port 与 {duplicate} 不能使用同一串口")
        used_ports[port] = f"gloria_m_dual.{side}.port"
    return left_enabled, right_enabled, rate_hz


def _connect_dual_grippers(teleop_yaml: Path) -> tuple[Any | None, Any | None, float]:
    runtime = load_runtime_config(teleop_yaml)
    left_enabled, right_enabled, rate_hz = _validate_gloria_dual_config(runtime)
    left_cfg = runtime.gloria_m_dual.side_config("left")
    right_cfg = runtime.gloria_m_dual.side_config("right")
    left = None
    right = None
    if left_enabled:
        left = GloriaMGripperFollower(left_cfg)
        left.connect()
    if right_enabled:
        right = GloriaMGripperFollower(right_cfg)
        right.connect()
    return left, right, rate_hz


def _read_hcx_joints(arm, fallback: list[float] | None) -> np.ndarray:
    if arm is None:
        if fallback is None:
            raise RuntimeError("无机械臂且无 fallback 关节角")
        return np.asarray(fallback, dtype=np.float32)
    values = np.asarray(arm.joint_angles(), dtype=np.float32)
    if values.shape == (7,) and np.isfinite(values).all():
        return values
    if fallback is not None:
        return np.asarray(fallback, dtype=np.float32)
    raise RuntimeError("HCX 关节反馈异常")


def _read_gripper(gripper, fallback: float = 1.0) -> float:
    if gripper is None:
        return float(fallback)
    opening = gripper.read_cached_normalized_opening()
    if opening is None:
        opening = gripper.read_normalized_opening()
    if opening is None or not np.isfinite(opening):
        return float(fallback)
    return float(np.clip(opening, 0.0, 1.0))


def _ramp_start_grippers(
    hw: HardwareBundle,
    *,
    left_target: float | None,
    right_target: float | None,
    duration_s: float,
    rate_hz: float,
) -> None:
    """启动时从当前开口线性插值到 start_pose 夹爪目标。"""
    if left_target is None and right_target is None:
        return
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("start_pose.gripper_ramp_s 必须是正的有限秒数")
    if not math.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("gripper rate_hz 必须 > 0")

    left_from = (
        _read_gripper(hw.left_gripper, fallback=1.0)
        if left_target is not None
        else None
    )
    right_from = (
        _read_gripper(hw.right_gripper, fallback=1.0)
        if right_target is not None
        else None
    )
    n_steps = max(1, int(math.ceil(duration_s * rate_hz)))
    period_s = duration_s / float(n_steps)
    print(
        f"[INFO] 启动夹爪渐变 steps={n_steps} duration_s={duration_s:.3f} "
        f"left={left_from}->{left_target} right={right_from}->{right_target}"
    )
    for step in range(1, n_steps + 1):
        alpha = float(step) / float(n_steps)
        if left_target is not None and left_from is not None:
            opening = float(left_from + alpha * (left_target - left_from))
            if hw.left_gripper_loop is not None:
                hw.left_gripper_loop.set_opening(opening)
            elif hw.left_gripper is not None:
                _ = hw.left_gripper.send_normalized(opening)
        if right_target is not None and right_from is not None:
            opening = float(right_from + alpha * (right_target - right_from))
            if hw.right_gripper_loop is not None:
                hw.right_gripper_loop.set_opening(opening)
            elif hw.right_gripper is not None:
                _ = hw.right_gripper.send_normalized(opening)
        time.sleep(period_s)


def _camera_status_label(camera: Any) -> str:
    status = getattr(camera, "status", None)
    if status is None:
        return "unknown"
    value = getattr(status, "value", status)
    return str(value)


def _describe_camera_frame(camera_manager: Any, name: str) -> str:
    cam = camera_manager.camera(name)
    status = _camera_status_label(cam)
    last_error = getattr(cam, "last_error", None)
    last_ns = getattr(cam, "latest_capture_monotonic_ns", None)
    frame = cam.get_frame()
    if frame is None:
        detail = "get_frame=None"
    elif getattr(frame, "rgb", None) is None:
        detail = "rgb=None"
    else:
        rgb = np.asarray(frame.rgb)
        detail = f"rgb={tuple(int(v) for v in rgb.shape)}"
    age = "n/a"
    if last_ns is not None:
        age = f"{max(0.0, time.perf_counter() - last_ns / 1e9):.2f}s"
    error = f" error={last_error}" if last_error else ""
    return f"{name}: status={status} {detail} last_frame_age={age}{error}"


def _as_uint8_rgb(rgb: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise RuntimeError(f"相机 {name} RGB 形状异常: {arr.shape}")
    return np.ascontiguousarray(arr)


def _read_images(
    camera_manager,
    camera_names: list[str],
    *,
    timeout_s: float = 2.0,
    wait_new: bool = False,
) -> dict[str, np.ndarray]:
    if camera_manager is None:
        raise RuntimeError("相机未连接，无法读取图像")
    deadline = time.perf_counter() + timeout_s
    missing: list[str] = list(camera_names)
    baselines: dict[str, int] | None = None
    if wait_new:
        # 先记下到位瞬间各相机已有序号，再等更新的物理帧，避免吃到运动中的缓冲。
        baselines = {
            name: int(camera_manager.camera(name).latest_sequence())
            for name in camera_names
        }
        images: dict[str, np.ndarray] = {}
        while time.perf_counter() < deadline:
            for name in camera_names:
                if name in images:
                    continue
                item = camera_manager.camera(name).get_next_frame_after(
                    baselines[name]
                )
                if item is None:
                    continue
                _seq, frame = item
                if frame is None or getattr(frame, "rgb", None) is None:
                    continue
                images[name] = _as_uint8_rgb(frame.rgb, name)
            if len(images) == len(camera_names):
                return images
            time.sleep(0.005)
        missing = [name for name in camera_names if name not in images]
    else:
        while time.perf_counter() < deadline:
            images = {}
            missing = []
            for name in camera_names:
                cam = camera_manager.camera(name)
                frame = cam.get_frame()
                if frame is None or frame.rgb is None:
                    missing.append(name)
                    continue
                images[name] = _as_uint8_rgb(frame.rgb, name)
            if not missing:
                return images
            time.sleep(0.01)
    details = [
        _describe_camera_frame(camera_manager, name) for name in camera_names
    ]
    kind = "新帧" if wait_new else "相机帧"
    print(
        f"[ERROR] 等待{kind}超时 ({timeout_s}s) 掉线={missing or list(camera_names)}",
        flush=True,
    )
    for line in details:
        print(f"[ERROR]   {line}", flush=True)
    raise TimeoutError(
        f"等待{kind}超时 ({timeout_s}s): 掉线={missing or list(camera_names)}; "
        + "; ".join(details)
    )


def _read_observation(
    hw: HardwareBundle,
    cameras: list[str],
    *,
    last_state: np.ndarray | None,
    state_dim: int,
    wait_new: bool = False,
) -> Observation:
    images = _read_images(hw.camera_manager, cameras, wait_new=wait_new)
    left_fb = last_state[:7].tolist() if last_state is not None else hw.left_start_joints_deg
    left = _read_hcx_joints(hw.left_arm, left_fb)
    if int(state_dim) == LEFT_ARM_DIM:
        lg_fb = float(last_state[7]) if last_state is not None else 1.0
        left_g = _read_gripper(hw.left_gripper, fallback=lg_fb)
        state = np.concatenate(
            [left.astype(np.float32), np.asarray([left_g], dtype=np.float32)]
        )
        return Observation(images=images, state=state, timestamp=time.time())

    right_fb = (
        last_state[7:14].tolist() if last_state is not None else hw.right_start_joints_deg
    )
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
    return Observation(images=images, state=state, timestamp=time.time())


def _clamp_joints_by_limits(
    joints: list[float] | np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    side: str,
) -> list[float]:
    """将 7 轴目标角截断到配置限位内。"""
    arr = np.asarray(joints, dtype=np.float64)
    if arr.shape != (7,):
        raise ValueError(f"{side} 关节必须是 7 维，实际为 {arr.shape}")
    clipped = np.clip(arr, lo, hi)
    over = (arr < lo - 1e-6) | (arr > hi + 1e-6)
    if np.any(over):
        axes = [int(i) for i, flag in enumerate(over) if flag]
        details = ", ".join(
            f"J{i} {float(arr[i]):+.3f}→{float(clipped[i]):+.3f}deg"
            for i in axes
        )
        print(f"[WARN] {side} 关节限位截断 轴={axes} {details}", flush=True)
    return clipped.astype(float).tolist()


def _clamp_joints_by_max_delta(
    target: np.ndarray,
    current: np.ndarray,
    max_delta_deg: float,
) -> tuple[list[float], np.ndarray]:
    """将目标关节角相对当前反馈限制在 ±max_delta_deg 内。"""
    target_np = np.asarray(target, dtype=np.float64)
    current_np = np.asarray(current, dtype=np.float64)
    if target_np.shape != current_np.shape:
        raise ValueError(
            f"关节维数不一致: target={target_np.shape} current={current_np.shape}"
        )
    raw_delta = target_np - current_np
    delta = np.clip(raw_delta, -max_delta_deg, max_delta_deg)
    return (current_np + delta).astype(float).tolist(), raw_delta


def _log_max_delta_clip(
    side: str,
    raw_delta: np.ndarray,
    *,
    max_delta_deg: float,
) -> None:
    clipped = np.abs(raw_delta) > (max_delta_deg + 1e-6)
    if not np.any(clipped):
        return
    axes = [int(i) for i, flag in enumerate(clipped) if flag]
    details = ", ".join(
        f"J{i} delta={float(raw_delta[i]):+.3f}deg" for i in axes
    )
    print(
        f"[WARN] {side} max_delta_deg={max_delta_deg:g} 裁剪轴={axes} {details}",
        flush=True,
    )


def _send_action(
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
    feedback_confirm: bool = True,
    move_interrupt: bool = False,
) -> None:
    if hw.left_arm is None or hw.right_arm is None:
        raise RuntimeError("HCX 双臂未连接")
    if not math.isfinite(max_delta_deg) or max_delta_deg <= 0.0:
        raise ValueError("move_joints.max_delta_deg 必须是正的有限数")
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape not in {(DUAL_ARM_DIM,), (LEFT_ARM_DIM,)}:
        raise ValueError(
            f"动作维数应为 {DUAL_ARM_DIM}（双臂）或 {LEFT_ARM_DIM}（左臂），"
            f"实际为 {action.shape}"
        )
    left_only = action.shape == (LEFT_ARM_DIM,)

    left_current = np.asarray(hw.left_arm.joint_angles(), dtype=np.float64)
    left, left_delta = _clamp_joints_by_max_delta(
        action[:7], left_current, max_delta_deg
    )
    _log_max_delta_clip("left", left_delta, max_delta_deg=max_delta_deg)
    left = _clamp_joints_by_limits(
        left, joint_limits_min_deg, joint_limits_max_deg, side="left"
    )
    left_g = float(np.clip(action[7] if left_only else action[14], 0.0, 1.0))

    right: list[float] | None = None
    right_g: float | None = None
    if not left_only:
        right_current = np.asarray(hw.right_arm.joint_angles(), dtype=np.float64)
        right, right_delta = _clamp_joints_by_max_delta(
            action[7:14], right_current, max_delta_deg
        )
        _log_max_delta_clip("right", right_delta, max_delta_deg=max_delta_deg)
        right = _clamp_joints_by_limits(
            right, joint_limits_min_deg, joint_limits_max_deg, side="right"
        )
        right_g = float(np.clip(action[15], 0.0, 1.0))
        print(
            f"  left={[round(v, 3) for v in left]} right={[round(v, 3) for v in right]} "
            f"left_gripper={left_g:.4f} right_gripper={right_g:.4f}",
            flush=True,
        )
    else:
        print(
            f"  left={[round(v, 3) for v in left]} left_gripper={left_g:.4f}",
            flush=True,
        )

    if hw.left_gripper_loop is not None:
        hw.left_gripper_loop.set_opening(left_g)
    elif hw.left_gripper is not None:
        _ = hw.left_gripper.send_normalized(left_g)
    if right_g is not None:
        if hw.right_gripper_loop is not None:
            hw.right_gripper_loop.set_opening(right_g)
        elif hw.right_gripper is not None:
            _ = hw.right_gripper.send_normalized(right_g)

    left_h = hw.left_arm.move_joints(
        left,
        interrupt=move_interrupt,
        acceleration_seconds=acceleration_seconds,
        deceleration_seconds=deceleration_seconds,
        speed_ratio=speed_ratio,
        smooth=1,
        wait=False,
    )
    right_h = None
    if right is not None:
        right_h = hw.right_arm.move_joints(
            right,
            interrupt=move_interrupt,
            acceleration_seconds=acceleration_seconds,
            deceleration_seconds=deceleration_seconds,
            speed_ratio=speed_ratio,
            smooth=1,
            wait=False,
        )
    del left_h, right_h

    if feedback_confirm:
        _confirm_targets_by_feedback(
            hw,
            left,
            right,
            timeout_s=feedback_confirm_timeout_s,
            poll_interval_s=feedback_confirm_poll_interval_s,
            angle_tolerance_deg=angle_tolerance_deg,
        )


def _confirm_targets_by_feedback(
    hw: HardwareBundle,
    left_target: list[float],
    right_target: list[float] | None,
    *,
    timeout_s: float,
    poll_interval_s: float,
    angle_tolerance_deg: float,
) -> None:
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("move_joints.feedback_confirm_timeout_s 必须是正的有限秒数")
    if not math.isfinite(poll_interval_s) or poll_interval_s <= 0.0:
        raise ValueError(
            "move_joints.feedback_confirm_poll_interval_s 必须是正的有限秒数"
        )
    if not math.isfinite(angle_tolerance_deg) or angle_tolerance_deg <= 0.0:
        raise ValueError("move_joints.angle_tolerance_deg 必须是正的有限数")
    if hw.left_arm is None or hw.right_arm is None:
        raise RuntimeError("HCX 双臂未连接，无法确认反馈到位")

    deadline = time.monotonic() + timeout_s
    left_target_np = np.asarray(left_target, dtype=np.float64)
    right_target_np = (
        None if right_target is None else np.asarray(right_target, dtype=np.float64)
    )
    while True:
        left_fb = np.asarray(hw.left_arm.joint_angles(), dtype=np.float64)
        left_ok = np.all(np.abs(left_fb - left_target_np) <= angle_tolerance_deg)
        right_ok = True
        if right_target_np is not None:
            right_fb = np.asarray(hw.right_arm.joint_angles(), dtype=np.float64)
            right_ok = np.all(np.abs(right_fb - right_target_np) <= angle_tolerance_deg)
        if left_ok and right_ok:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise RuntimeError("move_joints 反馈确认超时：关节未在容差内到位")
        _pump_camera_preview(hw)
        time.sleep(min(poll_interval_s, remaining))


def _shutdown(hw: HardwareBundle) -> None:
    if hw.obs_sampler is not None:
        try:
            hw.obs_sampler.stop()
        except Exception as exc:
            print(f"[WARN] 停止观测采样时出错: {exc}")
        hw.obs_sampler = None
    if hw.camera_preview is not None:
        try:
            hw.camera_preview.stop()
        except Exception as exc:
            print(f"[WARN] 停止相机预览时出错: {exc}")
        hw.camera_preview = None
    if hw.left_gripper_loop is not None:
        try:
            hw.left_gripper_loop.stop()
        except Exception as exc:
            print(f"[WARN] 停止左侧夹爪后台时出错: {exc}")
    if hw.right_gripper_loop is not None:
        try:
            hw.right_gripper_loop.stop()
        except Exception as exc:
            print(f"[WARN] 停止右侧夹爪后台时出错: {exc}")
    if hw.left_gripper is not None:
        try:
            hw.left_gripper.disable()
        except Exception:
            pass
        try:
            hw.left_gripper.disconnect()
        except Exception as exc:
            print(f"[WARN] 断开左侧夹爪时出错: {exc}")
    if hw.right_gripper is not None:
        try:
            hw.right_gripper.disable()
        except Exception:
            pass
        try:
            hw.right_gripper.disconnect()
        except Exception as exc:
            print(f"[WARN] 断开右侧夹爪时出错: {exc}")
    if hw.camera_manager is not None:
        try:
            hw.camera_manager.stop()
        except Exception as exc:
            print(f"[WARN] 停止相机时出错: {exc}")
    if hw.hcx_client is not None:
        try:
            hw.hcx_client.close()
        except Exception as exc:
            print(f"[WARN] 关闭 HCX 客户端时出错: {exc}")


def _resolve_train_config(ckpt_path: Path, config_arg: Path | None) -> Path | None:
    """CLI config > checkpoint 旁 config_source.yaml/config.yaml > None（用内嵌）。"""
    if config_arg is not None:
        return config_arg
    for name in ("config_source.yaml", "config.yaml"):
        cand = ckpt_path.parent / name
        if cand.is_file():
            return cand
    return None


def _validate_runtime_contract(cfg: Any, cameras: list[str], stats: dict) -> str:
    if not cameras:
        raise ValueError("训练配置 cameras 为空")
    layout = _runtime_layout(cameras, int(cfg.state_dim), int(cfg.action_dim))
    dim = LEFT_ARM_DIM if layout == "left" else DUAL_ARM_DIM
    if int(cfg.dataset.n_obs_steps) <= 0:
        raise ValueError("dataset.n_obs_steps 必须 > 0")
    if int(cfg.policy.n_action_steps) <= 0:
        raise ValueError("policy.n_action_steps 必须 > 0")
    # stats 为扁平键：state_mean/std/min/max、action_mean/std/min/max（见 robotfm.data.stats）
    required_stat_keys = (
        "state_mean",
        "state_std",
        "state_min",
        "state_max",
        "action_mean",
        "action_std",
        "action_min",
        "action_max",
    )
    missing = [k for k in required_stat_keys if k not in stats]
    if missing:
        raise ValueError(f"checkpoint stats 缺少归一化字段: {missing}")
    for key in required_stat_keys:
        shape = tuple(np.asarray(stats[key]).shape)
        if shape != (dim,):
            raise ValueError(
                f"checkpoint stats[{key}] 形状应为 ({dim},)，实际为 {shape}"
            )
    return layout


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

    _apply_rtc_overrides(cfg, deploy.get("rtc") or {})
    rtc_cfg = _normalize_rtc_config(cfg.policy.rtc)
    rtc_enabled = bool(rtc_cfg.enabled)

    # 训练 YAML 的 history_noise_std 只用于训练；真机 / 开环 val 一律不加噪
    train_history_noise = float(getattr(cfg.policy, "history_noise_std", 0.0) or 0.0)
    cfg.policy.history_noise_std = 0.0

    norm_mode = cfg.dataset.norm_mode
    n_obs = int(cfg.dataset.n_obs_steps)
    n_action_steps = int(cfg.policy.n_action_steps)
    horizon = int(cfg.dataset.horizon)
    train_fps = int(cfg.fps)
    obs_mode = str(deploy["obs_mode"])
    obs_fps = deploy["obs_fps"]
    if obs_mode == "after_action":
        obs_fps = None
    elif obs_fps is None:
        obs_fps = train_fps
    max_steps = deploy["max_steps"]
    exec_action_steps = deploy["exec_action_steps"]
    if exec_action_steps is None:
        # after_action 默认仍只跑 chunk[0]；fps 默认跑满 n_action_steps
        exec_action_steps = 1 if obs_mode == "after_action" else n_action_steps
    if exec_action_steps > n_action_steps:
        raise ValueError(
            f"deploy.yaml exec_action_steps={exec_action_steps} 不能大于 "
            f"policy.n_action_steps={n_action_steps}（后者决定模型输出长度，改大会导致权重 shape 不匹配）"
        )
    if obs_mode == "after_action" and rtc_enabled:
        raise ValueError(
            "obs_mode=after_action 与 rtc.enabled=true 互斥："
            "到位采观测按 exec_action_steps 执行后立刻再推理，不能走 RTC leftover"
        )
    policy_type = str(cfg.policy.type).lower()
    if rtc_enabled and policy_type in {"a2a", "n_a2a"} and n_action_steps != horizon:
        raise ValueError(
            "A2A RTC 要求 n_action_steps == horizon "
            f"(got n_action_steps={n_action_steps}, horizon={horizon})"
        )
    if rtc_enabled and int(rtc_cfg.execution_horizon) >= n_action_steps:
        raise ValueError(
            "RTC execution_horizon 必须小于 n_action_steps，否则 leftover 为空、"
            f"无法做 prefix 引导 (execution_horizon={rtc_cfg.execution_horizon}, "
            f"n_action_steps={n_action_steps})"
        )

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    policy = build_policy(cfg, stats)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()
    policy_cfg = getattr(policy, "cfg", None)
    if hasattr(policy_cfg, "history_noise_std"):
        # n_a2a 构建时若 std<=0 会回填 0.1；推理侧再强制关掉
        policy_cfg.history_noise_std = 0.0
    print(
        f"[INFO] 策略已就绪 layout={layout} device={device} cameras={cameras} "
        f"norm={norm_mode} n_obs={n_obs} n_action_steps={n_action_steps} "
        f"exec_action_steps={exec_action_steps} train_fps={train_fps} "
        f"obs_mode={obs_mode} obs_fps={obs_fps if obs_fps is not None else '-'} "
        f"history_noise_std=0 (train={train_history_noise:g})"
    )
    print(
        f"[INFO] rtc.enabled={rtc_enabled} guidance={rtc_cfg.guidance_enabled} "
        f"delay={rtc_cfg.inference_delay} exec_h={rtc_cfg.execution_horizon} "
        f"schedule={rtc_cfg.prefix_attention_schedule}"
    )
    if rtc_enabled:
        print(
            "[INFO] RTC 闭环：执行 execution_horizon 步后 replan，"
            "用未执行 leftover 做 prefix 引导；忽略 exec_action_steps"
        )

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
        hw.camera_manager = _connect_cameras(teleop_yaml, cameras)
        if deploy["display_cameras"]:
            hw.camera_preview = CameraPreviewLoop(
                hw.camera_manager, cameras, fps=PREVIEW_FPS
            )
            hw.camera_preview.start()
            if hw.camera_preview is not None and hw.camera_preview.is_active:
                print(
                    f"[INFO] 相机预览窗口已启动 window={PREVIEW_WINDOW_NAME} "
                    f"fps={int(PREVIEW_FPS)} cameras={cameras}"
                )
            else:
                hw.camera_preview = None

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
                _read_hcx_joints(hw.right_arm, hw.right_start_joints_deg).astype(float).tolist(),
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
            _confirm_targets_by_feedback(
                hw,
                _read_hcx_joints(hw.left_arm, hw.left_start_joints_deg).astype(float).tolist(),
                deploy["right_start_joints_deg"],
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

        pre_crop_size = cfg.dataset.pre_crop_size
        resize_size = cfg.dataset.resize_size
        crop_size = cfg.dataset.crop_size
        eval_fixed_crop = bool(cfg.dataset.eval_fixed_crop)
        print(
            f"[INFO] 观测入队即 pre_crop/resize/crop "
            f"(pre_crop={pre_crop_size} resize={resize_size} crop={crop_size} "
            f"fixed={eval_fixed_crop})"
        )
        if obs_mode == "after_action":
            obs_queue = StepObservationQueue(
                hw,
                cameras,
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
                "启动等各相机新帧并重复填充；之后每步动作到位再等新帧入队"
                "（跳过运动期间已在缓冲里的旧 RGB）"
            )
            print(
                f"[INFO] 开始闭环 obs_mode=after_action："
                f"每次推理执行 chunk 前 {exec_action_steps}/{n_action_steps} 步，"
                f"每步到位后等新帧，最多 {max_steps} 步"
            )
            step_i = 0
            while step_i < max_steps:
                _pump_camera_preview(hw)
                obs_history = obs_queue.snapshot()
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
                _log_inference_result(
                    step_i=step_i, pred_norm=pred, pred_phys=pred_phys
                )
                n_chunk = int(pred_phys.shape[0])
                n_exec = min(exec_action_steps, n_chunk, max_steps - step_i)
                print(
                    f"[INFO] step={step_i} 推理 chunk={n_chunk} "
                    f"执行前 {n_exec} 步"
                    + (
                        f"，丢弃其余 {n_chunk - n_exec}"
                        if n_exec < n_chunk
                        else ""
                    )
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
                        feedback_confirm_timeout_s=deploy[
                            "move_feedback_confirm_timeout_s"
                        ],
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
            return

        if obs_fps is None:
            raise ValueError("obs_mode=fps 需要有效的 obs_fps")
        hw.obs_sampler = FpsObservationSampler(
            hw,
            cameras,
            n_obs_steps=n_obs,
            fps=obs_fps,
            state_dim=int(cfg.state_dim),
            pre_crop_size=pre_crop_size,
            resize_size=resize_size,
            crop_size=crop_size,
            eval_fixed_crop=eval_fixed_crop,
        )
        hw.obs_sampler.start()
        fill_timeout_s = max(5.0, float(n_obs) / float(obs_fps) + 3.0)
        print(
            f"[INFO] 观测采样已启动 n_obs={n_obs} obs_fps={obs_fps} "
            f"train_fps={train_fps} fill_timeout_s={fill_timeout_s:.1f}"
        )
        hw.obs_sampler.wait_until_filled(timeout_s=fill_timeout_s)
        print(f"[INFO] 观测缓冲已就绪 frames={n_obs} obs_fps={obs_fps}")

        chunk_actions: list[np.ndarray] = []
        chunk_idx = 0
        action_queue = ActionQueue(rtc_cfg) if rtc_enabled else None
        rtc_replan_threshold = (
            n_action_steps - int(rtc_cfg.execution_horizon) if rtc_enabled else 0
        )
        print(
            f"[INFO] 开始闭环，最多 {max_steps} 步"
            + (
                f" (RTC qsize<={rtc_replan_threshold})"
                if rtc_enabled
                else ""
            )
        )

        for step_i in range(max_steps):
            t0 = time.perf_counter()
            _pump_camera_preview(hw)
            assert hw.obs_sampler is not None
            hw.obs_sampler.raise_if_unhealthy()
            if rtc_enabled:
                assert action_queue is not None
                if action_queue.qsize() <= rtc_replan_threshold:
                    leftover = action_queue.get_left_over()
                    if leftover is not None and leftover.shape[0] == 0:
                        leftover = None
                    leftover_len = 0 if leftover is None else int(leftover.shape[0])
                    if leftover is not None:
                        leftover = leftover.to(device)
                    obs_history = hw.obs_sampler.snapshot()
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
                    with torch.no_grad():
                        pred = policy.sample_actions(
                            batch,
                            prev_chunk_left_over=leftover,
                            inference_delay=rtc_cfg.inference_delay,
                            execution_horizon=rtc_cfg.execution_horizon,
                        )[0].cpu()
                    processed = denormalize_predicted_action(
                        pred,
                        stats,
                        norm_mode,
                        q_now_phys=np.asarray(obs_history[-1].state, dtype=np.float32),
                        predict_joint_delta=bool(cfg.policy.predict_joint_delta),
                        joint_mask=joint_mask_from_names(cfg.action_names, cfg.action_dim),
                    )
                    _log_inference_result(
                        step_i=step_i, pred_norm=pred, pred_phys=processed
                    )
                    # 阻塞推理：merge delay=0，连续性靠 leftover prefix 引导，不靠执行旧点。
                    action_queue.merge(pred, processed, real_delay=0)
                    print(
                        f"[INFO] step={step_i} RTC 重新规划 leftover={leftover_len} "
                        f"chunk={action_queue.qsize()}"
                    )
                action_t = action_queue.get()
                if action_t is None:
                    raise RuntimeError("RTC ActionQueue 为空")
                action = np.asarray(action_t.numpy(), dtype=np.float32)
            else:
                if chunk_idx >= len(chunk_actions):
                    obs_history = hw.obs_sampler.snapshot()
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
                    _log_inference_result(
                        step_i=step_i, pred_norm=pred, pred_phys=pred_phys
                    )
                    chunk_actions = [
                        np.asarray(a, dtype=np.float32)
                        for a in pred_phys[:exec_action_steps]
                    ]
                    chunk_idx = 0
                    print(f"[INFO] step={step_i} 重新规划 chunk={len(chunk_actions)}")

                action = chunk_actions[chunk_idx]
                chunk_idx += 1
            print(f"[INFO] step={step_i}", end="")
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
            hw.obs_sampler.raise_if_unhealthy()
            # 动作步节拍仍按训练 fps，与观测采样 obs_fps 解耦
            _pace_step(t0, train_fps)
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        _shutdown(hw)


if __name__ == "__main__":
    main()
