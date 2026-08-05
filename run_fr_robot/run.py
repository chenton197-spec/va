#!/usr/bin/env python3
"""FR3 + Gloria-M + Orbbec 上的策略闭环部署。

训练相关配置优先来自训练 YAML（``deploy.yaml`` 的 ``config`` 或
``--config``），否则回退到 checkpoint 内嵌 config（相机、维度、
norm_mode、resize/crop、n_obs_steps、n_action_steps、fps 等）。
权重与 stats 仍从 checkpoint 加载。机械臂走与
``ct/scripts/move_to_start_pose.py`` 相同的 FAIRINO SDK：

  /home/casbot/teleop_project/fairino390/linux/fairino/Robot.py

  - 启动时 MoveJ 到 deploy.yaml 中的 start_pose
  - 闭环用独立后台线程 ServoJ 下发关节角（度，固定 ``cmdT=8ms``）
  - 写入 ActionQueue 的同时做 Z 校验、五次插值与滤波，生成 8ms 密采样；
    ServoJ 只回放；主循环负责观测 / RTC 推理
  - 关节目标经样条插值 + 低通滤波，并用 ``max_joint_vel_deg_s`` 限速
    （默认 150 deg/s），避免 action chunk 衔接跳变
  - GetForwardKin / Z 限位校验同在 ServoJ 线程，避免 XML-RPC 并发
  - 夹爪目标写入后台循环，按 ``gripper_rate_hz``（默认 125）连续
    ``send_normalized``（MIT 不能只偶发一帧）

用法（在 ``ct/va`` 下，conda 环境 ``lerobot``）::

    # 部署 + 训练配置见同目录 deploy.yaml
    PYTHONPATH=. python run_robot/run_policy.py

    # CLI 覆盖训练配置 / checkpoint
    PYTHONPATH=. python run_robot/run_policy.py \\
        --config configs/shine_shoes_a2a_noise_limits.yaml \\
        --checkpoint model/a2a_noise_shine_shoes_limits_260730175409/checkpoint_090000.pt

    # 启用 RTC（也可在训练 YAML policy.rtc.enabled / deploy.yaml rtc 段开启）
    PYTHONPATH=. python run_fr_robot/run.py --rtc --inference-delay 2 --execution-horizon 4

    # 可选：torch.compile encoder + denoiser（也可在 deploy.yaml 设 compile: true）
    PYTHONPATH=. python run_fr_robot/run.py --compile
"""

from __future__ import annotations

import argparse
import importlib.util
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

VA_ROOT = Path(__file__).resolve().parents[1]
TELEOP_ROOT = Path("/home/casbot/teleop_project")
SDK_ROBOT_PY = TELEOP_ROOT / "fairino390" / "linux" / "fairino" / "Robot.py"
SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_YAML = SCRIPT_DIR / "deploy.yaml"
DEFAULT_ROBOT_IP = "192.168.57.3"

# 优先使用 teleop_project 下的 orbbec_sdk / teleop_sdk / gloria_m_sdk。
if str(TELEOP_ROOT) not in sys.path:
    sys.path.insert(0, str(TELEOP_ROOT))
if str(VA_ROOT) not in sys.path:
    sys.path.insert(0, str(VA_ROOT))

from robotfm.config import _normalize_rtc_config, load_config  # noqa: E402
from robotfm.data.dataset import crop_images, resize_images  # noqa: E402
from robotfm.data.stats import denormalize, normalize  # noqa: E402
from robotfm.policies.rtc import ActionQueue, RTCConfig  # noqa: E402
from robotfm.train import build_policy  # noqa: E402
from robotfm.types import Observation  # noqa: E402

from arm_control import (  # noqa: E402
    SERVOJ_CMDT_S,
    BackgroundServoJLoop,
    forward_kin_desc_pos,
)


# ---------------------------------------------------------------------------
# FAIRINO 辅助函数（对齐 move_to_start_pose.py）
# ---------------------------------------------------------------------------


def _load_robot_module():
    """从 Robot.py 路径动态加载 FAIRINO SDK 模块。"""
    if not SDK_ROBOT_PY.is_file():
        raise FileNotFoundError(f"找不到 FAIRINO SDK: {SDK_ROBOT_PY}")
    spec = importlib.util.spec_from_file_location("fairino_robot_sdk", SDK_ROBOT_PY)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 FAIRINO SDK: {SDK_ROBOT_PY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_robot_ip(robot_ip: str | None, teleop_yaml: Path) -> str:
    """优先 deploy.yaml 的 robot_ip，其次 teleop.yaml 的 fr3.robot_ip，最后默认 IP。"""
    if robot_ip:
        return robot_ip
    if teleop_yaml.is_file():
        with teleop_yaml.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        ip = (cfg.get("fr3") or {}).get("robot_ip")
        if ip:
            return str(ip)
    return DEFAULT_ROBOT_IP


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    """绝对路径原样返回；相对路径相对于 ``base``。"""
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def _load_deploy_config(path: Path) -> dict[str, Any]:
    """读取 deploy.yaml：checkpoint / config / teleop / MoveJ / max_steps / start_pose。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到部署配置: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"部署配置根节点必须是映射: {path}")

    required = ("checkpoint", "teleop_yaml", "vel", "max_steps", "start_pose", "z_limit")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"deploy.yaml 缺少字段: {missing}")

    joints_deg, tcp_pose = _parse_start_pose(data["start_pose"], path)

    ckpt = _resolve_path(data["checkpoint"], base=VA_ROOT)
    teleop_yaml = _resolve_path(data["teleop_yaml"], base=SCRIPT_DIR)
    train_cfg = data.get("config")
    train_cfg_path = (
        _resolve_path(train_cfg, base=VA_ROOT) if train_cfg else None
    )
    robot_ip = data.get("robot_ip")
    if robot_ip is not None:
        robot_ip = str(robot_ip)

    tool = data.get("tool")
    user = data.get("user")
    gripper_rate_hz = float(data.get("gripper_rate_hz", 125.0))
    if gripper_rate_hz <= 0.0:
        raise ValueError("deploy.yaml 的 gripper_rate_hz 必须 > 0")
    max_joint_vel_deg_s = float(data.get("max_joint_vel_deg_s", 150.0))
    if max_joint_vel_deg_s <= 0.0:
        raise ValueError("deploy.yaml 的 max_joint_vel_deg_s 必须 > 0")
    max_joint_step_raw = data.get("max_joint_step_deg", 15.0)
    max_joint_step_deg = (
        None if max_joint_step_raw is None else float(max_joint_step_raw)
    )
    if max_joint_step_deg is not None and max_joint_step_deg <= 0.0:
        raise ValueError("deploy.yaml 的 max_joint_step_deg 必须 > 0")

    # torch.compile for encoder + denoiser; default off (RTC autograd path).
    compile_enabled = bool(data.get("compile", False))

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
        "checkpoint": ckpt,
        "config": train_cfg_path,
        "teleop_yaml": teleop_yaml,
        "robot_ip": robot_ip,
        "vel": float(data["vel"]),
        "tool": int(tool) if tool is not None else None,
        "user": int(user) if user is not None else None,
        "max_steps": int(data["max_steps"]),
        "joints_deg": joints_deg,
        "tcp_pose": tcp_pose,
        "z_limit": float(data["z_limit"]),
        "gripper_rate_hz": gripper_rate_hz,
        "max_joint_vel_deg_s": max_joint_vel_deg_s,
        "max_joint_step_deg": max_joint_step_deg,
        "compile": compile_enabled,
        "rtc": rtc_override,
    }


def _apply_rtc_overrides(cfg: Any, deploy_rtc: dict[str, Any], args: argparse.Namespace) -> None:
    """按 CLI > deploy.yaml rtc > 训练 YAML 优先级覆盖 policy.rtc。"""
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

    if args.rtc:
        enabled = True
    if getattr(args, "no_guidance", False):
        guidance_enabled = False
    if args.inference_delay is not None:
        inference_delay = int(args.inference_delay)
    if args.execution_horizon is not None:
        execution_horizon = int(args.execution_horizon)

    need_rebuild = (
        args.rtc
        or getattr(args, "no_guidance", False)
        or args.inference_delay is not None
        or args.execution_horizon is not None
        or bool(deploy_rtc)
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


def _resolve_compile_enabled(deploy: dict[str, Any], args: argparse.Namespace) -> bool:
    """CLI --compile overrides deploy.yaml ``compile`` (default False)."""
    if args.compile:
        return True
    return bool(deploy.get("compile", False))


def _compile_policy_submodules(policy: torch.nn.Module) -> list[str]:
    """Compile encoder + denoiser + A2A action_decoder (RTC VJP hot path).

    Uses ``mode=\"default\"`` (not ``reduce-overhead``): CUDA graphs in
    reduce-overhead conflict with SpatialSoftmax grid buffers on repeated calls.
    """
    compiled: list[str] = []
    for name in ("encoder", "unet", "flow_net", "action_decoder"):
        mod = getattr(policy, name, None)
        if mod is None:
            continue
        setattr(policy, name, torch.compile(mod, mode="default"))
        compiled.append(name)
    return compiled


def _warmup_compiled_policy(
    policy: torch.nn.Module,
    cfg: Any,
    device: torch.device,
    *,
    crop_size: int | None,
) -> None:
    """One dummy ``sample_actions`` to finish Inductor compile + CUDA graphs."""
    n_cams = len(cfg.cameras)
    n_obs = int(cfg.dataset.n_obs_steps)
    h = w = int(crop_size or cfg.dataset.resize_size or 224)
    batch = {
        "obs_images": torch.zeros(
            1, n_cams, n_obs, 3, h, w, device=device, dtype=torch.float32
        ),
        "obs_state": torch.zeros(
            1, n_obs, int(cfg.state_dim), device=device, dtype=torch.float32
        ),
    }
    with torch.no_grad():
        _ = policy.sample_actions(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FR3 + Gloria-M + Orbbec 策略闭环部署"
    )
    parser.add_argument(
        "--deploy",
        type=str,
        default=str(DEPLOY_YAML),
        help="部署 YAML（默认 run_robot/deploy.yaml）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="训练配置 YAML（覆盖 deploy.yaml 的 config；相对路径相对于 ct/va）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint 路径（覆盖 deploy.yaml；相对路径相对于 ct/va）",
    )
    parser.add_argument(
        "--rtc",
        action="store_true",
        help="启用 RTC（覆盖训练 YAML / deploy.yaml；对齐 eval_flow_matching.py）",
    )
    parser.add_argument(
        "--no-guidance",
        action="store_true",
        help="RTC 调度保留（ahead+discard），关闭采样前缀引导（naive async）",
    )
    parser.add_argument(
        "--inference-delay",
        type=int,
        default=None,
        help="RTC 模拟推理延迟（动作步数）",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=None,
        help="RTC execution_horizon",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile encoder + unet/flow_net（覆盖 deploy.yaml compile）",
    )
    return parser.parse_args()


def _parse_start_pose(
    start: Any, source: Path
) -> tuple[list[float], list[float] | None]:
    """解析 deploy.yaml 中的 start_pose：joints_deg 必填，tcp_pose 可选。"""
    if not isinstance(start, dict):
        raise ValueError(f"start_pose 必须是映射: {source}")
    joints = start.get("joints_deg")
    if joints is None or len(joints) != 6:
        raise ValueError(f"start_pose 缺少合法 joints_deg (6 轴): {source}")
    joints_f = [float(v) for v in joints]
    tcp = start.get("tcp_pose")
    tcp_f = [float(v) for v in tcp] if tcp is not None and len(tcp) == 6 else None
    return joints_f, tcp_f


def _prepare_robot(robot) -> None:
    """切自动模式并上使能，便于执行 MoveJ。"""
    try:
        pkg = robot.robot_state_pkg
        if getattr(pkg, "main_code", 0) != 0:
            print(
                f"[WARN] 存在故障码 main={pkg.main_code} sub={pkg.sub_code}，尝试清除..."
            )
            ret = robot.ResetAllError()
            print(f"[INFO] ResetAllError -> {ret}")
            time.sleep(0.3)
        if getattr(pkg, "robot_mode", 0) != 0:
            ret = robot.Mode(0)
            print(f"[INFO] Mode(0) -> {ret}")
        else:
            print("[INFO] 已处于自动模式")
    except AttributeError:
        ret = robot.Mode(0)
        print(f"[INFO] Mode(0) -> {ret}")

    ret = robot.RobotEnable(1)
    print(f"[INFO] RobotEnable(1) -> {ret}")
    time.sleep(0.3)


def _current_tool_user(robot) -> tuple[int, int]:
    """读取当前工具号 / 工件号，失败则回退 0。"""
    tool, user = 0, 0
    try:
        ret, tool_id = robot.GetActualTCPNum(0)
        if ret == 0 and tool_id is not None:
            tool = int(tool_id)
    except Exception:
        pass
    try:
        ret, user_id = robot.GetActualWObjNum(0)
        if ret == 0 and user_id is not None:
            user = int(user_id)
    except Exception:
        pass
    return tool, user


def _pace_step(step_start: float, fps: int) -> None:
    """按 cfg.fps 节拍 sleep，使墙钟步频接近训练采集频率。"""
    if fps <= 0:
        return
    remain = (1.0 / fps) - (time.perf_counter() - step_start)
    if remain > 0:
        time.sleep(remain)


# ---------------------------------------------------------------------------
# 观测预处理（与 robotfm.eval / 训练评估时的固定中心裁剪一致）
# ---------------------------------------------------------------------------


def _preprocess_images(
    images: dict[str, np.ndarray],
    *,
    resize_size: int | None,
    crop_size: int | None,
    eval_fixed_crop: bool,
) -> dict[str, np.ndarray]:
    """相机原图 HWC uint8 → 策略分辨率 HWC float32 [0,1]（resize + 可选中心 crop）。

    入队时做一次，避免每次推理对整段历史重复预处理。
    """
    out: dict[str, np.ndarray] = {}
    for name, rgb in images.items():
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        t = resize_images(t, resize_size)
        if crop_size is not None and eval_fixed_crop:
            t = crop_images(t, crop_size, random=False)
        out[name] = t.squeeze(0).permute(1, 2, 0).contiguous().numpy()
    return out


def _prepare_observation(
    obs: Observation,
    *,
    resize_size: int | None,
    crop_size: int | None,
    eval_fixed_crop: bool,
) -> Observation:
    """替换图像为入队即用的 resize/crop 结果；state 原样保留。"""
    return Observation(
        images=_preprocess_images(
            obs.images,
            resize_size=resize_size,
            crop_size=crop_size,
            eval_fixed_crop=eval_fixed_crop,
        ),
        state=np.asarray(obs.state, dtype=np.float32),
        timestamp=obs.timestamp,
    )


def _append_observation(
    obs_history: list[Observation],
    obs: Observation,
    *,
    n_obs_steps: int,
) -> None:
    """追加观测并只保留最近 ``n_obs_steps`` 帧。"""
    obs_history.append(obs)
    overflow = len(obs_history) - n_obs_steps
    if overflow > 0:
        del obs_history[:overflow]


def _build_obs_batch(
    obs_history: list[Observation],
    cameras: list[str],
    n_obs_steps: int,
    stats: dict,
    norm_mode: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """将已预处理的观测历史转为策略输入 batch（batch_size=1）。

    - 图像：历史中已是 HWC float32 [0,1]（入队时 resize/crop），此处只转 CHW 堆叠
    - 状态：按 checkpoint 的 norm_mode 归一化
    - 历史不足时重复最早一帧（与训练 pad 一致）
    """
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
    }


def _clone_observation(obs: Observation) -> Observation:
    """深拷贝一帧观测，供推理线程异步组 batch 时不受主循环追加影响。"""
    return Observation(
        images={k: np.asarray(v).copy() for k, v in obs.images.items()},
        state=np.asarray(obs.state).copy(),
        timestamp=obs.timestamp,
    )


@dataclass
class _RtcInferJob:
    """单次 RTC 推理结果句柄（由 ``_RtcInferWorker`` 填写）。"""

    idx0: int
    leftover_len: int
    started_at: float
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _pred: torch.Tensor | None = None
    _error: BaseException | None = None
    _infer_ms: float | None = None
    _done: bool = False

    def _set_ok(self, pred: torch.Tensor, infer_ms: float) -> None:
        with self._lock:
            self._pred = pred
            self._infer_ms = infer_ms
            self._done = True

    def _set_err(self, exc: BaseException) -> None:
        with self._lock:
            self._error = exc
            self._done = True

    def poll(
        self,
    ) -> tuple[bool, torch.Tensor | None, BaseException | None, float | None]:
        """返回 ``(done, pred, error, infer_ms)``；未完成时其余为 None。"""
        with self._lock:
            if not self._done:
                return False, None, None, None
            return True, self._pred, self._error, self._infer_ms


class _RtcInferWorker:
    """常驻推理线程：只建一次 CUDA/cuBLAS context，避免每次 new Thread 冷启动。

    主线程不得在 busy 期间再调用 ``policy``；``batch_factory`` 在本线程执行。
    """

    def __init__(self, device: torch.device):
        self._device = device
        self._requests: queue.Queue[
            tuple[
                _RtcInferJob,
                Any,
                Any,
                torch.Tensor | None,
                int,
                int,
            ]
            | None
        ] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._warm_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._loop, name="rtc-infer", daemon=True
        )

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=60.0):
            raise RuntimeError("RTC infer worker CUDA warmup timed out")
        if self._warm_error is not None:
            raise RuntimeError(
                f"RTC infer worker CUDA warmup failed: {self._warm_error}"
            ) from self._warm_error

    def shutdown(self, *, join_timeout_s: float = 2.0) -> None:
        self._stop.set()
        # Unblock a waiting get(); if a job is in-flight, sentinel follows it.
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            try:
                self._requests.get_nowait()
            except queue.Empty:
                pass
            try:
                self._requests.put_nowait(None)
            except queue.Full:
                pass
        if self._thread.is_alive():
            self._thread.join(timeout=join_timeout_s)

    def submit(
        self,
        *,
        policy: Any,
        batch_factory: Any,
        leftover: torch.Tensor | None,
        inference_delay: int,
        execution_horizon: int,
        idx0: int,
        leftover_len: int,
    ) -> _RtcInferJob:
        if self._stop.is_set():
            raise RuntimeError("RTC infer worker already shut down")
        job = _RtcInferJob(
            idx0=idx0,
            leftover_len=leftover_len,
            started_at=time.perf_counter(),
        )
        self._requests.put(
            (
                job,
                policy,
                batch_factory,
                leftover,
                int(inference_delay),
                int(execution_horizon),
            )
        )
        return job

    def _loop(self) -> None:
        try:
            if self._device.type == "cuda":
                # bare "cuda" has no index; set_device needs cuda:N or int
                cuda_idx = (
                    self._device.index
                    if self._device.index is not None
                    else torch.cuda.current_device()
                )
                torch.cuda.set_device(cuda_idx)
                # Warm primary context + cuBLAS path used by RTC autograd.grad.
                x = torch.zeros(8, device=self._device, dtype=torch.float32)
                x = x.detach().requires_grad_(True)
                y = (x * x).sum()
                _ = torch.autograd.grad(y, x)[0]
                torch.cuda.synchronize()
        except BaseException as exc:
            self._warm_error = exc
        finally:
            self._ready.set()

        if self._warm_error is not None:
            return

        while not self._stop.is_set():
            req = self._requests.get()
            if req is None:
                break
            (
                job,
                policy,
                batch_factory,
                leftover,
                inference_delay,
                execution_horizon,
            ) = req
            try:
                t0 = time.perf_counter()
                batch = batch_factory()
                # RTC denoise uses torch.enable_grad() internally.
                with torch.no_grad():
                    pred = policy.sample_actions(
                        batch,
                        prev_chunk_left_over=leftover,
                        inference_delay=inference_delay,
                        execution_horizon=execution_horizon,
                    )[0]
                if self._device.type == "cuda":
                    torch.cuda.synchronize()
                infer_ms = (time.perf_counter() - t0) * 1000.0
                job._set_ok(pred, infer_ms)
            except BaseException as exc:
                job._set_err(exc)


# ---------------------------------------------------------------------------
# 硬件封装
# ---------------------------------------------------------------------------


@dataclass
class HardwareBundle:
    """真机资源句柄：机械臂、夹爪、相机与 ServoJ 相关状态。"""

    robot: Any | None = None
    gripper: Any | None = None
    gripper_loop: "BackgroundGripperLoop | None" = None
    servoj_loop: "BackgroundServoJLoop | None" = None
    camera_manager: Any | None = None
    camera_names: tuple[str, ...] = ()
    tool: int = 0
    user: int = 0
    start_joints_deg: list[float] | None = None


class BackgroundGripperLoop:
    """后台按固定频率刷 Gloria-M ``send_normalized``，策略只更新目标开合。"""

    def __init__(self, gripper: Any, *, rate_hz: float = 125.0):
        if rate_hz <= 0.0:
            raise ValueError("gripper rate_hz 必须 > 0")
        self._gripper = gripper
        self._period_s = 1.0 / float(rate_hz)
        self._rate_hz = float(rate_hz)
        self._lock = threading.Lock()
        self._target = 1.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warn_every = max(1, int(self._rate_hz))
        self._fail_streak = 0

    @property
    def rate_hz(self) -> float:
        return self._rate_hz

    def set_opening(self, opening: float) -> None:
        value = float(np.clip(opening, 0.0, 1.0))
        with self._lock:
            self._target = value

    def get_opening(self) -> float:
        with self._lock:
            return float(self._target)

    def start(self, initial_opening: float | None = None) -> None:
        if self._thread is not None:
            raise RuntimeError("夹爪后台循环已在运行")
        if initial_opening is not None:
            self.set_opening(initial_opening)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="gloria-m-gripper-loop",
            daemon=True,
        )
        self._thread.start()
        print(
            f"[INFO] Gloria-M 后台循环已启动: {self._rate_hz:g} Hz，"
            f"初始目标={self.get_opening():.4f}"
        )

    def stop(self, *, join_timeout_s: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout_s)
        self._thread = None

    def _run(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            with self._lock:
                target = self._target
            try:
                ok = bool(self._gripper.send_normalized(target))
            except Exception as exc:
                ok = False
                self._fail_streak += 1
                if self._fail_streak == 1 or self._fail_streak % self._warn_every == 0:
                    print(f"[WARN] Gloria-M 后台 send 异常: {exc}")
            else:
                if ok:
                    self._fail_streak = 0
                else:
                    self._fail_streak += 1
                    if self._fail_streak == 1 or self._fail_streak % self._warn_every == 0:
                        print("[WARN] Gloria-M 后台 send_normalized 失败")

            next_t += self._period_s
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0.0:
                if self._stop.wait(timeout=sleep_s):
                    break
            else:
                # 过载时丢掉落后节拍，避免追赶打爆串口。
                next_t = time.perf_counter()


def _connect_cameras(teleop_yaml: Path, required: list[str]) -> Any:
    """按训练相机名顺序启动 Orbbec（配置来自 teleop.yaml）。"""
    from orbbec_sdk import OrbbecManager, load_orbbec_camera_configs

    all_configs = load_orbbec_camera_configs(teleop_yaml)
    by_name = {c.name: c for c in all_configs}
    missing = [n for n in required if n not in by_name]
    if missing:
        raise ValueError(f"teleop.yaml 缺少训练所需相机: {missing}")
    configs = tuple(by_name[n] for n in required)
    manager = OrbbecManager(configs)
    manager.start()
    print(f"[INFO] Orbbec 已启动: {[c.name for c in configs]}")
    return manager


def _connect_gripper(teleop_yaml: Path) -> Any:
    """连接 Gloria-M 夹爪（与采集侧同一适配器）。"""
    from teleop_sdk.adapters.gloria_m import GloriaMGripperFollower
    from teleop_sdk.config import load_runtime_config

    runtime = load_runtime_config(teleop_yaml)
    if not runtime.gloria_m.enabled:
        raise RuntimeError("teleop.yaml 中 gloria_m.enabled=false，无法控制夹爪")
    gripper = GloriaMGripperFollower(runtime.gloria_m)
    gripper.connect()
    return gripper


def _connect_robot(robot_ip: str) -> Any:
    """连接 FR3：RPC → 清错 / 自动模式 / 上使能。"""
    robot_mod = _load_robot_module()
    print(f"[INFO] 连接 FR3: {robot_ip}")
    robot = robot_mod.RPC(robot_ip)
    time.sleep(0.5)
    _prepare_robot(robot)
    return robot


def _movej_start_pose(
    robot,
    joints_deg: list[float],
    tcp_pose: list[float] | None,
    vel: float,
    tool: int | None,
    user: int | None,
) -> tuple[list[float], int, int]:
    """阻塞 MoveJ 到给定起始关节位姿。"""
    if tool is None or user is None:
        cur_tool, cur_user = _current_tool_user(robot)
        tool = cur_tool if tool is None else tool
        user = cur_user if user is None else user

    desc_pos = tcp_pose if tcp_pose is not None else forward_kin_desc_pos(robot, joints_deg)
    print(f"[INFO] MoveJ joints_deg = {joints_deg}")
    if desc_pos is not None:
        print(f"[INFO] MoveJ desc_pos (tcp) = {desc_pos}")

    # blendT=-1.0：阻塞直到运动到位
    kwargs: dict[str, Any] = {
        "joint_pos": joints_deg,
        "tool": tool,
        "user": user,
        "vel": float(vel),
        "blendT": -1.0,
    }
    if desc_pos is not None:
        kwargs["desc_pos"] = desc_pos

    error = robot.MoveJ(**kwargs)
    if error != 0:
        raise RuntimeError(f"MoveJ 失败 error={error}")
    ret, now = robot.GetActualJointPosDegree(0)
    print(f"[INFO] MoveJ 完成，当前关节角 = {list(now) if ret == 0 else now}")
    return joints_deg, int(tool), int(user)


def _read_joints_deg(robot, fallback: list[float] | None) -> np.ndarray:
    """优先读状态包关节角，失败再 RPC；再失败用 fallback。"""
    if robot is None:
        if fallback is None:
            raise RuntimeError("无机器人且无 fallback 关节角")
        return np.asarray(fallback, dtype=np.float32)
    try:
        values = np.asarray(robot.robot_state_pkg.jt_cur_pos, dtype=np.float32)
        if values.shape == (6,) and np.isfinite(values).all():
            return values
    except Exception:
        pass
    ret, angles = robot.GetActualJointPosDegree(0)
    if ret == 0 and angles is not None and len(angles) == 6:
        return np.asarray(angles, dtype=np.float32)
    if fallback is not None:
        print("[WARN] 读关节失败，使用 start_pose / 上一帧")
        return np.asarray(fallback, dtype=np.float32)
    raise RuntimeError(f"读关节失败 ret={ret}")


def _read_gripper(gripper, fallback: float = 1.0) -> float:
    """读取归一化夹爪开合 [0, 1]。"""
    if gripper is None:
        return float(fallback)
    opening = gripper.read_cached_normalized_opening()
    if opening is None:
        opening = gripper.read_normalized_opening()
    if opening is None or not np.isfinite(opening):
        return float(fallback)
    return float(np.clip(opening, 0.0, 1.0))


def _read_images(
    camera_manager,
    camera_names: list[str],
    *,
    timeout_s: float = 2.0,
) -> dict[str, np.ndarray]:
    """读取各相机最新 RGB（HWC uint8）。"""
    if camera_manager is None:
        raise RuntimeError("相机未连接，无法读取图像")

    deadline = time.perf_counter() + timeout_s
    images: dict[str, np.ndarray] = {}
    while time.perf_counter() < deadline:
        images = {}
        ready = True
        for name in camera_names:
            cam = camera_manager.camera(name)
            frame = cam.get_frame()
            if frame is None or frame.rgb is None:
                ready = False
                break
            rgb = np.asarray(frame.rgb)
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            if rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise RuntimeError(f"相机 {name} RGB 形状异常: {rgb.shape}")
            images[name] = np.ascontiguousarray(rgb)
        if ready and len(images) == len(camera_names):
            return images
        time.sleep(0.01)
    raise TimeoutError(f"等待相机帧超时 ({timeout_s}s): {camera_names}")


def _read_observation(
    hw: HardwareBundle,
    cameras: list[str],
    *,
    last_state: np.ndarray | None,
) -> Observation:
    """拼一帧观测：多相机图像 + 7 维状态（6 关节度 + 夹爪）。"""
    images = _read_images(hw.camera_manager, cameras)
    fallback_joints = (
        last_state[:6].tolist()
        if last_state is not None
        else hw.start_joints_deg
    )
    joints = _read_joints_deg(hw.robot, fallback_joints)
    grip_fb = float(last_state[6]) if last_state is not None else 1.0
    gripper = _read_gripper(hw.gripper, fallback=grip_fb)
    state = np.concatenate(
        [joints.astype(np.float32), np.asarray([gripper], dtype=np.float32)]
    )
    return Observation(images=images, state=state, timestamp=time.time())


def _apply_gripper_opening(hw: HardwareBundle, opening: float) -> None:
    """更新夹爪目标开合。"""
    value = float(np.clip(opening, 0.0, 1.0))
    if hw.gripper_loop is not None:
        hw.gripper_loop.set_opening(value)
    elif hw.gripper is not None:
        ok = hw.gripper.send_normalized(value)
        if not ok:
            print("[WARN] Gloria-M send_normalized 失败")


def _wire_arm_queue(
    hw: HardwareBundle,
    action_queue: ActionQueue,
) -> None:
    """密采样回放时：推进 ActionQueue 索引 + 更新夹爪。"""
    if hw.servoj_loop is None:
        raise RuntimeError("ServoJ 后台循环未启动")

    def _advance() -> None:
        action_t = action_queue.get()
        if action_t is not None:
            arr = np.asarray(action_t.detach().cpu().numpy(), dtype=np.float64)
            joints = arr[:6].tolist()
            grip = float(arr[6]) if arr.shape[0] > 6 else float("nan")
            print(
                f"  [arm] sparse_done joints_deg={[round(v, 3) for v in joints]} "
                f"gripper={grip:.4f}",
                flush=True,
            )

    hw.servoj_loop.set_callbacks(
        on_sparse_advance=_advance,
        on_gripper=lambda g: _apply_gripper_opening(hw, g),
    )


def _enqueue_merged_actions(
    hw: HardwareBundle,
    action_queue: ActionQueue,
    *,
    fps: int,
    replace: bool,
) -> None:
    """merge 之后立刻把剩余稀疏路点送去 Z/插值/滤波密采样。"""
    if hw.servoj_loop is None:
        raise RuntimeError("ServoJ 后台循环未启动")
    leftover = action_queue.get_processed_left_over()
    if leftover is None or leftover.shape[0] == 0:
        return
    actions = np.asarray(leftover.detach().cpu().numpy(), dtype=np.float64)
    hw.servoj_loop.enqueue_waypoints(
        actions,
        action_period_s=1.0 / float(fps),
        replace=replace,
    )


def _shutdown(hw: HardwareBundle) -> None:
    """安全退出：停 ServoJ 后台、停运动、停夹爪后台、失能夹爪、停相机。"""
    if hw.servoj_loop is not None:
        try:
            hw.servoj_loop.stop()
            print("[INFO] ServoJ 后台循环已停止")
        except Exception as exc:
            print(f"[WARN] 停止 ServoJ 后台时出错: {exc}")
        hw.servoj_loop = None
    if hw.robot is not None:
        try:
            hw.robot.StopMotion()
            print("[INFO] FR3 StopMotion")
        except Exception as exc:
            print(f"[WARN] 停止 FR3 时出错: {exc}")
    if hw.gripper_loop is not None:
        try:
            hw.gripper_loop.stop()
            print("[INFO] Gloria-M 后台循环已停止")
        except Exception as exc:
            print(f"[WARN] 停止夹爪后台时出错: {exc}")
        hw.gripper_loop = None
    if hw.gripper is not None:
        try:
            hw.gripper.disable()
        except Exception:
            pass
        try:
            hw.gripper.disconnect()
        except Exception as exc:
            print(f"[WARN] 断开夹爪时出错: {exc}")
    if hw.camera_manager is not None:
        try:
            hw.camera_manager.stop()
            print("[INFO] Orbbec 已停止")
        except Exception as exc:
            print(f"[WARN] 停止相机时出错: {exc}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    deploy_path = Path(args.deploy)
    if not deploy_path.is_absolute():
        deploy_path = _resolve_path(deploy_path, base=VA_ROOT)
    deploy = _load_deploy_config(deploy_path)

    ckpt_path = (
        _resolve_path(args.checkpoint, base=VA_ROOT)
        if args.checkpoint
        else deploy["checkpoint"]
    )
    train_cfg_path = (
        _resolve_path(args.config, base=VA_ROOT)
        if args.config
        else deploy["config"]
    )
    teleop_yaml = deploy["teleop_yaml"]
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")
    if not teleop_yaml.is_file():
        raise FileNotFoundError(f"找不到 teleop.yaml: {teleop_yaml}")
    if train_cfg_path is not None and not train_cfg_path.is_file():
        raise FileNotFoundError(f"找不到训练配置: {train_cfg_path}")

    print(f"[INFO] 部署配置: {deploy_path}")
    print(f"[INFO] 加载 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if train_cfg_path is not None:
        cfg = load_config(train_cfg_path)
        print(f"[INFO] 训练配置: {train_cfg_path}")
    else:
        cfg = ckpt["config"]
        print("[INFO] 训练配置: checkpoint 内嵌 config")

    _apply_rtc_overrides(cfg, deploy.get("rtc") or {}, args)
    rtc_cfg = _normalize_rtc_config(cfg.policy.rtc)
    rtc_enabled = bool(rtc_cfg.enabled)

    stats = ckpt["stats"]
    cameras = list(cfg.cameras)
    norm_mode = cfg.dataset.norm_mode
    n_obs = int(cfg.dataset.n_obs_steps)
    n_action_steps = int(cfg.policy.n_action_steps)
    horizon = int(cfg.dataset.horizon)
    fps = int(cfg.fps)
    max_steps = deploy["max_steps"]
    policy_type = str(cfg.policy.type).lower()

    if rtc_enabled and policy_type in {"a2a", "n_a2a"} and n_action_steps != horizon:
        raise ValueError(
            "A2A RTC requires n_action_steps == horizon "
            f"(got n_action_steps={n_action_steps}, horizon={horizon})"
        )

    print(
        f"[INFO] cameras={cameras} state_dim={cfg.state_dim} action_dim={cfg.action_dim} "
        f"policy.type={cfg.policy.type}"
    )
    print(
        f"[INFO] norm_mode={norm_mode} n_obs={n_obs} horizon={horizon} "
        f"n_action_steps={n_action_steps} num_inference_steps={cfg.policy.num_inference_steps}"
    )
    print(
        f"[INFO] resize={cfg.dataset.resize_size} crop={cfg.dataset.crop_size} "
        f"eval_fixed_crop={cfg.dataset.eval_fixed_crop} fps={fps}"
    )
    print(f"[INFO] action_names={list(cfg.action_names)}")
    print(
        f"[INFO] rtc.enabled={rtc_enabled} guidance={rtc_cfg.guidance_enabled} "
        f"delay={rtc_cfg.inference_delay} exec_h={rtc_cfg.execution_horizon} "
        f"schedule={rtc_cfg.prefix_attention_schedule}"
    )
    if rtc_enabled:
        if rtc_cfg.guidance_enabled:
            print("[INFO] RTC mode: async think-while-moving + prefix guidance")
        else:
            print("[INFO] RTC mode: async think-while-moving, guidance OFF (naive discard)")

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    policy = build_policy(cfg, stats)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()

    compile_enabled = _resolve_compile_enabled(deploy, args)
    if compile_enabled:
        compiled = _compile_policy_submodules(policy)
        if compiled:
            print(f"[INFO] torch.compile({', '.join(compiled)}, mode=default)")
            crop_for_warmup = cfg.dataset.crop_size
            t_warm = time.perf_counter()
            _warmup_compiled_policy(
                policy, cfg, device, crop_size=crop_for_warmup
            )
            warm_ms = (time.perf_counter() - t_warm) * 1000.0
            print(f"[INFO] compile warmup done ({warm_ms:.0f}ms)")
        else:
            print("[WARN] --compile set but no encoder/unet/flow_net found; skipped")
    else:
        print("[INFO] torch.compile disabled (deploy.compile / --compile)")

    print(f"[INFO] 策略已就绪，设备={device} cudnn.benchmark={device.type == 'cuda'}")

    start_joints = deploy["joints_deg"]
    hw = HardwareBundle(start_joints_deg=start_joints)
    infer_worker: _RtcInferWorker | None = None

    try:
        ip = _resolve_robot_ip(deploy["robot_ip"], teleop_yaml)
        hw.robot = _connect_robot(ip)
        hw.gripper = _connect_gripper(teleop_yaml)
        # 先以当前反馈为持有目标，避免连接后到首帧动作之间 MIT 断流。
        hold_opening = _read_gripper(hw.gripper, fallback=1.0)
        hw.gripper_loop = BackgroundGripperLoop(
            hw.gripper, rate_hz=deploy["gripper_rate_hz"]
        )
        hw.gripper_loop.start(initial_opening=hold_opening)
        hw.camera_manager = _connect_cameras(teleop_yaml, cameras)
        hw.camera_names = tuple(cameras)

        joints, tool, user = _movej_start_pose(
            hw.robot,
            joints_deg=deploy["joints_deg"],
            tcp_pose=deploy["tcp_pose"],
            vel=deploy["vel"],
            tool=deploy["tool"],
            user=deploy["user"],
        )
        hw.start_joints_deg = joints
        hw.tool, hw.user = tool, user

        hw.servoj_loop = BackgroundServoJLoop(
            hw.robot,
            z_limit=deploy["z_limit"],
            max_joint_vel_deg_s=deploy["max_joint_vel_deg_s"],
            max_joint_step_deg=deploy["max_joint_step_deg"],
        )
        print(
            f"[INFO] ServoJ cmdT={SERVOJ_CMDT_S:g}s，"
            f"max_joint_vel_deg_s={deploy['max_joint_vel_deg_s']:g}，"
            f"max_joint_step_deg={deploy['max_joint_step_deg']}"
        )
        hw.servoj_loop.start(initial_joints_deg=joints)

        obs = _read_observation(hw, cameras, last_state=None)
        obs.validate(cameras, int(cfg.state_dim))
        resize_size = cfg.dataset.resize_size
        crop_size = cfg.dataset.crop_size
        eval_fixed_crop = bool(cfg.dataset.eval_fixed_crop)
        obs = _prepare_observation(
            obs,
            resize_size=resize_size,
            crop_size=crop_size,
            eval_fixed_crop=eval_fixed_crop,
        )
        obs_history: list[Observation] = [obs]
        print(
            f"[INFO] 观测入队即 resize/crop "
            f"(resize={resize_size} crop={crop_size} fixed={eval_fixed_crop})"
        )

        print(f"[INFO] 开始闭环，最多 {max_steps} 步" + (" (RTC)" if rtc_enabled else ""))

        action_queue = ActionQueue(rtc_cfg)

        def _pull_done() -> bool:
            assert hw.servoj_loop is not None
            return hw.servoj_loop.actions_executed >= max_steps

        if rtc_enabled:
            infer_worker = _RtcInferWorker(device)
            infer_worker.start()
            print("[INFO] RTC infer worker ready (persistent CUDA thread)")

            infer_job: _RtcInferJob | None = None
            cold_started = False
            arm_wired = False

            def _make_batch() -> dict[str, torch.Tensor]:
                return _build_obs_batch(
                    obs_history,
                    cameras=cameras,
                    n_obs_steps=n_obs,
                    stats=stats,
                    norm_mode=norm_mode,
                    device=device,
                )

            # Strictly inference_delay — do NOT use execution_horizon here.
            # threshold==real_delay after merge causes immediate replan forever.
            replan_threshold = int(rtc_cfg.inference_delay)
            print(
                f"[INFO] RTC replan when qsize<={replan_threshold} "
                f"(inference_delay={rtc_cfg.inference_delay}); "
                f"enqueue 时插值/滤波，ServoJ 回放密采样"
            )
            underrun_warned = False
            last_logged_exec = -1

            while not _pull_done():
                step_i = (
                    hw.servoj_loop.actions_executed if hw.servoj_loop is not None else 0
                )
                merged_this_iter = False

                # Merge completed async inference (use real consumed steps as delay).
                if infer_job is not None:
                    done, pred, err, infer_ms = infer_job.poll()
                    if done:
                        if err is not None:
                            print(f"[ERROR] RTC async inference failed: {err}")
                            infer_job = None
                        else:
                            assert pred is not None
                            real_delay = max(
                                0, action_queue.get_action_index() - infer_job.idx0
                            )
                            cfg_delay = int(rtc_cfg.inference_delay)
                            infer_ms_v = (
                                float(infer_ms)
                                if infer_ms is not None
                                else (time.perf_counter() - infer_job.started_at)
                                * 1000.0
                            )
                            if abs(real_delay - cfg_delay) >= 1:
                                print(
                                    f"[WARN] step={step_i} RTC real_delay={real_delay} "
                                    f"!= inference_delay={cfg_delay} "
                                    f"(推理耗时={infer_ms_v:.1f}ms); "
                                    f"prefix guidance uses cfg delay, merge uses real_delay"
                                )
                            print(
                                f"[INFO] step={step_i} RTC async replan done "
                                f"推理耗时={infer_ms_v:.1f}ms "
                                f"real_delay={real_delay} cfg_delay={cfg_delay} "
                                f"leftover_at_start={infer_job.leftover_len}"
                            )
                            processed = denormalize(
                                pred, stats, prefix="action", mode=norm_mode
                            )
                            action_queue.merge(
                                pred.cpu(),
                                processed.cpu(),
                                real_delay=real_delay,
                                action_index_before_inference=infer_job.idx0,
                            )
                            _enqueue_merged_actions(
                                hw, action_queue, fps=fps, replace=True
                            )
                            infer_job = None
                            underrun_warned = False
                            merged_this_iter = True

                # Cold start: one synchronous inference (no leftover to cover latency).
                if (
                    not cold_started
                    and infer_job is None
                    and action_queue.qsize() == 0
                ):
                    batch = _make_batch()
                    t_infer = time.perf_counter()
                    with torch.no_grad():
                        pred = policy.sample_actions(
                            batch,
                            prev_chunk_left_over=None,
                            inference_delay=rtc_cfg.inference_delay,
                            execution_horizon=rtc_cfg.execution_horizon,
                        )[0]
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    infer_ms = (time.perf_counter() - t_infer) * 1000.0
                    processed = denormalize(
                        pred, stats, prefix="action", mode=norm_mode
                    )
                    action_queue.merge(
                        pred.cpu(), processed.cpu(), real_delay=0
                    )
                    if not arm_wired:
                        _wire_arm_queue(hw, action_queue)
                        arm_wired = True
                    _enqueue_merged_actions(
                        hw, action_queue, fps=fps, replace=True
                    )
                    cold_started = True
                    merged_this_iter = True
                    print(
                        f"[INFO] step={step_i} RTC cold-start chunk merged "
                        f"推理耗时={infer_ms:.1f}ms qsize={action_queue.qsize()}"
                    )

                # Kick async replan early enough that leftover covers wall-clock latency.
                # Skip same iteration as merge so a late real_delay cannot immediately
                # re-arm when post-merge qsize happens to equal the threshold.
                if (
                    cold_started
                    and infer_job is None
                    and not merged_this_iter
                    and action_queue.qsize() <= replan_threshold
                ):
                    leftover = action_queue.get_left_over()
                    if leftover is not None and leftover.shape[0] == 0:
                        leftover = None
                    leftover_len = 0 if leftover is None else int(leftover.shape[0])
                    idx0 = action_queue.get_action_index()
                    # Snapshot obs so control loop can keep appending while we prep batch.
                    obs_snap = [
                        _clone_observation(o) for o in obs_history[-n_obs:]
                    ]
                    if len(obs_snap) == 0:
                        obs_snap = [_clone_observation(obs)]

                    def _batch_factory(
                        history: list[Observation] = obs_snap,
                    ) -> dict[str, torch.Tensor]:
                        return _build_obs_batch(
                            history,
                            cameras=cameras,
                            n_obs_steps=n_obs,
                            stats=stats,
                            norm_mode=norm_mode,
                            device=device,
                        )

                    print(
                        f"[INFO] step={step_i} RTC async replan start "
                        f"leftover={leftover_len} qsize={action_queue.qsize()} "
                        f"idx0={idx0} threshold={replan_threshold}"
                    )
                    assert infer_worker is not None
                    infer_job = infer_worker.submit(
                        policy=policy,
                        batch_factory=_batch_factory,
                        leftover=leftover,
                        inference_delay=int(rtc_cfg.inference_delay),
                        execution_horizon=int(rtc_cfg.execution_horizon),
                        idx0=idx0,
                        leftover_len=leftover_len,
                    )

                dense_left = (
                    hw.servoj_loop.dense_qsize if hw.servoj_loop is not None else 0
                )
                if (
                    cold_started
                    and dense_left == 0
                    and action_queue.qsize() == 0
                    and infer_job is not None
                    and not underrun_warned
                ):
                    waited_ms = (time.perf_counter() - infer_job.started_at) * 1000.0
                    print(
                        f"[WARN] step={step_i} dense/action underrun while "
                        f"inferring (~{waited_ms:.0f}ms so far); "
                        f"arm holding last ServoJ target (stutter). "
                        f"Increase rtc.execution_horizon / inference_delay "
                        f"or lower num_inference_steps.",
                        flush=True,
                    )
                    underrun_warned = True

                if (
                    cold_started
                    and dense_left == 0
                    and action_queue.qsize() == 0
                    and infer_job is None
                    and arm_wired
                ):
                    print(
                        "[WARN] action queue empty and no inference in flight; stopping"
                    )
                    break

                t0 = time.perf_counter()
                if step_i != last_logged_exec:
                    print(
                        f"[INFO] obs_loop exec_actions={step_i} "
                        f"qsize={action_queue.qsize()} dense={dense_left}",
                        flush=True,
                    )
                    last_logged_exec = step_i
                obs = _prepare_observation(
                    _read_observation(hw, cameras, last_state=obs.state),
                    resize_size=resize_size,
                    crop_size=crop_size,
                    eval_fixed_crop=eval_fixed_crop,
                )
                _append_observation(obs_history, obs, n_obs_steps=n_obs)
                _pace_step(t0, fps)
        else:
            # 非 RTC：merge 后立刻 enqueue 密采样；队列空再 replan。
            arm_wired = False
            last_logged_exec = -1

            while not _pull_done():
                step_i = (
                    hw.servoj_loop.actions_executed if hw.servoj_loop is not None else 0
                )
                if action_queue.qsize() == 0:
                    batch = _build_obs_batch(
                        obs_history,
                        cameras=cameras,
                        n_obs_steps=n_obs,
                        stats=stats,
                        norm_mode=norm_mode,
                        device=device,
                    )
                    t_infer = time.perf_counter()
                    with torch.no_grad():
                        pred = policy.sample_actions(batch)[0].cpu()
                    infer_ms = (time.perf_counter() - t_infer) * 1000.0
                    pred_phys = denormalize(
                        pred, stats, prefix="action", mode=norm_mode
                    )
                    chunk = pred_phys[:n_action_steps].cpu()
                    action_queue.merge(pred[:n_action_steps].cpu(), chunk, real_delay=0)
                    if not arm_wired:
                        _wire_arm_queue(hw, action_queue)
                        arm_wired = True
                    _enqueue_merged_actions(
                        hw, action_queue, fps=fps, replace=True
                    )
                    print(
                        f"[INFO] step={step_i} 重新规划 chunk={chunk.shape[0]} "
                        f"推理耗时={infer_ms:.1f}ms"
                    )

                dense_left = (
                    hw.servoj_loop.dense_qsize if hw.servoj_loop is not None else 0
                )
                t0 = time.perf_counter()
                if step_i != last_logged_exec:
                    print(
                        f"[INFO] obs_loop exec_actions={step_i} "
                        f"qsize={action_queue.qsize()} dense={dense_left}",
                        flush=True,
                    )
                    last_logged_exec = step_i
                obs = _prepare_observation(
                    _read_observation(hw, cameras, last_state=obs.state),
                    resize_size=resize_size,
                    crop_size=crop_size,
                    eval_fixed_crop=eval_fixed_crop,
                )
                _append_observation(obs_history, obs, n_obs_steps=n_obs)
                _pace_step(t0, fps)

        print("[INFO] 达到 max-steps，正常结束")
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        if infer_worker is not None:
            infer_worker.shutdown()
            infer_worker = None
        _shutdown(hw)


if __name__ == "__main__":
    main()
