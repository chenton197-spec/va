#!/usr/bin/env python3
"""FR3 + Gloria-M + Orbbec 上的策略闭环部署。

训练相关配置优先来自训练 YAML（``deploy.yaml`` 的 ``config`` 或
``--config``），否则回退到 checkpoint 内嵌 config（相机、维度、
norm_mode、resize/crop、n_obs_steps、n_action_steps、fps 等）。
权重与 stats 仍从 checkpoint 加载。机械臂走与
``ct/scripts/move_to_start_pose.py`` 相同的 FAIRINO SDK：

  /home/casbot/teleop_project/fairino390/linux/fairino/Robot.py

  - 启动时 MoveJ 到 deploy.yaml 中的 start_pose
  - 闭环用 MoveJ 下发关节角（度，速度默认 20）
  - 夹爪目标写入后台循环，按 ``gripper_rate_hz``（默认 125）连续
    ``send_normalized``（MIT 不能只偶发一帧）

用法（在 ``ct/va`` 下，conda 环境 ``lerobot``）::

    # 部署 + 训练配置见同目录 deploy.yaml
    PYTHONPATH=. python run_robot/run_policy.py

    # CLI 覆盖训练配置 / checkpoint
    PYTHONPATH=. python run_robot/run_policy.py \\
        --config configs/shine_shoes_a2a_noise_limits.yaml \\
        --checkpoint model/a2a_noise_shine_shoes_limits_260730175409/checkpoint_090000.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import threading
import time
from dataclasses import dataclass
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

from robotfm.config import load_config  # noqa: E402
from robotfm.data.dataset import spatial_preprocess_images  # noqa: E402
from robotfm.data.stats import denormalize, normalize  # noqa: E402
from robotfm.train import build_policy  # noqa: E402
from robotfm.types import Observation  # noqa: E402


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
    }


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
    pre_crop_size: int | None,
    resize_size: int | None,
    crop_size: int | None,
    eval_fixed_crop: bool,
) -> dict[str, np.ndarray]:
    """相机原图 HWC uint8 → 策略分辨率 HWC float32 [0,1]。

    顺序：中心 pre_crop → resize → 可选中心 crop。入队时做一次。
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
    """替换图像为入队即用的 pre_crop/resize/crop 结果；state 原样保留。"""
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


# ---------------------------------------------------------------------------
# 硬件封装
# ---------------------------------------------------------------------------


@dataclass
class HardwareBundle:
    """真机资源句柄：机械臂、夹爪、相机与 MoveJ 相关状态。"""

    robot: Any | None = None
    gripper: Any | None = None
    gripper_loop: "BackgroundGripperLoop | None" = None
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


def _forward_kin_desc_pos(robot, joints_deg: list[float]) -> list[float] | None:
    """用 GetForwardKin 由关节角求笛卡尔位姿 [x,y,z,rx,ry,rz]，失败返回 None。"""
    ret = robot.GetForwardKin(list(map(float, joints_deg)))
    if not isinstance(ret, (list, tuple)) or ret[0] != 0 or ret[1] is None:
        print(f"[WARN] GetForwardKin 失败: {ret}")
        return None
    return [float(v) for v in ret[1]]


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

    desc_pos = tcp_pose if tcp_pose is not None else _forward_kin_desc_pos(robot, joints_deg)
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


def _send_action(
    hw: HardwareBundle,
    action: np.ndarray,
    *,
    vel: float = 20.0,
    z_limit: float,
) -> None:
    """打印并下发一步动作：先更新夹爪目标，再阻塞 MoveJ。

    夹爪由后台高频循环持续 ``send_normalized``；本函数只改目标开合。
    正运动学得到的笛卡尔 Z 若小于 ``z_limit``（mm），则跳过 MoveJ。
    """
    joints = action[:6].astype(float).tolist()
    gripper = float(np.clip(action[6], 0.0, 1.0))
    print(
        f"  joints_deg={[round(v, 3) for v in joints]}  gripper={gripper:.4f}",
        flush=True,
    )
    if hw.robot is None:
        raise RuntimeError("未连接机器人，无法下发 MoveJ")

    # 先写目标，MoveJ 阻塞期间后台仍持续刷 MIT 帧。
    if hw.gripper_loop is not None:
        hw.gripper_loop.set_opening(gripper)
    elif hw.gripper is not None:
        ok = hw.gripper.send_normalized(gripper)
        if not ok:
            print("[WARN] Gloria-M send_normalized 失败")

    desc_pos = _forward_kin_desc_pos(hw.robot, joints)
    if desc_pos is not None:
        print(
            f"  desc_pos={[round(v, 3) for v in desc_pos]}",
            flush=True,
        )
        if desc_pos[2] < z_limit:
            print(
                f"[WARN] Z={desc_pos[2]:.3f} < z_limit={z_limit}，跳过 MoveJ",
                flush=True,
            )
        else:
            # blendT=-1.0：阻塞直到运动到位
            ret = hw.robot.MoveJ(
                joint_pos=joints,
                tool=hw.tool,
                user=hw.user,
                desc_pos=desc_pos,
                vel=float(vel),
                blendT=-1.0,
            )
            if ret != 0:
                print(f"[WARN] MoveJ 错误码: {ret}")
    else:
        print("[WARN] GetForwardKin 失败，跳过 MoveJ（无法校验 Z）", flush=True)


def _shutdown(hw: HardwareBundle) -> None:
    """安全退出：停运动、停夹爪后台、失能夹爪、停相机。"""
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
    stats = ckpt["stats"]
    cameras = list(cfg.cameras)
    norm_mode = cfg.dataset.norm_mode
    n_obs = int(cfg.dataset.n_obs_steps)
    n_action_steps = int(cfg.policy.n_action_steps)
    fps = int(cfg.fps)
    max_steps = deploy["max_steps"]

    print(
        f"[INFO] cameras={cameras} state_dim={cfg.state_dim} action_dim={cfg.action_dim} "
        f"policy.type={cfg.policy.type}"
    )
    print(
        f"[INFO] norm_mode={norm_mode} n_obs={n_obs} horizon={cfg.dataset.horizon} "
        f"n_action_steps={n_action_steps} num_inference_steps={cfg.policy.num_inference_steps}"
    )
    print(
        f"[INFO] pre_crop={cfg.dataset.pre_crop_size} resize={cfg.dataset.resize_size} "
        f"crop={cfg.dataset.crop_size} eval_fixed_crop={cfg.dataset.eval_fixed_crop} fps={fps}"
    )
    print(f"[INFO] action_names={list(cfg.action_names)}")

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    policy = build_policy(cfg, stats)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()
    print(f"[INFO] 策略已就绪，设备={device}")

    start_joints = deploy["joints_deg"]
    hw = HardwareBundle(start_joints_deg=start_joints)

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

        obs = _read_observation(hw, cameras, last_state=None)
        obs.validate(cameras, int(cfg.state_dim))
        pre_crop_size = cfg.dataset.pre_crop_size
        resize_size = cfg.dataset.resize_size
        crop_size = cfg.dataset.crop_size
        eval_fixed_crop = bool(cfg.dataset.eval_fixed_crop)
        obs = _prepare_observation(
            obs,
            pre_crop_size=pre_crop_size,
            resize_size=resize_size,
            crop_size=crop_size,
            eval_fixed_crop=eval_fixed_crop,
        )
        obs_history: list[Observation] = [obs]
        print(
            f"[INFO] 观测入队即 pre_crop/resize/crop "
            f"(pre_crop={pre_crop_size} resize={resize_size} crop={crop_size} "
            f"fixed={eval_fixed_crop})"
        )

        # action chunking：执行 n_action_steps 步后再 replan（与 eval.py 一致）
        chunk_actions: list[np.ndarray] = []
        chunk_idx = 0
        print(f"[INFO] 开始闭环，最多 {max_steps} 步")

        for step_i in range(max_steps):
            t0 = time.perf_counter()
            if chunk_idx >= len(chunk_actions):
                batch = _build_obs_batch(
                    obs_history,
                    cameras=cameras,
                    n_obs_steps=n_obs,
                    stats=stats,
                    norm_mode=norm_mode,
                    device=device,
                )
                with torch.no_grad():
                    pred = policy.sample_actions(batch)[0].cpu()
                pred_phys = denormalize(
                    pred, stats, prefix="action", mode=norm_mode
                ).numpy()
                chunk_actions = [
                    np.asarray(a, dtype=np.float32)
                    for a in pred_phys[:n_action_steps]
                ]
                chunk_idx = 0
                print(f"[INFO] step={step_i} 重新规划 chunk={len(chunk_actions)}")

            action = chunk_actions[chunk_idx]
            chunk_idx += 1
            print(f"[INFO] step={step_i}", end="")
            _send_action(hw, action, vel=20.0, z_limit=deploy["z_limit"])

            obs = _prepare_observation(
                _read_observation(hw, cameras, last_state=obs.state),
                pre_crop_size=pre_crop_size,
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
        _shutdown(hw)


if __name__ == "__main__":
    main()
