"""FAIRINO FR3 关节空间 ServoJ 控制。

在路点 ``enqueue``（写入 ActionQueue 的同时）完成：
  Z 校验 → 相邻点五次插值（名义 1/fps，超速拉长）→ 一阶低通，
并生成 ``cmdT=8ms`` 密采样指令缓冲。

ServoJ 后台只回放密采样，不再在取队列时做插值。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

SERVOJ_CMDT_S = 0.008
DEFAULT_MAX_JOINT_VEL_DEG_S = 150.0
DEFAULT_FILTER_TAU_S = 0.02
# 相邻稀疏路点允许的最大单轴跳变（度）；None 表示用 vel*period
DEFAULT_MAX_JOINT_STEP_DEG = 15.0
_N_JOINTS = 6

SparseAdvanceCallback = Callable[[], None]
GripperCallback = Callable[[float], None]


def forward_kin_desc_pos(robot: Any, joints_deg: list[float]) -> list[float] | None:
    """用 GetForwardKin 由关节角求笛卡尔位姿 [x,y,z,rx,ry,rz]，失败返回 None。"""
    ret = robot.GetForwardKin(list(map(float, joints_deg)))
    if not isinstance(ret, (list, tuple)) or ret[0] != 0 or ret[1] is None:
        print(f"[WARN] GetForwardKin 失败: {ret}")
        return None
    return [float(v) for v in ret[1]]


def _quintic_pos(
    s: float,
    q0: np.ndarray,
    v0: np.ndarray,
    a0: np.ndarray,
    q1: np.ndarray,
    v1: np.ndarray,
    a1: np.ndarray,
    T: float,
) -> np.ndarray:
    """单位区间 s∈[0,1] 上的五次 Hermite 位置。"""
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    s5 = s4 * s
    h00 = 1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5
    h10 = s - 6.0 * s3 + 8.0 * s4 - 3.0 * s5
    h20 = 0.5 * s2 - 1.5 * s3 + 1.5 * s4 - 0.5 * s5
    h01 = 10.0 * s3 - 15.0 * s4 + 6.0 * s5
    h11 = -4.0 * s3 + 7.0 * s4 - 3.0 * s5
    h21 = 0.5 * s3 - s4 + 0.5 * s5
    return (
        h00 * q0
        + h10 * (T * v0)
        + h20 * (T * T * a0)
        + h01 * q1
        + h11 * (T * v1)
        + h21 * (T * T * a1)
    )


def densify_waypoints(
    waypoints: np.ndarray,
    *,
    q0: np.ndarray,
    action_period_s: float,
    max_joint_vel_deg_s: float,
    filter_tau_s: float,
    robot: Any | None,
    z_limit: float,
    max_joint_step_deg: float | None = None,
    dt_s: float = SERVOJ_CMDT_S,
) -> list[tuple[list[float], float, bool]]:
    """对稀疏路点做跳变门控 + Z 校验 + 五次插值 + 低通，返回密采样。

    相对上一**接受**路点，若 ``max|Δq_i| > max_joint_step_deg`` 则丢弃该点
    （保持姿态，仍打 ``sparse_boundary`` 以推进 ActionQueue）。
    ``max_joint_step_deg is None`` 时默认 ``max_joint_vel_deg_s * action_period_s``。

    每个元素 ``(joints_deg[6], gripper, sparse_boundary)``。
    """
    if action_period_s <= 0.0:
        raise ValueError("action_period_s 必须 > 0")
    if max_joint_vel_deg_s <= 0.0:
        raise ValueError("max_joint_vel_deg_s 必须 > 0")
    if dt_s <= 0.0:
        raise ValueError("dt_s 必须 > 0")

    step_limit = (
        float(max_joint_vel_deg_s) * float(action_period_s)
        if max_joint_step_deg is None
        else float(max_joint_step_deg)
    )
    if step_limit <= 0.0:
        raise ValueError("max_joint_step_deg 必须 > 0")

    wps = np.asarray(waypoints, dtype=np.float64)
    if wps.ndim == 1:
        wps = wps.reshape(1, -1)
    if wps.ndim != 2 or wps.shape[1] < _N_JOINTS:
        raise ValueError(f"waypoints 形状非法: {wps.shape}")

    q_cursor = np.asarray(q0, dtype=np.float64).reshape(_N_JOINTS).copy()
    q_filt = q_cursor.copy()
    zero = np.zeros(_N_JOINTS, dtype=np.float64)
    out: list[tuple[list[float], float, bool]] = []

    for row in wps:
        q1 = row[:_N_JOINTS].copy()
        grip = float(np.clip(row[6], 0.0, 1.0)) if row.shape[0] > 6 else 1.0

        dq = q1 - q_cursor
        max_abs = float(np.max(np.abs(dq)))
        if max_abs > step_limit:
            print(
                f"[WARN] 路点跳变 max|Δq|={max_abs:.3f}° > "
                f"max_joint_step_deg={step_limit:.3f}°，丢弃该点",
                flush=True,
            )
            out.append((q_cursor.tolist(), grip, True))
            continue

        if robot is not None:
            try:
                desc_pos = forward_kin_desc_pos(robot, q1.tolist())
            except Exception as exc:
                print(f"[WARN] enqueue GetForwardKin 异常，跳过路点: {exc}", flush=True)
                out.append((q_cursor.tolist(), grip, True))
                continue
            if desc_pos is None:
                print("[WARN] enqueue GetForwardKin 失败，跳过路点", flush=True)
                out.append((q_cursor.tolist(), grip, True))
                continue
            print(f"  desc_pos={[round(v, 3) for v in desc_pos]}", flush=True)
            if desc_pos[2] < z_limit:
                print(
                    f"[WARN] Z={desc_pos[2]:.3f} < z_limit={z_limit}，跳过路点",
                    flush=True,
                )
                out.append((q_cursor.tolist(), grip, True))
                continue

        if max_abs < 1e-9:
            out.append((q_cursor.tolist(), grip, True))
            continue

        T = max(dt_s, float(action_period_s), max_abs / float(max_joint_vel_deg_s))
        n = max(1, int(round(T / dt_s)))
        T = float(n) * dt_s
        q0_seg = q_cursor.copy()

        for i in range(1, n + 1):
            s = float(i) / float(n)
            q_des = _quintic_pos(s, q0_seg, zero, zero, q1, zero, zero, T)
            if filter_tau_s <= 0.0:
                q_filt = q_des
            else:
                alpha = dt_s / (filter_tau_s + dt_s)
                q_filt = q_filt + alpha * (q_des - q_filt)
            boundary = i == n
            out.append((q_filt.tolist(), grip, boundary))

        q_cursor = q1.copy()
        q_filt = q1.copy()

    return out


class BackgroundServoJLoop:
    """后台 ``ServoJ``：回放 enqueue 时已插值/滤波好的密采样。"""

    _AXIS_POS = [0.0, 0.0, 0.0, 0.0]

    def __init__(
        self,
        robot: Any,
        *,
        z_limit: float,
        max_joint_vel_deg_s: float = DEFAULT_MAX_JOINT_VEL_DEG_S,
        filter_tau_s: float = DEFAULT_FILTER_TAU_S,
        max_joint_step_deg: float | None = DEFAULT_MAX_JOINT_STEP_DEG,
    ) -> None:
        self._robot = robot
        self._cmdt_s = SERVOJ_CMDT_S
        self._z_limit = float(z_limit)
        self._max_vel = float(max_joint_vel_deg_s)
        self._tau = float(filter_tau_s)
        self._max_step = (
            None if max_joint_step_deg is None else float(max_joint_step_deg)
        )
        if self._max_vel <= 0.0:
            raise ValueError("max_joint_vel_deg_s 必须 > 0")
        if self._tau < 0.0:
            raise ValueError("filter_tau_s 必须 >= 0")
        if self._max_step is not None and self._max_step <= 0.0:
            raise ValueError("max_joint_step_deg 必须 > 0")

        self._lock = threading.Lock()
        self._cmd_q: deque[tuple[list[float], float, bool]] = deque()
        self._q_cmd: list[float] | None = None
        self._ingest_req: tuple[np.ndarray, float, bool] | None = None
        self._on_sparse_advance: SparseAdvanceCallback | None = None
        self._on_gripper: GripperCallback | None = None
        self._actions_executed = 0
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._servo_started = False
        self._warn_every = max(1, int(round(1.0 / self._cmdt_s)))
        self._fail_streak = 0

    @property
    def cmdt_s(self) -> float:
        return self._cmdt_s

    @property
    def max_joint_vel_deg_s(self) -> float:
        return self._max_vel

    @property
    def actions_executed(self) -> int:
        with self._lock:
            return int(self._actions_executed)

    @property
    def dense_qsize(self) -> int:
        with self._lock:
            return len(self._cmd_q)

    def set_callbacks(
        self,
        *,
        on_sparse_advance: SparseAdvanceCallback | None = None,
        on_gripper: GripperCallback | None = None,
    ) -> None:
        with self._lock:
            self._on_sparse_advance = on_sparse_advance
            self._on_gripper = on_gripper

    def set_joints_deg(
        self,
        joints_deg: list[float] | np.ndarray,
        *,
        check_z: bool = True,
    ) -> None:
        """启动对齐用：直接设当前命令（不经密采样队列）。"""
        values = [float(v) for v in joints_deg]
        if len(values) != _N_JOINTS:
            raise ValueError(f"ServoJ 目标必须是 {_N_JOINTS} 轴，得到 {len(values)}")
        if check_z:
            desc = forward_kin_desc_pos(self._robot, values)
            if desc is None or desc[2] < self._z_limit:
                raise RuntimeError(f"初始关节 Z 校验失败: {desc}")
        with self._lock:
            self._q_cmd = values
            self._cmd_q.clear()
        self._wake.set()

    def get_joints_deg(self) -> list[float] | None:
        with self._lock:
            return None if self._q_cmd is None else list(self._q_cmd)

    def enqueue_waypoints(
        self,
        actions: np.ndarray | list,
        *,
        action_period_s: float,
        replace: bool = True,
    ) -> None:
        """在入队时刻请求：Z 校验 + 插值 + 滤波，写入密采样缓冲。

        实际规划在 ServoJ 线程执行，避免与 ``ServoJ`` RPC 并发。
        ``replace=True``（RTC）：丢掉未播放密采样，从当前关节重规划。
        ``replace=False``（非 RTC append）：接在缓冲末尾继续规划。
        """
        arr = np.asarray(actions, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            return
        with self._lock:
            self._ingest_req = (arr.copy(), float(action_period_s), bool(replace))
        self._wake.set()

    def start(self, initial_joints_deg: list[float] | np.ndarray | None = None) -> None:
        if self._thread is not None:
            raise RuntimeError("ServoJ 后台循环已在运行")
        if initial_joints_deg is not None:
            self.set_joints_deg(initial_joints_deg, check_z=False)
        if self.get_joints_deg() is None:
            raise RuntimeError("启动 ServoJ 前必须提供初始目标关节角")

        ret = self._robot.ServoMoveStart()
        print(f"[INFO] ServoMoveStart -> {ret}")
        if ret != 0:
            raise RuntimeError(f"ServoMoveStart 失败 error={ret}")
        self._servo_started = True

        self._stop.clear()
        self._wake.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="fairino-servoj-loop",
            daemon=True,
        )
        self._thread.start()
        print(
            f"[INFO] ServoJ 后台循环已启动: cmdT={self._cmdt_s:g}s，"
            f"max_joint_vel={self._max_vel:g} deg/s，"
            f"max_joint_step={self._max_step} deg，"
            f"z_limit={self._z_limit:g}，"
            f"初始目标={[round(v, 3) for v in self.get_joints_deg() or []]}"
        )

    def stop(self, *, join_timeout_s: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout_s)
        self._thread = None
        if self._servo_started:
            try:
                ret = self._robot.ServoMoveEnd()
                print(f"[INFO] ServoMoveEnd -> {ret}")
            except Exception as exc:
                print(f"[WARN] ServoMoveEnd 异常: {exc}")
            self._servo_started = False

    def _process_ingest(self) -> None:
        with self._lock:
            req = self._ingest_req
            self._ingest_req = None
            if req is None:
                return
            actions, period, replace = req
            q0 = (
                np.asarray(self._q_cmd, dtype=np.float64)
                if self._q_cmd is not None
                else actions[0, :_N_JOINTS].copy()
            )
            if not replace and self._cmd_q:
                # 从缓冲里最后一个密采样终点接着规划。
                q0 = np.asarray(self._cmd_q[-1][0], dtype=np.float64)

        samples = densify_waypoints(
            actions,
            q0=q0,
            action_period_s=period,
            max_joint_vel_deg_s=self._max_vel,
            filter_tau_s=self._tau,
            robot=self._robot,
            z_limit=self._z_limit,
            max_joint_step_deg=self._max_step,
            dt_s=self._cmdt_s,
        )
        with self._lock:
            if replace:
                self._cmd_q.clear()
            self._cmd_q.extend(samples)
            n_sparse = int(sum(1 for *_, b in samples if b))
        print(
            f"[INFO] enqueue 密采样: sparse≈{actions.shape[0]} → dense={len(samples)} "
            f"(boundaries={n_sparse}) replace={replace}",
            flush=True,
        )

    def _pop_dense(self) -> tuple[list[float], float, bool] | None:
        with self._lock:
            if not self._cmd_q:
                return None
            return self._cmd_q.popleft()

    def _run(self) -> None:
        next_servoj_t = time.perf_counter()
        while not self._stop.is_set():
            self._process_ingest()

            now = time.perf_counter()
            if now >= next_servoj_t:
                item = self._pop_dense()
                if item is not None:
                    joints, grip, boundary = item
                    with self._lock:
                        self._q_cmd = list(joints)
                        on_grip = self._on_gripper
                        on_adv = self._on_sparse_advance
                    if on_grip is not None:
                        try:
                            on_grip(grip)
                        except Exception as exc:
                            print(f"[WARN] gripper 回调异常: {exc}")
                    try:
                        ret = self._robot.ServoJ(
                            joint_pos=joints,
                            axisPos=self._AXIS_POS,
                            cmdT=self._cmdt_s,
                        )
                    except Exception as exc:
                        ret = -1
                        self._fail_streak += 1
                        if (
                            self._fail_streak == 1
                            or self._fail_streak % self._warn_every == 0
                        ):
                            print(f"[WARN] ServoJ 后台异常: {exc}")
                    else:
                        if ret == 0:
                            self._fail_streak = 0
                        else:
                            self._fail_streak += 1
                            if (
                                self._fail_streak == 1
                                or self._fail_streak % self._warn_every == 0
                            ):
                                print(f"[WARN] ServoJ 错误码: {ret}")
                    if boundary:
                        with self._lock:
                            self._actions_executed += 1
                        if on_adv is not None:
                            try:
                                on_adv()
                            except Exception as exc:
                                print(f"[WARN] sparse advance 回调异常: {exc}")
                else:
                    # 缓冲空：保持上一目标继续刷，避免 Servo 失联。
                    with self._lock:
                        joints = None if self._q_cmd is None else list(self._q_cmd)
                    if joints is not None:
                        try:
                            self._robot.ServoJ(
                                joint_pos=joints,
                                axisPos=self._AXIS_POS,
                                cmdT=self._cmdt_s,
                            )
                        except Exception:
                            pass

                next_servoj_t = now + self._cmdt_s

            if self._stop.is_set():
                break
            wait_s = max(0.0, next_servoj_t - time.perf_counter())
            self._wake.clear()
            with self._lock:
                has_ingest = self._ingest_req is not None
            if has_ingest:
                continue
            if self._stop.is_set():
                break
            self._wake.wait(timeout=wait_s)
