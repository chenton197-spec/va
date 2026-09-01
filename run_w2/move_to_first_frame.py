#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
VA_ROOT = SCRIPT_DIR.parent
TELEOP_ROOT = VA_ROOT.parent / "teleop_project"
DEFAULT_POSE = SCRIPT_DIR / "first_frame_pose.yaml"
JOINT_LO = np.array(
    [-169.0, -100.9, -169.9, -139.9, -169.0, -54.9, -59.9], dtype=np.float64
)
JOINT_HI = np.array(
    [169.0, 100.9, 169.9, 54.9, 169.9, 54.9, 59.9], dtype=np.float64
)

for p in (SCRIPT_DIR, TELEOP_ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from replay import (  # noqa: E402
    BackgroundGripperLoop,
    HardwareBundle,
    _confirm_targets_by_feedback,
    _connect_grippers,
    _connect_hcx_arms,
    _shutdown,
)


def _read_gripper(gripper, fallback: float = 1.0) -> float:
    if gripper is None:
        return float(fallback)
    opening = gripper.read_cached_normalized_opening()
    if opening is None:
        opening = gripper.read_normalized_opening()
    if opening is None or not np.isfinite(opening):
        return float(fallback)
    return float(np.clip(opening, 0.0, 1.0))


def _load_pose(path: Path) -> tuple[list[float], list[float], float, float, float]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到位姿 YAML: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"位姿 YAML 根节点必须是映射: {path}")
    left = np.asarray(data.get("left_joints_deg"), dtype=np.float64)
    right = np.asarray(data.get("right_joints_deg"), dtype=np.float64)
    if left.shape != (7,) or right.shape != (7,) or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("left_joints_deg / right_joints_deg 必须是 7 个有效角度")
    left_g = float(data.get("left_gripper", 1.0))
    right_g = float(data.get("right_gripper", 1.0))
    ramp_s = float(data.get("gripper_ramp_s", 0.9))
    if not np.isfinite(left_g) or not np.isfinite(right_g):
        raise ValueError("left_gripper / right_gripper 必须是有限数")
    if not math.isfinite(ramp_s) or ramp_s <= 0.0:
        raise ValueError("gripper_ramp_s 必须是正的有限秒数")
    left = np.clip(left, JOINT_LO, JOINT_HI).astype(float).tolist()
    right = np.clip(right, JOINT_LO, JOINT_HI).astype(float).tolist()
    return left, right, float(np.clip(left_g, 0.0, 1.0)), float(np.clip(right_g, 0.0, 1.0)), ramp_s


def _ramp_grippers(
    hw: HardwareBundle,
    *,
    left_target: float,
    right_target: float,
    duration_s: float,
    rate_hz: float,
) -> None:
    left_from = _read_gripper(hw.left_gripper, fallback=1.0)
    right_from = _read_gripper(hw.right_gripper, fallback=1.0)
    n_steps = max(1, int(math.ceil(duration_s * rate_hz)))
    period_s = duration_s / float(n_steps)
    for step in range(1, n_steps + 1):
        alpha = float(step) / float(n_steps)
        if hw.left_gripper_loop is not None:
            hw.left_gripper_loop.set_opening(left_from + alpha * (left_target - left_from))
        elif hw.left_gripper is not None:
            _ = hw.left_gripper.send_normalized(left_from + alpha * (left_target - left_from))
        if hw.right_gripper_loop is not None:
            hw.right_gripper_loop.set_opening(right_from + alpha * (right_target - right_from))
        elif hw.right_gripper is not None:
            _ = hw.right_gripper.send_normalized(right_from + alpha * (right_target - right_from))
        time.sleep(period_s)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", type=str, default=str(DEFAULT_POSE))
    parser.add_argument("--teleop", type=str, default=str(TELEOP_ROOT / "teleop.yaml"))
    args = parser.parse_args()

    pose_path = Path(args.pose).expanduser().resolve()
    teleop_yaml = Path(args.teleop).expanduser().resolve()
    left, right, left_g, right_g, ramp_s = _load_pose(pose_path)

    hw = HardwareBundle()
    try:
        hw.hcx_client, hw.left_arm, hw.right_arm = _connect_hcx_arms(teleop_yaml)
        hw.left_gripper, hw.right_gripper, gripper_rate_hz = _connect_grippers(teleop_yaml)
        if hw.left_gripper is not None:
            hw.left_gripper_loop = BackgroundGripperLoop(
                hw.left_gripper, rate_hz=gripper_rate_hz
            )
            hw.left_gripper_loop.start(
                initial_opening=_read_gripper(hw.left_gripper, fallback=1.0)
            )
        if hw.right_gripper is not None:
            hw.right_gripper_loop = BackgroundGripperLoop(
                hw.right_gripper, rate_hz=gripper_rate_hz
            )
            hw.right_gripper_loop.start(
                initial_opening=_read_gripper(hw.right_gripper, fallback=1.0)
            )
        hw.left_arm.move_joints(
            left,
            interrupt=False,
            wait=False,
            speed_ratio=0.1,
            acceleration_seconds=0.1,
            deceleration_seconds=0.1,
        )
        hw.right_arm.move_joints(
            right,
            interrupt=False,
            wait=False,
            speed_ratio=0.1,
            acceleration_seconds=0.1,
            deceleration_seconds=0.1,
        )
        _confirm_targets_by_feedback(
            hw,
            left,
            right,
            timeout_s=30.0,
            poll_interval_s=0.05,
            angle_tolerance_deg=0.5,
        )
        _ramp_grippers(
            hw,
            left_target=left_g,
            right_target=right_g,
            duration_s=ramp_s,
            rate_hz=gripper_rate_hz,
        )
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        _shutdown(hw, None)


if __name__ == "__main__":
    raise SystemExit(main())
