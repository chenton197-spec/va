#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

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

from replay import HardwareBundle, _connect_grippers, _connect_hcx_arms, _shutdown  # noqa: E402


def _read_gripper(gripper, fallback: float = 1.0) -> float:
    if gripper is None:
        return float(fallback)
    opening = gripper.read_cached_normalized_opening()
    if opening is None:
        opening = gripper.read_normalized_opening()
    if opening is None or not np.isfinite(opening):
        return float(fallback)
    return float(np.clip(opening, 0.0, 1.0))


def _fmt_joints(vals: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):.3f}" for v in vals) + "]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(DEFAULT_POSE))
    parser.add_argument("--teleop", type=str, default=str(TELEOP_ROOT / "teleop.yaml"))
    args = parser.parse_args()

    out_path = Path(args.out).expanduser().resolve()
    teleop_yaml = Path(args.teleop).expanduser().resolve()

    hw = HardwareBundle()
    try:
        hw.hcx_client, hw.left_arm, hw.right_arm = _connect_hcx_arms(teleop_yaml)
        hw.left_gripper, hw.right_gripper, _ = _connect_grippers(teleop_yaml)
        left = np.clip(
            np.asarray(hw.left_arm.joint_angles(), dtype=np.float64),
            JOINT_LO,
            JOINT_HI,
        )
        right = np.clip(
            np.asarray(hw.right_arm.joint_angles(), dtype=np.float64),
            JOINT_LO,
            JOINT_HI,
        )
        if left.shape != (7,) or right.shape != (7,) or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise RuntimeError("关节反馈必须是左右各 7 个有效角度")
        left_g = _read_gripper(hw.left_gripper)
        right_g = _read_gripper(hw.right_gripper)
        out_path.write_text(
            (
                f"left_joints_deg: {_fmt_joints(left)}\n"
                f"right_joints_deg: {_fmt_joints(right)}\n"
                f"left_gripper: {left_g:.4f}\n"
                f"right_gripper: {right_g:.4f}\n"
                f"gripper_ramp_s: 0.9\n"
            ),
            encoding="utf-8",
        )
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        _shutdown(hw, None)


if __name__ == "__main__":
    raise SystemExit(main())
