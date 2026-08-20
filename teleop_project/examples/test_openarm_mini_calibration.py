#!/usr/bin/env python3
"""独立标定双 OpenArm Mini 主臂并创建/更新组合标定 JSON。

串口、波特率和输出 JSON 路径由 ``teleop.yaml`` 的 ``openarm_mini`` 段提供。
本程序会永久修改每个 STS3215 的零位偏移与位置范围；它不依赖 LeRobot，也不
连接或驱动从臂。
"""

from __future__ import annotations

from teleop_sdk.adapters import OpenArmMiniLeaderCalibrator
from teleop_sdk.config import OpenArmMiniLeaderConfig, load_runtime_config


def _validate_config(config: OpenArmMiniLeaderConfig) -> None:
    """在打开串口前拒绝缺失或不安全的部署配置。"""

    if not config.port_left.strip() or not config.port_right.strip():
        raise ValueError("openarm_mini.port_left 和 port_right 均不能为空")
    if config.port_left == config.port_right:
        raise ValueError("openarm_mini.port_left 和 port_right 必须是两条不同串口")
    if not config.calibration_path.strip():
        raise ValueError("请在 teleop.yaml 设置 openarm_mini.calibration_path")
    if not isinstance(config.baudrate, int) or config.baudrate <= 0:
        raise ValueError("openarm_mini.baudrate 必须为正整数")


def _create_calibrators(config: OpenArmMiniLeaderConfig) -> list[OpenArmMiniLeaderCalibrator]:
    """按固定左右顺序创建两个独立的标定会话。"""

    return [
        OpenArmMiniLeaderCalibrator(
            port=config.port_left,
            calibration_path=config.calibration_path,
            side="left",
            baudrate=config.baudrate,
        ),
        OpenArmMiniLeaderCalibrator(
            port=config.port_right,
            calibration_path=config.calibration_path,
            side="right",
            baudrate=config.baudrate,
        ),
    ]


def main() -> int:
    """交互式标定左右 OpenArm Mini，并将结果保存到 YAML 指定路径。"""

    runtime = load_runtime_config()
    config = runtime.openarm_mini
    calibrators: list[OpenArmMiniLeaderCalibrator] = []

    try:
        _validate_config(config)
    except ValueError as exc:
        print(f"[ERROR] OpenArm Mini 配置无效: {exc}")
        return 2

    print("=" * 60)
    print("    OpenArm Mini 独立零位标定")
    print("=" * 60)
    print("  [WARN] 此操作会永久写入两侧电机的零位偏移和位置范围。")
    print("  [WARN] 请确保周围无人、主臂有支撑，并按提示手动移动主臂。")
    print(f"  标定结果将保存到: {config.calibration_path}")
    print("-" * 60)

    try:
        input("[INPUT] 确认开始标定后按 Enter，按 Ctrl+C 取消: ")
        calibrators = _create_calibrators(config)
        for side, calibrator in zip(("left", "right"), calibrators, strict=True):
            print(f"\n[INFO] 开始标定 {side} OpenArm Mini")
            calibrator.connect()
            calibrator.calibrate()
            print(f"[INFO] {side} 标定已保存")

        print("\n[PASS] 左右 OpenArm Mini 标定完成")
        return 0
    except KeyboardInterrupt:
        print("\n[STOP] OpenArm Mini 标定已取消")
        return 130
    except Exception as exc:
        print(f"[ERROR] OpenArm Mini 标定异常: {exc}")
        return 1
    finally:
        for calibrator in reversed(calibrators):
            calibrator.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
