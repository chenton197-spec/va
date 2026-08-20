"""新 SDK 模拟从臂的实时关节可视化。"""

from __future__ import annotations

import threading
import time

import numpy as np

from .adapters.mock_follower import MockFollower


# 保持原 FR3 模拟器的关节软限位和关节命名。
JOINT_MIN = np.array([-175.0, -265.0, -150.0, -265.0, -175.0, -360.0])
JOINT_MAX = np.array([175.0, 85.0, 150.0, 85.0, 175.0, 360.0])
JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]


def _configure_chinese_font(matplotlib) -> str | None:
    """选择系统已有的中文字体，避免 Matplotlib 回退到 DejaVu Sans。"""
    from matplotlib import font_manager

    candidates = ("Noto Sans CJK SC", "Source Han Sans CN", "Microsoft YaHei", "SimHei")
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in installed), None)
    if selected is not None:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    # 让坐标轴负数使用普通连字符，避免 Unicode 减号的字体兼容问题。
    matplotlib.rcParams["axes.unicode_minus"] = False
    return selected


def start_visualization(mock: MockFollower, stop_event: threading.Event) -> None:
    """在主线程显示模拟从臂的当前角度与目标角度。

    图形窗口关闭后会设置 ``stop_event``，与旧 ``--sim`` 模式相同。
    """
    # 延迟导入，确保命令行帮助和无图形测试不需要加载 Tk 后端。
    import matplotlib

    matplotlib.use("TkAgg")
    chinese_font = _configure_chinese_font(matplotlib)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    figure, (axis_bar, axis_info) = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [3, 1]}
    )
    figure.canvas.manager.set_window_title("FR3 Simulator -- teleop_sdk")
    figure.patch.set_facecolor("#1a1a2e")

    axis_bar.set_facecolor("#16213e")
    positions = np.arange(6)
    for index in range(6):
        axis_bar.barh(
            positions[index],
            JOINT_MAX[index] - JOINT_MIN[index],
            left=JOINT_MIN[index],
            height=0.55,
            color="#0f3460",
            alpha=0.5,
            zorder=1,
        )
        axis_bar.axvline(0, color="#444466", linewidth=0.8, zorder=1)

    bars = axis_bar.barh(positions, np.zeros(6), height=0.45, color="#4cc9f0", zorder=3)
    target_dots, = axis_bar.plot(
        np.zeros(6), positions, "o", color="#f72585", markersize=9, zorder=4
    )
    axis_bar.set_yticks(positions)
    axis_bar.set_yticklabels(JOINT_NAMES, color="white", fontsize=11)
    axis_bar.tick_params(axis="x", colors="#aaaacc")
    axis_bar.set_xlabel("角度（度）", color="#aaaacc")
    axis_bar.set_title("FR3 关节状态（模拟）", color="white", fontsize=12, pad=10)
    for spine in axis_bar.spines.values():
        spine.set_edgecolor("#333355")

    axis_info.set_facecolor("#16213e")
    axis_info.set_xticks([])
    axis_info.set_yticks([])
    axis_info.set_title("状态", color="white", fontsize=11, pad=8)
    for spine in axis_info.spines.values():
        spine.set_edgecolor("#333355")
    info_text = axis_info.text(
        0.05,
        0.95,
        "",
        transform=axis_info.transAxes,
        fontsize=9,
        va="top",
        color="#e0e0ff",
        # 不能强制使用 DejaVu Sans Mono，否则中文状态文本会缺字。
        fontfamily=chinese_font or "sans-serif",
    )
    figure.tight_layout(pad=2.0)

    def update(_frame: int) -> None:
        """以 20 Hz 刷新关节柱状图和状态文本。"""
        if stop_event.is_set():
            plt.close(figure)
            return

        joints, target, command_count, last_command_time = mock.get_visualization_state()
        left = np.minimum(joints, 0.0)
        widths = joints - left
        for bar, x_value, width in zip(bars, left, widths):
            bar.set_x(x_value)
            bar.set_width(width if width != 0 else 1e-9)

        margin = np.minimum(joints - JOINT_MIN, JOINT_MAX - joints)
        ratio = np.clip(margin / ((JOINT_MAX - JOINT_MIN) * 0.1 + 1e-9), 0, 1)
        for bar, value in zip(bars, ratio):
            bar.set_color(plt.cm.RdYlGn(0.2 + 0.8 * value))
        target_dots.set_xdata(target)

        all_values = np.concatenate([joints, target, JOINT_MIN, JOINT_MAX])
        axis_bar.set_xlim(all_values.min() - 10, all_values.max() + 10)
        age = f"{time.perf_counter() - last_command_time:.3f}s" if last_command_time else "N/A"
        lines = [""]
        for index, (joint, target_angle) in enumerate(zip(joints, target)):
            lines.append(f"J{index + 1}: {joint:+8.2f} -> {target_angle:+8.2f}")
        lines.extend(["", f"命令数: {command_count}", f"最近命令: {age}"])
        info_text.set_text("\n".join(lines))

    animation = FuncAnimation(figure, update, interval=50, cache_frame_data=False)
    _ = animation
    plt.show()
    stop_event.set()
