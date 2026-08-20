# OpenArm Mini -> HCX 现场运行说明

建议严格按以下顺序操作：先用上位机检查状态并将双臂调到合适的过渡位置，再用目标姿态脚本回到预设零位，最后启动 OpenArm Mini 双臂遥操数据采集。所有真机命令都可能改变机械臂或夹爪状态，运行前必须确认工作空间、硬件保护和独立急停可用。

## 1. HCX 双臂上位机检查与过渡位置调整

```bash
python -m examples.hcx_follower_gui
```

该命令启动 PySide6/Qt 上位机，连接参数、左右机器人 ID 和 EtherCAT 主站索引来自 `teleop.yaml` 的 `hcx` 段。它不启动直伺服，也不自动执行 `hcx.auto_*`；所有状态修改都需要用户点击对应按钮。

首次使用先安装独立 GUI 依赖：

```bash
python -m pip install -r requirements_hcx_follower_gui.txt
```

界面可用于：

- 连接 HCX，查看报警、软急停、示教器、EtherCAT、全局使能和左右单臂使能状态。
- 在确认报警原因已排除后清除报警，或在示教器已按规程物理拔除后请求脱离示教器。
- 按顺序完成全局使能和单臂使能。条件未满足时，相关按钮会被禁用并显示原因。
- 读取当前七轴角度；“当前位置填入”可将反馈角度填到规划目标输入框。
- 提交单臂规划关节运动，或按住每轴 `-` / `+` 按钮点动；松开按钮会清路停止。
- 对规划运动执行暂停、恢复和清路。

远程设备上运行时必须有可用的 X11/Wayland 图形会话。如果终端没有 `DISPLAY`/Wayland 环境，或系统缺少 Qt `xcb` 依赖，窗口无法启动；这是图形环境问题，不是 HCX 连接失败。

在上位机中将双臂调到距离障碍物和软限位都有足够余量的过渡位置，确认左右臂反馈稳定后关闭上位机，再执行下一步。

## 2. HCX 双臂回到预设零位

```bash
python -m examples.test_terminal_axis_to_zero
```

该脚本使用 HCX 的规划关节运动，将左右两臂移动到脚本顶部定义的绝对关节角度。这里的“回零”指移动到 `LEFT_TARGET_ANGLES_DEG` / `RIGHT_TARGET_ANGLES_DEG` 定义的预设姿态，不会重新标定驱动器机械零点。运动分两阶段执行：

1. 保持 J1 当前角度，先移动 J2-J7。
2. 持续读取实际关节反馈，确认 J2-J7 全部到位后，再单独移动 J1。

运行前直接修改 [examples/test_terminal_axis_to_zero.py](examples/test_terminal_axis_to_zero.py) 顶部的常量，该脚本不读取 `teleop.yaml`：

- `LOCAL_IP` / `REMOTE_IP` / `PORT`：本机和 HCX 控制器通信参数。
- `LEFT_ARM_ID` / `RIGHT_ARM_ID`：左右臂在 HCX 项目中的机器人 ID。
- `LEFT_TARGET_ANGLES_DEG` / `RIGHT_TARGET_ANGLES_DEG`：左右七轴绝对目标，单位为度；`None` 表示保持该轴当前位置。
- `MOTION_SPEED_RATIO`、`ACCELERATION_SECONDS`、`DECELERATION_SECONDS` 和 `SMOOTH`：控制器规划运动参数。
- `ETHERCAT_MASTER_INDICES`：使用 EtherCAT 时按现场配置填写 `0` 和/或 `1`；非 EtherCAT 保持空元组。
- `AUTO_DETACH_HMI_IF_NO_TEACH_PENDANT`、`AUTO_CLEAR_ALARMS` 和 `AUTO_ENABLE`：只有在现场确认允许脚本执行相应状态修改时才设为 `True`。
- `CONFIRM_MOTION`：真正允许发送运动命令的最终开关。

程序不依赖厂商完成回调来切换阶段，而是读取左右臂反馈并按 `ANGLE_TOLERANCE_DEG` 确认到位。成功后终端会输出两个阶段的运动结果和最终关节反馈。

## 3. OpenArm Mini -> HCX 双臂遥操与数据采集

```bash
python openarm_hcx_dual_arm_record.py
```

该入口在已有 OpenArm Mini -> HCX 双臂遥操和双 Gloria-M 夹爪控制上增加三路相机数据采集。HCX 左右臂仍由每侧独立的直伺服发送线程按 `hcx.direct_servo_rate_hz` 运行；相机采集和低频夹爪线程不进入该发送路径。

运行前检查 `teleop.yaml` 中的：

- `openarm_mini`：左右主臂串口和双侧标定文件。
- `teleop`：主臂采样、映射和通用平滑参数。
- `hcx`：控制器地址、机器人 ID、关节方向、直伺服频率、插值模式与明确授权的 `auto_*` 状态操作。
- `gloria_m_dual`：左右 Gloria-M 串口和共用控制参数。正式双臂数据采集要求两侧都启用。
- `hcx_orbbec`：`head`、`left_hand` 和 `right_hand` 三台相机的序列号、分辨率和帧率。
- `hcx_recording`：数据集目录、标称帧率、任务名称、图像格式和最小剩余空间。`hcx_recording.fps` 必须与三台 `hcx_orbbec` 相机一致，但它不是 HCX 伺服频率。

采集由头部相机驱动：每张新 `head` 图像触发一次实际双臂/双夹爪反馈读取，左右手图像选择不晚于该头部帧的最新缓存帧。实际数据行数由成功的头部帧和完整反馈决定，不保证等于标称 FPS。

启动并完成预检后，终端按键为：

| 按键 | 功能 |
| --- | --- |
| `s` | 开始一个 episode；再次按下停止接收新头部帧并在后台保存。 |
| `c` | 丢弃当前尚未完成的 episode。 |
| `q` | 停止遥操和夹爪线程，关闭相机流和采集进程。 |

第二次按下 `s` 后，等待终端显示 `episode 000000 已写入数据集` 再开始下一段。数据默认写入 `hcx_recording.root`；数据字段和 14 轴顺序见 [OPENARM_HCX_DUAL_ARM_RECORDING_DATA.md](OPENARM_HCX_DUAL_ARM_RECORDING_DATA.md)。

如果 episode 保存时提示没有完整帧，按错误中给出的 `meta/failed_recording_audit/*.jsonl` 路径查看跳过原因：

- `missing_camera_for_master_frame:<name>`：该手部相机没有可与头部帧配对的缓存帧。
- `feedback_snapshot_timeout`：双臂/双夹爪反馈未在时限内完成。
- `missing_gloria_m_feedback`：左或右 Gloria-M 没有有效反馈。

失败 episode 不占用正式 episode 编号；程序进入 `FAULT` 后会停止 HCX 直伺服和夹爪线程并关闭采集资源。
