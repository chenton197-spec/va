# teleop_sdk

`teleop_sdk` 是一个可扩展的 Python 主从遥操作与数据采集框架。仓库提供两套参考部署：Alicia-D 控制 FAIRINO FR3，以及 OpenArm Mini 控制 HCX 双七轴机械臂。两套部署均可配合 Gloria-M 夹爪和奥比中光相机进行数据采集。

本文只说明日常操作和采集。需要接入其他机械臂或相机时，查看 [设备适配指南](ADAPTER_GUIDE.md)。

# fairino390 sdk

https://hlrobots.feishu.cn/file/FuujbzjAEool4uxOinXccIHQnVb?from=from_copylink

## 开始前

正式运行前请确认以下事项：

- FR3 周围没有人员、工具或碰撞风险，机械臂处于安全初始姿态。
- FR3 已切换到自动模式并可上使能。
- Alicia-D、Gloria-M 和需要使用的相机均已上电并接好线缆。
- 使用奥比中光相机时，设备已获得 Linux USB 权限，且没有被其他程序占用。
- 数据盘剩余空间满足 `recording.min_free_disk_gb` 或 `hcx_recording.min_free_disk_gb`。

首次使用真实设备前，建议先完成下方的单设备检查。

根目录 `alicia_fr3_record.py` 和 `openarm_hcx_dual_arm_record.py` 分别是两套正式采集入口。OpenArm Mini -> HCX 只遥操不采集时，使用 `python openarm_hcx_dual_arm_teleop.py`。

## 安装环境

项目 Conda 环境：

```bash
conda create -name teleop python=3.11
```

激活虚拟环境，安装 Alicia-D + FR3 参考部署依赖

```bash
conda activate teleop
python -m pip install -r requirements-alicia.txt
```

根目录有四个依赖入口：`requirements.txt` 是两个示教臂共享的依赖；Alicia-D、OpenArm Mini 和 MuJoCo 虚拟从臂分别在此基础上增加各自的可选依赖。只使用其中一种设备时，无需安装其他设备的厂商 SDK。

```bash
# Alicia-D + FR3 参考部署，不安装 OpenArm Mini SDK
python -m pip install -r requirements-alicia.txt

# 只使用 OpenArm Mini 示教臂，不安装 Alicia-D 或 LeRobot
python -m pip install -r requirements-openarm.txt

# 使用 CASBOT URDF 的 MuJoCo 虚拟从臂，不安装 Alicia-D 或 OpenArm Mini SDK
python -m pip install -r requirements-mojoco.txt
```

摄像头usb权限

```bash
sudo "$(python -c 'import pyorbbecsdk,os;print(os.path.join(os.path.dirname(pyorbbecsdk.__file__),"pyorbbecsdk","shared","install_udev_rules.sh"))' | tail -n 1)"
```

Linux 如遇权限问题，可执行：

```bash
sudo usermod -a -G dialout $USER
```
或者
```bash
sudo chmod 666 /dev/ttyACM0
```

## 配置设备

所有日常配置都在根目录 [teleop.yaml](teleop.yaml)。开始前至少检查以下字段：

| 配置 | 用途 |
| --- | --- |
| `alicia.port` | Alicia-D 串口；留空时由 Alicia SDK 自动寻找。 |
| `openarm_mini.port_left` / `port_right` | OpenArm Mini 左右主臂串口。 |
| `openarm_mini.calibration_path` | 独立 OpenArm Mini 标定示例创建或更新的 `left` / `right` 组合 JSON。 |
| `teleop.rate_hz` | 主臂采样、映射和通用平滑计算频率；OpenArm -> HCX 部署设为 100 Hz。 |
| `teleop.filter_enabled` | 同时启用或关闭 One Euro 一级滤波与固定低通二级滤波。 |
| `teleop.spring_enabled` | 同时启用或关闭弹簧阻尼与前瞻预测。 |
| `teleop.latency_probe_enabled` | 目标提交时序诊断开关；不读取从臂反馈，实时遥操应保持关闭。 |
| `fr3.robot_ip` | FR3 的实际 IP 地址。 |
| `fr3.axis_sign` | Alicia-D -> FR3 的六轴部署方向数组；由 Alicia-FR3 入口注入通用控制器。 |
| `hcx.local_ip` / `remote_ip` | HCX 双臂状态检查与遥操作的本机和控制器地址。 |
| `hcx.left_robot_id` / `right_robot_id` | HCX 左右机械臂在控制器项目中配置的机器人 ID。 |
| `hcx.left_axis_sign` / `right_axis_sign` | OpenArm Mini -> HCX 关节方向数组；`1` 同向，`-1` 反向。 |
| `hcx.direct_servo_rate_hz` | HCX 每侧 Python 输出线程调用 `PluseToServo` 的频率，范围为 100–1000 Hz；`limited` 可独立设为 500、800、1000 Hz 等值。 |
| `hcx.direct_servo_interpolation` | `direct` 时 Python 线程重发最新目标；`linear` 时在 Python 队列中重采样低频主臂段；`limited` 时在 Python 中逐点执行低通、限速、限加速度，再调用单次 `set_target()`。 |
| `hcx.direct_servo_watchdog_s` | Python 会话的成功发送间隔监测阈值；输出线程会重复静止目标，因此主臂不动不会超时。 |
| `hcx.direct_servo_confirm_unsafe` | 直伺服的显式危险确认；默认 `false`，根目录 OpenArm demo 在其为 `false` 时不会连接硬件。 |
| `hcx.auto_detach_hmi` | 仅在现场确认未接示教器且已物理拔除时开启；请求控制器脱离示教器。 |
| `hcx.auto_clear_alarms` | 仅在已排除报警原因且允许复位后开启；有限次数请求清除 HCX 报警。 |
| `hcx.auto_enable` | 显式开启后，HCX 从臂启动流程才会自动开启全局及左右单臂使能。 |
| `gloria_m.port` | Gloria-M 串口，例如 `/dev/ttyACM0`。 |
| `gloria_m_dual` | OpenArm Mini 左右夹爪的共用控制参数和两个 Gloria-M 串口；普通遥操可只启用一侧，正式双臂采集要求两侧都启用。 |
| `orbbec.cameras` | 每台相机的名称、序列号、模式、分辨率和帧率。 |
| `recording.enabled_cameras` | 本次要写入数据集的相机名称；设为 `[]` 可不采集相机。 |
| `recording.master_camera` | 多相机时负责驱动采集的一台相机，必须在启用列表中。 |
| `recording.root` | 数据集保存目录。 |
| `recording.image_storage` | `png`、`jpg` 或 `video`。 |
| `hcx_orbbec.cameras` | HCX 正式采集的 `head` / `left_hand` / `right_hand` 三路相机。 |
| `hcx_recording.fps` | HCX 数据集标称帧率；必须与三台 `hcx_orbbec` 相机帧率一致，不是 HCX 伺服调度频率。 |
| `hcx_recording.root` | OpenArm Mini -> HCX 双臂数据集保存目录。 |

`openarm_mini.calibration_path` 可直接写相对路径，例如
`./my_openarm_mini.json`。该文件不需要预先创建；运行标定示例后会自动创建，并在
左右两侧均标定完成时包含 `left` 和 `right` 两个条目。

HCX 双臂从臂应共享一个连接，并通过运行时配置构造其启动参数。`start_servo()` 会
按配置依次复核示教器、报警、软急停和 EtherCAT 状态；只有相关 `auto_*` 项显式为
`true` 时才会修改控制器状态。规划模式的 `recover()` 仅复核状态；直伺服模式会在状态
仍正常时重建本侧软件会话，但不会重复脱离示教器、清报警或使能。规划模式退出时清除本侧
路径；直伺服退出时仅停止软件目标重发，不会自动关闭使能、暂停或发送物理急停。

```python
from teleop_sdk import load_runtime_config
from teleop_sdk.adapters import HcxConnection, HcxConnectionConfig, HcxFollower

runtime = load_runtime_config().hcx
connection = HcxConnection(HcxConnectionConfig.from_runtime_config(runtime))
left = HcxFollower(connection, robot_id=runtime.left_robot_id, side="left")
right = HcxFollower(connection, robot_id=runtime.right_robot_id, side="right")
```

根目录的 OpenArm Mini -> HCX 双臂入口使用相对同序七轴映射和
`hcx.left_axis_sign` / `right_axis_sign` 的部署方向数组，通过 `PluseToServo` 直接关节
伺服驱动两侧机械臂。当前参考配置以 100 Hz 读取和处理主臂；在 `limited` 模式下，
每次主臂更新只替换最新目标，HCX 适配器再按 `hcx.direct_servo_rate_hz` 独立输出单点
`set_target()`，因此该频率可设为 500、800 或 1000 Hz，而不依赖主臂采样频率的整数比。
`linear` 模式仍按相邻主臂目标在 Python 中生成队列。左右主臂读取在独立工作线程中并行执行，避免一侧串口
等待阻塞另一侧。`gloria_m_dual.left/right.enabled` 决定是否同时用两个独立低频线程控制左右 Gloria-M；这些线程不进入 HCX 每侧 500 Hz 直伺服发送路径。首次现场联调
应逐轴确认方向；启动前必须完成 OpenArm Mini 双侧标定，
按现场安全条件配置 HCX 的 `auto_*` 授权项，并在隔离单元、硬件保护与独立急停已确认后，
才将 `hcx.direct_servo_confirm_unsafe` 改为 `true`。

```bash
python openarm_hcx_dual_arm_teleop.py
```

### OpenArm Mini -> HCX 双臂正式采集

`openarm_hcx_dual_arm_record.py` 在上述已有遥操和夹爪控制上增加数据采集，不改变 HCX 每侧独立的直伺服发送线程。正式采集前必须配置三台相机和左右 Gloria-M：

```bash
python openarm_hcx_dual_arm_record.py
```

每张新的 `head` 图像触发一次双臂和双夹爪真实反馈读取，左右手图像选择不晚于该头部帧的最新缓存帧。`hcx_recording.fps` 只表示数据集标称帧率并用于校验三台相机配置；它不控制 500 Hz 伺服。当前参考配置为三路 15 FPS。

| 按键 | 操作 |
| --- | --- |
| `s` | 开始 episode；再次按下停止接收新帧并后台保存。 |
| `c` | 丢弃当前 episode。 |
| `q` 或 `Ctrl+C` | 停止遥操和夹爪线程，关闭相机与采集进程。 |

相机使用 `mode: "rgb"` 时只采集彩色图；使用 `mode: "rgbd"` 时额外保存原始深度图。Alicia-FR3 启用相机的 `fps` 必须与 `recording.fps` 相同；OpenArm-HCX 三台相机的 `fps` 必须与 `hcx_recording.fps` 相同。

常用存储设置如下：

```yaml
recording:
  root: "datasets/alicia_fr3"
  fps: 30
  image_storage: "jpg"
  quality: 90
```

`png` 无损但占用空间较大；`jpg` 占用较小，`quality` 范围为 1-100；`video` 写入 AV1 MP4。PNG 和 JPG 可以作为不同 episode 写入同一数据集，但一个 episode 内只能使用一种格式。视频数据集请使用新的 `recording.root`。

## 单设备检查

这些命令均从仓库根目录运行。它们不会写入正式数据集。

| 命令 | 用途 |
| --- | --- |
| `python -m examples.test_orbbec_camera` | 预览 YAML 中声明的奥比中光相机；按 `q` 或 Esc 关闭。 |
| `python -m examples.test_orbbec_collection_fps` | 检查采集接口的实际相机帧率和双相机配对情况。 |
| `python -m examples.test_fr3_observation_state` | 读取并打印 FR3 当前关节反馈，不启动 ServoJ。 |
| `python -m examples.test_hcx_dual_arm_state` | 一次性读取 HCX 左右七轴的角度与原始力矩反馈，不发送控制命令。 |
| `python -m examples.hcx_follower_gui` | 打开基于 PySide6/Qt 的 HCX 双七轴从臂上位机，查看状态与反馈，并在确认后执行单臂规划运动或状态管理；每轴 `-` / `+` 按住开始反转/正转，松开即清路停止。按说明书顺序完成脱离示教器、清除报警、EtherCAT OP、全局使能和单臂使能后，运动按钮才会解锁。链路断开时会自动关闭本地会话并禁用控制，需手动重新连接。首次运行前安装 `python -m pip install -r requirements_hcx_follower_gui.txt`。 |
| `python -m examples.test_limited_interpolation` | 使用 HCX SDK 单臂直伺服测试 LimitedInterpolator：按 `LIMITED_UPDATE_RATE_HZ` 生成受限轨迹点批次，并固定以 500 Hz 下发；每个发送时隙不重新执行算法。参数定义在脚本顶部，会产生真实运动。 |
| `python -m examples.test_gloria_m_observation_gripper` | 读取并打印 Gloria-M 当前开合反馈。 |
| `python -m examples.test_openarm_gloria_m_dual_gripper_teleop` | 不连接 HCX，单独验证 OpenArm Mini 左右夹爪到 Gloria-M 的映射。 |
| `python -m examples.test_parquet_collection_quality` | 分析脚本顶部 `PARQUET_PATH` 指定的 episode，输出帧数、时间连续性、关节跳变和图像完整性。 |
| `python -m examples.test_alicia_zero_calibration` | 按 Alicia SDK 的两次确认流程校准设备零位。 |
| `python -m examples.test_alicia_move_to_angles` | 移动 Alicia-D 到脚本内 `TARGET_JOINTS_DEG` 指定的六轴角度。 |
| `python -m examples.test_openarm_mini_calibration` | 交互式标定双 OpenArm Mini 的关节零位和夹爪行程，并写入 `openarm_mini.calibration_path`。 |
| `python -m examples.test_openarm_mini_calibration_verify` | 只读校验双 OpenArm Mini 的零位和夹爪标定，不写入电机。 |
| `python -m examples.test_openarm_mini_observation_state` | 持续只读打印左右 OpenArm Mini 的关节和夹爪状态。 |
| `python -m examples.test_openarm_mini_filtered_observation_state` | 持续输出左右 OpenArm Mini 的关节滤波和虚拟弹簧阻尼结果。 |
| `python -m examples.test_mujoco_casbot_dual_arm` | 在一个 MuJoCo 窗口中加载 CASBOT URDF，并同时展示左右七轴虚拟从臂。 |

若相机序列号不正确、没有 USB 权限或相机被其他程序占用，请先解决相机检查中的报错，再启动正式采集。

> **警告：** `test_alicia_zero_calibration` 会永久修改 Alicia-D 的设备零位，不是设置本次遥操的相对起点。SDK 会先要求确认、关闭扭矩，再要求你手动将机械臂摆到目标零位后确认。仅在机械臂有支撑、工作区安全且已确认机械零位姿态时运行。

> **动作警告：** `test_alicia_move_to_angles` 不接受 CLI 参数。运行前直接修改脚本顶部的 `TARGET_JOINTS_DEG` 和 `SPEED_DEG_S`，程序会显示目标并要求按 Enter 后才下发运动指令。目标角度必须符合实际机械限位。

## 启动遥操作与采集

```bash
conda activate alicia
python alicia_fr3_record.py # alicia
python openarm_hcx_dual_arm_record.py # openarm
```

程序会连接设备、确认主相机出帧，并启动 FR3 ServoJ。看到“正式遥操作与采集”及设备信息后即可移动 Alicia-D；默认是相对遥操作模式。

| 按键 | 操作 |
| --- | --- |
| `s` | 第一次按下开始一个 episode；再次按下停止采样并在后台写入数据。 |
| `c` | 丢弃当前尚未完成的 episode。 |
| `q` | 安全停止控制、关闭相机流并退出；未完成的 episode 会被丢弃。 |

停止采样后请等待终端出现 `episode 000000 已写入数据集` 一类提示，再开始下一段采集。遥操作控制会保持运行，数据写入在后台完成。

## 数据在哪里

以 `png` 或 `jpg` 存储时，默认 `recording.root` 的结构如下：

```text
datasets/alicia_fr3/
├── data/chunk-000/episode_000000.parquet
├── images/chunk-000/observation.images.hand/episode_000000/
├── images/chunk-000/observation.images.head/episode_000000/
├── videos/chunk-000/...                        # 仅 video 模式
├── depth/chunk-000/...                         # 仅 RGB-D 模式
└── meta/
```

- `Parquet`：每帧的 FR3 状态、动作、夹爪值、时间戳和图像路径。
- `images`：每台相机的 PNG 或 JPG 图像序列。
- `depth`：RGB-D 模式下的原始 `uint16` 深度数据。
- `meta/recording_audit`：采集时序、相机帧号和跳帧记录，用于排查同步问题。
- OpenArm Mini -> HCX 双臂数据的字段、关节顺序和最小读取示例见 [OPENARM_HCX_DUAL_ARM_RECORDING_DATA.md](OPENARM_HCX_DUAL_ARM_RECORDING_DATA.md)。

查看一个 episode 的表格内容时，先在 [examples/view_parquet.py](examples/view_parquet.py) 修改 `PARQUET_PATH`，再运行：

```bash
python -m examples.view_parquet
```

回放相机图像时，在 [examples/play_png_sequence.py](examples/play_png_sequence.py) 修改 `IMAGE_DIRECTORY`，再运行：

```bash
python -m examples.play_png_sequence
```

窗口中按空格暂停或继续，按 `a`、`d` 逐帧查看，按 `q` 或 Esc 退出。

## 常见问题

**相机提示未更新或没有第一帧**：先运行 `python -m examples.test_orbbec_camera`。确认相机接在稳定的 USB 3.x 接口、序列号正确，并退出可能占用相机的旧程序。

**提示 USB Access denied**：这是系统权限问题。按奥比中光 SDK 的 Linux udev 规则配置权限，重新插拔相机或重新登录后再试。

**数据集配置不匹配**：相机数量、图像分辨率、关节数量或 `recording.fps` 改变后，请使用新的 `recording.root`。PNG/JPG 之间切换不需要新目录；切换到 `video` 需要新目录。

**HCX episode 没有完整帧**：查看错误中给出的 `meta/failed_recording_audit/*.jsonl` 路径。`missing_camera_for_master_frame:<name>` 表示该手部相机没有可与头部帧因果配对的缓存帧；`feedback_snapshot_timeout` 表示双臂/双夹爪反馈超时；`missing_gloria_m_feedback` 表示夹爪反馈不完整。失败 episode 不占用正式 episode 编号。

**程序退出后相机仍无法打开**：优先使用 `q` 或 `Ctrl+C` 正常退出，让程序关闭采集进程和相机流；若进程异常中断，确认没有遗留 Python 进程后再重新连接相机。

## 更多资料

- [奥比中光相机说明](orbbec_sdk/README.md)
- [采集数据格式说明](leobot_scripts/README.md)
- [接入新设备](ADAPTER_GUIDE.md)
