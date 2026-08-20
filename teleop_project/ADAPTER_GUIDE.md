# 设备适配指南

`teleop_sdk` 将示教臂、从臂、从端夹爪和遥操作算法分开。接入新硬件时，在
`teleop_sdk/adapters/` 新增对应适配器；厂商 SDK 的单位、连接和错误处理必须
留在适配器中，`TeleopController` 不应依赖厂商方法名。

## 通用约定

- 公共关节角度统一使用**角度**，类型为 `numpy.ndarray`。
- 示教臂与从臂的 `joint_count` 是适配器固定能力，必须相等；不相等时控制器拒绝启动。
- 从臂默认关节限位由适配器的 `joint_limits_deg` 提供；YAML 可在部署时覆盖限位，
  但仍会接受长度和安全性校验。
- 厂商 SDK 使用弧度时，在适配器中转换为角度；不要把弧度传给控制器。
- 夹爪开合量统一为 `0.0`（闭合）到 `1.0`（张开）。
- 设备连接、错误码、伺服模式、资源释放和厂商异常都由适配器处理。

## 1. 示教臂

### 必需接口

所有示教臂继承 `LeaderArm` 并实现：

| 方法 | 职责 |
|---|---|
| `joint_count` | 固定关节数量；由适配器实现，不由用户配置。 |
| `connect()` | 建立设备连接；失败时抛出异常。 |
| `read_joint_angles_deg(timeout_s)` | 返回当前关节角度（度）；超时、无数据或临时失败时返回 `None`。 |
| `disconnect()` | 释放设备连接。 |

```python
import numpy as np

from teleop_sdk.interfaces import LeaderArm


class MyLeaderArm(LeaderArm):
    def __init__(self, address: str):
        self.address = address
        self.robot = None

    @property
    def joint_count(self) -> int:
        return 6  # 设备固定能力

    def connect(self) -> None:
        self.robot = vendor_sdk.connect(self.address)

    def read_joint_angles_deg(self, timeout_s: float) -> np.ndarray | None:
        if self.robot is None:
            return None
        joints_rad = self.robot.read_joints(timeout=timeout_s)
        if joints_rad is None:
            return None
        return np.rad2deg(np.asarray(joints_rad, dtype=float))

    def disconnect(self) -> None:
        if self.robot is not None:
            self.robot.close()
            self.robot = None
```

### OpenArm Mini（独立标定和读取）

`OpenArmMiniLeaderCalibrator` 不依赖 LeRobot，可完成单侧零位和夹爪行程标定，
并创建或更新组合标定 JSON。运行下面的示例会依次标定 `left` 和 `right`；
`teleop.yaml` 中的 `calibration_path` 可以使用相对路径，例如
`./my_openarm_mini.json`。文件不存在时会在首次成功标定后自动创建。

```bash
python -m examples.test_openarm_mini_calibration
```

完成标定后，可运行只读校验。它要求主臂依次处于零位且夹爪闭合、夹爪完全张开，
并检查七个关节是否均在 `+/-5 deg` 内，以及夹爪是否接近 `0.0` 和 `1.0`：

```bash
python -m examples.test_openarm_mini_calibration_verify
```

持续查看双主臂状态时，运行以下命令并按 `Ctrl+C` 退出。它以默认 20 Hz 输出左右侧
的七轴角度和归一化夹爪值：

```bash
python -m examples.test_openarm_mini_observation_state
```

若需观察关节原始值经过角度连续化、One Euro、固定低通和虚拟弹簧阻尼后的结果，
运行以下只读示例。它不会连接从臂，夹爪也不会参与滤波：

```bash
python -m examples.test_openarm_mini_filtered_observation_state
```

`OpenArmMiniLeaderArm` 读取同一格式的 JSON。字段采用 LeRobot
`MotorCalibration` 的格式，但本适配器使用一个根节点包含 `left` 和 `right` 的
组合文件；各侧包含 `joint_1` 到 `joint_7` 及 `gripper` 的 `id`、`drive_mode`、
`homing_offset`、`range_min`、`range_max` 字段。

先安装该适配器的独立依赖：

```bash
python -m pip install -r requirements-openarm.txt
```

```python
from teleop_sdk.adapters import OpenArmMiniLeaderArm

left_leader = OpenArmMiniLeaderArm(
    port="/dev/ttyACM1",
    calibration_path="/path/to/my_openarm_mini.json",
    side="left",
)
right_leader = OpenArmMiniLeaderArm(
    port="/dev/ttyACM0",
    calibration_path="/path/to/my_openarm_mini.json",
    side="right",
)

left_leader.connect()
try:
    frame = left_leader.read_joint_angles_and_gripper_opening(timeout_s=0.05)
finally:
    left_leader.disconnect()
```

每个实例固定输出 7 个按 `joint_1` 至 `joint_7` 排列的原始标定角度；侧别仅
选择 JSON 子对象，不做 LeRobot 专用的轴翻转或 `joint_6` / `joint_7` 重排。
夹爪通过 `LeaderArmWithGripper` 作为 `0.0`（闭合）到 `1.0`（张开）的输入提供。

### 示教端夹爪或扳机输入

示教端夹爪是**输入设备**，与从端真实夹爪不同。若示教臂只能单独读取夹爪，
额外实现 `LeaderGripperInput`：

```python
from teleop_sdk.interfaces import LeaderGripperInput


class MyLeaderGripper(LeaderGripperInput):
    def read_gripper_opening(self, timeout_s: float) -> float | None:
        raw = self.device.read_trigger(timeout=timeout_s)  # 例如原始范围 0~255
        if raw is None:
            return None
        return float(raw) / 255.0
```

构造控制器时显式传入：

```python
controller = TeleopController(
    leader, follower, config, gripper=follower_gripper, leader_gripper=leader_gripper
)
```

若厂商 SDK 可以在**同一状态帧**返回关节和夹爪输入，应继承
`LeaderArmWithGripper`。控制器会优先使用联合读取，避免同一周期读取两次设备：

```python
from teleop_sdk.interfaces import LeaderArmWithGripper


class MyLeaderArm(LeaderArmWithGripper):
    def read_joint_angles_and_gripper_opening(
        self, timeout_s: float
    ) -> tuple[np.ndarray, float] | None:
        state = self.robot.read_state(timeout=timeout_s)
        if state is None:
            return None
        joints_deg = np.rad2deg(np.asarray(state.joints_rad, dtype=float))
        opening = float(state.trigger) / 255.0
        return joints_deg, opening
```

原始夹爪量程、方向反转、校准范围和设备级死区属于**示教臂适配器配置**。例如
原始范围为 `0~255` 且方向相反时，在适配器内返回 `1.0 - raw / 255.0`；控制器
不需要知道厂商数据范围。

`AliciaLeaderArm` 还提供 Alicia 专用的：

```python
is_synced, is_locked, raw_gripper = leader.get_sync_lock_gripper()
```

该方法返回同步状态、锁定状态和 Alicia 原始夹爪值（通常为 `0~1000`），仅用于
Alicia 诊断或业务逻辑，不是新示教臂必须实现的接口。

## 2. 从臂

### 必需接口

从臂继承 `FollowerArm` 并实现：

| 方法 | 职责 |
|---|---|
| `joint_count` | 固定从臂关节数量；由适配器实现。 |
| `joint_limits_deg` | 返回该从臂默认最小/最大安全关节限位（度）。 |
| `connect()` | 连接并完成厂商要求的初始化。 |
| `read_joint_angles_deg()` | 返回当前从臂关节角度（度），用于建立相对遥操基准。 |
| `start_servo()` | 进入实时/伺服模式；成功返回 `True`。 |
| `send_joint_angles_deg(angles_deg, command_time_s)` | 下发一帧绝对关节目标（度）；成功返回 `True`。 |
| `recover()` | 伺服命令失败后执行厂商恢复流程；成功返回 `True`。 |
| `stop_servo()` | 停止实时/伺服模式和当前运动。 |
| `disconnect()` | 断开设备连接或释放句柄。 |

```python
import numpy as np

from teleop_sdk.interfaces import FollowerArm


class MyFollowerArm(FollowerArm):
    @property
    def joint_count(self) -> int:
        return 6  # 设备固定能力

    @property
    def joint_limits_deg(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-170] * 6), np.array([170] * 6)

    def connect(self) -> None:
        self.robot = vendor_sdk.connect(self.address)

    def read_joint_angles_deg(self) -> np.ndarray:
        return np.rad2deg(np.asarray(self.robot.read_joints(), dtype=float))

    def start_servo(self) -> bool:
        return self.robot.start_realtime_mode() == 0

    def send_joint_angles_deg(self, angles_deg: np.ndarray, command_time_s: float) -> bool:
        joints_rad = np.deg2rad(angles_deg)
        return self.robot.send_joint_target(joints_rad.tolist(), command_time_s) == 0

    def recover(self) -> bool:
        self.robot.stop_realtime_mode()
        self.robot.clear_errors()
        return self.start_servo()

    def stop_servo(self) -> None:
        self.robot.stop_realtime_mode()

    def disconnect(self) -> None:
        self.robot.close()
```

### MuJoCo URDF 虚拟从臂（可选）

`MujocoSimulation` 将一份完整 URDF 加载为共享 MuJoCo 场景；左右
`MujocoFollower` 各自绑定七个指定的旋转关节，但共用同一份模型、状态和查看器。
公共接口仍使用角度，适配器在边界处转换为 MuJoCo 的弧度 `qpos`，随后调用
`mj_forward` 更新正向运动学。

该适配器用于遥操链路、关节映射和姿态可视化验证：它不是动力学仿真，不模拟电机、
减速器、力矩控制、碰撞接触或真实控制周期。腰、头、腿、灵巧手和夹爪可以保留在
URDF 中显示，但只会控制传给 `MujocoFollower` 的关节名称。

安装可选依赖并运行仓库内的双臂示例：

```bash
conda activate alicia
python -m pip install -r requirements-mojoco.txt
python -m examples.test_mujoco_casbot_dual_arm
```

示例顶部的 `URDF_FILENAME` 是唯一的模型选择变量；默认加载
`simulation/urdfs/CASBOTWL12_WL12P1.urdf`。它没有命令行参数。被动查看器由
`simulation.open_viewer()` 在主线程创建，因此两个虚拟从臂会显示在同一个窗口中。

当前项目中的 CASBOT URDF 已是 MuJoCo 专用文件：它用 `meshdir="../meshes"` 定位
项目内网格，并将非左右七轴的可动关节固定在零位。因此右侧 `Joint` 面板只显示这
14 个手臂关节，但机器人其他部分仍会按零位姿态显示。适配器直接按文件路径加载
URDF，不处理 ROS 的 `package://` URI。

```python
from teleop_sdk.adapters import MujocoFollower, MujocoSimulation

simulation = MujocoSimulation("simulation/urdfs/CASBOTWL12_WL12P1.urdf")
left = MujocoFollower(simulation, LEFT_ARM_JOINT_NAMES)
right = MujocoFollower(simulation, RIGHT_ARM_JOINT_NAMES)
left.connect()
right.connect()
left.start_servo()
right.start_servo()
left.send_joint_angles_deg(left_target_deg, command_time_s=0.008)
right.send_joint_angles_deg(right_target_deg, command_time_s=0.008)
```

URDF 里的 `revolute` 限位会自动成为对应虚拟从臂的 `joint_limits_deg`。当前网格
路径依赖 `simulation/urdfs` 与 `simulation/meshes` 的同级目录布局；移动 URDF 时
需要同步调整其 `mujoco/compiler@meshdir`。

### 从臂配置归属

以下参数属于控制器运行配置；未在 YAML 覆盖时，映射默认按关节数量自动生成：

- `axis_order`：每个从臂关节由哪个示教臂关节驱动；默认 `J1->J1 ... Jn->Jn`。
- `axis_sign`：各轴方向，默认全为 `1`；YAML 可覆盖为 `-1`。
- `min_angles_deg` / `max_angles_deg`：可选部署覆盖；默认来自从臂适配器。
- `max_vel_deg_s`、`max_accel_deg_s2`、`spring_omega`：控制器的平滑与安全限制。

IP、串口、认证信息、厂商伺服模式、协议速度和错误码处理属于**从臂适配器自己的
配置**。根目录 `teleop.yaml` 可覆盖 IP、串口、CAN ID、控制参数、映射、方向与限位；
未填写时使用 `teleop_sdk/config.py` 的代码默认值。`joint_count` 不在 YAML 架构中，
写入会被拒绝。

## 3. 从端夹爪

从端夹爪是**执行设备**：它接收控制器的 `0.0~1.0` 开合目标并转换为真实位置、
速度、力或力矩命令。它不属于 `FollowerArm`，这样没有夹爪的从臂无需实现空方法。

继承 `GripperActuator`：

| 方法 | 职责 |
|---|---|
| `connect()` | 连接、初始化并使能夹爪。 |
| `send_normalized(opening)` | 接收 `0.0`（闭合）到 `1.0`（张开）的目标；成功返回 `True`。 |
| `disable()` | 退出时使夹爪失能。 |
| `disconnect()` | 关闭串口、网络或其他资源。 |

```python
from teleop_sdk.interfaces import GripperActuator


class MyFollowerGripper(GripperActuator):
    def connect(self) -> None:
        self.gripper = vendor_sdk.connect(self.address)
        self.gripper.enable()

    def send_normalized(self, opening: float) -> bool:
        opening = max(0.0, min(1.0, opening))
        position_mm = self.close_mm + opening * (self.open_mm - self.close_mm)
        return self.gripper.set_position(position_mm) == 0

    def disable(self) -> None:
        self.gripper.disable()

    def disconnect(self) -> None:
        self.gripper.close()
```

新夹爪应创建自己的配置类，例如 `MyGripperConfig`，其中包含：

- 连接参数：IP、串口、波特率、CAN ID、设备地址。
- 行程标定：全闭和全开对应的位置或角度。
- 安全限制：最大位置、速度、力、扭矩、电流或温度限制。
- 控制参数：位置 PID、阻抗刚度、阻尼或力控参数。
- 厂商协议参数：控制模式、寄存器、报文 ID 等。

`GloriaMGripperFollower` 与 `GloriaMGripperConfig` 是一个参考实现：它将归一化开合
量映射为位置误差，再以 MIT 模式输出限幅扭矩。夹爪 SDK、串口或连接失败时，
控制器会禁用夹爪，六轴从臂遥操继续运行。

## 4. 装配与验证

不带夹爪：

```python
controller = TeleopController(leader, follower, TELEOP_CONFIG)
```

示教臂支持联合读取，并使用从端夹爪：

```python
controller = TeleopController(
    leader, follower, TELEOP_CONFIG, gripper=follower_gripper
)
```

示教端夹爪输入与示教臂分离：

```python
controller = TeleopController(
    leader,
    follower,
    TELEOP_CONFIG,
    gripper=follower_gripper,
    leader_gripper=leader_gripper,
)
```

接入真实硬件前，先使用 `DryRunFollower` 确认示教端角度单位、轴顺序、轴方向和
夹爪开合方向；然后使用 `MockFollower` 检查控制器行为，最后再连接真实从臂并设置
实际关节限位、速度、加速度和夹爪安全参数。
