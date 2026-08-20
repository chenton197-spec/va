# HCX Python SDK API Reference

面向 HCX 机械臂关节角度控制和只读力矩反馈的 Python API 参考。本文只描述稳定的公开接口；`_hcx_native` 等以下划线开头的模块不是公开 API。

## 1. 约定与边界

| 项目 | 约定 |
| --- | --- |
| 角度 | 所有输入和反馈关节角度均为 **度**。 |
| `robot_id` | 控制器项目中配置的非负整数。SDK 不推断左臂/右臂映射，也不接受 `-1`。 |
| 连接 | 同一 Python 进程同时只能有一个已连接的 `RobotClient`。 |
| 平台 | 当前构建产物面向 Linux x86_64 和构建时的 CPython ABI。 |
| 范围 | 覆盖机械臂关节角度控制、角度反馈和原始力矩反馈；不包含轮组、IO、Modbus、CAN、力控或轨迹文件接口。 |

对象分类与关系：

```text
调用入口（使用者直接操作）
RobotClient(...)
  └── arm(robot_id) -> Arm
                         ├── move_joints(...)        -> MotionHandle
                         └── start_direct_servo(...) -> DirectServoSession

操作返回对象（由上述函数创建，不直接构造）
MotionHandle              # 查询或等待一条规划运动
DirectServoSession        # 持续设置目标并结束一次直伺服会话
```

| 对象类别 | 对象 | 获取方式 | 用途 |
| --- | --- | --- | --- |
| 调用入口 | `RobotClient` | 直接构造 | 管理控制器连接并取得机械臂对象。 |
| 调用入口 | `Arm` | `robot.arm(robot_id)` | 读取机械臂状态、执行控制和发起运动。 |
| 操作返回对象 | `MotionHandle` | `arm.move_joints(...)` 的返回值 | 查询或等待一条已提交的规划运动。 |
| 操作返回对象 | `DirectServoSession` | `arm.start_direct_servo(...)` 的返回值 | 在一次直接伺服期间设置最新关节目标，并显式停止下发。 |
| 值对象 | `MotionResult`、`DirectServoState` | 查询、等待或回调的返回值 | 承载状态快照，不用于发起控制。 |

`MotionHandle` 和 `DirectServoSession` 虽可从包中导入以便类型标注，但没有面向使用者的构造函数；应只使用入口函数返回的实例。

所有可能改变控制器状态的接口均在本文中标为“控制”或“高风险”。读取属性和反馈不会发送运动命令。

## 2. 导入与生命周期

```python
from hcx_sdk import RobotClient

robot = RobotClient(local_ip, remote_ip, port)
try:
    robot.connect(timeout_s=10.0)
    arm = robot.arm(robot_id)
    angles_deg = arm.joint_angles()  # 只读反馈
finally:
    robot.close()
```

`close()` 必须在 `finally` 中调用。它会停止本客户端尚未结束的直伺服软件下发并释放底层 SDK 资源；它不是物理急停。

华成底层库在网络异常时可能在 `init_data()` 内部持续重试。建议真实控制器程序为 `connect()` 传入有限的 `timeout_s`。等待过程可由 `Ctrl+C` 打断；超时或中断不能安全地强行终止厂商调用。此时 SDK 会保留一个守护线程等待厂商调用返回，若迟后连接成功会立即关闭该连接。在它返回前，同一进程不得再次调用 `connect()`；若一直不返回，请按 `Ctrl+C` 退出并重启 Python 进程。

## 3. 调用入口 API

本节中的对象是使用者直接持有和调用的控制入口。先创建并连接 `RobotClient`，再通过它获取 `Arm`。

### 3.1 `RobotClient`

#### 3.1.1 构造函数

```python
RobotClient(local_ip: str, remote_ip: str, port: int)
```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `local_ip` | `str` | 本机与控制器通信使用的 IP 地址。 |
| `remote_ip` | `str` | 控制器 IP 地址。 |
| `port` | `int` | 控制器端口，范围 `1..65535`。 |

无效端口会抛出 `ValueError`。构造函数不建立连接，必须随后调用 `connect()`。

#### 3.1.2 方法与属性

| 成员 | 返回 | 类型 | 说明 |
| --- | --- | --- | --- |
| `connect(timeout_s=None)` | `RobotClient` | 控制 | 初始化底层 SDK 并验证控制器链路；重复调用安全。连接失败或超时抛出 `ConnectionStateError`。 |
| `close()` | `None` | 控制 | 关闭连接并释放资源；重复调用安全。若存在直伺服会话，会先停止其软件下发。 |
| `connected` | `bool` | 只读 | Python 客户端和底层 SDK 都处于连接状态时为 `True`。 |
| `arm(robot_id)` | `Arm` | 只读 | 获取指定机器人控制对象。`robot_id` 必须为非负整数，且必须已连接。 |
| `active_alarms` | `tuple[str, ...]` | 只读 | 当前活动报警文本；仅读取，不清除报警。 |
| `clear_alarms()` | `None` | 控制 | 请求控制器清除当前报警。必须先由现场人员排除报警原因；控制器拒绝时抛出 `MotionRejectedError`。 |
| `hmi_detached` | `bool` | 只读 | 当前是否已处于脱离示教器状态。 |
| `soft_emergency_stop_normal` | `bool` | 只读 | 软急停状态是否正常；`False` 表示不可作为使能前的正常状态。 |
| `ethercat_master_operational(master_index)` | `bool` | 只读 | 读取 EtherCAT 主站的 OP 状态。仅 EtherCAT 伺服可调用，`master_index` 只能是 `0` 或 `1`。 |
| `detach_hmi()` | `None` | 控制 | 请求控制器脱离示教器。仅在现场确认未接示教器且会物理拔除时调用；控制器拒绝时抛出 `MotionRejectedError`。 |
| `set_global_enable(enabled)` | `None` | 控制 | 设置控制器全局机器人使能。控制器拒绝时抛出 `MotionRejectedError`。 |
| `global_enabled` | `bool` | 只读 | 获取控制器全局机器人使能状态。 |

```python
robot.connect(timeout_s=10.0)
robot.set_global_enable(True)   # 控制器状态变更
arm = robot.arm(robot_id)
```

`connect()` 的参数：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `timeout_s` | `float \| None` | `None` | 主线程等待连接结果的最长秒数，必须为正的有限数；`None` 表示不限时。超时只取消等待，不会并发调用厂商停止接口。 |

`connect()`、`arm()`、使能前诊断成员、`clear_alarms()`、`detach_hmi()`、`set_global_enable()` 和 `global_enabled` 在连接不可用时会抛出 `ConnectionStateError` 或其底层包装异常。

#### 3.1.3 使能前诊断与示教器状态

华成 V2.2.3 说明书的推荐顺序是：连接初始化后等待控制器稳定；在**未连接示教器**时请求脱离示教器并物理拔除；读取并处理报警；仅对 EtherCAT 伺服确认主站进入 OP；最后请求全局使能。SDK 提供显式的 `detach_hmi()` 和 `clear_alarms()` 调用，但不会自动清除报警、解除软急停或修改防护状态。

```python
import time

robot.connect()
time.sleep(2.0)

# 仅在现场确认未接示教器且已物理拔除时调用。
robot.detach_hmi()

if robot.active_alarms:
    # 必须先由现场人员排除报警原因，例如释放物理急停和恢复安全回路。
    robot.clear_alarms()
    if robot.active_alarms:
        raise RuntimeError(robot.active_alarms)
if not robot.soft_emergency_stop_normal:
    raise RuntimeError("控制器处于软急停状态")
if not robot.ethercat_master_operational(0):  # 仅 EtherCAT 伺服
    raise RuntimeError("EtherCAT 主站 0 尚未进入 OP")

robot.set_global_enable(True)
```

`hmi_detached == False` 并不单独证明异常，因为控制器也可能正在连接示教器。若现场未连接示教器，说明书要求调用 `detach_hmi()` 后物理拔除示教器，否则控制器可能报告急停而无法使能。

`soft_emergency_stop_normal == False` 时，必须由现场人员确认急停原因和安全条件；本 SDK 不提供自动解除软急停的行为。`active_alarms` 非空时，必须先排除报警原因，再显式调用 `clear_alarms()` 或使用控制器现场流程清除报警。报警未消失时不得继续使能或运动。

### 3.2 `Arm`

通过 `RobotClient.arm(robot_id)` 获取 `Arm`，不要直接构造。所有 `Arm` 接口均绑定到创建它的 `robot_id`。

#### 3.2.1 状态与反馈

| 成员 | 返回 | 类型 | 前置条件与行为 |
| --- | --- | --- | --- |
| `robot_id` | `int` | 只读 | 该对象绑定的控制器机器人 ID。 |
| `joint_angles()` | `tuple[float, ...]` | 只读 | 读取当前关节反馈，单位为度；每个数值四舍五入到小数点后三位，同时刷新轴数缓存。无反馈或读取失败时抛出 `ConnectionStateError`。 |
| `joint_torque_feedback()` | `tuple[int, ...]` | 只读 | 读取全轴原始力矩反馈；不切换运行模式，也不下发控制命令。厂商资料未定义物理单位，不得视为 `N·m`。无反馈、轴数不一致或读取失败时抛出 `ConnectionStateError`。 |
| `axis_count` | `int` | 只读 | 轴数。首次读取会调用 `joint_angles()` 获取反馈。 |
| `enabled` | `bool` | 只读 | 单臂使能状态。 |
| `protection_enabled` | `bool` | 只读 | 单臂防护状态。 |

#### 3.2.2 控制状态

| 方法 | 签名 | 类型 | 行为 |
| --- | --- | --- | --- |
| 单臂使能 | `set_enabled(enabled: bool) -> None` | 控制 | 设置该机械臂使能。控制器拒绝时抛出 `MotionRejectedError`。 |
| 防护 | `set_protection(enabled: bool, *, confirm_unsafe: bool = False) -> None` | 高风险 | 将 `enabled` 原样传给厂商 `setRobotProtectStatus`；`confirm_unsafe` 仅为旧调用方兼容保留，不参与判断。 |
| 暂停 | `pause() -> None` | 控制 | 请求控制器使该机械臂减速暂停。 |
| 恢复 | `resume() -> None` | 控制 | 从暂停状态恢复。 |
| 清路 | `clear_route(*, emergency_stop: bool = True) -> None` | 控制 | 清除控制器路径。`emergency_stop` 原样传递给厂商接口。 |

上述方法均要求连接可用；控制器拒绝时抛出 `MotionRejectedError`。

> 关闭防护或清路属于现场安全操作。SDK 不会在运动或直伺服时自动修改防护状态。

#### 3.2.3 规划关节运动

```python
arm.move_joints(
    angles_deg: Iterable[float],
    *,
    interrupt: bool = False,
    acceleration_seconds: float | None = None,
    deceleration_seconds: float | None = None,
    speed_ratio: float | None = None,
    smooth: int = 1,
    wait: bool = False,
    timeout_s: float | None = None,
) -> MotionHandle
```

通过厂商 `moveJoints2` 执行关节空间规划运动。

**返回：** [`MotionHandle`](#41-motionhandle) 操作返回对象。通过该对象查询或等待本次运动；不要直接构造它。

| 参数 | 约束 | 说明 |
| --- | --- | --- |
| `angles_deg` | 长度必须等于 `axis_count`；全部为有限数值 | 目标关节角度，单位为度。发送前由控制器检查关节限位。 |
| `interrupt` | `bool` | 传递给厂商运动接口的插补中断标志。 |
| `acceleration_seconds` | `None` 或 `0.1..1.0`，不能为 `0` | 加速时间。`None` 时向底层传递 `0.0`。 |
| `deceleration_seconds` | `None` 或 `0.1..1.0`，不能为 `0` | 减速时间。`None` 时向底层传递 `0.0`。 |
| `speed_ratio` | `None` 或 `(0, 1]` | 速度比例。`None` 时向底层传递 `0.0`。 |
| `smooth` | 整数 `0..9` | 平滑等级。 |
| `wait` | `bool` | 为 `True` 时，方法返回前等待运动完成。 |
| `timeout_s` | `None` 或非负有限秒数 | 仅在 `wait=True` 时传给 `MotionHandle.wait()`；`None` 表示无限等待。 |

发送前检查连接状态、控制器链路、活动报警、全局使能和单臂使能。检查失败时不会发送运动命令。

**可能抛出：** `ConnectionStateError`、`AlarmActiveError`、`MotionRejectedError`、`JointLimitError`、`MotionTimeoutError`（仅同步等待超时）、`TypeError`、`ValueError`。

```python
motion = arm.move_joints(
    target_angles_deg,
    acceleration_seconds=0.3,
    deceleration_seconds=0.3,
    speed_ratio=0.1,
)
result = motion.wait(timeout_s=15.0)
```

多个机械臂需要尽快并行执行时，分别调用各自的 `move_joints()`，先保存全部返回的 `MotionHandle`，再逐个等待。不要在第一条运动的 `wait()` 返回后才提交第二条。两次调用不是控制器级的原子或严格同步起动；必须由应用程序确认双臂轨迹不会互相干涉。

```python
left_motion = left_arm.move_joints(left_target_angles_deg)
right_motion = right_arm.move_joints(right_target_angles_deg)

left_result = left_motion.wait(timeout_s=15.0)
right_result = right_motion.wait(timeout_s=15.0)
```

#### 3.2.4 直接关节伺服

```python
arm.start_direct_servo(
    *,
    rate_hz: int | None = None,
    watchdog_s: float | None = None,
    confirm_unsafe: bool | None = None,
) -> DirectServoSession
```

创建基于厂商 `PluseToServo` 的直接关节伺服会话。该接口绕过厂商运动规划器；
原生扩展只把每次 Python 调用一对一地转发给厂商 SDK，不保存目标、不创建发送
线程，也不做插值、限速或轨迹规划。

**返回：** [`DirectServoSession`](#42-directservosession) 操作返回对象。通过该对象持续更新目标并结束本次会话；不要直接构造它。

| 参数 | 约束 | 说明 |
| --- | --- | --- |
| `rate_hz` | 任意值或 `None` | 旧调用方的元数据，不影响厂商调用。 |
| `watchdog_s` | 任意值或 `None` | 旧调用方的元数据，不创建 Python 看门狗。 |
| `confirm_unsafe` | 任意值或 `None` | 旧调用方兼容参数，不参与判断。 |

启动仅检查 Python 连接对象仍可用；不读取链路、报警、使能或关节限位，也不限制同一 `robot_id` 的会话数量。对于新代码，推荐直接使用 `Arm.pluse_to_servo(angles_deg) -> bool`，其方法名、参数单位和返回值与厂商 `RobotManager::PluseToServo` 对齐。

需要高频重发、插值、滤波或限速时，应在 Python 应用或适配器中实现自己的发送
线程。不同 `robot_id` 可以由不同 Python 线程同时调用 `set_target()`；原生
`pluse_to_servo` 调用期间会释放 GIL，但厂商库是否支持并发调用仍必须由现场验证。

**可能抛出：** `ConnectionStateError`、`DirectServoFault`（仅对已经显式 `stop()` 的兼容会话）、`TypeError`。

> 该接口不自动关闭防护，也不提供物理急停。只能在隔离、具备独立急停和硬件保护的安全单元内使用。

## 4. 操作返回对象 API

本节对象只由调用入口的函数返回，用于后续观察、等待或结束已经发起的操作。它们不是新的控制入口，也不应由使用者直接构造。

### 4.1 `MotionHandle`

**来源：** `Arm.move_joints()` 的返回值。

表示一条已被控制器接受的规划运动。

| 成员 | 返回 | 说明 |
| --- | --- | --- |
| `robot_id` | `int` | 目标机器人 ID。 |
| `sequence` | `int` | 厂商 `moveJoints2` 返回的运动序号。 |
| `done` | `bool` | 已完成或已取消时为 `True`。 |
| `succeeded` | `bool | None` | 未完成为 `None`；成功为 `True`；取消或失败为 `False`。 |
| `wait(timeout_s=None)` | `MotionResult` | 等待完成回调。`None` 表示无限等待；超时抛出 `MotionTimeoutError`。 |
| `add_done_callback(callback)` | `None` | 注册完成回调。回调在 Python 守护线程中执行。 |

`timeout_s` 必须为非负有限秒数或 `None`。未知序号、状态读取失败或等待失败会抛出 `MotionRejectedError` 或 `HcxSdkError`。

回调签名：

```python
def on_motion_done(result: MotionResult) -> None:
    print(result.sequence, result.succeeded)

motion.add_done_callback(on_motion_done)
```

回调内部抛出的异常只会被记录，不会回传给调用者。

### 4.2 `DirectServoSession`

**来源：** `Arm.start_direct_servo()` 的返回值。

表示一次已经启动的直接关节伺服会话。必须显式调用 `stop()` 结束软件下发。

| 成员 | 返回 | 说明 |
| --- | --- | --- |
| `set_target(angles_deg)` | `bool` | 立即向厂商 SDK 下发一个目标点，并返回厂商结果。 |
| `state` | `DirectServoState` | 当前会话状态快照。 |
| `stop()` | `None` | 停止软件侧发送并释放会话；幂等。 |

#### 4.2.1 `set_target(angles_deg)`

`angles_deg` 以度为单位。每次调用恰好触发一次厂商 `PluseToServo` 调用，并将厂商的 `bool` 返回值原样交给调用方；不会重发上一次目标，也不存在 `set_trajectory()` 或 `heartbeat()` 接口。

桥接层不检查轴数、关节限位、发送间隔、使能、报警或保护状态，也不会在一次 `false` 后停止后续调用。调用方负责自己的循环、算法和现场安全责任。`rate_hz` 与 `watchdog_s` 仅为旧会话调用形式保留，不影响厂商调用。

#### 4.2.2 `state: DirectServoState`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `running` | `bool` | Python 兼容会话是否被 `stop()` 关闭。 |
| `faulted` | `bool` | 始终为 `False`；厂商失败由 `set_target()` 的返回值表示。 |
| `sent_count` | `int` | 返回 `True` 的厂商调用次数。 |
| `error` | `str | None` | 最近一次厂商返回 `false` 的文本，仅供观察。 |
| `axis_count` | `int` | 不由会话探测，固定为 `0`。 |

#### 4.2.3 `stop()` 的安全语义

`stop()` 仅关闭该 Python 兼容会话；不会发送物理停止、急停、暂停、清路或防护状态变更命令。直接调用 `Arm.pluse_to_servo()` 不受会话状态影响。

```python
import time

servo = arm.start_direct_servo(
    rate_hz=100,
    watchdog_s=0.1,
    confirm_unsafe=True,
)
try:
    while keep_running:
        target_angles_deg = make_target_in_python()
        servo.set_target(target_angles_deg)
        time.sleep(1.0 / 250.0)
finally:
    servo.stop()
```

## 5. 值对象 / 数据模型

值对象仅表示某一时刻的结果或状态快照，不承担控制职责，也不应直接构造。

### 5.1 `MotionResult`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `robot_id` | `int` | 目标机器人 ID。 |
| `sequence` | `int` | 运动序号。 |
| `succeeded` | `bool` | 完成回调报告的成功状态。 |
| `cancelled` | `bool` | 该运动是否取消，默认 `False`。 |

### 5.2 `DirectServoState`

字段定义见 [4.2 `DirectServoSession`](#42-directservosession) 的 `state` 小节。该数据类为不可变快照。

## 6. 异常

所有业务异常继承自 `HcxSdkError`，后者继承 `RuntimeError`。

| 异常 | 触发场景 |
| --- | --- |
| `ConnectionStateError` | SDK 未连接、控制器链路异常、连接初始化失败或反馈读取失败。 |
| `AlarmActiveError` | 控制器存在活动报警，规划运动被阻止。 |
| `MotionRejectedError` | 控制器拒绝使能、暂停、清路、防护或规划运动请求；也可表示未知运动序号。 |
| `MotionTimeoutError` | 未在指定时间内收到规划运动完成回调。 |
| `JointLimitError` | 目标角度超出控制器配置限位，或读取到无效限位配置。 |
| `SafetyConfirmationError` | 旧版本兼容异常类型；当前桥接不再自动触发该安全确认。 |
| `DirectServoFault` | 已显式 `stop()` 的 Python 兼容会话又被调用。 |

参数类型、长度、范围或有限性不满足约束时，接口会抛出标准 `TypeError` 或 `ValueError`。

## 7. 公开导入

```python
from hcx_sdk import (
    AlarmActiveError,
    Arm,
    ConnectionStateError,
    DirectServoFault,
    DirectServoSession,
    DirectServoState,
    HcxSdkError,
    JointLimitError,
    MotionHandle,
    MotionRejectedError,
    MotionResult,
    MotionTimeoutError,
    RobotClient,
    SafetyConfirmationError,
)
```
