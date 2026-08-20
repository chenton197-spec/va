# orbbec_sdk

`orbbec_sdk` 是独立的奥比中光多相机硬件层。它不依赖遥操作控制器或上游
LeRobot；`leobot_scripts.orbbec` 提供与本仓库采集器的桥接。

## 安装

安装共享依赖（其中固定了当前项目使用的奥比中光官方 wheel）：

```bash
python -m pip install -r requirements.txt
```

Linux 首次接入前还需按奥比中光 SDK 文档安装 udev 规则。不要把厂商 wheel 或设备地址
提交到仓库。

仓库已迁移 RSDT 使用的
`orbbec_sdk/config/OrbbecSDKConfig_casbot.xml`，其中 `PipelineFrameQueueSize=1`。
`OrbbecManager` 始终加载它；需要调整 SDK 行为时直接编辑这份 XML。

## 直接读取图像

启动后直接从相机对象读取 `numpy` 图像；RGB 是 `uint8 RGB`，深度是原始 `uint16 Y16`。

```python
from orbbec_sdk import (
    OrbbecCameraConfig,
    OrbbecManager,
)

manager = OrbbecManager(
    [
        OrbbecCameraConfig(
            name="left",
            serial_number="CAMERA_SERIAL",
            rgb_resolution=[1280, 720],
            depth_resolution=[848, 480],
            fps=30,
        )
    ],
)
manager.start()
try:
    camera = manager.camera("left")
    frame = camera.get_frame()
    if frame is not None:
        rgb = frame.rgb
        depth = frame.depth
finally:
    manager.stop()
```

`get_frame()` 非破坏性地返回当前最新帧；相机尚未出帧或已停止时才返回 `None`。因此预览、
采集和诊断不会互相“消费”图像。SDK 不生成伪彩色、灰度归一化或 OpenCV 图像。

## 窗口测试

在根目录 `teleop.yaml` 的 `orbbec.cameras` 中声明相机。每项填写 `name`、`serial_number`、
`mode`、`rgb_resolution` 和 `fps`；`mode: rgbd` 额外使用 `depth_resolution` 并进行软件对齐，
`mode: rgb` 不启动深度流。RGB 模式可以保留 `depth_resolution`，供以后切回 RGB-D 时复用。然后运行
`python -m examples.test_orbbec_camera`。已启用的流各使用一个窗口；不做伪彩色或深度归一化，按
`q` 或 Escape 关闭。

## 多相机启动

一份内置 XML 配置由一个 `OrbbecManager` 的唯一 SDK `Context` 全局加载，所有相机共享它。

`CameraMode.RGB` 配合 `OrbbecRGBSource`，`DEPTH` 配合 `OrbbecDepthSource`，`RGBD`
配合 `OrbbecRGBDSource`。每个 bridge 保留自己的“新帧”游标，因此多个 bridge 可以独立
读取同一相机。`DatasetRecorder` 只写机器人状态，不接收这些 bridge。需要数据采集时，
将 `OrbbecCameraAdapterConfig` 传给 `CameraProcessDatasetRecorder`。该适配器在采集子进程
独占 manager 和相机缓冲，支持 RGB 与 RGB-D source；由 `recording.master_camera` 指定的主相机按序
新 RGB 帧驱动数据行，避免固定采样 tick 重复主相机图像。

## 配置与数据

默认模式为 RGB-D。未提供分辨率和 `fps` 时，SDK 使用 XML/设备默认 profile；RGB-D 显式配置时必须同时
声明 RGB 分辨率、深度分辨率和 FPS。RGB 固定使用 `MJPG`，深度固定使用 `Y16`，与 RSDT
原驱动一致。RGB-D 默认软件 D2C 对齐，也可设置 `AlignmentMode.HARDWARE`；RGB 模式只启用彩色 profile。

深度以原始 `Y16 uint16` 交给采集器。默认将 SDK 的 `get_depth_scale()` 视为毫米比例并
乘 `0.001` 得到米制标尺；如设备契约不同，请设置 `depth_scale_to_meters` 并用已知距离
验证。每帧同时保存主机单调时间、奥比中光全局时间戳和 SDK 帧序号。

所有配置序列号必须在 `start()` 时连接。设备移除、取帧异常或连续超时会使该相机进入
`failed` 状态且不自动重连；通过 `camera.status` 与 `camera.last_error` 诊断后重建 manager。
