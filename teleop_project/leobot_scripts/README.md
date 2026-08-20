# leobot_scripts

`leobot_scripts` 是面向 `teleop_sdk` 的采集 SDK。`DatasetRecorder` 包装任意
`FollowerArm`，固定频率写出状态、动作和夹爪反馈；相机由拥有相机生命周期的专用录制器
写入 RGB、原始深度和对应侧车数据。PNG 可无损保存 RGB，JPG 可逐帧压缩保存；不依赖上游 LeRobot 仓库。

从仓库根目录安装采集依赖：

```bash
python -m pip install -r requirements.txt
```

## 采集 Episode

创建控制器前包装从臂，开始和结束事件由键盘、GUI、ROS 等上层程序处理：

```python
from pathlib import Path

from leobot_scripts import DatasetRecorder, RecorderConfig, RecordingFollower

recorder = DatasetRecorder(
    RecorderConfig(
        root=Path("datasets/demo"),
        robot_type="my_follower",
        fps=30,
        image_storage="png",
    )
)
follower = RecordingFollower(my_follower, recorder)

# 在从臂至少成功接受过一条关节命令后：
recorder.start_episode("将物体移动到目标位置")
# 收到停止采集事件后：
recorder.stop_episode()
```

`action` 是从臂成功接受的最近关节目标，`observation.state` 是真实关节反馈，均为
角度制。输出时间戳固定为 `timestamp = frame_index / fps`。

## 相机进程隔离

`CameraProcessDatasetRecorder` 是通用的相机驱动录制器。FR3 的 ServoJ 控制仍按
`teleop.rate_hz` 运行；子进程独占相机 SDK 和磁盘写入，父进程按
`recording.numeric_sample_fps` 发送小型数值样本。
PNG/JPG 使用并行编码；RGB-D 相机的深度 Zarr 按 chunk 批量写入。短暂写入抖动由数值样本缓冲吸收，
不会阻塞机械臂控制。只有持续过载耗尽缓冲时才会跳过采样，跳过原因会写入 episode
的审计 JSONL。

一台相机时它自动成为主相机；多台相机时必须指定 `master_camera_name`。每张新的主相机
RGB 帧写入一行，并选择不晚于其捕获时间的最新机器人、夹爪和其他相机数据。相机帧不会
被预览或其他 source 全局消费。审计 JSONL 会记录主相机目标时间、数值样本年龄、各相机
捕获时间、源帧号和 `capture_age_ns`，用于检查时间对齐质量。

## 相机与深度

`DatasetRecorder` 不接收相机 source，也不轮询图像。相机录制器必须拥有相机 SDK、帧缓冲、
时间配对和写入生命周期。`CameraProcessDatasetRecorder` 只依赖
`CameraAdapterConfig`、`CameraAdapterSession`、`CameraFrameSource` 和可选的 `RGBDFrameSource`
通用契约；
奥比中光由 `OrbbecCameraAdapterConfig` 实现。接入其他相机时，只需实现同一适配层，
而不是修改录制器或向固定频率的 `DatasetRecorder` 添加图像参数。

相机适配器配置必须可被 Python 子进程序列化，并在 `open()` 中打开设备。会话需按相机名
暴露 RGB source；有深度时再实现 RGB-D source，并在 `close()` 中释放厂商资源。录制器不传递已打开的相机对象给子进程。

`meters_per_raw_unit` 必须是明确的“米/原始值单位”。奥比中光参考封装应在
`Y16` 缓冲区读取后、乘 `get_depth_scale()` 前取出 `raw`；其 BGR 图像需转换为 RGB。
请用已知距离验证设备标尺后再填入米制换算，不要将已缩放的浮点深度强转回 `uint16`。

## 输出与读取

PNG/JPG 模式（`image_storage="png"` 或 `image_storage="jpg"`）直接提交每帧 RGB，不会在结束 episode 时编码视频：

```text
images/chunk-000/observation.images.wrist/episode_000000/frame_000000.png
# 仅 mode: rgbd 时存在：
depth/chunk-000/observation.depth.wrist/episode_000000.zarr
```

PNG/JPG 模式的 `meta/info.json` 标记为 `leobot_image_sequence_v1`，Parquet 中
`observation.images.<name>.path` 是每帧图像的相对路径。因此它需要按路径读取图像，
不能直接交给期望 LeRobot v2.1 MP4 视频的读取器。
JPG 使用 `RecorderConfig.quality`（1-100，默认 75）；部署 YAML 对应 `recording.quality`。
同一 episode 的图像格式固定，但 PNG/JPG 可以作为不同 episode 追加到同一个 chunk。

RGB-D 模式下，Zarr 中的 `depth_raw[i]` 与 RGB 图像第 `i` 帧及 Parquet 的 `frame_index=i` 对应；
`meters_per_raw_unit[i]` 用于恢复米制深度，`0` 保留为无效值。静态格式说明、对齐关系
和可选内参写在 `meta/depth_sources.json`。
`source_timestamp_ns` 与 `source_frame_index` 记录相机 SDK 时间和源帧序号；RGB 的对应
时间写入审计 JSONL。

视频模式（`image_storage="video"`，默认值）继续写入标准路径：

```text
videos/chunk-000/observation.images.wrist/episode_000000.mp4
```

视频与逐帧 PNG/JPG 图像序列不能写入同一个数据集目录。

```python
import zarr

group = zarr.open_group(".../episode_000000.zarr", mode="r")
depth_m = group["depth_raw"][frame_index].astype("float32") * group["meters_per_raw_unit"][frame_index]
```

相机录制器应只写入完整 RGB 帧；RGB-D 模式还要求完整深度帧。缺失帧、缓冲溢出和时间配对缺口写入
`meta/recording_audit/episode_XXXXXX.jsonl`。

如果一个 episode 结束时没有产生任何完整帧，它不会占用正式 episode 编号。错误信息会列出各类
跳帧原因及次数，原始审计记录保存在 `meta/failed_recording_audit/`，用于区分相机缺帧、反馈超时、
夹爪反馈缺失等问题。

奥比中光多相机、共享 XML 配置与断连行为见 [orbbec_sdk/README.md](../orbbec_sdk/README.md)。
