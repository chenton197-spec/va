# OpenArm -> HCX 双臂采集数据说明

本说明面向使用数据训练或分析的人员。一个 episode 对应一个 Parquet 文件，包含双臂目标、双臂实际位置、左右夹爪开度和三路 RGB 图像。

## 数据位置

默认数据集目录：

```text
datasets/openarm_hcx_dual_arm/
├── data/chunk-000/episode_000000.parquet
├── images/chunk-000/observation.images.head/...
├── images/chunk-000/observation.images.left_hand/...
├── images/chunk-000/observation.images.right_hand/...
└── meta/
```

每个 Parquet 文件是一段连续采集的 episode。每一行对应一张头部相机图像；采集频率约为 30 Hz，但实际时间间隔应以 `timestamp` 为准。

## 每行字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `index` | `int64` | 数据集全局帧编号。 |
| `episode_index` | `int64` | 当前 episode 编号。 |
| `frame_index` | `int64` | 当前帧在 episode 内的编号，从 0 开始。 |
| `timestamp` | `float32`，秒 | 当前头部图像的 episode 相对时间。 |
| `action` | `float32[14]`，度 | HCX 双臂的目标关节角度。 |
| `observation.state` | `float32[14]`，度 | HCX 双臂读取到的实际关节角度。 |
| `action.left_gripper` | `float32` | 左夹爪目标开度，`0.0` 为闭合，`1.0` 为张开。 |
| `observation.left_gripper` | `float32` | 左夹爪实际开度，范围 `0.0` 到 `1.0`。 |
| `action.right_gripper` | `float32` | 右夹爪目标开度，`0.0` 为闭合，`1.0` 为张开。 |
| `observation.right_gripper` | `float32` | 右夹爪实际开度，范围 `0.0` 到 `1.0`。 |
| `observation.images.head` | `{path, timestamp}` | 头部 RGB 图像的相对路径和时间。 |
| `observation.images.left_hand` | `{path, timestamp}` | 左手 RGB 图像的相对路径和时间。 |
| `observation.images.right_hand` | `{path, timestamp}` | 右手 RGB 图像的相对路径和时间。 |

## 双臂数组顺序

`action` 和 `observation.state` 的 14 个元素顺序固定：

```text
[左 J1, 左 J2, 左 J3, 左 J4, 左 J5, 左 J6, 左 J7,
 右 J1, 右 J2, 右 J3, 右 J4, 右 J5, 右 J6, 右 J7]
```

所有关节值都是**角度**，不是弧度。可按以下方式拆分：

```python
left_arm = joints[:7]
right_arm = joints[7:]
```

## 图像时间关系

- `head` 是当前行的主图像。
- `left_hand` 和 `right_hand` 是不晚于该头部图像的最近手部图像，因此它们的时间戳可能略早于行 `timestamp`。
- 图像字段中的 `path` 相对于数据集根目录。对于 JPEG/PNG 数据集，路径直接对应一张图像文件。

## 最小读取示例

```python
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

root = Path("datasets/openarm_hcx_dual_arm")
table = pq.read_table(root / "data/chunk-000/episode_000000.parquet")

row = 0
target_deg = np.asarray(table["action"][row].as_py(), dtype=np.float32)
actual_deg = np.asarray(table["observation.state"][row].as_py(), dtype=np.float32)

left_target_deg = target_deg[:7]
right_target_deg = target_deg[7:]
left_actual_deg = actual_deg[:7]
right_actual_deg = actual_deg[7:]

head = table["observation.images.head"][row].as_py()
head_image_path = root / head["path"]

print("time (s):", table["timestamp"][row].as_py())
print("left target (deg):", left_target_deg)
print("left actual (deg):", left_actual_deg)
print("head image:", head_image_path)
```

## 使用时注意

- `action` 是从臂目标，`observation.state` 是从臂实际反馈；两者的差值可用于分析跟踪误差。
- 数据不包含 OpenArm 主臂原始关节角度。
- `action` 不是 HCX 500 Hz 伺服线程的逐点指令，而是遥操作上游产生的最近目标。
- `meta/recording_audit/` 中的 JSONL 是可选诊断信息；出现缺帧、图像时间配对或反馈读取问题时再查看它。

可使用 [examples/test_parquet_collection_quality.py](examples/test_parquet_collection_quality.py) 对单个 Parquet 文件检查帧数、时间连续性、关节跳变和图像路径。
