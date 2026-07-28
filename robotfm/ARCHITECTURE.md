# robotfm 包结构总览

本文档说明 `va/robotfm/` 各子模块职责与数据流，便于从整体理解项目。策略算法细节见 [policies/ARCHITECTURE.md](policies/ARCHITECTURE.md)。

## 目录与职责

| 模块 | 文件 | 职责 |
|------|------|------|
| 根 | `types.py` | `Observation`、`StepResult`、`EpisodeMeta` |
| 根 | `config.py` | YAML 配置加载（含嵌套 `policy.rtc`） |
| 根 | `train.py` | Flow Matching 训练循环 |
| 根 | `eval.py` | 仿真闭环评估（可选 RTC ActionQueue） |
| `envs/` | `base.py`, `pusht.py`, `registry.py` | 环境抽象与 PushT 适配 |
| `data/` | `schema.py`, `writer.py`, `dataset.py`, `stats.py` | NPZ 数据协议与 Dataset |
| `collect/` | `loop.py`, `drivers/` | 遥操作采集 |
| `policies/` | `flow_matching.py`, `encoders.py`, `video_encoders.py`, `unet1d.py`, `rtc/` | 条件 FM（ResNet / SlowFast）+ 推理期 RTC |
| `rl/` | `base.py` | RL 占位（Phase 3） |

## 端到端数据流

```text
采集:  TeleopDriver + BaseEnv  ->  collect_demos  ->  episodes/*.npz + meta.json + stats.json
训练:  EpisodeDataset  ->  FlowMatchingPolicy.compute_loss  ->  checkpoint_*.pt
评估:  checkpoint  ->  sample_actions (+ RTC) + env.step  ->  success_rate
```

## RTC（推理期）

- 训练目标不变；仅 `sample_actions` / `eval` 可选启用
- 算法核：Euler 每步 `RTCProcessor.denoise_step`（标准 RTC inpainting 引导）
- 控制核：`ActionQueue` leftover + `inference_delay` merge（与 LeRobot 同构；当前为同步模拟）
- 配置：`policy.rtc.enabled`（默认 `false`）

## 扩展真机 checklist

1. 实现 `RealRobotEnv`（相机 + 臂驱动）
2. 在 `registry.make_env` 注册 `backend`
3. 实现对应 `TeleopDriver`
4. 新建 yaml：`cameras`、`state_dim`、`action_dim` 与硬件一致
5. 策略与训练脚本无需改代码（维度由 config/meta 驱动）
6. （可选）后续加异步推理线程；RTC 算法接口已就绪

## 推荐阅读顺序

1. 本文档 + `policies/ARCHITECTURE.md`
2. `types.py` -> `config.py`
3. `data/schema.py` -> `data/dataset.py`
4. `policies/flow_matching.py` -> `policies/rtc/`
5. `train.py` -> `eval.py`
