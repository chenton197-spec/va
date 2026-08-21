# run_w2 使用说明（HCX 双臂）

## 启动命令

在 `ct/va` 目录下运行：

```bash
PYTHONPATH=. python run_w2/run.py --deploy run_w2/deploy.yaml
```

脚本仅支持 `--deploy` 参数。`checkpoint`、`teleop_yaml` 等从 deploy 文件读取；`config` 可选，不设时回退到 checkpoint 内嵌 config。
`hcx_sdk` / `teleop_sdk` / `orbbec_sdk` 来自仓库内的 `teleop_project/`，不再依赖外部 `/home/a/Code/teleop_project`。
机械臂接口已切换为 `hcx_sdk`（`RobotClient` + 双臂 `move_joints`），不再使用法奥 FR3 SDK。

## 轨迹回放

回放由 `openarm_hcx_dual_arm_record.py` 采集的数据集（默认 `teleop_project/datasets/to_init`），逐点用 HCX `move_joints` 下发 `action` 与夹爪目标（到位确认与 `run.py` 一致）：

```bash
# 先干跑校验数据
PYTHONPATH=. python run_w2/replay.py --config run_w2/replay.yaml --dry-run

# 实机回放 episode 0（需能 import hcx_sdk / teleop_sdk）
PYTHONPATH=. python run_w2/replay.py --config run_w2/replay.yaml
```

常用覆盖参数：`--episode`、`--speed`、`--dataset`、`--display-cameras`、`--loop`。
`move_joints` 速度/加减速/容差在 `replay.yaml` 中配置。

## 最小检查清单

- 启动日志包含 `部署配置`、`加载 checkpoint`、`训练配置`。
- 日志出现 `策略已就绪`，表示模型权重已加载成功并完成 `policy.eval()`。
- 日志持续输出 `step=...` 且无 `HCX` 报警/使能错误、无维度/相机键错误，表示推理与实机循环已进入稳定阶段。

