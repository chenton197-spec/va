"""robotfm：可扩展的机器人学习框架（Flow Matching 为核心）。

本包提供从数据采集、数据存储、策略训练到仿真评估的完整链路。
设计目标是：仿真（PushT）与真实机器人共用同一套数据协议和策略接口。

主要子模块：
- types:     统一观测 / 步进结果 / 数据集元信息
- config:    YAML 配置加载
- envs:      环境抽象（仿真 / 真机）
- data:      数据读写、Dataset、归一化统计
- collect:   遥操作采集循环
- policies:  Flow Matching 策略（见 policies/ARCHITECTURE.md）
- train/eval: 训练与评估入口
- rl:        强化学习占位（Phase 3）
"""

__version__ = "0.1.0"
