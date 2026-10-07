# Time-Gated Multi-Scale Flow Matching for Time-Series Imputation

对齐左右上下文、gap-only 损失、ODE 步数与 DC 投影；属于插补而非原生检测。

代码来源、提交、模型 entrypoint 和状态见 implementation_mapping.json。原始代码保留；后续修正存独立副本、补丁和行为检验，不能覆盖作者快照。

单独更新的[代码及消融覆盖审计](code_coverage.json)记录源码树哈希、已定位入口、消融原文页码和未完成项。先运行 scripts.flow_matching.audit_code_coverage 生成；不能由源码存在推断全部方法已实现。
