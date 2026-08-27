# Chapter 1 — Overview

## 知识提取

工业报警是过程变量越界或离散状态变化后面向操作员的通知。书中把 alarm overloading 的主因归为四类：噪声/扰动导致的 chattering，报警变量配置不当，各变量独立设计忽略相互关系，以及异常沿物理连接传播。alarm lifecycle 分为配置、设计和移除/整治阶段。

八个核心研究问题可压缩为：是否需要配置、如何定优先级、如何识别错误配置、报警机制是否性能合格、如何压低 nuisance load、能否提前预警、如何找根因、操作员应采取何种动作。

## Benchmark 映射

- 任务必须至少区分报警生成、性能评价、根因、洪泛、预测和运维可视化。
- 数据 card 必须说明报警是原生事件还是由过程变量阈值派生，避免把合成标签误作真实 DCS 配置。
- 指标既评价安全侧漏报/延迟，也评价操作员负荷侧虚警/报警率；单一 accuracy 不够。
- 正式结果按 lifecycle 阶段说明：design-time、online operation 或 retrospective analysis。

证据：`01_overview.txt`，PDF physical pages 13–59。
