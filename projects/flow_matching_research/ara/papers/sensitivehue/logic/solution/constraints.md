# SensitiveHUE: Multivariate Time Series Anomaly Detection by Enhancing the Sensitivity to Normal Patterns

官方源码及CPU模型前后向已验证；Table4九行损失/结构变体已实现，论文公式与发布源码的均值缩放差异分开记录；完整训练及数值对齐待验收。

冻结要求：作者版本与代码提交、数据实体/划分、训练专属归一化、掩码/窗口及回填、种子和预算、积分器/步数、分数聚合、阈值来源、点调整、指标尺度与不确定性。

可直接运行的本地模型配置及当前状态见 `src/code/implementation_mapping.json`；没有 entrypoint 的配置表示待适配，不能宣称方法已实现。
