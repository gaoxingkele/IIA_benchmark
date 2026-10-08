# SensitiveHUE: Multivariate Time Series Anomaly Detection by Enhancing the Sensitivity to Normal Patterns

研究任务：`time_series_anomaly_detection`；方法族：`heteroscedastic_sensitivity`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：官方源码及CPU模型前后向已验证；Table4九行损失/结构变体已实现，论文公式与发布源码的均值缩放差异分开记录；完整训练及数值对齐待验收。
