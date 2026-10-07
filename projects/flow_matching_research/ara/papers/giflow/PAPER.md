# Spatiotemporal Imputation with Graph-Informed Flow Matching

研究任务：`imputation`；方法族：`graph_informed_cfm`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：固定 Air-36/AQI 图、缺失掩码及 EMA/checkpoint；本地前向预检不代表论文 MAE/CRPS 已复现。
