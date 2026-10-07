# Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series

研究任务：`time_series_anomaly_detection`；方法族：`graph_normalizing_flow`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：归一化流的似然评分与 FM 的速度失配评分分开；关注训练污染和实体聚合。
