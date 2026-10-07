# DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection

研究任务：`time_series_anomaly_detection`；方法族：`contrastive_dual_attention`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：原作者阈值与 point adjustment 和严格验证阈值分轨；作为 MaelNet 组件不能等同整个集成。
