# Flow Mismatching: Unsupervised Anomaly Detection via Velocity Discrepancies in Flow Matching Models

研究任务：`image_anomaly_detection`；方法族：`velocity_mismatch_ad`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：原文验证 MVTec-AD 与 VisA；时序版本须另设窗口映射、变量/时间聚合和正常训练协议。
