# CrossAD: Time Series Anomaly Detection with Cross-scale Associations and Cross-window Modeling

研究任务：`time_series_anomaly_detection`；方法族：`cross_scale_cross_window`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：原作者对 SMAP/MSL 使用首个连续特征的轨道，不能与全变量版本混合排行。
