# Subsequence heterogeneity contrastive learning for time series anomaly detection

研究任务：`time_series_anomaly_detection`；方法族：`subsequence_contrastive`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：原文说阈值来自 validation，代码校准轨道须独立审计；Table2 是 Transformer/VAE，未找到此前所说 LSTM96.42 行。
