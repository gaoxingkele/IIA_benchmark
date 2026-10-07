# USAD: UnSupervised Anomaly Detection on Multivariate Time Series

研究任务：`time_series_anomaly_detection`；方法族：`adversarial_autoencoder`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：仓库已有稳定性消融不等于原始双解码器 USAD；原论文与 ablation 明确分名，不能替代复现。
