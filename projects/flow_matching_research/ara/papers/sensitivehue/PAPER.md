# SensitiveHUE: Multivariate Time Series Anomaly Detection by Enhancing the Sensitivity to Normal Patterns

研究任务：`time_series_anomaly_detection`；方法族：`heteroscedastic_sensitivity`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：需将测试校准和验证校准严格分轨；模型 NLL 与最终 F1 不是同一指标。 PDFp7分别报告best-threshold F1*与点调整F1*_PA；不能当验证集阈值结果。
