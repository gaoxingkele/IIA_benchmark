# Pi-transformer: A prior-informed dual-attention model for multivariate time-series anomaly detection

研究任务：`time_series_anomaly_detection`；方法族：`prior_dual_attention`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：镜像源码的CPU构造、phase reshape、causal mask与prior支持四项修复及single-head配置已物化；作者现仓库仍仅README/LICENSE，完整期刊等价性待核。
