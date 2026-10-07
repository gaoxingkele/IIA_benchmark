# TAB: Unified Benchmarking of Time Series Anomaly Detection Methods

研究任务：`evaluation_protocol`；方法族：`tsad_benchmark`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：严格复现 pinned TAB 与 validation-only 新协议分别报告；ratio-grid best-test F1 和 pooled校准须披露，TAB标签本身不保证无泄漏。 原文单变量构建先保留至少一种方法AUC>0.85的序列并作标签/异常率筛选；该选择条件必须随数据版本披露。
