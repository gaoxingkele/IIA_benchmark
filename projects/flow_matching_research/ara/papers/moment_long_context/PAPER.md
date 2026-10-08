# Towards Long-Context Time Series Foundation Models With A Handful Of Additional Parameters

研究任务：`forecasting_and_classification`；方法族：`compressive_channel_memory`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：代码、主消融或组件已按原文还原；仍缺：MOMENT patching/embedding/RevIN/head reused; T5-efficient-tiny sized random encoder prepared. Original MOMENT-Tiny/ICM checkpoints and exact pretraining not available here.；All context expansion baselines, 26 UEA SVM pipelines, one-epoch Time Series Pile and original forecasting datasets have not been run.；Original T5 relative bias/dropout retained, while compressed-memory aggregation excludes padded keys. Exact unpublished author ICM insertion semantics not independently verified.；Forecast and reconstruction/feature modes available; original SVM classification experiment remains separate.
