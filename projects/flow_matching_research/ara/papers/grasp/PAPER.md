# Graph-Spectral Flow Matching for Multivariate Time Series Anomaly Detection

研究任务：`time_series_anomaly_detection`；方法族：`graph_spectral_fm`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：mTSBench 的实体数和特征数须与经典五数据集版本分开；作者代码未定位，须重建谱路径、图构造及速度加权。
