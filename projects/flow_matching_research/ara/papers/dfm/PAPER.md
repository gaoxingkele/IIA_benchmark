# DFM: Interpolant-free Dual Flow Matching

研究任务：`time_series_anomaly_detection`；方法族：`dual_fm`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：代码、主消融或组件已按原文还原；仍缺：author U-Net and learnable prior；Dopri5；MLE-CNF and exact published FM/I-CFM U-Net baselines；author SMAP entity/feature protocol；threshold and adjustment protocol；objective degeneracy/bijectivity needs mathematical audit
