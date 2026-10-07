# DFM: Interpolant-free Dual Flow Matching

研究任务：`time_series_anomaly_detection`；方法族：`dual_fm`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：原文实验是 SMAP 无监督异常检测；旧 campaign 的 generation 标签不作为本工程的任务依据；冻结密度计算和积分容差。
