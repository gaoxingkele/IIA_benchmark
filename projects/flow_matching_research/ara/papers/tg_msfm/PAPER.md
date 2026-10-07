# Time-Gated Multi-Scale Flow Matching for Time-Series Imputation

研究任务：`imputation`；方法族：`time_gated_multiscale_fm`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：对齐左右上下文、gap-only 损失、ODE 步数与 DC 投影；属于插补而非原生检测。
