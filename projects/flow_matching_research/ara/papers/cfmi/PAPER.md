# CFMI: Flow Matching for Missing Data Imputation

研究任务：`imputation`；方法族：`conditional_fm`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：区分每数据集训练和零样本时序；PhysioNet/USHCN 的 CRPS、样本数和归一化沿用已冻结数值参考。
