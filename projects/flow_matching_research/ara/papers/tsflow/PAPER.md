# Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting

研究任务：`generation_and_forecasting`；方法族：`gp_conditional_fm`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：预测误差或生成保真度不是异常 F1；转为检测器前冻结正常训练集及分数定义。
