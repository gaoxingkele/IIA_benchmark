# Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models

研究任务：`imputation_and_forecasting`；方法族：`state_space_diffusion`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：对齐随机缺失与连续块缺失；S4/CUDA 依赖及窗口上下文会改变计算预算。
