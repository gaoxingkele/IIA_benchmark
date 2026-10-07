# Flow Matching for Generative Modeling

研究任务：`generative_foundation`；方法族：`conditional_vector_field`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：免积分仅指训练目标；采样仍需 ODE，似然和异常分数不是自动提供。
