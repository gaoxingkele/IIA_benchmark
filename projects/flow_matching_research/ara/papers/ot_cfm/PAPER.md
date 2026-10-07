# Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport

研究任务：`generative_foundation`；方法族：`minibatch_ot_cfm`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：TorchCFM 的多个 objective 类是训练组件，不能按数量当作已经完成的检测器。
