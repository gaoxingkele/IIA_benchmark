# MOMENT: A Family of Open Time-series Foundation Models

研究任务：`time_series_foundation`；方法族：`masked_pretraining`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：预训练数据重叠与权重哈希须核对；zero-shot/微调及论文自己的异常协议分别报告。
