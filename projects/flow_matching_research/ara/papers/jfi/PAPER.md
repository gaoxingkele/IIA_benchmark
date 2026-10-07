# Flow matching-based online imputation of chemical processes under arbitrary missing scenarios for enhanced process monitoring

研究任务：`online_imputation_and_supervised_downstream`；方法族：`joint_flow_imputation`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：TEP 是 DTU v1 mode1 前20故障的特定仿真子集；精确仿真 ID 尚缺。HP 私有炼化数据未公开；算法1插值公式需与公式核对。
