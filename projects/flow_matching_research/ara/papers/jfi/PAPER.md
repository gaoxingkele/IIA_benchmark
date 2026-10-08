# Flow matching-based online imputation of chemical processes under arbitrary missing scenarios for enhanced process monitoring

研究任务：`online_imputation_and_supervised_downstream`；方法族：`joint_flow_imputation`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：代码、主消融或组件已按原文还原；仍缺：Exact author code not located; branch reduction/separability and time embedding wiring are local choices.；Algorithm 1 line 6 omits factor t on z1; implementation follows explicit Eq.7 rather than that inconsistent line.；TEP simulation IDs, HP private data, pre-trained DCNN/SOFTS teacher pipelines, mask generator, preprocessing and exact paper hyperparameters not completed.；Online windows must contain only data available at deployment timestamp; LPD interpolates within supplied observed window.；Required frozen downstream teacher enforced when joint distillation weight > 0.；sigma/prior variance/loss weights here are explicit development defaults, not certified paper-best values.
