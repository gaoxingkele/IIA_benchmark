# Flow matching-based online imputation of chemical processes under arbitrary missing scenarios for enhanced process monitoring

代码、主消融或组件已按原文还原；仍缺：Exact author code not located; branch reduction/separability and time embedding wiring are local choices.；Algorithm 1 line 6 omits factor t on z1; implementation follows explicit Eq.7 rather than that inconsistent line.；TEP simulation IDs, HP private data, pre-trained DCNN/SOFTS teacher pipelines, mask generator, preprocessing and exact paper hyperparameters not completed.；Online windows must contain only data available at deployment timestamp; LPD interpolates within supplied observed window.；Required frozen downstream teacher enforced when joint distillation weight > 0.；sigma/prior variance/loss weights here are explicit development defaults, not certified paper-best values.

冻结要求：作者版本与代码提交、数据实体/划分、训练专属归一化、掩码/窗口及回填、种子和预算、积分器/步数、分数聚合、阈值来源、点调整、指标尺度与不确定性。

可直接运行的本地模型配置及当前状态见 `src/code/implementation_mapping.json`；没有 entrypoint 的配置表示待适配，不能宣称方法已实现。
