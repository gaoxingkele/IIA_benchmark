# Flow matching-based online imputation of chemical processes under arbitrary missing scenarios for enhanced process monitoring

代码、主消融或组件已按原文还原；仍缺：Exact author code not located; branch reduction/separability and time embedding wiring are local choices.；Algorithm 1 line 6 omits factor t on z1; implementation follows explicit Eq.7 rather than that inconsistent line.；TEP simulation IDs, HP private data, pre-trained DCNN/SOFTS teacher pipelines, mask generator, preprocessing and exact paper hyperparameters not completed.；Online windows must contain only data available at deployment timestamp; LPD interpolates within supplied observed window.；Required frozen downstream teacher enforced when joint distillation weight > 0.；sigma/prior variance/loss weights here are explicit development defaults, not certified paper-best values.

代码来源、提交、模型 entrypoint 和状态见 implementation_mapping.json。原始代码保留；后续修正存独立副本、补丁和行为检验，不能覆盖作者快照。

单独更新的[代码及消融覆盖审计](code_coverage.json)记录源码树哈希、已定位入口、消融原文页码和未完成项。[本篇方法及对比记录](method_inventory.json)绑定原文页码、代码候选与配置；由 scripts.flow_matching.refresh_paper_method_inventory 生成。先运行 scripts.flow_matching.audit_code_coverage 更新审计；不能由源码存在推断全部方法已实现。
