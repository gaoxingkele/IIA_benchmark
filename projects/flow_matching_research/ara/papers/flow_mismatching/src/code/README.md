# Flow Mismatching: Unsupervised Anomaly Detection via Velocity Discrepancies in Flow Matching Models

代码、主消融或组件已按原文还原；仍缺：No Flow Mismatching author repository located; architecture assembled from related cited Meta U-Net.；Exact StableAdamW with update RMS clipping training pipeline, MVTec/VisA splits, category training and checkpoint reproduction remain pending.；Compared Reflect, WT-Flow, TCCM, D-Flow, ReconFlow implementations from Appendix C are not represented by this detector primitive.；Time weighting switch is a local diagnostic supported by toy analysis; it is not a claim of all paper ablations.；K/T sweep list is a development grid; all paper Table 6 rows still require exact transcription.；Original task is image detection; do not report as TSAD performance.

代码来源、提交、模型 entrypoint 和状态见 implementation_mapping.json。原始代码保留；后续修正存独立副本、补丁和行为检验，不能覆盖作者快照。

单独更新的[代码及消融覆盖审计](code_coverage.json)记录源码树哈希、已定位入口、消融原文页码和未完成项。[本篇方法及对比记录](method_inventory.json)绑定原文页码、代码候选与配置；由 scripts.flow_matching.refresh_paper_method_inventory 生成。先运行 scripts.flow_matching.audit_code_coverage 更新审计；不能由源码存在推断全部方法已实现。
