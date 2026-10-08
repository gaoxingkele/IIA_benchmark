# Towards Long-Context Time Series Foundation Models With A Handful Of Additional Parameters

代码、主消融或组件已按原文还原；仍缺：MOMENT patching/embedding/RevIN/head reused; T5-efficient-tiny sized random encoder prepared. Original MOMENT-Tiny/ICM checkpoints and exact pretraining not available here.；All context expansion baselines, 26 UEA SVM pipelines, one-epoch Time Series Pile and original forecasting datasets have not been run.；Original T5 relative bias/dropout retained, while compressed-memory aggregation excludes padded keys. Exact unpublished author ICM insertion semantics not independently verified.；Forecast and reconstruction/feature modes available; original SVM classification experiment remains separate.

代码来源、提交、模型 entrypoint 和状态见 implementation_mapping.json。原始代码保留；后续修正存独立副本、补丁和行为检验，不能覆盖作者快照。

单独更新的[代码及消融覆盖审计](code_coverage.json)记录源码树哈希、已定位入口、消融原文页码和未完成项。[本篇方法及对比记录](method_inventory.json)绑定原文页码、代码候选与配置；由 scripts.flow_matching.refresh_paper_method_inventory 生成。先运行 scripts.flow_matching.audit_code_coverage 更新审计；不能由源码存在推断全部方法已实现。
