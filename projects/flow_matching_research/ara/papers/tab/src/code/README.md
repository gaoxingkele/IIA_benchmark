# TAB: Unified Benchmarking of Time Series Anomaly Detection Methods

严格复现 pinned TAB 与 validation-only 新协议分别报告；ratio-grid best-test F1 和 pooled校准须披露，TAB标签本身不保证无泄漏。 原文单变量构建先保留至少一种方法AUC>0.85的序列并作标签/异常率筛选；该选择条件必须随数据版本披露。

代码来源、提交、模型 entrypoint 和状态见 implementation_mapping.json。原始代码保留；后续修正存独立副本、补丁和行为检验，不能覆盖作者快照。

单独更新的[代码及消融覆盖审计](code_coverage.json)记录源码树哈希、已定位入口、消融原文页码和未完成项。[本篇方法及对比记录](method_inventory.json)绑定原文页码、代码候选与配置；由 scripts.flow_matching.refresh_paper_method_inventory 生成。先运行 scripts.flow_matching.audit_code_coverage 更新审计；不能由源码存在推断全部方法已实现。
