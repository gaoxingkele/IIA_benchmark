# TAB: Unified Benchmarking of Time Series Anomaly Detection Methods

严格复现 pinned TAB 与 validation-only 新协议分别报告；ratio-grid best-test F1 和 pooled校准须披露，TAB标签本身不保证无泄漏。 原文单变量构建先保留至少一种方法AUC>0.85的序列并作标签/异常率筛选；该选择条件必须随数据版本披露。

代码来源、提交、模型 entrypoint 和状态见 implementation_mapping.json。原始代码保留；后续修正存独立副本、补丁和行为检验，不能覆盖作者快照。
