# TAB: Unified Benchmarking of Time Series Anomaly Detection Methods

严格复现 pinned TAB 与 validation-only 新协议分别报告；ratio-grid best-test F1 和 pooled校准须披露，TAB标签本身不保证无泄漏。 原文单变量构建先保留至少一种方法AUC>0.85的序列并作标签/异常率筛选；该选择条件必须随数据版本披露。

冻结要求：作者版本与代码提交、数据实体/划分、训练专属归一化、掩码/窗口及回填、种子和预算、积分器/步数、分数聚合、阈值来源、点调整、指标尺度与不确定性。

可直接运行的本地模型配置及当前状态见 `src/code/implementation_mapping.json`；没有 entrypoint 的配置表示待适配，不能宣称方法已实现。
