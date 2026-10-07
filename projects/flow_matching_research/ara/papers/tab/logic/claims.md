# TAB: Unified Benchmarking of Time Series Anomaly Detection Methods

## C01

**Statement**: 所登记全文对应独立校验的来源及指定页范围。

**Conditions**: 来源版本、SHA256 和页范围匹配。

**Falsification criteria**: 文件哈希改变、标题/章节不符、页码越界。

**Proof**: E01

**Status**: source_verified

## C02

**Statement**: 系统评估时间序列异常检测方法并提供可复用的数据与评价管线。

**Conditions**: 这是原文机制描述，未声称完成理论证明或数值复现。

**Falsification criteria**: 锚点不能定位对应机制，或任务/机制被版本核对推翻。

**Proof**: E02

**Status**: source_reported

## C03

**Statement**: 该文数值尚未逐项转录；不能给出未核实的论文指标或本地等效性结论。 严格复现 pinned TAB 与 validation-only 新协议分别报告；ratio-grid best-test F1 和 pooled校准须披露，TAB标签本身不保证无泄漏。 原文单变量构建先保留至少一种方法AUC>0.85的序列并作标签/异常率筛选；该选择条件必须随数据版本披露。

**Conditions**: 数据实体、掩码、预算、阈值、PA、聚合及不确定性定义相同。

**Falsification criteria**: 多种子差异超出预先设定的等效边界，或原数据/协议无法对齐。

**Proof**: E02, E03

**Status**: pending_experiment

## C04

**Statement**: 对可适配检测的方法，在共同协议下的相对性能是待检验问题。

**Conditions**: 先通过任务适配门禁；基础/数据参考只参与设计，不进入检测排行。

**Falsification criteria**: 严格协议下性能下降或排序变化，或评分使用不可用的测试标签。

**Proof**: E04

**Status**: pending_experiment
