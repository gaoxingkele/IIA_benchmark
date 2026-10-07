# MaelNet: Memorable Anomaly Election Learning Detecting Anomalies in Time Series Data using Dual-Net Transformer

## C01

**Statement**: 所登记全文对应独立校验的来源及指定页范围。

**Conditions**: 来源版本、SHA256 和页范围匹配。

**Falsification criteria**: 文件哈希改变、标题/章节不符、页码越界。

**Proof**: E01

**Status**: source_verified

## C02

**Statement**: 快学习器使用 Anomaly Transformer 与 DCDetector，慢学习器使用掩码非平稳 Transformer，强化学习选择异常候选。

**Conditions**: 这是原文机制描述，未声称完成理论证明或数值复现。

**Falsification criteria**: 锚点不能定位对应机制，或任务/机制被版本核对推翻。

**Proof**: E02

**Status**: source_reported

## C03

**Statement**: 原文登记 2 条结果，具体数据集、值和页码见 evidence/tables/reported_results.json；其本地等效性尚待原协议重复实验。 奖励依赖真实标签，所属划分尚不清楚；Table II 三个数据集的最佳行实际是其他方法，不能声称五项皆由 MaelNet 获胜。

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
