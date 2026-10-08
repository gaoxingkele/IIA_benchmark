# Conditional Flow Matching for Time Series Modelling

## C01

**Statement**: 所登记全文对应独立校验的来源及指定页范围。

**Conditions**: 来源版本、SHA256 和页范围匹配。

**Falsification criteria**: 文件哈希改变、标题/章节不符、页码越界。

**Proof**: E01

**Status**: source_verified

## C02

**Statement**: 以 BB-Flows 的线性高斯边际插值或 GP-Flows 的状态/导数联合高斯后验形成轨迹条件速度目标。

**Conditions**: 这是原文机制描述，未声称完成理论证明或数值复现。

**Falsification criteria**: 锚点不能定位对应机制，或任务/机制被版本核对推翻。

**Proof**: E02

**Status**: source_reported

## C03

**Statement**: 原文登记 6 条结果，具体数据集、值和页码见 evidence/tables/reported_results.json；其本地等效性尚待原协议重复实验。 代码、主消融或组件已按原文还原；仍缺：author formula interpretation；exact simulation split/count version；author network and dopri5 adjoint NODE baseline；full five-seed MSE experiments

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
