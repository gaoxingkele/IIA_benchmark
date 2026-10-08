# Flow matching-based online imputation of chemical processes under arbitrary missing scenarios for enhanced process monitoring

## C01

**Statement**: 所登记全文对应独立校验的来源及指定页范围。

**Conditions**: 来源版本、SHA256 和页范围匹配。

**Falsification criteria**: 文件哈希改变、标题/章节不符、页码越界。

**Proof**: E01

**Status**: source_verified

## C02

**Statement**: 局部先验采样与混合卷积/通道注意力 UNet 学习插补速度，联合速度、目标重建和下游知识蒸馏损失。

**Conditions**: 这是原文机制描述，未声称完成理论证明或数值复现。

**Falsification criteria**: 锚点不能定位对应机制，或任务/机制被版本核对推翻。

**Proof**: E02

**Status**: source_reported

## C03

**Statement**: 该文数值尚未逐项转录；不能给出未核实的论文指标或本地等效性结论。 代码、主消融或组件已按原文还原；仍缺：Exact author code not located; branch reduction/separability and time embedding wiring are local choices.；Algorithm 1 line 6 omits factor t on z1; implementation follows explicit Eq.7 rather than that inconsistent line.；TEP simulation IDs, HP private data, pre-trained DCNN/SOFTS teacher pipelines, mask generator, preprocessing and exact paper hyperparameters not completed.；Online windows must contain only data available at deployment timestamp; LPD interpolates within supplied observed window.；Required frozen downstream teacher enforced when joint distillation weight > 0.；sigma/prior variance/loss weights here are explicit development defaults, not certified paper-best values.

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
