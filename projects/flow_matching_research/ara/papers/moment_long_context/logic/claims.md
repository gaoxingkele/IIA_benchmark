# Towards Long-Context Time Series Foundation Models With A Handful Of Additional Parameters

## C01

**Statement**: 所登记全文对应独立校验的来源及指定页范围。

**Conditions**: 来源版本、SHA256 和页范围匹配。

**Falsification criteria**: 文件哈希改变、标题/章节不符、页码越界。

**Proof**: E01

**Status**: source_verified

## C02

**Statement**: Infini-ChannelMixer 引入可压缩记忆，以少量额外参数增强 MOMENT 的长上下文及变量交互。

**Conditions**: 这是原文机制描述，未声称完成理论证明或数值复现。

**Falsification criteria**: 锚点不能定位对应机制，或任务/机制被版本核对推翻。

**Proof**: E02

**Status**: source_reported

## C03

**Statement**: 原文七张表格的702个标量目标已转录并校验（正文72、附录630），对应原始数据及缓存准备有独立证据；仍未取得可核实的原MOMENT-Tiny/ICM预训练检查点，完整一轮预训练与全部下游预测/SVM尚未完成。完整输入规划保留2162个候选及2086个原门槛入选来源，322个缺失值来源的有限输入对照仍需实际准备与闭合；不能把原文目标、数据准备或官方MOMENT零样本异常检测复核当作本论文性能复现。

**Conditions**: 对齐本论文的数据实体、通道分组、掩码、预训练预算、预测MSE/MAE、SVM准确率及不确定性定义；异常检测阈值/PA属于单独的任务适配。

**Falsification criteria**: 多种子差异超出预先设定的等效边界，或原数据/协议无法对齐。

**Proof**: E02, E03

**Status**: pending_experiment

## C04

**Statement**: 对可适配检测的方法，在共同协议下的相对性能是待检验问题。

**Conditions**: 先通过任务适配门禁；基础/数据参考只参与设计，不进入检测排行。

**Falsification criteria**: 严格协议下性能下降或排序变化，或评分使用不可用的测试标签。

**Proof**: E04

**Status**: pending_experiment

执行证据更新（C03状态仍为pending_experiment）：[原文702项目标与原始数据审计](../../../results/2026-10-15/MOMENT_ICM_primary_scope_and_raw_data_v1/README.md)、[四类输入完整一轮规划](../../../results/2026-10-15/MOMENT_ICM_full_PILE_paired_epoch_plan_tested_and_registered_v1/README.md)。Eq.(1)–(2)的所有通道压缩记忆、ELU+1及每头sigmoid门控与现有组件可逐项对应；历史作者插入、初始化和完整训练配方仍不以此认证。正文最大8通道抽样与附录D分组/余量过采样的描述差异保持为不同数据配方。
