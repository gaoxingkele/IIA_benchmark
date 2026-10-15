# Towards Long-Context Time Series Foundation Models With A Handful Of Additional Parameters

研究任务：`forecasting_and_classification`；方法族：`compressive_channel_memory`。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

原文七张表的702个标量目标（正文72、附录630）已有逐项索引、原始数据检查及页码证据。原文Eq.(1)–(2)用ELU+1聚合所有通道KV、归一化查询，并在每头以sigmoid门控融合局部注意力；4层×4头的16个额外门控参数与已准备组件对应。原文窗口固定256，论文报告的上下文扩展包含变量维度。原作者完整插入/初始化语义及历史训练配方尚未认证。

数据准备已有157分类、9长预测、47 Monash来源索引；异常检测全部1949次原来源准备已尝试，1627个成功缓存独立闭合，其余322个已完整审计出残留非有限值，有限输入对照及有效目标掩码仍需实际闭合与完整训练集成。分类零样本评估的117个原数据集与预训练157候选来源保持不同范围。

新完整一轮数据规划保留2162候选、2086个原门槛入选来源，共80850189个case/group/window引用、39479个批次槽；包括322个待修复来源，没有通过删掉它们声称完整。附录D的8通道分组/余量过采样、分类casewise6/7和Monash原计数门槛均明确登记。batch2048/seed13来自相关来源YAML，论文未公开的精确预训练批次量不据此认证。磁盘规划保持旧RandomState抽样与尾批次，18项验证通过；实际完整索引生成及神经训练以各自成功证明为准。

原MOMENT-Tiny/ICM检查点、完整一轮预训练、所有上下文扩展基线、冻结编码器的预测头/β微调、117数据集SVM及严格预处理对照仍需完成。官方MOMENT small/base/large异常检测检查点的独立复核属于另一任务，不能替代本论文原预测/分类结果。

[原文目标与数据证据](../../../results/2026-10-15/MOMENT_ICM_primary_scope_and_raw_data_v1/README.md)，[最新完整一轮规划](../../../results/2026-10-15/MOMENT_ICM_full_PILE_paired_epoch_plan_tested_and_registered_v1/README.md)。完整研究目标尚未完成。
