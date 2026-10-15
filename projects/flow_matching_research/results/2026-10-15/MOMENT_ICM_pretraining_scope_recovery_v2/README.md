# MOMENT/ICM：完整输入范围审计与流式内存修复

全部 2,441 个当前登记的 PILE 原始文件已重新校验 SHA256、Git blob SHA1、大小及读取前后状态，总计 8,705,622,732 字节；全部 47 个 Monash 文件均已逐值解析并完成训练长度统计。原数据没有覆盖。这里完成的是数据范围审计，神经网络预训练、推理、SVM及新增性能结果均为 0。

先前 v1 元数据任务在完成文件校验、处理 13 个 Monash 文件后，构建大型来源数据表时超过原定 6 GiB 进程树上限，被资源保护终止，没有最终成功证明。原失败回执和进度完整保留。v2 在新配置和新进程中逐个验证全部数值后丢弃数值对象，只保留长度；固定 pandas/NumPy 的 seed=13 行采样、训练比例 0.6、上下文 256，以及来源特殊长序列的分块和尾部丢弃规则。v2 未减少文件、样本或预算，准入与保护上限也没有降低；相关模型和正在运行的训练未修改。原数值对象的精确内存分配根因尚未单独证明。

两套共 28 项检查通过，其中 8 个真实小型来源文件的原解析器/数据类与流式实现的全部原有统计字段相同。最终 47 个文件均完成验证；大型文件没有再次构建完整数值表，因此不据此声称所有原数值表逐坐标重放。

相关 MOMENT 来源代码把 Monash 的 `length_timeseries` 设为拆分前的数据表行数，即序列条数。原筛选用该字段是否大于等于 256，选中 33/47 个文件、371,162 个训练条目。单独登记的时间长度对照判断训练序列最大长度是否大于等于 256，选中 28/47 个文件、278,699 个训练条目。它不要求每个条目长度均合格，也不等于原作者的已证实修复方案。完整逐文件数字在 CSV 中。

时间长度对照新增：australian_electricity_demand_dataset, bitcoin_dataset_without_missing_values, fred_md_dataset, nn5_daily_dataset_without_missing_values, pedestrian_counts_dataset, saugeenday_dataset, solar_10_minutes_dataset, us_births_dataset。

时间长度对照排除：car_parts_dataset_without_missing_values, covid_deaths_dataset, electricity_weekly_dataset, hospital_dataset, kaggle_web_traffic_weekly_dataset, m1_monthly_dataset, m3_monthly_dataset, m3_quarterly_dataset, m3_yearly_dataset, tourism_quarterly_dataset, tourism_yearly_dataset, traffic_weekly_dataset, vehicle_trips_dataset_without_missing_values。

例如 NN5 daily：来源字段为 111 条序列，训练有 66 条、各 791 个时间点；来源字段筛选将其排除，时间长度对照将其保留。两套筛选改变数据组成与一个 epoch 的数据预算，应分别报告，不能作为仅改变模型结构的消融。这里的训练条目只覆盖 Monash 分支，尚未证明整个多来源预训练集的优化器步数或 token 总预算。

来源 getter 在当前快照返回分类 157、长预测 9、短预测 47、可选 FRED 0、异常检测 1,949，共 2,162 个筛选前候选引用。这不是 2,162 个已认证可训练数据集。Windows 路径兼容层保留来源对 CharacterTrajectories 的明确排除，原生 Windows 差异另行记录。可选 `fred/preprocessed` 为空不能证明缺失论文必需素材；Monash FRED-MD 已包含。公开镜像的当前内容、README 描述和历史 ICM 预训练语料不应直接等同。

论文：[2409.13530v1](https://arxiv.org/abs/2409.13530v1)。相关来源提交 `3ab637e413f35f2c317573c0ace280d825c558de` 的关键原文代码按原 UTF8 字节封装归档，可以按 SHA256 恢复核对。它是相关 MOMENT 代码，尚未认证为 ICM 原作者完整流水线。原论文 Tiny/ICM 一整 epoch 预训练、全部预测、117 数据集 SVM、beta 消融及严格训练集预处理对照仍需执行。此前完成的 117 分类数据准备和单独的 72+6 异常检测复核保持独立。

研究的最低范围仍包含原登记的 45 篇论文、681 条方法/基线/消融和 273 条数据任务引用，以及后续原文发现的实验。本审计不代表全部研究实验已完成。
