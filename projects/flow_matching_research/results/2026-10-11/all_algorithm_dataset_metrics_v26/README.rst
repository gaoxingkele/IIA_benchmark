完整算法—数据集指标 v26
==========================

7,283项指标；1,805项有实测值；5,478项缺结果。

相对v24，本轮18项完整保存分数评价更新18项VUS／验证阈值Affiliation指标，其中12项由缺失变为实测；其余7,265条指标及采集时间逐字段一致。点级P/R/F1、AUROC、AP、所有其他任务和作者协议均未更新；原种子预算不变，无新增训练种子。

v26所有数值导出及逐行时间逐字节保留v25。v25继承了历史前缀一致旗标，该旗标不适用于v25与v24的比较；v26重新核验两者完整差异，当前base指v25，因此本版本的数值一致旗标适用于v26与v25。保留v25及失败登记作为历史证据。

18项评价覆盖全部原始实体与完整测试时间点，核验原模型/分数、原始与预处理数据、250个阈值及每一个整数缓冲区、验证校准阈值和宏平均。测试标签比例诊断未混入本次更新；单种子没有不确定性，多种子报告STD、SE和Student-t95%区间。

完整表：results.html、all_algorithm_dataset_metrics.csv、all_metrics.md、strict_algorithm_dataset.md。证据见publication_validation.json、range_increment_proof及independent_range_publication_review。

全部45篇论文、681项方法/基线/消融、273项原始数据任务及采用基线仍须完成。作者等价及完整TAB训练框架未认证。
