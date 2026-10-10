# 完整算法—数据集实验指标

发布：2026-10-10T20:20:17.049952+00:00。本版只更新GRASP原始完整训练结果；其他轨道保留原快照时间。

**7117条指标记录：1786条有实测值，5331条缺结果。指标行数不等于训练次数。全部原论文实验仍未完成。**

[可搜索完整表](results.html)；[完整CSV](all_algorithm_dataset_metrics.csv)；[完整可读表](all_metrics.md)；[全部严格异常检测F1矩阵](overview.md)。

GRASP现在有14/480个全实体种子组完成。新完成tau0.25/SWAN/seed1103：1500轮、93000次更新、3935验证点、72000测试点。

作者回顾性无PA Best-F1为53.0506%；验证第99百分位阈值、无PA的对照F1为52.9858%，Precision为38.7339%、Recall为83.8308%；原生PRC（源码字段AP）为0.520136、AUROC为0.721708。这里只完成1/10种子，未给单种子STD或置信区间。

保存分数对应的训练预算、模型和数据绑定已核验；该新训练模型的完整独立重推断正在执行。作者回顾性测试标签阈值与验证阈值对照分开，未取得作者等价或严格TAB认证。

[最新GRASP规范槽位](../grasp_canonical_listing_v6/grasp_execution_audit.json)；[USHCN完整训练接入和检查点状态](../ls4_native_ushcn_training_execution_v1/README.md)；[之前的完整协议说明](../all_algorithm_dataset_metrics_v10/README.md)。

USHCN新接入的100次完整500轮任务和20个完整GPU容量案例尚无500轮结果，不以注册、数据审计或容量诊断填性能分数。此前Pi和GiFlow结果沿用已核验快照；各行来源与时间均保留。
