# 完整算法—数据集实验指标

发布：2026-10-10T20:33:12.792009+00:00。继承v11全部原始数值与采集时间，本次增加GRASP τ=0.25完整检查点重推断证据。

主表7117条指标记录：1786条有实测值，5331条缺结果。指标行数不等于训练次数；全部原论文实验仍未完成。

[可搜索完整表](results.html)；[完整CSV](all_algorithm_dataset_metrics.csv)；[完整可读表](all_metrics.md)；[异常检测全部指标](strict_algorithm_dataset.md)；[全部F1矩阵](overview.md)。

GRASP τ=0.25/SWAN/seed1103实际完成1500轮、93000次优化更新；原检查点恢复后，3935个验证点与72000个测试点全部重新推断。观察到800次验证模型调用与14100次测试模型调用，覆盖194300与3597550次窗口计算。

验证第99百分位阈值、无PA的Precision/Recall/F1分别为38.7339%/83.8308%/52.9858%，与原运行分类完全一致。回顾性测试标签Best-F1为53.0506%；原生PRC和AUROC重算差异分别约4.0e-9和1.9e-9。CPU/GPU分数满足预注册数值容差，不声称分数逐位一致。

[完整模型重推断证据](../grasp_tau025_checkpoint_replay_v1/checkpoint_replay_audit.json)；[后续完整范围执行记录](../checkpoint_full_scope_resume_v3/README.md)。

这仍是1/10训练种子，重推断不增加训练种子、样本量或新的性能记录。作者等价和完整TAB认证未取得。原先13个训练模型中的7个完整重推断已保留，剩余6个模型24个配方继续执行；GiFlow保留9个完整重推断批次并继续其余5个批次。尚未完成的范围不能算通过。

新接入的USHCN100次完整500轮任务另有120条待完成聚合指标，见[待完成清单](../ls4_native_ushcn_training_execution_v1/pending_native_metrics.csv)。它们不以数据审计或容量诊断代替训练结果。
