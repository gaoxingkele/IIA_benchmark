完整算法—数据集指标 v20
==========================

发布UTC：2026-10-10T22:50:28.689199+00:00。继承v19的7283项指标：1790项有实测值，5493项缺完整结果，141个算法／配置名称。全部数值表和采集时间逐字节保留。本轮新增完整模型核验，不增加成绩、训练种子或置信区间。

全部指标见results.html、all_algorithm_dataset_metrics.csv、all_metrics.md。数值行中的旧核验状态描述保留为原采集快照；当前GiFlow核验以本README和publication_validation.json的证据链接为准。

GiFlow reviewed_corrected/Air36/seed393：原生验证选择检查点恢复后，全部1747测试窗口、268613缺失目标、14批次、20步Euler、280次前向重推断完成。独立核验所有保存数组与原数据、掩码、图、缩放器和划分；在原冻结CPU/GPU容差内通过。原生MAE10.951781375、MSE413.131974901、MAPE23.823091711%、sqrt(mean_batch_MSE)20.325648204维持原值。

证据：../giflow_corrected_full_replay_verified_v1/independent_full_replay_verification.json；完整模型原始证据：../giflow_corrected_checkpoint_replay_v1/reviewed_corrected_checkpoint_replay_audit.json。训练仍为1/5种子，原生重叠窗口和批次均值指标口径保留；不声称作者等价或完整TAB认证。

45篇论文、681项方法/基线/消融、273项原始任务及采用基线的全部范围仍未完成。TimeGAN、GT-GAN和其他未完成任务继续保留，不以预处理、检查点加载或重推断替代训练成绩。
