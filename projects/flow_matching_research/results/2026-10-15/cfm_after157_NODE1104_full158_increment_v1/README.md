158/285 deduplicated complete original CFM models

Current native/independent proof chains and artifact hashes were rechecked. Overlapping snapshots are deduplicated by canonical identity, never summed. The previous157 records remain identical. All newly closed original source grids and metrics were independently recomputed.

| New canonical model | Epochs | Updates | Original grid | MSE | MAE | RMSE | Bitwise native scores |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| cfm_ts_original__appendix_d__node_e10__sine__seed1104 | 10 | 20 | [80, 500, 1] | 0.119296264199 | 0.252603377631 | 0.34539291278 | True |

Appendix conditional GP600/Sine now has four complete finite seeds1103,1105,1106,1107 and the independently observed full80-grid failed seed1104. Full five-seed600 MSE remains undefined. The separate all-five original400/600 attempt-outcome audit retains that failure. Main20-test/93600-update GP600 recipes remain distinct from Appendix80-test/375000-update recipes; no combined performance mean, seed exclusion, altered solver or successful-subset benchmark. Original-author equivalence and strict TAB remain unproved; all285 slots and45/681/273scope remain required.

Paper: https://openreview.net/forum?id=Hqn4Aj7xrQ


## 新增NODE1104的预算、耗时与状态边界

原神经ODE/Sine/Appendix D种子1104实际完成320条训练轨迹、10个epoch和20次优化更新，150,865参数模型覆盖80条留出轨迹×500时间点×1观测维，共40,000个原测试网格点。原训练68856.05875秒（约19.12668小时），评分4.53747秒；长耗时发生在训练阶段。未因等待较久重启、减小预算、更改求解器或放松容差。具体运算热点未另做侵入式剖析。

|MSE|MAE|RMSE|80轨迹MSE描述性样本STD|
|---:|---:|---:|---:|
|0.11929626419875178|0.2526033776306255|0.34539291278014345|0.16658710331504847|

CPU原生检查点重推理与原保存预测逐元素相等，最大绝对差0；原先冻结rtol1e-7/atol1e-8保持。第二独立进程复核全部原始网格、轨迹身份、真值、完整指标和当前文件哈希。原训练、原生复核、独立检查均有实际成功退出且无资源中止的保护证明。

原checkpoint保留权重、预处理、10轮loss和更新计数，足以支持这次模型重推理；它没有完整优化器或训练RNG状态，因此未认证训练可从任意中间步骤精确续跑。80轨迹STD描述单个已拟合模型的轨迹差异，不是五个配对数据/模型种子的置信区间。本组1103..1107仍全部要求，不能用目前两项代替五种子汇总。

跨三个重叠快照按canonical ID去重后确认为158/285；单个收集器155不能当作全项目总数，也不能把三份快照相加。原157项保持相同，所有现有证明链、数据及模型/预测产物重新核对。已观察到的GP600失败种子和未定义五种子均值继续保留。42项相关检查通过，大型原产物留在原目录；本次归档未新增训练或重复模型重推理。
