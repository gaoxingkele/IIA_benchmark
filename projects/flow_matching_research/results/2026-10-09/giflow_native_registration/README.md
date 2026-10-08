# GiFlow 全程实验登记与继续执行

核验时间：2026-10-08T23:25:22.646095+00:00。完整目标保持不变，尚未全部完成。

新增冻结 140 项 GiFlow 全程实验；46 项来源哈希已核验。
Air-36/AQI 主实验；Air-36 公开代码的 graph/time/none 先验分支、20%–60% 点缺失率、六档图阈值。五个配对种子，原样与已审修正两条轨道。
正式训练保留 300 轮上限、patience40、batch128、窗口24、stride1、20步Euler。全量预测留存于配置指定的 F:/aicoding/IIA_Data/experiments/giflow_native_v1；原始数据不改写。

实际控制器 PID 8844，已按命令行和创建时间核验存活；当前状态 waiting_original_grin，等待先前原始 GRIN 队列 10 项完成。尚无 GiFlow 全程成绩。共用原有 GPU 锁、重任务锁与 8GiB 紧急提交量余量保护，没有终止现有模型或降低正式预算。

## 真实代码与数据检查

- 两个完整原始数据集、两条代码轨的划分/掩码/图/缩放统计审计通过；同种子配对数组逐元素完全相同，训练/验证/测试时间点无交叉，评价目标未进入条件输入。
- 两条原函数均完成真实 Air-36 训练一步和20步采样集成检查；保存预测与原函数误差一致，已审修正版实际测试权重逐张量等于验证选出的检查点。
- 集成检查使用CPU、batch2、1轮、1批测试、EMA起始轮0，仅验证连接与产物，不计入正式运行或benchmark。原样代码在EMA尚未初始化时结束会KeyError；原失败记录保留。
- 修正版数据加载调用原已传root，包装层现验证并去重同一路径参数；该修正经过真实代码检查，未改写冻结的原作者/已审源文件。

## 本次另外完成的严格/本地运行

源快照 2026-10-08T23:17:12.394975+00:00：已完成 916 次，比07:02表新增 9 次。每项 result/scores/model SHA256重新核验。

| 算法配置 | 数据集 | 种子 | Precision | Recall | 点级F1 | AUROC | AP |
|---|---|---:|---:|---:|---:|---:|---:|
| fm_objective_sb_stable_v2 | smd | 1104 | 0.482798 | 0.347940 | 0.404423 | 0.761566 | 0.327382 |
| fm_strict_dcdetector__msl__registered | msl | 1103 | 0.114583 | 0.011651 | 0.021151 | 0.504786 | 0.108140 |
| fm_strict_dcdetector__msl__registered | msl | 1104 | 0.096206 | 0.009400 | 0.017127 | 0.513247 | 0.108427 |
| fm_moment_pretrained_small__release512 | cicids | 42 | 0.363062 | 0.014655 | 0.028173 | 0.420767 | 0.285714 |
| fm_moment_pretrained_small__release512 | tep_classic | 42 | 0.857275 | 0.439762 | 0.581320 | 0.562498 | 0.852026 |
| fm_moment_pretrained_small__release512 | skab | 42 | 0.545031 | 0.154593 | 0.240866 | 0.594941 | 0.424080 |
| fm_moment_pretrained_small__release512 | pronto_day2 | 42 | 1.000000 | 0.023936 | 0.046752 | 0.544039 | 0.953758 |
| fm_moment_pretrained_small__release512 | pronto_day3 | 42 | 0.550725 | 0.053710 | 0.097875 | 0.375337 | 0.721401 |
| fm_moment_pretrained_small__release512 | pronto_day4 | 42 | 1.000000 | 0.001652 | 0.003298 | 0.586470 | 0.473271 |

这些为验证分位数阈值、无PA的本地完整结果；MOMENT为固定发布权重的单次评价。窗口预算与作者stride1训练未等效认证。

## 尚待补齐的原论文义务

- Paper Adam/patience10/random70-10-20 versus released AdamW/patience40/TemporalSplitter (validation10% of non-test).
- Learned filtering-factor Problem5 and composed spatial-temporal Eq4 absent in release.
- Block masks, PeMS08, attention/propagation ablations, Gaussian prior, Euler-step comparison and paper hyperparameter search remain required.
- Air36 release has 8759 timestamps versus paper8760; full AQI batch128 must actually fit; no reduced batch fallback.

原样 graph_time 中，时间过滤结果会覆盖此前的空间过滤结果；none 是观测/零先验，不能当作论文 FM-Gauss 消融。原代码固定过滤因子、AdamW、时序划分仍与论文不一致，不能用这140项代替全论文等效复现。

45篇ARA、681条方法/消融记录及其他原数据、作者全流程、工业迁移、预测、生成、连续时间、图像、单细胞、表格和TAB训练流程义务全部保留。

[完整登记与核验数据](registration_audit.json)。正式配置位于 benchmark/configs/experiments/fm_giflow_native_queue.v1.json。
