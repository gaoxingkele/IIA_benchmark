# 算法与数据集实验指标总览

异常检测快照：2026-10-09 05:26:21（北京时间）。

730 个登记组合：154 个有完整运行数值，144 个完成预定重复，10 个仅完成部分重复，576 个暂无完整结果。共核验728次完整运行，543次已有TAB范围指标。

[完整技术指标 Excel](complete_experiment_metrics.xlsx)；[全部154个有数值组合的P/R/F1、AUROC、AP、VUS、Affiliation](README.md)；[730个组合及全部状态](strict_summary.csv)；[逐种子、区间、事件指标和运行资源](strict_per_seed.csv)。

下列点级 F1 均为百分数（0–100）。使用验证分数第99百分位阈值，即1%尾部报警比例，不做点调整。无后缀为5/5种子；[n=3]等表示尚未完成5个种子；MOMENT为一次固定预训练模型评价[n=1]。—表示暂无完整运行数值，不代表零。

本地窗口适配、非重叠训练预算与作者stride=1流程尚未等效认证。PA-F1、测试集校准、Affiliation F及VUS与点级F1分别解释。

## 常用及扩展异常检测数据集

| 算法配置 | PSM | SMAP | MSL | SMD | SWAT | SWAN | CICIDS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Independent CFM | 3.23 | 23.53 | 7.27 | 41.07 | 28.96 | 55.52 | 4.87 |
| OT-CFM | 3.09 | 23.38 | 7.36 | 41.21 | 28.95 | 55.60 | 4.87 |
| 单阶段 Rectified | 3.23 | 23.53 | 7.27 | 41.07 | 28.96 | 55.52 | 4.87 |
| Target CFM | 3.27 | 23.51 | 7.28 | 40.95 | 28.95 | 55.45 | 4.81 |
| VP-CFM | 3.00 | 23.04 | 7.71 | 39.30 | 30.08 | 57.95 | 5.77 |
| SB 稳定版 v2 | 3.34 | 23.39 | 7.63 [n=3] | — | — | — | — |
| SF2M 仅流消融 v2 | 3.34 | 23.39 | — | — | — | — | — |
| SF2M 稳定版 v2 | 3.28 | 23.27 | — | — | — | — | — |
| Rectified 40轮对照 | 3.36 | 23.29 | 7.25 | 41.87 | 28.81 | 55.54 | 4.87 |
| Rectified 60轮对照 | 3.37 | 23.18 | 7.23 | 41.40 | 28.75 | 55.85 [n=4] | 4.84 [n=4] |
| Reflow 两阶段 | 2.99 | 23.03 | 7.55 | 39.02 | 29.62 | 57.77 | 5.89 |
| Reflow 三阶段 | 3.01 | 23.01 | 7.50 | 38.30 | 30.09 | 58.88 | 6.12 |
| Anomaly Transformer（PSM配方） | 1.46 | — | — | — | — | — | — |
| DCdetector（PSM配方） | 1.73 | — | — | — | — | — | — |
| iTransformer（PSM配方） | 4.84 | — | — | — | — | — | — |
| TimesNet（PSM配方） | 5.39 | — | — | — | — | — | — |
| TimesNet（SMAP配方） | — | 3.87 [n=3] | — | — | — | — | — |
| TranAD（PSM配方） | 2.76 | — | — | — | — | — | — |
| USAD（登记损失） | 2.88 | — | — | — | — | — | — |
| USAD（论文有符号损失） | 5.88 | — | — | — | — | — | — |
| MOMENT-small（预训练512） | 3.66 [n=1] | 10.56 [n=1] | 8.70 [n=1] | 17.48 [n=1] | 8.35 [n=1] | 42.62 [n=1] | — |

## 工业及过程数据集

| 算法配置 | SKAB | TEP_CLASSIC | PRONTO_DAY2 | PRONTO_DAY4 |
|---|---:|---:|---:|---:|
| Independent CFM | 50.92 | 73.06 | 16.48 | 0.41 |
| OT-CFM | 50.97 | 73.18 | 16.32 | 0.43 |
| 单阶段 Rectified | 50.92 | 73.06 | 16.48 | 0.41 |
| Target CFM | 50.95 | 73.21 | 16.43 | 0.40 |
| VP-CFM | 51.02 | 72.98 | 11.05 | 0.41 |
| SB 稳定版 v2 | 50.96 | 73.15 | 15.58 | 0.41 [n=3] |
| SF2M 仅流消融 v2 | 50.96 | 73.15 | 15.58 | 0.41 [n=3] |
| SF2M 稳定版 v2 | 50.98 | 73.06 | 15.43 | 0.40 [n=3] |
| Rectified 40轮对照 | 50.76 | 73.11 | 16.69 | 0.41 |
| Rectified 60轮对照 | 50.71 | 73.14 | 16.69 | 0.40 |
| Reflow 两阶段 | 50.90 | 73.04 | 8.62 | 0.41 |
| Reflow 三阶段 | 50.91 | 73.05 | 1.30 | 0.42 |
| Anomaly Transformer（PSM配方） | 1.94 [n=4] | 2.64 | — | 4.97 |
| DCdetector（PSM配方） | 2.39 [n=3] | 2.82 | — | 2.00 |
| iTransformer（PSM配方） | 19.58 | 55.74 | — | 1.39 |
| TimesNet（PSM配方） | 19.83 | 61.24 | 1.48 [n=2] | 0.44 |
| TimesNet（SMAP配方） | — | — | — | — |
| TranAD（PSM配方） | 50.98 | 76.13 | — | 0.41 |
| USAD（登记损失） | 51.60 | 83.21 | — | 0.00 |
| USAD（论文有符号损失） | 50.98 | 83.14 | — | 0.00 |
| MOMENT-small（预训练512） | — | — | — | — |

工业边界：TEP为经典单工况整运行；SKAB校准包含原生异常；PRONTO正常段选择使用标签且整日轮换相关。PSM配方迁移尚未目标调优。PRONTO_DAY3没有完整结果，仍在730组合表中保留。

## 其他实验和未完成项

- 插补：43/540个完整任务，34条已有数值的汇总指标；MAE/RMSE/CRPS/MRE、重复数及论文差值见[插补汇总](imputation_summary.csv)。
- MaelNet、Pi-Transformer、CrossAD：作者流程共1429项登记任务，尚无完整流程最终指标。详细配方和状态见[作者流程状态](author_pipeline_jobs.csv)。
- DT-LA、SHCL、KGL等尚未形成当前严格主表的完整成绩，不能用论文声称值代替本地成绩。
- 历史137次运行对应548条协议记录，独立保留在[历史协议结果](historical_protocol_results.csv)，不混入当前严格均值。
- 原文71条已核验指标见[论文报告值](paper_claims.csv)，45篇原始实验范围和缺口见[原文数据范围](original_dataset_scope.csv)。
- 11个资源事故恢复任务是原种子的重试，当前没有完整结果，不增加独立重复次数；登记保存在 benchmark 的 configs/experiments/fm_resource_recovery_queue.v2.json。

## 配置名称对应

| 表内名称 | 原模型配置名称 |
|---|---|
| Independent CFM | fm_objective_independent |
| OT-CFM | fm_objective_ot |
| 单阶段 Rectified | fm_objective_rectified |
| Target CFM | fm_objective_target |
| VP-CFM | fm_objective_vp |
| SB 稳定版 v2 | fm_objective_sb_stable_v2 |
| SF2M 仅流消融 v2 | fm_objective_sf2m_flow_only_stable_v2 |
| SF2M 稳定版 v2 | fm_objective_sf2m_stable_v2 |
| Rectified 40轮对照 | fm_rectified_40epoch_budget_control |
| Rectified 60轮对照 | fm_rectified_60epoch_budget_control |
| Reflow 两阶段 | fm_reflow_2stage |
| Reflow 三阶段 | fm_reflow_3stage |
| Anomaly Transformer（PSM配方） | fm_strict_anomaly_transformer__psm__registered |
| DCdetector（PSM配方） | fm_strict_dcdetector__psm__registered |
| iTransformer（PSM配方） | fm_strict_itransformer__psm__registered |
| TimesNet（PSM配方） | fm_strict_timesnet__psm__registered |
| TimesNet（SMAP配方） | fm_strict_timesnet__smap__registered |
| TranAD（PSM配方） | fm_strict_tranad__psm__registered |
| USAD（登记损失） | fm_strict_usad__psm__registered |
| USAD（论文有符号损失） | fm_strict_usad__psm__signed_paper_loss |
| MOMENT-small（预训练512） | fm_moment_pretrained_small__release512 |
