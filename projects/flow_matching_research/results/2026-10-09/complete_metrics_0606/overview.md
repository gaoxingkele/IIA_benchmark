# 算法—数据集实验指标总览

异常检测快照：2026-10-09 06:06:24（北京时间）。

730个登记组合：174个已有数值，160个完成预定重复，14个完成部分重复，556个暂无完整结果。核验821次完整运行，617次已有TAB范围指标。

[完整 Excel](complete_experiment_metrics.xlsx)；[全部技术指标长表](README.md)；[所有组合及状态](strict_summary.csv)；[逐种子技术指标](strict_per_seed.csv)。

下表为点级F1百分数（0–100）：验证分数第99百分位阈值，无PA。[n=数值]表示部分重复或固定预训练权重单次评价。—表示暂无完整运行结果；0.00是实测零值。

本地非重叠训练与作者stride=1预算尚未等效认证；作者阈值、PA-F1、Affiliation F和VUS分别解释。

## 常用与扩展数据集

| 算法配置 | PSM | SMAP | MSL | SMD | SWAT | SWAN | CICIDS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Independent CFM | 3.23 | 23.53 | 7.27 | 41.07 | 28.96 | 55.52 | 4.87 |
| OT-CFM | 3.09 | 23.38 | 7.36 | 41.21 | 28.95 | 55.60 | 4.87 |
| 单阶段 Rectified | 3.23 | 23.53 | 7.27 | 41.07 | 28.96 | 55.52 | 4.87 |
| Target CFM | 3.27 | 23.51 | 7.28 | 40.95 | 28.95 | 55.45 | 4.81 |
| VP-CFM | 3.00 | 23.04 | 7.71 | 39.30 | 30.08 | 57.95 | 5.77 |
| SB 稳定版 v2 | 3.34 | 23.39 | 7.62 | — | — | — | — |
| SF2M 仅流消融 v2 | 3.34 | 23.39 | 7.63 [n=3] | — | — | — | — |
| SF2M 稳定版 v2 | 3.28 | 23.27 | 7.57 | — | — | — | — |
| Rectified 40轮对照 | 3.36 | 23.29 | 7.25 | 41.87 | 28.81 | 55.54 | 4.87 |
| Rectified 60轮对照 | 3.37 | 23.18 | 7.23 | 41.40 | 28.75 | 55.85 [n=4] | 4.84 [n=4] |
| Reflow 两阶段 | 2.99 | 23.03 | 7.55 | 39.02 | 29.62 | 57.77 | 5.89 |
| Reflow 三阶段 | 3.01 | 23.01 | 7.50 | 38.30 | 30.09 | 58.88 | 6.12 |
| Anomaly Transformer（PSM配方） | 1.46 | — | — | — | — | — | — |
| Anomaly Transformer（SMAP配方） | — | 4.08 | — | — | — | — | — |
| DCdetector（PSM配方） | 1.73 | — | — | — | — | — | — |
| DCdetector（SMAP配方） | — | 1.06 [n=1] | — | — | — | — | — |
| iTransformer（PSM配方） | 4.84 | — | — | — | — | — | — |
| TimesNet（PSM配方） | 5.39 | — | — | — | — | — | — |
| TimesNet（SMAP配方） | — | 3.82 [n=4] | — | — | — | — | — |
| TranAD（PSM配方） | 2.76 | — | — | — | — | — | — |
| USAD（登记损失） | 2.88 | — | — | — | — | — | — |
| USAD（论文有符号损失） | 5.88 | — | — | — | — | — | — |
| MOMENT-small（预训练512） | 3.66 [n=1] | 10.56 [n=1] | 8.70 [n=1] | 17.48 [n=1] | 8.35 [n=1] | 42.62 [n=1] | — |

## 工业与过程数据集

| 算法配置 | SKAB | TEP_CLASSIC | PRONTO_DAY2 | PRONTO_DAY3 | PRONTO_DAY4 |
|---|---:|---:|---:|---:|---:|
| Independent CFM | 50.92 | 73.06 | 16.48 | 19.11 | 0.41 |
| OT-CFM | 50.97 | 73.18 | 16.32 | 20.02 | 0.43 |
| 单阶段 Rectified | 50.92 | 73.06 | 16.48 | 19.11 | 0.41 |
| Target CFM | 50.95 | 73.21 | 16.43 | 19.46 | 0.40 |
| VP-CFM | 51.02 | 72.98 | 11.05 | 17.92 | 0.41 |
| SB 稳定版 v2 | 50.96 | 73.15 | 15.58 | 17.36 [n=4] | 0.41 [n=3] |
| SF2M 仅流消融 v2 | 50.96 | 73.15 | 15.58 | 17.36 [n=4] | 0.41 [n=3] |
| SF2M 稳定版 v2 | 50.98 | 73.06 | 15.43 | 17.89 [n=4] | 0.40 [n=3] |
| Rectified 40轮对照 | 50.76 | 73.11 | 16.69 | — | 0.41 |
| Rectified 60轮对照 | 50.71 | 73.14 | 16.69 | — | 0.40 |
| Reflow 两阶段 | 50.90 | 73.04 | 8.62 | 17.41 | 0.41 |
| Reflow 三阶段 | 50.91 | 73.05 | 1.30 | 13.49 [n=1] | 0.42 |
| Anomaly Transformer（PSM配方） | 1.94 [n=4] | 2.64 | 0.86 | — | 4.97 |
| Anomaly Transformer（SMAP配方） | — | — | — | — | — |
| DCdetector（PSM配方） | 2.39 [n=3] | 2.82 | 1.94 | — | 2.00 |
| DCdetector（SMAP配方） | — | — | — | — | — |
| iTransformer（PSM配方） | 19.58 | 55.74 | 1.14 | — | 1.39 |
| TimesNet（PSM配方） | 19.83 | 61.24 | 1.35 | — | 0.44 |
| TimesNet（SMAP配方） | — | — | — | — | — |
| TranAD（PSM配方） | 50.98 | 76.13 | 8.22 | — | 0.41 |
| USAD（登记损失） | 51.60 | 83.21 | 16.61 | — | 0.00 |
| USAD（论文有符号损失） | 50.98 | 83.14 | 3.49 | — | 0.00 |
| MOMENT-small（预训练512） | — | — | — | — | — |

TEP是经典单工况整运行；SKAB校准含原生异常；PRONTO正常段选择使用标签且整日轮换相关。基线PSM配方迁移尚未目标调优。

## 其他实验及完整状态

- 作者流程共1429项，状态{'failed_or_partial_preserved': 6, 'running': 2, 'pending': 1419, 'completed': 2}；[作者实际技术指标](author_pipeline_numeric_metrics.csv)与[逐项状态](author_pipeline_jobs.csv)单列。发布权重评价不等于重新训练复现。
- 插补共540项，已有152条逐次技术指标；MAE/RMSE/CRPS/MRE、重复数、论文差值见[插补汇总](imputation_summary.csv)。
- 历史137次运行、548条协议记录见[历史结果](historical_protocol_results.csv)。
- DT-LA、SHCL、KGL等无当前严格主表完整成绩的组合保留缺口，论文值不能替代本地实测。
- [已核验论文报告值](paper_claims.csv)和[45篇原始实验范围](original_dataset_scope.csv)保留原文证据及复现缺口。

## 配置名称对应

| 表内名称 | 模型配置名称 |
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
| Anomaly Transformer（SMAP配方） | fm_strict_anomaly_transformer__smap__registered |
| DCdetector（PSM配方） | fm_strict_dcdetector__psm__registered |
| DCdetector（SMAP配方） | fm_strict_dcdetector__smap__registered |
| iTransformer（PSM配方） | fm_strict_itransformer__psm__registered |
| TimesNet（PSM配方） | fm_strict_timesnet__psm__registered |
| TimesNet（SMAP配方） | fm_strict_timesnet__smap__registered |
| TranAD（PSM配方） | fm_strict_tranad__psm__registered |
| USAD（登记损失） | fm_strict_usad__psm__registered |
| USAD（论文有符号损失） | fm_strict_usad__psm__signed_paper_loss |
| MOMENT-small（预训练512） | fm_moment_pretrained_small__release512 |
