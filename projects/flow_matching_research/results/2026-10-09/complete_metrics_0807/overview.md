# 算法—数据集实验结果总览

异常检测源快照：2026-10-09 08:07:57（北京时间）。插补、作者流程和附加原生队列在导出时另行核验。

登记 730 个异常检测组合，204 个已有数值，196 个完成预定重复，8 个仅完成部分重复，526 个暂无完整结果。已核验 944 次完整运行，625 次已有 TAB 范围指标。

[完整技术指标 Excel](complete_experiment_metrics.xlsx)；[全部有数值组合的 P/R/F1、AUROC、AP、VUS、Affiliation](README.md)；[全部组合与状态](strict_summary.csv)；[逐种子指标、区间和资源](strict_per_seed.csv)；[逐实体指标](strict_per_entity.csv)。

下表点级 F1 使用 0–100 百分数。验证分数第99百分位定阈值，不做 PA；不是论文原协议等效复现。无次数标记为5/5种子，n=x/y表示部分重复；MOMENT 1/1是固定预训练权重的一次评价，不是多种子重新训练。—表示暂无完整运行数值。

协议、模型配置与来源哈希均保存在完整表中。不同数据集异常比例和训练预算不同，不能直接凭 F1 排跨数据集名次。

## 常用异常检测数据集

| 算法配置 | PSM | SMAP | MSL | SMD | SWAT | SWAN | CICIDS |
|---|---:|---:|---:|---:|---:|---:|---:|
| fm_moment_pretrained_small__TAB100 | 5.11 [n=1/1] | 4.77 [n=1/1] | 7.73 [n=1/1] | 26.81 [n=1/1] | — | — | — |
| fm_moment_pretrained_small__release512 | 3.66 [n=1/1] | 10.56 [n=1/1] | 8.70 [n=1/1] | 17.48 [n=1/1] | 8.35 [n=1/1] | 42.62 [n=1/1] | 2.82 [n=1/1] |
| fm_objective_independent | 3.23 | 23.53 | 7.27 | 41.07 | 28.96 | 55.52 | 4.87 |
| fm_objective_ot | 3.09 | 23.38 | 7.36 | 41.21 | 28.95 | 55.60 | 4.87 |
| fm_objective_rectified | 3.23 | 23.53 | 7.27 | 41.07 | 28.96 | 55.52 | 4.87 |
| fm_objective_sb_stable_v2 | 3.34 | 23.39 | 7.62 | 40.89 [n=3/5] | — | — | — |
| fm_objective_sf2m_flow_only_stable_v2 | 3.34 | 23.39 | 7.62 | — | — | — | — |
| fm_objective_sf2m_stable_v2 | 3.28 | 23.27 | 7.57 | — | — | — | — |
| fm_objective_target | 3.27 | 23.51 | 7.28 | 40.95 | 28.95 | 55.45 | 4.81 |
| fm_objective_vp | 3.00 | 23.04 | 7.71 | 39.30 | 30.08 | 57.95 | 5.77 |
| fm_rectified_40epoch_budget_control | 3.36 | 23.29 | 7.25 | 41.87 | 28.81 | 55.54 | 4.87 |
| fm_rectified_60epoch_budget_control | 3.37 | 23.18 | 7.23 | 41.40 | 28.75 | 55.83 | 4.84 [n=4/5] |
| fm_reflow_2stage | 2.99 | 23.03 | 7.55 | 39.02 | 29.62 | 57.77 | 5.89 |
| fm_reflow_3stage | 3.01 | 23.01 | 7.50 | 38.30 | 30.09 | 58.88 | 6.12 |
| fm_strict_anomaly_transformer__msl__registered | — | — | 2.04 | — | — | — | — |
| fm_strict_anomaly_transformer__psm__registered | 1.46 | — | — | — | — | — | — |
| fm_strict_anomaly_transformer__smap__registered | — | 4.08 | — | — | — | — | — |
| fm_strict_dcdetector__msl__registered | — | — | 1.88 | — | — | — | — |
| fm_strict_dcdetector__psm__registered | 1.73 | — | — | — | — | — | — |
| fm_strict_dcdetector__smap__registered | — | 0.90 | — | — | — | — | — |
| fm_strict_itransformer__msl__registered | — | — | 7.99 | — | — | — | — |
| fm_strict_itransformer__psm__registered | 4.84 | — | — | — | — | — | — |
| fm_strict_itransformer__smap__registered | — | 4.52 | — | — | — | — | — |
| fm_strict_timesnet__msl__registered | — | — | 7.68 | — | — | — | — |
| fm_strict_timesnet__psm__registered | 5.39 | — | — | — | — | — | — |
| fm_strict_timesnet__smap__registered | — | 3.89 | — | — | — | — | — |
| fm_strict_tranad__msl__registered | — | — | 8.13 | — | — | — | — |
| fm_strict_tranad__psm__registered | 2.76 | — | — | — | — | — | — |
| fm_strict_usad__msl__registered | — | — | 6.52 | — | — | — | — |
| fm_strict_usad__msl__signed_paper_loss | — | — | 13.78 | — | — | — | — |
| fm_strict_usad__psm__registered | 2.88 | — | — | — | — | — | — |
| fm_strict_usad__psm__signed_paper_loss | 5.88 | — | — | — | — | — | — |
| fm_strict_usad__smap__registered | — | 24.99 | — | — | — | — | — |
| fm_strict_usad__smap__signed_paper_loss | — | 6.25 | — | — | — | — | — |

## 工业及过程数据集

| 算法配置 | PRONTO_DAY2 | PRONTO_DAY3 | PRONTO_DAY4 | SKAB | TEP_CLASSIC |
|---|---:|---:|---:|---:|---:|
| fm_moment_pretrained_small__TAB100 | — | — | — | — | — |
| fm_moment_pretrained_small__release512 | 4.68 [n=1/1] | 9.79 [n=1/1] | 0.33 [n=1/1] | 24.09 [n=1/1] | 58.13 [n=1/1] |
| fm_objective_independent | 16.48 | 19.11 | 0.41 | 50.92 | 73.06 |
| fm_objective_ot | 16.32 | 20.02 | 0.43 | 50.97 | 73.18 |
| fm_objective_rectified | 16.48 | 19.11 | 0.41 | 50.92 | 73.06 |
| fm_objective_sb_stable_v2 | 15.58 | 17.36 [n=4/5] | 0.41 [n=3/5] | 50.96 | 73.15 |
| fm_objective_sf2m_flow_only_stable_v2 | 15.58 | 17.36 [n=4/5] | 0.41 [n=3/5] | 50.96 | 73.15 |
| fm_objective_sf2m_stable_v2 | 15.43 | 17.89 [n=4/5] | 0.40 [n=3/5] | 50.98 | 73.06 |
| fm_objective_target | 16.43 | 19.46 | 0.40 | 50.95 | 73.21 |
| fm_objective_vp | 11.05 | 17.92 | 0.41 | 51.02 | 72.98 |
| fm_rectified_40epoch_budget_control | 16.69 | 20.37 | 0.41 | 50.76 | 73.11 |
| fm_rectified_60epoch_budget_control | 16.69 | 19.97 | 0.40 | 50.71 | 73.14 |
| fm_reflow_2stage | 8.62 | 17.41 | 0.41 | 50.90 | 73.04 |
| fm_reflow_3stage | 1.30 | 13.00 | 0.42 | 50.91 | 73.05 |
| fm_strict_anomaly_transformer__msl__registered | — | — | — | — | — |
| fm_strict_anomaly_transformer__psm__registered | 0.86 | 1.38 | 4.97 | 1.94 | 2.64 |
| fm_strict_anomaly_transformer__smap__registered | — | — | — | — | — |
| fm_strict_dcdetector__msl__registered | — | — | — | — | — |
| fm_strict_dcdetector__psm__registered | 1.94 | 2.16 | 2.00 | 2.26 | 2.82 |
| fm_strict_dcdetector__smap__registered | — | — | — | — | — |
| fm_strict_itransformer__msl__registered | — | — | — | — | — |
| fm_strict_itransformer__psm__registered | 1.14 | 3.60 | 1.39 | 19.58 | 55.74 |
| fm_strict_itransformer__smap__registered | — | — | — | — | — |
| fm_strict_timesnet__msl__registered | — | — | — | — | — |
| fm_strict_timesnet__psm__registered | 1.35 | 3.96 | 0.44 | 19.83 | 61.24 |
| fm_strict_timesnet__smap__registered | — | — | — | — | — |
| fm_strict_tranad__msl__registered | — | — | — | — | — |
| fm_strict_tranad__psm__registered | 8.22 | 23.45 | 0.41 | 50.98 | 76.13 |
| fm_strict_usad__msl__registered | — | — | — | — | — |
| fm_strict_usad__msl__signed_paper_loss | — | — | — | — | — |
| fm_strict_usad__psm__registered | 16.61 | 25.22 | 0.00 | 51.60 | 83.21 |
| fm_strict_usad__psm__signed_paper_loss | 3.49 | 7.12 | 0.00 | 50.98 | 83.14 |
| fm_strict_usad__smap__registered | — | — | — | — | — |
| fm_strict_usad__smap__signed_paper_loss | — | — | — | — | — |

## 其他任务及比较边界

- 插补完成 47/540 项；MAE、RMSE、CRPS、MRE、尺度、折/种子、误差区间和论文差值见[插补汇总](imputation_summary.csv)。
- 作者流程登记 1429 项，状态 {'failed_or_partial_preserved': 7, 'running': 2, 'pending': 1418, 'completed': 2}；[最终技术指标](author_pipeline_numeric_metrics.csv)与[逐项状态](author_pipeline_jobs.csv)独立保存。
- 原生附加队列登记 {'giflow': 140, 'grasp_mtsbench_statistical_replay': 80}，状态 {'pending': 220}；见[原生方法与消融状态](additional_native_paper_jobs.csv)。待运行不生成虚构成绩。
- MaelNet、Pi-Transformer、DT-LA、SHCL、KGL、CrossAD 等尚未在本次严格主表形成完整结果。论文报告值不能替代本地成绩。
- TAB测试集校准、最优比例和PA结果见[单列诊断](tab_ratio_diagnostic.csv)；不能与验证阈值严格结果混排。
- 原文指标见[论文值](paper_claims.csv)，45篇论文数据范围及缺口见[原文范围](original_dataset_scope.csv)。当前清单覆盖全部已登记任务及已保存指标，不表示全部论文实验已完成或所有原文表格已转录。
- 当前部分神经方法使用非重叠训练窗口，与作者stride=1的更新次数尚未对齐。Independent CFM与单阶段Rectified在本地sigma=0配置下等价；SF2M仅流消融也不能代表完整SF2M。
- TEP是经典单工况整运行；SKAB校准包含异常；PRONTO正常段选择使用标签且整日轮换相关。工业迁移成绩尚非原论文等效复现。
