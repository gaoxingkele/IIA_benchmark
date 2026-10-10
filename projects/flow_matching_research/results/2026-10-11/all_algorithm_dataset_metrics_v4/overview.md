# 完整算法—数据集结果入口与F1矩阵

本地严格异常检测快照：2026-10-10T16:40:28.278722+00:00（UTC）。概率指标0–1；下面F1矩阵专用百分数。

[全部6953条指标和技术明细入口](README.md)；[可搜索的全量表](results.html)；[全量CSV](all_algorithm_dataset_metrics.csv)；[全部730组合P/R/F1、AUROC、AP、VUS、Affiliation及区间](strict_algorithm_dataset.md)。

**以下为验证分数第99百分位、无PA的本地结果。非完整TAB作者训练等价认证；未完成值为—。**

矩阵保留90个冻结配置名称，不合并不同预算、不同数据集专用配置和消融。单元格为F1 (%) [已完成/预定种子]。

| 算法配置 | PSM | SMAP | MSL | SMD | SWAT | SWAN | CICIDS | TEP_CLASSIC | SKAB | PRONTO_DAY2 | PRONTO_DAY3 | PRONTO_DAY4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fm_comparator_dt_la | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_dt_la_l2_no_sparse | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_dt_la_l2_sparse | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_dt_la_lambda_0p1 | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_dt_la_lambda_0p3 | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_dt_la_lambda_0p7 | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_dt_la_lambda_1p0 | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_dt_la_mrh_no_sparse | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_dt_la_score_input | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_dt_la_score_latent | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_dt_la_score_softmax | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_kgl | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_kgl_without_gat | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_kgl_without_kan | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_kgl_without_lstm | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_shcl_transformer | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_shcl_transformer_backbone_cnn | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_shcl_transformer_backbone_lstm | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_shcl_transformer_backbone_rnn | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_shcl_transformer_backbone_vae | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_shcl_transformer_backbone_vae_only_pair | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_shcl_transformer_backbone_vae_only_point | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_shcl_transformer_backbone_vae_random_mask | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_shcl_transformer_backbone_vae_reconstruction | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_shcl_transformer_only_pair | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_shcl_transformer_only_point | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_shcl_transformer_random_mask | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_comparator_shcl_transformer_reconstruction | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_moment_pretrained_base__TAB100 | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_moment_pretrained_base__release512 | 3.92 [1/1] | 10.03 [1/1] | 8.85 [1/1] | 20.29 [1/1] | — | — | — | — | — | — | — | — |
| fm_moment_pretrained_large__TAB100 | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_moment_pretrained_large__release512 | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_moment_pretrained_small__TAB100 | 5.11 [1/1] | 4.77 [1/1] | 7.73 [1/1] | 26.81 [1/1] | 2.64 [1/1] | 36.33 [1/1] | 3.80 [1/1] | 52.97 [1/1] | 19.76 [1/1] | 1.04 [1/1] | 3.49 [1/1] | 0.82 [1/1] |
| fm_moment_pretrained_small__release512 | 3.66 [1/1] | 10.56 [1/1] | 8.70 [1/1] | 17.48 [1/1] | 8.35 [1/1] | 42.62 [1/1] | 2.82 [1/1] | 58.13 [1/1] | 24.09 [1/1] | 4.68 [1/1] | 9.79 [1/1] | 0.33 [1/1] |
| fm_native_dfm_exact | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_native_dfm_hutchinson | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_native_grasp_er | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_native_grasp_grasp | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_native_grasp_lin | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_native_grasp_mean | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_objective_independent | 3.23 [5/5] | 23.53 [5/5] | 7.27 [5/5] | 41.07 [5/5] | 28.96 [5/5] | 55.52 [5/5] | 4.87 [5/5] | 73.06 [5/5] | 50.92 [5/5] | 16.48 [5/5] | 19.11 [5/5] | 0.41 [5/5] |
| fm_objective_ot | 3.09 [5/5] | 23.38 [5/5] | 7.36 [5/5] | 41.21 [5/5] | 28.95 [5/5] | 55.60 [5/5] | 4.87 [5/5] | 73.18 [5/5] | 50.97 [5/5] | 16.32 [5/5] | 20.02 [5/5] | 0.43 [5/5] |
| fm_objective_rectified | 3.23 [5/5] | 23.53 [5/5] | 7.27 [5/5] | 41.07 [5/5] | 28.96 [5/5] | 55.52 [5/5] | 4.87 [5/5] | 73.06 [5/5] | 50.92 [5/5] | 16.48 [5/5] | 19.11 [5/5] | 0.41 [5/5] |
| fm_objective_sb | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_objective_sb_stable_v2 | 3.34 [5/5] | 23.39 [5/5] | 7.62 [5/5] | 40.89 [5/5] | — | — | — | 73.15 [5/5] | 50.96 [5/5] | 15.58 [5/5] | 17.36 [4/5] | 0.41 [3/5] |
| fm_objective_sf2m | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_objective_sf2m_flow_only | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_objective_sf2m_flow_only_stable_v2 | 3.34 [5/5] | 23.39 [5/5] | 7.62 [5/5] | 40.59 [3/5] | — | — | — | 73.15 [5/5] | 50.96 [5/5] | 15.58 [5/5] | 17.36 [4/5] | 0.41 [3/5] |
| fm_objective_sf2m_stable_v2 | 3.28 [5/5] | 23.27 [5/5] | 7.57 [5/5] | 40.73 [5/5] | — | — | — | 73.06 [5/5] | 50.98 [5/5] | 15.43 [5/5] | 17.89 [4/5] | 0.40 [3/5] |
| fm_objective_target | 3.27 [5/5] | 23.51 [5/5] | 7.28 [5/5] | 40.95 [5/5] | 28.95 [5/5] | 55.45 [5/5] | 4.81 [5/5] | 73.21 [5/5] | 50.95 [5/5] | 16.43 [5/5] | 19.46 [5/5] | 0.40 [5/5] |
| fm_objective_vp | 3.00 [5/5] | 23.04 [5/5] | 7.71 [5/5] | 39.30 [5/5] | 30.08 [5/5] | 57.95 [5/5] | 5.77 [5/5] | 72.98 [5/5] | 51.02 [5/5] | 11.05 [5/5] | 17.92 [5/5] | 0.41 [5/5] |
| fm_rectified_40epoch_budget_control | 3.36 [5/5] | 23.29 [5/5] | 7.25 [5/5] | 41.87 [5/5] | 28.81 [5/5] | 55.54 [5/5] | 4.87 [5/5] | 73.11 [5/5] | 50.76 [5/5] | 16.69 [5/5] | 20.37 [5/5] | 0.41 [5/5] |
| fm_rectified_60epoch_budget_control | 3.37 [5/5] | 23.18 [5/5] | 7.23 [5/5] | 41.40 [5/5] | 28.75 [5/5] | 55.83 [5/5] | 4.84 [4/5] | 73.14 [5/5] | 50.71 [5/5] | 16.69 [5/5] | 19.97 [5/5] | 0.40 [5/5] |
| fm_reflow_2stage | 2.99 [5/5] | 23.03 [5/5] | 7.55 [5/5] | 39.02 [5/5] | 29.62 [5/5] | 57.77 [5/5] | 5.89 [5/5] | 73.04 [5/5] | 50.90 [5/5] | 8.62 [5/5] | 17.41 [5/5] | 0.41 [5/5] |
| fm_reflow_3stage | 3.01 [5/5] | 23.01 [5/5] | 7.50 [5/5] | 38.30 [5/5] | 30.09 [5/5] | 58.88 [5/5] | 6.12 [5/5] | 73.05 [5/5] | 50.91 [5/5] | 1.30 [5/5] | 13.00 [5/5] | 0.42 [5/5] |
| fm_strict_anomaly_transformer__msl__registered | — | — | 2.04 [5/5] | — | — | — | — | — | — | — | — | — |
| fm_strict_anomaly_transformer__psm__registered | 1.46 [5/5] | — | — | — | — | — | — | 2.64 [5/5] | 1.94 [5/5] | 0.86 [5/5] | 1.38 [5/5] | 4.97 [5/5] |
| fm_strict_anomaly_transformer__smap__registered | — | 4.08 [5/5] | — | — | — | — | — | — | — | — | — | — |
| fm_strict_anomaly_transformer__smd__registered | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_strict_anomaly_transformer__swat__registered | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_strict_dcdetector__msl__registered | — | — | 1.88 [5/5] | — | — | — | — | — | — | — | — | — |
| fm_strict_dcdetector__psm__registered | 1.73 [5/5] | — | — | — | — | — | — | 2.82 [5/5] | 2.26 [5/5] | 1.94 [5/5] | 2.16 [5/5] | 2.00 [5/5] |
| fm_strict_dcdetector__smap__registered | — | 0.90 [5/5] | — | — | — | — | — | — | — | — | — | — |
| fm_strict_dcdetector__smd__registered | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_strict_dcdetector__swat__registered | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_strict_itransformer__msl__registered | — | — | 7.99 [5/5] | — | — | — | — | — | — | — | — | — |
| fm_strict_itransformer__psm__registered | 4.84 [5/5] | — | — | — | — | — | — | 55.74 [5/5] | 19.58 [5/5] | 1.14 [5/5] | 3.60 [5/5] | 1.39 [5/5] |
| fm_strict_itransformer__smap__registered | — | 4.52 [5/5] | — | — | — | — | — | — | — | — | — | — |
| fm_strict_itransformer__smd__registered | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_strict_itransformer__swat__registered | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_strict_timesnet__msl__registered | — | — | 7.68 [5/5] | — | — | — | — | — | — | — | — | — |
| fm_strict_timesnet__psm__registered | 5.39 [5/5] | — | — | — | — | — | — | 61.24 [5/5] | 19.83 [5/5] | 1.35 [5/5] | 3.96 [5/5] | 0.44 [5/5] |
| fm_strict_timesnet__smap__registered | — | 3.89 [5/5] | — | — | — | — | — | — | — | — | — | — |
| fm_strict_timesnet__smd__registered | — | — | — | 23.85 [4/5] | — | — | — | — | — | — | — | — |
| fm_strict_timesnet__swat__registered | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_strict_tranad__msl__registered | — | — | 8.13 [5/5] | — | — | — | — | — | — | — | — | — |
| fm_strict_tranad__psm__registered | 2.76 [5/5] | — | — | — | — | — | — | 76.13 [5/5] | 50.98 [5/5] | 8.22 [5/5] | 23.45 [5/5] | 0.41 [5/5] |
| fm_strict_tranad__smap__registered | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_strict_tranad__smd__registered | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_strict_tranad__swat__registered | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_strict_usad__msl__registered | — | — | 6.52 [5/5] | — | — | — | — | — | — | — | — | — |
| fm_strict_usad__msl__signed_paper_loss | — | — | 13.78 [5/5] | — | — | — | — | — | — | — | — | — |
| fm_strict_usad__psm__registered | 2.88 [5/5] | — | — | — | — | — | — | 83.21 [5/5] | 51.60 [5/5] | 16.61 [5/5] | 25.22 [5/5] | 0.00 [5/5] |
| fm_strict_usad__psm__signed_paper_loss | 5.88 [5/5] | — | — | — | — | — | — | 83.14 [5/5] | 50.98 [5/5] | 3.49 [5/5] | 7.12 [5/5] | 0.00 [5/5] |
| fm_strict_usad__smap__registered | — | 24.99 [5/5] | — | — | — | — | — | — | — | — | — | — |
| fm_strict_usad__smap__signed_paper_loss | — | 6.25 [5/5] | — | — | — | — | — | — | — | — | — | — |
| fm_strict_usad__smd__registered | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_strict_usad__smd__signed_paper_loss | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_strict_usad__swat__registered | — | — | — | — | — | — | — | — | — | — | — | — |
| fm_strict_usad__swat__signed_paper_loss | — | — | — | — | — | — | — | — | — | — | — | — |

## 作者流程的核心结果（不同协议单列）

| 方法 / 配方 | 数据集 | 种子 | 协议 | Precision (%) | Recall (%) | F1 (%) | AUROC | AP |
|---|---|---:|---|---:|---:|---:|---:|---:|
| pi_transformer_historical_mirror / causal_axis_repaired/full | PSM | 1103 | 作者阈值 + PA；单种子历史镜像 | 97.41 | 98.37 | 97.89 | — | — |
| pi_transformer_historical_mirror / causal_axis_repaired/full | PSM | 1103 | 同一模型、同一阈值，去掉PA | 27.45 | 0.99 | 1.91 | 0.484267 | 0.272921 |
| crossad_complete / release_checkpoint | GECCO | 2025 | 发布权重；测试标签最优点级F1 | 45.06 | 77.53 | 57.00 | 0.986657 | 0.551428 |
| crossad_complete / release_checkpoint | GECCO | 2025 | 发布权重；验证分数99百分位，无PA | 63.56 | 29.86 | 40.63 | 0.986659 | 0.551428 |
| crossad_complete / release_checkpoint | MSL | 2025 | 发布权重；测试标签最优点级F1 | 21.70 | 76.32 | 33.80 | 0.783921 | 0.289417 |
| crossad_complete / release_checkpoint | MSL | 2025 | 发布权重；验证分数99百分位，无PA | 32.40 | 23.46 | 27.22 | 0.783911 | 0.289394 |

Pi历史镜像尚未认证期刊版作者等价；CrossAD为发布权重评价，未重新训练。以上值不能与主表混排。

完整指标还包括插补MAE/RMSE/CRPS/MRE、连续动力学MSE/MAE/RMSE及生成指标；均值、STD、SE、95%区间、论文值及来源见全量表。

逐种子明细含事件召回、漏检、离线延迟、误报率、参数量、训练/评分耗时与显存峰值。论文报告值、273条原始范围和681条方法库存在入口单列。尚未全部完成。
