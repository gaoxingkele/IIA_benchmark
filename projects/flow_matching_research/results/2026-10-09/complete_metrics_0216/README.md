# 算法—数据集完整实验指标

快照：2026-10-08T18:17:04.938331+00:00。异常检测源快照：2026-10-08T18:16:27.603235+00:00。

这里完整列出已登记任务和已有指标，未完成项留空。全论文、全消融实验尚未全部完成。

strict_summary.csv 覆盖所有已登记算法配置—数据集组合；strict_per_seed.csv 保留三档验证阈值、P/R/F1、AUROC、AP、事件召回、延迟、误报率、置信区间、参数量、耗时和显存。
TAB 的 VUS/Affiliation 单列，已保存全实体及未定义计数。all_tsad_numeric_metrics.csv 保留每个数值及原字段路径。历史与论文报告值分别存储。

**比较约束：**严格主表只在验证段按 1% 分位数定阈值，不做 PA。TAB 比例诊断合并训练/测试分数校准，最优比例依赖测试标签，不能充当严格泛化成绩。非重叠训练比部分作者 stride=1 少很多梯度更新，不能据此判定原论文无效。

| 算法配置 | 数据集 | 完成/预定种子 | Precision | Recall | 点级 F1 | AUROC | AP | VUS ROC（种子数） | VUS PR | Affiliation F（种子数） |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fm_objective_independent | MSL | 5/5 | 0.063057 | 0.085794 | 0.072689 | 0.545220 | 0.115527 | 0.705197 (5) | 0.278498 | 0.725564 (5) |
| fm_objective_independent | PSM | 5/5 | 0.958813 | 0.016447 | 0.032333 | 0.735128 | 0.550857 | 0.698167 (5) | 0.501818 | 0.585950 (5) |
| fm_objective_independent | SMAP | 5/5 | 0.273926 | 0.206164 | 0.235262 | 0.536360 | 0.146863 | 0.690551 (5) | 0.294041 | 0.537718 (5) |
| fm_objective_independent | SMD | 5/5 | 0.527819 | 0.336168 | 0.410668 | 0.767886 | 0.337081 | 0.814265 (5) | 0.433968 | 0.799890 (5) |
| fm_objective_independent | SWAT | 5/5 | 0.174640 | 0.848194 | 0.289642 | 0.822661 | 0.727248 | 0.678608 (5) | 0.482889 | 0.717671 (5) |
| fm_objective_ot | MSL | 5/5 | 0.063852 | 0.086985 | 0.073644 | 0.545315 | 0.115515 | 0.706013 (5) | 0.277971 | 0.728107 (5) |
| fm_objective_ot | PSM | 5/5 | 0.963411 | 0.015709 | 0.030904 | 0.733514 | 0.553258 | 0.697567 (5) | 0.502244 | 0.578248 (5) |
| fm_objective_ot | SMAP | 5/5 | 0.272459 | 0.204767 | 0.233811 | 0.539347 | 0.147495 | 0.691844 (5) | 0.294302 | 0.536096 (5) |
| fm_objective_ot | SMD | 5/5 | 0.523223 | 0.340192 | 0.412128 | 0.767478 | 0.337461 | 0.814468 (5) | 0.436098 | 0.799317 (5) |
| fm_objective_ot | SWAT | 5/5 | 0.174778 | 0.842987 | 0.289527 | 0.821859 | 0.724678 | 0.679549 (5) | 0.485110 | 0.715820 (5) |
| fm_objective_rectified | MSL | 5/5 | 0.063057 | 0.085794 | 0.072689 | 0.545220 | 0.115527 | 0.705197 (5) | 0.278498 | 0.725564 (5) |
| fm_objective_rectified | PSM | 5/5 | 0.958813 | 0.016447 | 0.032333 | 0.735128 | 0.550857 | 0.698167 (5) | 0.501818 | 0.585950 (5) |
| fm_objective_rectified | SMAP | 5/5 | 0.273926 | 0.206164 | 0.235262 | 0.536360 | 0.146863 | 0.690551 (5) | 0.294041 | 0.537718 (5) |
| fm_objective_rectified | SMD | 5/5 | 0.527819 | 0.336168 | 0.410668 | 0.767886 | 0.337081 | 0.814265 (5) | 0.433968 | 0.799890 (5) |
| fm_objective_rectified | SWAT | 5/5 | 0.174640 | 0.848194 | 0.289642 | 0.822661 | 0.727248 | 0.678608 (5) | 0.482889 | 0.717671 (5) |
| fm_objective_sb_stable_v2 | PSM | 5/5 | 0.961223 | 0.017005 | 0.033416 | 0.717722 | 0.545647 | 0.687497 (5) | 0.497809 | 0.587712 (5) |
| fm_objective_target | MSL | 5/5 | 0.063195 | 0.085900 | 0.072819 | 0.544558 | 0.115526 | 0.704934 (5) | 0.279624 | 0.724266 (5) |
| fm_objective_target | PSM | 5/5 | 0.962764 | 0.016652 | 0.032726 | 0.735262 | 0.550569 | 0.701574 (5) | 0.502289 | 0.581274 (5) |
| fm_objective_target | SMAP | 5/5 | 0.273892 | 0.206000 | 0.235143 | 0.535192 | 0.146747 | 0.690140 (5) | 0.294203 | 0.537754 (5) |
| fm_objective_target | SMD | 5/5 | 0.521915 | 0.337308 | 0.409530 | 0.767068 | 0.334713 | 0.813133 (5) | 0.433818 | 0.800311 (5) |
| fm_objective_target | SWAT | 5/5 | 0.174506 | 0.849036 | 0.289505 | 0.823257 | 0.726629 | 0.680010 (5) | 0.485990 | 0.718252 (5) |
| fm_objective_vp | MSL | 5/5 | 0.066787 | 0.091249 | 0.077124 | 0.541408 | 0.117051 | 0.702600 (5) | 0.284205 | 0.733116 (5) |
| fm_objective_vp | PSM | 5/5 | 0.991510 | 0.015225 | 0.029989 | 0.712565 | 0.551002 | 0.671146 (5) | 0.492057 | 0.522960 (5) |
| fm_objective_vp | SMAP | 5/5 | 0.269066 | 0.201408 | 0.230372 | 0.549575 | 0.149446 | 0.691672 (5) | 0.289387 | 0.533601 (5) |
| fm_objective_vp | SMD | 5/5 | 0.408308 | 0.379794 | 0.392979 | 0.756467 | 0.324163 | 0.804581 (5) | 0.431433 | 0.798092 (5) |
| fm_reflow_2stage | MSL | 5/5 | 0.065598 | 0.089051 | 0.075546 | 0.539035 | 0.117010 | 0.700867 (5) | 0.285871 | 0.732293 (5) |
| fm_reflow_2stage | PSM | 5/5 | 0.976299 | 0.015209 | 0.029949 | 0.725765 | 0.552078 | 0.685073 (5) | 0.497841 | 0.547103 (5) |
| fm_reflow_2stage | SMAP | 5/5 | 0.269386 | 0.201170 | 0.230333 | 0.547532 | 0.148794 | 0.692927 (5) | 0.291242 | 0.535977 (5) |
| fm_reflow_2stage | SMD | 5/5 | 0.431110 | 0.356978 | 0.390249 | 0.753078 | 0.322908 | 0.795336 (5) | 0.420790 | 0.799751 (5) |
| fm_strict_anomaly_transformer__psm__registered | PSM | 5/5 | 0.421627 | 0.007416 | 0.014575 | 0.513439 | 0.295091 | 0.432373 (5) | 0.291488 | 0.635304 (5) |
| fm_strict_timesnet__psm__registered | PSM | 5/5 | 0.692853 | 0.028071 | 0.053941 | 0.598985 | 0.396761 | 0.597688 (5) | 0.398548 | 0.767881 (5) |
| fm_objective_sf2m_stable_v2 | PSM | 4/5 | 0.966661 | 0.016386 | 0.032222 | 0.719697 | 0.548439 | 0.687794 (4) | 0.499966 | 0.578675 (4) |
| fm_objective_vp | SWAT | 4/5 | 0.183880 | 0.823136 | 0.300606 | 0.821763 | 0.728644 | 0.683171 (3) | 0.490392 | 0.703010 (3) |
| fm_strict_dcdetector__psm__registered | PSM | 2/5 | 0.286382 | 0.009249 | 0.017919 | 0.501045 | 0.278295 | 0.439577 (2) | 0.280859 | 0.642914 (2) |

各指标可能由不同数量的已完成种子产生，范围指标种子数必须一起读取；不混合缺失值为零。独立 CFM 与单阶段 Rectified 在这里 sigma=0 时等价。迭代 reflow 独立列行，其完整队列尚未全部完成。

## 插补实验已有结果

| 算法 | 数据集 | 缺失率 | 掩码/协议 | 指标 | 均值 | SE | 完成/预定折或种子 | 论文值 | 比较状态 |
|---|---|---:|---|---|---:|---:|---:|---:|---|
| csdi | PhysioNet2012 | 0.1 | paper_mask / author_protocol | mae | 0.218292 | 0.000629 | 5/5 | 0.217000 | numerically_compatible |
| csdi | PhysioNet2012 | 0.1 | paper_mask / author_protocol | rmse | 0.505890 | 0.021293 | 5/5 | — | no_paper_reference |
| csdi | PhysioNet2012 | 0.1 | paper_mask / author_protocol | normalized_crps | 0.239517 | 0.001311 | 5/5 | 0.238000 | numerically_compatible |
| cfmi | PhysioNet2012 | 0.1 | paper_mask / author_protocol | mae | 0.241406 | 0.002110 | 5/5 | 0.234000 | numerically_compatible |
| cfmi | PhysioNet2012 | 0.1 | paper_mask / author_protocol | rmse | 0.549803 | 0.045544 | 5/5 | 0.538000 | numerically_compatible |
| cfmi | PhysioNet2012 | 0.1 | paper_mask / author_protocol | crps | 0.241007 | 0.007659 | 5/5 | 0.220000 | numerically_compatible |
| cfmi | PhysioNet2012 | 0.1 | paper_mask / author_protocol | normalized_crps | 0.255534 | 0.001830 | 5/5 | 0.250000 | numerically_compatible |
| csdi | PhysioNet2012 | 0.5 | paper_mask / author_protocol | mae | 0.301308 | 0.001173 | 5/5 | 0.301000 | numerically_compatible |
| csdi | PhysioNet2012 | 0.5 | paper_mask / author_protocol | rmse | 0.624481 | 0.032961 | 5/5 | — | no_paper_reference |
| csdi | PhysioNet2012 | 0.5 | paper_mask / author_protocol | normalized_crps | 0.330745 | 0.001490 | 5/5 | 0.330000 | numerically_compatible |
| cfmi | PhysioNet2012 | 0.5 | paper_mask / author_protocol | mae | 0.337274 | 0.002894 | 5/5 | 0.319000 | differs_from_paper |
| cfmi | PhysioNet2012 | 0.5 | paper_mask / author_protocol | rmse | 0.660839 | 0.016612 | 5/5 | 0.624000 | numerically_compatible |
| cfmi | PhysioNet2012 | 0.5 | paper_mask / author_protocol | crps | 0.331166 | 0.013786 | 5/5 | 0.282000 | differs_from_paper |
| cfmi | PhysioNet2012 | 0.5 | paper_mask / author_protocol | normalized_crps | 0.352914 | 0.002716 | 5/5 | 0.338000 | numerically_compatible |
| csdi | PhysioNet2012 | 0.9 | paper_mask / author_protocol | mae | 0.474957 | 0.001664 | 5/5 | 0.481000 | numerically_compatible |
| csdi | PhysioNet2012 | 0.9 | paper_mask / author_protocol | rmse | 0.793024 | 0.017760 | 5/5 | — | no_paper_reference |
| csdi | PhysioNet2012 | 0.9 | paper_mask / author_protocol | normalized_crps | 0.517923 | 0.002074 | 5/5 | 0.522000 | numerically_compatible |
| cfmi | PhysioNet2012 | 0.9 | paper_mask / author_protocol | mae | 0.554975 | 0.008909 | 5/5 | 0.504000 | differs_from_paper |
| cfmi | PhysioNet2012 | 0.9 | paper_mask / author_protocol | rmse | 0.887946 | 0.012818 | 5/5 | 0.822000 | differs_from_paper |
| cfmi | PhysioNet2012 | 0.9 | paper_mask / author_protocol | crps | 0.511680 | 0.032079 | 5/5 | 0.410000 | differs_from_paper |
| cfmi | PhysioNet2012 | 0.9 | paper_mask / author_protocol | normalized_crps | 0.580323 | 0.008218 | 5/5 | 0.538000 | numerically_compatible |
| csdi | Air-36 | None | historical_air36 / author_protocol | mae | 9.585215 | 0.089509 | 5/5 | 9.600000 | not_comparable |
| csdi | Air-36 | None | historical_air36 / author_protocol | rmse | 18.684381 | 0.313362 | 5/5 | — | no_paper_reference |
| csdi | Air-36 | None | historical_air36 / author_protocol | normalized_crps | 0.105955 | 0.001077 | 5/5 | 0.108000 | not_comparable |
| cfmi | Air-36 | None | historical_air36 / author_protocol | mae | 9.539493 | 0.090056 | 4/5 | 9.404000 | incomplete_repeats |
| cfmi | Air-36 | None | historical_air36 / author_protocol | rmse | 18.178026 | 0.255409 | 4/5 | 17.701000 | incomplete_repeats |
| cfmi | Air-36 | None | historical_air36 / author_protocol | crps | 6.807013 | 0.061356 | 4/5 | 6.726000 | incomplete_repeats |
| cfmi | Air-36 | None | historical_air36 / author_protocol | normalized_crps | 0.099229 | 0.000803 | 4/5 | 0.098000 | incomplete_repeats |

数值兼容筛查不等于统计等价证明；not_comparable、incomplete_repeats 等原因详见 imputation_summary.csv。插补误差不能代替异常检测成绩。

## 文件

[完整 Excel](complete_experiment_metrics.xlsx)、[所有组合与状态](strict_summary.csv)、[逐种子严格指标](strict_per_seed.csv)、[历史协议结果](historical_protocol_results.csv)、[插补汇总](imputation_summary.csv)、[论文报告值](paper_claims.csv)、[完整机器可读结果](complete_results.json)。

原文数据范围、未运行任务及每篇复现缺口均保留在 CSV/JSON 与 Excel 中。
