# 算法—数据集完整实验指标

快照：2026-10-08T23:35:06.926367+00:00。异常检测源快照：2026-10-08T23:33:00.518702+00:00。

这里完整列出已登记任务和已有指标，未完成项留空。全论文、全消融实验尚未全部完成。

strict_summary.csv 覆盖所有已登记算法配置—数据集组合；strict_per_seed.csv 保留三档验证阈值、P/R/F1、AUROC、AP、事件召回、延迟、误报率、置信区间、参数量、耗时和显存。
TAB 的 VUS/Affiliation 单列，已保存全实体及未定义计数。all_tsad_numeric_metrics.csv 保留每个数值及原字段路径。历史与论文报告值分别存储。

**比较约束：**严格主表使用验证分数第99百分位（1%尾部报警比例）定阈值，不做 PA。TAB 比例诊断合并训练/测试分数校准，最优比例依赖测试标签，不能充当严格泛化成绩。非重叠训练比部分作者 stride=1 少很多梯度更新，不能据此判定原论文无效。

| 算法配置 | 数据集 | 完成/预定种子 | Precision | Recall | 点级 F1 | AUROC | AP | VUS ROC（种子数） | VUS PR | Affiliation F（种子数） |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fm_objective_independent | CICIDS | 5/5 | 0.537502 | 0.025511 | 0.048710 | 0.620327 | 0.392767 | 0.695999 (1) | 0.342319 | 0.346026 (1) |
| fm_objective_independent | MSL | 5/5 | 0.063057 | 0.085794 | 0.072689 | 0.545220 | 0.115527 | 0.705197 (5) | 0.278498 | 0.725564 (5) |
| fm_objective_independent | PRONTO_DAY2 | 5/5 | 1.000000 | 0.089776 | 0.164760 | 0.717108 | 0.975313 | 0.976135 (5) | 0.998642 | 0.557584 (5) |
| fm_objective_independent | PRONTO_DAY3 | 5/5 | 0.719969 | 0.110212 | 0.191145 | 0.377849 | 0.727954 | 0.904335 (5) | 0.971246 | 0.666297 (5) |
| fm_objective_independent | PRONTO_DAY4 | 5/5 | 1.000000 | 0.002065 | 0.004121 | 0.463218 | 0.377897 | 0.646314 (5) | 0.629969 | 0.390340 (5) |
| fm_objective_independent | PSM | 5/5 | 0.958813 | 0.016447 | 0.032333 | 0.735128 | 0.550857 | 0.698167 (5) | 0.501818 | 0.585950 (5) |
| fm_objective_independent | SKAB | 5/5 | 0.390021 | 0.733302 | 0.509209 | 0.637441 | 0.549567 | 0.797160 (5) | 0.736057 | 0.774492 (5) |
| fm_objective_independent | SMAP | 5/5 | 0.273926 | 0.206164 | 0.235262 | 0.536360 | 0.146863 | 0.690551 (5) | 0.294041 | 0.537718 (5) |
| fm_objective_independent | SMD | 5/5 | 0.527819 | 0.336168 | 0.410668 | 0.767886 | 0.337081 | 0.814265 (5) | 0.433968 | 0.799890 (5) |
| fm_objective_independent | SWAN | 5/5 | 0.743857 | 0.442935 | 0.555188 | 0.807976 | 0.674500 | 0.715689 (5) | 0.585292 | 0.656444 (5) |
| fm_objective_independent | SWAT | 5/5 | 0.174640 | 0.848194 | 0.289642 | 0.822661 | 0.727248 | 0.678608 (5) | 0.482889 | 0.717671 (5) |
| fm_objective_independent | TEP_CLASSIC | 5/5 | 0.998597 | 0.576024 | 0.730604 | 0.889980 | 0.977490 | 0.976290 (5) | 0.995223 | 0.942948 (5) |
| fm_objective_ot | CICIDS | 5/5 | 0.536006 | 0.025489 | 0.048663 | 0.626055 | 0.393674 | — (0) | — | — (0) |
| fm_objective_ot | MSL | 5/5 | 0.063852 | 0.086985 | 0.073644 | 0.545315 | 0.115515 | 0.706013 (5) | 0.277971 | 0.728107 (5) |
| fm_objective_ot | PRONTO_DAY2 | 5/5 | 1.000000 | 0.088836 | 0.163174 | 0.710470 | 0.974522 | 0.975452 (5) | 0.998594 | 0.557396 (5) |
| fm_objective_ot | PRONTO_DAY3 | 5/5 | 0.727682 | 0.116090 | 0.200215 | 0.381933 | 0.728847 | 0.904840 (5) | 0.971282 | 0.671914 (5) |
| fm_objective_ot | PRONTO_DAY4 | 5/5 | 1.000000 | 0.002147 | 0.004286 | 0.462867 | 0.377187 | 0.645902 (5) | 0.628389 | 0.390393 (5) |
| fm_objective_ot | PSM | 5/5 | 0.963411 | 0.015709 | 0.030904 | 0.733514 | 0.553258 | 0.697567 (5) | 0.502244 | 0.578248 (5) |
| fm_objective_ot | SKAB | 5/5 | 0.390276 | 0.734250 | 0.509655 | 0.638719 | 0.550659 | 0.798622 (5) | 0.737487 | 0.773416 (5) |
| fm_objective_ot | SMAP | 5/5 | 0.272459 | 0.204767 | 0.233811 | 0.539347 | 0.147495 | 0.691844 (5) | 0.294302 | 0.536096 (5) |
| fm_objective_ot | SMD | 5/5 | 0.523223 | 0.340192 | 0.412128 | 0.767478 | 0.337461 | 0.814468 (5) | 0.436098 | 0.799317 (5) |
| fm_objective_ot | SWAN | 5/5 | 0.753168 | 0.440627 | 0.555968 | 0.810315 | 0.677442 | 0.720770 (5) | 0.589988 | 0.660748 (5) |
| fm_objective_ot | SWAT | 5/5 | 0.174778 | 0.842987 | 0.289527 | 0.821859 | 0.724678 | 0.679549 (5) | 0.485110 | 0.715820 (5) |
| fm_objective_ot | TEP_CLASSIC | 5/5 | 0.998335 | 0.577643 | 0.731832 | 0.890985 | 0.977679 | 0.976236 (5) | 0.995193 | 0.945853 (5) |
| fm_objective_rectified | CICIDS | 5/5 | 0.537502 | 0.025511 | 0.048710 | 0.620327 | 0.392767 | — (0) | — | — (0) |
| fm_objective_rectified | MSL | 5/5 | 0.063057 | 0.085794 | 0.072689 | 0.545220 | 0.115527 | 0.705197 (5) | 0.278498 | 0.725564 (5) |
| fm_objective_rectified | PRONTO_DAY2 | 5/5 | 1.000000 | 0.089776 | 0.164760 | 0.717108 | 0.975313 | 0.976135 (5) | 0.998642 | 0.557584 (5) |
| fm_objective_rectified | PRONTO_DAY3 | 5/5 | 0.719969 | 0.110212 | 0.191145 | 0.377849 | 0.727954 | 0.904335 (5) | 0.971246 | 0.666297 (5) |
| fm_objective_rectified | PRONTO_DAY4 | 5/5 | 1.000000 | 0.002065 | 0.004121 | 0.463218 | 0.377897 | 0.646314 (5) | 0.629969 | 0.390340 (5) |
| fm_objective_rectified | PSM | 5/5 | 0.958813 | 0.016447 | 0.032333 | 0.735128 | 0.550857 | 0.698167 (5) | 0.501818 | 0.585950 (5) |
| fm_objective_rectified | SKAB | 5/5 | 0.390021 | 0.733302 | 0.509209 | 0.637441 | 0.549567 | 0.797160 (5) | 0.736057 | 0.774492 (5) |
| fm_objective_rectified | SMAP | 5/5 | 0.273926 | 0.206164 | 0.235262 | 0.536360 | 0.146863 | 0.690551 (5) | 0.294041 | 0.537718 (5) |
| fm_objective_rectified | SMD | 5/5 | 0.527819 | 0.336168 | 0.410668 | 0.767886 | 0.337081 | 0.814265 (5) | 0.433968 | 0.799890 (5) |
| fm_objective_rectified | SWAN | 5/5 | 0.743857 | 0.442935 | 0.555188 | 0.807976 | 0.674500 | 0.715689 (5) | 0.585292 | 0.656444 (5) |
| fm_objective_rectified | SWAT | 5/5 | 0.174640 | 0.848194 | 0.289642 | 0.822661 | 0.727248 | 0.678608 (5) | 0.482889 | 0.717671 (5) |
| fm_objective_rectified | TEP_CLASSIC | 5/5 | 0.998597 | 0.576024 | 0.730604 | 0.889980 | 0.977490 | 0.976290 (5) | 0.995223 | 0.942948 (5) |
| fm_objective_sb_stable_v2 | MSL | 5/5 | 0.066049 | 0.090163 | 0.076244 | 0.534781 | 0.113941 | — (0) | — | — (0) |
| fm_objective_sb_stable_v2 | PRONTO_DAY2 | 5/5 | 1.000000 | 0.084493 | 0.155779 | 0.710824 | 0.974961 | 0.976188 (5) | 0.998645 | 0.557018 (5) |
| fm_objective_sb_stable_v2 | PSM | 5/5 | 0.961223 | 0.017005 | 0.033416 | 0.717722 | 0.545647 | 0.687497 (5) | 0.497809 | 0.587712 (5) |
| fm_objective_sb_stable_v2 | SKAB | 5/5 | 0.390226 | 0.734203 | 0.509601 | 0.644456 | 0.563593 | 0.807162 (5) | 0.750735 | 0.773940 (5) |
| fm_objective_sb_stable_v2 | SMAP | 5/5 | 0.272568 | 0.204790 | 0.233867 | 0.540958 | 0.147982 | — (0) | — | — (0) |
| fm_objective_sb_stable_v2 | TEP_CLASSIC | 5/5 | 0.998335 | 0.577214 | 0.731479 | 0.887587 | 0.976978 | 0.975428 (5) | 0.995032 | 0.948819 (5) |
| fm_objective_sf2m_flow_only_stable_v2 | MSL | 5/5 | 0.066049 | 0.090163 | 0.076244 | 0.534781 | 0.113941 | — (0) | — | — (0) |
| fm_objective_sf2m_flow_only_stable_v2 | PRONTO_DAY2 | 5/5 | 1.000000 | 0.084493 | 0.155779 | 0.710824 | 0.974961 | 0.976188 (5) | 0.998645 | 0.557018 (5) |
| fm_objective_sf2m_flow_only_stable_v2 | PSM | 5/5 | 0.961223 | 0.017005 | 0.033416 | 0.717722 | 0.545647 | — (0) | — | — (0) |
| fm_objective_sf2m_flow_only_stable_v2 | SKAB | 5/5 | 0.390226 | 0.734203 | 0.509601 | 0.644456 | 0.563593 | 0.807162 (5) | 0.750735 | 0.773940 (5) |
| fm_objective_sf2m_flow_only_stable_v2 | SMAP | 5/5 | 0.272568 | 0.204790 | 0.233867 | 0.540958 | 0.147982 | — (0) | — | — (0) |
| fm_objective_sf2m_flow_only_stable_v2 | TEP_CLASSIC | 5/5 | 0.998335 | 0.577214 | 0.731479 | 0.887587 | 0.976978 | 0.975428 (5) | 0.995032 | 0.948819 (5) |
| fm_objective_sf2m_stable_v2 | MSL | 5/5 | 0.065707 | 0.089263 | 0.075694 | 0.533194 | 0.115100 | — (0) | — | — (0) |
| fm_objective_sf2m_stable_v2 | PRONTO_DAY2 | 5/5 | 1.000000 | 0.083621 | 0.154316 | 0.708870 | 0.974728 | 0.975919 (5) | 0.998628 | 0.557062 (5) |
| fm_objective_sf2m_stable_v2 | PSM | 5/5 | 0.965382 | 0.016677 | 0.032784 | 0.718805 | 0.547731 | 0.687794 (4) | 0.499966 | 0.578675 (4) |
| fm_objective_sf2m_stable_v2 | SKAB | 5/5 | 0.390366 | 0.734576 | 0.509810 | 0.646947 | 0.568899 | 0.811590 (5) | 0.756376 | 0.774487 (5) |
| fm_objective_sf2m_stable_v2 | SMAP | 5/5 | 0.271455 | 0.203683 | 0.232735 | 0.541786 | 0.148027 | — (0) | — | — (0) |
| fm_objective_sf2m_stable_v2 | TEP_CLASSIC | 5/5 | 0.998249 | 0.576131 | 0.730588 | 0.888192 | 0.977092 | 0.975403 (5) | 0.995018 | 0.944569 (5) |
| fm_objective_target | CICIDS | 5/5 | 0.537548 | 0.025157 | 0.048064 | 0.619822 | 0.392276 | — (0) | — | — (0) |
| fm_objective_target | MSL | 5/5 | 0.063195 | 0.085900 | 0.072819 | 0.544558 | 0.115526 | 0.704934 (5) | 0.279624 | 0.724266 (5) |
| fm_objective_target | PRONTO_DAY2 | 5/5 | 1.000000 | 0.089502 | 0.164299 | 0.718803 | 0.975536 | 0.976382 (5) | 0.998662 | 0.557522 (5) |
| fm_objective_target | PRONTO_DAY3 | 5/5 | 0.720158 | 0.112544 | 0.194646 | 0.380926 | 0.728535 | 0.904979 (5) | 0.971362 | 0.663851 (5) |
| fm_objective_target | PRONTO_DAY4 | 5/5 | 1.000000 | 0.001982 | 0.003957 | 0.463794 | 0.378149 | 0.646883 (5) | 0.630595 | 0.390339 (5) |
| fm_objective_target | PSM | 5/5 | 0.962764 | 0.016652 | 0.032726 | 0.735262 | 0.550569 | 0.701574 (5) | 0.502289 | 0.581274 (5) |
| fm_objective_target | SKAB | 5/5 | 0.390134 | 0.733970 | 0.509466 | 0.637567 | 0.550253 | 0.797759 (5) | 0.737317 | 0.772947 (5) |
| fm_objective_target | SMAP | 5/5 | 0.273892 | 0.206000 | 0.235143 | 0.535192 | 0.146747 | 0.690140 (5) | 0.294203 | 0.537754 (5) |
| fm_objective_target | SMD | 5/5 | 0.521915 | 0.337308 | 0.409530 | 0.767068 | 0.334713 | 0.813133 (5) | 0.433818 | 0.800311 (5) |
| fm_objective_target | SWAN | 5/5 | 0.742943 | 0.442478 | 0.554533 | 0.806938 | 0.673441 | 0.714948 (5) | 0.584402 | 0.656056 (5) |
| fm_objective_target | SWAT | 5/5 | 0.174506 | 0.849036 | 0.289505 | 0.823257 | 0.726629 | 0.680010 (5) | 0.485990 | 0.718252 (5) |
| fm_objective_target | TEP_CLASSIC | 5/5 | 0.998377 | 0.577940 | 0.732081 | 0.888648 | 0.977200 | 0.976239 (5) | 0.995212 | 0.946545 (5) |
| fm_objective_vp | CICIDS | 5/5 | 0.574280 | 0.030402 | 0.057745 | 0.634149 | 0.397584 | — (0) | — | — (0) |
| fm_objective_vp | MSL | 5/5 | 0.066787 | 0.091249 | 0.077124 | 0.541408 | 0.117051 | 0.702600 (5) | 0.284205 | 0.733116 (5) |
| fm_objective_vp | PRONTO_DAY2 | 5/5 | 1.000000 | 0.058472 | 0.110471 | 0.710927 | 0.974623 | 0.975270 (5) | 0.998589 | 0.555089 (5) |
| fm_objective_vp | PRONTO_DAY3 | 5/5 | 0.710463 | 0.102615 | 0.179236 | 0.379484 | 0.727503 | 0.903143 (5) | 0.970629 | 0.650789 (5) |
| fm_objective_vp | PRONTO_DAY4 | 5/5 | 1.000000 | 0.002065 | 0.004121 | 0.426152 | 0.356261 | 0.616521 (5) | 0.600420 | 0.390340 (5) |
| fm_objective_vp | PSM | 5/5 | 0.991510 | 0.015225 | 0.029989 | 0.712565 | 0.551002 | 0.671146 (5) | 0.492057 | 0.522960 (5) |
| fm_objective_vp | SKAB | 5/5 | 0.390611 | 0.735352 | 0.510206 | 0.658022 | 0.591310 | 0.831504 (5) | 0.783663 | 0.776964 (5) |
| fm_objective_vp | SMAP | 5/5 | 0.269066 | 0.201408 | 0.230372 | 0.549575 | 0.149446 | 0.691672 (5) | 0.289387 | 0.533601 (5) |
| fm_objective_vp | SMD | 5/5 | 0.408308 | 0.379794 | 0.392979 | 0.756467 | 0.324163 | 0.804581 (5) | 0.431433 | 0.798092 (5) |
| fm_objective_vp | SWAN | 5/5 | 0.757899 | 0.469114 | 0.579502 | 0.836403 | 0.707246 | 0.747169 (5) | 0.617598 | 0.671753 (5) |
| fm_objective_vp | SWAT | 5/5 | 0.184022 | 0.823160 | 0.300797 | 0.821427 | 0.728332 | 0.682084 (5) | 0.487826 | 0.703106 (5) |
| fm_objective_vp | TEP_CLASSIC | 5/5 | 0.998512 | 0.575036 | 0.729787 | 0.891685 | 0.977797 | 0.976332 (5) | 0.995198 | 0.938951 (5) |
| fm_rectified_40epoch_budget_control | CICIDS | 5/5 | 0.535327 | 0.025496 | 0.048674 | 0.619370 | 0.394101 | — (0) | — | — (0) |
| fm_rectified_40epoch_budget_control | MSL | 5/5 | 0.062908 | 0.085608 | 0.072524 | 0.542974 | 0.114567 | — (0) | — | — (0) |
| fm_rectified_40epoch_budget_control | PRONTO_DAY2 | 5/5 | 1.000000 | 0.091058 | 0.166917 | 0.747698 | 0.978831 | 0.979421 (5) | 0.998867 | 0.557791 (5) |
| fm_rectified_40epoch_budget_control | PRONTO_DAY3 | 5/5 | 0.732350 | 0.118339 | 0.203669 | 0.426199 | 0.741105 | — (0) | — | — (0) |
| fm_rectified_40epoch_budget_control | PRONTO_DAY4 | 5/5 | 1.000000 | 0.002065 | 0.004121 | 0.471913 | 0.385524 | 0.653947 (5) | 0.641289 | 0.390389 (5) |
| fm_rectified_40epoch_budget_control | PSM | 5/5 | 0.957454 | 0.017087 | 0.033572 | 0.737732 | 0.550026 | — (0) | — | — (0) |
| fm_rectified_40epoch_budget_control | SKAB | 5/5 | 0.389012 | 0.730119 | 0.507581 | 0.624892 | 0.527667 | 0.780206 (5) | 0.712742 | 0.773067 (5) |
| fm_rectified_40epoch_budget_control | SMAP | 5/5 | 0.271670 | 0.203846 | 0.232921 | 0.535617 | 0.146476 | — (0) | — | — (0) |
| fm_rectified_40epoch_budget_control | SMD | 5/5 | 0.562429 | 0.334025 | 0.418723 | 0.772170 | 0.340303 | — (0) | — | — (0) |
| fm_rectified_40epoch_budget_control | SWAN | 5/5 | 0.745139 | 0.442806 | 0.555437 | 0.804995 | 0.672827 | — (0) | — | — (0) |
| fm_rectified_40epoch_budget_control | SWAT | 5/5 | 0.172858 | 0.864393 | 0.288095 | 0.822282 | 0.723544 | — (0) | — | — (0) |
| fm_rectified_40epoch_budget_control | TEP_CLASSIC | 5/5 | 0.998474 | 0.576631 | 0.731061 | 0.888723 | 0.977260 | 0.976414 (5) | 0.995308 | 0.944949 (5) |
| fm_rectified_60epoch_budget_control | MSL | 5/5 | 0.062751 | 0.085344 | 0.072324 | 0.543456 | 0.114749 | — (0) | — | — (0) |
| fm_rectified_60epoch_budget_control | PRONTO_DAY2 | 5/5 | 1.000000 | 0.091075 | 0.166946 | 0.740475 | 0.978277 | 0.978899 (5) | 0.998840 | 0.557825 (5) |
| fm_rectified_60epoch_budget_control | PRONTO_DAY3 | 5/5 | 0.730558 | 0.115654 | 0.199669 | 0.441464 | 0.745232 | — (0) | — | — (0) |
| fm_rectified_60epoch_budget_control | PRONTO_DAY4 | 5/5 | 1.000000 | 0.002024 | 0.004039 | 0.472890 | 0.386122 | 0.655000 (5) | 0.643326 | 0.390340 (5) |
| fm_rectified_60epoch_budget_control | PSM | 5/5 | 0.953541 | 0.017169 | 0.033730 | 0.736344 | 0.551276 | — (0) | — | — (0) |
| fm_rectified_60epoch_budget_control | SKAB | 5/5 | 0.388754 | 0.729218 | 0.507144 | 0.619909 | 0.521104 | 0.774063 (5) | 0.706324 | 0.775239 (5) |
| fm_rectified_60epoch_budget_control | SMAP | 5/5 | 0.270668 | 0.202750 | 0.231837 | 0.535460 | 0.146316 | — (0) | — | — (0) |
| fm_rectified_60epoch_budget_control | SMD | 5/5 | 0.623837 | 0.311497 | 0.413955 | 0.774513 | 0.341945 | — (0) | — | — (0) |
| fm_rectified_60epoch_budget_control | SWAN | 5/5 | 0.743784 | 0.446965 | 0.558325 | 0.806741 | 0.675242 | — (0) | — | — (0) |
| fm_rectified_60epoch_budget_control | SWAT | 5/5 | 0.171579 | 0.885941 | 0.287477 | 0.821908 | 0.723513 | — (0) | — | — (0) |
| fm_rectified_60epoch_budget_control | TEP_CLASSIC | 5/5 | 0.998579 | 0.577060 | 0.731432 | 0.890682 | 0.977676 | 0.976417 (5) | 0.995317 | 0.941918 (5) |
| fm_reflow_2stage | CICIDS | 5/5 | 0.580432 | 0.031007 | 0.058868 | 0.621817 | 0.390273 | 0.697292 (4) | 0.345104 | 0.348105 (4) |
| fm_reflow_2stage | MSL | 5/5 | 0.065598 | 0.089051 | 0.075546 | 0.539035 | 0.117010 | 0.700867 (5) | 0.285871 | 0.732293 (5) |
| fm_reflow_2stage | PRONTO_DAY2 | 5/5 | 1.000000 | 0.045050 | 0.086194 | 0.726795 | 0.976459 | 0.977087 (5) | 0.998712 | 0.554312 (5) |
| fm_reflow_2stage | PRONTO_DAY3 | 5/5 | 0.699425 | 0.099647 | 0.174145 | 0.385813 | 0.729001 | 0.904377 (5) | 0.970878 | 0.653527 (5) |
| fm_reflow_2stage | PRONTO_DAY4 | 5/5 | 1.000000 | 0.002065 | 0.004121 | 0.418756 | 0.352676 | 0.611032 (5) | 0.595473 | 0.390340 (5) |
| fm_reflow_2stage | PSM | 5/5 | 0.976299 | 0.015209 | 0.029949 | 0.725765 | 0.552078 | 0.685073 (5) | 0.497841 | 0.547103 (5) |
| fm_reflow_2stage | SKAB | 5/5 | 0.389903 | 0.733038 | 0.509045 | 0.641226 | 0.556431 | 0.803590 (5) | 0.745465 | 0.774031 (5) |
| fm_reflow_2stage | SMAP | 5/5 | 0.269386 | 0.201170 | 0.230333 | 0.547532 | 0.148794 | 0.692927 (5) | 0.291242 | 0.535977 (5) |
| fm_reflow_2stage | SMD | 5/5 | 0.431110 | 0.356978 | 0.390249 | 0.753078 | 0.322908 | 0.795336 (5) | 0.420790 | 0.799751 (5) |
| fm_reflow_2stage | SWAN | 5/5 | 0.765053 | 0.464189 | 0.577738 | 0.830896 | 0.703194 | 0.745971 (5) | 0.616491 | 0.675825 (5) |
| fm_reflow_2stage | SWAT | 5/5 | 0.180577 | 0.823877 | 0.296223 | 0.820484 | 0.725875 | 0.680466 (5) | 0.487292 | 0.706191 (5) |
| fm_reflow_2stage | TEP_CLASSIC | 5/5 | 0.998576 | 0.575833 | 0.730444 | 0.890811 | 0.977647 | 0.976223 (5) | 0.995191 | 0.939546 (5) |
| fm_reflow_3stage | CICIDS | 5/5 | 0.587823 | 0.032280 | 0.061199 | 0.624536 | 0.391916 | — (0) | — | — (0) |
| fm_reflow_3stage | MSL | 5/5 | 0.065159 | 0.088389 | 0.075017 | 0.538887 | 0.116631 | — (0) | — | — (0) |
| fm_reflow_3stage | PRONTO_DAY2 | 5/5 | 1.000000 | 0.006531 | 0.012962 | 0.733609 | 0.977437 | 0.977928 (5) | 0.998776 | 0.494028 (5) |
| fm_reflow_3stage | PRONTO_DAY3 | 5/5 | 0.639085 | 0.072426 | 0.130011 | 0.385244 | 0.728009 | — (0) | — | — (0) |
| fm_reflow_3stage | PRONTO_DAY4 | 5/5 | 1.000000 | 0.002106 | 0.004203 | 0.400798 | 0.344741 | 0.595280 (5) | 0.579246 | 0.390389 (5) |
| fm_reflow_3stage | PSM | 5/5 | 0.981046 | 0.015299 | 0.030127 | 0.720384 | 0.550238 | — (0) | — | — (0) |
| fm_reflow_3stage | SKAB | 5/5 | 0.389946 | 0.733163 | 0.509112 | 0.641799 | 0.557373 | 0.804914 (5) | 0.747594 | 0.777073 (5) |
| fm_reflow_3stage | SMAP | 5/5 | 0.269146 | 0.200999 | 0.230133 | 0.553047 | 0.149801 | — (0) | — | — (0) |
| fm_reflow_3stage | SMD | 5/5 | 0.398748 | 0.369052 | 0.382961 | 0.749359 | 0.319022 | — (0) | — | — (0) |
| fm_reflow_3stage | SWAN | 5/5 | 0.760988 | 0.480249 | 0.588824 | 0.842724 | 0.716953 | — (0) | — | — (0) |
| fm_reflow_3stage | SWAT | 5/5 | 0.184143 | 0.823053 | 0.300950 | 0.820895 | 0.726619 | — (0) | — | — (0) |
| fm_reflow_3stage | TEP_CLASSIC | 5/5 | 0.998495 | 0.575964 | 0.730526 | 0.891304 | 0.977736 | 0.976181 (5) | 0.995186 | 0.939388 (5) |
| fm_strict_anomaly_transformer__msl__registered | MSL | 5/5 | 0.068007 | 0.011995 | 0.020377 | 0.499501 | 0.103149 | — (0) | — | — (0) |
| fm_strict_anomaly_transformer__psm__registered | PRONTO_DAY2 | 5/5 | 0.996552 | 0.004308 | 0.008578 | 0.523648 | 0.950468 | 0.967297 (5) | 0.997646 | 0.905712 (5) |
| fm_strict_anomaly_transformer__psm__registered | PRONTO_DAY3 | 5/5 | 0.668853 | 0.006996 | 0.013840 | 0.566628 | 0.813593 | — (0) | — | — (0) |
| fm_strict_anomaly_transformer__psm__registered | PRONTO_DAY4 | 5/5 | 0.474066 | 0.026223 | 0.049669 | 0.480756 | 0.419808 | 0.717926 (5) | 0.694036 | 0.790997 (5) |
| fm_strict_anomaly_transformer__psm__registered | PSM | 5/5 | 0.421627 | 0.007416 | 0.014575 | 0.513439 | 0.295091 | 0.432373 (5) | 0.291488 | 0.635304 (5) |
| fm_strict_anomaly_transformer__psm__registered | SKAB | 5/5 | 0.391275 | 0.009970 | 0.019442 | 0.427430 | 0.320380 | 0.660282 (4) | 0.589757 | 0.749117 (4) |
| fm_strict_anomaly_transformer__psm__registered | TEP_CLASSIC | 5/5 | 0.859347 | 0.013393 | 0.026373 | 0.468803 | 0.835523 | 0.890125 (5) | 0.972942 | 0.904345 (5) |
| fm_strict_anomaly_transformer__smap__registered | SMAP | 5/5 | 0.119691 | 0.025828 | 0.040822 | 0.547435 | 0.144792 | — (0) | — | — (0) |
| fm_strict_dcdetector__msl__registered | MSL | 5/5 | 0.102026 | 0.010327 | 0.018754 | 0.510673 | 0.108932 | — (0) | — | — (0) |
| fm_strict_dcdetector__psm__registered | PRONTO_DAY2 | 5/5 | 0.945156 | 0.009797 | 0.019390 | 0.506659 | 0.941669 | 0.973202 (5) | 0.997794 | 0.967182 (5) |
| fm_strict_dcdetector__psm__registered | PRONTO_DAY3 | 5/5 | 0.729900 | 0.010978 | 0.021622 | 0.598332 | 0.828335 | — (0) | — | — (0) |
| fm_strict_dcdetector__psm__registered | PRONTO_DAY4 | 5/5 | 0.468616 | 0.010242 | 0.020039 | 0.468322 | 0.397537 | 0.702741 (5) | 0.675540 | 0.762463 (5) |
| fm_strict_dcdetector__psm__registered | PSM | 5/5 | 0.280011 | 0.008909 | 0.017267 | 0.501093 | 0.278458 | 0.439577 (2) | 0.280859 | 0.642914 (2) |
| fm_strict_dcdetector__psm__registered | SKAB | 5/5 | 0.370142 | 0.011662 | 0.022592 | 0.472803 | 0.338942 | 0.684734 (3) | 0.594531 | 0.717326 (3) |
| fm_strict_dcdetector__psm__registered | TEP_CLASSIC | 5/5 | 0.883269 | 0.014310 | 0.028155 | 0.404660 | 0.809885 | 0.876082 (5) | 0.969965 | 0.884707 (5) |
| fm_strict_dcdetector__smap__registered | SMAP | 5/5 | 0.073975 | 0.004783 | 0.008984 | 0.559623 | 0.138280 | — (0) | — | — (0) |
| fm_strict_itransformer__psm__registered | PRONTO_DAY2 | 5/5 | 1.000000 | 0.005727 | 0.011388 | 0.551696 | 0.952608 | 0.984508 (5) | 0.998937 | 0.763014 (5) |
| fm_strict_itransformer__psm__registered | PRONTO_DAY3 | 5/5 | 0.468146 | 0.018704 | 0.035969 | 0.341709 | 0.714053 | — (0) | — | — (0) |
| fm_strict_itransformer__psm__registered | PRONTO_DAY4 | 5/5 | 0.840732 | 0.007020 | 0.013923 | 0.539136 | 0.462087 | 0.745938 (5) | 0.724188 | 0.649683 (5) |
| fm_strict_itransformer__psm__registered | PSM | 5/5 | 0.667631 | 0.025093 | 0.048369 | 0.593232 | 0.385533 | — (0) | — | — (0) |
| fm_strict_itransformer__psm__registered | SKAB | 5/5 | 0.594325 | 0.117292 | 0.195811 | 0.555937 | 0.431904 | 0.747270 (5) | 0.673080 | 0.721763 (5) |
| fm_strict_itransformer__psm__registered | TEP_CLASSIC | 5/5 | 0.965836 | 0.391762 | 0.557398 | 0.723174 | 0.934172 | 0.950915 (5) | 0.989562 | 0.914512 (5) |
| fm_strict_itransformer__smap__registered | SMAP | 5/5 | 0.079510 | 0.031598 | 0.045223 | 0.421727 | 0.107702 | — (0) | — | — (0) |
| fm_strict_timesnet__msl__registered | MSL | 5/5 | 0.099431 | 0.062598 | 0.076810 | 0.566504 | 0.136118 | — (0) | — | — (0) |
| fm_strict_timesnet__psm__registered | PRONTO_DAY2 | 5/5 | 0.998214 | 0.006805 | 0.013506 | 0.555308 | 0.953766 | 0.984372 (5) | 0.998968 | 0.784650 (5) |
| fm_strict_timesnet__psm__registered | PRONTO_DAY3 | 5/5 | 0.428427 | 0.020766 | 0.039603 | 0.315425 | 0.703740 | — (0) | — | — (0) |
| fm_strict_timesnet__psm__registered | PRONTO_DAY4 | 5/5 | 0.900000 | 0.002230 | 0.004447 | 0.558825 | 0.475870 | 0.761214 (5) | 0.735129 | 0.545347 (5) |
| fm_strict_timesnet__psm__registered | PSM | 5/5 | 0.692853 | 0.028071 | 0.053941 | 0.598985 | 0.396761 | 0.597688 (5) | 0.398548 | 0.767881 (5) |
| fm_strict_timesnet__psm__registered | SKAB | 5/5 | 0.592598 | 0.119078 | 0.198290 | 0.554765 | 0.430283 | 0.748035 (5) | 0.674873 | 0.711315 (5) |
| fm_strict_timesnet__psm__registered | TEP_CLASSIC | 5/5 | 0.952964 | 0.451310 | 0.612447 | 0.732020 | 0.936110 | 0.955354 (5) | 0.990342 | 0.921253 (5) |
| fm_strict_timesnet__smap__registered | SMAP | 5/5 | 0.069500 | 0.027026 | 0.038918 | 0.411246 | 0.104513 | — (0) | — | — (0) |
| fm_strict_tranad__msl__registered | MSL | 5/5 | 0.070817 | 0.095326 | 0.081264 | 0.564902 | 0.123400 | — (0) | — | — (0) |
| fm_strict_tranad__psm__registered | PRONTO_DAY2 | 5/5 | 1.000000 | 0.042913 | 0.082245 | 0.675399 | 0.969625 | 0.971351 (5) | 0.998274 | 0.553244 (5) |
| fm_strict_tranad__psm__registered | PRONTO_DAY3 | 5/5 | 0.772787 | 0.138221 | 0.234463 | 0.395507 | 0.735216 | — (0) | — | — (0) |
| fm_strict_tranad__psm__registered | PRONTO_DAY4 | 5/5 | 1.000000 | 0.002065 | 0.004121 | 0.434710 | 0.360376 | 0.617609 (5) | 0.596592 | 0.390576 (5) |
| fm_strict_tranad__psm__registered | PSM | 5/5 | 0.950000 | 0.014027 | 0.027646 | 0.640820 | 0.478071 | — (0) | — | — (0) |
| fm_strict_tranad__psm__registered | SKAB | 5/5 | 0.390523 | 0.733986 | 0.509802 | 0.677208 | 0.637654 | 0.881295 (5) | 0.862251 | 0.765349 (5) |
| fm_strict_tranad__psm__registered | TEP_CLASSIC | 5/5 | 0.995008 | 0.616548 | 0.761331 | 0.887078 | 0.977274 | 0.974274 (5) | 0.994929 | 0.916555 (5) |
| fm_strict_usad__msl__registered | MSL | 5/5 | 0.056184 | 0.077585 | 0.065173 | 0.588159 | 0.130747 | — (0) | — | — (0) |
| fm_strict_usad__psm__registered | PRONTO_DAY2 | 5/5 | 1.000000 | 0.090614 | 0.166143 | 0.681805 | 0.970660 | 0.972063 (5) | 0.998345 | 0.557191 (5) |
| fm_strict_usad__psm__registered | PRONTO_DAY3 | 5/5 | 0.738000 | 0.152120 | 0.252246 | 0.354380 | 0.724316 | — (0) | — | — (0) |
| fm_strict_usad__psm__registered | PRONTO_DAY4 | 5/5 | 0.000000 | 0.000000 | 0.000000 | 0.456466 | 0.371131 | 0.640990 (5) | 0.627183 | — (0) |
| fm_strict_usad__psm__registered | PSM | 5/5 | 0.752067 | 0.014675 | 0.028775 | 0.720846 | 0.543615 | — (0) | — | — (0) |
| fm_strict_usad__psm__registered | SKAB | 5/5 | 0.393390 | 0.749670 | 0.516005 | 0.674037 | 0.633656 | 0.890857 (5) | 0.879593 | 0.757419 (5) |
| fm_strict_usad__psm__registered | TEP_CLASSIC | 5/5 | 0.949618 | 0.740476 | 0.832105 | 0.859262 | 0.965419 | 0.980942 (5) | 0.996367 | 0.935752 (5) |
| fm_strict_usad__psm__signed_paper_loss | PRONTO_DAY2 | 5/5 | 0.400000 | 0.018807 | 0.034891 | 0.709400 | 0.976639 | 0.979420 (5) | 0.998868 | 0.423748 (2) |
| fm_strict_usad__psm__signed_paper_loss | PRONTO_DAY3 | 5/5 | 0.367726 | 0.041107 | 0.071241 | 0.338023 | 0.706692 | — (0) | — | — (0) |
| fm_strict_usad__psm__signed_paper_loss | PRONTO_DAY4 | 5/5 | 0.000000 | 0.000000 | 0.000000 | 0.409522 | 0.353194 | 0.596755 (5) | 0.587822 | — (0) |
| fm_strict_usad__psm__signed_paper_loss | PSM | 5/5 | 0.104332 | 0.040975 | 0.058838 | 0.519146 | 0.304404 | — (0) | — | — (0) |
| fm_strict_usad__psm__signed_paper_loss | SKAB | 5/5 | 0.389492 | 0.737526 | 0.509769 | 0.679032 | 0.641558 | 0.898785 (5) | 0.891228 | 0.749148 (5) |
| fm_strict_usad__psm__signed_paper_loss | TEP_CLASSIC | 5/5 | 0.951254 | 0.738333 | 0.831361 | 0.857314 | 0.965089 | 0.980764 (5) | 0.996333 | 0.934338 (5) |
| fm_strict_usad__smap__registered | SMAP | 5/5 | 0.296483 | 0.216056 | 0.249936 | 0.463074 | 0.135130 | — (0) | — | — (0) |
| fm_strict_usad__smap__signed_paper_loss | SMAP | 5/5 | 0.066552 | 0.059209 | 0.062489 | 0.550997 | 0.132300 | — (0) | — | — (0) |
| fm_objective_sb_stable_v2 | PRONTO_DAY3 | 4/5 | 0.706547 | 0.098999 | 0.173649 | 0.396407 | 0.733941 | 0.907950 (4) | 0.972265 | 0.687009 (4) |
| fm_objective_sf2m_flow_only_stable_v2 | PRONTO_DAY3 | 4/5 | 0.706547 | 0.098999 | 0.173649 | 0.396407 | 0.733941 | 0.907950 (4) | 0.972265 | 0.687009 (4) |
| fm_objective_sf2m_stable_v2 | PRONTO_DAY3 | 4/5 | 0.713671 | 0.102282 | 0.178907 | 0.393857 | 0.733293 | 0.907328 (4) | 0.972101 | 0.681135 (4) |
| fm_rectified_60epoch_budget_control | CICIDS | 4/5 | 0.533353 | 0.025326 | 0.048355 | 0.615679 | 0.392124 | — (0) | — | — (0) |
| fm_objective_sb_stable_v2 | PRONTO_DAY4 | 3/5 | 1.000000 | 0.002065 | 0.004121 | 0.451857 | 0.369249 | 0.636532 (3) | 0.616044 | 0.390422 (3) |
| fm_objective_sf2m_flow_only_stable_v2 | PRONTO_DAY4 | 3/5 | 1.000000 | 0.002065 | 0.004121 | 0.451857 | 0.369249 | 0.636532 (3) | 0.616044 | 0.390422 (3) |
| fm_objective_sf2m_stable_v2 | PRONTO_DAY4 | 3/5 | 1.000000 | 0.001996 | 0.003984 | 0.452381 | 0.369946 | 0.637129 (3) | 0.618050 | 0.390340 (3) |
| fm_objective_sb_stable_v2 | SMD | 2/5 | 0.493244 | 0.342754 | 0.404325 | 0.762213 | 0.329719 | — (0) | — | — (0) |
| fm_moment_pretrained_small__TAB100 | PSM | 1/1 | 0.713812 | 0.026496 | 0.051095 | 0.585120 | 0.383768 | — (0) | — | — (0) |
| fm_moment_pretrained_small__TAB100 | SMAP | 1/1 | 0.133369 | 0.029023 | 0.047672 | 0.434559 | 0.111922 | — (0) | — | — (0) |
| fm_moment_pretrained_small__release512 | CICIDS | 1/1 | 0.363062 | 0.014655 | 0.028173 | 0.420767 | 0.285714 | — (0) | — | — (0) |
| fm_moment_pretrained_small__release512 | MSL | 1/1 | 0.135422 | 0.064080 | 0.086996 | 0.559740 | 0.132951 | 0.720217 (1) | 0.309355 | 0.696811 (1) |
| fm_moment_pretrained_small__release512 | PRONTO_DAY2 | 1/1 | 1.000000 | 0.023936 | 0.046752 | 0.544039 | 0.953758 | — (0) | — | — (0) |
| fm_moment_pretrained_small__release512 | PRONTO_DAY3 | 1/1 | 0.550725 | 0.053710 | 0.097875 | 0.375337 | 0.721401 | — (0) | — | — (0) |
| fm_moment_pretrained_small__release512 | PRONTO_DAY4 | 1/1 | 1.000000 | 0.001652 | 0.003298 | 0.586470 | 0.473271 | — (0) | — | — (0) |
| fm_moment_pretrained_small__release512 | PSM | 1/1 | 0.681009 | 0.018826 | 0.036639 | 0.555556 | 0.337271 | 0.547813 (1) | 0.333540 | 0.658476 (1) |
| fm_moment_pretrained_small__release512 | SKAB | 1/1 | 0.545031 | 0.154593 | 0.240866 | 0.594941 | 0.424080 | — (0) | — | — (0) |
| fm_moment_pretrained_small__release512 | SMAP | 1/1 | 0.232903 | 0.068287 | 0.105609 | 0.454654 | 0.126343 | 0.674969 (1) | 0.251157 | 0.545930 (1) |
| fm_moment_pretrained_small__release512 | SMD | 1/1 | 0.362595 | 0.115179 | 0.174824 | 0.813581 | 0.260849 | 0.810018 (1) | 0.326123 | 0.543465 (1) |
| fm_moment_pretrained_small__release512 | SWAN | 1/1 | 0.531190 | 0.355871 | 0.426205 | 0.684565 | 0.495399 | 0.597565 (1) | 0.445366 | 0.593244 (1) |
| fm_moment_pretrained_small__release512 | SWAT | 1/1 | 0.282845 | 0.048992 | 0.083518 | 0.262044 | 0.113999 | 0.369082 (1) | 0.162870 | 0.704033 (1) |
| fm_moment_pretrained_small__release512 | TEP_CLASSIC | 1/1 | 0.857275 | 0.439762 | 0.581320 | 0.562498 | 0.852026 | — (0) | — | — (0) |

各指标可能由不同数量的已完成种子产生，范围指标种子数必须一起读取；不混合缺失值为零。独立 CFM 与单阶段 Rectified 在这里 sigma=0 时等价。迭代 reflow 独立列行，其完整队列尚未全部完成。

工业数据的边界：TEP为经典单工况整运行；SKAB校准实验包含原生异常，1%只表示验证分数尾部分位数；PRONTO训练/校准的正常段选择使用标签，三个整日轮换相关。PSM基线超参数直接迁移，未做目标数据调优。工业结果不是原论文等效复现。

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
| cfmi | Air-36 | None | historical_air36 / author_protocol | mae | 9.534687 | 0.069923 | 5/5 | 9.404000 | numerically_compatible |
| cfmi | Air-36 | None | historical_air36 / author_protocol | rmse | 18.120535 | 0.206023 | 5/5 | 17.701000 | numerically_compatible |
| cfmi | Air-36 | None | historical_air36 / author_protocol | crps | 6.800040 | 0.048035 | 5/5 | 6.726000 | numerically_compatible |
| cfmi | Air-36 | None | historical_air36 / author_protocol | normalized_crps | 0.099087 | 0.000638 | 5/5 | 0.098000 | numerically_compatible |
| saits | PhysioNet2012 | None | paper_mask / author_protocol | mae | 0.189721 | 0.000301 | 3/5 | 0.186000 | incomplete_repeats |
| saits | PhysioNet2012 | None | paper_mask / author_protocol | rmse | 0.422238 | 0.000759 | 3/5 | 0.431000 | incomplete_repeats |
| saits | PhysioNet2012 | None | paper_mask / author_protocol | mre | 0.275389 | 0.000437 | 3/5 | 0.266000 | incomplete_repeats |
| brits | PhysioNet2012 | None | paper_mask / author_protocol | mae | 0.237949 | 0.000127 | 3/5 | 0.256000 | incomplete_repeats |
| brits | PhysioNet2012 | None | paper_mask / author_protocol | rmse | 0.576683 | 0.000349 | 3/5 | 0.767000 | incomplete_repeats |
| brits | PhysioNet2012 | None | paper_mask / author_protocol | mre | 0.345394 | 0.000184 | 3/5 | 0.365000 | incomplete_repeats |

数值兼容筛查不等于统计等价证明；not_comparable、incomplete_repeats 等原因详见 imputation_summary.csv。插补误差不能代替异常检测成绩。

## 文件

[完整 Excel](complete_experiment_metrics.xlsx)、[所有组合与状态](strict_summary.csv)、[逐种子严格指标](strict_per_seed.csv)、[历史协议结果](historical_protocol_results.csv)、[插补汇总](imputation_summary.csv)、[论文报告值](paper_claims.csv)、[完整机器可读结果](complete_results.json)。

原文数据范围、未运行任务及每篇复现缺口均保留在 CSV/JSON 与 Excel 中。

## 作者完整流程

独立作者流程登记 1429 项，状态 {'failed_or_partial_preserved': 7, 'running': 2, 'pending': 1418, 'completed': 2}。逐项配方、数据集、种子和阶段状态见 author_pipeline_jobs.csv；尚无完整流程指标时保留空值，不计入严格成绩。

已完成作者流程的全部数值字段见 [作者流程技术指标](author_pipeline_numeric_metrics.csv)，作者协议与附加验证阈值对照均保留原字段路径，分别解释。

## 新登记的原生流匹配实验

GiFlow 另登记 140 个作者镜像/已审修正实验，状态 {'pending': 140}。逐项方法、消融、数据集、种子和状态见 [原生实验清单](additional_native_paper_jobs.csv)。正式完整运行才列指标；集成检查不计入。
