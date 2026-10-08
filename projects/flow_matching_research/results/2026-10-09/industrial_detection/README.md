# 工业异常检测完整数据执行

快照：2026-10-08T19:16:46.175967+00:00。登记 1400 项，完整结果 38 项。

56个配置 × 5种数据划分 × 5个种子。CPU550项、GPU850项。保留完整容量和训练轮数；完整结果必须保存模型和每个原始时间点的分数。

| 数据划分 | 训练点数 | 测试运行/天数 | 完整测试点数 |
|---|---:|---:|---:|
| tep_classic | 500 | 21 | 20160 |
| skab | 9405 | 33 | 36656 |
| pronto_day4 | 722 | 1 | 11700 |
| pronto_day2 | 4320 | 1 | 12420 |
| pronto_day3 | 6857 | 1 | 21300 |

训练/验证/测试按整个运行或日期隔离。TEP训练d00.dat、验证d00_te.dat、测试全部21个故障运行；SKAB验证other/1.csv，其余33个实验测试；PRONTO轮换整日测试，训练/验证只保留各自日期的全部Normal连续段。所有分段单独开窗，不跨故障间隙；共享训练只拟合一次。

SKAB验证包含异常，1%是验证总体尾部分位数，不能称正常总体误报率保证。PRONTO训练/校准的正常段选择使用原标签，明确属于标签辅助协议；三个日期轮换有关联。PSM基线超参数直接迁移，未经目标数据调优。经典TEP不代表多工况HDF5，工业迁移不代表原论文等效复现。

| 算法配置 | 数据划分 | 种子 | Precision | Recall | 点级F1 | AUROC | AP |
|---|---|---:|---:|---:|---:|---:|---:|
| fm_objective_independent | tep_classic | 1103 | 0.998655 | 0.574643 | 0.729512 | 0.889729 | 0.977457 |
| fm_objective_independent | tep_classic | 1104 | 0.998854 | 0.570893 | 0.726536 | 0.890004 | 0.977523 |
| fm_objective_independent | tep_classic | 1105 | 0.998459 | 0.578571 | 0.732617 | 0.891130 | 0.977705 |
| fm_objective_independent | tep_classic | 1106 | 0.998558 | 0.577024 | 0.731402 | 0.887338 | 0.976946 |
| fm_objective_independent | tep_classic | 1107 | 0.998460 | 0.578988 | 0.732952 | 0.891697 | 0.977817 |
| fm_objective_ot | tep_classic | 1103 | 0.998060 | 0.581845 | 0.735128 | 0.891045 | 0.977700 |
| fm_objective_ot | tep_classic | 1104 | 0.998648 | 0.571429 | 0.726915 | 0.891359 | 0.977782 |
| fm_objective_ot | tep_classic | 1105 | 0.998458 | 0.578095 | 0.732235 | 0.891431 | 0.977750 |
| fm_objective_ot | tep_classic | 1106 | 0.998250 | 0.577321 | 0.731558 | 0.888525 | 0.977164 |
| fm_objective_ot | tep_classic | 1107 | 0.998257 | 0.579524 | 0.733326 | 0.892564 | 0.977996 |
| fm_objective_rectified | tep_classic | 1103 | 0.998655 | 0.574643 | 0.729512 | 0.889729 | 0.977457 |
| fm_objective_rectified | tep_classic | 1104 | 0.998854 | 0.570893 | 0.726536 | 0.890004 | 0.977523 |
| fm_objective_rectified | tep_classic | 1105 | 0.998459 | 0.578571 | 0.732617 | 0.891130 | 0.977705 |
| fm_objective_rectified | tep_classic | 1106 | 0.998558 | 0.577024 | 0.731402 | 0.887338 | 0.976946 |
| fm_objective_rectified | tep_classic | 1107 | 0.998460 | 0.578988 | 0.732952 | 0.891697 | 0.977817 |
| fm_objective_target | tep_classic | 1103 | 0.998259 | 0.580238 | 0.733898 | 0.892416 | 0.977952 |
| fm_objective_target | tep_classic | 1104 | 0.998064 | 0.583036 | 0.736079 | 0.889457 | 0.977417 |
| fm_objective_target | tep_classic | 1105 | 0.998758 | 0.574167 | 0.729156 | 0.887548 | 0.976943 |
| fm_objective_target | tep_classic | 1106 | 0.998359 | 0.579524 | 0.733353 | 0.887270 | 0.976889 |
| fm_objective_target | tep_classic | 1107 | 0.998443 | 0.572738 | 0.727919 | 0.886547 | 0.976801 |
| fm_objective_vp | tep_classic | 1103 | 0.998551 | 0.574464 | 0.729341 | 0.891093 | 0.977688 |
| fm_objective_vp | tep_classic | 1104 | 0.998646 | 0.570833 | 0.726433 | 0.890680 | 0.977602 |
| fm_objective_vp | tep_classic | 1105 | 0.998455 | 0.577143 | 0.731470 | 0.892525 | 0.977955 |
| fm_objective_vp | tep_classic | 1106 | 0.998347 | 0.575238 | 0.729909 | 0.889845 | 0.977401 |
| fm_objective_vp | tep_classic | 1107 | 0.998559 | 0.577500 | 0.731785 | 0.894284 | 0.978337 |
| fm_objective_sb_stable_v2 | tep_classic | 1103 | 0.998369 | 0.583095 | 0.736209 | 0.888637 | 0.977188 |
| fm_objective_sb_stable_v2 | tep_classic | 1104 | 0.997654 | 0.582262 | 0.735350 | 0.890017 | 0.977489 |
| fm_objective_sb_stable_v2 | tep_classic | 1105 | 0.998643 | 0.569464 | 0.725322 | 0.887767 | 0.976966 |
| fm_objective_sb_stable_v2 | tep_classic | 1106 | 0.998545 | 0.571905 | 0.727273 | 0.882502 | 0.975969 |
| fm_objective_sb_stable_v2 | tep_classic | 1107 | 0.998461 | 0.579345 | 0.733238 | 0.889013 | 0.977280 |
| fm_objective_sf2m_stable_v2 | tep_classic | 1103 | 0.998166 | 0.583214 | 0.736249 | 0.888779 | 0.977205 |
| fm_objective_sf2m_stable_v2 | tep_classic | 1104 | 0.997647 | 0.580476 | 0.733923 | 0.890391 | 0.977550 |
| fm_objective_sf2m_stable_v2 | tep_classic | 1105 | 0.998440 | 0.571429 | 0.726860 | 0.889214 | 0.977256 |
| fm_objective_sf2m_stable_v2 | tep_classic | 1106 | 0.998543 | 0.570952 | 0.726502 | 0.883175 | 0.976112 |
| fm_objective_sf2m_stable_v2 | tep_classic | 1107 | 0.998448 | 0.574583 | 0.729409 | 0.889403 | 0.977336 |
| fm_objective_sf2m_flow_only_stable_v2 | tep_classic | 1103 | 0.998369 | 0.583095 | 0.736209 | 0.888637 | 0.977188 |
| fm_objective_sf2m_flow_only_stable_v2 | tep_classic | 1104 | 0.997654 | 0.582262 | 0.735350 | 0.890017 | 0.977489 |
| fm_objective_sf2m_flow_only_stable_v2 | tep_classic | 1105 | 0.998643 | 0.569464 | 0.725322 | 0.887767 | 0.976966 |

这里只列已通过完整源数据、模型/分数哈希、覆盖和指标独立重算检查的完整结果；不是全部队列已经结束。预检不计性能。TAB VUS/Affiliation由独立固定版本队列跟随补算。

validation.json的seed_aggregates保存跨种子均值、STD、SE与t区间；这些区间描述初始化变异，不证明跨算法等效或不同运行的独立性。

[完整审计和逐次结果](validation.json)、[原始数据与划分审计](data_manifest.json)。原始论文、消融、多工况以及其他任务的已知缺口继续保留。
