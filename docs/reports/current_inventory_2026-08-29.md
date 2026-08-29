# IIA Benchmark 当前清单与完备性说明（2026-08-29）

仓库：<https://github.com/gaoxingkele/IIA_benchmark>；生成基线 revision：`6d030cb`。

## 1. 总体结论

| 范围 | 当前数量 | 完备性判断 |
|---|---:|---|
| 登记参考论文 | 28 | 本地全文 5，缺 23 |
| 书籍算法交付项 | 20 | 可调用 20；verified 0，partial 20 |
| SOTA 算法交付项 | 10 | 可调用 10；verified 0，partial 10 |
| 可调用方法族 / 模型配置 | 34 / 39 | 机制与单元测试可运行，不等于论文分数复现 |
| 逻辑数据集族 | 11 | 有效主载荷 11/11 |
| 下游任务 | 6 | 6/6 有真实或已取得数据入口；T4 正式专用数据实验仍待 adapter |
| 正式排行榜切分 | 0 | 尚无 leaderboard-eligible split |
| 真实数据验证报告 | 41 | 覆盖 13 个登记算法；严格分数闭环仍为 0 |

这里的‘可调用’表示本地实现有明确入口并通过机制/不变量测试；只有在论文原始数据、预处理、grouped split、指标、随机种子和参考分数均闭合后，才能升级为 `verified`。

## 2. 参考论文清单

当前登记 28 篇：全文已归档 5 篇，访问受限 22 篇，自动下载失败 1 篇。

| ID | 年份 | 题目 | 期刊/会议 | DOI | 全文状态 | 对应角色 |
|---|---:|---|---|---|---|---|
| `xu2012_far_mar_aad` | 2012 | Performance Assessment and Design for Univariate Alarm Systems Based on FAR, MAR, and AAD | IEEE Transactions on Automation Science and Engineering | `10.1109/TASE.2011.2176490` | 全文已归档 | book-chapter-2 mathematical baseline |
| `wang2022_delay_pmf` | 2022 | Design of Delay Timers Based on Estimated Probability Mass Functions of Alarm Durations | Journal of Process Control | `10.1016/j.jprocont.2022.01.002` | 全文未归档（访问受限） | book-chapter-2 non-IID delay design |
| `wang2023_deadband` | 2023 | Alarm Deadband Design Based on Maximum Amplitude Deviations and Bayesian Estimation | IEEE Transactions on Control Systems Technology | `10.1109/TCST.2023.3240020` | 全文未归档（访问受限） | book-chapter-2 non-IID deadband design |
| `yu2017_alarm_probability_plot` | 2017 | Design of Alarm Trippoints for Univariate Analog Process Variables Based on Alarm Probability Plots | IEEE Transactions on Industrial Electronics | `10.1109/TIE.2017.2682783` | 全文未归档（访问受限） | book-chapter-2 alarm probability plots |
| `yu2020_convex_noz` | 2020 | Alarm Monitoring for Multivariate Processes Based on a Convex-Hull Normal Operating Zone | IEEE Transactions on Control Systems Technology | `10.1109/TCST.2019.2943469` | 全文未归档（访问受限） | book-chapter-3 convex NOZ |
| `wang2024_search_cones` | 2024 | Multivariate Alarm Monitoring for Non-Convex Normal Operating Zones Based on Search Cones | IEEE Transactions on Automation Science and Engineering | `10.1109/TASE.2022.3222413` | 全文未归档（访问受限） | book-chapter-3 non-convex NOZ |
| `chen2017_variation_directions` | 2017 | Design of Multivariate Alarm Systems Based on Online Calculation of Variational Directions | Chemical Engineering Research and Design | `10.1016/j.cherd.2017.04.011` | 全文未归档（访问受限） | book-chapter-3 variation-direction alarms |
| `xiong2018_bayesian_pumps` | 2018 | Multivariate Alarm Systems for Time-Varying Processes Using Bayesian Filters With Applications to Electrical Pumps | IEEE Transactions on Industrial Informatics | `10.1109/TII.2017.2749332` | 全文未归档（访问受限） | book-chapter-3 time-varying pump alarms |
| `wang2024_condenser_noz` | 2024 | Multivariate Process Monitoring for Safe Operation of Condensers in Thermal Power Plants Based on Normal Operating Zones | IEEE Transactions on Control Systems Technology | `10.1109/TCST.2024.3370036` | 全文未归档（访问受限） | book-chapter-3 physics-guided condenser monitoring |
| `zhang2022_igte` | 2022 | Detection of Cause-Effect Relations Based on Information Granulation and Transfer Entropy | Entropy | `10.3390/e24020212` | 全文已归档 | book-chapter-4 continuous-variable causal inference |
| `hu2017_alarm_transfer_entropy` | 2017 | Cause-Effect Analysis of Industrial Alarm Variables Using Transfer Entropies | Control Engineering Practice | `10.1016/j.conengprac.2017.04.012` | 全文未归档（访问受限） | book-chapter-4 binary alarm NTE/NDTE |
| `zhang2023_igdte` | 2023 | A New Transfer Entropy Approach Based on Information Granulation and Clustering for Root Cause Analysis | Control Engineering Practice | `10.1016/j.conengprac.2023.105669` | 全文未归档（访问受限） | book-chapter-4 IGTE/IGDTE extension |
| `wang2018_recursive_bn_rca` | 2018 | Root-Cause Analysis of Occurring Alarms in Thermal Power Plants Based on Bayesian Networks | International Journal of Electrical Power & Energy Systems | `10.1016/j.ijepes.2018.05.029` | 全文未归档（访问受限） | book-chapter-4 online alarm root-cause analysis |
| `hu2022_qualitative_trend_rca` | 2022 | Analysis of Time-Varying Cause-Effect Relations Based on Qualitative Trends and Change Amplitudes | Computers & Chemical Engineering | `10.1016/j.compchemeng.2022.107813` | 全文未归档（访问受限） | book-chapter-4 continuous-variable root-cause contributions |
| `wang2018_flood_detection` | 2018 | Criteria and Algorithms for Online and Offline Detections of Industrial Alarm Floods | IEEE Transactions on Control Systems Technology | `10.1109/TCST.2017.2723578` | 全文未归档（访问受限） | book-chapter-5 flood segmentation |
| `hu2016_local_alignment` | 2016 | A Local Alignment Approach to Similarity Analysis of Industrial Alarm Flood Sequences | Control Engineering Practice | `10.1016/j.conengprac.2016.05.021` | 全文未归档（访问受限） | book-chapter-5 alarm-flood matching |
| `hu2018_closed_alarm_patterns` | 2018 | Detection of Frequent Alarm Patterns in Industrial Alarm Floods Using Itemset Mining Methods | IEEE Transactions on Industrial Electronics | `10.1109/TIE.2018.2795573` | 全文未归档（访问受限） | book-chapter-5 closed and representative patterns |
| `xu2021_max_entropy` | 2021 | A Maximum-Entropy-Based Method for Alarm Flood Prediction | Journal of Process Control | `10.1016/j.jprocont.2021.10.002` | 全文未归档（访问受限） | book-chapter-5 next-alarm prediction |
| `faulwasser2024_casim` | 2024 | Convolutional Kernel-Based Classification of Industrial Alarm Floods | Data-Centric Engineering | `10.1017/dce.2024.22` | 全文已归档 | SOTA early/open-set alarm flood classification |
| `faulwasser2024_cone_afc` | 2024 | Addressing Uncertainty in Online Alarm Flood Classification Using Conformal Prediction | IEEE Access | `10.1109/ACCESS.2024.3492348` | 全文已归档 | SOTA uncertainty-aware online alarm flood classification |
| `faulwasser2025_uncertainty_reduction` | 2025 | Predicting Uncertainty Reduction in Online Alarm Flood Classification | IFAC-PapersOnLine | `10.1016/j.ifacol.2025.11.935` | 全文已归档 | SOTA early decision timing |
| `faulwasser2025_etfa_robustness` | 2025 | Robustness Analysis of Industrial Alarm Flood Classification | IEEE ETFA | `10.1109/ETFA65518.2025.11205709` | 全文未归档（访问受限） | SOTA alarm-corruption robustness |
| `faulwasser2026_robustbench` | 2026 | AFC-RobustBench: A Robustness Benchmark for Alarm Flood Classification | SSRN preprint | `10.2139/ssrn.6999280` | 全文未归档（下载失败） | SOTA robustness benchmark |
| `manca2025_cross_conformal_afc` | 2025 | Data-Efficient Handling of Temporary Uncertainties in Online Alarm Flood Classification | IEEE ICPS | `10.1109/ICPS65515.2025.11087828` | 全文未归档（访问受限） | SOTA data-efficient uncertainty-aware online alarm flood classification |
| `rao2025_ctfh` | 2025 | Online Alarm Flood Classification via Deterministic Fingerprinting with Combinatorial Hashing: A Robust and Scalable Framework | Chemical Engineering Research and Design | `10.1016/j.cherd.2025.11.026` | 全文未归档（访问受限） | SOTA deterministic and scalable online alarm flood classification |
| `rahaman2025_modified_tfidf` | 2025 | Real-Time Classification and Early Warning of Industrial Alarm Floods Using Modified TF-IDF Methods | Control Engineering Practice | `10.1016/j.conengprac.2025.106485` | 全文未归档（访问受限） | SOTA n-gram clustering, fault isolation and early alarm-flood warning |
| `rahimi2026_structured_hdam` | 2026 | Online Alarm Flood Classification via Interpretable Template Extraction and Structured Convolutional Matching | Computers & Chemical Engineering | `10.1016/j.compchemeng.2026.109570` | 全文未归档（访问受限） | SOTA interpretable visual-template online alarm flood classification |
| `najafi2026_time_histogram_hybrid` | 2026 | Early Classification of Industrial Alarm Floods Using a Hybrid Neural Network and Optimal Time-Encoded Histograms | Engineering Applications of Artificial Intelligence | `10.1016/j.engappai.2025.113705` | 全文未归档（访问受限） | SOTA early alarm-flood classification without chattering preprocessing |

## 3. 算法集

核心闭环账本为 20 项书籍算法与 10 项 SOTA。`docs/algorithm_matrix.md` 的 34 个可调用方法族还包含支撑基线和一个交付项拆出的多个机制，因此数量不与 30 项闭环账本一一相等。

### 3.1 书籍算法（20 项）

| ID | 章节 | 算法/方法 | 页码 | 本地入口 | 状态 |
|---|---|---|---|---|---|
| `book_2_1_iid_delay_timer` | 2.1 | IID threshold and delay-timer design by FAR/MAR/AAD | 印刷页 49-67 / PDF 60-78 | `AlarmOnOffDelay`、`iid_delay_timer_performance`、`design_iid_delay_timer` | `verified`；Xu 2012 Examples 1-2 与 Table VII 已复现，工业原始载荷仍缺 |
| `book_2_2_non_iid_delay_timer` | 2.2 | Non-IID delay-timer design from alarm durations and intervals | 印刷页 68-84 / PDF 79-95 | `iia_benchmark.models.univariate_book.design_non_iid_delay_timer` | `partial` |
| `book_2_3_non_iid_deadband` | 2.3 | Deadband design by maximum amplitude deviations and Bayesian estimation | 印刷页 85-107 / PDF 96-118 | `iia_benchmark.models.univariate_book.deadband_index`<br>`iia_benchmark.models.univariate_book.design_deadband_width` | `partial` |
| `book_2_4_alarm_probability_plot` | 2.4 | Alarm Probability Plot threshold design | 印刷页 108-125 / PDF 119-136 | `iia_benchmark.models.univariate_book.build_alarm_probability_plot`<br>`iia_benchmark.models.univariate_book.select_alarm_probability_threshold` | `partial` |
| `book_3_1_convex_noz` | 3.1.2 | Convex-hull normal operating zone alarm | 印刷页 131-137 / PDF 142-148 | `ConvexHullNOZAlarm`<br>`convex_hull_fitness_index`<br>`closest_normal_point` | `partial`；书中 Figure 3.2 已复现，多数据集迁移为负结果 |
| `book_3_1_nonconvex_noz` | 3.1.3 | Non-convex normal operating zone by search cones | 印刷页 138-153 / PDF 149-164 | `iia_benchmark.models.multivariate_book.SearchConeNOZAlarm` | `partial` |
| `book_3_2_variation_direction` | 3.2 | Variation-direction multivariate alarm system | 印刷页 154-172 / PDF 165-183 | `iia_benchmark.models.multivariate_book.AdaptiveTimeGradient`<br>`iia_benchmark.models.multivariate_book.VariationDirectionAlarm` | `partial` |
| `book_3_3_electrical_pump` | 3.3 | Bayesian-filter multivariate alarms for electrical pumps | 印刷页 173-191 / PDF 184-202 | `iia_benchmark.models.multivariate_book.BayesianWindowRegressionAlarm` | `partial` |
| `book_3_4_condenser` | 3.4 | Physics-guided condenser normal operating zones | 印刷页 192-217 / PDF 203-228 | `CondenserPhysicalModel`<br>`CondenserNOZAlarm`<br>`condenser_alarm_rate_bounds` | `partial`；方程合成验证通过，但 99% 假警率上界门禁失败，工业原始数据缺失 |
| `book_4_1_nte` | 4.1 | Normalized Transfer Entropy for binary alarm variables | 印刷页 221-238 / PDF 232-249 | `iia_benchmark.models.root_cause_book.NormalizedTransferEntropyGraph` | `partial` |
| `book_4_1_ndte` | 4.1 | Normalized Direct Transfer Entropy | 印刷页 221-238 / PDF 232-249 | `iia_benchmark.models.root_cause_book.normalized_direct_transfer_entropy` | `partial` |
| `book_4_2_igte` | 4.2 | Information-granulation Transfer Entropy | 印刷页 239-261 / PDF 250-272 | `iia_benchmark.models.root_cause_book.information_granulation_transfer_entropy` | `partial` |
| `book_4_2_igdte` | 4.2 | Information-granulation Direct Transfer Entropy | 印刷页 239-261 / PDF 250-272 | `iia_benchmark.models.root_cause_book.information_granulation_direct_transfer_entropy` | `partial` |
| `book_4_3_recursive_bn` | 4.3 | Recursive Bayesian-network alarm root-cause analysis | 印刷页 262-276 / PDF 273-287 | `iia_benchmark.models.root_cause_book.RecursiveBayesianAlarmRCA` | `partial` |
| `book_4_4_plr_rca` | 4.4 | Piecewise-linear trend and non-negative-regression root-cause analysis | 印刷页 277-298 / PDF 288-309 | `iia_benchmark.models.root_cause_book.PLRContributionRCA` | `partial` |
| `book_5_1_flood_detection` | 5.1 | Online/offline alarm flood detection using newly appearing tags | 印刷页 303-321 / PDF 314-332 | `iia_benchmark.models.flood_book.criterion_c_alarm_flood_detection` | `partial` |
| `book_5_2_alarm_alignment` | 5.2 | Priority/time-aware BLAST-like alarm flood alignment | 印刷页 322-343 / PDF 333-354 | `iia_benchmark.models.flood_book.accelerated_alarm_alignment` | `partial` |
| `book_5_3_closed_patterns` | 5.3 | CHARM closed alarm patterns and representative clustering | 印刷页 344-355 / PDF 355-366 | `iia_benchmark.models.flood_book.charm_closed_alarm_patterns`<br>`iia_benchmark.models.flood_book.representative_alarm_patterns` | `partial` |
| `book_5_4_max_entropy_prediction` | 5.4 | Maximum-entropy next-alarm prediction | 印刷页 356-376 / PDF 367-387 | `iia_benchmark.models.flood_book.MaximumEntropyNextAlarmPredictor` | `partial` |
| `book_6_visual_analytics` | 6 | Alarm visual analytics verification suite | 印刷页 381-417 / PDF 392-428 | `iia_benchmark.visualization.build_alarm_visual_analytics`<br>`iia_benchmark.visualization.export_alarm_visual_report` | `partial` |

### 3.2 SOTA 算法（10 项）

| ID | 方法 | DOI | 本地入口 | 状态 | 完整闭环仍需 |
|---|---|---|---|---|---|
| `casim_2024` | CASIM MultiRocket-ridge-LoOP open-set AFC | `10.1017/dce.2024.22` | `iia_benchmark.models.CASIMClassifier` | `partial` | Code Ocean capsule plus an adapter, frozen grouped split, and reference-score run on the locally acquired TEP alarm payload |
| `cone_afc_2024` | ConE-AFC class- and stepwise conformal prediction | `10.1109/ACCESS.2024.3492348` | `iia_benchmark.models.ConEAlarmFloodClassifier` | `partial` | Code Ocean capsule, synthetic dataset, and Tables I-II |
| `uncertainty_reduction_2025` | Random-forest jackknife+ bifurcation forecasting | `10.1016/j.ifacol.2025.11.935` | `iia_benchmark.models.UncertaintyReductionForecaster` | `partial` | official Code Ocean capsule, adapter for the acquired TEP alarm archive, frozen splits, and Tables 1-4 |
| `etfa_robustness_2025` | Alarm-data-quality robustness analysis | `10.1109/ETFA65518.2025.11205709` | `iia_benchmark.evaluation.run_afc_robustness_benchmark` | `partial` | ETFA full text and original experiment data |
| `afc_robustbench_2026` | AFC-RobustBench prefix/severity/mixed-corruption protocol | `10.2139/ssrn.6999280` | `iia_benchmark.evaluation.run_afc_robustness_benchmark` | `partial` | SSRN PDF download endpoint and two process-industry datasets |
| `cross_conformal_afc_2025` | Data-efficient cross-conformal online AFC | `10.1109/ICPS65515.2025.11087828` | `iia_benchmark.models.CrossConformalAlarmFloodClassifier` | `partial` | full paper/capsule, exact empty-set postprocessing, synthetic dataset, and reference scores |
| `modified_tfidf_afc_2025` | Modified TF-IDF n-gram clustering and LSTM early warning | `10.1016/j.conengprac.2025.106485` | `iia_benchmark.models.TFIDFLSTMAlarmFloodClassifier` | `partial` | full modified-TF-IDF/KPCA equations, VAM simulator alarm data, hyperparameters, and reference scores |
| `ctfh_fingerprinting_2025` | AEM/CTFH deterministic fingerprint classification | `10.1016/j.cherd.2025.11.026` | `iia_benchmark.models.CTFHAlarmFloodClassifier` | `partial` | full CTFH equations/parameters, publisher tables, and a reference-score run on the acquired TEP five-class alarm data |
| `hybrid_histogram_afc_2026` | Hybrid neural network with optimal time-encoded histograms | `10.1016/j.engappai.2025.113705` | `iia_benchmark.models.OptimalTimeEncodedHistogramClassifier` | `partial` | full histogram/Transformer equations, exact hyperparameters, adapter for the acquired TEP alarm payload, and selected-paper reference scores |
| `structured_hdam_2026` | Interpretable HDAM template extraction and 2-D convolution matching | `10.1016/j.compchemeng.2026.109570` | `iia_benchmark.models.HDAMTemplateMatcher` | `partial` | full template/alignment equations, paper hyperparameters/tables, and reference-score validation on the acquired TEP alarm payload |

## 4. 数据集清单

机器登记包含 21 条来源记录，归并为 11 个逻辑数据集族；当前 11/11 均至少有一个有效主载荷。原始数据文件受 `.gitignore` 保护，仓库只提交来源、路径、哈希、profile 和审计状态。

| 数据集族 | 内容 | 有效主载荷 | 类型 | 可支持任务 | 当前边界 |
|---|---|---|---|---|---|
| `comopi` | 8 台包装设备、150,650 个十分钟 bin、123 类报警 | `comopi_alarm_counts` | public/acquired | `alarm_forecasting`, `bad_actor_analysis`, `machine_state_classification` | 已取得；正式榜单仍需冻结 split 与参考分数 |
| `enas` | 219,893 条离散传感器、执行器和人工错误状态记录 | `enas_event_log` | public/acquired | `alarm_sequence_modeling`, `anomaly_detection`, `root_cause_analysis` | 已取得；正式榜单仍需冻结 split 与参考分数 |
| `fcc_alarm` | 1,600 个 FCC 仿真 run、16 类异常、57 个报警位及 4,800 个配套时序 CSV | `fcc_alarm_series`, `fcc_alarm_timeseries` | public/acquired | `alarm_flood_classification`, `alarm_sequence_modeling`, `open_set_classification`, `robustness`, `root_cause_analysis` | alarm/process adapter、G0、grouped split 与首批 9 个实验已完成；多 seed/论文协议待补 |
| `imaks` | 211,200 条带异常和因果真值的合成 MQTT/传感器记录 | `imaks_synthetic` | synthetic | `alarm_sequence_modeling`, `robustness`, `root_cause_analysis` | 仅用于合成因果/鲁棒性验证，不得作为真实工业性能 |
| `npp_alarm_dataport` | 101 个阈值层；每层 1,212 个 run、12 类事故/扰动加 Normal、192 个二值报警位 | `npp_alarm_dataport_archive` | public/acquired | `alarm_flood_classification`, `alarm_sequence_modeling`, `open_set_classification`, `root_cause_analysis` | 原始载荷已取得；专用 adapter、grouped split 和正式实验待完成 |
| `piade` | 5 台包装设备；429,394 行原始记录及 23,376 行小时序列 | `piade_sequences`, `piade_raw` | public/acquired | `alarm_forecasting`, `alarm_sequence_modeling`, `bad_actor_analysis`, `machine_state_classification` | 已取得；正式榜单仍需冻结 split 与参考分数 |
| `pronto` | 1.72 GB 多相流实验设施数据；过程、报警和故障标签 | `pronto_full` | public/acquired | `alarm_flood_analysis`, `alarm_generation`, `root_cause_analysis` | T4 使用故障窗代理，不是专家洪泛类别 |
| `skab` | 35 个水循环异常实验 CSV | `skab` | public/acquired | `alarm_generation`, `anomaly_detection`, `robustness` | 已取得；正式榜单仍需冻结 split 与参考分数 |
| `smd10towfgr` | 10 台风机 SCADA；230,618 条日志、167 个事件代码 | `smd10towfgr` | public/acquired | `alarm_forecasting`, `alarm_sequence_modeling`, `bad_actor_analysis`, `machine_state_classification` | 已取得；正式榜单仍需冻结 split 与参考分数 |
| `tep_alarm_dataport` | 16.98 GB；100 个 Tests run、1,000 条五类报警序列及异常场景变体 | `tep_alarm_dataport_archive` | public/acquired | `alarm_flood_classification`, `alarm_sequence_modeling`, `open_set_classification`, `root_cause_analysis` | 五类 ZIP adapter、G0、seeded split 与首批 6 个实验已完成；100-run/异常变体及论文 exact protocol 待补 |
| `tep_classic` | TEP 经典过程仿真；44 个 run、52 个变量 | `tep_classic` | public/acquired | `alarm_generation`, `fault_detection`, `root_cause_analysis` | 已取得；正式榜单仍需冻结 split 与参考分数 |

另有 4 个不可报告成绩的 smoke 生成器：`synthetic_step_fault`、`synthetic_multivariate`、`synthetic_root_cause`、`synthetic_alarm_floods`。

## 5. 下游任务清单

| ID | 任务 | 当前状态 | 数据集族 | 输出 | 说明 |
|---|---|---|---|---|---|
| `T1` | 报警生成与参数设计 | `runnable_real_data` | `tep_classic`, `skab`, `pronto` | `binary_alarm`, `FAR`, `MAR`, `AAD` | 已有真实或已取得数据入口；正式榜单仍需统一 split。 |
| `T2` | 多变量动态报警限 | `runnable_real_data` | `tep_classic`, `skab`, `pronto` | `normal_operating_zone`, `dynamic_limit`, `alarm_state` | 已有真实或已取得数据入口；正式榜单仍需统一 split。 |
| `T3` | 因果图与根因排序 | `runnable_real_data` | `tep_classic`, `pronto`, `enas`, `imaks`, `tep_alarm_dataport`, `npp_alarm_dataport`, `fcc_alarm` | `directed_edges`, `root_cause_ranking` | 已有真实或已取得数据入口；正式榜单仍需统一 split。 |
| `T4` | 报警洪泛检测、聚类与分类 | `runnable_real_data_primary_partial` | `pronto`, `smd10towfgr`, `tep_alarm_dataport`, `npp_alarm_dataport`, `fcc_alarm` | `flood_intervals`, `class_label`, `open_set_label`, `prefix_prediction_set` | TEP 五类与 FCC 专用 adapter、G0、grouped split 和首批实验已完成；NPP adapter、多 seed 与论文 exact protocol 待完成；PRONTO 仅保留为错配哨兵。 |
| `T5` | next-alarm 与洪泛预测 | `runnable_real_data` | `piade`, `pronto`, `comopi`, `smd10towfgr`, `enas`, `tep_alarm_dataport`, `fcc_alarm` | `next_tag`, `future_alarm_set`, `early_warning` | 已有真实或已取得数据入口；正式榜单仍需统一 split。 |
| `T6` | 运维可视分析 | `runnable_real_data` | `piade`, `pronto`, `comopi`, `smd10towfgr`, `tep_alarm_dataport` | `KPI`, `bad_actor`, `correlation_graph`, `flood_visual_report` | 已有真实或已取得数据入口；正式榜单仍需统一 split。 |

## 6. 当前主要缺口与优先顺序

1. 补齐 23 篇论文全文，更新 PDF SHA-256、页码证据和 ARA evidence；其中 22 篇访问受限、1 篇自动下载遭遇 HTTP 403。
2. 为 NPP Alarm 建立只读 adapter；补齐 TEP 100-run/异常变体入口，并按 run/事故族/异常族生成稳定样本 ID。
3. 建立首个 leaderboard-eligible grouped split；训练期确定全部超参，测试期冻结，报告多 seed 与 95% CI。
4. TEP/FCC 已重跑 CASIM、CTFH、HDAM、ConE-AFC、Cross-Conformal；下一步补书籍序列方法、NPP、时间直方图与多 seed，保留 PRONTO 退化证据。
5. 合法取得 CASIM、ConE-AFC 等官方 Code Ocean 工件，并复跑论文代表表格；在此之前 30 项算法均保持 `partial`。
6. 完成 open-set 类别留一、prefix 早期分类、missing/spurious/jitter/delay 鲁棒性矩阵和跨数据集迁移实验。

## 7. 复现入口

```powershell
python scripts/data_acquisition/audit_public_datasets.py
python scripts/data_acquisition/profile_public_datasets.py
python scripts/literature/verify_ara_collection.py
python scripts/validate_scaffold.py
python scripts/audit_benchmark_coverage.py
python -m pytest -q
```

详细机器状态见 `docs/status_audit.json`，论文状态见 `papers/literature/download_manifest.json`，算法闭环账本见 `configs/algorithms/`。
