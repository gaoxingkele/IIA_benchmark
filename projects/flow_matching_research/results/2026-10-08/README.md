# 算法—数据集实验技术指标

抓取时间：2026-10-08T15:36:46.099531+00:00 至 2026-10-08T15:36:47.287974+00:00。

本表保留全部540个冻结任务、328条聚合指标、126条每折指标、71条已转录论文记录及45篇的273条原始数据集描述。36个任务已有结果；有本地数值的聚合指标28条。尚未转录的论文表不推断数值。

## 已有本地指标

MAE、RMSE、CRPS和normalized CRPS均越低越好。PhysioNet为标准化变量，Air-36的MAE/RMSE/CRPS为原物理尺度；不可跨数据集直接比较大小。Air-36当前三折均值是临时汇总；CSDI配置中的缺失率0仅关闭新随机掩码，评估使用历史掩码，不能解释为零缺失。

| 算法 | 数据集 | 缺失设置 | 完成/预定 | MAE | RMSE | CRPS | normalized CRPS |
|---|---|---|---|---|---|---|---|
| CSDI | PhysioNet2012 | 10% | 5/5 | 0.218292 | 0.505890 | — | 0.239517 |
| CFMI | PhysioNet2012 | 10% | 5/5 | 0.241406 | 0.549803 | 0.241007 | 0.255534 |
| CSDI | PhysioNet2012 | 50% | 5/5 | 0.301308 | 0.624481 | — | 0.330745 |
| CFMI | PhysioNet2012 | 50% | 5/5 | 0.337274 | 0.660839 | 0.331166 | 0.352914 |
| CSDI | PhysioNet2012 | 90% | 5/5 | 0.474957 | 0.793024 | — | 0.517923 |
| CFMI | PhysioNet2012 | 90% | 5/5 | 0.554975 | 0.887946 | 0.511680 | 0.580323 |
| CSDI | Air-36 | 原作者历史掩码 | 3/5 | 9.596554 | 18.741258 | — | 0.106694 |
| CFMI | Air-36 | 原作者历史掩码 | 3/5 | 9.577618 | 18.308786 | 6.830009 | 0.099528 |

## 原论文逐指标比较

标准误SE与95% t区间均保存在结构化文件；区间只描述已完成的折/种子，兼容筛查不证明统计等效。

| 算法 | 数据集 | 缺失设置 | 指标 | 本地均值 ± SE | 论文均值 ± SE | 状态 |
|---|---|---|---|---|---|---|
| CSDI | PhysioNet2012 | 10% | mae | 0.218292 ± 0.000629 | 0.217000 ± 0.001000 | 数值兼容筛查通过 |
| CSDI | PhysioNet2012 | 10% | rmse | 0.505890 ± 0.021293 | — ± — | 无登记论文参考 |
| CSDI | PhysioNet2012 | 10% | normalized_crps | 0.239517 ± 0.001311 | 0.238000 ± 0.001000 | 数值兼容筛查通过 |
| CFMI | PhysioNet2012 | 10% | mae | 0.241406 ± 0.002110 | 0.234000 ± 0.002000 | 数值兼容筛查通过 |
| CFMI | PhysioNet2012 | 10% | rmse | 0.549803 ± 0.045544 | 0.538000 ± 0.052000 | 数值兼容筛查通过 |
| CFMI | PhysioNet2012 | 10% | crps | 0.241007 ± 0.007659 | 0.220000 ± 0.004000 | 数值兼容筛查通过 |
| CFMI | PhysioNet2012 | 10% | normalized_crps | 0.255534 ± 0.001830 | 0.250000 ± 0.004000 | 数值兼容筛查通过 |
| CSDI | PhysioNet2012 | 50% | mae | 0.301308 ± 0.001173 | 0.301000 ± 0.002000 | 数值兼容筛查通过 |
| CSDI | PhysioNet2012 | 50% | rmse | 0.624481 ± 0.032961 | — ± — | 无登记论文参考 |
| CSDI | PhysioNet2012 | 50% | normalized_crps | 0.330745 ± 0.001490 | 0.330000 ± 0.002000 | 数值兼容筛查通过 |
| CFMI | PhysioNet2012 | 50% | mae | 0.337274 ± 0.002894 | 0.319000 ± 0.005000 | 超出比较范围 |
| CFMI | PhysioNet2012 | 50% | rmse | 0.660839 ± 0.016612 | 0.624000 ± 0.019000 | 数值兼容筛查通过 |
| CFMI | PhysioNet2012 | 50% | crps | 0.331166 ± 0.013786 | 0.282000 ± 0.010000 | 超出比较范围 |
| CFMI | PhysioNet2012 | 50% | normalized_crps | 0.352914 ± 0.002716 | 0.338000 ± 0.007000 | 数值兼容筛查通过 |
| CSDI | PhysioNet2012 | 90% | mae | 0.474957 ± 0.001664 | 0.481000 ± 0.003000 | 数值兼容筛查通过 |
| CSDI | PhysioNet2012 | 90% | rmse | 0.793024 ± 0.017760 | — ± — | 无登记论文参考 |
| CSDI | PhysioNet2012 | 90% | normalized_crps | 0.517923 ± 0.002074 | 0.522000 ± 0.002000 | 数值兼容筛查通过 |
| CFMI | PhysioNet2012 | 90% | mae | 0.554975 ± 0.008909 | 0.504000 ± 0.009000 | 超出比较范围 |
| CFMI | PhysioNet2012 | 90% | rmse | 0.887946 ± 0.012818 | 0.822000 ± 0.019000 | 超出比较范围 |
| CFMI | PhysioNet2012 | 90% | crps | 0.511680 ± 0.032079 | 0.410000 ± 0.014000 | 超出比较范围 |
| CFMI | PhysioNet2012 | 90% | normalized_crps | 0.580323 ± 0.008218 | 0.538000 ± 0.019000 | 数值兼容筛查通过 |
| CSDI | Air-36 | 历史掩码 | mae | 9.596554 ± 0.134053 | 9.600000 ± 0.040000 | 协议不可严格比较 |
| CSDI | Air-36 | 历史掩码 | rmse | 18.741258 ± 0.561071 | — ± — | 重复次数不足 |
| CSDI | Air-36 | 历史掩码 | normalized_crps | 0.106694 ± 0.001548 | 0.108000 ± 0.001000 | 协议不可严格比较 |
| CFMI | Air-36 | 历史掩码 | mae | 9.577618 ± 0.115383 | 9.404000 ± 0.050000 | 重复次数不足 |
| CFMI | Air-36 | 历史掩码 | rmse | 18.308786 ± 0.310275 | 17.701000 ± 0.106000 | 重复次数不足 |
| CFMI | Air-36 | 历史掩码 | crps | 6.830009 ± 0.080446 | 6.726000 ± 0.036000 | 重复次数不足 |
| CFMI | Air-36 | 历史掩码 | normalized_crps | 0.099528 ± 0.001054 | 0.098000 ± 0.001000 | 重复次数不足 |

## 未完成的实验范围

| 算法 | 数据集 | 已完成指标组/全部组 | 备注 |
|---|---|---|---|
| CSDI | Air-36 | 0/3 | author_protocol；完整缺失率/掩码/种子见任务清单 |
| CFMI | Air-36 | 0/4 | author_protocol；完整缺失率/掩码/种子见任务清单 |
| SAITS | PhysioNet2012 | 0/3 | author_protocol；完整缺失率/掩码/种子见任务清单 |
| BRITS | PhysioNet2012 | 0/3 | author_protocol；完整缺失率/掩码/种子见任务清单 |
| SAITS | AirQuality-132 | 0/3 | author；完整缺失率/掩码/种子见任务清单 |
| BRITS | AirQuality-132 | 0/3 | author；完整缺失率/掩码/种子见任务清单 |
| SAITS | Electricity | 0/27 | author；完整缺失率/掩码/种子见任务清单 |
| BRITS | Electricity | 0/27 | author；完整缺失率/掩码/种子见任务清单 |
| SAITS_BASE | ETT | 0/3 | author；完整缺失率/掩码/种子见任务清单 |
| SAITS | AirQuality-132 | 0/3 | corrected；完整缺失率/掩码/种子见任务清单 |
| BRITS | AirQuality-132 | 0/3 | corrected；完整缺失率/掩码/种子见任务清单 |
| SAITS | Electricity | 0/27 | corrected；完整缺失率/掩码/种子见任务清单 |
| BRITS | Electricity | 0/27 | corrected；完整缺失率/掩码/种子见任务清单 |
| SAITS_BASE | ETT | 0/3 | corrected；完整缺失率/掩码/种子见任务清单 |
| GRIN | Air-36 | 0/3 | author_protocol；完整缺失率/掩码/种子见任务清单 |
| GRIN | AQI-437 | 0/3 | author_protocol；完整缺失率/掩码/种子见任务清单 |
| CSDI | TEP | 0/27 | industrial_imputation_transfer；完整缺失率/掩码/种子见任务清单 |
| CFMI | TEP | 0/27 | industrial_imputation_transfer；完整缺失率/掩码/种子见任务清单 |
| CSDI | SKAB | 0/27 | industrial_imputation_transfer；完整缺失率/掩码/种子见任务清单 |
| CFMI | SKAB | 0/27 | industrial_imputation_transfer；完整缺失率/掩码/种子见任务清单 |
| CSDI | PRONTO | 0/27 | industrial_imputation_transfer；完整缺失率/掩码/种子见任务清单 |
| CFMI | PRONTO | 0/27 | industrial_imputation_transfer；完整缺失率/掩码/种子见任务清单 |

## 异常检测和消融

本项目目前没有完整登记的原论文或TAB异常检测F1/AUC结果。插补误差不代替异常检测成绩，PSM训练集21个配置预检也不构成F1。

| 原论文方法 | 原论文数据集范围 | 本地正式异常检测/消融成绩 |
|---|---|---|
| grasp | SMAP (mTSBench 51 series,26 variables)；SMD (mTSBench 18 series,39 variables)；CICIDS2017 (mTSBench six series,73 features)；SWAN (mTSBench one 39-variable series) | 尚未完成严格验收的正式结果 |
| dfm | SMAP | 尚未完成严格验收的正式结果 |
| ganf | SWaT Dec2015 attack_v0；METR-LA；PMU proprietary | 尚未完成严格验收的正式结果 |
| catch | MSL；PSM；SMD；CICIDS；CalIt2；NYC；Creditcard；GECCO；Genesis；ASD；TODS synthetic 12 sets | 尚未完成严格验收的正式结果 |
| dcdetector | SMD；MSL；SMAP；PSM；SWaT；NIPS-TS-SWAN；NIPS-TS-GECCO (repository NIPS_TS_Water)；UCR | 尚未完成严格验收的正式结果 |
| maelnet | SMAP；MSL；SMD；PSM；SWaT | 尚未完成严格验收的正式结果 |
| pi_transformer | SMAP；MSL；SMD；PSM；SWaT | 尚未完成严格验收的正式结果 |
| dt_la | SMAP；MSL；SMD；PSM；SWaT | 尚未完成严格验收的正式结果 |
| shcl | SMAP；MSL；SMD；PSM；SWaT；GECCO；SWAN；UCR | 尚未完成严格验收的正式结果 |
| kgl | SMD；SMAP | 尚未完成严格验收的正式结果 |
| crossad | SMD；MSL；SMAP；SWaT；PSM；GECCO；SWAN | 尚未完成严格验收的正式结果 |
| omnianomaly | SMD；MSL；SMAP | 尚未完成严格验收的正式结果 |
| sensitivehue | SWaT；WADI；MSL；SMD | 尚未完成严格验收的正式结果 |
| usad | SWaT；WADI；SMD；SMAP；MSL；Orange proprietary | 尚未完成严格验收的正式结果 |
| anomaly_transformer | SMAP；MSL；SMD；PSM；SWaT | 尚未完成严格验收的正式结果 |

## 数据与文件

- [完整结构化快照](experiment_results.json)：来源文件SHA256、协议、均值/STD/SE/区间、配置与运行结果路径。
- [聚合指标CSV](formal_results.csv)、[每折指标CSV](per_run_metrics.csv)、[全部任务CSV](jobs.csv)。
- [论文报告CSV](author_claims.csv)：原单位、物理PDF页码、表号、参考设置ID；SHCL Table2和Table7重复行按各原文来源保留。
- [原始数据集范围CSV](original_dataset_scope.csv)：45篇全部已登记范围；描述未建立结果绑定不等于数据集不存在。
- [完整Excel表](experiment_results.xlsx)；本地Excel位于 `outputs/fm_results_20261008/流匹配算法_数据集_完整实验指标.xlsx`，含六个可筛选工作表；其生成快照与本目录一致。

仅汇总已有实验记录，未启动训练、未覆盖原数据、未改动正在运行的队列和动态状态报告。没有选取最好一次测试结果。耗时按每次记录保存，共享checkpoint可能重复出现，不能按指标行求和。
