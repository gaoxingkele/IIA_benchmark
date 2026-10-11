# 完整算法—数据集实验技术指标清单 v37

当前登记147个算法配置，共8098项指标，其中2088项有实测值，6010项暂无结果。这是登记指标数量，不是已完成实验次数；不能称为全部论文已复现。

[全部8098项指标 CSV](all_algorithm_dataset_metrics_v37/all_algorithm_dataset_metrics.csv) · [完整逐项 Markdown](all_algorithm_dataset_metrics_v37/all_metrics.md) · [可筛选 HTML](all_algorithm_dataset_metrics_v37/results.html)

[异常检测全部点级与范围指标](all_algorithm_dataset_metrics_v37/strict_algorithm_dataset.md) · [异常检测F1矩阵](all_algorithm_dataset_metrics_v37/overview.md) · [CrossAD原生全部组合](all_algorithm_dataset_metrics_v37/crossad_complete_native_metric_grid.csv)

| 任务 | 登记指标 | 已有实测值 | 缺结果 |
| --- | ---: | ---: | ---: |
| 异常检测 | 7217 | 1979 | 5238 |
| 缺失值插补 | 440 | 54 | 386 |
| 连续时间动力系统 | 171 | 51 | 120 |
| 无条件时间序列生成 | 112 | 4 | 108 |
| 不规则时间序列生成 | 22 | 0 | 22 |
| 原生不规则时间序列重建 | 132 | 0 | 132 |
| 物理约束生成 | 4 | 0 | 4 |

每条指标保留算法/消融配置、数据集、任务、评价协议、指标名称、已完成与要求的种子数、均值、标准差、标准误、95%置信区间、论文值（有原始依据时）、原始结果路径、缺失模式与比例、训练轮数、选择边界及等价认证状态。

异常检测包含 Precision、Recall、F1、AUROC、AP、范围AUC、VUS-ROC、VUS-PR、Affiliation 等；插补包含 MAE、RMSE、MRE 等；其他任务指标按原论文与实际登记协议逐项保留。

验证集1%阈值/no-PA、测试标签最优阈值、PA、原作者范围指标和固定预训练权重评价分开。点级为完整测试点微平均，范围为实体宏平均时不可混用。SMAP历史预处理值保留历史身份，修正重跑尚未完成。

CrossAD原139任务登记为29个方法版本—数据集组合，每组29项指标。新增MSL原生主方法seed2025的29项经过完整模型与原作者指标重放的实测值；训练种子仅1/6，尚无种子置信区间。原生平铺数据与标准化边界保留，完整TAB及作者等价没有认证。

本轮30项相关测试通过；独立核验旧7283条完整字段和CSV前缀不变，新增815条槽位与29个实测值准确。完整失败尝试保留。
