# 完整算法—数据集实验技术指标

本清单列出全部已登记组合及缺失值。不同任务保留各自核验时间。全论文、全方法、全消融尚未全部完成。

[全部6923条汇总指标 CSV](all_algorithm_dataset_metrics.csv)；[全部6923条可阅读指标](all_metrics.md)。

严格异常检测：730个配置—数据集组合，206个完成预定重复，9个仅完成部分重复，515个暂无完整运行。快照为2026-10-09 10:24:49（北京时间），核验958次完整运行。

CFM-TS：{'completed': 54, 'pending': 231}；核验时间2026-10-09T04:39:37.089817+00:00（UTC）。285原始种子槽位、57组比较；精确重跑不增加种子。

GRASP：本次保留4个已完成原生组，并纳入main和Transformer，共6个全实体完成种子组。更新核验时间2026-10-09T04:44:46.180646+00:00（UTC）。main/Transformer在SWAN均只完成1/10种子，不能生成STD或置信区间。

概率指标按0–1列出，误差保留原量纲。—表示缺失，0为实测零值。n为该指标实际重复数，范围指标可能少于点级指标。

严格主表用验证分数第99百分位阈值，无PA；作者PA、测试标签最优阈值、发布权重评价及历史协议均单列。窗口与更新预算尚未全部对齐，未认证作者等价。PRONTO正常段选择使用标签，SKAB校准含原生异常，TEP为经典单工况整运行。

完整逐种子字段还包含事件召回、漏检、检测延迟、误报/千正常点、参数量、训练/评分秒数和CUDA峰值。延迟来自离线窗口分数。确定性统计方法重复拟合不作随机训练置信区间。

| 任务 | 汇总指标行数（含缺失） |
|---|---:|
| anomaly_detection | 6368 |
| imputation | 328 |
| continuous_time_dynamics | 171 |
| unconditional_time_series_generation | 56 |

完整明细入口：

- [全部730个异常检测组合的核心指标](strict_algorithm_dataset.md)
- [按数据集的F1矩阵](../complete_metrics_1026/overview.md)
- [16工作表Excel（严格与插补快照，CFM/GRASP最新更新见本清单）](../complete_metrics_1026/complete_experiment_metrics.xlsx)
- [2874行逐种子严格指标和资源](../complete_metrics_1026/strict_per_seed.csv)
- [42576行逐实体严格指标](../complete_metrics_1026/strict_per_entity.csv)
- [作者完整流程的314行数值字段](../complete_metrics_1026/author_pipeline_numeric_metrics.csv)
- [548行历史协议指标](../complete_metrics_1026/historical_protocol_results.csv)
- [9580行TAB比例诊断与PA对照](../complete_metrics_1026/tab_ratio_diagnostic.csv)
- [173行插补逐次指标](../complete_metrics_1026/imputation_per_run.csv)
- [已核验的71条论文报告值](../complete_metrics_1026/paper_claims.csv)
- [45篇论文的273条原始数据/任务范围及未完成义务](../complete_metrics_1026/original_dataset_scope.csv)
- [CFM-TS全部57组、论文值及差异](../cfm_recovery_1238/README.md)
- [最新GRASP完整数组独立审计](../native_recovery_1245/independent_array_audit.json)

这里的6923行是指标行数，并非6923次训练。作者流程与历史记录未混入严格主表。登记范围以外的原始任务缺口在原文范围清单中明示，不假设已有成绩。
