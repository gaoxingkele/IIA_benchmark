# 完整算法—数据集实验结果技术指标

发布：2026-10-10T16:43:03.776420+00:00。各轨道独立核验时间保留在CSV；持续运行期间的不可变快照。

**6953条汇总指标：1742条有实测值，5211条暂无完整结果。指标行数不等于训练次数。**

[可搜索、筛选的全部指标](results.html)；[全量CSV](all_algorithm_dataset_metrics.csv)；[全量可阅读表](all_metrics.md)。

| 任务 | 汇总指标行数（含缺失） |
|---|---:|
| anomaly_detection | 6368 |
| imputation | 328 |
| continuous_time_dynamics | 171 |
| unconditional_time_series_generation | 86 |

严格异常检测：730个配置—数据集组合，210个完成预定重复、9个部分完成、511个暂无完整结果；完整运行970次。核验时间2026-10-10T16:40:28.278722+00:00。

CFM-TS连续动力学：{'completed': 83, 'pending_exact_recovery': 2, 'running': 1, 'pending': 199}，16/57个五种子组完整，原始种子槽位共285。已有83个早先完成槽的检查点重推断逐位一致；本次新完成槽未自动获得这一复核标记。

GRASP：13/480个全实体种子组完整，精确恢复不增加种子。只纳入全实体完成组，不从部分实体推算论文宏平均。独立数组审计仍有待执行项。

插补：61/540项完整；MAE/RMSE/CRPS/MRE与缺失率、掩码分别列行。

概率指标0–1，误差保留原量纲；—为缺失、0为实测零。n为每一指标的实测重复数。单种子无种子STD/区间；确定性统计重复拟合不解释成随机训练不确定性。

严格主表：验证分数第99百分位阈值，无PA。测试标签最优阈值、作者PA、发布权重评价、历史记录与严格主表分别列出。非重叠窗口与部分作者stride=1的更新次数尚不一致；没有作者等价认证。PRONTO正常段选择使用标签，SKAB校准含异常，TEP为经典单工况整运行。

完整表和逐次技术明细：

- [全部严格异常检测组合：P/R/F1、AUROC、AP、VUS、Affiliation、STD/CI](strict_algorithm_dataset.md)
- [逐种子三档阈值、事件召回、漏检、延迟、误报率、参数量、训练/评分耗时、CUDA峰值](../complete_metrics_current_v4/strict_per_seed.csv)
- [逐实体结果](../complete_metrics_current_v4/strict_per_entity.csv)
- [全部插补指标与完成状态](../complete_metrics_current_v4/imputation_summary.csv)
- [CFM-TS全部57组、论文MSE及差值](../cfm_current_listing_v4/README.md)
- [作者完整流程及其协议数值](../complete_metrics_current_v4/author_pipeline_numeric_metrics.csv)
- [全部作者流程任务状态](../complete_metrics_current_v4/author_pipeline_jobs.csv)
- [历史协议结果](../complete_metrics_current_v4/historical_protocol_results.csv)
- [TAB比例与PA诊断；测试标签选比例，不用于严格泛化排名](../complete_metrics_current_v4/tab_ratio_diagnostic.csv)
- [71条已核验论文报告值](../complete_metrics_current_v4/paper_claims.csv)
- [45篇论文273条原始数据/任务范围与未完成义务](../complete_metrics_current_v4/original_dataset_scope.csv)
- [681条方法/基线/消融库存](../../../../../configs/reproducibility/fm_paper_method_inventory.v1.json)
- [生成原始任务状态，包括新增Table2/3待完成项](original_generation_jobs.csv)

表完整列出了已登记指标及其空缺。681条文献方法记录不等于681个已复现算法；未登记或未完成的原始任务继续保留在范围清单中。全部论文、全部方法和全部消融实验尚未完成。
