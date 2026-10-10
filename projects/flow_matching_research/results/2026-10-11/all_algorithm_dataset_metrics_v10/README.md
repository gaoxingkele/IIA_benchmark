# 完整算法—数据集实验结果技术指标

发布：2026-10-10T19:51:48.351625+00:00。本版更新 GiFlow 原始插补结果，其余轨道沿用已有快照，各自核验时间见CSV。

**7117条汇总指标：1780条有实测值，5337条暂无完整结果。指标行数不等于训练次数。**

[全部异常检测配置的F1矩阵](overview.md)；[可搜索、筛选的全部指标](results.html)；[全量CSV](all_algorithm_dataset_metrics.csv)；[全量可阅读表](all_metrics.md)。

Table4/5的n表示生成器训练种子数。5/10次评价的波动单列evaluator字段，不当作训练种子区间。

| 任务 | 汇总指标行数（含缺失） |
|---|---:|
| anomaly_detection | 6402 |
| imputation | 440 |
| continuous_time_dynamics | 171 |
| unconditional_time_series_generation | 86 |
| physics_informed_generation | 4 |
| irregular_time_series_generation | 14 |

严格异常检测：730个配置—数据集组合，210个完成预定重复、9个部分完成、511个暂无完整结果；完整运行970次。核验时间2026-10-10T17:38:48.560237+00:00。

CFM-TS连续动力学：{'completed': 83, 'pending_exact_recovery': 2, 'running': 1, 'pending': 199}，16/57个五种子组完整，原始种子槽位共285。已有83个早先完成槽的检查点重推断逐位一致；本次新完成槽未自动获得这一复核标记。

GRASP：13/480个全实体种子组完整，精确恢复不增加种子。只纳入全实体完成组，不从部分实体推算论文宏平均。13个早先完成槽已独立数组审计；新增完成槽需另行复核。训练模型重推断尚未完成。

插补：61/540项完整；MAE/RMSE/CRPS/MRE与缺失率、掩码分别列行。

概率指标0–1，误差保留原量纲；—为缺失、0为实测零。n为每一指标的实测重复数。单种子无种子STD/区间；确定性统计重复拟合不解释成随机训练不确定性。

严格主表：验证分数第99百分位阈值，无PA。测试标签最优阈值、作者PA、发布权重评价、历史记录与严格主表分别列出。非重叠窗口与部分作者stride=1的更新次数尚不一致；没有作者等价认证。PRONTO正常段选择使用标签，SKAB校准含异常，TEP为经典单工况整运行。

完整表和逐次技术明细：

- [全部严格异常检测组合：P/R/F1、AUROC、AP、VUS、Affiliation、STD/CI](strict_algorithm_dataset.md)
- [逐种子三档阈值、事件召回、漏检、延迟、误报率、参数量、训练/评分耗时、CUDA峰值](../complete_metrics_current_v5/strict_per_seed.csv)
- [逐实体结果](../complete_metrics_current_v5/strict_per_entity.csv)
- [全部插补指标与完成状态](../complete_metrics_current_v5/imputation_summary.csv)
- [CFM-TS全部57组、论文MSE及差值](../cfm_current_listing_v5/README.md)
- [作者完整流程及其协议数值](../complete_metrics_current_v5/author_pipeline_numeric_metrics.csv)
- [全部作者流程任务状态](../complete_metrics_current_v5/author_pipeline_jobs.csv)
- [历史协议结果](../complete_metrics_current_v5/historical_protocol_results.csv)
- [TAB比例与PA诊断；测试标签选比例，不用于严格泛化排名](../complete_metrics_current_v5/tab_ratio_diagnostic.csv)
- [71条已核验论文报告值](../complete_metrics_current_v5/paper_claims.csv)
- [45篇论文273条原始数据/任务范围与未完成义务](../complete_metrics_current_v5/original_dataset_scope.csv)
- [681条方法/基线/消融库存](../../../../../configs/reproducibility/fm_paper_method_inventory.v1.json)
- [生成原始任务状态，包括新增Table2/3/4/5待完成项](original_generation_jobs.csv)

表完整列出了已登记指标及其空缺。681条文献方法记录不等于681个已复现算法；未登记或未完成的原始任务继续保留在范围清单中。全部论文、全部方法和全部消融实验尚未完成。

## GiFlow 原始插补任务

本版本纳入 GiFlow 的 140 个原始种子槽位、28 个五种子组和 112 条插补指标及空缺。修正观察计数的重试不增加种子；未完成或缺失实际更新证据的旧结果不填数值。作者镜像的测试参与验证分支与修正的验证检查点分支分别保留。MAE、MSE、MAPE（百分数）和 sqrt(native mean batch MSE) 保持原代码单位；这些指标来自重叠测试窗口、batch 均值的不加权平均，不能当作异常检测 F1。训练模型独立重推断未完成。

## 最新Pi双种子指标

PSM causal_axis_repaired/full：2/6种子；作者阈值＋PA F1 97.8869%，同阈值无PA F1 1.9109%，无PA AUROC 0.487928、AP 0.275030。保留原测试早停和训练／测试联合校准，不进入严格泛化排名。更新8条已有指标，未新增训练种子槽位。[完整审计](../pi_two_full_seed_audit_v2/README.md)。

新注册LS4共轭修正与SaShiMi原始数据任务尚无本次核验的完整训练分数；其完整CPU数据审计和数学／接口诊断不列为实验性能。[LS4状态](../ls4_cauchy_corrected_resume_v3/README.md)、[SaShiMi状态](../sashimi_monash_gaussian_resume_v2/README.md)。

## 首个完整 GiFlow 原生种子

Air36／released_mirror／main／seed393：按原生早停实际完成247轮、11856次更新（预算300轮，每轮48个drop_last批次）。全部14个测试批次的已保存预测、目标和掩码位置已独立核验，完整数组重算MAE9.6546902、MSE359.7984835、MAPE21.5168332%、sqrt(native mean batch MSE)18.9683548。保持重叠窗口、不加权batch均值、测试参与验证及EMA重置行为。

这是1/5种子，没有种子STD或区间；140个原始槽位目前1个完整，28个五种子组尚无全部完成组。完整数组复核不等于训练模型重推断；未声明作者等价或严格TAB结果。

[完整规范槽位和指标](../giflow_canonical_listing_v2/README.md)、[独立全测试数组审计](../giflow_first_full_array_audit_v2/README.md)。

LS4 PhysioNet 已接入原生500轮训练和50次全测试评测，10个任务共750000次预定更新；完整CPU批量诊断达到内存上限的失败已保留，后继队列执行完整GPU容量验证，尚无该队列的完整500轮训练分数。[执行状态及协议限制](../ls4_native_physionet_execution_v2/README.md)。
