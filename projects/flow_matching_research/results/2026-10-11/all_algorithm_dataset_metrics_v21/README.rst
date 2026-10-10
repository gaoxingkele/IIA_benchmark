完整算法—数据集指标 v21
==========================

发布UTC：2026-10-10T22:55:19.801776+00:00。7283项指标、1790项有实测值、5493项尚缺完整结果。仅更新GiFlow发布源码分支Air36/main/20%point的四项聚合，独立完整训练种子由1/5增至2/5；条目数不是实验次数。

全部指标：results.html、all_algorithm_dataset_metrics.csv、all_metrics.md。

| 原生指标 | 均值 | STD | 完成／预定 |
| --- | --- | --- | --- |
| native_mean_batch_mae | 9.848085 | 0.273502 | 2/5 |
| native_mean_batch_mse | 397.380484 | 53.148964 | 2/5 |
| native_mean_batch_mape_percent | 22.047222 | 0.750083 | 2/5 |
| sqrt_native_mean_batch_mse | 19.912055 | 1.334593 | 2/5 |

新增seed394按原300轮上限、40轮早停完成196轮/9408次更新；全量1747窗口/14批次/264837缺失目标数组与原生批次均值指标独立核验通过。seed393为247轮/11856次更新。STD/SE/Student-t区间来自两个独立训练种子；不增加重复推断种子。发布源码的test-in-validation和每批EMA重置限制保留，不能称作者等价或严格TAB成绩。

seed394完整原生检查点重推断已启动，当前未声明完成；原始20步Euler、完整模型、数据和冻结容差均保留。此前两个seed393分支的完整模型重推断证据继续保留。

新增证据：../giflow_released_seed394_full_arrays_v1/full_array_audit.json；全部140原始槽：../giflow_canonical_listing_v4/canonical_jobs.csv；完整模型执行：../giflow_released_seed394_native_execution_v1/launch.json。

其余7279条指标与各轨道原采集时间不变。45论文/681方法基线消融/273原始任务及采用基线仍未全部完成；所有未完成范围继续保留。
