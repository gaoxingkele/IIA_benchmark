# 完整算法—数据集实验技术指标

发布UTC：2026-10-10T22:02:20.865698+00:00。共7275项指标、1790项有实测值、5485项缺完整结果，覆盖140个算法／配置名称。指标条数不是训练次数；原方向45论文/681方法基线消融/273原始任务仍未全部完成。

[可筛选全部结果](results.html)；[完整CSV](all_algorithm_dataset_metrics.csv)；[全部文字指标](all_metrics.md)；[异常检测F1矩阵](overview.md)；[异常检测全部指标](strict_algorithm_dataset.md)。

| 任务 | 登记指标 | 有实测值 | 缺完整结果 |
| --- | --- | --- | --- |
| anomaly_detection | 6402 | 1688 | 4714 |
| continuous_time_dynamics | 171 | 51 | 120 |
| imputation | 440 | 51 | 389 |
| irregular_time_series_generation | 22 | 0 | 22 |
| native_irregular_time_series_reconstruction | 132 | 0 | 132 |
| physics_informed_generation | 4 | 0 | 4 |
| unconditional_time_series_generation | 104 | 0 | 104 |

GiFlow／Air36／reviewed_corrected／main／20%point／seed393完成原生早停68轮、3264次优化；14批/1747窗口/268613目标全部预测数组和原生批均值指标独立重算通过。仍为1/5训练种子，不制造STD/SE/95%区间，原完整权重CPU重推断仍在运行，未称完成。[完整数组证据](../giflow_corrected_full_arrays_v1/full_array_audit.json)；[原生完整权重重推断登记](../giflow_corrected_execution_v1/launch.json)。

| 指标 | 原生完整训练值 | 完成／预定 |
| --- | --- | --- |
| native_mean_batch_mae | 10.951781 | 1/5 |
| native_mean_batch_mse | 413.131975 | 1/5 |
| native_mean_batch_mape_percent | 23.823092 | 1/5 |
| sqrt_native_mean_batch_mse | 20.325648 | 1/5 |

发布源码分支同种子MAE9.654690，修正分支10.951781；修正分支使用实际验证集、持续EMA和验证选中权重。多处修正及权重选择同时改变，单个配对种子不能归因，不能按低分丢弃结果或声称论文一致。

新增TimeGAN-native的三个原始数据集×三个原始RNN选项×两个原始指标=18待完成聚合槽；每槽5个完整生成器种子，每生成器各10次判别/预测评估训练。这些评估器重复不另计生成器种子。45任务仍保留每阶段50000循环、完整batch128/seq24/hidden24/layers3和原始2000/5000评估更新。[原生注册/构图/全量数据证据](../timegan_native_preparation_v1/README.rst)。暂无TimeGAN完整成绩；原始源码默认参数为明确本地选择，历史优化参数和采用它的其他论文不规则数据任务仍待完成。

其他全部轨道结果和采集时间与v17逐行保留；没有重新采集异常检测、SAITS、CFM等轨道。验证集阈值无PA、测试标签选阈值、PA、作者原生协议分别列出；尚未获完整TAB预算或作者等价认证。
