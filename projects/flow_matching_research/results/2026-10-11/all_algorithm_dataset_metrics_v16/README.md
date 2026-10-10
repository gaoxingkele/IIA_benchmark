# 完整算法—数据集实验技术指标

发布UTC：2026-10-10T21:24:44.980759+00:00。共7257条指标：1786条有实测值，5471条缺完整结果，覆盖137个算法／配置名称。指标条数不是独立训练次数；数据集别名和不同协议保留原名称。

[可搜索完整表](results.html)；[完整CSV](all_algorithm_dataset_metrics.csv)；[全部文字指标](all_metrics.md)；[异常检测F1总览](overview.md)；[异常检测全部技术指标](strict_algorithm_dataset.md)。

| 任务 | 登记指标 | 有实测值 | 缺完整结果 |
| --- | --- | --- | --- |
| anomaly_detection | 6402 | 1688 | 4714 |
| continuous_time_dynamics | 171 | 51 | 120 |
| imputation | 440 | 47 | 393 |
| irregular_time_series_generation | 22 | 0 | 22 |
| native_irregular_time_series_reconstruction | 132 | 0 | 132 |
| physics_informed_generation | 4 | 0 | 4 |
| unconditional_time_series_generation | 86 | 0 | 86 |

本版只更新CFM-TS的171条既有指标槽位；84/285个规范任务已完成，16个五种子组完整。所有84项的全量预测、原始数据身份、训练预算、MSE/MAE/RMSE与种子统计已独立重算。Pendulum/NODE/附录D/10轮由3/5增至4/5种子，其均值MSE=6.625421、MAE=1.887639、RMSE=2.261529；新增种子误差较高，未筛掉。[完整数组核验](../cfm_current_listing_v6/independent_full_metrics_audit.json)。本次未重新执行模型重推断；完整数组核验不等于作者实现等价认证。

异常检测采用各行明示的协议。验证集99百分位定阈值、无PA的工程结果，尚未获完整TAB实验预算或作者等价认证；测试标签选阈值、PA和原生作者协议分别列出。固定发布权重的1/1表示一次评估，不能当作一个重新训练种子。

LS4/PhysioNet原完整500轮训练曾在25轮后触发内存保护，部分结果未计入完整性能；新尝试和其他排队任务均不能作为实测完整指标。[保留中断和恢复记录](../ls4_physionet_memory_guard_recovery_v3/README.md)。本版不重新核验任何运行进程，不将发布日期视为各轨道采集时间。
