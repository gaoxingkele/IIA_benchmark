# GiFlow 首个完整原生种子及全部规范槽位

140个原始种子槽位、28个五种子组全部保留。当前快照：1个完整、2个旧失败／部分尝试保留、137个待运行；尚无完整五种子组。重试不增加种子数。

Air36 / released_mirror / main / seed393：原生预算300轮，按patience40早停实际完成247轮、11856次优化器更新（每轮48个batch128/drop_last批次，6267个原始训练窗口）。原生完整测试覆盖1747个窗口、14个批次。

| 原生指标 | 实测值 |
|---|---:|
| 不加权batch均值MAE | 9.6546900613 |
| 不加权batch均值MSE | 359.7984918867 |
| 不加权batch均值MAPE（%） | 21.5168329648 |
| sqrt(native mean batch MSE) | 18.9683550127 |

这是1/5种子，不提供种子STD或区间。保持原生重叠窗口及batch均值，不将其换算成异常检测F1。released_mirror保留测试加入验证、训练阶段每批重置EMA、最终EMA权重评价，不能当作无测试泄漏的泛化成绩。完整数组／指标已[独立复核](../giflow_first_full_array_audit_v2/README.md)，训练模型独立重推断尚未完成，作者等价未认证。

[全部规范任务状态](canonical_jobs.csv)、[全部112条指标与空缺](algorithm_dataset_metrics.csv)、[完整执行证据](execution_audit.json)、[SHA-256校验](validation.json)。论文及原始代码引用保留在冻结队列和模型配置中。

这不是全部140项实验完成，也不代表所有45篇论文、681条方法记录、273条原始数据／任务记录完成。
