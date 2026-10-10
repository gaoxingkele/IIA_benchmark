# GiFlow 完整原生插补结果衔接

本轮解决旧指标汇总未识别 v2 执行包装，以及原生任务一律归入异常检测的问题。没有改动已启动的 GiFlow 模型、原始作者代码、完整预算或已有结果。此处是只读、不可变的结果快照，不是新增训练完成记录。

## 完整范围与当前证据

核验时间见 [execution_audit.json](execution_audit.json)。140 个原始种子槽位、28 个五种子组完整保留；112 条 MAE、MSE、MAPE（百分数）和 sqrt(native mean batch MSE) 指标目前均为空。真实进程命令确认 1 项原生模型在运行，2 项此前失败／未完整登记的结果保留，137 项待执行。旧失败结果的临时计数没有被猜测补写，重试也没有变成新增种子。

作者镜像 `released_mirror` 与修正分支 `reviewed_corrected` 分开，主方法、图／时间／无先验、缺失率 0.3～0.6、图阈值 0.02／0.05／0.2／0.4／0.6 均保留。本地 393～397 五个种子，不代表已恢复作者历史随机状态。

完成判定必须同时满足冻结队列／任务绑定、全部产物 SHA、实际 DataLoader drop_last、逐 epoch 更新数、原生训练日志、完整测试 batch／目标覆盖，以及修正分支实际测试权重与验证检查点一致。未达到 300 epoch 时必须有原始 early stopping 日志。模型独立重推断与预测数组指标独立重算尚未完成，不能据这些登记证据宣布论文等价。

## 指标与协议

这里是插补任务，不是异常检测 F1。指标保持作者 masked overlapping test windows 上 batch 均值的不加权平均。原镜像的测试参与验证、最终 EMA 每 batch 重置，与修正分支的验证检查点／persistent EMA 单列，不能混合排名。

- [全部 140 个原始槽位与运行证据](canonical_jobs.csv)
- [全部 112 条插补指标及空缺](algorithm_dataset_metrics.csv)
- [核验与文件 SHA](validation.json)
- [合并后的全方向指标表](../all_algorithm_dataset_metrics_v8/README.md)

58 项本轮相关测试通过，包括实际更新证据丢失、短预算未提供 early stopping、测试覆盖缩短、错误检查点、错误队列／任务、错误 square-root MSE、诊断冒充正式结果、重试计为额外种子，以及插补错误归类等拒绝检查。单种子不造 STD／区间，缺失不填零。

全方向 v8 共 7117 条汇总指标，1776 条已有值、5341 条缺失；新增的 112 条是 GiFlow 明确缺失项，不是新增训练成绩。旧严格异常检测、CFM、GRASP 等数值沿用已核验快照，各自时间保留在 CSV。完整目标中的 45 篇论文、681 条文献方法／基线／消融记录和 273 条原始数据／任务仍未全部完成。

原论文与代码：[GiFlow](https://arxiv.org/abs/2606.06682)、[作者仓库](https://github.com/zepengzhang/GiFlow)。原模型配置和源获取证据由原始队列保存。
