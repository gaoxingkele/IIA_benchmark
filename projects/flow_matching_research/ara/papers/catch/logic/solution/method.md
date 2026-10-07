# CATCH: Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching

在频域进行通道关联建模和频率补丁重建，利用频域异常证据检测多变量变化。

原文导航：

- A01: PDF 第 1 页，检索标记 `frequency`；curated synopsis; marker verified in extracted text; not a full-page manual review。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
