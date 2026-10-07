# KGL: An Efficient Time Series Anomaly Detection Approach for the Industrial Internet of Things

以 KAN 激活增强图注意力并用 LSTM 学习时序动态，融合变量依赖和时间表示进行异常检测。

原文导航：

- A01: PDF 第 1 页，检索标记 `LSTM`；curated synopsis; marker verified in extracted text; not a full-page manual review。
- A02: PDF 第 6 页，检索标记 `validation`；selected passage reviewed; tabular rows marked visual separately。
- A03: PDF 第 6 页，检索标记 `Table 2`；selected passage reviewed; tabular rows marked visual separately。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
