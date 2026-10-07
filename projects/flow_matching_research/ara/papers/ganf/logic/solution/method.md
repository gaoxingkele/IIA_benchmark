# Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series

学习变量依赖图并以图条件化的归一化流估计时序密度，低似然作为异常证据。

原文导航：

- A01: PDF 第 1 页，检索标记 `graph`；curated synopsis; marker verified in extracted text; not a full-page manual review。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
