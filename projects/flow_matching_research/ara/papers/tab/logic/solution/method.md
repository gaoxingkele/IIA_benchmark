# TAB: Unified Benchmarking of Time Series Anomaly Detection Methods

系统评估时间序列异常检测方法并提供可复用的数据与评价管线。

原文导航：

- A01: PDF 第 1 页，检索标记 `benchmark`；curated synopsis; marker verified in extracted text; not a full-page manual review。
- A02: PDF 第 6 页，检索标记 `Statistics of multivariate datasets`；Table2/3 scope and subset selection passage reviewed。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
