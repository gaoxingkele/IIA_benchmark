# Conditional Flow Matching for Time Series Modelling

以 BB-Flows 的线性高斯边际插值或 GP-Flows 的状态/导数联合高斯后验形成轨迹条件速度目标。

原文导航：

- A01: PDF 第 2 页，检索标记 `Kalman`；curated synopsis; marker verified in extracted text; not a full-page manual review。
- A02: PDF 第 2 页，检索标记 `Kalman`；selected passage reviewed; tabular rows marked visual separately。
- A03: PDF 第 3 页，检索标记 `Gaussian`；selected passage reviewed; tabular rows marked visual separately。
- A04: PDF 第 4 页，检索标记 `Table 1`；selected passage reviewed; tabular rows marked visual separately。
- A05: PDF 第 8 页，检索标记 `256`；selected passage reviewed; tabular rows marked visual separately。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
