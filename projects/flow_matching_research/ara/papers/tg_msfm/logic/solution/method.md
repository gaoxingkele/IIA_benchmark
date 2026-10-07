# Time-Gated Multi-Scale Flow Matching for Time-Series Imputation

可见性掩码 Transformer 条件化速度，多尺度金字塔头通过时间门融合；Heun 积分每步投影保证观测一致性。

原文导航：

- A01: PDF 第 1 页，检索标记 `Heun`；curated synopsis; marker verified in extracted text; not a full-page manual review。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
