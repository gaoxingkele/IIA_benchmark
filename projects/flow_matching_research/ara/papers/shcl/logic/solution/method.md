# Subsequence heterogeneity contrastive learning for time series anomaly detection

固定间隔和时间层次掩码生成异质子序列，以对比目标增强任意时序骨干的正常模式学习。

原文导航：

- A01: PDF 第 1 页，检索标记 `masking`；curated synopsis; marker verified in extracted text; not a full-page manual review。
- A02: PDF 第 8 页，检索标记 `validation`；selected passage reviewed; tabular rows marked visual separately。
- A03: PDF 第 9 页，检索标记 `Table 2`；selected passage reviewed; tabular rows marked visual separately。
- A04: PDF 第 10 页，检索标记 `Table 4`；selected passage reviewed; tabular rows marked visual separately。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
