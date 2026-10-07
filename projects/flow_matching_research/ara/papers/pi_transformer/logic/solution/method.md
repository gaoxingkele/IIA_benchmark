# Pi-transformer: A prior-informed dual-attention model for multivariate time-series anomaly detection

相位与尺度信息形成先验注意力；结合关联差异加权重建能量、稳健归一化及多窗口分数融合。

原文导航：

- A01: PDF 第 1 页，检索标记 `point`；curated synopsis; marker verified in extracted text; not a full-page manual review。
- A02: PDF 第 8 页，检索标记 `point`；selected passage reviewed; tabular rows marked visual separately。
- A03: PDF 第 9 页，检索标记 `Table 2`；selected passage reviewed; tabular rows marked visual separately。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
