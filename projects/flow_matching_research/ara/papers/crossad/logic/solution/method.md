# CrossAD: Time Series Anomaly Detection with Cross-scale Associations and Cross-window Modeling

跨尺度关联与跨窗口建模联合学习异常表征，改善局部模式与长期依赖的检测。

原文导航：

- A01: PDF 第 1 页，检索标记 `cross-window`；curated synopsis; marker verified in extracted text; not a full-page manual review。
- A02: PDF 第 7 页，检索标记 `GECCO`；selected original dataset/experiment passage reviewed。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
