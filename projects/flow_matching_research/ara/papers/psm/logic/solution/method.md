# Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization

谱潜空间自编码器同步异步信号，再以集成分位数自编码器投票检测和定位；同时公开 PSM 数据。

原文导航：

- A01: PDF 第 1 页，检索标记 `asynchronous`；curated synopsis; marker verified in extracted text; not a full-page manual review。
- A02: PDF 第 10 页，检索标记 `SMD`；selected original dataset/experiment passage reviewed。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
