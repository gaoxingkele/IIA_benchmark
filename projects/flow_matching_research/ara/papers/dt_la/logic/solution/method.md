# Dual Transformers With Latent Amplification for Multivariate Time Series Anomaly Detection

双 Transformer 的潜变量放大增强正常与异常的表示差异，结合异常评分进行多变量检测。

原文导航：

- A01: PDF 第 1 页，检索标记 `amplification`；curated synopsis; marker verified in extracted text; not a full-page manual review。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
