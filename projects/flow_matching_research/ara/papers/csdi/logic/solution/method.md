# CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation

仅在缺失位置学习条件去噪扩散，观测值、时间及特征嵌入作为条件；多次采样形成插补分布。

原文导航：

- A01: PDF 第 1 页，检索标记 `conditional`；curated synopsis; marker verified in extracted text; not a full-page manual review。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
