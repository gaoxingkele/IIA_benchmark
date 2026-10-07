# Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models

以结构化状态空间模型 S4 建模长时依赖，扩散去噪器利用观测条件进行插补和预测。

原文导航：

- A01: PDF 第 1 页，检索标记 `state space`；curated synopsis; marker verified in extracted text; not a full-page manual review。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
