# Diffusion-TS: Interpretable Diffusion for General Time Series Generation

以趋势/季节性分解和重建驱动去噪构建可解释时序生成模型。

原文导航：

- A01: PDF 第 1 页，检索标记 `seasonal`；curated synopsis; marker verified in extracted text; not a full-page manual review。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
