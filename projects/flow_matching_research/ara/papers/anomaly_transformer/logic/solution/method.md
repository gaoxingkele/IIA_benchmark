# Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy

先验/序列双关联注意力及 minimax 学习构建关联差异，以关联差异加权重建误差评分。

原文导航：

- A01: PDF 第 1 页，检索标记 `discrepancy`；curated synopsis; marker verified in extracted text; not a full-page manual review。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
