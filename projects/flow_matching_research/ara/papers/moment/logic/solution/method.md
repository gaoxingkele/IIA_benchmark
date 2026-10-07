# MOMENT: A Family of Open Time-series Foundation Models

预训练编码器以掩码重建学习通用时序表示，再用于预测、分类、插补及异常检测。

原文导航：

- A01: PDF 第 1 页，检索标记 `pre-training`；curated synopsis; marker verified in extracted text; not a full-page manual review。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
