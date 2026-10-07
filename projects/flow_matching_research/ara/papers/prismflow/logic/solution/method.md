# PrismFlow: Residual Dynamics for Flow Matching in Time-Series Generation

全局速度场之外加入 Koopman 启发的潜空间残差专家，以置信度 Winner-Take-All 更新减少多模态速度平均化。

原文导航：

- A01: PDF 第 1 页，检索标记 `Winner`；curated synopsis; marker verified in extracted text; not a full-page manual review。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
