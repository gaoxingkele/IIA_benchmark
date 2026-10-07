# Flow Matching Guide and Code

统一介绍连续流、概率路径、条件速度回归及离散流等对象，并提供可执行示例与算法接口。

原文导航：

- A01: PDF 第 1 页，检索标记 `flow matching`；curated synopsis; marker verified in extracted text; not a full-page manual review。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
