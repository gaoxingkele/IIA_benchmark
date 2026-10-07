# Building Normalizing Flows with Stochastic Interpolants

从随机端点插值构造中间密度，以二次回归目标学习其速度场并得到归一化流。

原文导航：

- A01: PDF 第 1 页，检索标记 `interpolant`；curated synopsis; marker verified in extracted text; not a full-page manual review。
- A02: PDF 第 8 页，检索标记 `HEPMASS`；selected original dataset/experiment passage reviewed。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
