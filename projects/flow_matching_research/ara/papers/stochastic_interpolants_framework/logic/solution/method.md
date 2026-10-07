# Stochastic Interpolants: A Unifying Framework for Flows and Diffusions

用含噪随机插值统一概率流和扩散过程，关联速度、score 及相应训练目标。

原文导航：

- A01: PDF 第 1 页，检索标记 `diffusions`；curated synopsis; marker verified in extracted text; not a full-page manual review。
- A02: PDF 第 51 页，检索标记 `Oxford`；selected original dataset/experiment passage reviewed。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
