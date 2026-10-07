# Simulation-free Schrödinger bridges via score and flow matching

联合估计 score 与概率流速度，以无仿真目标学习 Schrödinger 桥随机过程。

原文导航：

- A01: PDF 第 1 页，检索标记 `score`；curated synopsis; marker verified in extracted text; not a full-page manual review。
- A02: PDF 第 8 页，检索标记 `Cite`；selected original dataset/experiment passage reviewed。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
