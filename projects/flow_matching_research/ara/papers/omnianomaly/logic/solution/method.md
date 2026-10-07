# Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network

随机循环潜状态结合 planar normalizing flows 及重建概率刻画多变量正常行为；论文引入 SMD。

原文导航：

- A01: PDF 第 1 页，检索标记 `stochastic`；curated synopsis; marker verified in extracted text; not a full-page manual review。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
