# TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis

以 FFT 找主要周期，将一维时序重排为多个二维结构，用 Inception 卷积刻画周期内/周期间变化。

原文导航：

- A01: PDF 第 1 页，检索标记 `Inception`；curated synopsis; marker verified in extracted text; not a full-page manual review。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
