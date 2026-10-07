# Graph-Spectral Flow Matching for Multivariate Time Series Anomaly Detection

通过图 Dirichlet 能量与动能构造固定端点输运路径，在图频率上求解插值；以正常训练分布的速度失配构造异常分数。

原文导航：

- A01: PDF 第 1 页，检索标记 `Dirichlet`；curated synopsis; marker verified in extracted text; not a full-page manual review。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
