# DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection

多尺度补丁的双注意力产生不同关联视角，采用对比学习直接构造异常关联差异。

原文导航：

- A01: PDF 第 1 页，检索标记 `contrastive`；curated synopsis; marker verified in extracted text; not a full-page manual review。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
