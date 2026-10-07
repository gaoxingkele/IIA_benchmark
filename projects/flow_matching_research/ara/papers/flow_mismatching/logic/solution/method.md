# Flow Mismatching: Unsupervised Anomaly Detection via Velocity Discrepancies in Flow Matching Models

在 Gaussian 到测试图像的仿射路径上比较正常生成速度与几何目标速度，跨路径/时间聚合失配形成热图和图像分数。

原文导航：

- A01: PDF 第 1 页，检索标记 `affine`；curated synopsis; marker verified in extracted text; not a full-page manual review。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
