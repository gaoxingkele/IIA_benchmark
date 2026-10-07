# Sequence Modeling with Spectral Mean Flows

通过 HMM 的分布均值嵌入及谱张量分解建模序列，将时间依赖 Hilbert 空间的 MMD 梯度流关联到流匹配。

原文导航：

- A01: PDF 第 1 页，检索标记 `Hilbert`；curated synopsis; marker verified in extracted text; not a full-page manual review。
- A02: PDF 第 9 页，检索标记 `FRED`；selected original dataset/experiment passage reviewed。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
