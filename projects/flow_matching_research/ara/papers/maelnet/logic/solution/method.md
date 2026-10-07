# MaelNet: Memorable Anomaly Election Learning Detecting Anomalies in Time Series Data using Dual-Net Transformer

快学习器使用 Anomaly Transformer 与 DCDetector，慢学习器使用掩码非平稳 Transformer，强化学习选择异常候选。

原文导航：

- A01: PDF 第 4 页，检索标记 `reward`；curated synopsis; marker verified in extracted text; not a full-page manual review。
- A02: PDF 第 3 页，检索标记 `threshold`；selected passage reviewed; tabular rows marked visual separately。
- A03: PDF 第 4 页，检索标记 `reward`；selected passage reviewed; tabular rows marked visual separately。
- A04: PDF 第 4 页，检索标记 `Table II`；selected passage reviewed; tabular rows marked visual separately。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
