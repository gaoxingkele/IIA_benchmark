# SensitiveHUE: Multivariate Time Series Anomaly Detection by Enhancing the Sensitivity to Normal Patterns

移除统计捷径增强正常依赖敏感性，利用异方差预测的不确定性与负对数似然评分。

原文导航：

- A01: PDF 第 1 页，检索标记 `heteroscedastic`；curated synopsis; marker verified in extracted text; not a full-page manual review。
- A02: PDF 第 6 页，检索标记 `WADI`；selected original dataset/experiment passage reviewed。
- A03: PDF 第 7 页，检索标记 `Evaluation Metrics`；selected F1*/F1*_PA definitions reviewed。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
