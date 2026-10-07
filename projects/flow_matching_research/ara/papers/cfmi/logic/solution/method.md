# CFMI: Flow Matching for Missing Data Imputation

共享条件向量场在不同缺失掩码下学习条件分布；以条件流匹配提供概率插补，包含表格任务与零样本时序评估。

原文导航：

- A01: PDF 第 1 页，检索标记 `conditional`；curated synopsis; marker verified in extracted text; not a full-page manual review。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
