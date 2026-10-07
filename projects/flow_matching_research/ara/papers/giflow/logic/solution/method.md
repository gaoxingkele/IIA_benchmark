# Spatiotemporal Imputation with Graph-Informed Flow Matching

图滤波形成观测条件先验，结合空间/时间注意力及图传播学习条件速度，积分补全缺失坐标。

原文导航：

- A01: PDF 第 1 页，检索标记 `prior`；curated synopsis; marker verified in extracted text; not a full-page manual review。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
