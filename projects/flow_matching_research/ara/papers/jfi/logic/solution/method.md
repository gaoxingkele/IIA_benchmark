# Flow matching-based online imputation of chemical processes under arbitrary missing scenarios for enhanced process monitoring

局部先验采样与混合卷积/通道注意力 UNet 学习插补速度，联合速度、目标重建和下游知识蒸馏损失。

原文导航：

- A01: PDF 第 1 页，检索标记 `local`；curated synopsis; marker verified in extracted text; not a full-page manual review。
- A02: PDF 第 7 页，检索标记 `13385936`；selected passage reviewed; tabular rows marked visual separately。
- A03: PDF 第 7 页，检索标记 `12341`；selected passage reviewed; tabular rows marked visual separately。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
