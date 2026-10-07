# Towards Long-Context Time Series Foundation Models With A Handful Of Additional Parameters

Infini-ChannelMixer 引入可压缩记忆，以少量额外参数增强 MOMENT 的长上下文及变量交互。

原文导航：

- A01: PDF 第 1 页，检索标记 `Infini`；curated synopsis; marker verified in extracted text; not a full-page manual review。
- A02: PDF 第 4 页，检索标记 `Table 3`；selected passage reviewed; tabular rows marked visual separately。

工程分解：数据/掩码条件 → 方法族专属路径或表示 → 损失训练 → 原任务输出 → 单独冻结的异常分数与阈值。不得将原任务误差直接换成异常检测指标。
