# Chapter 4 — Root-cause analysis

## 数学与方法

对二值报警，Normalized Transfer Entropy（NTE）和其 direct 版本在多个 lag 上取最大因果强度；独立 Bernoulli surrogate Monte Carlo 给显著性阈值，间接路径需要剪枝。书中特别提示样本量：报警出现过少时 TE 不可靠，案例采用至少约 50 次出现的筛选。

对非平稳连续变量，IGTE/IGDTE 先将变量抽象成 information granules，再用 OPTICS 聚类与 PDF 估计构造二阶传递熵。在线根因识别还可用 Bayesian network：递归更新条件概率、显式放置 “unknown cause”，更新率可写为 `λ = 1 - 0.5^(1/m)`。

另一条可解释路线把关键过程变量做 piecewise-linear representation，计算滞后相关和 qualitative trend，再用 non-negative multiple linear regression 分配贡献因子。

## 算法抽取

`transfer_entropy` 和 `TransferEntropyRanker` 已实现离散一阶估计、lag 搜索和 permutation threshold。正式 RCA 必须报告 discretization、最大 lag、surrogate 次数、显著性及 occurrence count。IGTE、直接路径剪枝和在线 BN 是 B 级扩展位。

## 数据与测试

只有 TEP/仿真或受监督装置拥有可信根因标签；PIADE 的机器状态不能冒充根因。指标：edge precision/recall、Top-k、MRR、lag error、online detection delay。

证据：`04_root_cause.txt`，PDF physical pages 231–311。
