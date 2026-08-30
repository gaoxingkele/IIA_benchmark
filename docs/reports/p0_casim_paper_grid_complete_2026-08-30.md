# CASIM 开放集完整计算网格报告（2026-08-30）

## 结论

CASIM 的论文开放集计算网格已经完成：14 个 held-out 类 × 5 folds × 10 个随机 MultiRocket 实例，共 **700/700 model fits**，seeds 为 42–51，失败任务为 0。该状态表示论文计算覆盖完成，不表示 P3 已闭合。

## Figure 13c 对照

| 指标 | 论文 | 本地 10 实例均值 | 差值 | 容差 | 结果 |
|---|---:|---:|---:|---:|---|
| 最大 balanced accuracy | 0.947 | 0.955520 | +0.008520 | ±0.02 | Pass |
| 最佳 novelty threshold | 0.324 | 0.335 | +0.011 | ±0.0005（论文舍入） | Fail |
| 全阈值 mean balanced accuracy | 0.879 | 0.903267 | +0.024267 | ±0.02 | Fail |

最大性能已经接近论文，十实例平均后最佳阈值也由单 seed 的 0.425 移到 0.335，但完整曲线仍系统性高于论文，导致全阈值均值超差。当前证据更像外层 split/未知样本分配或指标聚合协议仍有差异，而不是单纯随机性。

## Figure 14a 随机性包络

- 十实例曲线的平均全范围宽度为 0.027033，最大宽度为 0.088970（tau=0.957）。
- 十实例曲线的平均 IQR 宽度为 0.009853，最大 IQR 宽度为 0.036494（tau=0.976）。
- 在聚合峰值 tau=0.335 处，十实例 BA 范围为 0.947809–0.962849，宽度 0.015040；IQR 宽度为 0.008428。
- 各 seed 自身最佳阈值范围较大（0.167–0.472），但在聚合峰值附近的性能包络较窄。这支持“性能对随机实例较稳健”，但不支持“最佳阈值稳定”。

完整 threshold-wise mean、IQR、minimum 和 maximum 数组已存入 `repetitions_10/summary.json`，可直接重绘 Figure 14a。

## 数据与协议

- 数据：官方 v1 Capsule 的 310 条 TEP 报警子序列、76 个报警变量；数据 manifest SHA-256 为 `77643dfb40749472f5dfb752611b1db3c462822453e20d6e4669e9cb14f14dd2`。
- 模型：作者 CASIM 代码；672 features、10 estimators、LoOP k=10、lambda=3。
- 切分：每个已知类整体重标为 `-1`，复用作者 `get_train_test(..., open_set=True)` 的五折划分。
- 判定：严格执行 O2.4，只有 `p_out < tau` 接受已知类，等于阈值时拒绝为未知。
- 结果文件 SHA-256：`1b39242e07ab79af60729a0c4a5e09f785018dba36b287c0827069d76b2a4e18`。

## 仍未闭合的必要实验

1. 向作者确认 Capsule 遗漏的 14 类外层循环，重点确认 novel samples 是每 fold 分配一次还是在五个 test split 中重复出现，以及平均顺序。
2. 重建 Figure 14b 的 `nclf={1,5,10,25}` 和 Figure 14c 的 `nfeat={672,10000,50000}` 消融；这些不属于当前默认 700 项。
3. 在归档 Docker image 中执行；当前 Windows/Python 3.9.25 精确依赖兼容运行只满足 P2 环境证据。
4. 在获得作者外层脚本后重新判断全阈值均值 +0.024267 的来源；在此之前不得调阈值或选择 seed 来压低差距。

## 产物

- `experiments/paper_harness/p0_paper_exact/run_1/paper_grid/repetitions_10/seed_results.jsonl`
- `experiments/paper_harness/p0_paper_exact/run_1/paper_grid/repetitions_10/summary.json`
- `experiments/paper_harness/p0_paper_exact/Figure_3.png`
