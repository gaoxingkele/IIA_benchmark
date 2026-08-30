# CASIM Paper-Exact 开放集迭代（2026-08-30）

## 结论

checkpoint 中剩余的 22 个 held-out-class/fold 任务已经完成，因而随机种子 42 的 70 个开放集组合完整。论文第 4.2 节同时规定：为控制 MultiRocket 的非确定性，每个测试需运行 10 个随机实例并报告均值。因此当前进度是 **70/700 model fits（1/10 随机实例）**，不是完整 P3 复现。

## 当前结果与论文 Figure 13c

| 指标 | 论文 | 本地 seed 42 | 差值 | 判定 |
|---|---:|---:|---:|---|
| 最大 balanced accuracy | 0.947 | 0.955905 | +0.008905 | 数值在 ±0.02 容差内 |
| 最佳 novelty threshold | 0.324 | 0.425 | +0.101 | 未闭合 |
| 全阈值 mean balanced accuracy | 0.879 | 0.907038 | +0.028038 | 超出 ±0.02 容差 |

峰值性能接近，但最佳阈值和整条曲线尚未接近论文。因此不能只依据最大 balanced accuracy 宣称复现成功。

## 协议与数据审计

- 使用官方 CASIM v1 Capsule 的 310 条 TEP 报警子序列、76 个报警变量和作者模型代码。
- 每个已知类依次整体重标为 `-1`，复用作者的 `get_train_test(..., open_set=True)` 五折划分；70 个 `(held_out_class, fold)` 组合唯一且无失败。
- 固定参数为 672 个 MultiRocket 特征、10 个 ridge estimators、10 个 LoOP 邻居、`lambda=3`，本实例种子为 42。
- 按论文 O2.4，仅 `p_out < tau` 接受已知类；`p_out == tau` 与 `p_out > tau` 均拒绝为未知类。
- Capsule 未提供论文的 14 类外层循环，当前 relabel-before-split wrapper 仍需作者确认。
- 当前机器没有 Docker；Windows/Python 3.9.25 精确依赖兼容运行不满足最终环境 P3 门禁。

## 可恢复产物

- 逐任务结果：`experiments/paper_harness/p0_paper_exact/run_1/paper_grid/repetitions_1/seed_results.jsonl`
- 阈值曲线摘要：`experiments/paper_harness/p0_paper_exact/run_1/paper_grid/repetitions_1/summary.json`
- 结果 SHA-256：`ca8d7fbd51fdfe835c20887ab488aa385797b8a9448035f214558bdd6140f3b8`
- 下一阶段命令：`paper_grid.py run-casim --workers 8 --repetitions 10`。runner 会引导已完成的前 70 项，只计算任务 70-699，并在每个任务后原子化重写有序 checkpoint。

## 剩余门禁

1. 完成随机实例 2-10，汇总 10 条曲线的均值、IQR 与全范围（Figure 14a）。
2. 对 Figure 13c 的峰值、最佳阈值和全阈值均值执行最终容差判断。
3. 获得作者对 held-out-class wrapper 的确认，或取得遗漏的原始实验脚本。
4. 在归档 Docker image 中复跑。
