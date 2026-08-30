# ConE-AFC 原子断点验证（2026-08-30）

## 目的与结果

原 full-grid runner 以“一个 split 的五个模型全部完成”为最小落盘单位。实测六个并行 split 在约 27 分钟内都未形成可恢复结果，任何中断都会丢失已完成的较快模型。现已把网格等价拆分为 **50 splits × 5 models = 250 个 `(split, model)` 原子任务**。

拆分只改变实验编排，不改变以下论文条件：

- `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)`；
- 作者 `create_cali_dataset` 生成的训练/校准/测试分区；
- 五个作者模型实现及超参数；
- 10–60 分钟的 51 个前缀；
- `alpha={0.01,0.05,0.10}` × `ncal/class={22,102,2491}`；
- accuracy、coverage、average set size，以及补充的 singleton/empty rate。

## 首个正式任务

`split=0|model=WDI_1NN` 已完整运行并落盘，当前进度 1/250。该单折的原分类 accuracy 为 0.305825，论文 50-fold 均值为 0.313；coverage 在九个 conformal 条件下均与论文均值非常接近。单折 average set size 的偏差不能作为最终复现结论，需等 50 folds 聚合。

对同一命令再次运行时，runner 在 0.27 秒内返回，checkpoint 行数仍为 1，SHA-256 未变化，证明它会跳过已完成的 `(split, model)` 而不重训。

## 产物与恢复

- 原子任务：`experiments/paper_harness/p0_paper_exact/run_2/paper_grid/model_split_results.jsonl`
- 兼容的 split 汇总：`experiments/paper_harness/p0_paper_exact/run_2/paper_grid/seed_results.jsonl`
- 当前摘要：`experiments/paper_harness/p0_paper_exact/run_2/paper_grid/summary.json`
- 原子 checkpoint SHA-256：`1a844d6ce59d2565bc795c690140e16b3a16c710daf53f4bef6d539e8570cb20`
- 恢复命令：`paper_grid.py run-cone --workers 8`

所有 checkpoint 使用临时文件加 `os.replace`，并以 UTF-8 bytes 写入，保证 Windows 上记录的 SHA-256 与实际磁盘文件一致。
