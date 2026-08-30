# BiP-AFC Tables 1–4 与切分控制复现报告（2026-08-31）

## 结论

BiP-AFC / Predicting Uncertainty Reduction 的原始域计算已完成 160 个可恢复任务：2 个数据集 × 4 个分类器 × 5 折 × 4 条协议 lane。每折均覆盖 10–60 min 的全部前缀和 jackknife+ `alpha=0.01–0.30`。

计算完整不等于数值闭合。官方 v3 代码原样保留 calibration/RF 重叠时，16 个论文条目仅 6 个通过冻结的联合门槛；改为论文文本要求的无重叠切分后为 7/16。增加全局 NumPy seed=42 后，overlap/disjoint 分别为 7/16 和 8/16。结果证明索引重叠能解释部分差距，但不能解释全部论文差距。

当前证据等级为 P2，不标记 P3：仍缺归档 Docker 内的文件顺序/数据版本复核，CASIM 的 Numba RNG 未被模型 `random_state` 控制，且本仓库独立实现尚未在完全相同的折上运行。

## 数据与协议先验

| 数据集 | 形状 | 类别分布 | 每类 AFC / calibration / RF / test |
|---|---:|---:|---:|
| TEP | 1,000 × 50 × 300 | 5 × 200 | 60 / 50 / 50 / 40 |
| synthetic | 1,875 × 10 × 60 | 5 × 375 | 100 / 100 / 100 / 75 |

缓存与作者 CSV 逐值比较通过：TEP 的 15,000,000 个二值元素、synthetic 数据及全部标签保持一致；缓存仅把 `int64` 无损转为 `uint8`，原始文件未被覆盖。source manifest SHA-256 为 `f11f64d6615ad08cc5c5a300275c194d5a21097115fc416acc9ffe3701dd9a8d`。

四条 lane 为：

1. `author_overlap`：保留 v3 `main.py`，calibration 与 RF 使用相同的每类前 50/100 条。
2. `paper_disjoint`：calibration 在前、AFC train 居中、RF 在后，三者严格无交叉。
3. `seeded_author_overlap`：在 lane 1 基础上，每次作者模型 fit 前设置 NumPy seed=42。
4. `seeded_paper_disjoint`：在 lane 2 基础上设置同一全局 seed。

40 对任务的 AFC train、calibration 和 test 索引哈希完全相同。作者 lane 的 calibration/RF 交叉量为 TEP 250、synthetic 500；disjoint lane 均为 0。

## Tables 1 与 3：bifurcation 数

表中单元为 `train / test` 五折均值；Pass 同时要求两项相对误差不超过 5%。

| 数据集 | 模型 | 论文 | v3 overlap | Pass | paper disjoint | Pass |
|---|---|---|---|---:|---|---:|
| synthetic | ACM-SVM | 1853.4 / 1395.6 | 1898.2 / 1420.6 | 是 | 1903.4 / 1421.2 | 是 |
| synthetic | EAC-1NN | 1094.2 / 814.0 | 1126.6 / 849.0 | 是 | 1135.8 / 849.0 | 是 |
| synthetic | MBW-LR | 1747.6 / 1311.2 | 1579.6 / 1184.2 | 否 | 1580.8 / 1184.2 | 否 |
| synthetic | CASIM | 1425.0 / 1065.6 | 1362.0 / 1027.8 | 是 | 1331.2 / 1003.4 | 否 |
| TEP | ACM-SVM | 603.6 / 471.8 | 595.4 / 476.0 | 是 | 594.8 / 475.8 | 是 |
| TEP | EAC-1NN | 227.0 / 178.0 | 233.6 / 193.6 | 否 | 248.2 / 193.6 | 否 |
| TEP | MBW-LR | 460.6 / 367.4 | 625.0 / 505.8 | 否 | 633.0 / 505.8 | 否 |
| TEP | CASIM | 299.8 / 233.4 | 283.8 / 226.6 | 否 | 294.2 / 227.0 | 是 |

## Tables 2 与 4：coverage MAE / point MAE

Pass 同时要求 coverage MAE 绝对差不超过 0.01、point MAE 相对差不超过 5%。

| 数据集 | 模型 | 论文 | v3 overlap | Pass | paper disjoint | Pass |
|---|---|---|---|---:|---|---:|
| synthetic | ACM-SVM | 0.0145 / 1.2930 | 0.00966 / 1.25248 | 是 | 0.01011 / 1.26592 | 是 |
| synthetic | EAC-1NN | 0.0156 / 0.5240 | 0.01128 / 0.43484 | 否 | 0.01324 / 0.44545 | 否 |
| synthetic | MBW-LR | 0.0126 / 0.6475 | 0.01239 / 0.67288 | 是 | 0.01174 / 0.67316 | 是 |
| synthetic | CASIM | 0.0111 / 0.7127 | 0.00906 / 0.61584 | 否 | 0.01325 / 0.61873 | 否 |
| TEP | ACM-SVM | 0.1004 / 3.5827 | 0.09574 / 3.18081 | 否 | 0.10366 / 3.17512 | 否 |
| TEP | EAC-1NN | 0.2142 / 4.0613 | 0.18861 / 4.28709 | 否 | 0.20922 / 4.43904 | 否 |
| TEP | MBW-LR | 0.0228 / 2.0768 | 0.02368 / 2.21907 | 否 | 0.02682 / 2.15878 | 是 |
| TEP | CASIM | 0.0104 / 2.3272 | 0.02121 / 2.26490 | 否 | 0.02355 / 2.22772 | 否 |

## 受控切分消融

设置 NumPy seed 后，MBW、EAC、ACM 的 30/30 配对任务具有完全相同的 test bifurcation，因而可把变化归因于 RF 训练子集。主要变化如下：

| 数据集 | 模型 | point MAE overlap→disjoint | coverage MAE 变化 | 平均区间宽度变化 |
|---|---|---:|---:|---:|
| TEP | MBW-LR | 2.21907 → 2.15878 | +0.00036 | -0.43147 |
| TEP | EAC-1NN | 4.28709 → 4.43904 | +0.02356 | -0.33702 |
| TEP | ACM-SVM | 3.18543 → 3.18373 | +0.00607 | -0.50549 |
| synthetic | MBW-LR | 0.67288 → 0.67316 | -0.00121 | -0.00172 |
| synthetic | EAC-1NN | 0.43484 → 0.44545 | +0.00165 | -0.08195 |
| synthetic | ACM-SVM | 1.26269 → 1.26049 | +0.00116 | +0.03877 |

因此，无重叠修正使 TEP MBW 的 point MAE 和区间宽度改善，并使 Table 4 MBW 进入容差，但会使 TEP EAC 的 point/coverage MAE 变差；不存在统一收益。

CASIM 的 10/10 受控配对仍出现 test bifurcation 差异，最大单折差为 32。根因证据位于 vendored `CASIM_multirocket.py`：`_fit_multi` 和 bias 采样调用 Numba `np.random`，而 `MultiRocketMultivariate` 构造器不接收 Arsenal 传入的 `random_state`。因此 CASIM 的 overlap/disjoint 数值只能作描述性比较，不能作纯切分因果估计。

## 可审计产物

- 160 行原子结果：`experiments/paper_harness/p0_paper_exact/run_3/paper_grid/fold_results.jsonl`
- 汇总、论文逐项比较、两组消融和配对审计：`experiments/paper_harness/p0_paper_exact/run_3/paper_grid/summary.json`
- 进度与 checkpoint 哈希：`experiments/paper_harness/p0_paper_exact/run_3/paper_grid/progress.json`
- 配置源：`configs/experiments/p0_bip_paper_grid.json`
- 协议卡：`paper_harness/paper_exact/faulwasser2025_uncertainty_reduction.v1.json`
- checkpoint SHA-256：`427ebb056602e105f998c8f11948caea5eceb070f1172e408fcfd38a4ca32b7f`

完整幂等恢复耗时 0.535 秒，未重新训练。所有 alpha 曲线、interval width、每折误差标准差和分区索引哈希均保存在上述 JSON/JSONL 中。

## 复跑命令

```powershell
& .\tmp\codeocean_envs\bip\Scripts\python.exe `
  experiments\paper_harness\p0_paper_exact\bip_grid.py `
  run-bip --workers 8 `
  --lanes author_overlap paper_disjoint seeded_author_overlap seeded_paper_disjoint
```

## 仍需完成及必要性

1. 在归档 Docker 镜像中复跑，核对作者 `os.listdir` 未排序造成的 Windows/Linux 文件顺序差异；该差异会改变外层 fold 和每类内部抽样。
2. 为 CASIM 增加不修改作者模型逻辑的 Numba RNG 显式播种对照，否则无法把 RF 子集效应和特征采样随机性分开。
3. 在相同 5 折索引、相同 30 个 alpha 和相同目标上运行本仓库独立实现，分离作者代码版本差距与实现差距。
4. 向作者核实论文生成 Tables 1–4 时使用的 commit、文件顺序、Mapie 版本和统计汇总公式；当前 v3 Capsule 自身不能完整重现论文表格。
