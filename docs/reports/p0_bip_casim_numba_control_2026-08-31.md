# BiP CASIM Numba RNG 受控切分验证（2026-08-31）

## 结论

CASIM 的 Numba 随机性混杂已经闭合。在真实 Code Ocean TEP 与 synthetic 数据、相同五折索引和相同作者模型参数下，同时重置 Python-level NumPy 与 Numba RNG 后，overlap/disjoint 两条通道的 **10/10 test bifurcation 完全一致，最大绝对差由 32 降为 0**。

这使 RF 训练子集效应可以在当前 `n_estimators=1`、`n_jobs_multirocket=1` 条件下作受控归因。结果不支持“删除 calibration/RF 重叠会统一改善性能”：TEP 的 point MAE 和区间宽度小幅改善，synthetic 的两项均变差；两者 coverage MAE 均仅小幅改善。

## 冻结协议

- 数据：官方 BiP-AFC v3 TEP `1,000 × 50 × 300` 与 synthetic `1,875 × 10 × 60`。
- 外层切分：`RepeatedStratifiedKFold(5, 1, random_state=42)`。
- 固定项：AFC train、calibration、test、10–60 min 前缀、30 个 alpha、作者 CASIM/BiP 实现和超参数。
- 唯一配对差异：RF train 使用 calibration 同一子集，或使用严格不相交子集。
- 新增控制：作者模型 import 完成后、每次 fit 前重置 `np.random.seed(42)`，并通过 Numba JIT seeder 重置 Numba 进程内 RNG；不修改 Capsule 源码。

真实数据参数探针显示：只重置全局 NumPy 时 MultiRocket bias 不一致；同时重置 Numba RNG 后，全部参数数组逐字节一致。

## 结果

| 数据 | test bifurcation overlap/disjoint | point MAE 变化 | coverage MAE 变化 | 区间宽度变化 |
|---|---:|---:|---:|---:|
| TEP | 259.6 / 259.6 | -0.017131 | -0.000282 | -0.044184 |
| synthetic | 1034.8 / 1034.8 | +0.031906 | -0.001689 | +0.255506 |

变化均为 `disjoint - overlap`。负值表示对应误差或宽度下降。

CASIM 对应的四个论文目标中，两条新通道均为 2/4 通过；显式播种解决的是实验可归因性，而不是论文数值差距。

## 审计产物

- 180 行总 checkpoint：`experiments/paper_harness/p0_paper_exact/run_3/paper_grid/fold_results.jsonl`
- 汇总与 paired audit：`experiments/paper_harness/p0_paper_exact/run_3/paper_grid/summary.json`
- 图：`experiments/paper_harness/p0_paper_exact/Figure_4.png`
- 配置：`configs/experiments/p0_bip_paper_grid.json`
- checkpoint SHA-256：`46d054efd950cfa77dd749f688ba3858b04cd27e636252f5e0b829772f462ffd`

## 仍未闭合

1. 本机缺少 Docker CLI，尚不能在归档 Code Ocean 镜像中核对 Linux 文件顺序、数据加载次序和依赖版本。
2. 本仓库独立 CASIM/ConE/BiP 实现尚未在完全相同的五折上完成逐项对照。
3. 官方 v3 作者通道仍只有 6/16 论文条目通过冻结门槛；RNG 控制不能解释剩余数值差距。
