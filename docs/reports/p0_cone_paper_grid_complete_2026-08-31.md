# ConE-AFC Paper-Exact 全网格复现报告（2026-08-31）

## 结论

ConE-AFC 的作者代码论文网格已经完整运行：10 次重复、每次 5 折、5 个基分类器、3 个 alpha、3 个每类校准样本量和全部 51 个前缀。250 个 `split × model` 原子任务全部完成，组装为 50 个完整 split；Table 1–2 的 95 个命名均值全部落在预先冻结的绝对误差 `0.02` 内。

这是 Code Ocean 原始 synthetic alarm-flood 数据上的正式作者代码结果，不是 synthetic smoke。当前证据等级为 P2：数值网格闭合，但尚未在归档 Docker 镜像中重跑，也未完成本仓库独立实现的同折对照，因此不标记为 P3。

## 协议闭合

| 项目 | 实际执行 |
|---|---:|
| 原始样本 | 18,750 |
| 类别 | 5 类，每类 3,750 |
| 重复分层折 | 10 × 5 = 50 |
| 基分类器 | WDI-1NN、MBW-LR、EAC-1NN、ACM-SVM、CASIM |
| alpha | 0.01、0.05、0.10 |
| 每类校准量 | 22、102、2,491 |
| 前缀 | 10–60 min，步长 1 min |
| 原子任务 | 250/250 |
| 论文表格均值 | 95/95 通过 |

## 原始分类准确率

| 模型 | 论文 mean ± std | 本地作者代码 mean ± std | mean 差值 |
|---|---:|---:|---:|
| WDI-1NN | 0.313 ± 0.019 | 0.312874 ± 0.018731 | -0.000126 |
| MBW-LR | 0.665 ± 0.001 | 0.665364 ± 0.000696 | +0.000364 |
| EAC-1NN | 0.684 ± 0.001 | 0.684437 ± 0.001124 | +0.000437 |
| ACM-SVM | 0.629 ± 0.001 | 0.628891 ± 0.001277 | -0.000109 |
| CASIM | 0.706 ± 0.002 | 0.706494 ± 0.001595 | +0.000494 |

在全部 95 项中，最大的均值绝对差为 `0.002271`，出现在 CASIM、`alpha=0.10`、每类校准量 102 的 average set size：论文 `1.599`，复跑 `1.601271`。各模型最大均值绝对差分别为 WDI `0.000419`、MBW `0.000650`、EAC `0.000488`、ACM `0.001146`、CASIM `0.002271`。最大的标准差绝对差为 `0.002431`。

这也推翻了单个 split 阶段对 WDI set size 偏差的过早担忧：50 折聚合后，全部 WDI 命名均值均闭合。

## 可审计产物

- 原子 checkpoint：`experiments/paper_harness/p0_paper_exact/run_2/paper_grid/model_split_results.jsonl`
  - 250 行、250 个唯一任务；SHA-256 `db63065f2b8703737782c891a04911e6ff334ac6de9b99e7990fe941f5509a6d`
- 完整 split：`experiments/paper_harness/p0_paper_exact/run_2/paper_grid/seed_results.jsonl`
  - 50 行、50 个唯一 split；SHA-256 `850beca912aa28c5550d396d9c2fa130b1aa9a0f7fb61262d18e6022988ef135`
- 汇总与 95 行逐项比较：`experiments/paper_harness/p0_paper_exact/run_2/paper_grid/summary.json`
- 运行配置：`configs/experiments/p0_paper_exact.json`
- 论文协议卡：`paper_harness/paper_exact/faulwasser2024_cone_afc.v1.json`

checkpoint 使用字节级 UTF-8/LF 原子替换；记录哈希与文件实际哈希一致。随机种子固定为 42，未筛选 seed，未修改测试集，未在结果出现后调参。

## 复跑与恢复

完整或幂等复跑命令：

```powershell
& .\tmp\codeocean_envs\cone\Scripts\python.exe `
  experiments\paper_harness\p0_paper_exact\paper_grid.py `
  run-cone --workers 8
```

runner 会按 `split × model` 任务键加载 checkpoint；已有 250 项时不会重新训练，只会重新校验并生成 split 汇总、进度文件和哈希。

## 仍未闭合的必要工作

1. 在 Code Ocean 归档 Docker 镜像中运行同一完整网格，满足项目预先冻结的 P3 环境门槛。
2. 在相同 50 个 fold、相同前缀、相同超参数上运行本仓库独立实现，将剩余误差明确归因到实现而非数据或协议。
3. 保留现有 TEP/NPP/FCC transfer 结果作为跨域迁移轨；不得用本次原域结果替换迁移结果，也不得把迁移结果描述为论文复现。
