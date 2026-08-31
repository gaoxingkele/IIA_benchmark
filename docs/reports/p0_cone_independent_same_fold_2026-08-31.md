# ConE 独立 conformal 层同折验证（2026-08-31）

## 结论

本仓库独立 `ConEAFCCalibrator` 已在官方 18,750 条 synthetic alarm-flood 数据的完整 50 折上完成同折验证。使用与作者网格完全相同的 MBW-LR 基础分数、51 个时间前缀、3 个 alpha、3 个每类校准量时，coverage、average set size、singleton rate 和 empty rate 共 **1,800/1,800 个配对指标逐项完全一致，最大绝对差为 0**。

这闭合了独立 conformal 校准与集合生成子门槛，但不是完整 P3：基础 MBW-LR 分数仍由作者实现生成，其余四个基础分类器和端到端独立 wrapper 尚未完成；本机也没有 Docker CLI。

## 协议

- 数据：Code Ocean ConE-AFC v2，`18,750 × 10 × 60`，五类各 3,750 条。
- 切分：`RepeatedStratifiedKFold(5, 10, random_state=42)`，50/50 folds。
- 基础分类器：作者 MBW-LR，保持论文参数和逐前缀训练。
- 独立部分：`src/iia_benchmark/models/cone_afc.py:ConEAFCCalibrator`。
- 前缀：10–60 min，共 51 个。
- alpha：0.01、0.05、0.10。
- 每类校准量：22、102、2,491。
- 指标：class-balanced coverage、average set size、singleton rate、empty rate。
- 验收：与冻结作者网格逐折绝对差不超过 `1e-12`。

每折保存 outer-train/test 索引 SHA-256、分区大小、seed、36 个条件指标和逐项差值；未筛选 fold，未修改测试集，未按结果调参。

## 结果

| 项目 | 结果 |
|---|---:|
| 完成 folds | 50/50 |
| 配对指标 | 1,800 |
| 通过 `1e-12` | 1,800/1,800 |
| 最大绝对指标差 | 0.0 |
| 最大绝对原始 accuracy 差 | 0.0 |
| 各任务累计计算时间 | 10,097.756 s |

四种指标各自的平均差和最大绝对差均为 0。

## 产物

- 原子结果：`experiments/paper_harness/p0_paper_exact/run_2/independent_same_fold/mbw_lr/fold_results.jsonl`
- 汇总：`experiments/paper_harness/p0_paper_exact/run_2/independent_same_fold/mbw_lr/summary.json`
- 进度：`experiments/paper_harness/p0_paper_exact/run_2/independent_same_fold/mbw_lr/progress.json`
- 配置：`configs/experiments/p0_cone_independent_same_fold.json`
- 图：`experiments/paper_harness/p0_paper_exact/Figure_5.png`
- checkpoint SHA-256：`8c046c07cdf767cd76126330914baefdb2b335e47f641dbca9f4ca95fdce308e`

## 剩余 P3 门槛

1. 在相同 50 折上运行本仓库独立 MBW-LR、EAC、WDI、ACM 和 CASIM 基础分类器，而不是复用作者分数。
2. 运行完整独立 `ConEAlarmFloodClassifier` 端到端链路并分离 base-model delta 与 conformal-layer delta。
3. 在归档 Code Ocean Docker 镜像中复跑作者网格；当前机器缺少 Docker CLI。
