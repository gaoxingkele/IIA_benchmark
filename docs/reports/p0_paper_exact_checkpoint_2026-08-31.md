# P0 论文精确复现 checkpoint（2026-08-31）

## 当前结论

CASIM、ConE-AFC 和 BiP-AFC 三项 P0 作者代码计算网格已经全部完成，共形成 **1,130 个可审计原子任务**：CASIM 700 个 model fit、ConE 250 个 `split × model` 任务、BiP 180 个 `lane × dataset × model × fold` 任务。三者的任务粒度不同，因此总数只表示恢复与审计覆盖量，不用于比较算法复杂度。

当前均为 **P2**，还不能标记为 P3：ConE 已实现论文数值闭合，并闭合了 MBW-LR 作者分数上的独立 conformal 层同折对照；CASIM 和 BiP 保留了未通过容差的负结果；三项仍缺归档 Docker 镜像内复跑，ConE 也仍缺独立基础分类器与端到端同折对照。

| 论文/方法 | 原域数据 | 完整计算 | 冻结数值门槛 | 当前等级 | 核心结论 |
|---|---|---:|---:|---|---|
| CASIM | Code Ocean v1 TEP，310 条序列、16 报警变量 | 700/700 model fits | 1/3 通过 | P2 | 最大 BA 接近论文；最佳阈值和全阈值均值未闭合 |
| ConE-AFC | Code Ocean synthetic，18,750 条五类洪泛 | 作者 250/250；独立层 50/50 folds | 作者表 95/95；独立层 1,800/1,800 | P2 | 作者表格闭合；独立 conformal 层精确闭合，base/end-to-end 仍开放 |
| BiP-AFC / uncertainty reduction | Code Ocean v3 TEP + synthetic | 180/180 六通道任务 | 原四通道 6/16、7/16、7/16、8/16；CASIM 控制 2/4、2/4 | P2 | 官方 v3 不能完整复现论文；Numba 控制闭合，切分重叠只解释部分差距 |

`transfer_result` 与 `paper_exact_result` 始终分开保存。现有 TEP/NPP/FCC 的 E2 迁移结果没有被原域复现结果覆盖，也没有被描述为论文精确复现。

## 逐项结果

### CASIM

- 14 个 held-out class、5 folds、10 个随机实例，seeds 42–51，共 700/700 fits，失败 0。
- 论文/本地最大 balanced accuracy：`0.947 / 0.955520`，差值 `+0.008520`，通过 ±0.02。
- 论文/本地最佳 novelty threshold：`0.324 / 0.335`，未通过论文舍入精度 ±0.0005。
- 论文/本地全阈值 mean balanced accuracy：`0.879 / 0.903267`，差值 `+0.024267`，未通过 ±0.02。
- 结果 checkpoint SHA-256：`1b39242e07ab79af60729a0c4a5e09f785018dba36b287c0827069d76b2a4e18`。

结论：计算覆盖完整，但外层 held-out-class/novel-sample 聚合协议仍可能与论文生成 Figure 13c 的私有包装脚本不同。不得通过选择 seed 或事后调阈值消除差距。

### ConE-AFC

- 10 次重复、每次 5 folds、5 个分类器、3 个 alpha、3 个校准量、51 个时间前缀。
- 250/250 `split × model` 任务组装为 50/50 完整 folds；论文 Tables 1–2 的 95/95 个命名均值全部通过 ±0.02。
- 最大均值绝对差为 `0.002271`；原子 checkpoint SHA-256 为 `db63065f2b8703737782c891a04911e6ff334ac6de9b99e7990fe941f5509a6d`；完整 split SHA-256 为 `850beca912aa28c5550d396d9c2fa130b1aa9a0f7fb61262d18e6022988ef135`。
- 独立 `ConEAFCCalibrator` 在相同 MBW-LR 分数和 50 折上完成 1,800 个指标对照，1,800/1,800 精确一致，最大绝对差 0；checkpoint SHA-256 为 `8c046c07cdf767cd76126330914baefdb2b335e47f641dbca9f4ca95fdce308e`。

结论：这是本轮唯一实现完整论文表格数值闭合的作者代码实验。尚缺归档 Docker 环境与独立实现对照，因此仍为 P2。

### BiP-AFC / uncertainty reduction

- TEP 数据形状为 `1,000 × 50 × 300`，synthetic 为 `1,875 × 10 × 60`；缓存和作者 CSV 逐值一致，原始数据未覆盖。
- 四条完整模型通道：`author_overlap`、`paper_disjoint`、`seeded_author_overlap`、`seeded_paper_disjoint`；另有两条 CASIM-only Numba 控制通道。
- 完整模型通道各有 2 数据集 × 4 模型 × 5 folds，共 160 个任务；CASIM 控制新增 20 个任务，合计 180/180。
- 16 个论文条目的联合门槛通过数依次为 `6/16`、`7/16`、`7/16`、`8/16`。
- 作者通道的 calibration/RF 交叉量为 TEP 250、synthetic 500；disjoint 通道均为 0。
- 只固定 Python-level NumPy seed 时，MBW、EAC、ACM 的 30/30 配对 test bifurcation 完全一致，而 CASIM 的 10/10 仍不一致；根因定位到 vendored MultiRocket 的 Numba `np.random`。
- 同时显式播种 Numba RNG 后，CASIM 的 10/10 配对 test bifurcation 完全一致，最大绝对差由 32 降为 0。TEP point MAE 改善 0.017131，synthetic 则变差 0.031906，证明删除重叠没有统一收益。
- checkpoint SHA-256：`46d054efd950cfa77dd749f688ba3858b04cd27e636252f5e0b829772f462ffd`。

结论：切分重叠会改变部分点预测误差和区间宽度，但不存在统一收益，不能解释全部论文差距。CASIM 的切分消融已经在单进程、`n_jobs_multirocket=1` 条件下具备纯 RF 子集归因能力。

## 完整性与验证

- 三个 Code Ocean capsule 的 archive SHA-256、代码 manifest 和数据 manifest 均通过完整哈希校验。
- 全测试套件：`131 passed`。
- paper-exact harness：0 issues。
- JSON 配置/状态文件：8 个全部可解析。
- BiP 四通道幂等复跑只重建汇总，未重新训练；用时 0.535 秒。
- 运行中由 Python 生成的 capsule `__pycache__` 文件未纳入结果或 Git，已移动到 `tmp/paper_exact_runtime_cache/quarantine/`；原始 capsule 内容恢复为 manifest 一致状态。

## 可审计产物

- CASIM：`experiments/paper_harness/p0_paper_exact/run_1/paper_grid/repetitions_10/`
- ConE：`experiments/paper_harness/p0_paper_exact/run_2/paper_grid/`
- ConE 独立层：`experiments/paper_harness/p0_paper_exact/run_2/independent_same_fold/mbw_lr/`
- BiP：`experiments/paper_harness/p0_paper_exact/run_3/paper_grid/`
- 协议状态：`paper_harness/paper_exact/status.v1.json`
- 冻结总配置：`configs/experiments/p0_paper_exact.json`
- CASIM 报告：`docs/reports/p0_casim_paper_grid_complete_2026-08-30.md`
- ConE 报告：`docs/reports/p0_cone_paper_grid_complete_2026-08-31.md`
- BiP 报告：`docs/reports/p0_bip_paper_grid_complete_2026-08-31.md`

## 恢复命令

```powershell
.\scripts\resume_p0_checkpoint.ps1 -SkipInstall
```

各 runner 以细粒度 checkpoint 为准，已完成任务不会重新训练。所有数据路径继续由 `configs/` 提供；原始书籍、下载论文和 raw data 不在预处理阶段覆盖。

## 尚未闭合及必要性

1. **归档 Docker 复跑（三项）**：隔离 Windows/Linux 文件顺序、NumPy/Numba/Mapie/sklearn 版本差异，满足 P3 环境等价门槛。
2. **本仓库独立实现同折运行（三项）**：把作者代码复现误差与本地算法实现误差分离，避免“作者代码能跑”等同于“算法已完整落地”。
3. **CASIM 外层协议与 Figure 14b–c**：需要作者确认 held-out-class wrapper，并复现 `nclf`、`nfeat` 消融，解释全阈值均值差距。
4. **BiP 文件顺序和版本**：需要在 Docker 中核对 `os.listdir` 顺序，并向作者确认论文 commit、数据生成次序和 Mapie 版本；Numba RNG 控制已经闭合，不再列为阻塞项。
5. **ConE 独立基础模型与端到端对照**：独立 conformal 层已在 MBW-LR 作者分数上精确闭合；还需把五个基础分类器换成本仓库实现，并运行完整 `ConEAlarmFloodClassifier`，才能闭合实现链路。

P3 完成标准保持不变：同一数据、预处理、分组切分、超参数、种子、指标和目标表格；作者代码与本地实现都在可复现环境中运行，并对差值作逐表审计。

## 本轮提交

- `6475548` — complete CASIM paper open-set grid
- `9e42f0d` — close ConE 50-fold paper grid
- `5c47993` — close BiP paper grids and split controls

以上提交均已推送到 canonical `main`。
