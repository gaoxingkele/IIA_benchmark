# IIA Paper Harness 多数据集复现计划

截止 `2026-08-29`，Paper Harness 已把 **20 个书籍交付项 + 10 个 SOTA 方法**全部纳入实验矩阵。
矩阵展开为 **121 个算法×数据集目标**：其中 **111 个 M2/M3 有效匹配**，**10 个 M1/P0
错配哨兵**。30 个算法均规划了至少 3 个有效数据集，因此目标层已经满足多数据集设计；当前
还不等于全部实验已完成。

## 当前可执行面

| 项目 | 数量 | 含义 |
|---|---:|---|
| 注册算法 | 30 | 20 book + 10 SOTA，全部 callable |
| 已有 E2 真实数据证据的算法 | 10 | 其余 20 个仍需统一 runner 和真实数据配置 |
| 数据集族 | 11 | 11/11 主载荷已在本地；FCC、TEP 五类与 NPP alpha=0.50 adapter/G0 已完成 |
| 算法×数据集目标 | 121 | 包含有效匹配与诊断哨兵 |
| M2/M3 有效目标 | 111 | 可进入跨数据集汇总 |
| M1/P0 哨兵目标 | 10 | 仅保留错配退化证据 |
| 按现有数据适配器可调度 | 118 | 其中 108 个为 M2/M3，10 个为错配哨兵 |
| 被适配器阻塞 | 3 | CoMoPI/EnAS/iMAKS 各 1 个仍待统一入口 |
| 论文实验 backlog | 28 | 与本地 literature registry 28/28 对齐 |

“数据已下载”和“可进入公平实验”是两件事。当前适配器按解锁收益排序：

| 优先级 | 数据集适配器 | 可解锁目标 |
|---:|---|---:|
| 已完成 | FCC Alarm | 25 个有效目标已具备数据入口；T4 首批 6 runs 已执行 |
| 已完成 | TEP Alarm 五类载荷 | 18 个目标已有统一 episode/split 入口；首批 6 runs 已执行 |
| 已完成 | NPP Alarm DataPort | 17 个目标已有统一入口；alpha=0.50 首批 6 runs 已执行 |
| 3 | CoMoPI | 1 |
| 3 | EnAS | 1 |
| 3 | iMAKS | 1 |

FCC G0 发现其 1600 个 run 均为异常场景、没有独立正常工况，因此已从 Chapter 3 NOZ 数据目标
中移除并由 SMD/SKAB 替代；FCC 继续用于报警状态复现、RCA、洪泛分类、预测和可视分析。NPP
alpha=0.50 的 grouped adapter 已解锁 17 个目标。下一步补 TEP 100-run/异常变体与 NPP 跨 alpha
处理鲁棒性入口，并把书籍 Chapter 4/5 及 AFC SOTA 的其余 callable 实现接入这三类报警载荷。

## 固定内核与允许修改范围

固定内核 K 包括数据哈希、adapter 输出 schema、grouped split、指标公式、随机种子、负结果保留和
ARA 证据绑定。允许修改的 X 只包括预注册的模型超参数、窗口/表示适配、训练预算、校准量和扰动
强度。测试标签、metric、group membership、事后挑选数据集/seed 均不能进入 X。

每项实验依次通过：

1. `G0 validity`：完整性、先验分布、标签语义、split 与泄漏审计；
2. `G1 activation`：算法专属 beacon，先确认机制真正激活；
3. `G2 multi-dataset`：至少 3 个 M2/M3 数据集、3 个固定 seed；
4. `G3 competitive credit`：只有主张优于基线时才要求 paired mean Δ > 0 且 `|z| >= 1.96`；
5. `G4 reference reproduction`：明确书/论文表、图或案例，记录 P2/P3 差异和容差；
6. `G5 held-out`：协议哈希冻结后只评一次。

有效但性能差的结果可以通过 G0/G1 并作为负结果发布；它只是不获得“更优”信用。这样不会把
CTFH/HDAM/ConE 的退化隐藏成失败，也不会把 coverage 1.0、集合大小 4.0 误写成有效不确定性。

## 六个执行阶段

| 阶段 | 工作 | 退出条件 |
|---|---|---|
| S0 数据预检 | 11 族完整先验统计；补 6 个 adapter；冻结 run/device split | G0 全通过；matrix/reference 校验通过 |
| S1 初始实现 | 将 30 个算法全部接入统一 runner；为每种机制写 activation beacon | 30/30 有 bounded config、unit test、activation 记录 |
| S2 基线调参 | 每算法先在至少 2 个数据集 train/calibration 调参；固定 lane parent | 不触碰 test；搜索预算、超参范围冻结 |
| S3 多数据集复现 | 112 个有效目标，正式 seed `1103/2207/3301`；复现书/论文条目 | 逐数据集+macro 结果；P1/P2/P3 明示；G2/G4 结论 |
| S4 消融与鲁棒性 | 组件消融；missing/spurious/jitter/delay；prefix/open-set | 与 S3 同数据/seed；配对差异与负结果保留 |
| S5 最终留出 | 一次性 held-out；独立审计；表图、claim ledger、ARA | G5 通过；结果、引用、章节和代码形成闭环 |

## 任务与方法波次

| 波次 | 范围 | 首选数据集 | 重点复现 |
|---|---|---|---|
| W1 | T1 Chapter 2 四算法 | TEP classic、PRONTO、SKAB、FCC | FAR/MAR/AAD；PMF/Bayesian interval；deadband；APP |
| W2 | T2 Chapter 3 五算法 | TEP classic、PRONTO、SKAB、FCC | convex/search-cone NOZ；variation direction；pump/condenser |
| W3 | T3 Chapter 4 六算法 | TEP classic/Alarm、FCC、NPP、PRONTO、iMAKS | NTE/NDTE、IGTE/IGDTE、BN、PLR；Top-k/MRR/edge F1 |
| W4 | T4 Chapter 5 + AFC SOTA | TEP Alarm、FCC、NPP；SMD 仅检测；PRONTO 仅哨兵 | flood interval、alignment、patterns、closed/open/prefix AFC |
| W5 | T5/T6 | PIADE、TEP Alarm、FCC、SMD、EnAS、CoMoPI | next alarm、uncertainty reduction、书 Chapter 6 事实一致性 |
| W6 | 全 lane 鲁棒性/消融 | 与各自 W1-W5 相同 | 统一 corruption、组件消融、资源开销、跨域汇总 |

## 书籍和论文实验闭环

书籍按 Chapter 2–6 建立了 5 组条目；28 篇论文逐篇登记在
`paper_harness/reference_experiments.v1.json`。当前只有 ConE-AFC 的实验协议已经抽取到足够细的
数字级条目，但其 Code Ocean 数据仍被 403 阻塞；5 篇 TEP 系 SOTA 已有近同族载荷，可先做 P2，
仍需抽取/冻结 exact split、参数和表格；其余论文必须先完成全文/原始数据门禁，不能用跨数据集
结果冒充论文表格复现。

各条目的输出必须同时回答：复现的是哪一章/哪篇论文、哪张表/图/案例、数据是否相同、协议差异、
目标值与容差、实际值与误差、是否通过 G4。若原始数据缺失，则执行 P1 多数据集工程验证，并保留
`exact_data_blocked`，而不是把状态改为 verified。

## 首轮实际执行顺序

1. 给 FCC alarm/process/valve/disturbance 四类文件建立 run-aligned adapter、16 类 grouped split 和先验报告；
2. TEP Alarm 的 1000 个五类 CSV 已完成统一 episode schema；100-run Original/Filter/Deadband 与异常变体仍待选择性适配；
3. 给 NPP 101 runs/12 fault families+Normal 建 episode 与 open-set split；
4. 把尚无真实 runner 的 20 个算法接入 lane runner，并逐个通过 G1；
5. 先跑 CPU 经典方法的 W1-W3，再跑 T4 经典方法，最后调度 MultiRocket/LSTM/Transformer/HDAM；
6. 每完成一个波次生成独立结果目录、gate ledger、ARA validation，再按项目规则单独 Git 提交。

## FCC Wave 1 已执行结果

FCC 使用 16 类、每类 100 个完整 run；run 1–60 训练、61–80 校准、81–100 测试。G0 检查确认
1600 个 `57×60` 报警矩阵无跨类别完全重复，简单训练类中心的独立测试准确率为 0.9875。过程侧
1530 个缺失值占 0.0664%，已登记 run 内插值和训练折中位数规则。

| 方法 | 表示 | Balanced accuracy / coverage | Macro-F1 / set size | G1 |
|---|---|---:|---:|---|
| CTFH | state | 0.512500 | 0.464254 | pass，62 hashes |
| CTFH | rising edge | 0.365625 | 0.324116 | pass，45 hashes |
| HDAM | state | 0.993750 | 0.993734 | pass，16 类 |
| CASIM | state | 1.000000 | 1.000000 | pass，16 类 |
| ConE + CTFH | rising edge | coverage 0.940625 | set size 8.009375/16 | partial，singleton 0 |
| Cross-Conformal + CTFH | rising edge | coverage 0.965625 | set size 7.671875/16 | partial，singleton 0 |
| Accelerated alignment | activation sequence | 0.887500 | 0.890699 | pass，16 类 |
| CHARM representatives | activation set | 0.931250 | 0.929589 | pass，130 closed / 16 reps |
| Maximum entropy | activation sequence | top-1 0.187324 | top-3 0.412958 | pass，coverage 0.919927 |

这些是 P1 单 split 工程结果，不授予 E3、paired significance 或论文分数信用。它们证明 PRONTO
退化主要是表示/标签错配，而不是所有实现均失效；同时 state 优于 rising edge，必须作为正式消融保留。
CHARM 的父实现还暴露出运行时病理：完整 FCC 上超过 4 分钟且观测内存约 754 MB 未完成；直接
枚举 TID 闭包后同一数据/参数在 1.54 秒完成，并通过小规模 brute-force 等价测试。该修复只获得
runtime gene 信用，不被解释为预测性能提升。

## TEP Alarm Wave 1 已执行结果

TEP 使用 IEEE DataPort DOI `10.21227/326k-qr90` 的五类官方载荷：IDV1、IDV2、IDV6、IDV14、
IDV1+IDV5 各 200 个完整 `50×300` 报警样本。固定 seed 1103，每类 120/40/40 个完整样本进入
训练/校准/测试。G0 完整 ZIP CRC、ground truth 一一映射、二值/分钟网格、跨类重复、分割重叠均通过；
简单训练类中心诊断在 state/rising-edge 上均为 0.975，这说明该固定 split 本身较易，不能据此授予
CASIM 的普适 SOTA 信用。

| 方法 | 表示 | Balanced accuracy / coverage | Macro-F1 / set size | G1 |
|---|---|---:|---:|---|
| CTFH | state | 0.725000 | 0.655143 | pass，996 hashes；IDV14 recall 0 |
| CTFH | rising edge | 0.750000 | 0.683298 | pass，995 hashes；IDV14 recall 0 |
| HDAM | state/full episode | 0.975000 | 0.974902 | pass，5 类 |
| CASIM | state | 1.000000 | 1.000000 | pass，5 类 |
| ConE + CTFH | rising edge | coverage 0.890000 | set size 1.080000/5 | efficient but under target，singleton 0.73 |
| Cross-Conformal + CTFH | rising edge | coverage 0.965000 | set size 1.260000/5 | pass，singleton 0.74 |

HDAM 的 PRONTO 12-bin 父参数在 TEP 60-bin episode 上超过 482.6 秒仍未产生预测，原因是约
3.07 亿次模板位置比较；在不查看测试预测的前提下改为完整 60-bin 模板后，9.67 秒完成。父失败与
修复均保留，修复仅获得运行时信用。上述结果虽使用准确的公开五类载荷，仍是单 seed、本地 60/20/20
split；在论文 split、重复次数、超参和表格逐项闭合前只记 P2/E2，不记严格 paper-score reproduction。

## NPP Alarm Wave 1 已执行结果

NPP 使用 IEEE DataPort DOI `10.21227/g2fa-9y43` 的 alpha=0.50 阈值层。原始 CSV 实际为
`TIME + 192` 个二值报警位、10 秒采样。G0 在 160 样本完整窗口上发现：Normal 只有 1 个 run；
MD 的 100 个 run 具有同一状态轨迹；19 个 state-or-edge 连通分量含 45 个跨标签冲突 run。
固定协议移除这些不可独立识别样本，并为其余 11 类各选 48 个唯一轨迹分量代表，以 seed 1103
分成 28/10/10 train/calibration/test；最终 528 个 run 的 state 与 rising-edge 轨迹均无跨分区重复。

| 方法 | 表示 | Balanced accuracy / coverage | Macro-F1 / set size | G1 |
|---|---|---:|---:|---|
| CTFH | state | 0.836364 | 0.827591 | pass；3,317 hashes，11 类 |
| CTFH | rising edge | 0.818182 | 0.812328 | pass；3,164 hashes，11 类 |
| HDAM | state/full episode | 0.763636 | 0.752807 | pass；11 类，minimum stability 0.802876 |
| CASIM | state | 0.972727 | 0.973137 | pass；11 类，110 个测试中错 3 个 |
| ConE + CTFH | rising edge | coverage 0.900000 | set size 1.300000/11 | pass；singleton 0.572727，empty 0.063636 |
| Cross-Conformal + CTFH | rising edge | coverage 0.963636 | set size 1.536364/11 | pass；singleton 0.463636，empty 0 |

这些是 P1/E2 单阈值、单 seed 的分组工程验证。它们验证了 NPP 事件载荷能激活洪泛指纹与
conformal 效率机制，但不复现 TEP 论文分数。下一步必须以相同 base-run identity 对 alpha 层分组，
执行 `1103/2207/3301` 与跨阈值鲁棒性，避免把同一仿真轨迹的阈值变体泄漏到训练和测试两侧。

## Chapter 4.1 NTE/NDTE 三数据集结构验证

为复现书籍“时滞 NTE 网络—Bernoulli surrogate 显著性—发生次数守卫—NDTE 间接边剪枝”条目，
FCC、TEP Alarm 与 NPP 统一使用每类 2 个固定完整测试 episode、最高二元熵的 6 个报警变量、
至少 3 次发生与 3 次清除、lag 0–3、9 次 surrogate、显著性 0.10。该协议只评估图机制；三个
载荷均无 alarm-tag 级因果真值，因此禁止把故障类别改造成根报警 tag 并汇报虚假的 top-k/MRR。

| 数据集 | 运行数 | 图激活率 | NTE 显著边 | NDTE 剪枝 | 同类/跨类直接边 Jaccard |
|---|---:|---:|---:|---:|---:|
| FCC | 32 | 0.9375 | 394 | 182 | 0.1709 / 0.0165 |
| TEP Alarm | 10 | 1.0000 | 193 | 87 | 0.4550 / 0.0378 |
| NPP alpha=0.50 | 22 | 1.0000 | 525 | 201 | 0.4432 / 0.0313 |

三套数据上同类图重合均高于跨类，且 NTE 与 NDTE 激活门禁均通过；但这些仍是 P1/P2
结构迁移证据，不是原论文私有工业记录的 edge-F1 或根因准确率复现。FCC 的 Chapter 4.4 PLR
适用性审计另外发现：1,600/1,600 个 run 中的注入扰动列均为时间常数，0 个 run 含根扰动变化，
因此无法识别时滞或趋势贡献。`FCC × PLR` 已从 M3/P1 降为 M1/P0 哨兵，负门禁保留而不造分。

校验命令：

```powershell
python scripts/paper_harness.py check
python scripts/paper_harness.py status
python scripts/paper_harness.py plan
python -m pytest -q tests/test_paper_harness.py
```

## Chapter 4.3/4.4 equation and controlled-recreation validation

Chapter 4.4 PLR now executes equations (4.90)-(4.93) for all 3,200 samples,
the published 32 monotone segments, and seeds 41/42/43. All three runs recover
the exact delays `x1=10`, `x2=8`; all 32 active segments rank `x1` first in
segments 1-16 and `x2` first in segments 17-32. Mean contribution in the
published mixed-driver blocks is 0.7459 for x1 (segments 1-8) and 0.7485 for
x2 (segments 17-24). This closes the equation-defined numerical entry at P2,
but remains E1 because the cited two-month thermal-plant payload is missing.

Chapter 4.3 recursive Bayesian RCA now uses the paper's zero probability
initialization, `lambda=1-(0.5)^(1/20)`, online posterior decisions, known,
co-existing, and unknown causes, plus every false/missing-alarm interval listed
in the coal-feeder example. Stable-state accuracy is 1.0 and nuisance-sample
accuracy is 0.8627 versus 0.0588 for instantaneous lookup. The recursive method
has lower overall point accuracy (0.9491 versus 0.9850) and 19-sample transition
delays; this cost is retained. Because the original plant waveforms are absent,
the run is P1/E1 controlled evidence rather than Figure 4.37 reproduction.
