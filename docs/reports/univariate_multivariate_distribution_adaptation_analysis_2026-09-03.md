# IIA Benchmark 单变量与多变量报警验证、分布分析及适应性改进报告

**报告日期：** 2026-09-03  
**项目仓库：** https://github.com/gaoxingkele/IIA_benchmark  
**验证范围：** 书籍 Chapter 2 单变量报警、Chapter 3 多变量报警、经典基线及分布适配层  
**证据边界：** 真实公开数据上的工程迁移验证（M2/P1）与书籍公式/命名条目验证；不将合成 smoke 或代理数据结果表述为论文原分数复现

## 1. 执行摘要

本轮完成了四项工作：第一，重新核对书籍 Chapter 2 的四类单变量算法和 Chapter 3 的五类多变量算法是否具有可调用实现、配置、测试、数据和实验记录；第二，在 TEP Classic、PRONTO、SKAB 的冻结分区上分析边际分布、位置、协方差、相关结构、有效秩和时间依赖；第三，对单变量方法验证 ECDF、稳健缩放、近期窗口、block delay、污染保护和自动拒绝等适配策略；第四，新建多变量稳健收缩 Mahalanobis、近期 block 校准和 `static/adapt/reject` 路由，并完成三数据集、三种子、逐点/事件/不确定性验证。

最重要的结论不是“适配一定提高性能”，而是不同数据集需要不同级别的适配：

1. **TEP 的分布迁移最轻。** 单变量双侧 ECDF 将 F1 从 0.8678 提高到 0.9502，同时把 MAR 从 0.1840 降到 0.0458；多变量传统 Mahalanobis 仍优于新增稳健/block 适配。
2. **PRONTO 的主要矛盾是强时间依赖和异常阶段漂移。** 单变量 block 适配能提高召回，但 FAR 升至 0.4313；多变量 block 适配 F1 从 0.7229 降到 0.5640。只在正常训练段上重校准，无法预见未来工况变化。
3. **SKAB 的主要矛盾是正常基线、协方差与相关结构同时改变。** 单变量近期窗口将 FAR 从 0.5949 降到 0.1446，但仍约有 55.1 次虚警事件/小时；多变量 M2 的 FAR 反而从 0.8965 升至 0.9414。
4. **稳健缩放和协方差收缩只能解决尺度与病态矩阵，不能解决正常流形整体移动。** 多变量 M1/M2 在三个数据集上均未超过 M0，必须保留这一负结果。
5. **自动拒绝是必要能力。** 单变量路由拒绝全部 TEP d11 单元；多变量路由同样拒绝三个 d11 seed 单元。选择性 F1 不能与全覆盖方法直接比较，报告必须同时给出 coverage。

因此，下一阶段的优先级应从“继续统一调阈值”转为：工况识别与条件化模型、多源正常域训练、动态残差建模、事件级校准，以及在无法证明适用时回退到多变量或显式拒绝。

## 2. 为什么要这样验证

### 2.1 原问题

早期结果表明，相同单变量算法在 TEP、PRONTO、SKAB 上差异极大；多变量 NOZ、Mahalanobis 和变化方向方法也出现“一侧低 FAR、另一侧高 MAR”或近乎持续报警。这种现象既可能来自实现错误，也可能来自数据分布、标签定义、采样周期、工况混合和时序结构不一致。

若不先拆分原因，继续调参会产生三类错误结论：

- 用测试集选择变量或阈值，形成数据泄漏；
- 把高召回但持续报警误称为性能优秀；
- 把代理数据迁移结果误称为书籍或论文原实验复现。

### 2.2 本轮采用的验证原则

- `configs/` 是唯一协议来源，模型中不硬编码数据路径。
- 原始书籍和下载数据只读；实验只写入 harness 与 report 目录。
- 正常训练、异常校准、正常评估、异常评估严格分离。
- 变量选择、方向、阈值、delay 和路由只读训练/校准分区。
- 测试分布统计是冻结后的解释性证据，不反向调参。
- 三个随机种子为 1103、2207、3301；同一物理 episode 的多 seed 只反映算法随机性，不冒充独立工业样本。
- 同时报告 FAR、MAR、F1、AAD、事件召回、FAR/hour、报警持续时间和 moving-block bootstrap。
- synthetic 只验证公式和管线，不计入公开数据性能结论。

## 3. 数据集与任务匹配

| 数据集 | 连续变量 | 采样特征 | 本轮用法 | 优点 | 主要局限 |
|---|---:|---|---|---|---|
| TEP Classic | 52 | 约 180 秒/点；独立正常训练/测试文件 | d01、d04、d11；单变量 T1、多变量 T2 | 故障定义稳定、分区清晰、便于机制对照 | 仿真过程，不是专家确认的真实报警载荷 |
| PRONTO | 17 个过程变量，另含报警/控制列 | 约 1 秒/点；Testday2/3/4 | Air blockage、Air leakage、Slugging | 真实试验过程、具有工况和故障阶段 | 正常段短、强自相关、同一天内复用，不能进严格 leaderboard |
| SKAB | 8 | 约 1 秒/点；anomaly-free 与多故障 run | valve1/0、valve1/1、valve2/0；扩展验证覆盖 34 run | 多传感器、点级异常标签、公开可重复 | 正常基线明显漂移、异常比例低、同一 anomaly-free 文件再切分 |
| Chapter 3 condenser synthetic | 4 个物理量 | 按书中参数采样 | 仅验证 3.90-3.103 方程和区间计算 | 可验证单位、拟合和边界公式 | 不是 300 MW 机组工业结果，不能计入真实性能 |

Chapter 2/3 多数据集协议均选择三个数据集的三个 episode，再在三个种子上重复，共 27 个 episode-seed 单元。多变量每个单元由异常校准段选择四个变量；正常评估和异常评估保持冻结。

## 4. 已验证的算法

### 4.1 单变量算法

| ID | 章节/论文 | 落地内容 | 当前证据 |
|---|---|---|---|
| `book_2_1_iid_delay_timer` | 2.1；Xu et al. 2012，DOI 10.1109/TASE.2011.2176490 | IID FAR/MAR/AAD、on/off delay、设计搜索 | E4：Examples 1-2 与 Table VII 数值复现；三数据集迁移完成 |
| `book_2_2_non_iid_delay_timer` | 2.2；Wang et al. 2022，DOI 10.1016/j.jprocont.2022.01.002 | 报警持续/间隔 PMF 与 non-IID delay | E2：可执行；19/27 单元触发零事件退化，不能给完整激活信用 |
| `book_2_3_non_iid_deadband` | 2.3；Wang et al. 2023，DOI 10.1109/TCST.2023.3240020 | 最大幅值偏差、Beta 后验、deadband 宽度 | E2：仅 7/27 单元满足机制前提；其余保留拒绝/退化状态 |
| `book_2_4_alarm_probability_plot` | 2.4；Yu et al. 2017，DOI 10.1109/TIE.2017.2682783 | APP 状态、转移矩阵与 trippoint | E3 negative：可运行但跨数据 FAR 偏高 |
| `AdaptiveUnivariateAlarmRouter` | 适配层 | 稳健缩放、ECDF、block/recent、safe rolling、Page/CUSUM、时间 delay、拒绝 | 三数据集三种子 B0-B7 消融与不确定性完成 |

### 4.2 多变量算法

| ID | 章节/论文 | 落地内容 | 当前证据 |
|---|---|---|---|
| `book_3_1_convex_noz` | 3.1.2；Yu et al. 2020，DOI 10.1109/TCST.2019.2943469 | 凸包 NOZ、fitness、最近正常点与动态限 | E3：Figure 3.2 精确通过；三数据迁移未胜 Mahalanobis |
| `book_3_1_nonconvex_noz` | 3.1.3；Wang et al. 2024，DOI 10.1109/TASE.2022.3222413 | 搜索锥、角度分区、径向 membership | E3 negative：执行完整但 PRONTO/SKAB FAR 高；原 CSTR 序列缺失 |
| `book_3_2_variation_direction` | 3.2；Chen et al. 2017，DOI 10.1016/j.cherd.2017.04.011 | 自适应时间梯度与方向规则矩阵 | E2：书中方向例通过；代理数据 MAR 接近 1 |
| `book_3_3_electrical_pump` | 3.3；Xiong et al. 2018，DOI 10.1109/TII.2017.2749332 | Bayesian 窗口回归、异常冻结、R²/正态门禁 | E2：可执行；仅 4/27 统计激活，0/27 获得泵领域信用 |
| `book_3_4_condenser` | 3.4；Wang et al. 2024，DOI 10.1109/TCST.2024.3370036 | 物理压力模型、NOZ、Beta-binomial 最坏 FAR/MAR | E2 synthetic：拟合 >0.99998，但 99% FAR 上界 0.3500 未通过；工业数据缺失 |
| `MahalanobisAlarm` | 经典 MSPC 基线 | 全局均值、协方差逆与经验分位数 | TEP/PRONTO/SKAB 多协议验证完成 |
| `AdaptiveMultivariateAlarmRouter` | 本轮适配层 | median/MAD、收缩协方差、近期 block 校准、延迟和拒绝 | 真实三数据集三种子 M0-M3 验证完成；总体为负结果 |

## 5. 统计分布分析

### 5.1 单变量分布

| 数据集 | 正常 train-eval KS 中位数 | 异常 cal-eval KS 中位数 | 评估 AUC 中位数（最低） | 正常 lag-1 特征 | 异常比例中位数 | 解释 |
|---|---:|---:|---:|---:|---:|---|
| TEP | 0.0547 | 0.0990 | 1.0000（0.5127） | 多数较弱，部分可达 0.68 | 0.3333 | 正常分布相对稳定；d11 单变量可分性不足 |
| PRONTO | 0.3649 | 0.8876 | 0.5343（0.1338） | 中位数 0.9916 | 0.4738 | 正常漂移、异常前后阶段变化及强自相关同时存在 |
| SKAB | 0.4777 | 0.1440 | 1.0000（1.0000） | 约 0.7660 | 0.0487 | 排序可分但阈值失配；低 prevalence 放大虚警负担 |

这些统计量解释了为什么“同一阈值 + 同一 delay”不能迁移。TEP 主要是少数故障不可分；PRONTO 的异常校准段和评估段并非同分布；SKAB 虽然 AUC=1，正常基线却跨越了训练阈值，因此 AUC 高并不代表 FAR 可控。

### 5.2 多变量分布

下表基于每个 episode 的四个校准期选中变量，汇总正常 train→evaluation 的中位数。

| 数据集 | 边际 KS | 最大标准化中位数漂移 | 协方差相对 Frobenius 漂移 | 最大相关系数漂移 | 评估有效秩/4 | lag-1 中位数 |
|---|---:|---:|---:|---:|---:|---:|
| TEP | 0.0930 | 0.5116 | 0.5141 | 0.1184 | 3.27 | 0.7343 |
| PRONTO | 0.3126 | 0.3474 | 0.2394 | 0.2864（最大 1.0296） | 1.52 | 0.9917 |
| SKAB | 0.5690 | 1.3154 | 0.9458 | 1.3411 | 2.91 | 0.7476 |

含义如下：

- TEP 的边际漂移较小，但协方差变化约为训练协方差范数的 51%，因此“静态椭球”仍有误差；不过漂移尚不足以抵消传统 Mahalanobis 的优势。
- PRONTO 的有效秩只有约 1.52/4，说明四个选中变量高度共线；lag-1 约 0.992，逐点 IID 假设严重不成立。收缩协方差能改善数值条件，但不能恢复未来故障阶段。
- SKAB 同时出现位置、协方差和相关结构的大幅变化。单一全局椭球、凸包和搜索锥都会把新的正常工况判到 NOZ 外，因此出现 0.90 以上 FAR。

## 6. 单变量实验结果

### 6.1 书籍原方法迁移基线

| 数据集 | IID F1/FAR/MAR | non-IID F1/FAR/MAR | deadband F1/FAR/MAR | APP F1/FAR/MAR |
|---|---|---|---|---|
| TEP | 0.8678 / 0.0067 / 0.1840 | 0.7231 / 0.2190 / 0.1993 | 0.8480 / 0.0044 / 0.2049 | 0.5780 / 0.6472 / 0.1579 |
| PRONTO | 0.3011 / 0.1119 / 0.7905 | 0.3332 / 0.1214 / 0.7211 | 0.0479 / 0.1207 / 0.9730 | 0.2311 / 0.1161 / 0.8496 |
| SKAB | 0.1448 / 0.5949 / 0.0125 | 0.1093 / 0.8277 / 0.0130 | 0.3095 / 0.2283 / 0.0000 | 0.1545 / 0.5586 / 0.0000 |

结论：原始书籍机制具有明确的前提。TEP 上 IID 方法有效；PRONTO 上所有方法均存在严重漏报；SKAB 的低 MAR 主要由大量报警换来。APP、non-IID 和 deadband 的机制激活状态必须与分数一同报告。

### 6.2 分布适配消融

| 数据集 | B0 冻结基线 | 最佳工程候选 | 候选 F1 | 候选 FAR | 候选 MAR | 判断 |
|---|---:|---|---:|---:|---:|---|
| TEP | 0.8678 | B2 双侧 ECDF | 0.9502 | 0.0271 | 0.0458 | 性能保持门禁通过 |
| PRONTO | 0.3011 | B5 block-recent 双侧 ECDF | 0.6738 | 0.4313 | 0.2824 | 召回改善但 FAR≤0.15 失败 |
| SKAB | 0.1448 | B4 block-recent 单侧 ECDF | 0.4124 | 0.1446 | 0.0042 | 逐点改善；事件虚警仍不可接受 |

自动路由 B7 在 TEP 上拒绝 3/9 个 d11 单元，其余 6 个单元 F1=0.9501、FAR=0.0526、MAR=0；这是 **67% coverage 的选择性结果**。PRONTO B7 F1=0.6552、FAR=0.2935，仍未通过门禁。SKAB B7 等同 B4。

SKAB valve1/0 的端到端接入进一步显示：F1=0.4134，95% moving-block bootstrap 区间为 [0.3283, 0.5322]；FAR=0.1446，区间为 [0.0901, 0.2097]；正常段产生 72 个虚警事件，约 55.1 次/小时，最长持续 214 秒。60 秒正常块出现报警的 Beta(1,1) 后验均值为 0.4074，95% 区间 [0.3035, 0.5156]。因此逐点 FAR 低于 0.15 仍不等于现场负担可接受。

## 7. 多变量实验结果

### 7.1 书籍 Chapter 3 方法与经典基线

| 数据集 | Convex NOZ F1/FAR/MAR | Search-cone F1/FAR/MAR | Variation F1/FAR/MAR | Pump proxy F1/FAR/MAR | Mahalanobis F1/FAR/MAR |
|---|---|---|---|---|---|
| TEP | 0.7210 / 0.3470 / 0.0440 | 0.6385 / 0.5385 / 0.0278 | 0.1455 / 0.0600 / 0.9093 | 0.4655 / 0.0936 / 0.5491 | **0.8921 / 0.0692 / 0.0766** |
| PRONTO | 0.6766 / 0.8069 / 0.0502 | 0.6661 / 0.5600 / 0.0825 | 0.0294 / 0.0105 / 0.9848 | 0.7037 / 0.4209 / 0.1115* | **0.7229 / 0.4604 / 0.0902** |
| SKAB | 0.0950 / 0.9744 / 0.0000 | 0.0939 / 0.9862 / 0.0000 | 0.0077 / 0.0063 / 0.9953 | 0.1271 / 0.7650 / 0.0000* | 0.1049 / 0.8965 / 0.0000 |

`*` PRONTO/SKAB pump proxy 的统计门禁激活率为 0，不能因 F1 数值较高而获得方法有效信用。多变量几何模型在 SKAB 上几乎持续报警；变化方向模型在三个数据集上过于保守或规则覆盖不足。

独立全数据协议还得到：TEP Mahalanobis F1=0.8998、PRONTO F1=0.3978、SKAB F1=0.5442，PRONTO Search-cone F1=0.5767。它们与上表不同是因为变量数、故障集合和切分不同，不能跨协议直接排名。

### 7.2 多变量适配前后

| 数据集 | M0 F1/FAR/MAR | M1 robust+shrinkage | M2 + recent/block | M3 auto/selective |
|---|---|---|---|---|
| TEP | **0.8921 / 0.0692 / 0.0766** | 0.8574 / 0.1230 / 0.0611 | 0.8351 / 0.1493 / 0.0678 | 0.8765 / 0.1410 / 0.0003，coverage 0.667 |
| PRONTO | **0.7229 / 0.4604 / 0.0902** | 0.6637 / 0.3870 / 0.1747 | 0.5640 / 0.4703 / 0.2391 | 与 M2 相同，coverage 1.0 |
| SKAB | **0.1049 / 0.8965 / 0.0000** | 0.0993 / 0.9368 / 0.0000 | 0.0984 / 0.9414 / 0.0037 | 与 M2 相同，coverage 1.0 |

M2 相对 M0 的 F1 变化为 TEP -0.0570、PRONTO -0.1589、SKAB -0.0065，FAR 则分别增加 0.0801、0.0099、0.0448。三项均未获得晋级。M3 在 TEP 的低 MAR 来自拒绝 d11，不能解释为全覆盖性能提升。

多变量适配失败的原因是：

1. robust/MAD 只稳定每个变量的边际尺度，不会对齐旋转或移动后的相关结构；
2. 收缩协方差降低条件数，却仍假设单一椭球正常区；
3. recent block 只读训练期尾部，无法预见 PRONTO/SKAB 的未来正常工况；
4. delay 能滤除孤立越界，无法把整段新正常流形移回 NOZ；
5. 校准期按异常中位数选变量可能偏向某一阶段，异常评估阶段发生变化时排序失效。

## 8. 改进方向

### 8.1 P0：先解决工况错配

**工况条件化。** 使用可观测工况标签、控制设定值或无监督 regime clustering，将正常数据划分为工况，再分别建立 ECDF、Mahalanobis/NOZ 或残差模型。新工况不能静默并入旧阈值，应进入 `unknown_regime` 并拒绝或人工确认。

**多源正常域。** 当前 PRONTO/SKAB 正常训练段覆盖不足。需要按设备、日期、负荷、批次收集多个正常 group，采用 leave-one-group-out 验证，而不是继续细切同一文件。模型只有在未见 group 上满足 FAR/hour 才能晋级。

**动态残差替代原值距离。** 对强自相关和共线数据，优先比较 VAR/DPCA/CVA、Kalman innovation、动态 PLS 或状态空间残差。报警对象应是预测残差和创新，而不是原始测量的静态欧氏/马氏距离。

### 8.2 P1：把适配变成受控决策

**双层变点机制。** 第一层 Page/CUSUM 或多变量 score 变点只负责识别“分布发生变化”；第二层在污染保护、最小正常证据和人工/工况确认后重校准。不能在未知异常段直接滚动更新正常参考。

**稳定变量选择。** 从“最大校准 shift”改为跨 episode 最小 AUC、方向一致性、正常漂移惩罚和组稳定性联合评分。变量在任一关键 group 失效时降权或拒绝。

**相关结构适配。** 对协方差旋转明显的 SKAB，比较 per-regime shrinkage、CORAL 式二阶对齐、子空间角度门禁和 mixture-of-local-NOZ；只在正常 group 之间学习对齐，禁止使用测试异常标签。

**选择性预测。** 路由必须同时输出 coverage、拒绝原因和回退模型。若 calibration 最差 block AUC 低、方向不一致或未知工况分数高，则转动态多变量模型，不强制给出单变量分数。

### 8.3 P1：从逐点指标转向报警工程指标

- 阈值目标改为每小时虚警事件数、最长持续虚警、事件召回和秒级检测延迟。
- 按真实采样周期定义 delay，不以“样本数”跨数据集复用。
- 对正常/异常分区分别使用 moving-block bootstrap，保留时间依赖。
- 零事件使用显式 Beta 后验及先验参数，不使用隐藏常数回退。
- 排行榜同时报告 point、event、uncertainty、robustness 和 coverage；任一缺失不得发布单一 F1 排名。

### 8.4 P2：闭合书籍与论文原条件

- Chapter 2 仍需 non-IID delay、deadband、APP 对应的原始工业波形、持续时间 PMF 和论文 split。
- Chapter 3 仍需 CSTR 原生成序列、四组电泵数据、300 MW 冷凝器 100 天工况、变量单位和 Tables 3.6-3.7。
- 取得原数据后按同一数据、预处理、切分、参数、指标和目标表格执行 P3；代理数据结果仅保留为 transfer_result。

## 9. 新数据集适配决策流程

```text
注册数据与 citation/SHA-256
  → group 隔离与时间 purge
  → 正常/异常 train-cal-eval 四分区
  → 边际 KS/Wasserstein + 位置/协方差/相关/有效秩/lag-1 审计
  → 校准期最差 block 可分性
      ├─ 弱或不稳定：reject，转动态残差/任务专用模型
      ├─ 分布稳定：static ECDF 或 static robust Mahalanobis
      └─ 漂移但可分：regime/recent/block 适配
  → 冻结后评估 point + event + uncertainty + coverage
  → 未满足 FAR/hour、MAR、delay 或 coverage：不晋级
```

推荐的最低晋级门禁为：不存在分区重叠；所有超参数冻结；至少三个独立 group；FAR、MAR、事件指标和置信区间完整；没有利用评估标签路由；相对基线改善在多个 group 上方向一致；拒绝单元单独计数。

## 10. 可复现性与质量控制

核心命令：

```powershell
python experiments/paper_harness/chapter2_multidataset/experiment.py --out_dir experiments/paper_harness/chapter2_multidataset/run_1
python experiments/paper_harness/chapter3_multidataset/experiment.py --out_dir experiments/paper_harness/chapter3_multidataset/run_1
python experiments/paper_harness/univariate_adaptation/experiment.py --out_dir experiments/paper_harness/univariate_adaptation/run_1
python experiments/paper_harness/multivariate_adaptation/experiment.py --out_dir experiments/paper_harness/multivariate_adaptation/run_1
python experiments/paper_harness/multivariate_adaptation/plot.py
python -m pytest -q
```

主要机器证据：

- `experiments/reports/book_ch2_multidataset_validation.json`
- `experiments/reports/univariate_distribution_audit_validation.json`
- `experiments/reports/skab_univariate_onboarding_validation.json`
- `experiments/reports/book_ch3_multidataset_validation.json`
- `experiments/reports/multivariate_distribution_adaptation_validation.json`
- `experiments/reports/tep_mahalanobis_validation.json`
- `experiments/reports/pronto_mahalanobis_validation.json`
- `experiments/reports/skab_mahalanobis_validation.json`
- `experiments/reports/pronto_search_cone_validation.json`

本轮新增多变量适配的 M0 与冻结 Chapter 3 Mahalanobis 在三个数据集上的 FAR/MAR/F1 完全一致；所有 scored 单元均含 moving-block bootstrap；单元测试、全仓测试和 JSON 校验在提交前执行。

## 11. 最终结论

1. 单变量方法具备可落地的条件适配能力，但不存在一个对 TEP、PRONTO、SKAB 都满足低 FAR/低 MAR 的统一适配器。
2. TEP 可优先使用双侧 ECDF；d11 应拒绝单变量并转动态多变量残差。
3. PRONTO 需要工况/变点路由与动态模型，block-recent 只能交换 FAR 和 MAR，不能形成部署级改进。
4. SKAB 的单变量 recent-window 有工程改善，但事件虚警仍高；多变量静态椭球及其稳健收缩变体均失败。
5. 新增多变量适配实验否证了“稳健缩放 + 协方差收缩 + block calibration 足以解决迁移”的假设。后续应建模正常工况流形与时序状态，而不是继续对全局阈值做局部优化。
6. 当前成果可作为可重复的 M2/P1 transfer benchmark；书籍/论文 P3 仍取决于原始泵、冷凝器、CSTR 和专用报警载荷。

