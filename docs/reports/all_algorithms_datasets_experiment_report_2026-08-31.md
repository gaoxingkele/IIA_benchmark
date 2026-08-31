# Intelligence Industrial Alarm Benchmark：全部算法与数据集整体实验报告

日期：2026-08-31  
证据截止：`main` 分支 `4dbca3d` 及其之前的实验产物  
仓库：https://github.com/gaoxingkele/IIA_benchmark  
配置真值：`configs/`；实验矩阵：`paper_harness/experiment_matrix.v1.json`；论文精确状态：`paper_harness/paper_exact/status.v1.json`

## 1. 报告口径

本报告覆盖仓库当前登记的全部 20 个书籍算法、10 个 SOTA 算法、11 个公共/已取得数据集家族、4 个合成 smoke 数据生成器及 6 个下游任务。结果严格区分以下证据：

- `callable`：有配置、可调用实现和测试，不等于论文复现。
- `E1`：受控机制或合成验证。
- `E2`：真实数据工程迁移验证，不能替代论文原域分数。
- `E3/E4`：命名公式、表格或数值条目被复现；E4 为当前最严格书籍条目。
- `P2`：作者代码/官方数据上的完整计算已执行，但协议、数值或独立实现门槛仍未完全闭合。
- `P3`：同数据、同预处理、同切分、同超参数、同指标、独立实现和权威环境均闭合。当前仍无完整 P3 条目。

合成 smoke 结果只用于验证管线，不作为 benchmark 性能。没有专家洪泛区间、因果真值或论文原始载荷的实验，不报告为正式排行榜成绩。

## 2. 总体状态

| 项目 | 当前状态 | 判断 |
|---|---:|---|
| 书籍算法交付项 | 20/20 可调用 | 1 个 E4，19 个 partial/E1-E3；原始工业数据仍是主要缺口 |
| SOTA 算法交付项 | 10/10 可调用 | 全部有真实 TEP/NPP/FCC 工程验证；只有 ConE 达到作者表格闭合，但仍非 P3 |
| 可调用方法族 / 模型配置 | 34 / 39 | 单元和机制测试可运行 |
| 公共/已取得逻辑数据集家族 | 11/11 有主载荷 | 10 个真实/高保真载荷，1 个登记的合成因果载荷 iMAKS |
| 合成 smoke 数据 | 4 | 仅用于管线与不变量验证 |
| 下游任务 | 6/6 可运行 | T4 的论文精确和专家洪泛真值仍不完整 |
| 有真实数据执行的正式算法 | 29/30 | 冷凝器算法只有方程定义合成载荷；无语义匹配真实冷凝器数据 |
| 登记参考论文 | 28 | `access_audit.json` 记录 5 篇全文已取得；其余多为订阅/作者请求门槛 |
| ARA 论文验证命令 | 28/28 通过 | 表示登记命令和证据产物可运行，不等于 28 篇论文分数全部复现 |
| P0 作者代码计算 | 1,130 个任务 | CASIM 700、ConE 250、BiP 180；另有 ConE 独立层 50 折 |
| 当前测试 | 133/133 通过 | 2026-08-31 复核 |
| Capsule 哈希 | 3/3 通过 | 归档、代码清单、数据清单均一致 |
| leaderboard-eligible split | 0 | 现有结果为工程迁移或论文复现审计，尚未发布正式排行榜 |

核心结论：仓库已经完成“算法可调用 + 多数据集工程验证 + 三项 P0 作者工件验证”的主体框架，但不能称为“全部论文完整复现”。当前最可靠的闭合结果是 Xu 2012 的公式/表格复现和 ConE 作者表格及独立 conformal 层复核；最重要的负结果集中在表示错配、原始工业载荷缺失和论文协议不闭合。

## 3. 数据集完整清单与先验审计

| 数据集家族 | 已取得主载荷与主要规模 | 表示/先验结论 | 任务与实验角色 | 当前边界 |
|---|---|---|---|---|
| `tep_classic` | 44 个运行文件，52 个过程变量，2 个正常、42 个故障文件 | 连续变量、故障起点已登记；用于分组迁移 | T1、T2、T3；Chapter 2/3/4 多数据集 | 报警由阈值派生，不是专家工业报警流 |
| `pronto` | 3 个对齐测试日、45,420 个样本、17 个过程变量；报警适配器使用 12 个报警列 | 状态与 rising-edge 两种表示均已审计 | T1、T2、T3、T5、T6；T4 仅作错配哨兵 | 连续状态窗口与洪泛事件指纹严重错配 |
| `piade` | 429,394 条原始记录、5 台设备、92,084 条非 sentinel 报警记录；23,376 条序列行 | 离散制造报警/状态事件，时间顺序可用 | T3 PLR、T5 next-alarm、T6 可视分析 | 无 tag-level 因果真值 |
| `skab` | 35 个实验文件、8 个传感器特征 | 原生点异常标签，不做 point adjustment | T1、T2、T3 多变量迁移 | 工况分布漂移强，部分方法 FAR 很高 |
| `smd10towfgr` | 10 台风机；每台约 26,209 行、132 列；1,002 条 Alarm、688 条 Warning | 时间戳事件可用于密度/序列 | T4 检测、T5、T6 | 无专家洪泛 episode 标签 |
| `tep_alarm_dataport` | 1,000 个运行、5 类、每类 200；每运行 300×50；无缺失、无精确重复 | G0 通过；状态/上升沿共用 600/200/200 分组切分 | T3、T4、T5、T6；SOTA 主数据 | 工程切分不是所有论文的原始 split |
| `npp_alarm_dataport` | alpha=0.50 源切片 1,212 个运行、192 报警变量；11 个评测故障族、160 点窗口 | G0 通过；排除退化 MD 和单例 Normal；去重后 308/110/110 | T3、T4；TEP 外迁移 | 只闭合 alpha=0.50，跨 alpha 和论文 exact 协议待做 |
| `fcc_alarm` | 1,600 个运行、16 类、60×57 报警状态；4,800 个过程/阀门/扰动 CSV | G0 通过；无跨场景精确重复；过程数据缺失率 0.0664% 已量化 | T3、T4、T5；SOTA 主数据 | 高保真仿真，不是实厂日志；旧固定切分存在重复乐观偏差 |
| `comopi` | 150,650 个十分钟 bin、8 台设备、123 类报警、194,974 次报警出现 | 只有 bin 计数，无 bin 内事件顺序；AL_53/54 极稀有 | T5、T6 候选 | 主载荷已取得，但 adapter 仍为 pending |
| `enas` | 219,893 行、33 列，2020-09 至 2021-02；ME/HE/UE 人工错误事件 | 单行错误脉冲；递归 BN 使用显式 5 行前向持久化适配 | T3 递归 BN、T5 | 无专家 tag-level 根因标签 |
| `imaks` | 211,200 条标注传感器记录、9 个站、22 个传感器 | 合成因果数据；209,740 NORMAL，1,460 异常 | T3 因果 smoke/结构诊断 | 只能作为 synthetic causal evidence |

### 3.1 三个核心报警数据集的分布审计

- TEP Alarm：1,000/1,000 唯一状态轨迹；每分钟活动报警中位数 5，单运行 activation edge 中位数 210.5，单运行活动 tag 中位数 29；五类完全均衡，G0 通过。
- NPP Alarm：源切片 13 个家族含 Normal；评测切片保留 11 个非退化故障族。单样本活动报警中位数 44，单运行 activation edge 中位数 107；去除 125 个重复非代表运行和 45 个跨标签冲突运行，分区间无重复组。
- FCC Alarm：每运行 activation event 中位数 13、活动 tag 中位数 12；完整报警状态质心先验准确率 0.9875，仅作为可分性诊断。旧固定切分的 alignment BA 0.8875，按唯一轨迹分组后三种子均值降至 0.7736，证明旧结果受重复泄漏影响。

### 3.2 官方 Code Ocean 数据载荷

| 载荷 | 规模 | 先验验证 |
|---|---|---|
| CASIM TEP | 310 个变长序列、76 通道、15 个标签（含 `-1` outlier） | 308 条唯一轨迹；官方样本级 CV 的第 3-5 折有同标签重复轨迹跨 train/test |
| ConE synthetic | 18,750 条、5 类均衡、60×10 | 18,750 条唯一轨迹，二值、有限、非空，全部检查通过 |
| BiP synthetic | 1,875 条、5 类均衡、60×10 | 唯一、二值、有限、非空，全部检查通过 |
| BiP TEP | 1,000 条、5 类均衡、300×50 | 1,000 条唯一轨迹，全部检查通过 |

另有 `synthetic_step_fault`、`synthetic_multivariate`、`synthetic_root_cause` 和 `synthetic_alarm_floods` 四个生成器，只用于 smoke、单元测试和机制不变量。

## 4. 下游任务覆盖

| ID | 任务 | 已验证数据 | 核心算法 | 当前完成度 |
|---|---|---|---|---|
| T1 | 报警生成与设计 | TEP Classic、PRONTO、SKAB | IID/non-IID delay、deadband、APP | 可运行；只有 Xu 2012 公式/表格达到 E4 |
| T2 | 多变量动态报警限 | TEP Classic、PRONTO、SKAB；冷凝器方程合成 | Convex/Search-cone NOZ、方向报警、Bayesian pump、condenser | 可运行；迁移总体偏弱，缺泵/冷凝器原始载荷 |
| T3 | 因果图与根因排序 | TEP/NPP/FCC Alarm、TEP Classic、PRONTO、SKAB、PIADE、EnAS、iMAKS | NTE/NDTE、IGTE/IGDTE、递归 BN、PLR | 结构/激活已验证；缺专家根因真值，不能报告 RCA accuracy |
| T4 | 洪泛检测、聚类与分类 | TEP/NPP/FCC Alarm；SMD detection；PRONTO mismatch | Criterion C、alignment、patterns、CASIM、CTFH、HDAM、TF-IDF、histogram、conformal | 三数据集多种子工程验证完成；论文 exact 和专家洪泛区间未全闭合 |
| T5 | 下一报警与洪泛预测 | PIADE、TEP/NPP/FCC Alarm；EnAS/SMD 候选 | MaxEnt、empirical、uncertainty-reduction | 可运行；MaxEnt 原域数据缺失，BiP 仍为 P2 |
| T6 | 运行可视分析 | PIADE 已生成完整 HTML/JSON；其他载荷有适配入口 | Chapter 6 visualization suite | 真实数据描述性验证完成；原书工业案例缺失 |

## 5. 书籍算法：章节、论文、数据与实验结论

### 5.1 全部 20 个书籍算法状态

| 算法 ID | 章节 / 对照论文 | 实际验证数据 | 证据 | 实验结论与未闭合项 |
|---|---|---|---|---|
| `book_2_1_iid_delay_timer` | 2.1；Xu 2012 | 方程案例、TEP/PRONTO/SKAB | E4 | Examples 1-2 与 Table VII 通过；最大 Monte Carlo 误差 0.046386，Table VII 最大误差 5.94e-05；工业蒸汽压力数据缺失 |
| `book_2_2_non_iid_delay_timer` | 2.2；Wang 2022 | TEP/PRONTO/SKAB | E2 partial | 27 个 episode-seed 单元中 19 个缺 duration-PMF 机制，不能给完整激活信用 |
| `book_2_3_non_iid_deadband` | 2.3；Wang 2023 | TEP/PRONTO/SKAB | E2 SKAB-only | 45° suitability 仅 7/27 单元通过；论文 case data 缺失 |
| `book_2_4_alarm_probability_plot` | 2.4；Yu 2017 | TEP/PRONTO/SKAB | E3 negative | 方法激活但跨域 FAR/MAR 高，属于负迁移 |
| `book_3_1_convex_noz` | 3.1.2；Yu 2020 | 书中 Figure 3.2、TEP/PRONTO/SKAB | E3 | Figure 3.2 fitness 与 Eq. 3.15 通过；多数据迁移不优于 Mahalanobis |
| `book_3_1_nonconvex_noz` | 3.1.3；Wang 2024 | TEP/PRONTO/SKAB | E3 negative | Eq. 3.18 坐标和 radial membership 执行；高 FAR，原 CSTR 序列缺失 |
| `book_3_2_variation_direction` | 3.2；Chen 2017 | TEP/PRONTO/SKAB | E2 | 数值变化方向恢复，但三数据 MAR 接近 1；精确 ATG 和工业数据缺失 |
| `book_3_3_electrical_pump` | 3.3；Xiong 2018 | 书中 Tables 3.2-3.4、TEP/PRONTO/SKAB | E2 | 表格选择规则通过；统计门仅 4/27，泵语义门 0/27；四组泵数据缺失 |
| `book_3_4_condenser` | 3.4；Wang 2024 | Table 3.5 方程合成 | E2 synthetic | kPa 方程和单模型区间执行；99% FAR 上界 0.3500 未过，300 MW 冷凝器数据缺失 |
| `book_4_1_nte` | 4.1；Hu 2017 | TEP/NPP/FCC Alarm | E2 | 三数据均产生 surrogate-significant 图；原工业报警记录缺失 |
| `book_4_1_ndte` | 4.1；Hu 2017 | TEP/NPP/FCC Alarm | E2 | 三数据均发生间接边剪枝；没有专家真值评估 edge-F1 |
| `book_4_2_igte` | 4.2；Zhang 2022/2023 | TEP Classic、PRONTO、SKAB | E2 | 图能激活，但 TEP Table 4.8 F1=0，PRONTO/SkAB 稳定性弱 |
| `book_4_2_igdte` | 4.2；Zhang 2023 | 受控链、iMAKS、TEP/PRONTO/SKAB | E1 | 受控链 3/3 剪枝；真实/已取得数据 0/21 剪枝，iMAKS 0/3 过阈值 |
| `book_4_3_recursive_bn` | 4.3；Wang 2018 | 受控公式、EnAS、iMAKS | E2 | 原始单行错误脉冲 0/160 决策；5 行持久化后 160/160 激活；缺 tag 根因真值 |
| `book_4_4_plr_rca` | 4.4；Hu 2022 | 书中数值、PIADE、iMAKS | E2 | 书中 delay 10/8 和 32/32 dominant driver 恢复；PIADE 129/300 窗口激活；iMAKS 未恢复 180 样本 lag |
| `book_5_1_flood_detection` | 5.1；Wang 2018 | TEP/NPP/FCC Alarm | E2 descriptive | 三数据 mechanism 全激活；无专家洪泛区间，不能计算真实 FAR/MAR/delay |
| `book_5_2_alarm_alignment` | 5.2；Hu 2016 | Table 5.5/Eq. 5.16、TEP/NPP/FCC | E3 negative | 命名条目通过；9/9 seed-dataset 单元均未胜 set-Jaccard |
| `book_5_3_closed_patterns` | 5.3；Hu 2018 | TEP/NPP/FCC Alarm | E3 | 分类稳定且压缩有效，但 BA 与 class-core Jaccard 相同；921→207 原数据库缺失 |
| `book_5_4_max_entropy_prediction` | 5.4；Xu 2021 | Table 5.15、TEP/NPP/FCC | E3 negative | Table 5.15 通过；三数据 9/9 eta surrogate 未达 0.8；26 条历史洪泛及 Monte Carlo 数据缺失 |
| `book_6_visual_analytics` | Chapter 6 | PIADE | E2 descriptive | 30 天切片生成 HTML/JSON，含 2,951 events、1,476 activations、26 tags；原书工业可视案例缺失 |

### 5.2 Chapter 2 三数据集指标

表内为三种子平均 `F1 / FAR / MAR`。

| 数据集 | IID delay | non-IID delay | deadband | APP |
|---|---:|---:|---:|---:|
| TEP Classic | 0.8678 / 0.0067 / 0.1840 | 0.7231 / 0.2190 / 0.1993 | 0.8480 / 0.0044 / 0.2049 | 0.5780 / 0.6472 / 0.1579 |
| PRONTO | 0.3011 / 0.1119 / 0.7905 | 0.3332 / 0.1214 / 0.7211 | 0.0479 / 0.1207 / 0.9730 | 0.2311 / 0.1161 / 0.8496 |
| SKAB | 0.1448 / 0.5949 / 0.0125 | 0.1093 / 0.8277 / 0.0130 | 0.3095 / 0.2283 / 0.0000 | 0.1545 / 0.5586 / 0.0000 |

### 5.3 Chapter 3 三数据集 F1

| 数据集 | Convex NOZ | Search-cone | Variation direction | Bayesian pump proxy | Mahalanobis baseline |
|---|---:|---:|---:|---:|---:|
| TEP Classic | 0.7210 | 0.6385 | 0.1455 | 0.4655 | 0.8921 |
| PRONTO | 0.6766 | 0.6661 | 0.0294 | 0.7037 | 0.7229 |
| SKAB | 0.0950 | 0.0939 | 0.0077 | 0.1271 | 0.1049 |

结果说明：书中专用几何/物理方法没有在通用代理数据上稳定超过 Mahalanobis。这里保留负结果，不能通过继续调参把代理迁移成绩描述成原论文复现。

### 5.4 Chapter 4 结构与根因结果

| 数据/方法 | 图激活 | NTE 显著边 | NDTE 剪枝 | 稳定性/根因结果 |
|---|---:|---:|---:|---|
| TEP Alarm NTE/NDTE | 1.0000 | 193/10 runs | 87 | 同类 direct-edge Jaccard 0.4550，跨类 0.0378 |
| NPP Alarm NTE/NDTE | 1.0000 | 525/22 runs | 201 | 同类 Jaccard 0.4432，跨类 0.0313 |
| FCC Alarm NTE/NDTE | 0.9375 | 394/32 runs | 182 | 同类 Jaccard 0.1709，跨类 0.0165 |
| IGTE/IGDTE TEP IDV1 | IGTE 1.0 | 平均 2.0 | 0 | Table 4.8 F1=0；published V1 root rank=2 |
| IGTE/IGDTE PRONTO | IGTE 0.6667 | 平均 1.3333 | 0 | 同 episode 跨 seed Jaccard 0.3148 |
| IGTE/IGDTE SKAB | IGTE 1.0 | 平均 3.8889 | 0 | 同 episode 跨 seed Jaccard 0.5417 |
| Recursive BN / EnAS | — | — | — | 5 行持久化后 ME/HE/UE 非空率均 1.0；无 tag-level accuracy |
| PLR / PIADE | — | — | — | 3 个时间折，129/300 窗口激活，平均 activation 0.43 |

### 5.5 Chapter 5 三数据集结果

| 数据集 | Criterion-C candidate rate | Alignment BA / set-Jaccard | Closed-pattern BA / baseline | compression | MaxEnt top1 / global-frequency |
|---|---:|---:|---:|---:|---:|
| TEP Alarm | 0.8000 | 0.7300 / 0.9500 | 0.9667 / 0.9667 | 0.000756 | 0.1109 / 0.1027 |
| NPP Alarm | 0.9364 | 0.1485 / 0.6061 | 0.7182 / 0.7182 | 0.1686 | 0.0881 / 0.4976 |
| FCC Alarm | 0.8653 | 0.7736 / 0.8806 | 0.9347 / 0.9347 | 0.1211 | 0.2209 / 0.0918 |

Criterion-C 没有专家区间，所以 candidate rate 只是描述性结果。Alignment 在三数据集均弱于集合 Jaccard；closed-pattern 分类与其 class-core 基线完全持平；MaxEnt 只在 TEP/FCC 的 top-1 上超过全局频率，NPP 明显失败。

## 6. SOTA 算法：论文、数据与实验结论

### 6.1 全部 10 个 SOTA 状态

| 算法 | 论文 DOI | 实际验证数据 | 当前结果 | 仍缺 |
|---|---|---|---|---|
| `casim_2024` / CASIM | `10.1017/dce.2024.22` | TEP/NPP/FCC；官方 CASIM TEP | 工程 BA 1.0000/0.8182/0.9922；作者 open-set 700/700 模型 fit 完成，P2 | full-threshold mean 数值差、Figure 14b-c、Docker、作者 wrapper 确认 |
| `cone_afc_2024` / ConE-AFC | `10.1109/ACCESS.2024.3492348` | TEP/NPP/FCC；官方 18,750 synthetic | 250/250 作者任务，95/95 论文表均值过容差；独立 conformal 层 1,800/1,800 完全一致，P2 | Docker、五种独立基础分类器、端到端 wrapper |
| `uncertainty_reduction_2025` / BiP | `10.1016/j.ifacol.2025.11.935` | TEP/NPP/FCC；官方 synthetic+TEP | 工程预测优于 median baseline；作者/切分/RNG 180/180 任务完成，P2 | 论文-v3 数值差、Docker 文件序、独立同折实现 |
| `etfa_robustness_2025` | `10.1109/ETFA65518.2025.11205709` | TEP/NPP/FCC | 五类扰动、两强度、三 prefix、三 seed 的 E2 验证完成 | ETFA 全文和原数据 |
| `afc_robustbench_2026` | `10.2139/ssrn.6999280` | TEP/NPP/FCC | prefix/severity/mixed corruption 与 normalized AUC 已完成 | preprint PDF/两套原始行业数据及论文表 |
| `cross_conformal_afc_2025` | `10.1109/ICPS65515.2025.11087828` | TEP/NPP/FCC | 三折、八 prefix 的 coverage-efficiency 验证完成 | 全文/capsule、精确 empty-set 后处理、原 synthetic 数据 |
| `modified_tfidf_afc_2025` | `10.1016/j.conengprac.2025.106485` | TEP/NPP/FCC | 1-4 gram、spectral clustering、100 epoch LSTM 完成 | VAM 数据、完整公式/超参、正常类不足时 KPCA 不适用 |
| `ctfh_fingerprinting_2025` | `10.1016/j.cherd.2025.11.026` | TEP/NPP/FCC；PRONTO mismatch | NPP 可胜 baseline，TEP/FCC 不胜；PRONTO 退化 | 全公式/参数/论文表及原 TEP 协议 |
| `structured_hdam_2026` | `10.1016/j.compchemeng.2026.109570` | TEP/NPP/FCC；PRONTO mismatch | TEP/FCC 高 BA，NPP 与 Jaccard 持平；PRONTO 退化 | 完整 template/alignment 公式、超参和论文表 |
| `hybrid_histogram_afc_2026` | `10.1016/j.engappai.2025.113705` | TEP/NPP/FCC | 三阶段训练均激活，但三数据 BA 偏低 | 完整 histogram/Transformer 公式、原 split、超参和论文分数 |

### 6.2 三数据集分类 Balanced Accuracy

三种子均值 ± 标准差；Jaccard 为支撑基线。

| 方法 | TEP Alarm | NPP Alarm | FCC Alarm |
|---|---:|---:|---:|
| Jaccard class-core | 0.9667 ± 0.0104 | 0.6932 ± 0.0341 | 0.9115 ± 0.0119 |
| CTFH | 0.7350 ± 0.0100 | **0.8371 ± 0.0365** | 0.3828 ± 0.0590 |
| Structured HDAM | 0.9967 ± 0.0029 | 0.6932 ± 0.0455 | 0.9375 ± 0.0156 |
| CASIM | **1.0000 ± 0.0000** | 0.8182 ± 0.0455 | **0.9922 ± 0.0078** |
| Modified TF-IDF | 0.8450 ± 0.1039 | 0.6250 ± 0.1311 | 0.9896 ± 0.0045 |
| Time histogram hybrid | 0.7017 ± 0.1056 | 0.2652 ± 0.0459 | 0.3698 ± 0.1398 |

说明：粗体表示该数据集表内最高方法，不表示论文原始排行榜。TEP/FCC 上 CASIM 最强；NPP 上 CTFH 最强。Time histogram 在 NPP/FCC 显著退化，Modified TF-IDF 在 TEP/NPP 有较大 seed 方差。

### 6.3 Robustness normalized AUC

| 方法 | TEP Alarm | NPP Alarm | FCC Alarm |
|---|---:|---:|---:|
| Jaccard class-core | 0.8750 | 0.6856 | 0.7234 |
| CTFH | 0.7967 | 0.6864 | 0.2922 |
| Structured HDAM | **1.0000** | 0.6621 | 0.7359 |
| CASIM | 0.9733 | **0.7379** | **0.8318** |
| Modified TF-IDF | 0.7783 | 0.5348 | 0.7151 |
| Time histogram hybrid | 0.6850 | 0.2159 | 0.2599 |

### 6.4 Conformal full-prefix结果

| 数据集 | ConE coverage / set size | Cross-Conformal coverage / set size | 判断 |
|---|---:|---:|---|
| TEP Alarm | 0.8167 / 1.0050 | 0.9617 / 1.2467 | Cross-Conformal coverage 更高，集合稍大 |
| NPP Alarm | 0.7689 / 1.0871 | 0.9735 / 1.5606 | ConE coverage 偏低；Cross-Conformal 更保守 |
| FCC Alarm | 0.9010 / 6.6745 | 0.9661 / 8.0703 | coverage 尚可，但集合很大，判别效率弱 |

### 6.5 Uncertainty Reduction 迁移结果

| 数据集 | MAE(min) | median baseline MAE(min) | Jackknife+ coverage | mean interval width(min) |
|---|---:|---:|---:|---:|
| TEP Alarm | 12.5743 | 16.0377 | 0.9032 | 68.1009 |
| NPP Alarm | 3.7065 | 4.0506 | 0.8666 | 14.8100 |
| FCC Alarm | 0.8197 | 2.3580 | 0.9230 | 4.8849 |

三数据 MAE 均优于 median baseline，但 NPP coverage 低于 0.9，TEP 区间很宽；这属于迁移结论，不等于 IFAC Tables 1-4 复现。

### 6.6 支撑性基线与非正式交付方法

除 30 个正式算法交付项外，仓库还提供固定阈值/延迟/死区、通用 `design_alarm`、Mahalanobis MSPC、基础 Transfer-Entropy ranker、简单洪泛检测、Smith-Waterman 相似度、经验 next-alarm 和 Jaccard class-core 等支撑方法。它们用于消融、机制对照或低复杂度基线，不单独计入“20+10”闭环账本。

- Mahalanobis：TEP Classic F1=0.8998、PRONTO F1=0.3978、SKAB F1=0.5442；SKAB FAR=0.8836，说明高 recall 来自严重过报。
- PIADE `EmpiricalNextAlarmPredictor`：top-1=0.1267、top-3=0.2881、vocabulary coverage=0.9813，属于工程基线。
- SMD10toWFGR Criterion-C：1,690 条 Alarm/Warning 事件中得到 4 个候选区间，分布在 3 个 device-day；无专家标签，不能作准确率结论。
- Jaccard class-core 是洪泛分类的重要低复杂度基线；它在 TEP/NPP/FCC 的 BA 分别为 0.9667/0.6932/0.9115，并揭示多种复杂方法没有稳定产生增益。

## 7. P0 论文精确验证

### 7.1 CASIM

- 作者默认闭集 mean BA：0.9938186813；排除重复测试轨迹后 0.9934981685，说明默认高分不是主要由两个重复组驱动。
- open-set 700/700 模型 fit 完成，10 个 seed、70 个 held-out-class/fold 组合。
- 本地最大 BA 0.9555196，论文 0.947，差值在 0.02 内。
- 本地最优 threshold 0.335，论文 0.324，不通过舍入门。
- 全 threshold mean BA 0.9032669，论文 0.879，差值 0.0242669，不通过 0.02 门。
- 状态：`P2_open_set_paper_compute_complete_numeric_and_protocol_gates_open`，不是 P3。

### 7.2 ConE-AFC

- 默认条件 5 folds、alpha=0.05、每类 calibration=22：命名表 9/10 行过容差。
- 完整 50 split × 5 model 的 250/250 作者任务完成。
- Tables I-II 共 95/95 个均值在 0.02 容差内，最大绝对均值差 0.0022710。
- 固定作者 MBW-LR 分数后，本仓库独立 conformal layer 在 50 折、51 prefix、3 alpha、3 calibration size 上得到 1,800/1,800 完全一致指标，最大差 0。
- 状态：作者表格和独立 conformal 子层已闭合；基础分类器、端到端 wrapper 和 Docker 尚未闭合，仍非 P3。

### 7.3 BiP / Uncertainty Reduction

- 180/180 lane-dataset-model-fold 任务完成。
- 每 lane 16 个论文表格条目：author overlap 6/16、paper disjoint 7/16、seeded author 7/16、seeded disjoint 8/16 通过。
- Python RNG 控制后仍有 10 个 CASIM 配对 bifurcation 不同；进一步同时重置 Numba RNG 后，CASIM 10/10 同折配对完全相同，最大差由 32 降为 0。
- 删除 calibration/RF overlap 没有统一收益：TEP 的 point MAE/width 小幅改善，synthetic 的两项变差；因此 overlap 只能解释部分差距。
- 状态：`P2_author_v3_and_deterministic_split_ablation_grids_complete_numeric_mismatch_retained`，仍非 P3。

## 8. 需要保留的负结果

1. PRONTO 洪泛表示错配：CTFH 在状态和 rising-edge 适配下均坍缩为单类预测，BA=0.25；HDAM 状态全窗口 BA=0.1010，activation 版本 BA=0.25。
2. 以退化 CTFH 分数构建 ConE/Cross-Conformal 时，PRONTO coverage=1.0，但平均集合大小固定为 4、singleton rate=0，完全没有判别效率。
3. Chapter 3 几何/物理方法在 TEP/PRONTO/SKAB 代理数据上没有稳定超过 Mahalanobis，说明方法依赖设备语义和原工况。
4. IGDTE 在受控链可工作，但在 21 个已取得 episode-seed 图上剪枝为 0；iMAKS 已知边也低于冻结 surrogate threshold。
5. Chapter 5 alignment 在 9/9 单元未胜集合 Jaccard；closed patterns 与 class-core baseline 持平；MaxEnt 的 eta surrogate 在 9/9 单元未达 0.8。
6. FCC 旧固定切分含精确 rising-edge 重复跨分区；按唯一轨迹重做后 alignment BA 从 0.8875 降至 0.7736。

这些结果表明，应优先匹配事件载荷、标签语义和原论文 split，而不是继续在错配数据上事后调参。

## 9. 完整性判断与剩余缺口

### 9.1 已经完整的部分

- 30/30 正式算法都有配置和调用入口；133 项测试通过。
- 11/11 数据集家族主载荷已取得，TEP/NPP/FCC 的数据先验、重复、分组切分均已有审计。
- 6/6 下游任务有真实或已取得数据入口。
- TEP/NPP/FCC 上 SOTA 分类、conformal、robustness、uncertainty 三种子实验完成。
- CASIM/ConE/BiP 三套官方 Capsule 的归档、代码和数据哈希完整。
- Xu 2012 命名公式/表格、ConE 作者 Tables I-II 和独立 conformal 层已形成可核查闭环。

### 9.2 尚不完整的部分

- 仍有 23/28 登记论文在 2026-08-28 access audit 中没有本地全文闭合；后续取得的文件需更新 audit 后再计数。
- 书籍 Chapter 2-5 的蒸汽压力、泵、冷凝器、热电报警流、389 个洪泛、921→207 pattern 数据库、26 条历史 TEP 洪泛等原始载荷缺失。
- CTFH、HDAM、Modified TF-IDF、Time Histogram、Cross-Conformal 等全文公式、超参数、原数据和论文表尚未完整取得。
- 本机无 Docker CLI，三套归档镜像尚未在权威容器中重跑。
- ConE 只独立闭合了 conformal layer；五个基础分类器和端到端 wrapper 待做。
- CASIM 仍有 threshold/平均 BA 数值差与 Figure 14b-c 缺口；BiP 仍有多数论文表格条目未过容差。
- T3 缺专家根因真值；T4 缺专家洪泛区间；当前 0 个正式 leaderboard-eligible split。
- `comopi` 主载荷存在，但适配器尚未实现。

## 10. 建议的下一轮优先级

1. 在有 Docker-compatible engine 的机器上原样复跑三个 Code Ocean 镜像，确认文件顺序、依赖和数值。
2. 将 ConE 独立同折验证扩展到五个基础分类器和完整 wrapper，争取首个 P3。
3. 对 CASIM 复现 Figure 14b-c，并向作者确认 held-out-class outer loop；对 BiP 对齐 v3 文件顺序和论文表格生成逻辑。
4. 优先补齐 TEP 专用事件载荷、VAM、泵/冷凝器和专家洪泛区间，停止在 PRONTO 代理窗口上调 CTFH/HDAM。
5. 完成 `comopi` adapter，冻结至少一个完全无重复、组级切分、带置信区间的正式 leaderboard protocol。

## 11. 复核命令与证据位置

```powershell
pytest -q
python scripts\paper_exact.py check --require-local --full-hash
python experiments\paper_harness\chapter2_multidataset\plot.py
python experiments\paper_harness\chapter3_multidataset\plot.py
python experiments\paper_harness\chapter4_gap_closure\plot.py
python experiments\paper_harness\chapter5_multidataset\plot.py
python experiments\paper_harness\sota_wave2_multidataset\plot.py
```

主要证据：

- `experiments/reports/ara_algorithm_validation.json`
- `experiments/reports/book_ch2_multidataset_validation.json`
- `experiments/reports/book_ch3_multidataset_validation.json`
- `experiments/reports/book_ch4_igte_igdte_multidataset_validation.json`
- `experiments/reports/book_ch4_gap_closure_validation.json`
- `experiments/reports/book_ch5_multidataset_validation.json`
- `experiments/reports/sota_wave2_multidataset_validation.json`
- `experiments/reports/p0_codeocean_data_prior.json`
- `paper_harness/paper_exact/status.v1.json`
- `docs/reports/p0_paper_exact_checkpoint_2026-08-31.md`
- `docs/reports/p0_bip_casim_numba_control_2026-08-31.md`
- `docs/reports/p0_cone_independent_same_fold_2026-08-31.md`

最终判断：算法集和数据工程主体已经落地，数据先验与多数据集实验覆盖较完整；但“全部论文完整复现”尚未完成。当前可以对外发布工程 benchmark 状态报告和负结果，不应发布为全部论文分数排行榜。
