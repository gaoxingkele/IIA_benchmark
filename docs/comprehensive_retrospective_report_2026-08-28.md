# IIA Benchmark 整体回顾报告

报告日期：2026-08-28（Asia/Shanghai）  
统计基线提交：`122530a620998ac8ae1f0953e38aff059da3d3a8`  
项目仓库：<https://github.com/gaoxingkele/IIA_benchmark>  
理论主线：Wang、Hu、Chen，*Intelligent Industrial Alarm Systems: Advanced Analysis and Design Methods*（2024），DOI [10.1007/978-981-97-6516-4](https://doi.org/10.1007/978-981-97-6516-4)

## 1. 执行摘要与当前结论

本项目已形成一个面向 Intelligent Industrial Alarm（IIA）的可运行、可审计 benchmark 骨架，覆盖“书籍章节拆解—数学/方法提取—算法实现—任务配置—数据获取—实验执行—论文 ARA 证据包—覆盖审计”的完整工程链路，并完成了三轮“数据集 → 引用文献 → 方法论提取 → 算法提取 → 相似/SOTA 论文 → 下一批数据集”扩展。

当前可确认的规模如下：

| 项目 | 当前数量 | 完备性结论 |
|---|---:|---|
| 可调用方法族 | 34 | 可运行、单元测试覆盖；不等于论文级完整复现 |
| 模型配置 | 31 | 已配置化 |
| 书籍交付项 | 20（19 个算法 + 1 个第 6 章可视分析套件） | 20/20 可调用，20 个均为 `partial`，0 个 `verified` |
| SOTA 交付项 | 10 | 10/10 可调用，10 个均为 `partial`，0 个 `verified` |
| 下游任务 | 6 | 6/6 均已有真实数据运行记录；T4 仅为代理验证 |
| 公共数据登记记录 | 10 条、7 个逻辑数据集族 | 4 个主载荷已取得，3 个仍受访问/站点限制 |
| 真实数据验证报告 | 8 | 工程验证有效，0 个满足正式排行榜闭环 |
| 登记论文 | 28 | 本地 PDF 4 篇，缺 24 篇 |
| ARA 文献工程包 | 28 | 28/28 本地验证通过，14 条唯一验证命令 |
| 全仓测试 | 93 | `93 passed`（2026-08-28 重新执行） |
| 严格论文分数闭环 | 0 | 当前不能宣称“全部算法完整复现”或“SOTA 已复现” |

最重要的判断是：**算法接口和主要机制已经落地，但算法集、论文集、数据集和实验集尚未达到论文同协议、同数据、同切分、同指标、同分数的严格完备状态。** 当前最成熟的成果是工程化可调用基线、真实数据适配、证据链和缺口审计；最主要的阻塞是 24 篇全文、TEP/NPP/FCC 报警载荷、官方 Code Ocean 工件以及论文级参考分数未闭环。

## 2. 方法体系与三轮扩展逻辑

整体方法体系由六条主线构成：

1. 报警性能与参数设计：阈值、延时器、死区，使用 FAR、MAR、AAD 等指标进行权衡优化。
2. 多变量正常运行区：Mahalanobis、凸包、非凸搜索锥、变化方向、时变回归与物理机理约束。
3. 因果与根因分析：NTE/NDTE、IGTE/IGDTE、递归贝叶斯网络、分段线性趋势和贡献度。
4. 报警洪泛分析：Criterion C 检测、局部/加速序列对齐、闭频繁模式、最大熵 next-alarm 预测。
5. 在线、早期、开集与不确定性分类：CASIM、ConE-AFC、cross-conformal、CTFH、modified TF-IDF、HDAM、时间编码直方图。
6. 鲁棒性与运维表达：缺失/虚警/时间扰动/检测延迟压力测试，以及 KPI、bad actor、关联图、时间线和洪泛序列可视分析。

三轮扩展形成了如下闭环：

| 轮次 | 起始数据 | 文献/方法推进 | 落地结果 | 下一数据节点 |
|---|---|---|---|---|
| Round 1 | TEP classic、PRONTO | 书第 2–5 章；报警设计、NOZ、因果、洪泛序列 | 经典透明基线、书籍算法清单和 T1–T5 雏形 | TEP Alarm DataPort |
| Round 2 | TEP Alarm 元数据、PIADE | CASIM、ConE-AFC；早期/开集/不确定性 | 分类与 conformal 接口、PIADE next-alarm 和可视分析 | NPP、FCC |
| Round 3 | PIADE、SKAB、NPP/FCC 元数据 | uncertainty reduction、AFC-RobustBench 及 2025–2026 方法 | 扰动矩阵、鲁棒性协议、更多 SOTA 可调用复现位 | DataPort 授权、跨工厂验证 |

此循环的价值是将“读书所得的算法”置于真实数据与新论文中反复校验；当前短板是循环中的最后一个“数据/论文闭环”仍被访问权限和官方工件可得性限制。

## 3. 书籍章节拆解与 benchmark 对照

书籍源文件共 433 个 PDF 物理页，SHA-256 为 `d002db25abf162a7afcd06dfe08953ee38e37e708ff425120cabf326fac2e4b0`。章节抽取记录位于 `papers/extracted_text/book/manifest.json`，知识拆解位于 `knowledge_base/book/`。

| 章 | 书页 / PDF 物理页 | 内容 | benchmark 对应 |
|---|---|---|---|
| 1 | 1–47 / 13–59 | 工业报警概念、报警过载、研究现状与问题 | 任务边界、数据合同、评价原则 |
| 2 | 49–127 / 60–138 | 单变量报警系统最优设计 | T1；IID/non-IID delay、deadband、APP |
| 3 | 129–220 / 139–230 | 多变量报警系统最优设计 | T1/T2；凸/非凸 NOZ、变化方向、泵、凝汽器 |
| 4 | 221–301 / 231–311 | 报警事件根因分析 | T3；NTE/NDTE、IGTE/IGDTE、BN、PLR |
| 5 | 303–379 / 312–388 | 工业报警洪泛分析 | T4/T5；检测、相似性、模式挖掘、预测 |
| 6 | 381–420 / 389–428 | 报警可视分析及应用 | T6；KPI、关联、时间线、洪泛可视报告 |

## 4. 书中全部交付项：算法—章节—论文—落地状态

状态含义：`partial` 表示存在可调用实现和本地测试，但仍缺至少一项书中方程/优化细节、原始工业数据、论文协议或参考分数；只有这些证据全部闭环后才能标为 `verified`。

| 章节 | 方法/算法 | 主要实现 | 对照论文 | 状态与主要缺口 |
|---|---|---|---|---|
| 2.1 | IID 阈值与延时器 FAR/MAR/AAD 设计 | `iid_delay_timer_performance` | Xu et al. 2012，[10.1109/TASE.2011.2176490](https://doi.org/10.1109/TASE.2011.2176490) | partial；缺论文工业分数闭环 |
| 2.2 | 非 IID 报警持续时间/间隔 PMF 延时器设计 | `design_non_iid_delay_timer` | Wang et al. 2022，[10.1016/j.jprocont.2022.01.002](https://doi.org/10.1016/j.jprocont.2022.01.002) | partial；全文和工业数据缺失 |
| 2.3 | 最大幅值偏差 + 贝叶斯死区设计 | `deadband_index`、`design_deadband_width` | Wang et al. 2023，[10.1109/TCST.2023.3240020](https://doi.org/10.1109/TCST.2023.3240020) | partial；论文精确协议/分数缺失 |
| 2.4 | Alarm Probability Plot 阈值设计 | `build_alarm_probability_plot`、`select_alarm_probability_threshold` | Yu et al. 2017，[10.1109/TIE.2017.2682783](https://doi.org/10.1109/TIE.2017.2682783) | partial；量化/目标的论文级复测缺失 |
| 3.1.2 | 凸包正常运行区 | `ConvexHullNOZAlarm` | Yu et al. 2020，[10.1109/TCST.2019.2943469](https://doi.org/10.1109/TCST.2019.2943469) | partial；稳健正常点筛选与原表缺失 |
| 3.1.3 | 搜索锥非凸正常运行区 | `SearchConeNOZAlarm` | Wang et al. 2024，[10.1109/TASE.2022.3222413](https://doi.org/10.1109/TASE.2022.3222413) | partial；精确 cone-facet 调参和论文分数缺失 |
| 3.2 | 在线变化方向多变量报警 | `AdaptiveTimeGradient`、`VariationDirectionAlarm` | Chen et al. 2017，[10.1016/j.cherd.2017.04.011](https://doi.org/10.1016/j.cherd.2017.04.011) | partial；工业规则矩阵和论文分数缺失 |
| 3.3 | 电泵时变回归 + 贝叶斯滤波报警 | `BayesianWindowRegressionAlarm` | Xiong et al. 2018，[10.1109/TII.2017.2749332](https://doi.org/10.1109/TII.2017.2749332) | partial；原始泵数据/分数缺失 |
| 3.4 | 物理引导凝汽器 NOZ | `CondenserPhysicalModel`、`CondenserNOZAlarm` | Wang et al. 2024，[10.1109/TCST.2024.3370036](https://doi.org/10.1109/TCST.2024.3370036) | partial；最坏 FAR/MAR 贝叶斯界及工业分数缺失 |
| 4.1 | 二值报警 NTE | `NormalizedTransferEntropyGraph` | Hu et al. 2017，[10.1016/j.conengprac.2017.04.012](https://doi.org/10.1016/j.conengprac.2017.04.012) | partial；工业报警标签与参考图缺失 |
| 4.1 | 二值报警 NDTE | `normalized_direct_transfer_entropy` | 同上 | partial；间接边剔除和论文图表闭环缺失 |
| 4.2 | 信息粒化 IGTE | `information_granulation_transfer_entropy` | Zhang et al. 2022，[10.3390/e24020212](https://doi.org/10.3390/e24020212) | partial；两水箱/工业案例分数未复测 |
| 4.2 | 信息粒化 IGDTE | `information_granulation_direct_transfer_entropy` | Zhang et al. 2023，[10.1016/j.conengprac.2023.105669](https://doi.org/10.1016/j.conengprac.2023.105669) | partial；全文、TEP 协议和分数缺失 |
| 4.3 | 递归贝叶斯网络报警根因 | `RecursiveBayesianAlarmRCA` | Wang et al. 2018，[10.1016/j.ijepes.2018.05.029](https://doi.org/10.1016/j.ijepes.2018.05.029) | partial；热电厂数据/根因真值缺失 |
| 4.4 | 分段线性趋势 + 非负回归根因贡献 | `PLRContributionRCA` | Hu et al. 2022，[10.1016/j.compchemeng.2022.107813](https://doi.org/10.1016/j.compchemeng.2022.107813) | partial；工业时变因果分数缺失 |
| 5.1 | Criterion C 在线/离线洪泛检测 | `criterion_c_alarm_flood_detection` | Wang et al. 2018，[10.1109/TCST.2017.2723578](https://doi.org/10.1109/TCST.2017.2723578) | partial；缺专家洪泛区间与年度工业复测 |
| 5.2 | 优先级/时间感知 BLAST-like 对齐 | `accelerated_alarm_alignment` | Hu et al. 2016，[10.1016/j.conengprac.2016.05.021](https://doi.org/10.1016/j.conengprac.2016.05.021) | partial；原工业序列、精度/加速表缺失 |
| 5.3 | CHARM 闭模式与代表模式聚类 | `charm_closed_alarm_patterns`、`representative_alarm_patterns` | Hu et al. 2018，[10.1109/TIE.2018.2795573](https://doi.org/10.1109/TIE.2018.2795573) | partial；921→207 等论文结果未复现 |
| 5.4 | 最大熵 next-alarm 预测 | `MaximumEntropyNextAlarmPredictor` | Xu et al. 2021，[10.1016/j.jprocont.2021.10.002](https://doi.org/10.1016/j.jprocont.2021.10.002) | partial；原 Monte Carlo/工业分数缺失 |
| 6 | 报警可视分析验证套件 | `build_alarm_visual_analytics`、`export_alarm_visual_report` | 书第 6 章 | partial；原工业案例的图/事实一致性未闭环 |

结论：书中 19 个算法和第 6 章套件均已提供可调用入口，没有 `missing` 条目；但 20 个交付项没有任何一个达到严格 `verified`。

## 5. SOTA 算法层：论文—方法—落地状态

| SOTA 方法 | 论文 | 主要实现 | 当前缺口 |
|---|---|---|---|
| CASIM MultiRocket + ridge + LoOP | Faulwasser et al. 2024，[10.1017/dce.2024.22](https://doi.org/10.1017/dce.2024.22) | `CASIMClassifier` | Code Ocean、TEP Alarm 载荷、论文分数 |
| ConE-AFC 类别/步骤 conformal prediction | Faulwasser et al. 2024，[10.1109/ACCESS.2024.3492348](https://doi.org/10.1109/ACCESS.2024.3492348) | `ConEAlarmFloodClassifier` | Code Ocean、合成原数据、Tables I–II |
| 不确定性下降预测 | Faulwasser et al. 2025，[10.1016/j.ifacol.2025.11.935](https://doi.org/10.1016/j.ifacol.2025.11.935) | `UncertaintyReductionForecaster` | 全文、完整特征定义、两套评测数据 |
| 报警数据质量鲁棒性 | Faulwasser et al. 2025，[10.1109/ETFA65518.2025.11205709](https://doi.org/10.1109/ETFA65518.2025.11205709) | `run_afc_robustness_benchmark` | ETFA 全文和原实验数据 |
| AFC-RobustBench | Faulwasser et al. 2026，[10.2139/ssrn.6999280](https://doi.org/10.2139/ssrn.6999280) | `run_afc_robustness_benchmark` | SSRN PDF 被 403 阻断、两套过程工业数据 |
| Cross-conformal AFC | Manca et al. 2025，[10.1109/ICPS65515.2025.11087828](https://doi.org/10.1109/ICPS65515.2025.11087828) | `CrossConformalAlarmFloodClassifier` | 全文/工件、精确空集策略、原合成数据和分数 |
| Modified TF-IDF + KPCA + LSTM | Rahaman et al. 2025，[10.1016/j.conengprac.2025.106485](https://doi.org/10.1016/j.conengprac.2025.106485) | `TFIDFLSTMAlarmFloodClassifier` 等 | 公式/超参、VAM 数据、参考分数 |
| AEM/CTFH 确定性指纹 | Rao et al. 2025，[10.1016/j.cherd.2025.11.026](https://doi.org/10.1016/j.cherd.2025.11.026) | `CTFHAlarmFloodClassifier` | 精确峰值/哈希参数、论文表、TEP Alarm 数据 |
| 最优时间编码直方图混合网络 | Najafi et al. 2026，[10.1016/j.engappai.2025.113705](https://doi.org/10.1016/j.engappai.2025.113705) | `OptimalTimeEncodedHistogramClassifier` | 完整公式/超参、TEP Alarm 切分和论文分数 |
| Structured HDAM 模板与二维匹配 | Rahimi et al. 2026，[10.1016/j.compchemeng.2026.109570](https://doi.org/10.1016/j.compchemeng.2026.109570) | `HDAMTemplateMatcher` | 模板/对齐精确公式、超参/表、TEP Alarm 数据 |

这些实现均通过了机制/不变量测试，但不能替代作者工件和原协议复测，因此 10/10 均保持 `partial`。

## 6. 下游任务支持情况

| ID | 下游任务 | 输入/输出 | 已有真实数据 | 当前实验覆盖 | 完备性 |
|---|---|---|---|---|---|
| T1 | 报警生成与设计 | 连续过程量 → binary alarm、FAR、MAR、AAD | TEP、SKAB、PRONTO | Mahalanobis/阈值派生报警验证 | 可运行；未形成统一排行榜 |
| T2 | 多变量动态报警限 | 多变量状态 → NOZ、dynamic limit、alarm state | TEP、SKAB、PRONTO | Mahalanobis、PRONTO search-cone | 可运行；跨模型同切分比较不足 |
| T3 | 因果图与根因排序 | 报警/过程序列 → directed edges、root ranking | TEP；PRONTO 可接入 | TEP D01/D04/D11 NTE 结构验证 | 可运行；缺根因真值精度 |
| T4 | 洪泛检测、聚类与分类 | 报警事件/窗口 → flood interval、class、unknown、prediction set | PRONTO 代理；TEP/NPP/FCC 受限 | PRONTO fault-window CASIM + Criterion C | 仅 `runnable_real_data_surrogate`；非专家洪泛标签 |
| T5 | next-alarm 与洪泛预测 | 报警前缀 → next tag、future set、early warning | PIADE、PRONTO | PIADE next distinct alarm | 可运行；尚无论文同协议早期预测榜 |
| T6 | 运维可视分析 | 统一 AlarmEvent/Episode → KPI、bad actor、关联图、洪泛报告 | PIADE、PRONTO | PIADE 30 日 HTML/JSON 工件 | 可运行；原书工业案例未复刻 |

六个任务均有真实数据执行记录，但只有“执行覆盖”，没有任何任务已经满足正式 benchmark 的全部公平比较门槛。T4 尤其需要 TEP/NPP/FCC 专用报警洪泛载荷与专家标签。

## 7. 数据集总表与验证状态

10 条下载登记记录归并为 7 个数据集族；PRONTO 包含 README、技术报告和完整载荷 3 条记录，PIADE 包含 raw 与 hourly sequences 2 条记录。

| 数据集族 | 当前状态 | 本地画像/用途 | 任务 | 来源 |
|---|---|---|---|---|
| `tep_classic` | 主载荷已下载并记录 Git revision | 44 个 run 文件：2 normal、42 fault；52 特征 | T1/T2/T3 | [GitHub](https://github.com/jkitchin/tennessee-eastman-profbraatz)，[10.1016/0098-1354(93)80018-I](https://doi.org/10.1016/0098-1354(93)80018-I) |
| `pronto` | 1,720,560,937 字节完整 ZIP 已下载；MD5 与全 ZIP CRC 通过；安全抽取官方 aligned/labelled 子集 | 3 个测试日、45,420 个 1 Hz 样本、13 个报警 tag、17 个过程特征；Normal 11,899，四类故障共 33,521 | T1–T6；T4 仅代理 | [Zenodo 1341583](https://zenodo.org/records/1341583)，CC BY 4.0 |
| `piade` | 两个 CSV 已下载、MD5 通过 | raw 429,394 行、5 台设备、92,084 个非哨兵报警行；小时表 23,376×164 | T5/T6；可扩展 T4 | [Zenodo 7071747](https://zenodo.org/records/7071747) |
| `skab` | Git 数据已下载并记录 revision | 35 个水循环异常实验 CSV | T1/T2/鲁棒性 | [SKAB](https://github.com/waico/SKAB)，[10.1007/s41060-022-00355-4](https://doi.org/10.1007/s41060-022-00355-4) |
| `tep_alarm_dataport` | 仅 landing metadata；主载荷需 IEEE DataPort 登录/接受条款 | 论文所需过程报警洪泛、开集/前缀分类主数据；登记文件约 61.88 MB、6.5 GB、9.25 GB | T3/T4/T6 | [IEEE DataPort](https://ieee-dataport.org/open-access/tennessee-eastman-process-alarm-management-dataset)，[10.21227/326k-qr90](https://doi.org/10.21227/326k-qr90) |
| `npp_alarm_dataport` | 仅 landing metadata；主载荷需登录/接受条款 | 核电报警跨域数据 | T3/T4 | [IEEE DataPort](https://ieee-dataport.org/open-access/nuclear-power-plant-alarm-dataset)，[10.21227/g2fa-9y43](https://doi.org/10.21227/g2fa-9y43) |
| `fcc_alarm` | DOI 已登记；本次站点重定向不稳定，landing/payload 均缺 | FCC 报警跨装置验证 | T3/T4 | [10.60517/2v23vv393](https://doi.org/10.60517/2v23vv393) |

另外有 4 个合成 smoke 数据集，用于单变量、非凸 NOZ、根因和洪泛相似性代码路径验证；它们不能进入正式排行榜。

## 8. 已执行真实数据实验及实际效果

以下数值直接来自 `experiments/reports/*.json`。它们是带配置和数据哈希的工程验证，不是论文或排行榜成绩。

| 数据/方法 | 任务 | 实际结果 | 解释与限制 |
|---|---|---|---|
| SKAB / Mahalanobis | T1/T2 | Precision 0.3955，Recall 0.9710，F1 0.5442，FAR 0.8836，MAR 0.0290，AAD 0.0882 | 高召回但虚警极高，说明单一椭球正常区不适合该异构工况 |
| TEP / Mahalanobis | T1/T2 | Precision 0.9463，Recall 0.8806，F1 0.8998，FAR 0.2092，MAR 0.1194，AAD 1.9048 | 当前效果最好，但仍是工程切分，非论文同协议榜单 |
| PRONTO / Mahalanobis | T1/T2 | Precision 0.9823，Recall 0.2609，F1 0.3978，FAR 0.0082，MAR 0.7391，AAD 1546.19 | 极保守：虚警低、漏报和延迟高；源测试日复用，不可进榜 |
| PRONTO / Search-cone NOZ | T1/T2 | 10° 下 2,901 个 cones；Precision 0.8049，Recall 0.4718，F1 0.5767，FAR 0.4782，MAR 0.5282，AAD 391.11 | 相比 PRONTO Mahalanobis 提高召回/F1并降低延迟，但 FAR 明显升高 |
| TEP / normalized TE | T3 | D01、D04、D11 共 3 个因果图；候选根节点分别以 `XMEAS_01/XMV_03`、`XMEAS_13`、`XMEAS_07` 居前 | 只验证图结构生成和候选排序；不能解释为故障根因准确率 |
| PRONTO / CASIM fault-window | T4 代理 | 60 train / 38 test；Accuracy 0.3421，Balanced Accuracy 0.2520，Macro-F1 0.1944 | 标签是故障工况而非专家确认洪泛；闭集、同源日、非论文协议 |
| PRONTO / Criterion C | T4 代理 | 3 日最大 attention tag 数为 3/4/4；按书中默认阈值发现 0 个 confirmed flood interval | 不能据此断言无洪泛；数据和参数缺少专家洪泛标注校准 |
| PIADE / empirical next alarm | T5 | 1,551 train windows、669 test windows、4,294 个转移；Top-1 0.1267，Top-3 0.2881，词表覆盖 0.9813 | 可作为真实设备日志起点；尚非书中最大熵论文同协议结果 |
| PIADE / visual analytics | T6 | `s_1` 前 30 日：2,951 事件、1,476 activations、26 tags、1 个检测洪泛；输出 HTML+JSON | 描述性验证，不是分类分数；证明第 6 章工件链可运行 |

辅助 synthetic smoke 结果如下，仅证明管线和已知不变量：单变量 F1 0.9950；凸 NOZ F1 0.9852；合成根因 `ROOT` 在 lag 3 排第一；10 个合成洪泛的留一 1-NN 相似分类准确率 1.0。

## 9. 论文获取现状与缺失论文

28 篇论文已全部登记 DOI、角色、实现映射和 ARA 验证；本地实际获得 4 篇 PDF：

1. Xu et al. 2012 FAR/MAR/AAD 作者稿。
2. Zhang et al. 2022 IGTE 开放获取论文。
3. Faulwasser et al. 2024 CASIM 开放获取论文。
4. Faulwasser et al. 2024 ConE-AFC Ruhr OPUS 作者稿。

其余 24 篇并未下载成功：

- 23 篇记录为 `not_openly_downloadable`，包括书第 2–5 章对应的 16 篇和 SOTA 的 7 篇。
- 其中 7 篇书籍对照论文发现 ResearchGate 作者全文候选，但需要已登录浏览器人工获取：deadband、APP、convex NOZ、search-cone NOZ、variation direction、electrical pumps、condenser NOZ。
- AFC-RobustBench 预印本 1 篇为 `download_failed`，SSRN 下载端点返回 HTTP 403。
- uncertainty-reduction 论文虽被标记为开放/complimentary，但远程全文仍受阻。
- 其余论文需要学校机构订阅、作者索取或合法开放副本；项目不会绕过登录、订阅或站点条款。

论文全文的必要性不只是“留档”：它决定方程、伪代码分支、特征定义、超参数、数据切分、随机种子、指标口径和参考表的精确性。没有全文时只能实现经书籍/元数据确认的机制，不能判定为论文完整复现。

## 10. 缺失数据、缺失实验及必要性

| 优先级 | 缺失项 | 为什么必要 | 闭环条件 |
|---|---|---|---|
| P0 | TEP Alarm DataPort 主载荷 | CASIM、ConE-AFC、CTFH、HDAM、时间直方图等 SOTA 的主要论文数据；也是 T4 专用报警洪泛协议的核心 | 授权下载、checksum、adapter、官方 run/类别映射、grouped split |
| P0 | NPP Alarm DataPort 主载荷 | 验证方法能否从化工仿真迁移到核电报警，并支撑 T3/T4 跨域结论 | 授权载荷、标签说明、设备/工况分组切分 |
| P0 | FCC Alarm 主载荷 | 防止 benchmark 只覆盖 TEP/PRONTO，提供真实炼化跨装置测试 | 稳定来源、许可、checksum、事件适配器 |
| P0 | CASIM/ConE-AFC Code Ocean 工件 | 锁定作者环境、预处理、随机种子和表格生成逻辑 | 合法取得 capsule，冻结依赖并复跑参考表 |
| P0 | 24 篇缺失全文 | 确保方程、算法分支、超参与分数口径完整 | 合法 PDF、SHA-256、页码证据、ARA 更新 |
| P0 | 统一 leaderboard-eligible split | 目前 0 个正式榜单切分，现有数据复用/代理标签会造成泄漏或不可比 | 按 run/设备/工厂分组；训练期定超参；测试冻结；版本化 split |
| P1 | 书第 2–5 章逐论文分数复测 | 这是 20 个书籍交付项从 `partial` 到 `verified` 的关键 | 同数据/切分/指标/容差重现代表表或图 |
| P1 | 10 个 SOTA 同协议横评 | 当前只能比较实现机制，无法回答谁是 SOTA | 同一事件合同、同一 split、prefix/open-set/coverage/鲁棒性指标 |
| P1 | T3 根因真值实验 | 当前 TEP 只输出候选图，没有 ground-truth root ranking accuracy | 带故障注入源/根因标签；Top-1/Top-3/MRR/edge precision-recall |
| P1 | T4 专家洪泛区间、开集和前缀实验 | PRONTO fault label 不是 flood label；闭集窗口无法验证 unknown 与在线早期能力 | 专家 episode 边界、留一类别 open-set、10–100% prefix 曲线 |
| P1 | 多随机种子和置信区间 | 单次结果可能受初始化、样本和扰动随机性影响 | 固定 seeds；均值、标准差、95% bootstrap CI |
| P1 | 鲁棒性矩阵 | 工业日志普遍有 missing/spurious/jitter/delay，干净数据分数不足以判断可用性 | 多强度、多 seed、clean degradation、worst group、robustness AUC |
| P2 | leave-one-site/device-out 跨域实验 | 衡量算法是否只记住单一装置模式 | TEP/PRONTO/NPP/FCC 或 PIADE 设备级留一 |
| P2 | 运行资源报告 | 深度/核方法与传统方法的工业部署代价差异显著 | CPU/GPU、训练/推理时间、峰值内存、参数量 |
| P2 | 第 6 章原工业图表复刻 | 保证可视化不仅能生成，而且与书中分析事实一致 | 原案例数据或等价公开案例、图表数值核对和导出回归测试 |

## 11. ARA 文献工程化验证

每篇登记论文在 `papers/literature/ara/<paper_id>/` 下具有一致的 ARA 结构：

- `metadata.json` 和 `PAPER.md`：标识、DOI、角色与状态。
- `logic/`：problem、concepts、claims、related work、experiments、solution/method、constraints。
- `src/`：代码映射、配置说明和运行环境。
- `evidence/`：来源、表/图占位、带命令的本地验证记录。
- `trace/exploration_tree.json`：探索与证据路径。

最新验证结果为 28/28 paper package 通过，覆盖 14 条唯一命令。该结果表示“登记的本地实现/测试可运行”，不表示 28 篇论文的原始分数已复现。严格证据状态以 `configs/algorithms/*.json`、`papers/literature/download_manifest.json` 和 `docs/status_audit.json` 为准。

## 12. 项目目录结构

```text
IIA_benchmark/
├─ configs/                    # 唯一配置事实源
│  ├─ algorithms/              # 书籍 20 项与 SOTA 10 项状态清单
│  ├─ datasets/                # 公共/合成数据登记
│  ├─ experiments/             # 冻结实验配置
│  ├─ metrics/                 # 各任务指标定义
│  ├─ models/                  # 31 个模型配置
│  ├─ splits/                  # 分组/时间切分
│  ├─ systems/                 # 工业系统定义
│  └─ tasks/                   # T1–T6 合同
├─ data/public_datasets/       # 下载数据、审计、画像和安全抽取结果
├─ docs/                       # 范围、矩阵、协议、路线图、审计与本报告
├─ experiments/
│  ├─ reports/                 # 8 个真实数据报告 + ARA 验证汇总
│  └─ runs/                    # 运行工件（HTML/JSON 等）
├─ knowledge_base/
│  ├─ book/                    # 六章知识、算法伪代码和任务映射
│  ├─ literature/              # 文献地图
│  └─ datasets/                # 数据知识卡
├─ papers/
│  ├─ extracted_text/book/     # 书籍按章、物理页和 hash 的证据层
│  └─ literature/              # 28 篇登记、4 个本地 PDF、28 个 ARA 工程包
├─ scripts/
│  ├─ book/                    # 章节抽取
│  ├─ data_acquisition/        # 下载、checksum、PRONTO 安全抽取/画像
│  └─ literature/              # OA 发现、下载、ARA 构建与验证
├─ src/iia_benchmark/
│  ├─ config/                  # 配置加载
│  ├─ data/                    # 统一数据对象和 adapters
│  ├─ evaluation/              # 指标与鲁棒性协议
│  ├─ models/                  # 经典、书籍与 SOTA 可调用实现
│  ├─ tasks/                   # 任务执行逻辑
│  └─ visualization/           # 第 6 章可视分析
└─ tests/                      # 93 个当前通过测试
```

数据原始载荷和论文 PDF 按项目策略通常只保存在本地；Git 跟踪其元数据、来源和哈希，不把大文件或受许可约束的内容直接提交到公开仓库。

## 13. 安装与使用说明

### 13.1 环境安装

要求 Python 3.10 或以上。PowerShell 中执行：

```powershell
git clone https://github.com/gaoxingkele/IIA_benchmark.git
Set-Location IIA_benchmark
python -m pip install -e ".[test,ml,deep]"
```

仅运行经典基线和测试时可使用：

```powershell
python -m pip install -e ".[test]"
```

### 13.2 书籍、论文与 ARA

```powershell
python scripts/book/extract_book.py
python scripts/literature/discover_open_access.py
python scripts/literature/download_representative_papers.py
python scripts/literature/build_ara_collection.py
python scripts/literature/verify_ara_collection.py
python scripts/literature/run_ara_validations.py
```

### 13.3 数据下载与审计

```powershell
# 下载默认、无需交互授权的登记数据
python scripts/data_acquisition/download_public_datasets.py

# 显式下载 PRONTO 1.72 GB 完整包
python scripts/data_acquisition/download_public_datasets.py --dataset pronto_full

# 校验、审计、安全抽取和画像
python scripts/data_acquisition/audit_public_datasets.py
python scripts/data_acquisition/audit_pronto_archive.py
python scripts/data_acquisition/extract_pronto_subset.py
python scripts/data_acquisition/profile_public_datasets.py
```

DataPort 数据必须在合法登录并接受条款后由授权人员下载，再放入登记目录并补充 checksum；脚本不会绕过访问控制。

### 13.4 执行实验

统一入口为：

```powershell
python -m iia_benchmark.runner <experiment-config.json>
```

代表性真实数据命令：

```powershell
python -m iia_benchmark.runner configs/experiments/tep_mahalanobis_validation.json
python -m iia_benchmark.runner configs/experiments/skab_mahalanobis_validation.json
python -m iia_benchmark.runner configs/experiments/tep_normalized_te_causal_validation.json
python -m iia_benchmark.runner configs/experiments/pronto_mahalanobis_validation.json
python -m iia_benchmark.runner configs/experiments/pronto_search_cone_validation.json
python -m iia_benchmark.runner configs/experiments/pronto_casim_fault_classification_validation.json
python -m iia_benchmark.runner configs/experiments/piade_next_alarm_validation.json
python -m iia_benchmark.runner configs/experiments/piade_visual_analytics_validation.json
```

### 13.5 全量校验与覆盖审计

```powershell
python scripts/validate_scaffold.py
python scripts/audit_benchmark_coverage.py
python -m pytest -q
```

新增算法必须同时提供配置、可调用实现或明确缺失状态、测试、文献/数据引用与 ARA 记录；新增榜单成绩还必须提供 grouped split、冻结超参、数据审计、随机种子和不确定性/鲁棒性报告。

## 14. 建议的下一次迭代顺序

1. 由有权限的成员获取 TEP/NPP DataPort 载荷，并补齐 FCC 稳定下载；这是 T4 和多数 SOTA 闭环的最高杠杆步骤。
2. 通过机构订阅、作者索取或已登录的合法作者副本渠道补齐 24 篇全文，同时补充 PDF hash 和页码证据。
3. 获取 CASIM、ConE-AFC 官方 Code Ocean 工件，锁定环境并复跑代表表格。
4. 发布第一个 leaderboard v0：优先 TEP/PRONTO 的 T1/T2，建立无源日复用的 grouped split、固定 seeds 和 95% CI。
5. 发布 T4 专项榜：必须使用专家洪泛 episode、open-set 留一类别、10–100% prefix 和鲁棒性矩阵。
6. 逐项推动书籍 20 项与 SOTA 10 项从 `partial` 升级为 `verified`；每次升级都应在同一提交中加入实现、测试、配置、引用和参考分数证据。

## 15. 可审计事实入口

- 算法状态：`configs/algorithms/book_algorithms.json`、`configs/algorithms/sota_algorithms.json`
- 下游任务：`configs/tasks/downstream_tasks.json`
- 数据登记/本机状态：`configs/datasets/public_sources.json`、`data/public_datasets/audit.json`、`data/public_datasets/profile.json`
- 文献登记/下载：`papers/literature/registry.json`、`papers/literature/download_manifest.json`、`papers/literature/access_audit.json`
- 实验结果：`experiments/reports/`
- 总体覆盖：`docs/status_audit.json`、`docs/status_audit.md`
- 评估边界：`docs/evaluation_protocol.md`
- 完整复现路线：`docs/full_reproduction_roadmap.md`

本报告中的“已实现”“已验证”“已复现”严格分层：可调用和单元测试通过只代表工程机制成立；只有论文全文、作者/官方数据与工件、同协议切分、同指标和参考分数全部闭环，才称为完整复现。
