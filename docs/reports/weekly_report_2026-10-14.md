# 学术小组周例会汇报材料

- **汇报主题**：时间序列异常检测方法与数据集盘点及实验进展
- **日期**：2026-10-14
- **仓库**：IIA_benchmark（工业报警 / 时间序列异常检测基准）

---

## 一、本周工作概述

本周完成了仓库内单变量 / 多变量时间序列异常检测**方法与数据集的全面盘点**，梳理了三条研究线（工业报警主线、MTSAD 深度复现线、flow_matching 生成式检测子项目）的已有实验结果与证据等级，明确了当前最可靠的研究起点和存在的风险项。

## 二、方法盘点

### 2.1 单变量异常检测（对应书 Chapter 2，任务 T1：过程变量→二值报警）

| 方法 | 出处 | 实现位置 |
|---|---|---|
| 阈值 + on-delay + deadband 基线 | — | `src/iia_benchmark/models/univariate.py` |
| IID on/off delay timer（闭式 FAR/MAR/AAD） | Xu et al., 2012 | `models/univariate_book.py` |
| Non-IID delay timer / deadband 设计 | Wang et al., 2022 / 2023 | `models/univariate_book.py` |
| Alarm Probability Plot (APP) | Yu et al., 2017 | `models/univariate_book.py` |
| 自适应套件：ECDF 报警、Page-Hinkley 变点、工况条件化、路由+拒绝 | 本仓库适配层（B0–B7 消融） | `models/adaptive_univariate.py` |

### 2.2 多变量异常检测

**工业报警主线**（书 Chapter 3–4）：Mahalanobis MSPC、凸包正常运行区（NOZ）、搜索锥 NOZ、变化方向报警、贝叶斯时变回归、自适应多变量套件；另含因果/根因族（TE/NTE、IGTE、递归 BN、PLR 贡献度）。

**MTSAD 深度复现族**（`models/mtsad_detectors.py`，统一 Detector 接口）：

- 经典：PCA、kNN、Isolation Forest、OCSVM、Mahalanobis
- 深度：USAD、Anomaly Transformer、DCdetector、TimesNet、iTransformer、TranAD、CATCH、GCAD

**flow_matching 子项目**（进行中）：流匹配目标变体（independent / OT / rectified / SB / SF2M / VP / reflow）、GRASP、CFM-TS、MOMENT 零样本、MaelNet、CrossAD、BRITS/SAITS 等。

**报警洪泛分类**（报警序列任务）：CASIM、ConE、CTFH、HDAM、TF-IDF-LSTM、conformal 系列等 10 个 SOTA。

## 三、数据集盘点

### 3.1 多变量、逐点标签（核心基准，已下载并 SHA-256 固定）

| 数据集 | 规模 | 异常率 |
|---|---|---|
| SMD | 708,405 × 38 | 4.16% |
| MSL | 58,317 × 55 | 10.5% |
| SMAP | 135,183 × 25 | 12.8% |
| PSM | 132,481 × 25 | 27.76% |
| SWaT | 495,000 × 51 | 12.2% |

### 3.2 工业过程数据

- **TEP**：52 过程变量，44 run（2 正常 + 42 故障）
- **PRONTO**：17 过程变量 + 报警列，1.72 GB，CRC 校验通过
- **SKAB**：8 传感器，35 组实验，原生逐点标签（不做 point adjustment）

### 3.3 单变量数据

NAB、UCR 子集、MBA 心律（TranAD 镜像）；UCR Archive 2021 整包（184 MB）；GECCO 2018、SWAN-SF（约 6.5 GB）。

### 3.4 报警序列与其他

TEP / NPP / FCC 二值报警序列（洪泛分类用）；iMAKS（合成因果真值，仅 smoke）。**WADI、MSDS 仅登记未下载（需申请）**；4 个 synthetic 生成器仅用于管线验证。

## 四、实验结果进展

### 4.1 MTSAD 复现（本周重点，证据最充分）

124 个 run、32 个对比对：**5 matches / 9 near / 18 off**。

| 方法 | 复现结论 | 代表性数值（复现 vs 论文/第三方，point-adjusted F1） |
|---|---|---|
| **TimesNet** | **5/5 数据集全部吻合（Δ ≤ 0.03）** | SMD 0.8542 vs 0.8512；SWaT 0.9293 vs 0.9210 |
| Anomaly Transformer | 部分吻合 | SMD 0.9034 vs 0.9033（Δ 0.0001）；MSL/SWaT off |
| DCdetector | 4/5 接近 | SWaT 0.9518 vs 0.9633；MSL 缺口经 4 种子 + 200 分位阈值扫描确认为开放差异 |
| OCSVM | MSL 吻合 | 0.7015 vs 0.7082 |
| TranAD / USAD / GCAD | **复现失败** | GCAD AUROC 0.39–0.70 vs 论文 0.73–0.95 |

**方法学发现**：同一预测下 point-wise 与 point-adjusted F1 差 2.8–116 倍；affiliation 与 PA 两族指标差达 0.53。跨论文比较前必须对齐指标族、阈值规则与数据载荷版本。

### 4.2 工业报警主线

- 单变量 TEP：IID delay F1 0.8678 / FAR 0.0067 / MAR 0.1840；适配后双侧 ECDF **F1 0.9502**；SKAB 基线仅 0.1448（适配 B4 升至 0.4124，但 FAR 仍 55 次/小时）。
- 多变量 TEP：Mahalanobis F1 **0.8921** > 凸包 NOZ 0.7210 > 搜索锥 0.6385；PRONTO/SKAB 全面偏弱（负结果保留）。
- 报警洪泛 BA（TEP/NPP/FCC）：CASIM 1.0000 / 0.8182 / 0.9922；HDAM 0.9967 / 0.6932 / 0.9375；多种复杂方法不超 Jaccard 基线。
- 论文级验证：Xu 2012 达 **E4 级复现**（Table VII 最大误差 5.94e-05）；ConE-AFC conformal 层 1,800/1,800 完全一致。

### 4.3 flow_matching 子项目（进行中，不宜引用）

7,117 条登记指标中 5,341 条缺结果；严格协议（无 point adjustment、验证集 99 分位阈值）下 F1 百分数仅 3–41（如 rectified：SMD 41.07、SWaT 28.96、PSM 3.23）。多个全预算训练被中止重排队。

## 五、问题与风险

1. **排行榜为空**：leaderboard-eligible split 数量为 0，目前没有可对外发布的正式榜单，只有复现审计表。
2. **证据等级偏低**：30 个正式算法仅 1 个 E4、0 个 P3，多数停留在 E2/P2——"可调用 + 单测过"不等于论文复现。
3. **复现失败未闭环**：TranAD / USAD / GCAD 无法复现，DCdetector-MSL 缺口仍是开放差异。
4. **协议敏感性强**：point adjustment 可放大 F1 达 116 倍，结果表述必须绑定协议说明，否则易误导。
5. **合成结果纪律**：synthetic smoke 高分（如 F1 0.9950）只验证管线，严禁当作 benchmark 性能（AGENTS.md 明文规定）。

## 六、下周计划

1. 推进 MTSAD 复现缺口：排查 TranAD / USAD / GCAD 失败原因（载荷差异 or 实现差异），DCdetector-MSL 做归因实验。
2. 补齐 leaderboard-eligible split：分组切分 + 冻结超参 + 数据审计 + 多种子不确定性报告。
3. flow_matching 子项目：优先完成 730 个严格组合中已预定的重复实验（当前仅 210 个完成）。
4. 单变量方向：基于自适应 ECDF 套件在 UCR / NAB 上建立基线表。
5. 启动 WADI / MSDS 数据申请流程。

---

**一句话总结**：MTSAD 五基准 + TimesNet / Anomaly Transformer / DCdetector 复现管线是当前最可靠的研究起点；单变量方向有自适应 ECDF 套件 + UCR/NAB 数据可用；flow_matching 子项目尚需时间，暂不引用其结果。
