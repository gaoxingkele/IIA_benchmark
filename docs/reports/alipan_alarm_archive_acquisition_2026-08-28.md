# 认证传输报警数据归档审计（2026-08-28）

## 结论

项目所有者提供的阿里云盘快传已通过已登录的官方桌面端下载到
`data/public_datasets/alipan_anomaly_archives/`。三个原始文件均保持原名，目录受
`.gitignore` 保护，不提交 Git。下载完成记录中的 `loaded`、`fullSize` 与最终文件字节数一致。

本轮解决了 TEP 与 NPP 专用报警载荷的“未取得”问题，但没有自动解决 adapter、数据切分、
模型调参与论文表格复现。现有 PRONTO 负结果仍然有效；后续应在本轮取得的五类 TEP 报警序列上
重新运行 CTFH、HDAM、CASIM、ConE/Cross-Conformal 等方法。

## 原始归档与哈希

| 文件 | 字节数 | SHA-256 | 审计结论 |
|---|---:|---|---|
| `TennseAlarmDataset.rar` | 16,983,510,811 | `be77ff57c24074ecefe3380070e27575cb726b668c7699776497083369189bdc` | RAR5，UnRAR 全量测试退出码 0；TEP 专用载荷 |
| `Industrial-Alarm-Datasets.rar` | 199,576,419 | `5ce297029e075d1c47f4af59cb0757bc6cdc0bf187489e31acc6b1d2e8daefd1` | UnRAR 全量测试退出码 0；NPP 报警载荷 |
| `FCC Alarm Dataset.zip` | 821,853 | `90cb3a411aaac527df32c69d62f1e0d146ec57f65d6e42f382924088982584b6` | ZIP 全量解码通过，但内容与文件名不符，禁止进入 FCC alarm 榜单 |

## TEP RAR5 内部结构

外层 RAR 以 `-m0` 存储四个 ZIP，四个成员总大小为 16,983,509,969 字节；不是填充文件或稀疏文件。

| 内层 ZIP | 字节数 | 目录审计 | 完整性范围 |
|---|---:|---|---|
| `1_Tests.zip` | 9,934,972,496 | 1,701 个成员；100 个 run；每个 run 含 `1_Original`、`2_Filter`、`3_Deadband`，每版含 `ALARMS/SIMOUT/TOUT/XMV.xlsx`；共 1,200 个 XLSX、100 个随机种子 | 外层 RAR CRC 通过；内层 ZIP 中央目录通过，未做 9.93 GB 内层全量解压 CRC |
| `2nd_Alarm_Dataset_5Classes.zip` | 2,874,462 | 1,000 个报警 CSV，`class_0`–`class_4` 各 200 个，另有 `ground_truth.xlsx`；每个 CSV 为 300 行、52 列，其中 50 个 HI/LO 二值报警位 | 内层全量解码/CRC 通过，解码 33,828,354 字节 |
| `2_Extracted Abnormal Situations.zip` | 6,980,781,537 | 4,052 个成员、3,200 个 XLSX；ARC、Normalization、组合三阶段；18 个 IDV/XMV 场景变体 | 外层 RAR CRC 通过；内层 ZIP 中央目录通过，未做 6.98 GB 内层全量解压 CRC |
| `3_Normal Steady-State (1).zip` | 64,881,474 | 3 个正常稳态 run；每个含 `SIMOUT/TOUT/XMV.xlsx` 和种子 | 内层全量解码/CRC 通过，解码 66,257,101 字节 |

大型内层 ZIP 仅临时展开用于读取中央目录，清点后删除临时副本；原始 RAR 始终保留。

## NPP RAR 内部结构

- 123,930 个归档成员，其中 122,510 个 CSV。
- 101 个顶层数字 run/组；故障族为 `FLB`、`LLB`、`LOCA`、`LOCAC`、`LR`、`MD`、
  `RI`、`RW`、`SGATR`、`SGBTR`、`SLBIC`、`SLBOC`，另有 `Normal`。
- 101 个 `AlarmLimits/AL_alpha*.csv` 阈值文件。
- 数据 CSV 以 `TIME` 加 194 个 LO/HI 二值报警位组织，10 秒采样抽样可读。
- UnRAR 全量归档测试退出码 0；另对 13 类代表 CSV 和全部 101 个阈值文件做了解码抽样，退出码 0。

## 误命名 FCC 候选包

嵌入的 `metadata.json` 将该包标识为 *Experimental results for sorption kinetics and equilibria
of CO2 on samples of Na-Y beads and Al-fumarate MOF powder*，实际 DOI 为
`10.60517/19027803-fec9-41f2-8a02-408cc176554e`，许可为 CC BY 4.0。它包含吸附动力学和
平衡吸附 ZIP，不是 FCC 工业报警数据。原包保留用于可追溯审计，但登记为
`fcc_alarm_candidate_rejected`，真正 FCC alarm DOI `10.60517/2v23vv393` 的载荷仍缺失。

## 下游任务与下一步

| 载荷 | 可支持任务 | 当前缺口 |
|---|---|---|
| TEP 五类报警 CSV | T4 五分类、prefix early classification、open-set、Conformal coverage/set size、CTFH/HDAM 指纹 | 实现 ZIP/RAR adapter、按原始实例分组切分、确认 `ground_truth.xlsx` 语义、冻结种子 |
| TEP Tests/异常工况 XLSX | T1/T2 报警处理比较，T3 根因候选，T4 洪泛/鲁棒性，T6 可视分析 | 选择性提取、工作簿 schema profile、官方 run/阶段标签映射 |
| NPP 报警 CSV | T3 根因/事故族识别、T4 跨域分类/open-set、序列扰动鲁棒性 | adapter、run/alpha 层级语义核对、跨 run grouped split |
| FCC 候选 | 仅数据质量审计 | 真正 FCC alarm payload 仍未取得 |

严格顺序应为：先实现只读 adapter 与数据卡，再生成 grouped split，随后重跑 CTFH/HDAM 与
ConE/Cross-Conformal 的代表性实验；在同数据、同 split、同指标下与论文表格对齐前，不把
“载荷已下载”写成“算法已完整复现”。
