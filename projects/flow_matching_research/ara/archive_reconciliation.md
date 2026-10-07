# 两个用户压缩包核对

两包 ZIP CRC 及登记 SHA256 均通过。共 16 个 PDF 条目、15 个不同文件哈希、13 篇独立论文。原压缩包保留；导入以完整 SHA256 命名，重跑不会覆盖已有不同文件。

| 补充内容 | 核对结论 |
|---|---|
| MaelNet、SHCL、KGL、JFI、CFM-TS | 补齐此前五项全文访问缺口；作者代码缺口仍分别登记 |
| Pi-Transformer | 新增期刊16页版本，原 arXiv29页版本保留；数值引用按各自版本页码 |
| MOMENT long-context、SensitiveHUE | 新增可解析全文；后续 MOMENT 是预测/分类增强，不能直接当异常检测增强结果 |
| TimesNet | 用户 conference PDF 与既有文件哈希不同，保留两份；不据此推断算法发生变化 |
| KGL | 两包文件完全相同，按哈希去重；原文会议时间为2025，历史元数据2024不沿用 |
| SHCL | 两个文件哈希不同，归一化提取文本相同；图形及排版未据此宣称一致 |
| MaelNet | 两份哈希与归一化提取文本均不同，保留版本，不宣称完全相同 |
| SWaT 来源论文 | 文件实际是360页 CRITIS 2016 会议论文集；目标章节 PDF100–111、印刷88–99，DOI 10.1007/978-3-319-71368-7_8 |
| PSM、OmniAnomaly、USAD | 数据来源及竞争方法一并建 ARA 工程 |

新全文还确认 JFI 的 HP 数据来自私有炼化装置，66变量、六个月162907点；没有公开下载地址。本地 TEP 不能补上 HP 原数据复现。TEP 仍需与论文的具体 mode/fault/simulation 子集对齐。

历史审计保留。本轮补件报告只更新全文可获取性，不追认旧实验已经完成。素材详情见 benchmark 中 `papers/literature/flow_matching/user_supplied_manifest.json` 与 `docs/reports/flow_matching_user_papers_ara_2026-10-07.json`。
