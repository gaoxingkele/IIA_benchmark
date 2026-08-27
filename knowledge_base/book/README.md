# Book-to-benchmark map

源书：Jiandong Wang, Wenkai Hu, Tongwen Chen, *Intelligent Industrial Alarm Systems: Advanced Analysis and Design Methods*, Springer, 2024, DOI `10.1007/978-981-97-6516-4`。

`scripts/book/extract_book.py` 是本仓库的 book-to-skill 等价流水线：验证 433 页 PDF、按目录边界抽取六章、为每页写入 `PDF_PAGE` 标记、规范化 ligature，并保存源文件 SHA-256 和章元数据。它不做不可审计的自动摘要；本目录的六个 notes 是从证据文本再提炼出的“概念→数学→算法→任务→测试”层。

| 章 | printed pages | PDF physical pages | benchmark 产物 |
|---|---:|---:|---|
| 1 Overview | 1–47 | 13–59 | 任务分类与报警生命周期 |
| 2 Univariate | 49–127 | 60–138 | threshold/delay/deadband、FAR/MAR/AAD |
| 3 Multivariate | 129–220 | 139–230 | Mahalanobis、convex/nonconvex NOZ、动态阈值 |
| 4 Root cause | 221–301 | 231–311 | TE/IGTE、Bayesian network、贡献度排序 |
| 5 Alarm floods | 303–379 | 312–388 | 洪泛检测、对齐、模式挖掘、next alarm |
| 6 Visual analytics | 381–420 | 389–428 | KPI、bad actor、相关图、洪泛可视化 |

证据文本位于 `papers/extracted_text/book/`。重新提取后 manifest 哈希不变才可认为知识层仍对应同一版书。
