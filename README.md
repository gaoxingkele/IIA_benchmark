# IIA Benchmark

面向智能工业报警（Intelligent Industrial Alarm, IIA）的可复现基准框架。项目以 Wang、Hu、Chen 的 2024 年专著 *Intelligent Industrial Alarm Systems* 为理论主线，目录与运行方式参考 `D:\aicoding\powergrid_benchmark`，将“知识提取、数据获取、可调用基线、统一评测”放在同一个仓库里。

Canonical repository: <https://github.com/gaoxingkele/IIA_benchmark>

## 当前覆盖

| 任务轨 | 可调用基线 | 核心指标 | 当前数据入口 |
|---|---|---|---|
| 单变量报警设计 | threshold + delay + deadband、参数网格设计 | FAR、MAR、AAD、F1 | synthetic、PRONTO/TEP 接口位 |
| 多变量报警设计 | Mahalanobis、convex-hull NOZ、动态阈值 | FAR、MAR、AAD | synthetic、TEP、SKAB |
| 根因分析 | 离散 transfer entropy + surrogate threshold | Top-k、MRR、时延 | synthetic、PRONTO、TEP alarm（受限） |
| 报警洪泛 | 新出现标签检测、Smith–Waterman、next-alarm | 检出率、序列准确率、提前量 | synthetic、PIADE、TEP alarm（受限） |
| 开集/早期分类 | 协议与指标层，CASIM/ConE-AFC 复现位 | macro-F1、AUROC、coverage、set size | TEP alarm、NPP/FCC（受限） |
| 鲁棒性 | missing/spurious/timing perturbation | clean score、退化量、最坏组 | synthetic、PIADE、SKAB |
| 可视分析验证 | Chapter 6 KPI/bad actor/HDAP/burst/correlation/timeline/similarity/spiral 报告 | 数值事实层、HTML、JSON、事件行追溯 | 任意统一 `AlarmEvent`/`AlarmEpisode` 数据 |

“已实现”“论文复现位”“仅元数据”被明确分层，避免把尚未拿到受限文件的方法写成可比较结果。

## 快速开始

```powershell
python -m pip install -e ".[test]"
python scripts/book/extract_book.py
python scripts/data_acquisition/download_public_datasets.py
python scripts/data_acquisition/audit_public_datasets.py
python -m iia_benchmark.runner configs/experiments/synthetic_univariate_smoke.json
pytest -q
```

另外三个 smoke 配置覆盖洪泛相似性、凸包正常运行区和 transfer-entropy 根因排序：

```powershell
python -m iia_benchmark.runner configs/experiments/synthetic_flood_similarity_smoke.json
python -m iia_benchmark.runner configs/experiments/pronto_casim_fault_classification_validation.json
python -m iia_benchmark.runner configs/experiments/pronto_mahalanobis_validation.json
python -m iia_benchmark.runner configs/experiments/synthetic_multivariate_noz_smoke.json
python -m iia_benchmark.runner configs/experiments/synthetic_root_cause_smoke.json
```

## 目录

- `knowledge_base/book/`：按六章抽取的理论、算法伪代码、任务映射。
- `papers/extracted_text/book/`：带 PDF 物理页标记的证据层及哈希 manifest。
- `docs/three_round_expansion.md`：三轮“数据集→文献→方法→算法/SOTA→数据集”轨迹。
- `docs/build_report.md`：本轮数据 profile、四个 smoke 效果与验证状态。
- `docs/status_audit.md`：严格区分可调用、部分复现、闭环验证及数据门禁的自动覆盖审计。
- `configs/`：系统、数据、切分、模型、指标、实验的 JSON 唯一事实来源。
- `src/iia_benchmark/`：统一数据对象、可调用经典模型、指标和 runner。
- `scripts/data_acquisition/`：aria2/代理优先的下载器与校验审计。
- `data/public_datasets/audit.json`：已下载资源、字节数、校验和或 Git revision。

## 结果有效性边界

`synthetic_*_smoke` 只验证代码路径，禁止进入正式排行榜。公开结果必须按设备/工况/仿真 run 分组切分，训练期完成阈值选择，测试期冻结所有超参，并同时报告不确定性、开放集和扰动结果。详见 [评测协议](docs/evaluation_protocol.md)。

书籍官方信息：[Springer, DOI 10.1007/978-981-97-6516-4](https://link.springer.com/book/10.1007/978-981-97-6516-4)。数据和论文出处见 [数据登记](docs/dataset_registry.md) 与 [文献地图](knowledge_base/literature/paper_map.md)。
