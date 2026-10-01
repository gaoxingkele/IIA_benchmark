# Flow Matching 论文与实验数据获取审计（2026-10-01）

生成时间：2026-10-01T10:50:14.475871+00:00。

注册论文 20 篇；下载 manifest 中当前本地可用 PDF 19 篇。注册数据/配方来源 2536 项；成功记录且现场可用 601 项，其中去重文件 600 个、合计 4.550 GiB。

本报告只审计资料获取，不报告模型性能，也不声明论文实验已经复现。原始数据、作者预处理包、生成配方与访问受限项分别标注；完整实验列表来自三个配置的论文条目。数据存在不代表作者划分、缺失掩码、超参数、随机种子或实验子集一致。

现场检查包括 manifest 成功状态、本地存在、非空、与 manifest 尺寸一致及排除 HTML 占位。SHA256/发布方校验结果沿用下载 manifest，不在本报告脚本中重新读取全部大型数据。目录存在只表明可复用目录，不能代替逐文件审计。

## 输入与完成状态

| 输入 | 情况 |
|---|---|
| papers\literature\flow_matching\data_download_manifest.json | {"selected": 393, "completed": 358, "summary": {"available": 312, "failed": 40, "login_terms_required": 3, "manual_drive_folder": 1, "manual_reconstruction_required": 1, "proprietary_not_released": 1}} |
| papers\literature\flow_matching\paper_download_manifest.json | {"selected": 20, "completed": 20, "summary": {"available": 19, "needs_institution": 1}} |

## 逐篇论文、完整实验列表与资料边界

### Graph-Spectral Flow Matching for Multivariate Time Series Anomaly Detection

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/grasp.pdf](../../papers/literature/flow_matching/pdfs/grasp.pdf)

实验来源：[原论文/官方证据](https://arxiv.org/html/2609.36765#A5.SS1)。

论文列明实验数据集：SMAP (mTSBench 51 series,26 variables)；SMD (mTSBench 18 series,39 variables)；CICIDS2017 (mTSBench six series,73 features)；SWAN (mTSBench one 39-variable series)。

当前可用资料分类：论文指定 mTSBench 发布数据（实验索引仍需核对） 154 项。

- **download_note**：The exact paper benchmark preprocessing/splits are mTSBench, not generic source datasets.

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| mtsbench_SMAP_.DS_Store | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/.DS_Store](../../data/public_datasets/flow_matching/mtsbench/SMAP/.DS_Store) |
| mtsbench_SMAP_SMAP_A-1_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-1_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-1_test.csv) |
| mtsbench_SMAP_SMAP_A-1_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-1_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-1_train.csv) |
| mtsbench_SMAP_SMAP_A-2_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-2_test.csv) |
| mtsbench_SMAP_SMAP_A-2_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-2_train.csv) |
| mtsbench_SMAP_SMAP_A-3_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-3_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-3_test.csv) |
| mtsbench_SMAP_SMAP_A-3_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-3_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-3_train.csv) |
| mtsbench_SMAP_SMAP_A-4_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-4_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-4_test.csv) |
| mtsbench_SMAP_SMAP_A-4_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-4_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-4_train.csv) |
| mtsbench_SMAP_SMAP_A-5_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-5_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-5_test.csv) |
| mtsbench_SMAP_SMAP_A-5_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-5_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-5_train.csv) |
| mtsbench_SMAP_SMAP_A-6_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-6_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-6_test.csv) |
| mtsbench_SMAP_SMAP_A-6_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-6_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-6_train.csv) |
| mtsbench_SMAP_SMAP_A-7_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-7_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-7_test.csv) |
| mtsbench_SMAP_SMAP_A-7_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-7_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-7_train.csv) |
| mtsbench_SMAP_SMAP_A-8_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-8_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-8_test.csv) |
| mtsbench_SMAP_SMAP_A-8_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-8_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-8_train.csv) |
| mtsbench_SMAP_SMAP_A-9_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-9_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-9_test.csv) |
| mtsbench_SMAP_SMAP_A-9_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-9_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_A-9_train.csv) |
| mtsbench_SMAP_SMAP_B-1_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_B-1_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_B-1_test.csv) |
| mtsbench_SMAP_SMAP_B-1_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_B-1_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_B-1_train.csv) |
| mtsbench_SMAP_SMAP_D-11_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-11_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-11_test.csv) |
| mtsbench_SMAP_SMAP_D-11_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-11_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-11_train.csv) |
| mtsbench_SMAP_SMAP_D-1_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-1_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-1_test.csv) |
| mtsbench_SMAP_SMAP_D-1_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-1_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-1_train.csv) |
| mtsbench_SMAP_SMAP_D-2_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-2_test.csv) |
| mtsbench_SMAP_SMAP_D-2_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-2_train.csv) |
| mtsbench_SMAP_SMAP_D-3_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-3_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-3_test.csv) |
| mtsbench_SMAP_SMAP_D-3_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-3_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-3_train.csv) |
| mtsbench_SMAP_SMAP_D-4_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-4_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-4_test.csv) |
| mtsbench_SMAP_SMAP_D-4_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-4_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-4_train.csv) |
| mtsbench_SMAP_SMAP_D-5_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-5_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-5_test.csv) |
| mtsbench_SMAP_SMAP_D-5_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-5_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-5_train.csv) |
| mtsbench_SMAP_SMAP_D-6_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-6_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-6_test.csv) |
| mtsbench_SMAP_SMAP_D-6_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-6_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-6_train.csv) |
| mtsbench_SMAP_SMAP_D-7_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-7_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-7_test.csv) |
| mtsbench_SMAP_SMAP_D-7_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-7_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-7_train.csv) |
| mtsbench_SMAP_SMAP_D-8_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-8_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-8_test.csv) |
| mtsbench_SMAP_SMAP_D-8_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-8_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-8_train.csv) |
| mtsbench_SMAP_SMAP_D-9_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-9_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-9_test.csv) |
| mtsbench_SMAP_SMAP_D-9_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-9_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_D-9_train.csv) |
| mtsbench_SMAP_SMAP_E-10_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-10_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-10_test.csv) |
| mtsbench_SMAP_SMAP_E-10_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-10_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-10_train.csv) |
| mtsbench_SMAP_SMAP_E-11_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-11_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-11_test.csv) |
| mtsbench_SMAP_SMAP_E-11_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-11_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-11_train.csv) |
| mtsbench_SMAP_SMAP_E-12_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-12_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-12_test.csv) |
| mtsbench_SMAP_SMAP_E-12_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-12_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-12_train.csv) |
| mtsbench_SMAP_SMAP_E-13_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-13_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-13_test.csv) |
| mtsbench_SMAP_SMAP_E-13_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-13_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-13_train.csv) |
| mtsbench_SMAP_SMAP_E-1_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-1_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-1_test.csv) |
| mtsbench_SMAP_SMAP_E-1_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-1_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-1_train.csv) |
| mtsbench_SMAP_SMAP_E-2_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-2_test.csv) |
| mtsbench_SMAP_SMAP_E-2_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-2_train.csv) |
| mtsbench_SMAP_SMAP_E-3_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-3_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-3_test.csv) |
| mtsbench_SMAP_SMAP_E-3_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-3_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-3_train.csv) |
| mtsbench_SMAP_SMAP_E-4_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-4_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-4_test.csv) |
| mtsbench_SMAP_SMAP_E-4_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-4_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-4_train.csv) |
| mtsbench_SMAP_SMAP_E-5_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-5_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-5_test.csv) |
| mtsbench_SMAP_SMAP_E-5_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-5_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-5_train.csv) |
| mtsbench_SMAP_SMAP_E-6_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-6_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-6_test.csv) |
| mtsbench_SMAP_SMAP_E-6_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-6_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-6_train.csv) |
| mtsbench_SMAP_SMAP_E-7_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-7_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-7_test.csv) |
| mtsbench_SMAP_SMAP_E-7_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-7_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-7_train.csv) |
| mtsbench_SMAP_SMAP_E-8_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-8_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-8_test.csv) |
| mtsbench_SMAP_SMAP_E-8_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-8_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-8_train.csv) |
| mtsbench_SMAP_SMAP_E-9_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-9_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-9_test.csv) |
| mtsbench_SMAP_SMAP_E-9_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-9_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_E-9_train.csv) |
| mtsbench_SMAP_SMAP_F-1_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_F-1_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_F-1_test.csv) |
| mtsbench_SMAP_SMAP_F-1_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_F-1_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_F-1_train.csv) |
| mtsbench_SMAP_SMAP_F-2_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_F-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_F-2_test.csv) |
| mtsbench_SMAP_SMAP_F-2_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_F-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_F-2_train.csv) |
| mtsbench_SMAP_SMAP_F-3_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_F-3_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_F-3_test.csv) |
| mtsbench_SMAP_SMAP_F-3_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_F-3_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_F-3_train.csv) |
| mtsbench_SMAP_SMAP_G-1_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-1_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-1_test.csv) |
| mtsbench_SMAP_SMAP_G-1_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-1_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-1_train.csv) |
| mtsbench_SMAP_SMAP_G-2_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-2_test.csv) |
| mtsbench_SMAP_SMAP_G-2_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-2_train.csv) |
| mtsbench_SMAP_SMAP_G-3_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-3_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-3_test.csv) |
| mtsbench_SMAP_SMAP_G-3_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-3_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-3_train.csv) |
| mtsbench_SMAP_SMAP_G-4_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-4_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-4_test.csv) |
| mtsbench_SMAP_SMAP_G-4_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-4_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-4_train.csv) |
| mtsbench_SMAP_SMAP_G-6_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-6_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-6_test.csv) |
| mtsbench_SMAP_SMAP_G-6_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-6_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-6_train.csv) |
| mtsbench_SMAP_SMAP_G-7_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-7_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-7_test.csv) |
| mtsbench_SMAP_SMAP_G-7_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-7_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-7_train.csv) |
| mtsbench_SMAP_SMAP_G-7_val.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-7_val.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_G-7_val.csv) |
| mtsbench_SMAP_SMAP_P-1_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_P-1_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_P-1_test.csv) |
| mtsbench_SMAP_SMAP_P-1_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_P-1_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_P-1_train.csv) |
| mtsbench_SMAP_SMAP_P-2_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_P-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_P-2_test.csv) |
| mtsbench_SMAP_SMAP_P-2_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_P-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_P-2_train.csv) |
| mtsbench_SMAP_SMAP_P-3_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_P-3_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_P-3_test.csv) |
| mtsbench_SMAP_SMAP_P-3_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_P-3_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_P-3_train.csv) |
| mtsbench_SMAP_SMAP_P-4_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_P-4_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_P-4_test.csv) |
| mtsbench_SMAP_SMAP_P-4_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_P-4_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_P-4_train.csv) |
| mtsbench_SMAP_SMAP_P-7_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_P-7_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_P-7_train.csv) |
| mtsbench_SMAP_SMAP_R-1_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_R-1_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_R-1_test.csv) |
| mtsbench_SMAP_SMAP_R-1_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_R-1_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_R-1_train.csv) |
| mtsbench_SMAP_SMAP_S-1_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_S-1_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_S-1_test.csv) |
| mtsbench_SMAP_SMAP_S-1_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_S-1_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_S-1_train.csv) |
| mtsbench_SMAP_SMAP_T-1_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_T-1_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_T-1_test.csv) |
| mtsbench_SMAP_SMAP_T-1_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_T-1_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_T-1_train.csv) |
| mtsbench_SMAP_SMAP_T-2_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_T-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_T-2_test.csv) |
| mtsbench_SMAP_SMAP_T-2_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_T-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_T-2_train.csv) |
| mtsbench_SMAP_SMAP_T-3_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_T-3_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_T-3_test.csv) |
| mtsbench_SMAP_SMAP_T-3_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_T-3_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMAP/SMAP_T-3_train.csv) |
| mtsbench_SMD_SMD_machine-1-2_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-2_test.csv) |
| mtsbench_SMD_SMD_machine-1-2_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-2_train.csv) |
| mtsbench_SMD_SMD_machine-1-2_val.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-2_val.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-2_val.csv) |
| mtsbench_SMD_SMD_machine-1-3_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-3_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-3_test.csv) |
| mtsbench_SMD_SMD_machine-1-3_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-3_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-3_train.csv) |
| mtsbench_SMD_SMD_machine-1-4_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-4_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-4_test.csv) |
| mtsbench_SMD_SMD_machine-1-4_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-4_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-4_train.csv) |
| mtsbench_SMD_SMD_machine-1-6_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-6_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-6_test.csv) |
| mtsbench_SMD_SMD_machine-1-6_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-6_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-6_train.csv) |
| mtsbench_SMD_SMD_machine-1-7_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-7_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-7_test.csv) |
| mtsbench_SMD_SMD_machine-1-7_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-7_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-7_train.csv) |
| mtsbench_SMD_SMD_machine-1-8_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-8_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-8_test.csv) |
| mtsbench_SMD_SMD_machine-1-8_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-8_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-8_train.csv) |
| mtsbench_SMD_SMD_machine-2-1_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-1_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-1_test.csv) |
| mtsbench_SMD_SMD_machine-2-1_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-1_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-1_train.csv) |
| mtsbench_SMD_SMD_machine-2-2_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-2_test.csv) |
| mtsbench_SMD_SMD_machine-2-2_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-2_train.csv) |
| mtsbench_SMD_SMD_machine-2-3_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-3_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-3_test.csv) |
| mtsbench_SMD_SMD_machine-2-3_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-3_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-3_train.csv) |
| mtsbench_SMD_SMD_machine-2-4_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-4_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-4_test.csv) |
| mtsbench_SMD_SMD_machine-2-4_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-4_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-4_train.csv) |
| mtsbench_SMD_SMD_machine-2-5_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-5_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-5_test.csv) |
| mtsbench_SMD_SMD_machine-2-5_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-5_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-5_train.csv) |
| mtsbench_SMD_SMD_machine-2-7_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-7_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-7_test.csv) |
| mtsbench_SMD_SMD_machine-2-7_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-7_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-7_train.csv) |
| mtsbench_SMD_SMD_machine-2-9_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-9_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-9_test.csv) |
| mtsbench_SMD_SMD_machine-2-9_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-9_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-9_train.csv) |
| mtsbench_SMD_SMD_machine-3-10_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-10_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-10_test.csv) |
| mtsbench_SMD_SMD_machine-3-10_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-10_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-10_train.csv) |
| mtsbench_SMD_SMD_machine-3-2_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-2_test.csv) |
| mtsbench_SMD_SMD_machine-3-2_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-2_train.csv) |
| mtsbench_SMD_SMD_machine-3-3_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-3_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-3_test.csv) |
| mtsbench_SMD_SMD_machine-3-3_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-3_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-3_train.csv) |
| mtsbench_SMD_SMD_machine-3-5_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-5_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-5_test.csv) |
| mtsbench_SMD_SMD_machine-3-5_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-5_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-5_train.csv) |
| mtsbench_SMD_SMD_machine-3-6_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-6_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-6_test.csv) |
| mtsbench_SMD_SMD_machine-3-6_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-6_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-6_train.csv) |
| mtsbench_cicids_.DS_Store | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/.DS_Store](../../data/public_datasets/flow_matching/mtsbench/cicids/.DS_Store) |
| mtsbench_cicids_cicids_0_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_0_test.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_0_test.csv) |
| mtsbench_cicids_cicids_0_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_0_train.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_0_train.csv) |
| mtsbench_cicids_cicids_0_val.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_0_val.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_0_val.csv) |
| mtsbench_cicids_cicids_2_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_2_test.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_2_test.csv) |
| mtsbench_cicids_cicids_2_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_2_train.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_2_train.csv) |
| mtsbench_cicids_cicids_3_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_3_test.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_3_test.csv) |
| mtsbench_cicids_cicids_3_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_3_train.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_3_train.csv) |
| mtsbench_cicids_cicids_5_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_5_test.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_5_test.csv) |
| mtsbench_cicids_cicids_5_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_5_train.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_5_train.csv) |
| mtsbench_cicids_cicids_6_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/cicids/cicids_6_test.csv:                                                                                  [#3dc9a1 0B/0B CN:1 DL:0B]                                                                                 [#3dc9a1 0B/0B CN:1 DL:0B]                                                                                 [#3dc9a1 0B/0B CN:1 DL:0B] 10/01 18:29:13 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/cicids/cicids_6_test.csv Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/cicids/cicids_6_test.csv   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 3dc9a1\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/mtsbench/cicids/cicids_6_test.csv.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_6_test.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_6_test.csv) |
| mtsbench_cicids_cicids_6_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_6_train.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_6_train.csv) |
| mtsbench_cicids_cicids_7_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/cicids/cicids_7_test.csv:                                                                                  [#aa79fe 0B/0B CN:1 DL:0B]                                                                                 [#aa79fe 0B/0B CN:1 DL:0B] 10/01 18:29:16 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/cicids/cicids_7_test.csv Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/cicids/cicids_7_test.csv   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= aa79fe\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/mtsbench/cicids/cicids_7_test.csv.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_7_test.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_7_test.csv) |
| mtsbench_cicids_cicids_7_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_7_train.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_7_train.csv) |
| mtsbench_swan_swan_sf_test.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_test.csv:                                                                                  [#a88e2c 0B/0B CN:1 DL:0B]                                                                                 [#a88e2c 0B/0B CN:1 DL:0B] 10/01 18:29:16 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_test.csv Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_test.csv   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= a88e2c\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/mtsbench/swan/swan_sf_test.csv.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/mtsbench/swan/swan_sf_test.csv](../../data/public_datasets/flow_matching/mtsbench/swan/swan_sf_test.csv) |
| mtsbench_swan_swan_sf_train.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_train.csv:                                                                                  [#ed89a6 0B/0B CN:1 DL:0B]                                                                                 [#ed89a6 0B/0B CN:1 DL:0B] 10/01 18:29:16 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_train.csv Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_train.csv   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= ed89a6\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/mtsbench/swan/swan_sf_train.csv.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/mtsbench/swan/swan_sf_train.csv](../../data/public_datasets/flow_matching/mtsbench/swan/swan_sf_train.csv) |
| mtsbench_swan_swan_sf_val.csv | 论文指定 mTSBench 发布数据（实验索引仍需核对） | https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_val.csv:                                                                                  [#eb07e9 0B/0B CN:1 DL:0B]                                                                                 [#eb07e9 0B/0B CN:1 DL:0B] 10/01 18:29:16 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_val.csv Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_val.csv   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= eb07e9\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/mtsbench/swan/swan_sf_val.csv.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/mtsbench/swan/swan_sf_val.csv](../../data/public_datasets/flow_matching/mtsbench/swan/swan_sf_val.csv) |

### Time-Gated Multi-Scale Flow Matching for Time-Series Imputation

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/tg_msfm.pdf](../../papers/literature/flow_matching/pdfs/tg_msfm.pdf)

实验来源：[原论文/官方证据](https://proceedings.iclr.cc/paper_files/paper/2026/file/51d317df78eded9eb3c9d3fb1091c279-Paper-Conference.pdf)。

论文列明实验数据集：ETTh1；ETTh2；ETTm1；ETTm2；Electricity；Traffic；Weather；Illness；Exchange；PEMS03。

当前可用资料分类：原始/基础公开数据或作者数据包 13 项。

- **reproduction_boundary**：Paper section4.1 names all ten datasets but exact author preprocessing artifacts/code not located; public TSL/tsl sources are registered as underlying data.

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| pems03 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/pems/pems03.zip](../../data/public_datasets/flow_matching/pems/pems03.zip) |
| tg_msfm_ETT-small_ETTh1.csv | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/tslib/ETT-small/ETTh1.csv](../../data/public_datasets/flow_matching/tslib/ETT-small/ETTh1.csv) |
| tg_msfm_ETT-small_ETTh2.csv | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/tslib/ETT-small/ETTh2.csv](../../data/public_datasets/flow_matching/tslib/ETT-small/ETTh2.csv) |
| tg_msfm_ETT-small_ETTm1.csv | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/tslib/ETT-small/ETTm1.csv](../../data/public_datasets/flow_matching/tslib/ETT-small/ETTm1.csv) |
| tg_msfm_ETT-small_ETTm2.csv | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/tslib/ETT-small/ETTm2.csv](../../data/public_datasets/flow_matching/tslib/ETT-small/ETTm2.csv) |
| tg_msfm_electricity_.DS_Store | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/tslib/electricity/.DS_Store](../../data/public_datasets/flow_matching/tslib/electricity/.DS_Store) |
| tg_msfm_electricity_electricity.csv | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/tslib/electricity/electricity.csv](../../data/public_datasets/flow_matching/tslib/electricity/electricity.csv) |
| tg_msfm_exchange_rate_.DS_Store | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/tslib/exchange_rate/.DS_Store](../../data/public_datasets/flow_matching/tslib/exchange_rate/.DS_Store) |
| tg_msfm_exchange_rate_exchange_rate.csv | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/tslib/exchange_rate/exchange_rate.csv](../../data/public_datasets/flow_matching/tslib/exchange_rate/exchange_rate.csv) |
| tg_msfm_illness_national_illness.csv | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/tslib/illness/national_illness.csv](../../data/public_datasets/flow_matching/tslib/illness/national_illness.csv) |
| tg_msfm_traffic_.DS_Store | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/tslib/traffic/.DS_Store](../../data/public_datasets/flow_matching/tslib/traffic/.DS_Store) |
| tg_msfm_traffic_traffic.csv | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/tslib/traffic/traffic.csv](../../data/public_datasets/flow_matching/tslib/traffic/traffic.csv) |
| tg_msfm_weather_weather.csv | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/tslib/weather/weather.csv](../../data/public_datasets/flow_matching/tslib/weather/weather.csv) |

### Spatiotemporal Imputation with Graph-Informed Flow Matching

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/giflow.pdf](../../papers/literature/flow_matching/pdfs/giflow.pdf)

实验来源：[原论文/官方证据](https://arxiv.org/html/2606.06682#A3.SS1)。

论文列明实验数据集：Air-36 (36 stations,8760 hourly samples)；AQI (437 stations,43 cities,8760 hourly samples)；PeMS08 (170 sensors,17856 five-minute samples)；Synthetic smooth spatiotemporal graph (50 nodes,3000 steps)；Synthetic scalability graphs (10000,20000,30000,50000 nodes)。

当前可用资料分类：原始/基础公开数据或作者数据包 2 项。

- **synthetic_generation**：{"evidence_url": "https://arxiv.org/html/2606.06682#S4.SS1", "recipe": "Uniform random positions in 50x50 square, k=5 KNN graph; Laplacian pseudoinverse square root with first eigenvalue set to zero; low-frequency initial signal, Gaussian increments propagated by L^-1/2; R=3000; Gaussian observation noise sigma=0.1 or0.3;20% point masking. Scalability uses same generation procedure.", "boundary": "Exact sampled seeds and initial spectral coefficients are not fully specified. Repo currently has README only; cannot reproduce exact authors samples by guessing."}

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| pems08 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/pems/pems08.zip](../../data/public_datasets/flow_matching/pems/pems08.zip) |
| air_quality_tsl | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/air_quality/air_quality.zip](../../data/public_datasets/flow_matching/air_quality/air_quality.zip) |

### CFMI: Flow Matching for Missing Data Imputation

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/cfmi.pdf](../../papers/literature/flow_matching/pdfs/cfmi.pdf)

实验来源：[原论文/官方证据](https://arxiv.org/pdf/2506.09258)。

论文列明实验数据集：airfoil_self_noise；banknote；blood_transfusion；breast_cancer_diagnostic；california；climate_model_crashes；concrete_compression；concrete_slump；connectionist_bench_sonar；connectionist_bench_vowel；ecoli；glass；ionosphere；iris；libras；parkinsons；planning_relax；qsar_biodegradation；seeds；wine；wine_quality_red；wine_quality_white；yacht_hydrodynamics；yeast；PhysioNet/Computing in Cardiology Challenge2012 set-a；PM2.5/STMVL Air-36；Synthetic2D cosine,banana,funnel,ring,spiral。

当前可用资料分类：原始/基础公开数据或作者数据包 42 项。

- **synthetic_generation**：{"evidence_url": "https://github.com/vsimkus/cfmi/blob/main/imp_cfm/data/toy_synthetic.py", "recipe": "create_and_save_dataset(dataset,num_samples=25000,seed=20240513),20K training points; saved train/val/test arrays provided in author repo.", "boundary": "Repo additionally includes mring; paper reports five distributions. mring is supplementary available code/data, not one of confirmed five paper experiments."}

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| cfmi_data_physionet_set-a-processed.npz | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/physionet/set-a-processed.npz](../../data/public_datasets/flow_matching/cfmi/physionet/set-a-processed.npz) |
| cfmi_data_physionet_set-a.tar.gz | 原始/基础公开数据或作者数据包 | https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/physionet/set-a.tar.gz:                                                                                  [#fee9c0 0B/0B CN:1 DL:0B]                                                                                 [#fee9c0 0B/0B CN:1 DL:0B] 10/01 18:29:16 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/physionet/set-a.tar.gz Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/physionet/set-a.tar.gz   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= fee9c0\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/cfmi/physionet/set-a.tar.gz.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/cfmi/physionet/set-a.tar.gz](../../data/public_datasets/flow_matching/cfmi/physionet/set-a.tar.gz) |
| cfmi_data_pm25_STMVL-Release.zip | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/pm25/STMVL-Release.zip](../../data/public_datasets/flow_matching/cfmi/pm25/STMVL-Release.zip) |
| cfmi_data_pm25_data_Code_STMVL_SampleData_pm25_ground.txt | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/pm25/data/Code/STMVL/SampleData/pm25_ground.txt](../../data/public_datasets/flow_matching/cfmi/pm25/data/Code/STMVL/SampleData/pm25_ground.txt) |
| cfmi_data_pm25_data_Code_STMVL_SampleData_pm25_latlng.txt | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/pm25/data/Code/STMVL/SampleData/pm25_latlng.txt](../../data/public_datasets/flow_matching/cfmi/pm25/data/Code/STMVL/SampleData/pm25_latlng.txt) |
| cfmi_data_pm25_data_Code_STMVL_SampleData_pm25_missing.txt | 原始/基础公开数据或作者数据包 | https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/pm25/data/Code/STMVL/SampleData/pm25_missing.txt:                                                                                  [#beb084 0B/0B CN:1 DL:0B]                                                                                 [#beb084 0B/0B CN:1 DL:0B] 10/01 18:29:16 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/pm25/data/Code/STMVL/SampleData/pm25_missing.txt Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/pm25/data/Code/STMVL/SampleData/pm25_missing.txt   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= beb084\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/cfmi/pm25/data/Code/STMVL/SampleData/pm25_missing.txt.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/cfmi/pm25/data/Code/STMVL/SampleData/pm25_missing.txt](../../data/public_datasets/flow_matching/cfmi/pm25/data/Code/STMVL/SampleData/pm25_missing.txt) |
| cfmi_data_pm25_data_STMVL-Readme.pdf | 原始/基础公开数据或作者数据包 | https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/pm25/data/STMVL-Readme.pdf:                                                                                  [#4bed72 0B/0B CN:1 DL:0B]                                                                                 [#4bed72 0B/0B CN:1 DL:0B] 10/01 18:29:17 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/pm25/data/STMVL-Readme.pdf Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/pm25/data/STMVL-Readme.pdf   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 4bed72\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/cfmi/pm25/data/STMVL-Readme.pdf.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/cfmi/pm25/data/STMVL-Readme.pdf](../../data/public_datasets/flow_matching/cfmi/pm25/data/STMVL-Readme.pdf) |
| cfmi_data_pm25_data_pm25_meanstd.pk | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/pm25/data/pm25_meanstd.pk](../../data/public_datasets/flow_matching/cfmi/pm25/data/pm25_meanstd.pk) |
| cfmi_data_synthetic_banana_test.npz | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/synthetic/banana/test.npz](../../data/public_datasets/flow_matching/cfmi/synthetic/banana/test.npz) |
| cfmi_data_synthetic_banana_train.npz | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/synthetic/banana/train.npz](../../data/public_datasets/flow_matching/cfmi/synthetic/banana/train.npz) |
| cfmi_data_synthetic_banana_val.npz | 原始/基础公开数据或作者数据包 | https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/banana/val.npz:                                                                                  [#7d0a31 0B/0B CN:1 DL:0B]                                                                                 [#7d0a31 0B/0B CN:1 DL:0B] 10/01 18:29:18 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/banana/val.npz Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/banana/val.npz   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 7d0a31\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/cfmi/synthetic/banana/val.npz.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/cfmi/synthetic/banana/val.npz](../../data/public_datasets/flow_matching/cfmi/synthetic/banana/val.npz) |
| cfmi_data_synthetic_cosine_test.npz | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/synthetic/cosine/test.npz](../../data/public_datasets/flow_matching/cfmi/synthetic/cosine/test.npz) |
| cfmi_data_synthetic_cosine_train.npz | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/synthetic/cosine/train.npz](../../data/public_datasets/flow_matching/cfmi/synthetic/cosine/train.npz) |
| cfmi_data_synthetic_cosine_val.npz | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/synthetic/cosine/val.npz](../../data/public_datasets/flow_matching/cfmi/synthetic/cosine/val.npz) |
| cfmi_data_synthetic_funnel_test.npz | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/synthetic/funnel/test.npz](../../data/public_datasets/flow_matching/cfmi/synthetic/funnel/test.npz) |
| cfmi_data_synthetic_funnel_train.npz | 原始/基础公开数据或作者数据包 | https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/funnel/train.npz:                                                                                  [#583c15 0B/0B CN:1 DL:0B]                                                                                 [#583c15 0B/0B CN:1 DL:0B] 10/01 18:29:19 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/funnel/train.npz Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/funnel/train.npz   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 583c15\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/cfmi/synthetic/funnel/train.npz.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/cfmi/synthetic/funnel/train.npz](../../data/public_datasets/flow_matching/cfmi/synthetic/funnel/train.npz) |
| cfmi_data_synthetic_funnel_val.npz | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/synthetic/funnel/val.npz](../../data/public_datasets/flow_matching/cfmi/synthetic/funnel/val.npz) |
| cfmi_data_synthetic_mring_test.npz | 原始/基础公开数据或作者数据包 | https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/mring/test.npz:                                                                                  [#6aa481 0B/0B CN:1 DL:0B]                                                                                 [#6aa481 0B/0B CN:1 DL:0B] 10/01 18:29:19 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/mring/test.npz Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/mring/test.npz   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 6aa481\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/cfmi/synthetic/mring/test.npz.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/cfmi/synthetic/mring/test.npz](../../data/public_datasets/flow_matching/cfmi/synthetic/mring/test.npz) |
| cfmi_data_synthetic_mring_train.npz | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/synthetic/mring/train.npz](../../data/public_datasets/flow_matching/cfmi/synthetic/mring/train.npz) |
| cfmi_data_synthetic_mring_val.npz | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/synthetic/mring/val.npz](../../data/public_datasets/flow_matching/cfmi/synthetic/mring/val.npz) |
| cfmi_data_synthetic_ring_test.npz | 原始/基础公开数据或作者数据包 | https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/ring/test.npz:                                                                                  [#130e42 0B/0B CN:1 DL:0B]                                                                                 [#130e42 0B/0B CN:1 DL:0B] 10/01 18:29:20 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/ring/test.npz Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/ring/test.npz   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 130e42\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/cfmi/synthetic/ring/test.npz.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/cfmi/synthetic/ring/test.npz](../../data/public_datasets/flow_matching/cfmi/synthetic/ring/test.npz) |
| cfmi_data_synthetic_ring_train.npz | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/synthetic/ring/train.npz](../../data/public_datasets/flow_matching/cfmi/synthetic/ring/train.npz) |
| cfmi_data_synthetic_ring_val.npz | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/synthetic/ring/val.npz](../../data/public_datasets/flow_matching/cfmi/synthetic/ring/val.npz) |
| cfmi_data_synthetic_spiral_test.npz | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/synthetic/spiral/test.npz](../../data/public_datasets/flow_matching/cfmi/synthetic/spiral/test.npz) |
| cfmi_data_synthetic_spiral_train.npz | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/synthetic/spiral/train.npz](../../data/public_datasets/flow_matching/cfmi/synthetic/spiral/train.npz) |
| cfmi_data_synthetic_spiral_val.npz | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/synthetic/spiral/val.npz](../../data/public_datasets/flow_matching/cfmi/synthetic/spiral/val.npz) |
| cfmi_data_uci_airfoil_self_noise_airfoil_self_noise.dat | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/airfoil_self_noise/airfoil_self_noise.dat](../../data/public_datasets/flow_matching/cfmi/uci/airfoil_self_noise/airfoil_self_noise.dat) |
| cfmi_data_uci_banknote_data_banknote_authentication.txt | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/banknote/data_banknote_authentication.txt](../../data/public_datasets/flow_matching/cfmi/uci/banknote/data_banknote_authentication.txt) |
| cfmi_data_uci_blood_transfusion_transfusion.data | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/blood_transfusion/transfusion.data](../../data/public_datasets/flow_matching/cfmi/uci/blood_transfusion/transfusion.data) |
| cfmi_data_uci_breast_cancer_diagnostic_wdbc.data | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/breast_cancer_diagnostic/wdbc.data](../../data/public_datasets/flow_matching/cfmi/uci/breast_cancer_diagnostic/wdbc.data) |
| cfmi_data_uci_cal_housing_py3.pkz | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/cal_housing_py3.pkz](../../data/public_datasets/flow_matching/cfmi/uci/cal_housing_py3.pkz) |
| cfmi_data_uci_climate_model_crashes_pop_failures.dat | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/climate_model_crashes/pop_failures.dat](../../data/public_datasets/flow_matching/cfmi/uci/climate_model_crashes/pop_failures.dat) |
| cfmi_data_uci_concrete_compression_Concrete_Data.xls | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/concrete_compression/Concrete_Data.xls](../../data/public_datasets/flow_matching/cfmi/uci/concrete_compression/Concrete_Data.xls) |
| cfmi_data_uci_concrete_slump_slump_test.data | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/concrete_slump/slump_test.data](../../data/public_datasets/flow_matching/cfmi/uci/concrete_slump/slump_test.data) |
| cfmi_data_uci_connectionist_bench_sonar_sonar.all-data | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/connectionist_bench_sonar/sonar.all-data](../../data/public_datasets/flow_matching/cfmi/uci/connectionist_bench_sonar/sonar.all-data) |
| cfmi_data_uci_connectionist_bench_vowel_vowel-context.data | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/connectionist_bench_vowel/vowel-context.data](../../data/public_datasets/flow_matching/cfmi/uci/connectionist_bench_vowel/vowel-context.data) |
| cfmi_data_uci_ecoli_ecoli.data | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/ecoli/ecoli.data](../../data/public_datasets/flow_matching/cfmi/uci/ecoli/ecoli.data) |
| cfmi_data_uci_glass_glass.data | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/glass/glass.data](../../data/public_datasets/flow_matching/cfmi/uci/glass/glass.data) |
| cfmi_data_uci_ionosphere_ionosphere.data | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/ionosphere/ionosphere.data](../../data/public_datasets/flow_matching/cfmi/uci/ionosphere/ionosphere.data) |
| cfmi_data_uci_libras_movement_libras.data | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/libras/movement_libras.data](../../data/public_datasets/flow_matching/cfmi/uci/libras/movement_libras.data) |
| cfmi_data_uci_parkinsons_parkinsons.data | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/parkinsons/parkinsons.data](../../data/public_datasets/flow_matching/cfmi/uci/parkinsons/parkinsons.data) |
| cfmi_data_uci_planning_relax_plrx.txt | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/planning_relax/plrx.txt](../../data/public_datasets/flow_matching/cfmi/uci/planning_relax/plrx.txt) |
| cfmi_data_uci_qsar_biodegradation_biodeg.csv | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/qsar_biodegradation/biodeg.csv](../../data/public_datasets/flow_matching/cfmi/uci/qsar_biodegradation/biodeg.csv) |
| cfmi_data_uci_seeds_seeds_dataset.txt | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/seeds/seeds_dataset.txt](../../data/public_datasets/flow_matching/cfmi/uci/seeds/seeds_dataset.txt) |
| cfmi_data_uci_wine_quality_red_winequality-red.csv | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/wine_quality_red/winequality-red.csv](../../data/public_datasets/flow_matching/cfmi/uci/wine_quality_red/winequality-red.csv) |
| cfmi_data_uci_wine_quality_white_winequality-white.csv | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/wine_quality_white/winequality-white.csv](../../data/public_datasets/flow_matching/cfmi/uci/wine_quality_white/winequality-white.csv) |
| cfmi_data_uci_yacht_hydrodynamics_yacht_hydrodynamics.data | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/yacht_hydrodynamics/yacht_hydrodynamics.data](../../data/public_datasets/flow_matching/cfmi/uci/yacht_hydrodynamics/yacht_hydrodynamics.data) |
| cfmi_data_uci_yeast_yeast.data | 原始/基础公开数据或作者数据包 | https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/uci/yeast/yeast.data:                                                                                  [#1b1599 0B/0B CN:1 DL:0B]                                                                                 [#1b1599 0B/0B CN:1 DL:0B] 10/01 18:29:22 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/uci/yeast/yeast.data Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/uci/yeast/yeast.data   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 1b1599\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/cfmi/uci/yeast/yeast.data.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/cfmi/uci/yeast/yeast.data](../../data/public_datasets/flow_matching/cfmi/uci/yeast/yeast.data) |
| cfmi_iris | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/iris.csv](../../data/public_datasets/flow_matching/cfmi/uci/iris.csv) |
| cfmi_wine | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/cfmi/uci/wine_data.csv](../../data/public_datasets/flow_matching/cfmi/uci/wine_data.csv) |

### Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/tsflow.pdf](../../papers/literature/flow_matching/pdfs/tsflow.pdf)

实验来源：[原论文/官方证据](https://arxiv.org/html/2410.03024#S4)。

论文列明实验数据集：Electricity (electricity_nips)；Exchange(exchange_rate_nips)；KDDCup(kdd_cup_2018_without_missing)；M4-Hourly(m4_hourly)；Solar(solar_nips)；Traffic(traffic_nips)；UberTLC-Hourly(uber_tlc_hourly)；Wikipedia(wiki2000_nips)。

当前可用资料分类：原始/基础公开数据或作者数据包 7 项。

- **preprocessing_note**：Uber source ZIP is raw Jan-Jun2015 trips; official GluonTS _uber_tlc.py aggregates hourly by location ID. KDD TSF requires GluonTS converter. Other GluonTS archives contain prepared splits.

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| tsflow_electricity_nips | 原始/基础公开数据或作者数据包 | https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/mv_release/datasets/electricity_nips.tar.gz:                                                 [#345fe5 9.3MiB/42MiB(21%) CN:2 DL:3.6MiB ETA:9s]                                                                                 [#345fe5 12MiB/42MiB(29%) CN:2 DL:3.5MiB ETA:8s]                                                                                 [#345fe5 16MiB/42MiB(38%) CN:2 DL:3.6MiB ETA:7s]                                                                                 [#345fe5 18MiB/42MiB(44%) CN:2 DL:3.4MiB ETA:6s]                                                                                 [#345fe5 21MiB/42MiB(50%) CN:2 DL:3.3MiB ETA:6s]                                                                                 [#345fe5 23MiB/42MiB(56%) CN:2 DL:3.1MiB ETA:5s]                                                                                 [#345fe5 26MiB/42MiB(61%) CN:2 DL:3.0MiB ETA:5s]                                                                                 [#345fe5 27MiB/42MiB(65%) CN:2 DL:2.9MiB ETA:5s]                                                                                 [#345fe5 31MiB/42MiB(73%) CN:2 DL:2.8MiB ETA:4s]                                                                                 [#345fe5 33MiB/42MiB(79%) CN:2 DL:2.7MiB ETA:3s]                                                                                 [#345fe5 34MiB/42MiB(81%) CN:1 DL:2.2MiB ETA:3s]                                                                                 [#345fe5 34MiB/42MiB(81%) CN:1 DL:1.8MiB ETA:4s]                                                                                 [#345fe5 34MiB/42MiB(81%) CN:1 DL:1.5MiB ETA:5s] 10/01 18:29:35 [[1;31mERROR[0m] CUID#8 - Download aborted. URI=https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/mv_release/datasets/electricity_nips.tar.gz Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/mv_release/datasets/electricity_nips.tar.gz   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 345fe5\|ERR \|   2.2MiB/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/gluonts/electricity_nips.tar.gz.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/gluonts/electricity_nips.tar.gz](../../data/public_datasets/flow_matching/gluonts/electricity_nips.tar.gz) |
| tsflow_exchange_rate_nips | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/gluonts/exchange_rate_nips.tar.gz](../../data/public_datasets/flow_matching/gluonts/exchange_rate_nips.tar.gz) |
| tsflow_solar_nips | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/gluonts/solar_nips.tar.gz](../../data/public_datasets/flow_matching/gluonts/solar_nips.tar.gz) |
| tsflow_traffic_nips | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/gluonts/traffic_nips.tar.gz](../../data/public_datasets/flow_matching/gluonts/traffic_nips.tar.gz) |
| tsflow_wiki2000 | 原始/基础公开数据或作者数据包 | https://github.com/awslabs/gluonts/raw/b89f203595183340651411a41eeb0ee60570a4d9/datasets/wiki2000_nips.tar.gz:                                                                                  [#d487b1 0B/0B CN:1 DL:0B]                                                                                 [#d487b1 0B/0B CN:1 DL:0B] 10/01 18:29:24 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://github.com/awslabs/gluonts/raw/b89f203595183340651411a41eeb0ee60570a4d9/datasets/wiki2000_nips.tar.gz Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/awslabs/gluonts/b89f203595183340651411a41eeb0ee60570a4d9/datasets/wiki2000_nips.tar.gz   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= d487b1\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/gluonts/wiki2000_nips.tar.gz.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/gluonts/wiki2000_nips.tar.gz](../../data/public_datasets/flow_matching/gluonts/wiki2000_nips.tar.gz) |
| tsflow_kdd | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/gluonts/kdd_cup_2018_dataset_without_missing_values.zip](../../data/public_datasets/flow_matching/gluonts/kdd_cup_2018_dataset_without_missing_values.zip) |
| tsflow_m4_train | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/gluonts/M4_Hourly-train.csv](../../data/public_datasets/flow_matching/gluonts/M4_Hourly-train.csv) |
| tsflow_m4_test | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/gluonts/M4_Hourly-test.csv](../../data/public_datasets/flow_matching/gluonts/M4_Hourly-test.csv) |
| tsflow_uber | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/gluonts/uber-raw-data-janjune-15.csv.zip](../../data/public_datasets/flow_matching/gluonts/uber-raw-data-janjune-15.csv.zip) |

### DFM: Interpolant-free Dual Flow Matching

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/dfm.pdf](../../papers/literature/flow_matching/pdfs/dfm.pdf)

实验来源：[原论文/官方证据](https://arxiv.org/html/2410.09246#S4)。

论文列明实验数据集：SMAP (entity-unconditional,55 entities,25 data dimensions as reported in paper)。

当前可用资料分类：原始/基础公开数据或作者数据包 1 项。

- **boundary**：Paper SMAP dimensions/entities differ from mTSBench GRASP. Do not relabel mTSBench as exact DFM data. Original telemanom README now refers to public Kaggle NASA anomaly detection dataset, and exact DFM author preprocessing artifacts not located.

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| dfm_nasa_smap_msl_author_mirror | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/nasa_author/nasa_smap_msl.zip](../../data/public_datasets/flow_matching/nasa_author/nasa_smap_msl.zip) |

### PrismFlow: Residual Dynamics for Flow Matching in Time-Series Generation

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/prismflow.pdf](../../papers/literature/flow_matching/pdfs/prismflow.pdf)

实验来源：[原论文/官方证据](https://arxiv.org/html/2605.28867#S5)。

论文列明实验数据集：Stocks；ETTh (baseline uses ETTh1)；Energy；fMRI；Sines(synthetic)；MuJoCo(synthetic simulation)。

当前可用资料分类：原始/基础公开数据或作者数据包 1 项。

- **boundary**：Section5 cites TimeGAN and Diffusion-TS data protocol. Baseline author archive registered; exact PrismFlow code and generator configuration not located. Do not invent MuJoCo initial conditions or claim recreated data matches paper.

- **synthetic_generation**：{"evidence_url": "https://github.com/Y-debug-sys/Diffusion-TS", "recipe": "Use baseline sine and MuJoCo simulator implementations after reviewing baseline configs; exact PrismFlow generation seed/count/config not stated in accessible paper.", "status": "author_generator_not_found"}

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| prismflow_diffusionts_archive | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/diffusion_ts/dataset.zip](../../data/public_datasets/flow_matching/diffusion_ts/dataset.zip) |

### Flow matching-based online imputation of chemical processes under arbitrary missing scenarios for enhanced process monitoring

PDF：未取得可用全文；needs_institution。 [papers/literature/flow_matching/pdfs/jfi.pdf](../../papers/literature/flow_matching/pdfs/jfi.pdf)

实验来源：[原论文/官方证据](https://www.sciencedirect.com/science/article/pii/S0009250926011826)。

论文列明实验数据集：Tennessee Eastman process(DTU DOI10.11583/DTU.13385936.v1)；Hydrocracking process(HP,real-world industrial data)。

当前可用资料分类：没有已完成且现场可用的注册资料。

- **boundary**：Publisher snippets say200 simulations; DTU official source says500 seeds/fault. Preserve full original data; exact paper subset needs full methodology or author artifacts.

- **restricted_items**：[{"dataset": "HP", "status": "no_public_download_link_found", "reason": "Publisher snippets verify experiment exists; author HP data link not published in accessible material. Exact access/release terms unknown; request from authors through authorized channel."}]

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| jfi_tep_26003087 | 原始/基础公开数据或作者数据包 | https://ndownloader.figshare.com/files/26003087?download=1&cachebust=1790850549810505300: HTTPSConnectionPool(host='ndownloader.figshare.com', port=443): Max retries exceeded with url: /files/26003087?download=1&cachebust=1790850549810505300 (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1010)'))) | [data/public_datasets/flow_matching/tep_dtu_v1/TEP_Mode1.h5](../../data/public_datasets/flow_matching/tep_dtu_v1/TEP_Mode1.h5) |
| jfi_tep_26003162 | 原始/基础公开数据或作者数据包 | https://ndownloader.figshare.com/files/26003162?download=1&cachebust=1790850549813843400: HTTPSConnectionPool(host='ndownloader.figshare.com', port=443): Max retries exceeded with url: /files/26003162?download=1&cachebust=1790850549813843400 (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1010)'))) | [data/public_datasets/flow_matching/tep_dtu_v1/TEP_Mode2.h5](../../data/public_datasets/flow_matching/tep_dtu_v1/TEP_Mode2.h5) |
| jfi_tep_26003492 | 原始/基础公开数据或作者数据包 | https://ndownloader.figshare.com/files/26003492?download=1&cachebust=1790850549812294400: HTTPSConnectionPool(host='s3q.ait.dtu.dk', port=9000): Max retries exceeded with url: /figshare/26003492/TEP_Mode3.h5?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=00000005001330512eb1/20261001/oha/s3/aws4_request&X-Amz-Date=20261001T102912Z&X-Amz-Expires=10&X-Amz-SignedHeaders=host&X-Amz-Signature=9665065290cf3607f6221d219fbeed2b38902aa2a21ec69cac467e656ae1a4cd (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1010)'))) | [data/public_datasets/flow_matching/tep_dtu_v1/TEP_Mode3.h5](../../data/public_datasets/flow_matching/tep_dtu_v1/TEP_Mode3.h5) |
| jfi_tep_26024015 | 原始/基础公开数据或作者数据包 | https://ndownloader.figshare.com/files/26024015?download=1&cachebust=1790850549819016400: HTTPSConnectionPool(host='ndownloader.figshare.com', port=443): Max retries exceeded with url: /files/26024015?download=1&cachebust=1790850549819016400 (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1010)'))) | [data/public_datasets/flow_matching/tep_dtu_v1/TEP_Mode4.h5](../../data/public_datasets/flow_matching/tep_dtu_v1/TEP_Mode4.h5) |
| jfi_tep_26028068 | 原始/基础公开数据或作者数据包 | https://ndownloader.figshare.com/files/26028068?download=1&cachebust=1790850549819016400: HTTPSConnectionPool(host='ndownloader.figshare.com', port=443): Max retries exceeded with url: /files/26028068?download=1&cachebust=1790850549819016400 (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1010)'))) | [data/public_datasets/flow_matching/tep_dtu_v1/TEP_Mode5.h5](../../data/public_datasets/flow_matching/tep_dtu_v1/TEP_Mode5.h5) |
| jfi_tep_26034659 | 原始/基础公开数据或作者数据包 | https://ndownloader.figshare.com/files/26034659?download=1&cachebust=1790850549819016400: HTTPSConnectionPool(host='ndownloader.figshare.com', port=443): Max retries exceeded with url: /files/26034659?download=1&cachebust=1790850549819016400 (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1010)'))) | [data/public_datasets/flow_matching/tep_dtu_v1/TEP_Mode6.h5](../../data/public_datasets/flow_matching/tep_dtu_v1/TEP_Mode6.h5) |
| jfi_tep_26775368 | 原始/基础公开数据或作者数据包 | https://ndownloader.figshare.com/files/26775368?download=1&cachebust=1790850549820197500: HTTPSConnectionPool(host='s3q.ait.dtu.dk', port=9000): Max retries exceeded with url: /figshare/26775368/Readme.html?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=00000005001330512eb1/20261001/oha/s3/aws4_request&X-Amz-Date=20261001T102912Z&X-Amz-Expires=10&X-Amz-SignedHeaders=host&X-Amz-Signature=b493280c5f1891c9030cf30ba42edd69fbc1677b81a9982daf69516fbca6b80e (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1010)'))) | [data/public_datasets/flow_matching/tep_dtu_v1/Readme.html](../../data/public_datasets/flow_matching/tep_dtu_v1/Readme.html) |

### CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/csdi.pdf](../../papers/literature/flow_matching/pdfs/csdi.pdf)

实验来源：[原论文/官方证据](https://arxiv.org/html/2107.03502)。

论文列明实验数据集：PhysioNet2012 set-a；STMVL Beijing PM2.5 36 stations；solar_nips；electricity_nips；traffic_nips；taxi_30min；wiki2000_nips。

当前可用资料分类：原始/基础公开数据或作者数据包 6 项。

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| physionet2012_set_a | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/physionet2012/set-a.tar.gz](../../data/public_datasets/flow_matching/physionet2012/set-a.tar.gz) |
| beijing_stmvl | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/air_quality/STMVL-Release.zip](../../data/public_datasets/flow_matching/air_quality/STMVL-Release.zip) |
| solar_nips | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/forecasting/solar_nips.tar.gz](../../data/public_datasets/flow_matching/forecasting/solar_nips.tar.gz) |
| electricity_nips | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/forecasting/electricity_nips.tar.gz](../../data/public_datasets/flow_matching/forecasting/electricity_nips.tar.gz) |
| traffic_nips | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/forecasting/traffic_nips.tar.gz](../../data/public_datasets/flow_matching/forecasting/traffic_nips.tar.gz) |
| taxi_30min | 原始/基础公开数据或作者数据包 | https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/mv_release/datasets/taxi_30min.tar.gz:                                                                                  [#8c1875 23MiB/43MiB(53%) CN:1 DL:0B]                                                                                 [#8c1875 23MiB/43MiB(53%) CN:1 DL:0B] 10/01 18:30:00 [[1;31mERROR[0m] CUID#8 - Download aborted. URI=https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/mv_release/datasets/taxi_30min.tar.gz Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/mv_release/datasets/taxi_30min.tar.gz   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 8c1875\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/forecasting/taxi_30min.tar.gz.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/forecasting/taxi_30min.tar.gz](../../data/public_datasets/flow_matching/forecasting/taxi_30min.tar.gz) |
| wiki2000_nips | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/forecasting/wiki2000_nips.tar.gz](../../data/public_datasets/flow_matching/forecasting/wiki2000_nips.tar.gz) |

### Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/sssd.pdf](../../papers/literature/flow_matching/pdfs/sssd.pdf)

实验来源：[原论文/官方证据](https://github.com/AI4HealthUOL/SSSD/tree/main/docs/instructions)。

论文列明实验数据集：Electricity；ETTm1；MuJoCo；PTB-XL 1.0.1；METR-LA；PEMS-BAY。

当前可用资料分类：原始/基础公开数据或作者数据包 5 项。

- **notes**：src/get_data.py covers first four; traffic sources/preprocessing are documented separately.

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| electricity_load_diagrams | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/electricity/electricityloaddiagrams20112014.zip](../../data/public_datasets/flow_matching/electricity/electricityloaddiagrams20112014.zip) |
| ettm1 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/ett/ETTm1.csv](../../data/public_datasets/flow_matching/ett/ETTm1.csv) |
| nrtsi_mujoco_train | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/mujoco/mujoco_train.npy](../../data/public_datasets/flow_matching/mujoco/mujoco_train.npy) |
| nrtsi_mujoco_test | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/mujoco/mujoco_test.npy](../../data/public_datasets/flow_matching/mujoco/mujoco_test.npy) |
| ptbxl_1_0_1 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/ptbxl/ptbxl_1_0_1.zip](../../data/public_datasets/flow_matching/ptbxl/ptbxl_1_0_1.zip) |
| grin_public_bundle | 原始/基础公开数据或作者数据包 | https://mega.nz/folder/qwwG3Qba#c6qFTeT7apmZKKyEunCzSg: HTML returned instead of dataset | [data/public_datasets/flow_matching/grin/public_bundle](../../data/public_datasets/flow_matching/grin/public_bundle) |
| sssd_processed_bundle | 公开预处理/作者包（精确实验版本未确认） | https://mega.nz/folder/kT91jYpI#97GyTkVVUk97fzs1Oy4nBQ: HTML returned instead of dataset | [data/public_datasets/flow_matching/sssd/processed_bundle](../../data/public_datasets/flow_matching/sssd/processed_bundle) |
| dcrnn_traffic_bundle | 原始/基础公开数据或作者数据包 | https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX: HTML returned instead of dataset | [data/public_datasets/flow_matching/traffic/dcrnn_bundle](../../data/public_datasets/flow_matching/traffic/dcrnn_bundle) |
| drive_1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/traffic/dcrnn_bundle/metr-la.h5](../../data/public_datasets/flow_matching/traffic/dcrnn_bundle/metr-la.h5) |
| drive_1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/traffic/dcrnn_bundle/pems-bay.h5](../../data/public_datasets/flow_matching/traffic/dcrnn_bundle/pems-bay.h5) |
| sssd_individual_1 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/sssd/processed_bundle/individual_files](../../data/public_datasets/flow_matching/sssd/processed_bundle/individual_files) |
| sssd_individual_2 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/sssd/processed_bundle/individual_files](../../data/public_datasets/flow_matching/sssd/processed_bundle/individual_files) |
| sssd_individual_3 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/sssd/processed_bundle/individual_files](../../data/public_datasets/flow_matching/sssd/processed_bundle/individual_files) |
| sssd_individual_4 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/sssd/processed_bundle/individual_files](../../data/public_datasets/flow_matching/sssd/processed_bundle/individual_files) |
| sssd_individual_5 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/sssd/processed_bundle/individual_files](../../data/public_datasets/flow_matching/sssd/processed_bundle/individual_files) |
| sssd_individual_6 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/sssd/processed_bundle/individual_files](../../data/public_datasets/flow_matching/sssd/processed_bundle/individual_files) |
| sssd_individual_7 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/sssd/processed_bundle/individual_files](../../data/public_datasets/flow_matching/sssd/processed_bundle/individual_files) |
| sssd_individual_8 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/sssd/processed_bundle/individual_files](../../data/public_datasets/flow_matching/sssd/processed_bundle/individual_files) |
| sssd_individual_9 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/sssd/processed_bundle/individual_files](../../data/public_datasets/flow_matching/sssd/processed_bundle/individual_files) |
| sssd_individual_10 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/sssd/processed_bundle/individual_files](../../data/public_datasets/flow_matching/sssd/processed_bundle/individual_files) |
| sssd_individual_11 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/sssd/processed_bundle/individual_files](../../data/public_datasets/flow_matching/sssd/processed_bundle/individual_files) |
| sssd_individual_12 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/sssd/processed_bundle/individual_files](../../data/public_datasets/flow_matching/sssd/processed_bundle/individual_files) |
| sssd_processed_bundle:data/public_datasets/flow_matching/sssd/processed_bundle/PTB-XL/lbl_itos.npy | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/sssd/processed_bundle/PTB-XL/lbl_itos.npy](../../data/public_datasets/flow_matching/sssd/processed_bundle/PTB-XL/lbl_itos.npy) |

### BRITS: Bidirectional Recurrent Imputation for Time Series

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/brits.pdf](../../papers/literature/flow_matching/pdfs/brits.pdf)

实验来源：[原论文/官方证据](https://arxiv.org/html/1805.10572)。

论文列明实验数据集：STMVL Beijing PM2.5 36 stations；PhysioNet2012 set-a；UCI Localization Data for Person Activity。

当前可用资料分类：原始/基础公开数据或作者数据包 4 项。

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| physionet2012_set_a | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/physionet2012/set-a.tar.gz](../../data/public_datasets/flow_matching/physionet2012/set-a.tar.gz) |
| physionet2012_outcomes_a | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/physionet2012/Outcomes-a.txt](../../data/public_datasets/flow_matching/physionet2012/Outcomes-a.txt) |
| beijing_stmvl | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/air_quality/STMVL-Release.zip](../../data/public_datasets/flow_matching/air_quality/STMVL-Release.zip) |
| human_activity_localization | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/activity/localization_person_activity.zip](../../data/public_datasets/flow_matching/activity/localization_person_activity.zip) |

### SAITS: Self-Attention-based Imputation for Time Series

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/saits.pdf](../../papers/literature/flow_matching/pdfs/saits.pdf)

实验来源：[原论文/官方证据](https://arxiv.org/html/2202.08516)。

论文列明实验数据集：PhysioNet2012 sets a,b,c and outcomes；UCI Beijing Multi-Site Air Quality；Electricity Load Diagrams 2011-2014；ETTm1；NRTSI Air；NRTSI Gas。

当前可用资料分类：原始/基础公开数据或作者数据包 15 项。

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| physionet2012_set_a | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/physionet2012/set-a.tar.gz](../../data/public_datasets/flow_matching/physionet2012/set-a.tar.gz) |
| physionet2012_outcomes_a | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/physionet2012/Outcomes-a.txt](../../data/public_datasets/flow_matching/physionet2012/Outcomes-a.txt) |
| physionet2012_set_b | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/physionet2012/set-b.tar.gz](../../data/public_datasets/flow_matching/physionet2012/set-b.tar.gz) |
| physionet2012_outcomes_b | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/physionet2012/Outcomes-b.txt](../../data/public_datasets/flow_matching/physionet2012/Outcomes-b.txt) |
| physionet2012_set_c | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/physionet2012/set-c.tar.gz](../../data/public_datasets/flow_matching/physionet2012/set-c.tar.gz) |
| physionet2012_outcomes_c | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/physionet2012/Outcomes-c.txt](../../data/public_datasets/flow_matching/physionet2012/Outcomes-c.txt) |
| beijing_multisite | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/air_quality/beijing_multisite.zip](../../data/public_datasets/flow_matching/air_quality/beijing_multisite.zip) |
| electricity_load_diagrams | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/electricity/electricityloaddiagrams20112014.zip](../../data/public_datasets/flow_matching/electricity/electricityloaddiagrams20112014.zip) |
| ettm1 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/ett/ETTm1.csv](../../data/public_datasets/flow_matching/ett/ETTm1.csv) |
| nrtsi_air_quality_train | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/nrtsi/air_quality_train.npy](../../data/public_datasets/flow_matching/nrtsi/air_quality_train.npy) |
| nrtsi_air_quality_val | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/nrtsi/air_quality_val.npy](../../data/public_datasets/flow_matching/nrtsi/air_quality_val.npy) |
| nrtsi_air_quality_test | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/nrtsi/air_quality_test.npy](../../data/public_datasets/flow_matching/nrtsi/air_quality_test.npy) |
| nrtsi_gas_train | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/nrtsi/gas_train.npy](../../data/public_datasets/flow_matching/nrtsi/gas_train.npy) |
| nrtsi_gas_val | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/nrtsi/gas_val.npy](../../data/public_datasets/flow_matching/nrtsi/gas_val.npy) |
| nrtsi_gas_test | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/nrtsi/gas_test.npy](../../data/public_datasets/flow_matching/nrtsi/gas_test.npy) |

### Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/grin.pdf](../../papers/literature/flow_matching/pdfs/grin.pdf)

实验来源：[原论文/官方证据](https://arxiv.org/html/2108.00298)。

论文列明实验数据集：AQI-437；AQI-36；METR-LA；PEMS-BAY；CER-E SME subset；synthetic VAR。

当前可用资料分类：原始/基础公开数据或作者数据包 1 项。

- **synthetic**：{"source_url": "https://github.com/Graph-Machine-Learning-Group/grin/blob/main/lib/datasets/synthetic.py", "runner_url": "https://github.com/Graph-Machine-Learning-Group/grin/blob/main/scripts/run_synthetic.py", "method": "Graph-based vector autoregressive process; use official script/config for exact coefficients and seeds."}

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| beijing_stmvl | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/air_quality/STMVL-Release.zip](../../data/public_datasets/flow_matching/air_quality/STMVL-Release.zip) |
| grin_public_bundle | 原始/基础公开数据或作者数据包 | https://mega.nz/folder/qwwG3Qba#c6qFTeT7apmZKKyEunCzSg: HTML returned instead of dataset | [data/public_datasets/flow_matching/grin/public_bundle](../../data/public_datasets/flow_matching/grin/public_bundle) |
| dcrnn_traffic_bundle | 原始/基础公开数据或作者数据包 | https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX: HTML returned instead of dataset | [data/public_datasets/flow_matching/traffic/dcrnn_bundle](../../data/public_datasets/flow_matching/traffic/dcrnn_bundle) |
| cer_e | 受限/未公开 | https://www.ucd.ie/issda/data/commissionforenergyregulationcer/:  10/01 18:30:34 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://www.ucd.ie/issda/data/commissionforenergyregulationcer/ Exception: [AbstractCommand.cc:351] errorCode=22 URI=https://www.ucd.ie/issda/data/commissionforenergyregulationcer/   -> [HttpSkipResponseCommand.cc:239] errorCode=22 The response status is not successful. status=403  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= dfa5f4\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/cer_e/pending.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/cer_e/pending](../../data/public_datasets/flow_matching/cer_e/pending) |
| drive_1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/traffic/dcrnn_bundle/metr-la.h5](../../data/public_datasets/flow_matching/traffic/dcrnn_bundle/metr-la.h5) |
| drive_1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/traffic/dcrnn_bundle/pems-bay.h5](../../data/public_datasets/flow_matching/traffic/dcrnn_bundle/pems-bay.h5) |

### Diffusion-TS: Interpretable Diffusion for General Time Series Generation

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/diffusion_ts.pdf](../../papers/literature/flow_matching/pdfs/diffusion_ts.pdf)

实验来源：[原论文/官方证据](https://github.com/Y-debug-sys/Diffusion-TS)。

论文列明实验数据集：Stocks；ETTh1；Energy；fMRI；Sine；MuJoCo。

当前可用资料分类：原始/基础公开数据或作者数据包 2 项。

- **notes**：Paper Table 11 lists MuJoCo dimension 14 and 10000 samples; current repository generator defaults to 12 dimensions/30000 samples. Match frozen paper/version configuration before regeneration.

- **synthetic**：[{"source_url": "https://github.com/Y-debug-sys/Diffusion-TS/blob/main/Utils/Data_utils/sine_dataset.py", "method": "Uniform frequency and phase in [0,0.1], (sin(freq*t+phase)+1)/2, default seed 123; configurations specify sample count/length/dimension."}, {"source_url": "https://github.com/Y-debug-sys/Diffusion-TS/blob/main/Utils/Data_utils/mujoco_dataset.py", "method": "dm_control suite.load(hopper,stand), uniformly initialized qpos/qvel, physics.step(), 12 qpos/qvel channels, default seed 123; separate from downloaded NRTSI MuJoCo."}]

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| etth1 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/ett/ETTh1.csv](../../data/public_datasets/flow_matching/ett/ETTh1.csv) |
| diffusion_ts_realworld_bundle | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/diffusion_ts/dataset.zip](../../data/public_datasets/flow_matching/diffusion_ts/dataset.zip) |
| diffusion_ts_eeg_extension | 原始/基础公开数据或作者数据包 | https://drive.google.com/uc?export=download&id=1IqwE0wbCT1orVdZpul2xFiNkGnYs4t89: File is not a zip file | [data/public_datasets/flow_matching/diffusion_ts/eeg.zip](../../data/public_datasets/flow_matching/diffusion_ts/eeg.zip) |

### Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/ganf.pdf](../../papers/literature/flow_matching/pdfs/ganf.pdf)

实验来源：[原论文/官方证据](https://github.com/EnyanDai/GANF#datasets)。

论文列明实验数据集：SWaT Dec2015 attack_v0；METR-LA；PMU proprietary。

当前可用资料分类：没有已完成且现场可用的注册资料。

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| dcrnn_traffic_bundle | 原始/基础公开数据或作者数据包 | https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX: HTML returned instead of dataset | [data/public_datasets/flow_matching/traffic/dcrnn_bundle](../../data/public_datasets/flow_matching/traffic/dcrnn_bundle) |
| ganf_pmu | 受限/未公开 | proprietary_not_released | [data/public_datasets/flow_matching/pmu/unavailable](../../data/public_datasets/flow_matching/pmu/unavailable) |
| existing_mtsad_swat | 原始/基础公开数据或作者数据包 | missing or empty payload | [data/public_datasets/mtsad/swat](../../data/public_datasets/mtsad/swat) |
| drive_1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/traffic/dcrnn_bundle/metr-la.h5](../../data/public_datasets/flow_matching/traffic/dcrnn_bundle/metr-la.h5) |
| drive_1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/traffic/dcrnn_bundle/pems-bay.h5](../../data/public_datasets/flow_matching/traffic/dcrnn_bundle/pems-bay.h5) |

### CATCH: Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/catch.pdf](../../papers/literature/flow_matching/pdfs/catch.pdf)

实验来源：[原论文/官方证据](https://arxiv.org/html/2410.12261v2#A1.SS1)。

论文列明实验数据集：MSL；PSM；SMD；CICIDS；CalIt2；NYC；Creditcard；GECCO；Genesis；ASD；TODS synthetic 12 sets。

当前可用资料分类：公开预处理/作者包（精确实验版本未确认） 116 项；生成配方/源码 1 项。

- **synthetic**：{"evidence_url": "https://arxiv.org/html/2410.12261v2#A1.SS6", "source_url": "https://github.com/datamllab/tods", "method": "MultivariateDataGenerator, 5 channels alternating sine/cosine, freq .04, coef 1.5, noise_amp .05, training length 20000, test length 5000; 2 rates each for global/contextual/seasonal/shapelet/trend/mixture anomalies. Follow Appendix A.6 parameters; paper excerpt gives no random seed, so newly generated data cannot be asserted byte-identical to released sets."}

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| mtsbench_SMD_SMD_machine-1-2_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-2_test.csv) |
| mtsbench_SMD_SMD_machine-1-2_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-2_train.csv) |
| mtsbench_SMD_SMD_machine-1-2_val.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-2_val.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-2_val.csv) |
| mtsbench_SMD_SMD_machine-1-3_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-3_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-3_test.csv) |
| mtsbench_SMD_SMD_machine-1-3_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-3_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-3_train.csv) |
| mtsbench_SMD_SMD_machine-1-4_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-4_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-4_test.csv) |
| mtsbench_SMD_SMD_machine-1-4_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-4_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-4_train.csv) |
| mtsbench_SMD_SMD_machine-1-6_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-6_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-6_test.csv) |
| mtsbench_SMD_SMD_machine-1-6_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-6_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-6_train.csv) |
| mtsbench_SMD_SMD_machine-1-7_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-7_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-7_test.csv) |
| mtsbench_SMD_SMD_machine-1-7_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-7_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-7_train.csv) |
| mtsbench_SMD_SMD_machine-1-8_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-8_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-8_test.csv) |
| mtsbench_SMD_SMD_machine-1-8_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-8_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-8_train.csv) |
| mtsbench_SMD_SMD_machine-2-1_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-1_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-1_test.csv) |
| mtsbench_SMD_SMD_machine-2-1_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-1_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-1_train.csv) |
| mtsbench_SMD_SMD_machine-2-2_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-2_test.csv) |
| mtsbench_SMD_SMD_machine-2-2_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-2_train.csv) |
| mtsbench_SMD_SMD_machine-2-3_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-3_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-3_test.csv) |
| mtsbench_SMD_SMD_machine-2-3_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-3_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-3_train.csv) |
| mtsbench_SMD_SMD_machine-2-4_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-4_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-4_test.csv) |
| mtsbench_SMD_SMD_machine-2-4_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-4_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-4_train.csv) |
| mtsbench_SMD_SMD_machine-2-5_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-5_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-5_test.csv) |
| mtsbench_SMD_SMD_machine-2-5_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-5_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-5_train.csv) |
| mtsbench_SMD_SMD_machine-2-7_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-7_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-7_test.csv) |
| mtsbench_SMD_SMD_machine-2-7_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-7_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-7_train.csv) |
| mtsbench_SMD_SMD_machine-2-9_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-9_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-9_test.csv) |
| mtsbench_SMD_SMD_machine-2-9_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-9_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-9_train.csv) |
| mtsbench_SMD_SMD_machine-3-10_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-10_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-10_test.csv) |
| mtsbench_SMD_SMD_machine-3-10_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-10_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-10_train.csv) |
| mtsbench_SMD_SMD_machine-3-2_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-2_test.csv) |
| mtsbench_SMD_SMD_machine-3-2_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-2_train.csv) |
| mtsbench_SMD_SMD_machine-3-3_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-3_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-3_test.csv) |
| mtsbench_SMD_SMD_machine-3-3_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-3_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-3_train.csv) |
| mtsbench_SMD_SMD_machine-3-5_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-5_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-5_test.csv) |
| mtsbench_SMD_SMD_machine-3-5_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-5_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-5_train.csv) |
| mtsbench_SMD_SMD_machine-3-6_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-6_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-6_test.csv) |
| mtsbench_SMD_SMD_machine-3-6_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-6_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-6_train.csv) |
| mtsbench_cicids_cicids_0_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_0_test.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_0_test.csv) |
| mtsbench_cicids_cicids_0_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_0_train.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_0_train.csv) |
| mtsbench_cicids_cicids_0_val.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_0_val.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_0_val.csv) |
| mtsbench_cicids_cicids_2_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_2_test.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_2_test.csv) |
| mtsbench_cicids_cicids_2_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_2_train.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_2_train.csv) |
| mtsbench_cicids_cicids_3_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_3_test.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_3_test.csv) |
| mtsbench_cicids_cicids_3_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_3_train.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_3_train.csv) |
| mtsbench_cicids_cicids_5_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_5_test.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_5_test.csv) |
| mtsbench_cicids_cicids_5_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_5_train.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_5_train.csv) |
| mtsbench_cicids_cicids_6_test.csv | 公开预处理/作者包（精确实验版本未确认） | https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/cicids/cicids_6_test.csv:                                                                                  [#3dc9a1 0B/0B CN:1 DL:0B]                                                                                 [#3dc9a1 0B/0B CN:1 DL:0B]                                                                                 [#3dc9a1 0B/0B CN:1 DL:0B] 10/01 18:29:13 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/cicids/cicids_6_test.csv Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/cicids/cicids_6_test.csv   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 3dc9a1\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/mtsbench/cicids/cicids_6_test.csv.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_6_test.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_6_test.csv) |
| mtsbench_cicids_cicids_6_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_6_train.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_6_train.csv) |
| mtsbench_cicids_cicids_7_test.csv | 公开预处理/作者包（精确实验版本未确认） | https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/cicids/cicids_7_test.csv:                                                                                  [#aa79fe 0B/0B CN:1 DL:0B]                                                                                 [#aa79fe 0B/0B CN:1 DL:0B] 10/01 18:29:16 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/cicids/cicids_7_test.csv Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/cicids/cicids_7_test.csv   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= aa79fe\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/mtsbench/cicids/cicids_7_test.csv.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_7_test.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_7_test.csv) |
| mtsbench_cicids_cicids_7_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/cicids/cicids_7_train.csv](../../data/public_datasets/flow_matching/mtsbench/cicids/cicids_7_train.csv) |
| catch_bundle | 原始/基础公开数据或作者数据包 | https://1drv.ms/u/c/801ce36c4ff3f93b/EVTDLHyvegpEn_Oxa6ZiuFIBjTsKk6m9JldUqWDqvrVCnQ?e=P2T3Vc:                                                                                  [#c08a9d 0B/0B CN:1 DL:0B] 10/01 18:30:36 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://1drv.ms/u/c/801ce36c4ff3f93b/EVTDLHyvegpEn_Oxa6ZiuFIBjTsKk6m9JldUqWDqvrVCnQ?e=P2T3Vc Exception: [AbstractCommand.cc:351] errorCode=22 URI=https://onedrive.live.com/:u:/g/personal/801CE36C4FF3F93B/EVTDLHyvegpEn_Oxa6ZiuFIBjTsKk6m9JldUqWDqvrVCnQ?resid=801CE36C4FF3F93B!s7c2cc3547aaf440a9ff3b16ba662b852&e=P2T3Vc&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvYy84MDFjZTM2YzRmZjNmOTNiL0VWVERMSHl2ZWdwRW5fT3hhNlppdUZJQmpUc0trNm05SmxkVXFXRHF2clZDblE_ZT1QMlQzVmM   -> [HttpSkipResponseCommand.cc:239] errorCode=22 The response status is not successful. status=403  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= c08a9d\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/catch/author_bundle.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/catch/author_bundle](../../data/public_datasets/flow_matching/catch/author_bundle) |
| existing_mtsad_smd | 原始/基础公开数据或作者数据包 | missing or empty payload | [data/public_datasets/mtsad/smd](../../data/public_datasets/mtsad/smd) |
| existing_mtsad_msl | 原始/基础公开数据或作者数据包 | missing or empty payload | [data/public_datasets/mtsad/msl](../../data/public_datasets/mtsad/msl) |
| existing_mtsad_psm | 原始/基础公开数据或作者数据包 | missing or empty payload | [data/public_datasets/mtsad/psm](../../data/public_datasets/mtsad/psm) |
| tods_generator | 生成配方/源码 | 可用 | [data/public_datasets/flow_matching/catch/multivariate_generator.py](../../data/public_datasets/flow_matching/catch/multivariate_generator.py) |
| mtsbench_CalIt2_CalIt2_traffic_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/CalIt2/CalIt2_traffic_test.csv](../../data/public_datasets/flow_matching/mtsbench/CalIt2/CalIt2_traffic_test.csv) |
| mtsbench_CalIt2_CalIt2_traffic_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/CalIt2/CalIt2_traffic_train.csv](../../data/public_datasets/flow_matching/mtsbench/CalIt2/CalIt2_traffic_train.csv) |
| mtsbench_CalIt2_CalIt2_traffic_val.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/CalIt2/CalIt2_traffic_val.csv](../../data/public_datasets/flow_matching/mtsbench/CalIt2/CalIt2_traffic_val.csv) |
| mtsbench_GECCO_GECCO_water_quality_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/GECCO/GECCO_water_quality_test.csv](../../data/public_datasets/flow_matching/mtsbench/GECCO/GECCO_water_quality_test.csv) |
| mtsbench_GECCO_GECCO_water_quality_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/GECCO/GECCO_water_quality_train.csv](../../data/public_datasets/flow_matching/mtsbench/GECCO/GECCO_water_quality_train.csv) |
| mtsbench_GECCO_GECCO_water_quality_val.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/GECCO/GECCO_water_quality_val.csv](../../data/public_datasets/flow_matching/mtsbench/GECCO/GECCO_water_quality_val.csv) |
| mtsbench_Genesis_Genesis_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/Genesis/Genesis_test.csv](../../data/public_datasets/flow_matching/mtsbench/Genesis/Genesis_test.csv) |
| mtsbench_Genesis_Genesis_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/Genesis/Genesis_train.csv](../../data/public_datasets/flow_matching/mtsbench/Genesis/Genesis_train.csv) |
| mtsbench_Genesis_Genesis_val.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/Genesis/Genesis_val.csv](../../data/public_datasets/flow_matching/mtsbench/Genesis/Genesis_val.csv) |
| mtsbench_MSL_MSL_C-1_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_C-1_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_C-1_test.csv) |
| mtsbench_MSL_MSL_C-1_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_C-1_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_C-1_train.csv) |
| mtsbench_MSL_MSL_C-2_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_C-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_C-2_test.csv) |
| mtsbench_MSL_MSL_C-2_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_C-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_C-2_train.csv) |
| mtsbench_MSL_MSL_D-14_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-14_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-14_test.csv) |
| mtsbench_MSL_MSL_D-14_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-14_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-14_train.csv) |
| mtsbench_MSL_MSL_D-15_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-15_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-15_test.csv) |
| mtsbench_MSL_MSL_D-15_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-15_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-15_train.csv) |
| mtsbench_MSL_MSL_D-16_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-16_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-16_test.csv) |
| mtsbench_MSL_MSL_D-16_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-16_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-16_train.csv) |
| mtsbench_MSL_MSL_F-4_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-4_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-4_test.csv) |
| mtsbench_MSL_MSL_F-4_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-4_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-4_train.csv) |
| mtsbench_MSL_MSL_F-5_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-5_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-5_test.csv) |
| mtsbench_MSL_MSL_F-5_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-5_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-5_train.csv) |
| mtsbench_MSL_MSL_F-7_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-7_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-7_test.csv) |
| mtsbench_MSL_MSL_F-7_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-7_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-7_train.csv) |
| mtsbench_MSL_MSL_F-8_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-8_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-8_test.csv) |
| mtsbench_MSL_MSL_F-8_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-8_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-8_train.csv) |
| mtsbench_MSL_MSL_M-1_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-1_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-1_test.csv) |
| mtsbench_MSL_MSL_M-1_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-1_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-1_train.csv) |
| mtsbench_MSL_MSL_M-2_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-2_test.csv) |
| mtsbench_MSL_MSL_M-2_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-2_train.csv) |
| mtsbench_MSL_MSL_M-3_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-3_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-3_test.csv) |
| mtsbench_MSL_MSL_M-3_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-3_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-3_train.csv) |
| mtsbench_MSL_MSL_M-4_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-4_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-4_test.csv) |
| mtsbench_MSL_MSL_M-4_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-4_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-4_train.csv) |
| mtsbench_MSL_MSL_M-5_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-5_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-5_test.csv) |
| mtsbench_MSL_MSL_M-5_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-5_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-5_train.csv) |
| mtsbench_MSL_MSL_M-6_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-6_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-6_test.csv) |
| mtsbench_MSL_MSL_M-6_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-6_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-6_train.csv) |
| mtsbench_MSL_MSL_M-7_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-7_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-7_test.csv) |
| mtsbench_MSL_MSL_M-7_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-7_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-7_train.csv) |
| mtsbench_MSL_MSL_P-10_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-10_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-10_test.csv) |
| mtsbench_MSL_MSL_P-10_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-10_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-10_train.csv) |
| mtsbench_MSL_MSL_P-11_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-11_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-11_test.csv) |
| mtsbench_MSL_MSL_P-11_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-11_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-11_train.csv) |
| mtsbench_MSL_MSL_P-14_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-14_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-14_test.csv) |
| mtsbench_MSL_MSL_P-14_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-14_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-14_train.csv) |
| mtsbench_MSL_MSL_P-15_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-15_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-15_test.csv) |
| mtsbench_MSL_MSL_P-15_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-15_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-15_train.csv) |
| mtsbench_MSL_MSL_S-2_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_S-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_S-2_test.csv) |
| mtsbench_MSL_MSL_S-2_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_S-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_S-2_train.csv) |
| mtsbench_MSL_MSL_T-12_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-12_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-12_test.csv) |
| mtsbench_MSL_MSL_T-12_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-12_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-12_train.csv) |
| mtsbench_MSL_MSL_T-13_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-13_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-13_test.csv) |
| mtsbench_MSL_MSL_T-13_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-13_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-13_train.csv) |
| mtsbench_MSL_MSL_T-13_val.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-13_val.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-13_val.csv) |
| mtsbench_MSL_MSL_T-4_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-4_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-4_test.csv) |
| mtsbench_MSL_MSL_T-4_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-4_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-4_train.csv) |
| mtsbench_MSL_MSL_T-5_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-5_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-5_test.csv) |
| mtsbench_MSL_MSL_T-5_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-5_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-5_train.csv) |
| mtsbench_MSL_MSL_T-8_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-8_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-8_test.csv) |
| mtsbench_MSL_MSL_T-8_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-8_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-8_train.csv) |
| mtsbench_PSM_PSM_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/PSM/PSM_test.csv](../../data/public_datasets/flow_matching/mtsbench/PSM/PSM_test.csv) |
| mtsbench_PSM_PSM_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/PSM/PSM_train.csv](../../data/public_datasets/flow_matching/mtsbench/PSM/PSM_train.csv) |
| mtsbench_PSM_PSM_val.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/PSM/PSM_val.csv](../../data/public_datasets/flow_matching/mtsbench/PSM/PSM_val.csv) |
| mtsbench_creditcard_creditcard_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/creditcard/creditcard_test.csv](../../data/public_datasets/flow_matching/mtsbench/creditcard/creditcard_test.csv) |
| mtsbench_creditcard_creditcard_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/creditcard/creditcard_train.csv](../../data/public_datasets/flow_matching/mtsbench/creditcard/creditcard_train.csv) |
| mtsbench_creditcard_creditcard_val.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/creditcard/creditcard_val.csv](../../data/public_datasets/flow_matching/mtsbench/creditcard/creditcard_val.csv) |
| tab_author_dataset_zip | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/catch/TAB_dataset.zip](../../data/public_datasets/flow_matching/catch/TAB_dataset.zip) |

### DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/dcdetector.pdf](../../papers/literature/flow_matching/pdfs/dcdetector.pdf)

实验来源：[原论文/官方证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md)。

论文列明实验数据集：SMD；MSL；SMAP；PSM；SWaT；NIPS-TS-SWAN；NIPS-TS-GECCO (repository NIPS_TS_Water)；UCR。

当前可用资料分类：公开预处理/作者包（精确实验版本未确认） 96 项；原始/基础公开数据或作者数据包 7 项。

- **notes**：Paper Section 4.1 names NIPS-TS-GECCO; repository uses NIPS_TS_Water script. Do not confuse with SWaT.

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| mtsbench_SMD_SMD_machine-1-2_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-2_test.csv) |
| mtsbench_SMD_SMD_machine-1-2_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-2_train.csv) |
| mtsbench_SMD_SMD_machine-1-2_val.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-2_val.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-2_val.csv) |
| mtsbench_SMD_SMD_machine-1-3_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-3_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-3_test.csv) |
| mtsbench_SMD_SMD_machine-1-3_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-3_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-3_train.csv) |
| mtsbench_SMD_SMD_machine-1-4_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-4_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-4_test.csv) |
| mtsbench_SMD_SMD_machine-1-4_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-4_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-4_train.csv) |
| mtsbench_SMD_SMD_machine-1-6_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-6_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-6_test.csv) |
| mtsbench_SMD_SMD_machine-1-6_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-6_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-6_train.csv) |
| mtsbench_SMD_SMD_machine-1-7_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-7_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-7_test.csv) |
| mtsbench_SMD_SMD_machine-1-7_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-7_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-7_train.csv) |
| mtsbench_SMD_SMD_machine-1-8_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-8_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-8_test.csv) |
| mtsbench_SMD_SMD_machine-1-8_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-8_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-1-8_train.csv) |
| mtsbench_SMD_SMD_machine-2-1_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-1_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-1_test.csv) |
| mtsbench_SMD_SMD_machine-2-1_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-1_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-1_train.csv) |
| mtsbench_SMD_SMD_machine-2-2_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-2_test.csv) |
| mtsbench_SMD_SMD_machine-2-2_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-2_train.csv) |
| mtsbench_SMD_SMD_machine-2-3_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-3_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-3_test.csv) |
| mtsbench_SMD_SMD_machine-2-3_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-3_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-3_train.csv) |
| mtsbench_SMD_SMD_machine-2-4_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-4_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-4_test.csv) |
| mtsbench_SMD_SMD_machine-2-4_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-4_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-4_train.csv) |
| mtsbench_SMD_SMD_machine-2-5_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-5_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-5_test.csv) |
| mtsbench_SMD_SMD_machine-2-5_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-5_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-5_train.csv) |
| mtsbench_SMD_SMD_machine-2-7_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-7_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-7_test.csv) |
| mtsbench_SMD_SMD_machine-2-7_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-7_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-7_train.csv) |
| mtsbench_SMD_SMD_machine-2-9_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-9_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-9_test.csv) |
| mtsbench_SMD_SMD_machine-2-9_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-9_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-2-9_train.csv) |
| mtsbench_SMD_SMD_machine-3-10_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-10_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-10_test.csv) |
| mtsbench_SMD_SMD_machine-3-10_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-10_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-10_train.csv) |
| mtsbench_SMD_SMD_machine-3-2_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-2_test.csv) |
| mtsbench_SMD_SMD_machine-3-2_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-2_train.csv) |
| mtsbench_SMD_SMD_machine-3-3_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-3_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-3_test.csv) |
| mtsbench_SMD_SMD_machine-3-3_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-3_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-3_train.csv) |
| mtsbench_SMD_SMD_machine-3-5_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-5_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-5_test.csv) |
| mtsbench_SMD_SMD_machine-3-5_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-5_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-5_train.csv) |
| mtsbench_SMD_SMD_machine-3-6_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-6_test.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-6_test.csv) |
| mtsbench_SMD_SMD_machine-3-6_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-6_train.csv](../../data/public_datasets/flow_matching/mtsbench/SMD/SMD_machine-3-6_train.csv) |
| mtsbench_swan_swan_sf_test.csv | 公开预处理/作者包（精确实验版本未确认） | https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_test.csv:                                                                                  [#a88e2c 0B/0B CN:1 DL:0B]                                                                                 [#a88e2c 0B/0B CN:1 DL:0B] 10/01 18:29:16 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_test.csv Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_test.csv   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= a88e2c\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/mtsbench/swan/swan_sf_test.csv.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/mtsbench/swan/swan_sf_test.csv](../../data/public_datasets/flow_matching/mtsbench/swan/swan_sf_test.csv) |
| mtsbench_swan_swan_sf_train.csv | 公开预处理/作者包（精确实验版本未确认） | https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_train.csv:                                                                                  [#ed89a6 0B/0B CN:1 DL:0B]                                                                                 [#ed89a6 0B/0B CN:1 DL:0B] 10/01 18:29:16 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_train.csv Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_train.csv   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= ed89a6\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/mtsbench/swan/swan_sf_train.csv.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/mtsbench/swan/swan_sf_train.csv](../../data/public_datasets/flow_matching/mtsbench/swan/swan_sf_train.csv) |
| mtsbench_swan_swan_sf_val.csv | 公开预处理/作者包（精确实验版本未确认） | https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_val.csv:                                                                                  [#eb07e9 0B/0B CN:1 DL:0B]                                                                                 [#eb07e9 0B/0B CN:1 DL:0B] 10/01 18:29:16 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_val.csv Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_val.csv   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= eb07e9\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/mtsbench/swan/swan_sf_val.csv.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [data/public_datasets/flow_matching/mtsbench/swan/swan_sf_val.csv](../../data/public_datasets/flow_matching/mtsbench/swan/swan_sf_val.csv) |
| dcdetector_bundle | 原始/基础公开数据或作者数据包 | https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR: HTML returned instead of dataset | [data/public_datasets/flow_matching/dcdetector/author_bundle](../../data/public_datasets/flow_matching/dcdetector/author_bundle) |
| existing_mtsad_smd | 原始/基础公开数据或作者数据包 | missing or empty payload | [data/public_datasets/mtsad/smd](../../data/public_datasets/mtsad/smd) |
| existing_mtsad_msl | 原始/基础公开数据或作者数据包 | missing or empty payload | [data/public_datasets/mtsad/msl](../../data/public_datasets/mtsad/msl) |
| existing_mtsad_psm | 原始/基础公开数据或作者数据包 | missing or empty payload | [data/public_datasets/mtsad/psm](../../data/public_datasets/mtsad/psm) |
| existing_mtsad_smap | 原始/基础公开数据或作者数据包 | missing or empty payload | [data/public_datasets/mtsad/smap](../../data/public_datasets/mtsad/smap) |
| existing_mtsad_swat | 原始/基础公开数据或作者数据包 | missing or empty payload | [data/public_datasets/mtsad/swat](../../data/public_datasets/mtsad/swat) |
| existing_ucr | 原始/基础公开数据或作者数据包 | missing or empty payload | [data/public_datasets/mtsad_paper_repos/tranad/data/UCR](../../data/public_datasets/mtsad_paper_repos/tranad/data/UCR) |
| mtsbench_GECCO_GECCO_water_quality_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/GECCO/GECCO_water_quality_test.csv](../../data/public_datasets/flow_matching/mtsbench/GECCO/GECCO_water_quality_test.csv) |
| mtsbench_GECCO_GECCO_water_quality_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/GECCO/GECCO_water_quality_train.csv](../../data/public_datasets/flow_matching/mtsbench/GECCO/GECCO_water_quality_train.csv) |
| mtsbench_GECCO_GECCO_water_quality_val.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/GECCO/GECCO_water_quality_val.csv](../../data/public_datasets/flow_matching/mtsbench/GECCO/GECCO_water_quality_val.csv) |
| mtsbench_MSL_MSL_C-1_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_C-1_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_C-1_test.csv) |
| mtsbench_MSL_MSL_C-1_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_C-1_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_C-1_train.csv) |
| mtsbench_MSL_MSL_C-2_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_C-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_C-2_test.csv) |
| mtsbench_MSL_MSL_C-2_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_C-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_C-2_train.csv) |
| mtsbench_MSL_MSL_D-14_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-14_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-14_test.csv) |
| mtsbench_MSL_MSL_D-14_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-14_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-14_train.csv) |
| mtsbench_MSL_MSL_D-15_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-15_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-15_test.csv) |
| mtsbench_MSL_MSL_D-15_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-15_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-15_train.csv) |
| mtsbench_MSL_MSL_D-16_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-16_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-16_test.csv) |
| mtsbench_MSL_MSL_D-16_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-16_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_D-16_train.csv) |
| mtsbench_MSL_MSL_F-4_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-4_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-4_test.csv) |
| mtsbench_MSL_MSL_F-4_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-4_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-4_train.csv) |
| mtsbench_MSL_MSL_F-5_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-5_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-5_test.csv) |
| mtsbench_MSL_MSL_F-5_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-5_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-5_train.csv) |
| mtsbench_MSL_MSL_F-7_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-7_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-7_test.csv) |
| mtsbench_MSL_MSL_F-7_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-7_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-7_train.csv) |
| mtsbench_MSL_MSL_F-8_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-8_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-8_test.csv) |
| mtsbench_MSL_MSL_F-8_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-8_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_F-8_train.csv) |
| mtsbench_MSL_MSL_M-1_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-1_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-1_test.csv) |
| mtsbench_MSL_MSL_M-1_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-1_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-1_train.csv) |
| mtsbench_MSL_MSL_M-2_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-2_test.csv) |
| mtsbench_MSL_MSL_M-2_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-2_train.csv) |
| mtsbench_MSL_MSL_M-3_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-3_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-3_test.csv) |
| mtsbench_MSL_MSL_M-3_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-3_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-3_train.csv) |
| mtsbench_MSL_MSL_M-4_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-4_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-4_test.csv) |
| mtsbench_MSL_MSL_M-4_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-4_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-4_train.csv) |
| mtsbench_MSL_MSL_M-5_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-5_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-5_test.csv) |
| mtsbench_MSL_MSL_M-5_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-5_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-5_train.csv) |
| mtsbench_MSL_MSL_M-6_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-6_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-6_test.csv) |
| mtsbench_MSL_MSL_M-6_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-6_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-6_train.csv) |
| mtsbench_MSL_MSL_M-7_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-7_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-7_test.csv) |
| mtsbench_MSL_MSL_M-7_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-7_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_M-7_train.csv) |
| mtsbench_MSL_MSL_P-10_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-10_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-10_test.csv) |
| mtsbench_MSL_MSL_P-10_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-10_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-10_train.csv) |
| mtsbench_MSL_MSL_P-11_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-11_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-11_test.csv) |
| mtsbench_MSL_MSL_P-11_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-11_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-11_train.csv) |
| mtsbench_MSL_MSL_P-14_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-14_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-14_test.csv) |
| mtsbench_MSL_MSL_P-14_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-14_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-14_train.csv) |
| mtsbench_MSL_MSL_P-15_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-15_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-15_test.csv) |
| mtsbench_MSL_MSL_P-15_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-15_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_P-15_train.csv) |
| mtsbench_MSL_MSL_S-2_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_S-2_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_S-2_test.csv) |
| mtsbench_MSL_MSL_S-2_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_S-2_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_S-2_train.csv) |
| mtsbench_MSL_MSL_T-12_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-12_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-12_test.csv) |
| mtsbench_MSL_MSL_T-12_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-12_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-12_train.csv) |
| mtsbench_MSL_MSL_T-13_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-13_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-13_test.csv) |
| mtsbench_MSL_MSL_T-13_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-13_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-13_train.csv) |
| mtsbench_MSL_MSL_T-13_val.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-13_val.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-13_val.csv) |
| mtsbench_MSL_MSL_T-4_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-4_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-4_test.csv) |
| mtsbench_MSL_MSL_T-4_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-4_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-4_train.csv) |
| mtsbench_MSL_MSL_T-5_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-5_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-5_test.csv) |
| mtsbench_MSL_MSL_T-5_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-5_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-5_train.csv) |
| mtsbench_MSL_MSL_T-8_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-8_test.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-8_test.csv) |
| mtsbench_MSL_MSL_T-8_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-8_train.csv](../../data/public_datasets/flow_matching/mtsbench/MSL/MSL_T-8_train.csv) |
| mtsbench_PSM_PSM_test.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/PSM/PSM_test.csv](../../data/public_datasets/flow_matching/mtsbench/PSM/PSM_test.csv) |
| mtsbench_PSM_PSM_train.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/PSM/PSM_train.csv](../../data/public_datasets/flow_matching/mtsbench/PSM/PSM_train.csv) |
| mtsbench_PSM_PSM_val.csv | 公开预处理/作者包（精确实验版本未确认） | 可用 | [data/public_datasets/flow_matching/mtsbench/PSM/PSM_val.csv](../../data/public_datasets/flow_matching/mtsbench/PSM/PSM_val.csv) |
| drive_1OcNc0YQsOMw9jQIIHgiOXVG03wjXbEiM | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/MSL/MSL_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/MSL/MSL_test.npy) |
| drive_19vR0QvKluuiIT2H5mCFNIJh6xGVwshDd | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/dcdetector/author_bundle/MSL/MSL_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/MSL/MSL_test_label.npy) |
| drive_1PMzjODVFblVnwq8xo7pKHrdbczPxdqTa | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/MSL/MSL_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/MSL/MSL_train.npy) |
| drive_1-QvPvJFzBiUTrtHycQaw_XYlDaFjQbk_ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/NIPS_TS_Creditcard/NIPS_TS_creditcard_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/NIPS_TS_Creditcard/NIPS_TS_creditcard_test.npy) |
| drive_1NwyKxucNSwrq9gf-dHoIbtWnR8Jxy7k_ | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/dcdetector/author_bundle/NIPS_TS_Creditcard/NIPS_TS_creditcard_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/NIPS_TS_Creditcard/NIPS_TS_creditcard_test_label.npy) |
| drive_1mmaVfK_Ukp35z58YV9S4h64nJyDkM5N_ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/NIPS_TS_Creditcard/NIPS_TS_creditcard_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/NIPS_TS_Creditcard/NIPS_TS_creditcard_train.npy) |
| drive_16oiQRvH-0qYDBKj2FRIRnWbWGdfwXyOZ | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/dcdetector/author_bundle/NIPS_TS_GECCO/NIPS_TS_Water_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/NIPS_TS_GECCO/NIPS_TS_Water_test.npy) |
| drive_1ezC01a074tlYa-3nnwu49ywmFuibfaQh | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/dcdetector/author_bundle/NIPS_TS_GECCO/NIPS_TS_Water_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/NIPS_TS_GECCO/NIPS_TS_Water_test_label.npy) |
| drive_1qCi9dXNidRvQ6haGbmjGxwSDn17KWSha | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/dcdetector/author_bundle/NIPS_TS_GECCO/NIPS_TS_Water_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/NIPS_TS_GECCO/NIPS_TS_Water_train.npy) |
| drive_1wkyqT5E0H_1QFJ64xvVPwcM45Zalz5GQ | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/dcdetector/author_bundle/NIPS_TS_Swan/NIPS_TS_Swan_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/NIPS_TS_Swan/NIPS_TS_Swan_test.npy) |
| drive_1cA1n8g6fS5czsZtGG6bb3lKPInRiL6Ao | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/dcdetector/author_bundle/NIPS_TS_Swan/NIPS_TS_Swan_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/NIPS_TS_Swan/NIPS_TS_Swan_test_label.npy) |
| drive_1nGyc0xiUGvLyU6ztyRLqqhhKms0QO2ZV | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/NIPS_TS_Swan/NIPS_TS_Swan_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/NIPS_TS_Swan/NIPS_TS_Swan_train.npy) |
| drive_10-r-Zm0nfQJp0i-mVg3iXs6x0u9Ua25a | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/SMAP/SMAP_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/SMAP/SMAP_test.npy) |
| drive_1uYiXqmK3Cgyxk4U6-LgUni7JddQnlggs | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/SMAP/SMAP_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/SMAP/SMAP_test_label.npy) |
| drive_1e_JhpIURDLluw4IcHJF-dgtjjJXsPEKE | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/SMAP/SMAP_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/SMAP/SMAP_train.npy) |
| drive_162gZw7v7XBEhrsLv5Bq5r12M3bGCdvwr | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/SMD/SMD_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/SMD/SMD_test.npy) |
| drive_1NHwCJXUDXNXWq9gO3mopJbTGJfC6A7b4 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/SMD/SMD_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/SMD/SMD_test_label.npy) |
| drive_1ETJkCImUSk11p-p9B5CAnduszoNOnSiF | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/SMD/SMD_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/SMD/SMD_train.npy) |
| drive_1RQH7igHhm_0GAgXyVpkJk6TenDl9rd53 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/PSM/test.csv](../../data/public_datasets/flow_matching/dcdetector/author_bundle/PSM/test.csv) |
| drive_1SYgcRt0DH--byFbvkKTkezJKU5ZENZhw | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/PSM/test_label.csv](../../data/public_datasets/flow_matching/dcdetector/author_bundle/PSM/test_label.csv) |
| drive_1d3tAbYTj0CZLhB7z3IDTfTRg3E7qj_tw | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/PSM/train.csv](../../data/public_datasets/flow_matching/dcdetector/author_bundle/PSM/train.csv) |
| drive_1SO_j0apQAh0lvh4-t0HOQnwuYiAVyGNh | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/SWAT/swat2.csv](../../data/public_datasets/flow_matching/dcdetector/author_bundle/SWAT/swat2.csv) |
| drive_1TVppX8PzyEtRbJ45ooWic4HeSXHAQ5kj | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/SWAT/swat_train2.csv](../../data/public_datasets/flow_matching/dcdetector/author_bundle/SWAT/swat_train2.csv) |
| drive_1_Z7VEXY1vlzmj8uk-U8JOlWalyRZXzda | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_100_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_100_test.npy) |
| drive_1my20t6D-BG6hEa-LhxFopQAShDAG5HBy | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_100_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_100_test_label.npy) |
| drive_1tOgE3yMsthn3uHIsgQRB-3rB3ojJldcf | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_100_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_100_train.npy) |
| drive_10wJBgNJgKDzrWE3qznHJrdyQ0ANJm80L | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_101_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_101_test.npy) |
| drive_1vSQU4vonAOYpZ70k30SEOPKKQMEvTiPY | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_101_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_101_test_label.npy) |
| drive_1XCt31gFbB7iIAwLBD_gJLjZJLZjYQLLg | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_101_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_101_train.npy) |
| drive_1kmhQZNYYuKO5uI4tSPaU9QLOWsvMaBru | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_102_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_102_test.npy) |
| drive_19ZVhVAfrsb6pnHg4Pn1zJipztevKhvpQ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_102_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_102_test_label.npy) |
| drive_1Yi7Ssy8WcF1aM_ZEJE__1LHuYhTl1yVv | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_102_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_102_train.npy) |
| drive_1JmwOOKdshlCUEbxmAHr_Nh37LKWfO04g | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_103_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_103_test.npy) |
| drive_1A4sA9TIGZK2mpvxirfPmzcYQIlRGgeRI | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_103_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_103_test_label.npy) |
| drive_1BHi82HxZ5A6lnDApqRH2KOnjOFSbvnj1 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_103_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_103_train.npy) |
| drive_14N_MSTZJiD85IS5mlo0zBu3GKcuGYtIf | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_104_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_104_test.npy) |
| drive_1Lkxkz78n1VcF-hJpmi7B7mBqqpE1p1OP | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_104_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_104_test_label.npy) |
| drive_1UFgAjAY1XYNRu3OpcMEJ8-svOWAvjv-u | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_104_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_104_train.npy) |
| drive_1SS_Iz0uvAFRP-naXBXfCE50YjKaDO_7e | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_105_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_105_test.npy) |
| drive_10nfevHuV6vjm1A1L1h9NJPCoOaqGM5lA | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_105_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_105_test_label.npy) |
| drive_1CxRO-PySrzbxGfAo-JSPMCGYN7gz7Tnw | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_105_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_105_train.npy) |
| drive_1KfGIDMJE50HJTqtzzMNetj8ZevTRiNqe | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_106_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_106_test.npy) |
| drive_1VHEsYzMfHL-bFTXxWRKeonmQ0TqwhsoQ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_106_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_106_test_label.npy) |
| drive_1MYT46_yZRY7ZTz2SFYC7WTs4zFzh00so | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_106_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_106_train.npy) |
| drive_1MLY28wEK7gFWo4WotJb9KgUzZQt8IcoO | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_107_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_107_test.npy) |
| drive_1OrInecE3qNwLhCIqllNAdUY_rmhQ2bnV | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_107_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_107_test_label.npy) |
| drive_1PQndrNteb_2VDfjGseQ5M_g4AGftr-5L | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_107_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_107_train.npy) |
| drive_1XzGScG93qTY_IUkE8GxK7WgsYv68AsvE | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_108_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_108_test.npy) |
| drive_1mrhvGgVOdC77SlymwNJSODtiYeTjhTbQ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_108_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_108_test_label.npy) |
| drive_1fTYZhnden18RbJ9WsARwl9HuCyuxD0KG | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_108_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_108_train.npy) |
| drive_1aUCK0tEJBxdhH3sS4SSph23DNTSQvzG2 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_109_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_109_test.npy) |
| drive_1jorGzYQPzA6rk4MH-GBaC-BjyJXxgazg | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_109_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_109_test_label.npy) |
| drive_1FhrUAySwv-PS-P94pedlgjOcCHSu4a67 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_109_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_109_train.npy) |
| drive_1gAVqw8WL5SgRwL_xrgWl41G2XCMNUXoB | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_10_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_10_test.npy) |
| drive_1Zjcy5vBgVW10SD_uMMtZ6MRsqVMtMXUE | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_10_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_10_test_label.npy) |
| drive_19iMcYpaLh4W02QSFWVkc7_VcGyr96gYK | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_10_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_10_train.npy) |
| drive_1VNFt2ABjzaOAa_uTKx54ujaVgVv_qG74 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_110_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_110_test.npy) |
| drive_1XxyKVux0cSPGkPpmweog7LAV-_4ziLW8 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_110_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_110_test_label.npy) |
| drive_1lWB1jC3Rgt3VNfyNBCHtSUfCPQiD8qn6 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_110_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_110_train.npy) |
| drive_1lSrkT1uKpRxScRsm2WNgzvuUWa6uiIkr | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_111_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_111_test.npy) |
| drive_1bg1-Cv3PWSiLCXWd3Q9YZ-B19Cy3ZkUB | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_111_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_111_test_label.npy) |
| drive_1rP0CvsRWyOv6jtEaN-onAYFZeUCPw4V_ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_111_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_111_train.npy) |
| drive_1JLuCxNV8rUio8mRyOsw9kB2XCkNc5r-L | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_112_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_112_test.npy) |
| drive_1PuMS2uZyWeoOYoH1Eya-3Dc0IHBbFCGC | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_112_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_112_test_label.npy) |
| drive_1UN91-zGMyVvSvikHSiimsA37Ci7fFwAT | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_112_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_112_train.npy) |
| drive_1TbIdPjKianlVOg4kjB15hvbuRGvq3DNe | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_113_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_113_test.npy) |
| drive_1NuUiBp8_-jYVeuhZ39fMzXeqq_oYC87C | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_113_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_113_test_label.npy) |
| drive_1EFivVitAxHensOwpz3ZUUVCtLw-PWNiO | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_113_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_113_train.npy) |
| drive_12n43A9F_5FwgNeSrnrT8NcbCA9kuRr0_ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_114_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_114_test.npy) |
| drive_1iOy-sUdI7CcrflhwhWmsKgc7-W5HrM6M | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_114_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_114_test_label.npy) |
| drive_1NRug5Cc9EL_bGdRYgJcsjnXSFxlVX-bv | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_114_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_114_train.npy) |
| drive_1u5fs1CbJyPshUtfNCF_bW-n6TA3TAmRl | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_115_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_115_test.npy) |
| drive_12qlnHdybFZsDL2zqO9KY7XiWhTwTqOp4 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_115_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_115_test_label.npy) |
| drive_1Bx_gfKxob4gdlU37FQ7oqIp6mIp4PSIz | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_115_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_115_train.npy) |
| drive_1hQ1-Uw5I6SXQ9WXcpU6DrzpFhzGz2o9o | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_116_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_116_test.npy) |
| drive_1nnVl-uwVURvRV9KVRNhnplafgxBlZn9E | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_116_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_116_test_label.npy) |
| drive_1tZ92yomgNo9B16yOrlEO1N7L7U40dfTD | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_116_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_116_train.npy) |
| drive_1erLCw61wRjq9DvJByDs3XQyFpFikhgm3 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_117_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_117_test.npy) |
| drive_1gzNDXk7X1HVOGn3VsFMBI1m7vyeHoZCH | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_117_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_117_test_label.npy) |
| drive_1AR3ujRKWALXyq2n-566IqpTc-U5LIukM | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_117_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_117_train.npy) |
| drive_1qbNx81LQ-X3VhD980A63vzWIBBwimluf | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_118_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_118_test.npy) |
| drive_1Sz83_lklI3pY3ZbJTABBYa26doCLktPU | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_118_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_118_test_label.npy) |
| drive_1zv2APEKaWVeDmcq6WjFY0WsJ2f9ntz8D | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_118_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_118_train.npy) |
| drive_1COMlrlTZQ2mZAa7dvst-wa9-7ildSeVm | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_119_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_119_test.npy) |
| drive_1vFUvNIxJaefG-5_-MyU4Qv1Z5_cBgpsn | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_119_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_119_test_label.npy) |
| drive_1hrb-Zd_kZdkh-HRsPocWGmWIUpbYZ69B | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_119_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_119_train.npy) |
| drive_10-XK0QBgx8MERS1iiJDMrxca7r1Hmg8l | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_11_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_11_test.npy) |
| drive_10WS_xWE2UdrmdM4ZfrukdW9E_Z49L61X | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_11_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_11_test_label.npy) |
| drive_1q1j5EmSlrvToAWOm5iO49lSUPVgefQDB | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_11_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_11_train.npy) |
| drive_1CETwYuY4bNAKJljQMdU1tFPvR_6vPWXd | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_120_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_120_test.npy) |
| drive_1Csd2ieZqGn2lcRZX8CX93oVEGPnx-hm5 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_120_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_120_test_label.npy) |
| drive_1dFEjsQ4_Z8v2McdmQrC0aZ9wNpLSAq6A | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_120_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_120_train.npy) |
| drive_1z1SEXsNQXafk_A_UcHUdHACWdal3B2ic | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_121_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_121_test.npy) |
| drive_1n8RlN7BBN9DQB5pEa9m1PqFmt8vD7PLD | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_121_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_121_test_label.npy) |
| drive_1jaTQ6HIixkFSTmU_Rz2QmcepT3ZlxbvK | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_121_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_121_train.npy) |
| drive_1cbpmJBrnm8_1dGFLASVu6-YjPjlSMl9C | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_122_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_122_test.npy) |
| drive_10blY__V4MzB4D_UhS29sNPZC559FS9ss | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_122_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_122_test_label.npy) |
| drive_1u-tsHIWVeBkziFot8aQrhIaqVYIvg88k | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_122_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_122_train.npy) |
| drive_1u7ioLCqHaZYlx3YbLXbN3gE3tDvcHJ85 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_123_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_123_test.npy) |
| drive_1X-hINQk3738DMNoYKEwBwF2_xwLtCJwo | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_123_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_123_test_label.npy) |
| drive_1P-6TtPijKWFpQBOJxzz2FnnwZIJXx5Is | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_123_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_123_train.npy) |
| drive_1vkhoIsf6tbfkcCOfoMSGCHdUtzZ7pX-n | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_124_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_124_test.npy) |
| drive_1HuTAjmZMqzsr5JvPcHOBQgLM5VKJ1E6m | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_124_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_124_test_label.npy) |
| drive_1pxeaU5VjR0kEdo4uwX95mc2C7CbYwlwN | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_124_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_124_train.npy) |
| drive_1oJCD-Cpz0ret_dzld99q-O0V35uNvPl9 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_125_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_125_test.npy) |
| drive_1bSRF7cvtXtirjf5s06kANNz9tTDo1-uU | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_125_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_125_test_label.npy) |
| drive_1YfFVqAjYmXuLUsnoD9CJbnPQgyCCeG34 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_125_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_125_train.npy) |
| drive_17f14nmRm5lT3B_LyVnSM4RaKrejgrHIL | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_126_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_126_test.npy) |
| drive_1wJ-H7hEUROeadoLtyFpqqaKXLBRjREM5 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_126_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_126_test_label.npy) |
| drive_1rSJZUjEX1Gb3rLddr1VSYoftk-W9si1t | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_126_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_126_train.npy) |
| drive_1-yzeof4LEjBik1fTBvsr5SP8VJrcFuku | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_127_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_127_test.npy) |
| drive_1spJTalICYxBiQ-gCBr2XfWuCODaB9dNF | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_127_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_127_test_label.npy) |
| drive_1qlWdDce9IaLu9LCq7xf9Bmd85bjhozCL | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_127_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_127_train.npy) |
| drive_19NB76gExrV2KwI_8bmoGgmo3UAtVykSD | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_128_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_128_test.npy) |
| drive_1OpSCbPYEzp8yixMbG06JcYKBHlGYrvkg | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_128_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_128_test_label.npy) |
| drive_1qUX06ZS-bzdEf1uwqoTE-miODBbK2M7k | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_128_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_128_train.npy) |
| drive_1OogtABH4WAp3e3X7dTgqaH7OYp3H5xuo | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_129_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_129_test.npy) |
| drive_17x7Qs3TAryQpXL9j00NJhIyUXnjcbQ0r | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_129_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_129_test_label.npy) |
| drive_1Azu4LnJyuwT5b3jnJ8ZsryKwQNliOncv | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_129_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_129_train.npy) |
| drive_1upAyfdBWRyNZiRDyZ4ym5QVSqQZxTYZZ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_12_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_12_test.npy) |
| drive_1NNEwGMhN5NpE4E7yvthpn17RP-zgvOeU | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_12_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_12_test_label.npy) |
| drive_1Uui0FmIvtjpZ6xWj-4jiYMMWUoO3a_lh | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_12_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_12_train.npy) |
| drive_1mLs9MEihLoHN6mE2oGZMYvAaO-H5eXHs | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_130_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_130_test.npy) |
| drive_1mK5_6dcvIayCuuqCS2mU8igK_rLOXKRt | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_130_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_130_test_label.npy) |
| drive_1C2PgR5KQkoouufuvKzkPMIaHC7s_GtQ3 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_130_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_130_train.npy) |
| drive_1a_BiwUQISZM95xOzGM5T6NHUkZ1bX2QR | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_131_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_131_test.npy) |
| drive_1Bwb3jY06Lqjr3auycOqweLZ89r0rLLmV | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_131_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_131_test_label.npy) |
| drive_1ruf8qTArjpVlrExunyDCDj59Ud3zXZMq | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_131_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_131_train.npy) |
| drive_1Kb5YtTPfkoLGf5ZJiVRdGS1tWc9ZmO02 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_132_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_132_test.npy) |
| drive_116AJam56k2mTHSKUzDF5_oGYnrN2DsMH | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_132_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_132_test_label.npy) |
| drive_1Exl-QSnadIDvWTw1CIUoEW4RmmvDeZMA | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_132_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_132_train.npy) |
| drive_1g9fpeqZjYAbAuBXAnr6bl4xQG8WvkprK | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_133_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_133_test.npy) |
| drive_1fhJ_RWkkTHlBYMWfedHZ29ZoagpncP0d | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_133_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_133_test_label.npy) |
| drive_1DwaDLBgjrPhmfGot8f9f5rHr2cX5_m2Q | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_133_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_133_train.npy) |
| drive_1G4BtXoEWKK8iaVo3Ovmga6aqkJaBvS_J | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_134_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_134_test.npy) |
| drive_1wedZKAK64-OzQ48kW93w-ODTlNkmSQdh | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_134_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_134_test_label.npy) |
| drive_17z2YY6n7302ymTfUydh6zWHa9aDBGx-f | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_134_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_134_train.npy) |
| drive_15liHJMrxn7c2ElE4IURRALoO2KnfSJ0r | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_135_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_135_test.npy) |
| drive_1tCLJgcx2Kv_nUu_t8TqvDK5rXQ59XbDw | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_135_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_135_test_label.npy) |
| drive_1SLVPaWl5dj0I2sLduu29VqNhv_BbzLY4 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_135_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_135_train.npy) |
| drive_163iyOWQe9QKYvIsrT9kGtU0x4ty1oCga | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_136_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_136_test.npy) |
| drive_16S95mZ0Lx_VT-cKZdZNUqkffiYasDEUS | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_136_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_136_test_label.npy) |
| drive_1G2mRIVVQYye0v81IkRp7GEYJ1371_CFM | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_136_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_136_train.npy) |
| drive_1-vKaRCrBPTshSvCao4QChO_bMpBoMBFp | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_137_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_137_test.npy) |
| drive_1kEydXT_-WakFPHRJaI7MMo-rAInHQKYC | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_137_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_137_test_label.npy) |
| drive_1h1OgZ6NYk6x-xo57vLYnnmcoyA4pkoGt | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_137_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_137_train.npy) |
| drive_1jhzIVXecqH7d4l8d-lNgUtwJukP77LVf | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_138_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_138_test.npy) |
| drive_14p9rgfJrBwNeVbaPQdKmgj87cIp_nJ2y | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_138_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_138_test_label.npy) |
| drive_1ZzCvfY2MUWcC_hTvit9GzaPawKoiDXkk | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_138_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_138_train.npy) |
| drive_1Mgc_d-6XuYIM6KS1VC7P5lp9ewa7UV10 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_139_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_139_test.npy) |
| drive_1oA8oGwogXLDNmJE2YZRQrBP6Cq3HCTNI | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_139_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_139_test_label.npy) |
| drive_15AwoqFM-uQ4aPwef5ddHAliKfgNLHEnb | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_139_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_139_train.npy) |
| drive_1ylm-qujDC8JpLD0QsZa-YlzSwq-jduVH | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_13_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_13_test.npy) |
| drive_1Az-a-e4BSSGodWmELO55xseTbAX3A4dG | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_13_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_13_test_label.npy) |
| drive_1TyKSmaV-d26XhYRQHuigiLawn2cSDdPE | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_13_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_13_train.npy) |
| drive_1VxxxDEv48EY2BWGMKQhlWn6yWkFB11CP | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_140_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_140_test.npy) |
| drive_1PCGWKETqSkrO6tDPSoEfNosmNOE7U8bL | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_140_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_140_test_label.npy) |
| drive_1OEEW7Gg1OBgCPif_N1kUp2SykHlT0KCg | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_140_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_140_train.npy) |
| drive_1yELigKSZEx4z7Ph7htFwnEDVKJZXoj4z | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_141_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_141_test.npy) |
| drive_12c4vTL0w9dQoMCzF1-ZNtXLVAa2h9Oux | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_141_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_141_test_label.npy) |
| drive_1xC8ofV_6iXkrwhkTmMWSrfFlWNRTQHpW | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_141_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_141_train.npy) |
| drive_1rpNBPZSEKr8VtdloWIK_9ZZiSyON4RkI | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_142_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_142_test.npy) |
| drive_1mJUsgR4sUu4clVgLbkxoRuF6Z_Wi3xP7 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_142_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_142_test_label.npy) |
| drive_1GzK1HJ_t3-Dj3_272Nk8rg1RKRpWycuU | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_142_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_142_train.npy) |
| drive_1tPgHSrcZvmpWaSaZLhjN_724YZI6kNHF | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_143_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_143_test.npy) |
| drive_1X_mUCKu7EK8_ZHz7-ROu-_FGzXYJesno | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_143_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_143_test_label.npy) |
| drive_1bWuzmmtGwlM54I9XlK5HHDy8IX3eXDoG | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_143_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_143_train.npy) |
| drive_1XXk3h73W8RRJ4-cD7sf0jHchObW5ZBkX | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_144_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_144_test.npy) |
| drive_1ix4oTmJhzlNa0-hUGun45_Qv5Qm9LDl5 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_144_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_144_test_label.npy) |
| drive_1_y9hfacxWplkQx9ALGxym5IBUYFzmrCS | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_144_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_144_train.npy) |
| drive_1LpnmGCHNimW8ylYINKqMiC_XGKKqpg0G | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_145_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_145_test.npy) |
| drive_1eYpEhdXsnoqPReaKsIMgR077TCBqo6GQ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_145_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_145_test_label.npy) |
| drive_1vVknS1Npsuy9GQ0xKtk15Xh_6Hc9oLAH | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_145_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_145_train.npy) |
| drive_18pMmBOUegfru7lJWcMcQxqoLMWiiOT7O | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_146_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_146_test.npy) |
| drive_1tT6kqUD9TwMyY1G6hQrmA5Lw8VpBaEmS | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_146_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_146_test_label.npy) |
| drive_16FNYDbuuh3LzzkYLmHKmo-b-3wKiyKc8 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_146_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_146_train.npy) |
| drive_1nAmEIP2VX-wjciriNOeMV4UMTuh5Gvyk | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_147_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_147_test.npy) |
| drive_1P_QhnA7Zcv0zPDSJqee-O3jlQnX7bNQf | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_147_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_147_test_label.npy) |
| drive_1RcKF7V4WguqtV4dQ0yNVMAutbNSNDgD_ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_147_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_147_train.npy) |
| drive_1uqZ3Wen1jsJ7-DRwlhWxKawv2S4dHeIn | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_148_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_148_test.npy) |
| drive_1C4MMsocbxcevvC4GiObKLEc2Dy86vSlA | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_148_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_148_test_label.npy) |
| drive_11zYZhtbKMECPWE9PCacyCe-AaqUJub16 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_148_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_148_train.npy) |
| drive_18AwOijgGJdrNKmTLTIR7him9ENorS3Ux | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_149_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_149_test.npy) |
| drive_1XwA-LVrl3BR9eNWUIq90VLXdGAKN798b | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_149_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_149_test_label.npy) |
| drive_1XN6Yr0JO4E85p7YpcDBcHozFHqOwxPyv | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_149_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_149_train.npy) |
| drive_1-VO_7LJgf0RuAZ2TvoffhlB3uV02ZHQJ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_14_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_14_test.npy) |
| drive_1PcRuIC7TPXrhvL-hcY86CcbjSEg0TaXo | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_14_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_14_test_label.npy) |
| drive_1BYuZXnrLlqa9jVhY3RhOJLTf2y85DYSa | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_14_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_14_train.npy) |
| drive_155JinbeR_z7D6IAl2QiT5EUJEnb9uTDQ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_150_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_150_test.npy) |
| drive_1jkb1eRtx25-n_ndf5BcEEpZyyV_c82cy | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_150_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_150_test_label.npy) |
| drive_1pfpt6Z3vFc3E5ZhuIJ2LrH1VxpcQ-qCk | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_150_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_150_train.npy) |
| drive_1Vo9pDbY6cZY_XWIxcKi6tZrfGdwjflec | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_151_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_151_test.npy) |
| drive_1lcTYRQCd9uR_0f-vF76ST1sY9CaDsWVq | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_151_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_151_test_label.npy) |
| drive_1LSldrkKMQCh_BKZktrNX8ou7Y1iZDwqA | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_151_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_151_train.npy) |
| drive_1uSehq69CHwmwE1K4r-l5xhH9W0qYDJ72 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_152_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_152_test.npy) |
| drive_19lUCeWvNIjN3LEdoQHvuwxCxbSSvrytp | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_152_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_152_test_label.npy) |
| drive_1JwkUMUvMAlk-s5D91i1XfK_vUAqPUWnt | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_152_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_152_train.npy) |
| drive_14D1putfiem6Ioe_-aWvDNW-A6x2lIZmb | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_153_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_153_test.npy) |
| drive_1TxEkNgNbugS_JswDP0Urse8bVf7IwJSF | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_153_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_153_test_label.npy) |
| drive_1pltTxxtlvGjigxdAFxu4NBqgbCoWSxPL | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_153_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_153_train.npy) |
| drive_1SuxVXnVdkGCrQxjqBqIMtSJtms56rxEA | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_154_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_154_test.npy) |
| drive_1pXLYjmhc2y95p6bYmHDczjFL2A-hpNSJ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_154_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_154_test_label.npy) |
| drive_1QdQPGEs1wrmpt_7zellhWoaYTYosTXxn | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_154_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_154_train.npy) |
| drive_1fGxsUDO0Guq22qaRgxjJXwDBhTWGwTe8 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_155_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_155_test.npy) |
| drive_1Y_lKaJ5zcEZeI5JGVsZxT738leetCL_3 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_155_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_155_test_label.npy) |
| drive_13_hKYaJNSQ04U72-gWm1mEfeg2sPi4Qg | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_155_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_155_train.npy) |
| drive_1cj50OqVnIUtOVhM6cWnM3YAtx7yfkOC7 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_156_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_156_test.npy) |
| drive_1FH_P8OC9zzBOH89kTTNKdmRRnoklxd6U | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_156_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_156_test_label.npy) |
| drive_1yzjFWCXNE7WaAYjUEiQKny_mdZpNKMuZ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_156_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_156_train.npy) |
| drive_1TTzPumR3BCXO0TRWyR3Eyu3bBh6B-2jX | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_157_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_157_test.npy) |
| drive_1o6n_Wx537ITw9ZLw11vvuZqx5Qvtx-Wd | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_157_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_157_test_label.npy) |
| drive_1_ER7QsHheQmBuSYagmb63KGDUlB78xL- | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_157_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_157_train.npy) |
| drive_1WKixx2JNRimzJxQFFEX6CccXjYmocM6P | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_158_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_158_test.npy) |
| drive_11UWZL4Ikt5tmrgnZJhYkatXkQDhfVckV | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_158_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_158_test_label.npy) |
| drive_1Lsh893Z36pXqSqPqPhPijJGSdbKg14lh | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_158_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_158_train.npy) |
| drive_1sc5HJ9QTyVaNEaK2RBmbNVLuqBdIJn7r | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_159_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_159_test.npy) |
| drive_16alqaIO3-QXhPlHGrl9YngN4PUReiXFM | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_159_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_159_test_label.npy) |
| drive_1XEHScaZ_xxJ7EvCMRdN0k1CeRvol6m0d | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_159_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_159_train.npy) |
| drive_1oyHLbEEp5bRcd02k_MqXU4WIye2wq2dM | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_15_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_15_test.npy) |
| drive_1gxhAinUBXi5ZKkuh_8JQYxdKSwsoGrvc | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_15_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_15_test_label.npy) |
| drive_1MWonEXJensrAvWQmWGpBsNepQw6FMxFa | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_15_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_15_train.npy) |
| drive_1vLVt7KiVQNQiX16BcSLkWiMWTrHUEhl8 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_160_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_160_test.npy) |
| drive_1fZaoqmafavWH7ucTNyviNyQX57jo9Tld | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_160_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_160_test_label.npy) |
| drive_1qvp2A8D67DAL-f7rNtbpPZVYlShga57U | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_160_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_160_train.npy) |
| drive_1RjtFjX4j5CWDe5C7y-LWTs1GkK0ty9E3 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_161_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_161_test.npy) |
| drive_1qLGbMAFiXO5yid_Nbo1Y6x9bcMowyr-C | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_161_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_161_test_label.npy) |
| drive_1vMjEhIBQHwZyb6qm7yaQyjIrs0ia4_n- | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_161_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_161_train.npy) |
| drive_1DPW-edvF5gEEVmfqo_kiQu46KFMSzPAn | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_162_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_162_test.npy) |
| drive_1Z-feRkBQKYj3GHfxtFa4Zs37BmVkqemD | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_162_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_162_test_label.npy) |
| drive_1yoeSahySlmujTmZ68JoFAdsrxZ2cyvnw | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_162_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_162_train.npy) |
| drive_1Gwb5INec1456yO1vO8kMNdF6MTE9QG7t | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_163_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_163_test.npy) |
| drive_1OYNOdqXsxPYq14vlLoy5bfupHhNuFAQi | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_163_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_163_test_label.npy) |
| drive_1W6qceK1ouYSqsih2wkYJneJv1KgduPtj | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_163_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_163_train.npy) |
| drive_1yQvE2SBFJDdRGy89irrI5Ea5uUXAEY4F | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_164_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_164_test.npy) |
| drive_1Q6z6NR7fBDYqnjusQJShT7k3EW9Y1frZ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_164_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_164_test_label.npy) |
| drive_1ejx7Zwx0pycITgd3V6Ps4wyyBte36lY2 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_164_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_164_train.npy) |
| drive_1EgRpq06FmEk5NmLzNPa36QOnmzi-vI0M | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_165_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_165_test.npy) |
| drive_1WXQ6mT-UTrRHOzDBxe7pVzAXzL2-Bpv3 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_165_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_165_test_label.npy) |
| drive_1yptQeeQkvkVttVuXW3s03_f4AonxdLUU | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_165_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_165_train.npy) |
| drive_1z0vfadRxtQjDzkkLq71Xx6dQkMDiseOE | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_166_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_166_test.npy) |
| drive_10W0GtL8nFRUlEGFO0xOEK0eTfcoPrazY | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_166_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_166_test_label.npy) |
| drive_118NoUEQqlt2pjp2igbik7tcr23INzO7v | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_166_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_166_train.npy) |
| drive_18iDupYhTrha1pawmpd_DP4vFPVe-zQnu | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_167_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_167_test.npy) |
| drive_1Vt-NWApDK82Z55WK3pwJE7ND7erx-9jp | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_167_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_167_test_label.npy) |
| drive_1qvSJDYl_ZQBoFd0hijW-x5TzlFcGlmcR | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_167_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_167_train.npy) |
| drive_1djCvOv8nzYpkSKS7xBiglLzA2yoAU50t | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_168_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_168_test.npy) |
| drive_1iAggcFuqLJeG3QQrY_gehUNoobAF6LlR | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_168_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_168_test_label.npy) |
| drive_1S4nA7hoo4pVMXr5QbTZEfh_gkNwQPYf0 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_168_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_168_train.npy) |
| drive_1sMh9nSqxZ3OnVUzhpMF5aaBbtVT9rc4X | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_169_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_169_test.npy) |
| drive_11qc1oArgXgDOhT6VnO3H_St0Hya0Iwja | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_169_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_169_test_label.npy) |
| drive_1HXRNFQL50nsh_mIoaPH1S3j-bi6bFkSW | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_169_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_169_train.npy) |
| drive_1AWvfrtqXaxrYIf8t7KG9NK8nn0xUprKZ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_16_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_16_test.npy) |
| drive_12USxtEAImiv9wm0PEUUaIEiqx2rKzTgA | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_16_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_16_test_label.npy) |
| drive_1jY2QWJwpwCOiQ8ofU8bXuziLutdLEncm | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_16_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_16_train.npy) |
| drive_1TiPlb7LbszpaQ4IvdDJJ2Hpinw_Rz83g | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_170_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_170_test.npy) |
| drive_1tv2Hjy0ZSKm8emv5LSGZVuvHNtt0daIQ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_170_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_170_test_label.npy) |
| drive_1SXr7BAwTDEiEUk99DRjnLu0At89hhQEp | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_170_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_170_train.npy) |
| drive_1xS5HmqbrmKdLkxUUs8oOAUN2qzR9lBsr | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_171_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_171_test.npy) |
| drive_1-9QeKETBW4uZeFf-Fsj_Rij-LY0kCDED | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_171_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_171_test_label.npy) |
| drive_1XBG0f-V8efInP06qkk_fOHFrRpZsf6e- | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_171_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_171_train.npy) |
| drive_14qaRLlALo1Nd0d7zs-LOWqV7aLuSoLLu | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_172_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_172_test.npy) |
| drive_1Y6qiOs2bihNXbfSF3WAH8iiP6C1og_G3 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_172_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_172_test_label.npy) |
| drive_1M8PimBOd4KkCL9CsbcfllZ0qlaODcyzK | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_172_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_172_train.npy) |
| drive_15smwVGc6YFP7IYGPmopdqVjaNS3rwu3r | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_173_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_173_test.npy) |
| drive_1Gv2JnxoMrayQHQrExoRNxlrEjvPRpMVw | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_173_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_173_test_label.npy) |
| drive_1pcoUhfePZ-6PJ5QvuaVZmV_W2-LDI183 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_173_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_173_train.npy) |
| drive_1P_MPwskLacgH9Lu8p-8qCBdUpJc6NslX | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_174_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_174_test.npy) |
| drive_1mY5qUd8MJWgJFokq_vH0qcOV-1lg0q6Z | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_174_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_174_test_label.npy) |
| drive_1vrUhTY-x_Ca9uk69DXVfU--RZPDwn1hI | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_174_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_174_train.npy) |
| drive_1AZNYfPw5VBB-hjXmEXAT_QE68bRLyUVg | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_175_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_175_test.npy) |
| drive_1sa2WoevBO9CnPWNUJ9sFEGfA4G4aZ6HN | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_175_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_175_test_label.npy) |
| drive_16-_7zotm6qmu-95arEa-gYwqDszGGi0b | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_175_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_175_train.npy) |
| drive_16it3bY0FUkpmZ4Tlp1o-rxoSjdeZUYDx | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_176_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_176_test.npy) |
| drive_1V89Zvwa2UeAxYEm-13ovUKhYTAYXH6E7 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_176_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_176_test_label.npy) |
| drive_1za1mK1WW_jIU9vxEbmtu7WGczH2eSshN | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_176_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_176_train.npy) |
| drive_1xjsqMVJK2NDdBxtw6p9k1GiNEsfK8dMX | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_177_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_177_test.npy) |
| drive_1ataJrznpkgxs_Dlr0LpXHGPppYNHxtGJ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_177_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_177_test_label.npy) |
| drive_16LQbYtfbLNvTsgijv_0gO0TOXl_5l9zh | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_177_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_177_train.npy) |
| drive_1NouZnb6g_OK8yX67bFGNvfTUn892dr05 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_178_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_178_test.npy) |
| drive_1j_58OiVtE20Fce1ORKAKpqkKr5rdqpk5 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_178_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_178_test_label.npy) |
| drive_1ot742yTQDq5tMl7MFwCOrO3R1EQMvx81 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_178_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_178_train.npy) |
| drive_14SrpuZebvPP8JvUxp0xJsrfwjqEwg1zj | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_179_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_179_test.npy) |
| drive_1H_Gfrqr9GneGOTFTerUvIrhIg8FOol3R | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_179_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_179_test_label.npy) |
| drive_1GjIKk6jwg1gQ2_Z2b1fGX3NJoRsN4xRZ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_179_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_179_train.npy) |
| drive_1puZrO3tfVCFFoM9XA-YJSuUx0G9KfFjP | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_17_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_17_test.npy) |
| drive_1vy_0RMBxnnu3a58SmqC_mYvItV4a8S8m | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_17_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_17_test_label.npy) |
| drive_1M0Mes3CqksZFO50lA9aDDAXgGqthHdjS | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_17_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_17_train.npy) |
| drive_1nD0tWHBv-VaCN0wnE_Ym0rJvS-Ed2dmO | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_180_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_180_test.npy) |
| drive_1ZEjEUB0jVnrJyJ-lialMnRGv2n82kgmq | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_180_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_180_test_label.npy) |
| drive_1oPa_z5XvJ0_rOa7oYTVauZhY6YX135h8 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_180_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_180_train.npy) |
| drive_1EAWrbra-jU4aAH2rOXl8WnovGxrzkfZj | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_181_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_181_test.npy) |
| drive_1r78RF1m2GlB5TsRwb7LvKzzzyafmPhhZ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_181_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_181_test_label.npy) |
| drive_1VqtnwPbHiDbN0Q6jA02G3YGUlx51v-LN | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_181_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_181_train.npy) |
| drive_13a4-mPk0HDnDb2BQ5_eWOjfAPUZ4NdEL | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_182_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_182_test.npy) |
| drive_1tsuLtmRjl1JhRSkMe7ggn2KiW9SlnYX9 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_182_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_182_test_label.npy) |
| drive_1ircxNm9Qy_hb81tYeMUsbCn9oCuaMAnr | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_182_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_182_train.npy) |
| drive_1hWf6f8-1OLbu-0b9HiecWFRfKpovwAwg | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_183_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_183_test.npy) |
| drive_1lC1AlVJ5NDeWJGTkBzFz6_8Q9wABzPLJ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_183_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_183_test_label.npy) |
| drive_1NuqXOhW-ndaStRWRTZ9czLQ9DHNnrlfY | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_183_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_183_train.npy) |
| drive_1RVPVOr5Z0fBQ1CZEXMcsVSTLnl2QZOBz | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_184_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_184_test.npy) |
| drive_1843RAREYoILT-7jJI7VzVu8UYyUY44P4 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_184_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_184_test_label.npy) |
| drive_1GE62zCYTnNKERzF2TfBciGbJwoD-6TKb | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_184_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_184_train.npy) |
| drive_1IqK_wkxzkJP4p0MjfttbficCyApEvEI5 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_185_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_185_test.npy) |
| drive_1HiNNn6XtL7Ff3V2K6Rt2KEa3TpBDdHiH | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_185_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_185_test_label.npy) |
| drive_1P7NeIysFLTYYAd2zcSbFFD4PZRi8Pq3I | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_185_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_185_train.npy) |
| drive_19IgNjclJRVuL17KdTtmwZbULUC_tpBQ4 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_186_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_186_test.npy) |
| drive_1AmqYZ0mYL_FtO-dYqU86jxh8zwTxgBKS | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_186_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_186_test_label.npy) |
| drive_14tUSGytztrVkywTYymMRNtCH6cOZUSln | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_186_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_186_train.npy) |
| drive_1saq9MHgurE0kj7hv-R-dhoczsAQDtONA | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_187_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_187_test.npy) |
| drive_1aRwbnppCTrYbH4nCXbaiY6tdITmj8v5J | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_187_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_187_test_label.npy) |
| drive_1p4Sc42wJp7gl3LmaVCRHuEWhCGKqQtB3 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_187_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_187_train.npy) |
| drive_1LqjQn1E7lUI2E7eQBfdPAfxrhJxowwnQ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_188_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_188_test.npy) |
| drive_1ZrevW7KRmE3kZFEn6nHybfN-B1NcyG-F | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_188_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_188_test_label.npy) |
| drive_1fcFrXA2xebGS1kxCSvwyA1-0Hj73MD1Z | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_188_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_188_train.npy) |
| drive_1VEG_pH2Fu2wDXgeKo0_rLurRr2PA6gcZ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_189_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_189_test.npy) |
| drive_1FuJDfoy1HCIPQ1ZNpgJfRAKt4dizwHt6 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_189_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_189_test_label.npy) |
| drive_1w-Mlw2QKMoAtyoXheBXwwVvYJwMBGB0o | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_189_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_189_train.npy) |
| drive_1xUBiGkAur2Kf1QaECTq9BE1JA3X7OplH | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_18_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_18_test.npy) |
| drive_1KtuXxZiZGOulj2IwONGOEmmg82JED-FN | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_18_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_18_test_label.npy) |
| drive_1sn04wby3ufmYCeFDhegowG7M2yvj1v8_ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_18_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_18_train.npy) |
| drive_1qHfRLHniqrADeuPh-OVXC8Vg8mGnj53b | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_190_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_190_test.npy) |
| drive_1kg5euSTrTbMBV7hfXUtbeYTioxTYZaRa | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_190_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_190_test_label.npy) |
| drive_1pCZyeLCyLl4HmT7IpognS3VN_x0DG0aX | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_190_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_190_train.npy) |
| drive_1uzfTKmHtFZz20BBUyWgIlCGrRNLjrMoy | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_191_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_191_test.npy) |
| drive_1wZQFP38S7J7_FhecGTGOC4VcGUWDiVZ8 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_191_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_191_test_label.npy) |
| drive_1qZuC34u1Q_1mUvbEgJaneUR7NSMVloEA | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_191_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_191_train.npy) |
| drive_15L563IoaGTIzBzLExHCTmnA2_P6kMBlz | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_192_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_192_test.npy) |
| drive_1rmcPzFD-0hN0Yo9VhCIzq9K8rtNOGJAz | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_192_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_192_test_label.npy) |
| drive_1j_Xrz879h_EFZWPUkwAgNtjrD7GgEnqU | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_192_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_192_train.npy) |
| drive_1E_ufZZ16k6j-nzuNAEHIIkzGUZ85LCUa | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_193_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_193_test.npy) |
| drive_1KTpkKnRHPjnJh1dRgU-tt8xq8ElDalHX | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_193_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_193_test_label.npy) |
| drive_13fBKWes03kd03tCeZlwNIQqexs8EfGZZ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_193_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_193_train.npy) |
| drive_1AG83lInMAi4pnvvKKGdQd4er2ug7zPpc | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_194_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_194_test.npy) |
| drive_14brIJ6B8v-BiwM3JvfgADWOO4y0hyvqq | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_194_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_194_test_label.npy) |
| drive_1TuQgLJeY2S-1LN9A4mTIdV2rptxP4yBV | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_194_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_194_train.npy) |
| drive_16LG4IXQ53yK9yEH49816oMe8SJM9DLiu | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_195_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_195_test.npy) |
| drive_1Ib0JLrzDGM13Bzf238aA8HLxTN4VGHKE | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_195_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_195_test_label.npy) |
| drive_1LQS92mDewq0sfAn6fq3adnqjBP-9sq5u | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_195_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_195_train.npy) |
| drive_18jSNtnleyo-RtndtP3kyD5_Q15axuU3n | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_196_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_196_test.npy) |
| drive_1tKqoO8qX-6p-ddjsIoNBtUvce2oF9n2z | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_196_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_196_test_label.npy) |
| drive_1fuDJ8Q8YOjj6Tzn_Bs4B7axv15vuZhzp | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_196_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_196_train.npy) |
| drive_12eYwsguOzYK9kcHtBncwnTrJiABKnTjt | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_197_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_197_test.npy) |
| drive_1FezRSA5aqw8IcvJO0B76VvTkjxW7C3Or | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_197_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_197_test_label.npy) |
| drive_1nrPJx0snm-GL9n2VCMLZQ4P9GGL0c-OQ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_197_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_197_train.npy) |
| drive_1jZTduPvUGqOKak56KT_AwsBEgJHwEVr7 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_198_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_198_test.npy) |
| drive_17x86ua6QMwxWIH9n6ek1cJza4Hfnjr7B | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_198_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_198_test_label.npy) |
| drive_1kzZMXnUCS1MZyZ0DakRx1whIjC6YncoO | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_198_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_198_train.npy) |
| drive_17unFEEklMZDVRS89q9KW0Sem02-lbnM0 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_199_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_199_test.npy) |
| drive_1Lzs80CiluWL5BQ3G2QzU50zuRYVmeQir | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_199_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_199_test_label.npy) |
| drive_198BnQuQL2g8-ZRhm7VfOUKUZXBMaYdoO | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_199_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_199_train.npy) |
| drive_1Y61qsy8Ebb5Op4AyvX__dljUFbFfeH8V | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_19_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_19_test.npy) |
| drive_1efs41XVquwXWUKcjoIxEgC9oMcm93wEJ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_19_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_19_test_label.npy) |
| drive_1jmrxNY2LK_5c5WVfnLvNj19pD3e_X_bS | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_19_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_19_train.npy) |
| drive_1mBJsNCxdQ8HsnwGvI3W_Y9qj1YBl-bxj | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_1_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_1_test.npy) |
| drive_1dH8hKrdq2fm_LpxJCX5VLpXRsZ28d9H4 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_1_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_1_test_label.npy) |
| drive_1g9zDd3-3HK5d-GqPYtp4GZ_Tiaj4_0yQ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_1_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_1_train.npy) |
| drive_1QRBPrROHc_iAHIdRzpWoAbVbzF93KsSa | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_200_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_200_test.npy) |
| drive_1_vSn4ewkT7y1FrZ_auHu_Wv3cDuzzVJi | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_200_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_200_test_label.npy) |
| drive_1LZfZ2CAWpQI0Xlk4gP7iBR0jyHpF1oLa | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_200_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_200_train.npy) |
| drive_1AhOL2g8a4aIxBxc2reojV-LzVFT4B694 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_201_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_201_test.npy) |
| drive_1fATuS2qHTdyyAZ9BCKCv0_yDcBqt16Yj | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_201_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_201_test_label.npy) |
| drive_1p-h1z-CUx8fEefFwrQT0xxKemNfyZO0S | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_201_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_201_train.npy) |
| drive_1XI1it2S0gvDWFLsyJdNwpuqPDnpSf2dM | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_202_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_202_test.npy) |
| drive_1Lq8LB6Tg-Vp_5NcfW2H4Cg_EbXVr1ycq | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_202_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_202_test_label.npy) |
| drive_18GSrLfyjvKEULdr8BhRkz8JZg1qvVmxA | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_202_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_202_train.npy) |
| drive_1ecXAaD7ETb5As03UABnaFmoy11gmZ3jK | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_203_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_203_test.npy) |
| drive_1tDzAAlD8ZIzK7UjPw-sxLxDVgSfYZL1y | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_203_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_203_test_label.npy) |
| drive_1ptb3fyn-QxOziuLfZfVM2htridZJUXHa | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_203_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_203_train.npy) |
| drive_1y2JylqTYb2_a0JRIuucafkHaaW9Mz4Vs | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_204_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_204_test.npy) |
| drive_13d7lzlUF7emAjLR6S0cyYFGH_YttcHcL | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_204_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_204_test_label.npy) |
| drive_1Y2JiLv_3I08mK9vfYFnf9WyjKyDXc12Q | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_204_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_204_train.npy) |
| drive_18vpSgKRJFI9BhsSpJ4YSPhKpAMih0ooj | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_205_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_205_test.npy) |
| drive_10FYVrvxRnDL2rhFiXAuOIGGoqWWSUDIO | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_205_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_205_test_label.npy) |
| drive_1KlGDSxJDGnQH6Zu3Cx3v8OdzwwDArlRZ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_205_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_205_train.npy) |
| drive_1wmzqHMSbKbZ2fBx1EQj_owHuZad1S5xW | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_206_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_206_test.npy) |
| drive_1htU9IBTF5tL0fVNBCauP5_s2oVZC34Zm | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_206_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_206_test_label.npy) |
| drive_1WtjLlmxrHxgkJhQN4laOAN3f64-I27mS | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_206_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_206_train.npy) |
| drive_1B7YbZ0LhLB1-dRN9UpUu9gZ_P1iYETxz | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_207_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_207_test.npy) |
| drive_1GUs2iDcU8eGvznXKF4zhf6wZ1PFt7NO4 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_207_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_207_test_label.npy) |
| drive_19upP_piqJcaUf-sie3nLuAnXMzCzra7g | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_207_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_207_train.npy) |
| drive_1vHJcQTifONWqlNpWyeYZcToJ2YrnfRx6 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_208_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_208_test.npy) |
| drive_16C-lyb969Z_09W5I22xU_e95sMJF5H_5 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_208_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_208_test_label.npy) |
| drive_1KwJNat5D49QhuLQr6PQvmZPZWVL7gU5W | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_208_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_208_train.npy) |
| drive_1c_vpGkuZGIK_w-5KEF4oa0Gu73cv6Ptg | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_209_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_209_test.npy) |
| drive_1CbpIbU5KlkIF6DMwLscLqpE6TF8MtLTR | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_209_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_209_test_label.npy) |
| drive_1gFsXPiJlsvfi7sqhW5JCi6i97gdzuO9e | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_209_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_209_train.npy) |
| drive_1xmVsK0oKKof1E4kjIY3PbJC8LDAFaRim | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_20_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_20_test.npy) |
| drive_1vUEveVREubnCxaq5bagamFPEoyZrM9xa | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_20_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_20_test_label.npy) |
| drive_1o8a9n56E5EBHbErZQlEEOZXUetgtTrSv | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_20_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_20_train.npy) |
| drive_1usGJF4cXQBtOm95qnUd_MnXkApXfat8h | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_210_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_210_test.npy) |
| drive_1AU1nwg1_GiIs0kxdV7RdEwWKcjwy6S2e | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_210_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_210_test_label.npy) |
| drive_1Vmh4gBCD4MUMpUDEiMZ-6Jd8YFhYplJB | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_210_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_210_train.npy) |
| drive_19qanvzff0SU_F1r5qN7AVQXMjMqG60Bh | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_211_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_211_test.npy) |
| drive_1ZSJuijpSKHji7TK4A0vAc_5Ar-yLEJ9O | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_211_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_211_test_label.npy) |
| drive_14_dGspfTK_BwTmTYMU4PDjOPhbuj8vP_ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_211_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_211_train.npy) |
| drive_1yypRZIeKBgDs4dnm140atPufL9Z1zwUs | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_212_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_212_test.npy) |
| drive_1ZCM9IAbPo6NwB86qkcDDHd2t1kZfG0hF | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_212_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_212_test_label.npy) |
| drive_14wtyNJC5WQL9KVIZ0T8-_OgpfwVsjRNq | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_212_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_212_train.npy) |
| drive_1JDlNEDhzBDO0fVoBUyHhnbfOqzmt3qp8 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_213_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_213_test.npy) |
| drive_1OajcTD6IR4XuiBfyIy9Eg-DO_qJcn5RF | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_213_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_213_test_label.npy) |
| drive_16IoRe6fQ018lX_r5QsSUq3Vkom34Ajg5 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_213_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_213_train.npy) |
| drive_1C5qw9dTjMDIEt0WoZ2VDIyTI48JBstrp | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_214_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_214_test.npy) |
| drive_1DOH1C8zfMhWjBXPnFkYK_jgQaN7EbsW4 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_214_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_214_test_label.npy) |
| drive_11P1OfGqTLvjYuidScUmA0dyUQVlxYaAl | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_214_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_214_train.npy) |
| drive_1bE5phUvmrZ56eZ3QDBZBuRn982vmvPi3 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_215_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_215_test.npy) |
| drive_1b_Vq_hOMszKwiEh9eSdFdjuT0dOWTC1_ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_215_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_215_test_label.npy) |
| drive_1JjenaRRj8F_XwHNZX8Xn2nzqfpeyMEn_ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_215_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_215_train.npy) |
| drive_1SHipJ67Ar5Za3SD92FHBuIZd3JXYulAq | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_216_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_216_test.npy) |
| drive_13s_UJP43MJ7cZxLCnCq1DcXGVMy8-AiB | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_216_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_216_test_label.npy) |
| drive_1zn3yJlOhIiougqM21APCCK9O_ziTgaGd | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_216_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_216_train.npy) |
| drive_1BueXl4Rms9vLHmESjlcl6PmtDMJEWA7_ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_217_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_217_test.npy) |
| drive_1ZzCfvaPRw7ahC5O9Ie8XZca0M_klj_Qb | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_217_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_217_test_label.npy) |
| drive_1EGklFFwWWRNJjiisYEu_qnGiwqYK-WjR | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_217_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_217_train.npy) |
| drive_1MSHsdqW1TdQttkfrvwC54IkeMLsvILgh | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_218_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_218_test.npy) |
| drive_1ZJAvmyXbt6cC1B8Jkdg4qwgbiqsgBaIM | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_218_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_218_test_label.npy) |
| drive_1FBxIcL5jYNv9HxTpDKr8FZT0316y2XCW | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_218_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_218_train.npy) |
| drive_1jOAfu_0lGLeLd5xyekCwm8ZQiQ_5RbFF | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_219_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_219_test.npy) |
| drive_14DEqyp5ym7MCZImgF7Dy7QGbqEfNsI7p | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_219_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_219_test_label.npy) |
| drive_1xL9GP6Y0kgAmxpujk972g309x1Dt40pO | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_219_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_219_train.npy) |
| drive_1qpslV2aavaN6MD4N9YXOIG6UkjfDL2tJ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_21_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_21_test.npy) |
| drive_18Jb9K4aaHPVUwIvU7lwJ1zlILUvAD0wi | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_21_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_21_test_label.npy) |
| drive_1uLX2bKLSJl5edEYNe4DkijuglI-zKLqC | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_21_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_21_train.npy) |
| drive_1QmNeKLIQfsskry2Grn2CPsZOkvbfmLM0 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_220_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_220_test.npy) |
| drive_1p8vU8i5ZyCbcnXSO-W_y2HsomZN_t4c_ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_220_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_220_test_label.npy) |
| drive_16yP8bEtbPHRkdNHpRCtSlUHki0fkUG_3 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_220_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_220_train.npy) |
| drive_1ch6GAUnPMv3UjUokKJ-8Wj7o8iSMLyFP | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_221_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_221_test.npy) |
| drive_1e-aA0GtztS39sjLEbhzwjn-akKjHnh4v | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_221_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_221_test_label.npy) |
| drive_17hnFAtfuaZLC-gKbh2OOmxO3SSVSQDyh | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_221_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_221_train.npy) |
| drive_1oCvI-JcWbQ8R2r45xkTIHB_OLr9jXw10 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_222_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_222_test.npy) |
| drive_1WAaAcnu0HARssxit0tYHoN7z3HAVsd5N | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_222_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_222_test_label.npy) |
| drive_1-zOr-kzmcxE5FKXnpC7bS4DeXSdnLYwx | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_222_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_222_train.npy) |
| drive_13yYgJEKU8UHvEdCjJ4XKEWIC9UitessI | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_223_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_223_test.npy) |
| drive_15DMcGLsaY2UNpJG1KvzT9YKMeNp7_T0I | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_223_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_223_test_label.npy) |
| drive_1zPGS-RjZOJE9six0HeU90f6WEc_NfO6T | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_223_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_223_train.npy) |
| drive_1qdy_dWlVhurjM44sd2_Zj3h7g1G0vwg9 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_224_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_224_test.npy) |
| drive_1SLjgGI_JTI_vI7PSnkNs0GlvwmH-R9Be | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_224_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_224_test_label.npy) |
| drive_1i8x7-Y4ff6pzaAJbZFD5kUFWU97zuPnn | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_224_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_224_train.npy) |
| drive_1UOk5KN1pL-a9hs9-0vTztrM9XtxQ6QRh | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_225_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_225_test.npy) |
| drive_1dZUFLG107b6EPND3G_S5_bG71fw0xRU6 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_225_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_225_test_label.npy) |
| drive_1VTrBnU6T4_IgsCHQTaRQHv3EgMLNN1OW | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_225_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_225_train.npy) |
| drive_1YBy133LC1kLsdaoZroPQNwReeA7buIIl | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_226_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_226_test.npy) |
| drive_1ivkijdZh063-WX_FieLp-4ZetUqqBygV | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_226_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_226_test_label.npy) |
| drive_1ZMCPVHeqy1nRqquZ4qp2stNnsYGpbJkt | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_226_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_226_train.npy) |
| drive_1u2YZstVFCSnm2X3qeFUg6ySI3AHWWTHN | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_227_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_227_test.npy) |
| drive_133rOwXxDY_iGddoknuAahHU0JJQ-5htH | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_227_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_227_test_label.npy) |
| drive_1IJ8OK3V8-C40zSPyzFdrN-BVY5bLadkG | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_227_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_227_train.npy) |
| drive_1fzFJLqbKT2BrXBgZvd-ytB1jM082fyO5 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_228_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_228_test.npy) |
| drive_19DXmfZMUR_m05yKJ6YC20akRm2EJj1Zq | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_228_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_228_test_label.npy) |
| drive_197NvCDCT_z-FX4BVANyvRyiRJSr_8HSu | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_228_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_228_train.npy) |
| drive_1relGjpVASbPQoOOq6pb_BFmYse4KXXp_ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_229_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_229_test.npy) |
| drive_12zsnQZ2bLo51Dh1jA4DGNC3n2ie55n1B | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_229_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_229_test_label.npy) |
| drive_1ew1I1CR3KIuBNmERjZn3yTss21PU16u- | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_229_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_229_train.npy) |
| drive_1d470xTiA1JWVFh1SXMaoSBDFcmQAM-qO | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_22_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_22_test.npy) |
| drive_1f4PAXh9F0RZTf0R6_YPDBV0nmX5yMfta | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_22_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_22_test_label.npy) |
| drive_1mh5MrWVUr7B-CDn2gBfcw8iURGKwrBVy | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_22_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_22_train.npy) |
| drive_1HeCaqkkv4m88EV1XBi2V6Z9pnDBhVZZl | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_230_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_230_test.npy) |
| drive_1m8XldeYRRNfQE1zdJ8F5Xdp8CKADJAf0 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_230_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_230_test_label.npy) |
| drive_1Ld7vz_zKGnD8QV8no-o6QX-1BUQ6wI4v | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_230_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_230_train.npy) |
| drive_15oVL2lQZz9PnrSeTHzUmAfffU2vJ6RBQ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_231_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_231_test.npy) |
| drive_1f8b7x-PX5azFvyXmUpEdfOKUyRef2mox | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_231_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_231_test_label.npy) |
| drive_1ZzsRAyZMkb5cn9VvbL3L4BI7_B_3cmzw | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_231_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_231_train.npy) |
| drive_1gDp30lvieN1MaBx5rPc5-QKOMmgDRsj0 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_232_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_232_test.npy) |
| drive_1sMAU-k3b9GBWGa_qgg3nq1XgQc92jtbD | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_232_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_232_test_label.npy) |
| drive_1oY8G_ihH0TvMQbfRdktkPI5TPrj9yBbs | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_232_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_232_train.npy) |
| drive_1I6ZjqoIYboDnG6ZH3pAxRi5n64a-KTPs | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_233_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_233_test.npy) |
| drive_1SNvSfnxobmiAaXAyzbRbfS2KQ5WZ3fbf | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_233_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_233_test_label.npy) |
| drive_15bw1TrmO54DnU_BKMLW1-921cXRp4_p7 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_233_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_233_train.npy) |
| drive_1IV5aJv8LDVvjDGkK1-KvqIWyBgTEx-mR | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_234_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_234_test.npy) |
| drive_1y_ItdLMHc8bb4NCOFIwWeP0w1Srv2F7F | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_234_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_234_test_label.npy) |
| drive_14PyuDUXFhkAX4iq-RKLWErdB9ID3YbAJ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_234_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_234_train.npy) |
| drive_1u6TW19IZ498oil5ZesCdgs4QQNvPjkyL | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_235_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_235_test.npy) |
| drive_1lQ6yuAVBu7i7ODAEnOsjcu1sTJoQQhr3 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_235_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_235_test_label.npy) |
| drive_1sW4WYliD-xr_nCYThaagSKCSbldHnGRc | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_235_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_235_train.npy) |
| drive_1kirVYu_c6Pu0nt1rX4n6UFFWn5vJLbsc | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_236_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_236_test.npy) |
| drive_1XlWIXX488akj5VQU-Ao7Bv0K9_R4Z1b- | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_236_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_236_test_label.npy) |
| drive_1nsdXI6Tl_JSq9IKIdnB07HKN3EMDsSpD | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_236_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_236_train.npy) |
| drive_1GnUlf-yhjP3_60ROY4U-ncDVeX0-89VE | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_237_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_237_test.npy) |
| drive_1MmlmOhsft4DFtUgpCq_7PBkVZHglxA7T | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_237_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_237_test_label.npy) |
| drive_15l2M5g09RSAiACSyAVIAqQdvkFrWbC0S | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_237_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_237_train.npy) |
| drive_1aseJRq81CaKWIEfZAEKx_IOHtmq6s5Kd | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_238_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_238_test.npy) |
| drive_1H7MibZI3uw0Gfm929Rgvf7X2c6YJOCur | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_238_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_238_test_label.npy) |
| drive_1ALzzMVdgIgyAisEs7c2XdHtRmYVPOHGB | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_238_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_238_train.npy) |
| drive_1xHH3ObxaCz0u5fAtU6s-u8_uNPQnLXWv | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_239_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_239_test.npy) |
| drive_1fLpQX71qllZNJCLbNgn1C0uOmg-4iXq9 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_239_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_239_test_label.npy) |
| drive_1XbjfuvexBT-9H4iBJ9I2sSC1_PgCmpQd | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_239_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_239_train.npy) |
| drive_1tPHnZLk4dlP2IqJSj7McRJm4haRrihuF | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_23_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_23_test.npy) |
| drive_1QS9wxSM_JFdGYeR5_xIsGshTZy_Wx5RT | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_23_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_23_test_label.npy) |
| drive_1iMT9gHt1lGY_WGEYCFGBz5xJvxrz3-nf | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_23_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_23_train.npy) |
| drive_1ypssyM_WsgG6fx2qz1ZOhmRm9Tu2RhwI | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_240_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_240_test.npy) |
| drive_19b-c0pNzlxVvUIUqKSUk6GgRsHuc2vRO | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_240_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_240_test_label.npy) |
| drive_1iyujZ1x-d18XvjtcFVSbV_KaJTk0ulDb | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_240_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_240_train.npy) |
| drive_1ojUG4RMoNhEO-qCaIwg5lQmW6AO0O-Vr | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_241_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_241_test.npy) |
| drive_1dN5mo0_Q8bMCieky2y-n6LYrdBMDr5cM | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_241_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_241_test_label.npy) |
| drive_1qoh_Kdfzctlzmu6OP0l5pMGeiEclBNX5 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_241_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_241_train.npy) |
| drive_1RLjVYmoZNABn_TQMZp8ZA0gnt7ws_AWu | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_242_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_242_test.npy) |
| drive_14r2z3Ka6l99P3a8IoF6dnui18ZBzNI9C | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_242_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_242_test_label.npy) |
| drive_1jc0wX0CekV6W4W_260-sBsNpDhEyNs-e | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_242_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_242_train.npy) |
| drive_14ofAKMgMYx1syRHqxFNQ3YcKFNh4Yg4Q | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_243_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_243_test.npy) |
| drive_18I0szuN7KG8aj6x7UB3EIaNos6NYNVIt | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_243_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_243_test_label.npy) |
| drive_1CEeJdlsZc_17o_6wJ7Ct2iRq4AipYqIR | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_243_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_243_train.npy) |
| drive_1YcjvZFdOgRuo8PfZ5Nmwup1GYj_BLQRw | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_244_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_244_test.npy) |
| drive_1YdIHPyjP498TlTQJKC23mrqPJHa19tUZ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_244_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_244_test_label.npy) |
| drive_1LRHzpCeTCuXPHyHaJ52L_SEmOFhU-oKt | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_244_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_244_train.npy) |
| drive_1p1ROdd2vSOs9-hpWPp0C4AF2k990H1M5 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_245_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_245_test.npy) |
| drive_1uVeJP_PoHL-Rr7tHdyoqr-1Cu18YDdMZ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_245_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_245_test_label.npy) |
| drive_17q8dsroJEJuy9Wd_f7xz7MJgkBiRCiMl | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_245_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_245_train.npy) |
| drive_1PxvmtZtgblT-rbuxCVoNHc0qFEzO28z5 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_246_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_246_test.npy) |
| drive_15QVYoUJGc9PwPxr4RIc240MYWLJ-WXhX | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_246_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_246_test_label.npy) |
| drive_1ggcNcqYOqZw31GuC74UnKePPRiSsnWmD | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_246_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_246_train.npy) |
| drive_1U5WHtRNypvkA1X4a4yq_dNwfFzwN-pMh | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_247_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_247_test.npy) |
| drive_1o5WmA9DjGxR0pXGVWKapzM1Twl3braxQ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_247_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_247_test_label.npy) |
| drive_1HYgpzneyRnw9q0P8zcOYRY10V0JVqUK8 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_247_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_247_train.npy) |
| drive_1wP2QzEVOFg3rNo2C5PT5lplvCdSv5E0o | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_248_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_248_test.npy) |
| drive_1PT12c2QVLKdc162NROjAy0NyHGgOTQq0 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_248_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_248_test_label.npy) |
| drive_1MyynOgDCTwGOQbeIfoOj88IY1S4M7u0N | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_248_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_248_train.npy) |
| drive_1lNnon2hIk19Xj4S5QLkTMhnQF1GyNoVP | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_249_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_249_test.npy) |
| drive_1JV8epRxJZv2cUkOIzgFSCbpsiWkE4dOa | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_249_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_249_test_label.npy) |
| drive_1VlafMs4AiNpcJs5mKk9wI-avHBXg0gzJ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_249_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_249_train.npy) |
| drive_1wrhn4SiqyiSk4YXM6xNmkvqnafmNQqdQ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_24_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_24_test.npy) |
| drive_1olwBuw4-bM4_3u4L2c3wJiyWMkD9zLUm | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_24_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_24_test_label.npy) |
| drive_1XnsAvtVUGC1Aai_oCCBqQH_F0ON3wKar | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_24_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_24_train.npy) |
| drive_1-7mI3AGVL1Jv4eY7k2sf11l1lkMisvyS | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_250_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_250_test.npy) |
| drive_19Ui7-wYOTVat_8DFv0AwYFGQIswSYM6P | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_250_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_250_test_label.npy) |
| drive_1ofI_wZ5gWBMXUpSE19JgkRrPCSiNCEo_ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_250_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_250_train.npy) |
| drive_1tUWfW2fT9WUCtO8m62v2EhXnxj8OVbBG | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_25_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_25_test.npy) |
| drive_1TcDzhpMWMUodBw3lNFDJtMSJBHAA_yIG | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_25_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_25_test_label.npy) |
| drive_1Bsfrmtkl1wQtNgWw7_iPcziF9yXLxerJ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_25_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_25_train.npy) |
| drive_1EBy-9Dmg3fpUEt1Ah3XjwB3Ubwn4orbz | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_26_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_26_test.npy) |
| drive_1OzmYY8L-exBZOEUk674NT97UIybc1O4F | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_26_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_26_test_label.npy) |
| drive_1bI_6L_ADFBx9Q0fatqdq72VjIAlhzvKf | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_26_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_26_train.npy) |
| drive_1BwVRogdDJgRGvUSgpcNIFPHZAJa6Pzsc | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_27_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_27_test.npy) |
| drive_1T8685JMJxtbw76k_cmc4gfeVjgpZs4Zr | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_27_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_27_test_label.npy) |
| drive_1ISkXt4EzfolVbogUiGKmb_Xdn_WaVW6V | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_27_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_27_train.npy) |
| drive_1xcYIMOkigW--eA2fJFHfDMvjOK3JHX1o | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_28_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_28_test.npy) |
| drive_1Srimg8SnTDD-vdArJfPGqWZpVHxEN9eR | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_28_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_28_test_label.npy) |
| drive_1V86awzl-ntjMHz3fW2bxsQ4USUG8y7q1 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_28_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_28_train.npy) |
| drive_11m9POizww8kUVsYbML0ZW7NA_0jrUYSh | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_29_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_29_test.npy) |
| drive_1uqaLDyYkzqpkfwmBgLDBZEb2g5Rnlieh | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_29_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_29_test_label.npy) |
| drive_1bqyUF2Wb2dF-QWXfikyEuvTowTPJzhAB | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_29_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_29_train.npy) |
| drive_1CxKApX3vtk4qcc6BtDLZd9QLBnmmnSAX | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_2_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_2_test.npy) |
| drive_1jKrmoTkkQmlr4m7x-N9OhWCxA74WqiAh | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_2_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_2_test_label.npy) |
| drive_1vadd-L44mdSLcf89yiNnPk5Br5TVdvDt | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_2_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_2_train.npy) |
| drive_17wQQux_b3YfcZnIHQQa8l9z8wm47dTKA | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_30_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_30_test.npy) |
| drive_1N3ftapKRmx4cEOv1rt9XHMzAO7zwwWUe | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_30_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_30_test_label.npy) |
| drive_1IbMfiI9BJpTYRZQzER37h3HrjW6C9-Ot | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_30_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_30_train.npy) |
| drive_1CTiHANnB4prCx_-HWOuKh2Lu08ASXnbV | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_31_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_31_test.npy) |
| drive_1w-y2izen_-l89zv3dX4cQxwSiw2wlFkY | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_31_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_31_test_label.npy) |
| drive_1sT00dDJkKbD02eMkz8-nNawd1V_eo8S_ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_31_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_31_train.npy) |
| drive_1vqYz6sBsmn6v93m2_0R5CGt_O8OYFdWo | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_32_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_32_test.npy) |
| drive_1r_75aybrdLi5JC_vWONyW69s9kDWpT2R | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_32_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_32_test_label.npy) |
| drive_1mam4KVchGm1Ac0lerFybXHKyc6aVw54k | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_32_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_32_train.npy) |
| drive_1QCDQq0GUs_1vo54N0Jz5h-u-8sfBeT69 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_33_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_33_test.npy) |
| drive_1rwd6pV3uahozkdMktJI1W70IxzPulU_W | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_33_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_33_test_label.npy) |
| drive_1VYCmqJHCeywIMjh-2JP6aDCUNnsX5rEk | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_33_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_33_train.npy) |
| drive_16rmQspt5v3dejB9i2TyXna_ao7HieOmd | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_34_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_34_test.npy) |
| drive_1jutwn7zOpqo02jq7d2epUG4PWXJDQk4B | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_34_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_34_test_label.npy) |
| drive_1cS7uYM39p21BUwMOPb0eF1kEpCPyJ6wq | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_34_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_34_train.npy) |
| drive_1CNDecv66DVAj-UcDDcBGyG8UyBpKniSu | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_35_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_35_test.npy) |
| drive_17B2vmLIt1rC9gBzJrwbrl3t-MJx6RCee | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_35_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_35_test_label.npy) |
| drive_1mUqvFsd-eElebhGam46c08Ztc2dmLQ7g | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_35_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_35_train.npy) |
| drive_1mCWx6SkMp93ZmCo9NFTPPo9fxh3srWz7 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_36_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_36_test.npy) |
| drive_1ee-7ku3M0nGKt27248VajRaXg7gTIuDq | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_36_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_36_test_label.npy) |
| drive_1vbJNqMrM8h212W3801xFGrATVALRiFSj | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_36_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_36_train.npy) |
| drive_1ZCwXd-44IUlkYVurVw-BpjexfbOIbOMb | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_37_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_37_test.npy) |
| drive_1d0LUtjN38O-KPKFw4R9coIFbrs27uy2j | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_37_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_37_test_label.npy) |
| drive_1L7c6Yw-8EvuhFT_q88Y-0yZDIjTfaBNX | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_37_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_37_train.npy) |
| drive_14pcB_CSZWflVAd1JngmCZIpw0G2f5qGw | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_38_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_38_test.npy) |
| drive_1fPYxrlbirH6j11Sgp5MXdsOWKBaMcsYf | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_38_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_38_test_label.npy) |
| drive_1PGefZmdhu07Lx8a9rLTQXBUcvhs1EZ2X | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_38_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_38_train.npy) |
| drive_1ql5SA2Hav_A113aczhizHQOo05loTjTO | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_39_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_39_test.npy) |
| drive_1SCvXvrmdfxPXjX-e7zhom8t_OIZXNS7l | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_39_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_39_test_label.npy) |
| drive_1V_pz0QNn20X9QrPhoYqpja6w70L-m_Qi | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_39_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_39_train.npy) |
| drive_1ylvK2nWhxTMkmlKsTGf9hngSP0_dLgOy | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_3_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_3_test.npy) |
| drive_1RhF4ayp89y0AFmnc42GdC6vjJ0OPNplW | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_3_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_3_test_label.npy) |
| drive_13xVG-V9QVyJyIC4cN6RSqmuKBSyguEpd | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_3_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_3_train.npy) |
| drive_1AIMB_1mwomxIxdlUkRltN8MbgSy53p3j | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_40_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_40_test.npy) |
| drive_1VtJT1Sgjo3rU6YdURT35pwxGVzM3VoF7 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_40_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_40_test_label.npy) |
| drive_1O7wYO5gWt6CVNkHRo2ZanHOEyCjodt6q | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_40_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_40_train.npy) |
| drive_13FM1M7HdYk11j808H9q7KL296z8NCGvK | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_41_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_41_test.npy) |
| drive_19VEjCLxwuDiMvz6FcDTps1dftg7RK00Q | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_41_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_41_test_label.npy) |
| drive_1XFWTmiFpaRBv4ZXJBYvtHXib7HntaF67 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_41_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_41_train.npy) |
| drive_1dOEEWE6N_Rj3aivKT5p7ygujIIVFmbwc | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_42_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_42_test.npy) |
| drive_1wVzscBkLvtcfn2kekp04luIx5eZElWf5 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_42_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_42_test_label.npy) |
| drive_1igEkiEZjSgmTXqSD0BZ2Yq4qqDVGLFlp | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_42_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_42_train.npy) |
| drive_1cQ47WOhuzIhwvltPul6-q-f1RuFRHhsX | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_43_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_43_test.npy) |
| drive_1DdeoPc5hXOo-RUQkaWqEaPpTUdafzvVM | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_43_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_43_test_label.npy) |
| drive_1Ls4AREyXaQjVoJhjNfinuYJrPLzQueVY | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_43_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_43_train.npy) |
| drive_1Ssi6yHLq7EHyRNuVG-MFpRhcvw4yTiV2 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_44_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_44_test.npy) |
| drive_1zsN3Y38etRbMQZ-c-1nkoa7lJ6a0ag38 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_44_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_44_test_label.npy) |
| drive_1ECPUVmUWZYXgW5hbIDvMjEmEjyr9MNl0 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_44_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_44_train.npy) |
| drive_1bbA7B048_bFX5EZz1qyq5N_-wLZXbq-z | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_45_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_45_test.npy) |
| drive_1cKXHAge95XfXp8TJMEYX6CDBoFcfBidp | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_45_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_45_test_label.npy) |
| drive_14m8c3GE4DeI5RZt_2eV_TkEZiTGiU0TH | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_45_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_45_train.npy) |
| drive_1V0cIN60TPk-I5fhqIWPZdUm9POnvxHQf | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_46_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_46_test.npy) |
| drive_1Lr5OD5odSCWp6nkDE7lEy4N5DFLvBhnQ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_46_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_46_test_label.npy) |
| drive_1bTQwMtJ3rpRKOEl6SJ-nB7w1bsbt0yiQ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_46_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_46_train.npy) |
| drive_1X4yEAhvPqQ13yNOLrViTktD_ty-toOjf | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_47_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_47_test.npy) |
| drive_1h4OS7vWNoFhHHk2yFIe5tr2fVMK3Tqib | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_47_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_47_test_label.npy) |
| drive_1cShKu8nXK7yupnyrRQjAKp1I5WkDEeTG | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_47_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_47_train.npy) |
| drive_1fVFfd7OBghkdsJ6xNZQudz6cAJZXEpB0 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_48_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_48_test.npy) |
| drive_16M1lw1St9AHpAj0oIYOMObcSgCf6pjp1 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_48_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_48_test_label.npy) |
| drive_1Vp3dNI4BysKsyV_MMNYfc_6jYd3HvtaT | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_48_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_48_train.npy) |
| drive_17wNIKF_DR1bioaE0euJ_G8ioig9xYnaJ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_49_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_49_test.npy) |
| drive_19d5jxbWsdQ5gszMOZoaGZN3Ep66PKe5o | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_49_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_49_test_label.npy) |
| drive_1PrK9snsyskU7r2gP08CFc_TfWMzdp0Hv | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_49_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_49_train.npy) |
| drive_1kWkkBDN9Ul4VhphzlHaijaRiZlKL0nDT | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_4_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_4_test.npy) |
| drive_1odiuWwczJxkKV3efcliyjG5-JFPG9MrG | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_4_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_4_test_label.npy) |
| drive_1obcBhP_86sMrppD_uCyW6IQZmF4UDvUa | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_4_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_4_train.npy) |
| drive_1f3llt34ph5xbGzt20_kdykAPD0zz92NN | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_50_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_50_test.npy) |
| drive_10tu2fJb8j2dXYA622_K3EdTBhZC8ZcOS | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_50_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_50_test_label.npy) |
| drive_1XNWWeYqDGU3DFM22XNaH0ornZxMsjUu2 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_50_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_50_train.npy) |
| drive_1l5l-TYwGHNp0LWferqb9up_KeVrOUmkH | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_51_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_51_test.npy) |
| drive_1Vgpuitak8Z9r9Hr6bNxz9bTClQe-JLCB | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_51_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_51_test_label.npy) |
| drive_1-cWRUDG8Vixh2AtWNKN03zAJbpS5Gx4R | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_51_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_51_train.npy) |
| drive_1s9XRpAab2cmqSg8plmVZlxQcNjmAbP90 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_52_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_52_test.npy) |
| drive_1qNIJp8kiKHShHoNqAivWM-wh3cTCLu7U | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_52_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_52_test_label.npy) |
| drive_1qGQ5X44S9_RIf90mw0lt5e01IsA3zwsX | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_52_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_52_train.npy) |
| drive_1IY6sI7ZRWQUBkIjhpcoJgUqEdCd8kng8 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_53_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_53_test.npy) |
| drive_1eLvFjCq2U3VFjaz10siz8777Vlf3syj5 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_53_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_53_test_label.npy) |
| drive_13pe8AyCW-wvs5eB78qZRgX-jFZ5S1ZwL | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_53_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_53_train.npy) |
| drive_1mE_co4khCMvtS4-6wc8Bj66bmy0tkyyk | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_54_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_54_test.npy) |
| drive_1fTCpIHM8kIhD4SppYyHPgw5pUoBT1pTc | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_54_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_54_test_label.npy) |
| drive_1Dlu2IGLK3ljrpj9rjbRals0AAV5z2JH0 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_54_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_54_train.npy) |
| drive_17qPwKB9MJgq_yEvCNwGSC2VHK-QJt-ik | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_55_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_55_test.npy) |
| drive_1HdTML8FLmrzZPhDKo9I4pWpocHL9MpRy | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_55_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_55_test_label.npy) |
| drive_11DOrHbuJlDE0GK54Z98FtI8KBVnkrTGM | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_55_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_55_train.npy) |
| drive_1bl-kHMbhHpei5a_WJ-cZfjktbVuEayPg | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_56_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_56_test.npy) |
| drive_1j5UaPKMxJZFPRFnPDDmfCii4nMy0g5FG | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_56_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_56_test_label.npy) |
| drive_17pBKYSHybL5NfbWiLoJ9aWlnN9vK0a5H | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_56_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_56_train.npy) |
| drive_1cGGrmHgHqsE_rulm6Bvw34rFxwICI5_l | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_57_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_57_test.npy) |
| drive_1NjvjYkYv3Z-_JxT9VZoGV9mGScWZ6v69 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_57_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_57_test_label.npy) |
| drive_1xCVJZodyG62hjMC_r8HwMQPd2gitVX1c | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_57_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_57_train.npy) |
| drive_1csUjRFBMYx7snOjl6j3BiSBSM3I2HWpD | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_58_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_58_test.npy) |
| drive_18o1CMZGubntmL_MkSKKt8Ndc6CExZeWW | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_58_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_58_test_label.npy) |
| drive_1x6fJh2R76qHJgEdZf1JkuSFEJWjDKF7p | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_58_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_58_train.npy) |
| drive_1zCmF7EfCMmcX3dq6SyaoE-pIeJW7IQUO | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_59_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_59_test.npy) |
| drive_1KmZLlMtFfWPH_aK7TiS7fC6yD46YMkhY | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_59_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_59_test_label.npy) |
| drive_1Xmec1B1T2tXSh2y5TxlXs3OJFfmoD81g | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_59_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_59_train.npy) |
| drive_1sd7SiCtVD41jxUDr4nn6mqJj8hixA15O | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_5_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_5_test.npy) |
| drive_1-oSVr0BFmtFCcYlfqm0xMJJ5IPlza0V4 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_5_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_5_test_label.npy) |
| drive_1ShwcFpnRGUAPKQa1dzucgz7hO6QfOF7D | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_5_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_5_train.npy) |
| drive_1OVKLfnYCQ11t-H1xSI7b9h9g_OHSVyec | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_60_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_60_test.npy) |
| drive_1o4boxpQEm_BgqXpTKsbEXKcdBdX9RqGf | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_60_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_60_test_label.npy) |
| drive_1B3gmv6VNCIA5bipj3V6MU0fDScT11QTf | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_60_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_60_train.npy) |
| drive_1R5q6DzK7qIj5Y_KwXICQ0XvXxIZ2XPqQ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_61_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_61_test.npy) |
| drive_1UfYO7R01RAPmqDqOR2MlcneejwP9TlLj | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_61_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_61_test_label.npy) |
| drive_1bGqaoAR2T19KqNgdUsfJumhACIfM4JRh | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_61_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_61_train.npy) |
| drive_1x7vNRC91gvIbU580aGpmiFw5Kqzoqpne | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_62_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_62_test.npy) |
| drive_1q3IFFXRnbs7eplEhyLqBekwxKP_9oCj7 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_62_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_62_test_label.npy) |
| drive_1EQH1f5M5iO_fBncado1p2LmS0XIOAMgc | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_62_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_62_train.npy) |
| drive_1WO_s5HuEf4dhNd6qvd8Pgdeie4DHnM7M | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_63_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_63_test.npy) |
| drive_1ChNvDHgACNr7MANKnK5SGTPmQuyoAhzN | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_63_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_63_test_label.npy) |
| drive_1gVLJHjIAy658l4Mk3Zuc4Q7Rl4F1EbXe | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_63_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_63_train.npy) |
| drive_1gbtfnq60q81tSyadi60BM68oUl7YM_7b | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_64_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_64_test.npy) |
| drive_1hrg_pY1LqRrn1xusvPBDccpbHEy_YiaV | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_64_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_64_test_label.npy) |
| drive_1m0KYuJcT2d9zWBggOYgGOW0OFjHUav3b | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_64_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_64_train.npy) |
| drive_1YdAXzycfbqw2YW1Q0ay7yrnEXDQK7vow | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_65_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_65_test.npy) |
| drive_1u_8DLJzwwJTc4tZiwJegh7cy0G0I1KVx | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_65_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_65_test_label.npy) |
| drive_1A1gheTj2WhqE9j6z2-wlES6koqYNCB0c | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_65_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_65_train.npy) |
| drive_1oi6fvFXQapD3VdX1KBT6nq1eSH9BHVkE | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_66_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_66_test.npy) |
| drive_1x1hFe6fl9N08mj4-HtYgqMV8XfdPm9sG | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_66_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_66_test_label.npy) |
| drive_1St8gAkGVC7uKXOf-m8Nm310JEY69K-NL | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_66_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_66_train.npy) |
| drive_1lVkql_p-Re5rQ5UGhzP4nKdimsIdv2QH | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_67_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_67_test.npy) |
| drive_10pOkKF-vfU4OsQ-EygG4QNAUZJMcrSTL | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_67_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_67_test_label.npy) |
| drive_1cJeHKXXEC5S5dwwWcrMdZ6_1c7fbd0Me | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_67_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_67_train.npy) |
| drive_1b5W9JWT6rvcbD-E6Xe2azfmFjIAtd1b4 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_68_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_68_test.npy) |
| drive_1mErDZ54Gl__G_HihUa550f2PqIVTV_sA | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_68_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_68_test_label.npy) |
| drive_1uTObsbpYC3x1bnjyrosxOyerFcK2rrJd | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_68_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_68_train.npy) |
| drive_15U2WHomNnjRjQgcaH_ExK1nJRlMssmw7 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_69_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_69_test.npy) |
| drive_1L0o9ddt967SF1uwmWCGnVwi0bOKuD1eD | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_69_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_69_test_label.npy) |
| drive_17riRcMd6WjDmURk4HD9zi8XYVSTawRVs | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_69_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_69_train.npy) |
| drive_16aygIkNMJKEnKVvhDLThhZx2P8u8WTO6 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_6_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_6_test.npy) |
| drive_1kBgl6Kgf6JVPGrAztCdnt4pll6c_ANhI | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_6_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_6_test_label.npy) |
| drive_1KPyJ2TNjroE0tbU3PfNifB1SmCGaY04n | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_6_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_6_train.npy) |
| drive_1fGcVgQd-U-T7TUGgoKygI_d2hBVrAzBq | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_70_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_70_test.npy) |
| drive_1B_aB4IV5NAv_UnpgRDS6s4BxyJgnGo48 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_70_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_70_test_label.npy) |
| drive_1xBifxBAqsEN1WDeBoiTHzKoVE05x1vxm | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_70_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_70_train.npy) |
| drive_1CRBpUozIEgGgp2FDVyYdVsyAPN9mokip | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_71_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_71_test.npy) |
| drive_14Dv3jZrhW4jG2YnTB0Jhmes7OC3hXdaf | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_71_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_71_test_label.npy) |
| drive_1MZeiF0LeGJ9BfaNlhBS3Jv44cqk2xT6a | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_71_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_71_train.npy) |
| drive_1-pXyWqWYkeVEECqSITsXUe3jTFS9NBeR | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_72_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_72_test.npy) |
| drive_1d0I0Ge7Wyc_cGkClaY7lUGBGNah1Mt1E | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_72_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_72_test_label.npy) |
| drive_1-e-7dQvGAV3TwjCv4i-orFlBYOQwtWVu | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_72_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_72_train.npy) |
| drive_1Rx56KevrTs7KF1B9maWDEwtLjapI8rWo | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_73_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_73_test.npy) |
| drive_1B32ArYW0gp51KWaWo-oqrqNQUg30heHM | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_73_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_73_test_label.npy) |
| drive_197GS0tBd7xI9j3AlFJxRXPQTMp4_Pr24 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_73_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_73_train.npy) |
| drive_1xMlCT3eleQjXMzVd42ycp4LFvLmZC-0i | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_74_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_74_test.npy) |
| drive_16VAd_oeqriX2jSRXSzUWlmF2PSXjmfUq | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_74_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_74_test_label.npy) |
| drive_1VFz3nTDBpIuwZKZ_BctuHzxRgnolQCOC | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_74_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_74_train.npy) |
| drive_1vAdYCZ7Vb38XkeKTTOGlpkcZOlSjOeUW | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_75_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_75_test.npy) |
| drive_1ruMwcg8-y2KJPlue2fIHtSKYbzKqjnit | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_75_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_75_test_label.npy) |
| drive_19Iq7SWja31YdU8dvNAzkfqe75HAcGv3C | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_75_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_75_train.npy) |
| drive_1j_kGSWbqHFCEvFSo83V_dwWuE0bYmZAX | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_76_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_76_test.npy) |
| drive_19f16ERaGbvFFgOUxz6UjoFyIkvialSLD | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_76_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_76_test_label.npy) |
| drive_1fBJ-1xTNRLjH85JuXQT_viHNkf-sX7M8 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_76_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_76_train.npy) |
| drive_1gv2BwJrp6BZX3ZJQrw39t6KuBSwLL2ju | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_77_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_77_test.npy) |
| drive_10WMdtVm57MhBSC1bVILtzjaIBUow4xVM | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_77_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_77_test_label.npy) |
| drive_1gsrM62MtsMKvQbD3mdLbb9LKCQCtYb7z | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_77_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_77_train.npy) |
| drive_1T5rn2yFcxaiXVOvWKo7wmlJETnsXfWFR | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_78_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_78_test.npy) |
| drive_1VrRme1KIfbOf1_I4jgQGKgzgjSDihKwg | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_78_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_78_test_label.npy) |
| drive_12AaHG3XBY95AEbaV4mzwSSLqd8-xn-9s | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_78_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_78_train.npy) |
| drive_1UySZfID42AFyjPSXuLi_H8wOoBmCppSP | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_79_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_79_test.npy) |
| drive_10OCvA6l349U0oiOgqf9uVUVJWPJJ4kpv | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_79_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_79_test_label.npy) |
| drive_1rNvKX8uoMJRzIj3G-4CZculNVtyMh--5 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_79_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_79_train.npy) |
| drive_1_a7q_R5gXKh_sb-6U5Z3ksgySO13vbYK | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_7_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_7_test.npy) |
| drive_1ema4WKwJALPjrQ_vcHJ5TcUojt_4E4bg | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_7_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_7_test_label.npy) |
| drive_1lU00YDWl1MGxN5laFsUmmAj2Z2gNXxsd | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_7_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_7_train.npy) |
| drive_1JuoUtdUM6P9Wkjiq0gaosyv8VGmkplsB | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_80_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_80_test.npy) |
| drive_1GxyXFJMD-5tSIBEOk-zyfgPnuhRPvUqV | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_80_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_80_test_label.npy) |
| drive_1OyGx5zrSLVWPmDLgqMZTInxyV00BXQyN | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_80_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_80_train.npy) |
| drive_1kRoImuIPee5dE-m9MGU3q2E-wNMwPkoF | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_81_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_81_test.npy) |
| drive_1J7RgppGGyJuA4ycpa7duX-Iea-WfgDfS | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_81_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_81_test_label.npy) |
| drive_1MPK04BIc9pZyjGrog_Wrzgjig7f6vet5 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_81_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_81_train.npy) |
| drive_1SpJ0WBwjkUyG9Z4d6eNMf38fjDE7uCkN | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_82_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_82_test.npy) |
| drive_1u1lpDYNwR65gXlm2eV_N1n8r7J2z2OlX | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_82_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_82_test_label.npy) |
| drive_1FRzu36xZpLx6iWr26wo92jl7Kdx1Qycd | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_82_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_82_train.npy) |
| drive_1LBackPqEIOvePYdVLdkjfGwY5SaWZb2r | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_83_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_83_test.npy) |
| drive_1TqBbVlHxi8HsdjiJgU_q5uUD1IwCzTA3 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_83_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_83_test_label.npy) |
| drive_1D_Mnmc2bqd2H_meN5Lf5fGmuN7KEsd4s | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_83_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_83_train.npy) |
| drive_1tqLF0yPo2qAz6E6QNMrqPsEbUGZgGLXm | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_84_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_84_test.npy) |
| drive_1tdJfkDTDjqu8F-ZXOEinDOr5pmBSbvsw | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_84_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_84_test_label.npy) |
| drive_1YrJYJnzpl4D6Mey2t2zvRIOSksAr58jL | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_84_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_84_train.npy) |
| drive_1_UPPfYNUFFQu8ZqJyuK9h-oQ9llRRGys | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_85_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_85_test.npy) |
| drive_1FSXFS9YdHQMRj5IIwkNRY7gOB3RO9POD | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_85_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_85_test_label.npy) |
| drive_1NcXi9L97yMdlV77O0AWrbn27PYXzBHJw | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_85_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_85_train.npy) |
| drive_1Tpl-1i_4KGGAauKoatoFhSbx4aoUgh5K | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_86_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_86_test.npy) |
| drive_18jyLiDJVz8MTasMW-_LBqypoIPWc6R0B | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_86_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_86_test_label.npy) |
| drive_1sGuW7F_n6qFOBdmhCrlGFoq9BDHq9-e7 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_86_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_86_train.npy) |
| drive_1faWLqFsCpBxFUwGQZj9tNZPyrK6TgY80 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_87_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_87_test.npy) |
| drive_1UDe3T7YbVzh8l-Jdp9ZqPcattWS1Mmh_ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_87_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_87_test_label.npy) |
| drive_1e18EJUFgA8clmf3Bw-TMmJLAP09mP1JD | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_87_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_87_train.npy) |
| drive_1F2aTA-PnG6sylFxfTXbr4jlOIIbk31pG | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_88_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_88_test.npy) |
| drive_1wQO-dhb9sd-Ra8x4kP-5EXDRxCaA1fPq | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_88_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_88_test_label.npy) |
| drive_1OjoOT8Uphq4XOJvdPOdurvbzMpdT-RS- | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_88_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_88_train.npy) |
| drive_1dYzeyZZgjLNH7eTYFdusq7ye7pvgvAlv | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_89_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_89_test.npy) |
| drive_10_2cmGxur7NJLEzaTG-r8E3E4-bo9kk9 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_89_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_89_test_label.npy) |
| drive_1kQh-WZse0wCJ9G-0pR4hTVS82_YQQzAX | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_89_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_89_train.npy) |
| drive_1WCOA7268KQ6A9AnbUZLeZu7rtUgh1aMZ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_8_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_8_test.npy) |
| drive_11yXrHk164boyU2HggFdRzHcv-gdb3mFt | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_8_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_8_test_label.npy) |
| drive_1pT-AftNlCmCCcz0TSZbjMUpyEvw0cctA | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_8_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_8_train.npy) |
| drive_1VEV7iTQG-HSDmTC4I5XKmE8e2Mr8AIUd | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_90_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_90_test.npy) |
| drive_17O9C9VMdkuVWlfG4S4Yr2vZbxDP7Toh6 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_90_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_90_test_label.npy) |
| drive_1LLTCHqaFGmBIVHSF-5plrac8zec2n3Nt | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_90_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_90_train.npy) |
| drive_1DIQrMB1IeWMf-XuLAUz9PEFfTQdG9STL | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_91_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_91_test.npy) |
| drive_1Jt967hGbf1nKkI68tjkYvZ60viKpCrit | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_91_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_91_test_label.npy) |
| drive_1O3sMNkYl_EMDY-rEjOe_pB0a-iUUriUb | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_91_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_91_train.npy) |
| drive_1bD67UOBQ3GHmzYsoPXq5vsTmM-iRAYZ4 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_92_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_92_test.npy) |
| drive_1KbHDG-FOHgF1TQmQcw1ztIerVqSWJQoM | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_92_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_92_test_label.npy) |
| drive_1d_GTydrLdOU1xxB_VQxC_cCLIlTUKIT7 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_92_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_92_train.npy) |
| drive_17l_MUj9IlqGoURgnXAFYMNw_H6rIfjsI | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_93_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_93_test.npy) |
| drive_1g7rvC4q4T07_5Xz81KIIVqDzv0Nl0sEB | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_93_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_93_test_label.npy) |
| drive_1m55nsDVlgcx_taJ0MSXDuvRbMC1-WiAA | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_93_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_93_train.npy) |
| drive_1jDIFslzYbeb9wUbduLFrWWWAWZvlEnET | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_94_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_94_test.npy) |
| drive_1pQmSvJkFnEyeA_NjTiGLxIq1a5qhGmSr | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_94_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_94_test_label.npy) |
| drive_1ztb1o0PQjcL5Mbz4zo7j19OmNQEZ16QR | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_94_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_94_train.npy) |
| drive_1S9fAAGY8eKy3HzTrXVuiVTwHpf4lqJ-c | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_95_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_95_test.npy) |
| drive_1wHmGZBf6m_lVFpzldf01xqBDRJILrz_F | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_95_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_95_test_label.npy) |
| drive_1YD_yyrXoRIBqMyF3ovZiI--Wb7mWCY7b | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_95_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_95_train.npy) |
| drive_1nCH3UJQmNTf1W5id5n1PYXG2q35GpakA | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_96_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_96_test.npy) |
| drive_1TMmJZJBDtSiU_7fUtM3kmhVbYEgdibFx | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_96_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_96_test_label.npy) |
| drive_1vipHkn2Xa1MMCpnsiq5YFe12vD0mZYzX | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_96_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_96_train.npy) |
| drive_1Zy13pqI2cuMpvis7FheDIhtSilgl8n-L | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_97_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_97_test.npy) |
| drive_1gTOF8lWDNqXGj9fLllWsurLTmD6eBpVP | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_97_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_97_test_label.npy) |
| drive_1Lw18ClwGbcDOMGiGG1RvWFJ1JkvbbKYZ | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_97_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_97_train.npy) |
| drive_1u6RFErfFWnAgUUtugqdiAc169j9fSUq4 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_98_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_98_test.npy) |
| drive_1iDIgCPaHEgxGlFal7yAQmCMpxjGBMR74 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_98_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_98_test_label.npy) |
| drive_1-35aitCESvhPO9bU5lEQLZSUy5Dz7krB | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_98_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_98_train.npy) |
| drive_1RaOl3AP44jhEx8YveQnovluwUViqoRhm | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_99_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_99_test.npy) |
| drive_1QMgWWmu_mnBS4RPgFSdEgWgCjX7CK0n9 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_99_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_99_test_label.npy) |
| drive_1uiGe9ggsK3yhnin04leI2MKx-QcWXomb | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_99_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_99_train.npy) |
| drive_1fcWa13XwFEQQbEaUW5rWNtT742CDRMIh | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_9_test.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_9_test.npy) |
| drive_17thH_49POiYU3AjbdPwnwjLxHBc80nBp | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_9_test_label.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_9_test_label.npy) |
| drive_1Z6f5sn4S3XWIQ7-Fcum-5Nim3Pq--URx | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_9_train.npy](../../data/public_datasets/flow_matching/dcdetector/author_bundle/UCR/UCR_9_train.npy) |

### Flow Matching for Generative Modeling

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/flow_matching.pdf](../../papers/literature/flow_matching/pdfs/flow_matching.pdf)

实验来源：[原论文/官方证据](https://arxiv.org/html/2210.02747)。

论文列明实验数据集：cifar10；imagenet_32；imagenet_64；imagenet_128。

当前可用资料分类：没有已完成且现场可用的注册资料。

- **synthetic_note**：Gaussian base noise and illustrative vector-field plots are generated, not separate external benchmark datasets.

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| cifar10 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/cifar10/cifar-10-python.tar.gz](../../data/public_datasets/flow_matching/cifar10/cifar-10-python.tar.gz) |
| imagenet_32 | 受限/未公开 | login_terms_required | [data/public_datasets/flow_matching/imagenet/imagenet32.manual](../../data/public_datasets/flow_matching/imagenet/imagenet32.manual) |
| imagenet_64 | 受限/未公开 | login_terms_required | [data/public_datasets/flow_matching/imagenet/imagenet64.manual](../../data/public_datasets/flow_matching/imagenet/imagenet64.manual) |
| imagenet_128 | 受限/未公开 | login_terms_required | [data/public_datasets/flow_matching/imagenet/imagenet128.manual](../../data/public_datasets/flow_matching/imagenet/imagenet128.manual) |

### Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/rectified_flow.pdf](../../papers/literature/flow_matching/pdfs/rectified_flow.pdf)

实验来源：[原论文/官方证据](https://arxiv.org/html/2209.03003)。

论文列明实验数据集：cifar10；lsun_bedroom_train；lsun_bedroom_val；lsun_church_outdoor_train；lsun_church_outdoor_val；afhq_original；celeba_hq_stargan_release；celeba_hq_original_reconstruction；metfaces_aligned；officehome；domainnet_clipart；domainnet_infograph；domainnet_painting；domainnet_quickdraw；domainnet_real；domainnet_sketch；rectified_flow_toy；rectified_flow_reflow_pairs；domainnet_clipart_train_index；domainnet_clipart_test_index；domainnet_infograph_train_index；domainnet_infograph_test_index；domainnet_painting_train_index；domainnet_painting_test_index；domainnet_quickdraw_train_index；domainnet_quickdraw_test_index；domainnet_real_train_index；domainnet_real_test_index；domainnet_sketch_train_index；domainnet_sketch_test_index；metfaces_metadata。

当前可用资料分类：原始/基础公开数据或作者数据包 272 项。

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| cifar10 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/cifar10/cifar-10-python.tar.gz](../../data/public_datasets/flow_matching/cifar10/cifar-10-python.tar.gz) |
| lsun_bedroom_train | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/lsun/bedroom_train_lmdb.zip](../../data/public_datasets/flow_matching/lsun/bedroom_train_lmdb.zip) |
| lsun_bedroom_val | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/lsun/bedroom_val_lmdb.zip](../../data/public_datasets/flow_matching/lsun/bedroom_val_lmdb.zip) |
| lsun_church_outdoor_train | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/lsun/church_outdoor_train_lmdb.zip](../../data/public_datasets/flow_matching/lsun/church_outdoor_train_lmdb.zip) |
| lsun_church_outdoor_val | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/lsun/church_outdoor_val_lmdb.zip](../../data/public_datasets/flow_matching/lsun/church_outdoor_val_lmdb.zip) |
| afhq_original | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/afhq/afhq.zip](../../data/public_datasets/flow_matching/afhq/afhq.zip) |
| celeba_hq_stargan_release | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/celeba_hq/celeba_hq.zip](../../data/public_datasets/flow_matching/celeba_hq/celeba_hq.zip) |
| celeba_hq_original_reconstruction | 受限/未公开 | manual_reconstruction_required | [data/public_datasets/flow_matching/celeba_hq/original_reconstruction.manual](../../data/public_datasets/flow_matching/celeba_hq/original_reconstruction.manual) |
| metfaces_aligned | 原始/基础公开数据或作者数据包 | manual_drive_folder | [data/public_datasets/flow_matching/metfaces/images.manual](../../data/public_datasets/flow_matching/metfaces/images.manual) |
| domainnet_clipart | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/domainnet/clipart.zip](../../data/public_datasets/flow_matching/domainnet/clipart.zip) |
| domainnet_infograph | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/domainnet/infograph.zip](../../data/public_datasets/flow_matching/domainnet/infograph.zip) |
| domainnet_painting | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/domainnet/painting.zip](../../data/public_datasets/flow_matching/domainnet/painting.zip) |
| domainnet_quickdraw | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/domainnet/quickdraw.zip](../../data/public_datasets/flow_matching/domainnet/quickdraw.zip) |
| domainnet_real | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/domainnet/real.zip](../../data/public_datasets/flow_matching/domainnet/real.zip) |
| domainnet_sketch | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/domainnet/sketch.zip](../../data/public_datasets/flow_matching/domainnet/sketch.zip) |
| officehome | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/officehome/OfficeHomeDataset_10072016.zip](../../data/public_datasets/flow_matching/officehome/OfficeHomeDataset_10072016.zip) |
| celeba_original | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/celeba/img_align_celeba.manual](../../data/public_datasets/flow_matching/celeba/img_align_celeba.manual) |
| rectified_flow_toy | 生成配方/源码 | 无下载记录 | [data/public_datasets/flow_matching/synthetic/rectified_flow_toy.generated](../../data/public_datasets/flow_matching/synthetic/rectified_flow_toy.generated) |
| rectified_flow_reflow_pairs | 生成配方/源码 | 无下载记录 | [data/public_datasets/flow_matching/synthetic/reflow_pairs.generated](../../data/public_datasets/flow_matching/synthetic/reflow_pairs.generated) |
| domainnet_clipart_train_index | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/domainnet/clipart_train.txt](../../data/public_datasets/flow_matching/domainnet/clipart_train.txt) |
| domainnet_clipart_test_index | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/domainnet/clipart_test.txt](../../data/public_datasets/flow_matching/domainnet/clipart_test.txt) |
| domainnet_infograph_train_index | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/domainnet/infograph_train.txt](../../data/public_datasets/flow_matching/domainnet/infograph_train.txt) |
| domainnet_infograph_test_index | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/domainnet/infograph_test.txt](../../data/public_datasets/flow_matching/domainnet/infograph_test.txt) |
| domainnet_painting_train_index | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/domainnet/painting_train.txt](../../data/public_datasets/flow_matching/domainnet/painting_train.txt) |
| domainnet_painting_test_index | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/domainnet/painting_test.txt](../../data/public_datasets/flow_matching/domainnet/painting_test.txt) |
| domainnet_quickdraw_train_index | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/domainnet/quickdraw_train.txt](../../data/public_datasets/flow_matching/domainnet/quickdraw_train.txt) |
| domainnet_quickdraw_test_index | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/domainnet/quickdraw_test.txt](../../data/public_datasets/flow_matching/domainnet/quickdraw_test.txt) |
| domainnet_real_train_index | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/domainnet/real_train.txt](../../data/public_datasets/flow_matching/domainnet/real_train.txt) |
| domainnet_real_test_index | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/domainnet/real_test.txt](../../data/public_datasets/flow_matching/domainnet/real_test.txt) |
| domainnet_sketch_train_index | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/domainnet/sketch_train.txt](../../data/public_datasets/flow_matching/domainnet/sketch_train.txt) |
| domainnet_sketch_test_index | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/domainnet/sketch_test.txt](../../data/public_datasets/flow_matching/domainnet/sketch_test.txt) |
| metfaces_png_10075-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10075-00.png](../../data/public_datasets/flow_matching/metfaces/images/10075-00.png) |
| metfaces_png_10076-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10076-00.png](../../data/public_datasets/flow_matching/metfaces/images/10076-00.png) |
| metfaces_png_10092-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10092-00.png](../../data/public_datasets/flow_matching/metfaces/images/10092-00.png) |
| metfaces_png_10093-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10093-00.png](../../data/public_datasets/flow_matching/metfaces/images/10093-00.png) |
| metfaces_png_10132-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10132-00.png](../../data/public_datasets/flow_matching/metfaces/images/10132-00.png) |
| metfaces_png_10136-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10136-00.png](../../data/public_datasets/flow_matching/metfaces/images/10136-00.png) |
| metfaces_png_10165-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/10165-00.png](../../data/public_datasets/flow_matching/metfaces/images/10165-00.png) |
| metfaces_png_10176-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/10176-00.png](../../data/public_datasets/flow_matching/metfaces/images/10176-00.png) |
| metfaces_png_10185-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10185-00.png](../../data/public_datasets/flow_matching/metfaces/images/10185-00.png) |
| metfaces_png_10210-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10210-00.png](../../data/public_datasets/flow_matching/metfaces/images/10210-00.png) |
| metfaces_png_10217-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10217-00.png](../../data/public_datasets/flow_matching/metfaces/images/10217-00.png) |
| metfaces_png_10218-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10218-00.png](../../data/public_datasets/flow_matching/metfaces/images/10218-00.png) |
| metfaces_png_10219-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10219-00.png](../../data/public_datasets/flow_matching/metfaces/images/10219-00.png) |
| metfaces_png_10242-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10242-00.png](../../data/public_datasets/flow_matching/metfaces/images/10242-00.png) |
| metfaces_png_10373-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10373-00.png](../../data/public_datasets/flow_matching/metfaces/images/10373-00.png) |
| metfaces_png_10426-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10426-00.png](../../data/public_datasets/flow_matching/metfaces/images/10426-00.png) |
| metfaces_png_10427-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10427-00.png](../../data/public_datasets/flow_matching/metfaces/images/10427-00.png) |
| metfaces_png_10466-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10466-00.png](../../data/public_datasets/flow_matching/metfaces/images/10466-00.png) |
| metfaces_png_10472-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10472-00.png](../../data/public_datasets/flow_matching/metfaces/images/10472-00.png) |
| metfaces_png_10474-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10474-00.png](../../data/public_datasets/flow_matching/metfaces/images/10474-00.png) |
| metfaces_png_10483-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10483-00.png](../../data/public_datasets/flow_matching/metfaces/images/10483-00.png) |
| metfaces_png_10484-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10484-00.png](../../data/public_datasets/flow_matching/metfaces/images/10484-00.png) |
| metfaces_png_10503-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10503-00.png](../../data/public_datasets/flow_matching/metfaces/images/10503-00.png) |
| metfaces_png_10504-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10504-00.png](../../data/public_datasets/flow_matching/metfaces/images/10504-00.png) |
| metfaces_png_10516-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10516-00.png](../../data/public_datasets/flow_matching/metfaces/images/10516-00.png) |
| metfaces_png_10522-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10522-00.png](../../data/public_datasets/flow_matching/metfaces/images/10522-00.png) |
| metfaces_png_10524-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10524-00.png](../../data/public_datasets/flow_matching/metfaces/images/10524-00.png) |
| metfaces_png_10525-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10525-00.png](../../data/public_datasets/flow_matching/metfaces/images/10525-00.png) |
| metfaces_png_10527-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10527-00.png](../../data/public_datasets/flow_matching/metfaces/images/10527-00.png) |
| metfaces_png_10528-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10528-00.png](../../data/public_datasets/flow_matching/metfaces/images/10528-00.png) |
| metfaces_png_10529-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10529-00.png](../../data/public_datasets/flow_matching/metfaces/images/10529-00.png) |
| metfaces_png_10533-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10533-00.png](../../data/public_datasets/flow_matching/metfaces/images/10533-00.png) |
| metfaces_png_10535-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10535-00.png](../../data/public_datasets/flow_matching/metfaces/images/10535-00.png) |
| metfaces_png_10537-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10537-00.png](../../data/public_datasets/flow_matching/metfaces/images/10537-00.png) |
| metfaces_png_10592-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10592-00.png](../../data/public_datasets/flow_matching/metfaces/images/10592-00.png) |
| metfaces_png_10596-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10596-00.png](../../data/public_datasets/flow_matching/metfaces/images/10596-00.png) |
| metfaces_png_10598-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10598-00.png](../../data/public_datasets/flow_matching/metfaces/images/10598-00.png) |
| metfaces_png_10599-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10599-00.png](../../data/public_datasets/flow_matching/metfaces/images/10599-00.png) |
| metfaces_png_10602-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10602-00.png](../../data/public_datasets/flow_matching/metfaces/images/10602-00.png) |
| metfaces_png_10603-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10603-00.png](../../data/public_datasets/flow_matching/metfaces/images/10603-00.png) |
| metfaces_png_10605-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10605-00.png](../../data/public_datasets/flow_matching/metfaces/images/10605-00.png) |
| metfaces_png_10751-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10751-00.png](../../data/public_datasets/flow_matching/metfaces/images/10751-00.png) |
| metfaces_png_10753-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10753-00.png](../../data/public_datasets/flow_matching/metfaces/images/10753-00.png) |
| metfaces_png_10754-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10754-00.png](../../data/public_datasets/flow_matching/metfaces/images/10754-00.png) |
| metfaces_png_10758-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10758-00.png](../../data/public_datasets/flow_matching/metfaces/images/10758-00.png) |
| metfaces_png_10759-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10759-00.png](../../data/public_datasets/flow_matching/metfaces/images/10759-00.png) |
| metfaces_png_10760-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10760-00.png](../../data/public_datasets/flow_matching/metfaces/images/10760-00.png) |
| metfaces_png_10761-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10761-00.png](../../data/public_datasets/flow_matching/metfaces/images/10761-00.png) |
| metfaces_png_10774-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10774-00.png](../../data/public_datasets/flow_matching/metfaces/images/10774-00.png) |
| metfaces_png_10775-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10775-00.png](../../data/public_datasets/flow_matching/metfaces/images/10775-00.png) |
| metfaces_png_10776-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10776-00.png](../../data/public_datasets/flow_matching/metfaces/images/10776-00.png) |
| metfaces_png_10782-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10782-00.png](../../data/public_datasets/flow_matching/metfaces/images/10782-00.png) |
| metfaces_png_10783-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10783-00.png](../../data/public_datasets/flow_matching/metfaces/images/10783-00.png) |
| metfaces_png_10784-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10784-00.png](../../data/public_datasets/flow_matching/metfaces/images/10784-00.png) |
| metfaces_png_10787-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10787-00.png](../../data/public_datasets/flow_matching/metfaces/images/10787-00.png) |
| metfaces_png_10791-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10791-00.png](../../data/public_datasets/flow_matching/metfaces/images/10791-00.png) |
| metfaces_png_10794-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10794-00.png](../../data/public_datasets/flow_matching/metfaces/images/10794-00.png) |
| metfaces_png_10795-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10795-00.png](../../data/public_datasets/flow_matching/metfaces/images/10795-00.png) |
| metfaces_png_10801-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10801-00.png](../../data/public_datasets/flow_matching/metfaces/images/10801-00.png) |
| metfaces_png_10834-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10834-00.png](../../data/public_datasets/flow_matching/metfaces/images/10834-00.png) |
| metfaces_png_10836-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10836-00.png](../../data/public_datasets/flow_matching/metfaces/images/10836-00.png) |
| metfaces_png_10837-02 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10837-02.png](../../data/public_datasets/flow_matching/metfaces/images/10837-02.png) |
| metfaces_png_10845-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10845-00.png](../../data/public_datasets/flow_matching/metfaces/images/10845-00.png) |
| metfaces_png_10849-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10849-00.png](../../data/public_datasets/flow_matching/metfaces/images/10849-00.png) |
| metfaces_png_10850-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10850-00.png](../../data/public_datasets/flow_matching/metfaces/images/10850-00.png) |
| metfaces_png_10852-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10852-00.png](../../data/public_datasets/flow_matching/metfaces/images/10852-00.png) |
| metfaces_png_10853-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10853-00.png](../../data/public_datasets/flow_matching/metfaces/images/10853-00.png) |
| metfaces_png_10859-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10859-00.png](../../data/public_datasets/flow_matching/metfaces/images/10859-00.png) |
| metfaces_png_10862-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10862-00.png](../../data/public_datasets/flow_matching/metfaces/images/10862-00.png) |
| metfaces_png_10863-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10863-00.png](../../data/public_datasets/flow_matching/metfaces/images/10863-00.png) |
| metfaces_png_10864-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10864-00.png](../../data/public_datasets/flow_matching/metfaces/images/10864-00.png) |
| metfaces_png_10866-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10866-00.png](../../data/public_datasets/flow_matching/metfaces/images/10866-00.png) |
| metfaces_png_10867-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10867-00.png](../../data/public_datasets/flow_matching/metfaces/images/10867-00.png) |
| metfaces_png_10880-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10880-00.png](../../data/public_datasets/flow_matching/metfaces/images/10880-00.png) |
| metfaces_png_10881-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10881-00.png](../../data/public_datasets/flow_matching/metfaces/images/10881-00.png) |
| metfaces_png_10882-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10882-00.png](../../data/public_datasets/flow_matching/metfaces/images/10882-00.png) |
| metfaces_png_10885-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10885-00.png](../../data/public_datasets/flow_matching/metfaces/images/10885-00.png) |
| metfaces_png_10886-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10886-00.png](../../data/public_datasets/flow_matching/metfaces/images/10886-00.png) |
| metfaces_png_10890-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10890-00.png](../../data/public_datasets/flow_matching/metfaces/images/10890-00.png) |
| metfaces_png_10896-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10896-00.png](../../data/public_datasets/flow_matching/metfaces/images/10896-00.png) |
| metfaces_png_10902-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10902-00.png](../../data/public_datasets/flow_matching/metfaces/images/10902-00.png) |
| metfaces_png_10905-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10905-00.png](../../data/public_datasets/flow_matching/metfaces/images/10905-00.png) |
| metfaces_png_10907-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10907-00.png](../../data/public_datasets/flow_matching/metfaces/images/10907-00.png) |
| metfaces_png_10923-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/10923-00.png](../../data/public_datasets/flow_matching/metfaces/images/10923-00.png) |
| metfaces_png_10927-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10927-00.png](../../data/public_datasets/flow_matching/metfaces/images/10927-00.png) |
| metfaces_png_10929-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10929-00.png](../../data/public_datasets/flow_matching/metfaces/images/10929-00.png) |
| metfaces_png_10956-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10956-00.png](../../data/public_datasets/flow_matching/metfaces/images/10956-00.png) |
| metfaces_png_10957-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10957-00.png](../../data/public_datasets/flow_matching/metfaces/images/10957-00.png) |
| metfaces_png_10958-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10958-00.png](../../data/public_datasets/flow_matching/metfaces/images/10958-00.png) |
| metfaces_png_10981-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10981-00.png](../../data/public_datasets/flow_matching/metfaces/images/10981-00.png) |
| metfaces_png_10986-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10986-00.png](../../data/public_datasets/flow_matching/metfaces/images/10986-00.png) |
| metfaces_png_10988-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10988-00.png](../../data/public_datasets/flow_matching/metfaces/images/10988-00.png) |
| metfaces_png_10989-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10989-00.png](../../data/public_datasets/flow_matching/metfaces/images/10989-00.png) |
| metfaces_png_10990-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10990-00.png](../../data/public_datasets/flow_matching/metfaces/images/10990-00.png) |
| metfaces_png_10991-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/10991-00.png](../../data/public_datasets/flow_matching/metfaces/images/10991-00.png) |
| metfaces_png_11015-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11015-00.png](../../data/public_datasets/flow_matching/metfaces/images/11015-00.png) |
| metfaces_png_11054-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11054-00.png](../../data/public_datasets/flow_matching/metfaces/images/11054-00.png) |
| metfaces_png_11057-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11057-00.png](../../data/public_datasets/flow_matching/metfaces/images/11057-00.png) |
| metfaces_png_11058-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11058-00.png](../../data/public_datasets/flow_matching/metfaces/images/11058-00.png) |
| metfaces_png_11079-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11079-00.png](../../data/public_datasets/flow_matching/metfaces/images/11079-00.png) |
| metfaces_png_11106-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11106-00.png](../../data/public_datasets/flow_matching/metfaces/images/11106-00.png) |
| metfaces_png_11166-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11166-00.png](../../data/public_datasets/flow_matching/metfaces/images/11166-00.png) |
| metfaces_png_11175-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11175-00.png](../../data/public_datasets/flow_matching/metfaces/images/11175-00.png) |
| metfaces_png_11187-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11187-00.png](../../data/public_datasets/flow_matching/metfaces/images/11187-00.png) |
| metfaces_png_11189-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11189-00.png](../../data/public_datasets/flow_matching/metfaces/images/11189-00.png) |
| metfaces_png_11191-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11191-00.png](../../data/public_datasets/flow_matching/metfaces/images/11191-00.png) |
| metfaces_png_11194-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11194-00.png](../../data/public_datasets/flow_matching/metfaces/images/11194-00.png) |
| metfaces_png_11195-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11195-00.png](../../data/public_datasets/flow_matching/metfaces/images/11195-00.png) |
| metfaces_png_11201-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11201-00.png](../../data/public_datasets/flow_matching/metfaces/images/11201-00.png) |
| metfaces_png_11206-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11206-00.png](../../data/public_datasets/flow_matching/metfaces/images/11206-00.png) |
| metfaces_png_11207-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11207-00.png](../../data/public_datasets/flow_matching/metfaces/images/11207-00.png) |
| metfaces_png_11208-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11208-00.png](../../data/public_datasets/flow_matching/metfaces/images/11208-00.png) |
| metfaces_png_11209-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11209-00.png](../../data/public_datasets/flow_matching/metfaces/images/11209-00.png) |
| metfaces_png_11212-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/11212-00.png](../../data/public_datasets/flow_matching/metfaces/images/11212-00.png) |
| metfaces_png_11213-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/11213-00.png](../../data/public_datasets/flow_matching/metfaces/images/11213-00.png) |
| metfaces_png_11214-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11214-00.png](../../data/public_datasets/flow_matching/metfaces/images/11214-00.png) |
| metfaces_png_11223-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11223-00.png](../../data/public_datasets/flow_matching/metfaces/images/11223-00.png) |
| metfaces_png_11240-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11240-00.png](../../data/public_datasets/flow_matching/metfaces/images/11240-00.png) |
| metfaces_png_11241-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11241-00.png](../../data/public_datasets/flow_matching/metfaces/images/11241-00.png) |
| metfaces_png_11243-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11243-00.png](../../data/public_datasets/flow_matching/metfaces/images/11243-00.png) |
| metfaces_png_11244-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/11244-00.png](../../data/public_datasets/flow_matching/metfaces/images/11244-00.png) |
| metfaces_png_11247-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11247-00.png](../../data/public_datasets/flow_matching/metfaces/images/11247-00.png) |
| metfaces_png_11249-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/11249-00.png](../../data/public_datasets/flow_matching/metfaces/images/11249-00.png) |
| metfaces_png_11250-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11250-00.png](../../data/public_datasets/flow_matching/metfaces/images/11250-00.png) |
| metfaces_png_11267-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/11267-00.png](../../data/public_datasets/flow_matching/metfaces/images/11267-00.png) |
| metfaces_png_11269-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11269-00.png](../../data/public_datasets/flow_matching/metfaces/images/11269-00.png) |
| metfaces_png_11269-01 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11269-01.png](../../data/public_datasets/flow_matching/metfaces/images/11269-01.png) |
| metfaces_png_11273-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11273-00.png](../../data/public_datasets/flow_matching/metfaces/images/11273-00.png) |
| metfaces_png_11274-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/11274-00.png](../../data/public_datasets/flow_matching/metfaces/images/11274-00.png) |
| metfaces_png_11274-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/11274-01.png](../../data/public_datasets/flow_matching/metfaces/images/11274-01.png) |
| metfaces_png_11284-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/11284-00.png](../../data/public_datasets/flow_matching/metfaces/images/11284-00.png) |
| metfaces_png_11347-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11347-00.png](../../data/public_datasets/flow_matching/metfaces/images/11347-00.png) |
| metfaces_png_11365-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11365-00.png](../../data/public_datasets/flow_matching/metfaces/images/11365-00.png) |
| metfaces_png_11366-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11366-00.png](../../data/public_datasets/flow_matching/metfaces/images/11366-00.png) |
| metfaces_png_11375-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11375-00.png](../../data/public_datasets/flow_matching/metfaces/images/11375-00.png) |
| metfaces_png_11406-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11406-00.png](../../data/public_datasets/flow_matching/metfaces/images/11406-00.png) |
| metfaces_png_11407-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11407-00.png](../../data/public_datasets/flow_matching/metfaces/images/11407-00.png) |
| metfaces_png_11408-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11408-00.png](../../data/public_datasets/flow_matching/metfaces/images/11408-00.png) |
| metfaces_png_11418-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11418-00.png](../../data/public_datasets/flow_matching/metfaces/images/11418-00.png) |
| metfaces_png_11433-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11433-00.png](../../data/public_datasets/flow_matching/metfaces/images/11433-00.png) |
| metfaces_png_11496-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11496-00.png](../../data/public_datasets/flow_matching/metfaces/images/11496-00.png) |
| metfaces_png_11504-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11504-00.png](../../data/public_datasets/flow_matching/metfaces/images/11504-00.png) |
| metfaces_png_11506-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11506-00.png](../../data/public_datasets/flow_matching/metfaces/images/11506-00.png) |
| metfaces_png_11507-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11507-00.png](../../data/public_datasets/flow_matching/metfaces/images/11507-00.png) |
| metfaces_png_11512-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11512-00.png](../../data/public_datasets/flow_matching/metfaces/images/11512-00.png) |
| metfaces_png_11587-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11587-00.png](../../data/public_datasets/flow_matching/metfaces/images/11587-00.png) |
| metfaces_png_11602-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11602-00.png](../../data/public_datasets/flow_matching/metfaces/images/11602-00.png) |
| metfaces_png_11614-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11614-00.png](../../data/public_datasets/flow_matching/metfaces/images/11614-00.png) |
| metfaces_png_11620-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11620-00.png](../../data/public_datasets/flow_matching/metfaces/images/11620-00.png) |
| metfaces_png_11621-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11621-00.png](../../data/public_datasets/flow_matching/metfaces/images/11621-00.png) |
| metfaces_png_11624-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11624-00.png](../../data/public_datasets/flow_matching/metfaces/images/11624-00.png) |
| metfaces_png_11626-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11626-00.png](../../data/public_datasets/flow_matching/metfaces/images/11626-00.png) |
| metfaces_png_11651-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11651-00.png](../../data/public_datasets/flow_matching/metfaces/images/11651-00.png) |
| metfaces_png_11674-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11674-00.png](../../data/public_datasets/flow_matching/metfaces/images/11674-00.png) |
| metfaces_png_11683-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11683-00.png](../../data/public_datasets/flow_matching/metfaces/images/11683-00.png) |
| metfaces_png_11699-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11699-00.png](../../data/public_datasets/flow_matching/metfaces/images/11699-00.png) |
| metfaces_png_11704-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11704-00.png](../../data/public_datasets/flow_matching/metfaces/images/11704-00.png) |
| metfaces_png_11705-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11705-00.png](../../data/public_datasets/flow_matching/metfaces/images/11705-00.png) |
| metfaces_png_11709-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11709-00.png](../../data/public_datasets/flow_matching/metfaces/images/11709-00.png) |
| metfaces_png_11712-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11712-00.png](../../data/public_datasets/flow_matching/metfaces/images/11712-00.png) |
| metfaces_png_11712-01 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11712-01.png](../../data/public_datasets/flow_matching/metfaces/images/11712-01.png) |
| metfaces_png_11713-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11713-00.png](../../data/public_datasets/flow_matching/metfaces/images/11713-00.png) |
| metfaces_png_11715-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11715-00.png](../../data/public_datasets/flow_matching/metfaces/images/11715-00.png) |
| metfaces_png_11716-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11716-00.png](../../data/public_datasets/flow_matching/metfaces/images/11716-00.png) |
| metfaces_png_11717-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11717-00.png](../../data/public_datasets/flow_matching/metfaces/images/11717-00.png) |
| metfaces_png_11718-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11718-00.png](../../data/public_datasets/flow_matching/metfaces/images/11718-00.png) |
| metfaces_png_11720-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11720-00.png](../../data/public_datasets/flow_matching/metfaces/images/11720-00.png) |
| metfaces_png_11721-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11721-00.png](../../data/public_datasets/flow_matching/metfaces/images/11721-00.png) |
| metfaces_png_11722-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11722-00.png](../../data/public_datasets/flow_matching/metfaces/images/11722-00.png) |
| metfaces_png_11724-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11724-00.png](../../data/public_datasets/flow_matching/metfaces/images/11724-00.png) |
| metfaces_png_11725-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11725-00.png](../../data/public_datasets/flow_matching/metfaces/images/11725-00.png) |
| metfaces_png_11728-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11728-00.png](../../data/public_datasets/flow_matching/metfaces/images/11728-00.png) |
| metfaces_png_11732-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11732-00.png](../../data/public_datasets/flow_matching/metfaces/images/11732-00.png) |
| metfaces_png_11733-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11733-00.png](../../data/public_datasets/flow_matching/metfaces/images/11733-00.png) |
| metfaces_png_11736-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11736-00.png](../../data/public_datasets/flow_matching/metfaces/images/11736-00.png) |
| metfaces_png_11739-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11739-00.png](../../data/public_datasets/flow_matching/metfaces/images/11739-00.png) |
| metfaces_png_11741-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11741-00.png](../../data/public_datasets/flow_matching/metfaces/images/11741-00.png) |
| metfaces_png_11742-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11742-00.png](../../data/public_datasets/flow_matching/metfaces/images/11742-00.png) |
| metfaces_png_11743-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11743-00.png](../../data/public_datasets/flow_matching/metfaces/images/11743-00.png) |
| metfaces_png_11746-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11746-00.png](../../data/public_datasets/flow_matching/metfaces/images/11746-00.png) |
| metfaces_png_11764-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11764-00.png](../../data/public_datasets/flow_matching/metfaces/images/11764-00.png) |
| metfaces_png_11764-01 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11764-01.png](../../data/public_datasets/flow_matching/metfaces/images/11764-01.png) |
| metfaces_png_11772-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11772-00.png](../../data/public_datasets/flow_matching/metfaces/images/11772-00.png) |
| metfaces_png_11778-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11778-00.png](../../data/public_datasets/flow_matching/metfaces/images/11778-00.png) |
| metfaces_png_11779-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11779-00.png](../../data/public_datasets/flow_matching/metfaces/images/11779-00.png) |
| metfaces_png_11781-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11781-00.png](../../data/public_datasets/flow_matching/metfaces/images/11781-00.png) |
| metfaces_png_11798-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11798-00.png](../../data/public_datasets/flow_matching/metfaces/images/11798-00.png) |
| metfaces_png_11798-01 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11798-01.png](../../data/public_datasets/flow_matching/metfaces/images/11798-01.png) |
| metfaces_png_11839-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11839-00.png](../../data/public_datasets/flow_matching/metfaces/images/11839-00.png) |
| metfaces_png_11840-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11840-00.png](../../data/public_datasets/flow_matching/metfaces/images/11840-00.png) |
| metfaces_png_11842-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11842-00.png](../../data/public_datasets/flow_matching/metfaces/images/11842-00.png) |
| metfaces_png_11843-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11843-00.png](../../data/public_datasets/flow_matching/metfaces/images/11843-00.png) |
| metfaces_png_11846-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11846-00.png](../../data/public_datasets/flow_matching/metfaces/images/11846-00.png) |
| metfaces_png_11929-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11929-00.png](../../data/public_datasets/flow_matching/metfaces/images/11929-00.png) |
| metfaces_png_11931-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11931-00.png](../../data/public_datasets/flow_matching/metfaces/images/11931-00.png) |
| metfaces_png_11932-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11932-00.png](../../data/public_datasets/flow_matching/metfaces/images/11932-00.png) |
| metfaces_png_11943-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11943-00.png](../../data/public_datasets/flow_matching/metfaces/images/11943-00.png) |
| metfaces_png_11945-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11945-00.png](../../data/public_datasets/flow_matching/metfaces/images/11945-00.png) |
| metfaces_png_11946-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11946-00.png](../../data/public_datasets/flow_matching/metfaces/images/11946-00.png) |
| metfaces_png_11972-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11972-00.png](../../data/public_datasets/flow_matching/metfaces/images/11972-00.png) |
| metfaces_png_11973-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11973-00.png](../../data/public_datasets/flow_matching/metfaces/images/11973-00.png) |
| metfaces_png_11984-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/11984-00.png](../../data/public_datasets/flow_matching/metfaces/images/11984-00.png) |
| metfaces_png_12532-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12532-00.png](../../data/public_datasets/flow_matching/metfaces/images/12532-00.png) |
| metfaces_png_12533-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12533-00.png](../../data/public_datasets/flow_matching/metfaces/images/12533-00.png) |
| metfaces_png_12545-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12545-00.png](../../data/public_datasets/flow_matching/metfaces/images/12545-00.png) |
| metfaces_png_12551-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12551-00.png](../../data/public_datasets/flow_matching/metfaces/images/12551-00.png) |
| metfaces_png_12564-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12564-00.png](../../data/public_datasets/flow_matching/metfaces/images/12564-00.png) |
| metfaces_png_12591-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12591-00.png](../../data/public_datasets/flow_matching/metfaces/images/12591-00.png) |
| metfaces_png_12594-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12594-00.png](../../data/public_datasets/flow_matching/metfaces/images/12594-00.png) |
| metfaces_png_12600-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12600-00.png](../../data/public_datasets/flow_matching/metfaces/images/12600-00.png) |
| metfaces_png_12602-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12602-00.png](../../data/public_datasets/flow_matching/metfaces/images/12602-00.png) |
| metfaces_png_12612-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12612-00.png](../../data/public_datasets/flow_matching/metfaces/images/12612-00.png) |
| metfaces_png_12613-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12613-00.png](../../data/public_datasets/flow_matching/metfaces/images/12613-00.png) |
| metfaces_png_12615-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12615-00.png](../../data/public_datasets/flow_matching/metfaces/images/12615-00.png) |
| metfaces_png_12631-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12631-00.png](../../data/public_datasets/flow_matching/metfaces/images/12631-00.png) |
| metfaces_png_12634-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12634-00.png](../../data/public_datasets/flow_matching/metfaces/images/12634-00.png) |
| metfaces_png_12641-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12641-00.png](../../data/public_datasets/flow_matching/metfaces/images/12641-00.png) |
| metfaces_png_12644-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12644-00.png](../../data/public_datasets/flow_matching/metfaces/images/12644-00.png) |
| metfaces_png_12644-01 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12644-01.png](../../data/public_datasets/flow_matching/metfaces/images/12644-01.png) |
| metfaces_png_12655-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12655-00.png](../../data/public_datasets/flow_matching/metfaces/images/12655-00.png) |
| metfaces_png_12656-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12656-00.png](../../data/public_datasets/flow_matching/metfaces/images/12656-00.png) |
| metfaces_png_12657-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12657-00.png](../../data/public_datasets/flow_matching/metfaces/images/12657-00.png) |
| metfaces_png_12658-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12658-00.png](../../data/public_datasets/flow_matching/metfaces/images/12658-00.png) |
| metfaces_png_12661-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12661-00.png](../../data/public_datasets/flow_matching/metfaces/images/12661-00.png) |
| metfaces_png_12662-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12662-00.png](../../data/public_datasets/flow_matching/metfaces/images/12662-00.png) |
| metfaces_png_12663-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12663-00.png](../../data/public_datasets/flow_matching/metfaces/images/12663-00.png) |
| metfaces_png_12664-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12664-00.png](../../data/public_datasets/flow_matching/metfaces/images/12664-00.png) |
| metfaces_png_12665-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12665-00.png](../../data/public_datasets/flow_matching/metfaces/images/12665-00.png) |
| metfaces_png_12666-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12666-00.png](../../data/public_datasets/flow_matching/metfaces/images/12666-00.png) |
| metfaces_png_12667-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12667-00.png](../../data/public_datasets/flow_matching/metfaces/images/12667-00.png) |
| metfaces_png_12668-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12668-00.png](../../data/public_datasets/flow_matching/metfaces/images/12668-00.png) |
| metfaces_png_12669-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12669-00.png](../../data/public_datasets/flow_matching/metfaces/images/12669-00.png) |
| metfaces_png_12670-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12670-00.png](../../data/public_datasets/flow_matching/metfaces/images/12670-00.png) |
| metfaces_png_12671-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12671-00.png](../../data/public_datasets/flow_matching/metfaces/images/12671-00.png) |
| metfaces_png_12673-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12673-00.png](../../data/public_datasets/flow_matching/metfaces/images/12673-00.png) |
| metfaces_png_12674-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12674-00.png](../../data/public_datasets/flow_matching/metfaces/images/12674-00.png) |
| metfaces_png_12675-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12675-00.png](../../data/public_datasets/flow_matching/metfaces/images/12675-00.png) |
| metfaces_png_12677-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12677-00.png](../../data/public_datasets/flow_matching/metfaces/images/12677-00.png) |
| metfaces_png_12680-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12680-00.png](../../data/public_datasets/flow_matching/metfaces/images/12680-00.png) |
| metfaces_png_12681-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12681-00.png](../../data/public_datasets/flow_matching/metfaces/images/12681-00.png) |
| metfaces_png_12683-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12683-00.png](../../data/public_datasets/flow_matching/metfaces/images/12683-00.png) |
| metfaces_png_12692-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12692-00.png](../../data/public_datasets/flow_matching/metfaces/images/12692-00.png) |
| metfaces_png_12693-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12693-00.png](../../data/public_datasets/flow_matching/metfaces/images/12693-00.png) |
| metfaces_png_12694-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12694-00.png](../../data/public_datasets/flow_matching/metfaces/images/12694-00.png) |
| metfaces_png_12695-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12695-00.png](../../data/public_datasets/flow_matching/metfaces/images/12695-00.png) |
| metfaces_png_12696-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12696-00.png](../../data/public_datasets/flow_matching/metfaces/images/12696-00.png) |
| metfaces_png_12698-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12698-00.png](../../data/public_datasets/flow_matching/metfaces/images/12698-00.png) |
| metfaces_png_12700-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12700-00.png](../../data/public_datasets/flow_matching/metfaces/images/12700-00.png) |
| metfaces_png_12701-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12701-00.png](../../data/public_datasets/flow_matching/metfaces/images/12701-00.png) |
| metfaces_png_12702-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12702-00.png](../../data/public_datasets/flow_matching/metfaces/images/12702-00.png) |
| metfaces_png_12705-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12705-00.png](../../data/public_datasets/flow_matching/metfaces/images/12705-00.png) |
| metfaces_png_12706-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12706-00.png](../../data/public_datasets/flow_matching/metfaces/images/12706-00.png) |
| metfaces_png_12707-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12707-00.png](../../data/public_datasets/flow_matching/metfaces/images/12707-00.png) |
| metfaces_png_12708-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12708-00.png](../../data/public_datasets/flow_matching/metfaces/images/12708-00.png) |
| metfaces_png_12791-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12791-00.png](../../data/public_datasets/flow_matching/metfaces/images/12791-00.png) |
| metfaces_png_12792-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12792-00.png](../../data/public_datasets/flow_matching/metfaces/images/12792-00.png) |
| metfaces_png_12793-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12793-00.png](../../data/public_datasets/flow_matching/metfaces/images/12793-00.png) |
| metfaces_png_12804-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12804-00.png](../../data/public_datasets/flow_matching/metfaces/images/12804-00.png) |
| metfaces_png_12812-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12812-00.png](../../data/public_datasets/flow_matching/metfaces/images/12812-00.png) |
| metfaces_png_12813-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12813-00.png](../../data/public_datasets/flow_matching/metfaces/images/12813-00.png) |
| metfaces_png_12814-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12814-00.png](../../data/public_datasets/flow_matching/metfaces/images/12814-00.png) |
| metfaces_png_12815-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12815-00.png](../../data/public_datasets/flow_matching/metfaces/images/12815-00.png) |
| metfaces_png_12818-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12818-00.png](../../data/public_datasets/flow_matching/metfaces/images/12818-00.png) |
| metfaces_png_12819-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12819-00.png](../../data/public_datasets/flow_matching/metfaces/images/12819-00.png) |
| metfaces_png_12824-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12824-00.png](../../data/public_datasets/flow_matching/metfaces/images/12824-00.png) |
| metfaces_png_12825-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12825-00.png](../../data/public_datasets/flow_matching/metfaces/images/12825-00.png) |
| metfaces_png_12827-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12827-00.png](../../data/public_datasets/flow_matching/metfaces/images/12827-00.png) |
| metfaces_png_12830-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12830-00.png](../../data/public_datasets/flow_matching/metfaces/images/12830-00.png) |
| metfaces_png_12882-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12882-00.png](../../data/public_datasets/flow_matching/metfaces/images/12882-00.png) |
| metfaces_png_12907-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12907-00.png](../../data/public_datasets/flow_matching/metfaces/images/12907-00.png) |
| metfaces_png_12921-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12921-00.png](../../data/public_datasets/flow_matching/metfaces/images/12921-00.png) |
| metfaces_png_12923-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12923-00.png](../../data/public_datasets/flow_matching/metfaces/images/12923-00.png) |
| metfaces_png_12932-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12932-00.png](../../data/public_datasets/flow_matching/metfaces/images/12932-00.png) |
| metfaces_png_12933-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12933-00.png](../../data/public_datasets/flow_matching/metfaces/images/12933-00.png) |
| metfaces_png_12940-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/12940-00.png](../../data/public_datasets/flow_matching/metfaces/images/12940-00.png) |
| metfaces_png_12948-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12948-00.png](../../data/public_datasets/flow_matching/metfaces/images/12948-00.png) |
| metfaces_png_12950-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12950-00.png](../../data/public_datasets/flow_matching/metfaces/images/12950-00.png) |
| metfaces_png_12975-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12975-00.png](../../data/public_datasets/flow_matching/metfaces/images/12975-00.png) |
| metfaces_png_12977-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/12977-00.png](../../data/public_datasets/flow_matching/metfaces/images/12977-00.png) |
| metfaces_png_12980-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/12980-00.png](../../data/public_datasets/flow_matching/metfaces/images/12980-00.png) |
| metfaces_png_12994-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/12994-00.png](../../data/public_datasets/flow_matching/metfaces/images/12994-00.png) |
| metfaces_png_12997-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/12997-00.png](../../data/public_datasets/flow_matching/metfaces/images/12997-00.png) |
| metfaces_png_13000-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13000-00.png](../../data/public_datasets/flow_matching/metfaces/images/13000-00.png) |
| metfaces_png_13027-00 | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/images/13027-00.png](../../data/public_datasets/flow_matching/metfaces/images/13027-00.png) |
| metfaces_png_13044-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13044-00.png](../../data/public_datasets/flow_matching/metfaces/images/13044-00.png) |
| metfaces_png_13045-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13045-00.png](../../data/public_datasets/flow_matching/metfaces/images/13045-00.png) |
| metfaces_png_13049-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13049-00.png](../../data/public_datasets/flow_matching/metfaces/images/13049-00.png) |
| metfaces_png_13051-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13051-00.png](../../data/public_datasets/flow_matching/metfaces/images/13051-00.png) |
| metfaces_png_13053-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13053-00.png](../../data/public_datasets/flow_matching/metfaces/images/13053-00.png) |
| metfaces_png_13057-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13057-00.png](../../data/public_datasets/flow_matching/metfaces/images/13057-00.png) |
| metfaces_png_13077-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13077-00.png](../../data/public_datasets/flow_matching/metfaces/images/13077-00.png) |
| metfaces_png_13093-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13093-00.png](../../data/public_datasets/flow_matching/metfaces/images/13093-00.png) |
| metfaces_png_13094-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13094-00.png](../../data/public_datasets/flow_matching/metfaces/images/13094-00.png) |
| metfaces_png_13096-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13096-00.png](../../data/public_datasets/flow_matching/metfaces/images/13096-00.png) |
| metfaces_png_13097-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13097-00.png](../../data/public_datasets/flow_matching/metfaces/images/13097-00.png) |
| metfaces_png_13098-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13098-00.png](../../data/public_datasets/flow_matching/metfaces/images/13098-00.png) |
| metfaces_png_13099-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13099-00.png](../../data/public_datasets/flow_matching/metfaces/images/13099-00.png) |
| metfaces_png_13100-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13100-00.png](../../data/public_datasets/flow_matching/metfaces/images/13100-00.png) |
| metfaces_png_13102-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13102-00.png](../../data/public_datasets/flow_matching/metfaces/images/13102-00.png) |
| metfaces_png_13104-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13104-00.png](../../data/public_datasets/flow_matching/metfaces/images/13104-00.png) |
| metfaces_png_13105-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13105-00.png](../../data/public_datasets/flow_matching/metfaces/images/13105-00.png) |
| metfaces_png_13187-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13187-00.png](../../data/public_datasets/flow_matching/metfaces/images/13187-00.png) |
| metfaces_png_13196-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13196-00.png](../../data/public_datasets/flow_matching/metfaces/images/13196-00.png) |
| metfaces_png_13204-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13204-00.png](../../data/public_datasets/flow_matching/metfaces/images/13204-00.png) |
| metfaces_png_13209-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13209-00.png](../../data/public_datasets/flow_matching/metfaces/images/13209-00.png) |
| metfaces_png_13320-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13320-00.png](../../data/public_datasets/flow_matching/metfaces/images/13320-00.png) |
| metfaces_png_13324-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13324-00.png](../../data/public_datasets/flow_matching/metfaces/images/13324-00.png) |
| metfaces_png_13325-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13325-00.png](../../data/public_datasets/flow_matching/metfaces/images/13325-00.png) |
| metfaces_png_13327-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13327-00.png](../../data/public_datasets/flow_matching/metfaces/images/13327-00.png) |
| metfaces_png_13334-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13334-00.png](../../data/public_datasets/flow_matching/metfaces/images/13334-00.png) |
| metfaces_png_13335-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13335-00.png](../../data/public_datasets/flow_matching/metfaces/images/13335-00.png) |
| metfaces_png_13336-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13336-00.png](../../data/public_datasets/flow_matching/metfaces/images/13336-00.png) |
| metfaces_png_13337-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13337-00.png](../../data/public_datasets/flow_matching/metfaces/images/13337-00.png) |
| metfaces_png_13338-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13338-00.png](../../data/public_datasets/flow_matching/metfaces/images/13338-00.png) |
| metfaces_png_13341-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13341-00.png](../../data/public_datasets/flow_matching/metfaces/images/13341-00.png) |
| metfaces_png_13342-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13342-00.png](../../data/public_datasets/flow_matching/metfaces/images/13342-00.png) |
| metfaces_png_13346-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13346-00.png](../../data/public_datasets/flow_matching/metfaces/images/13346-00.png) |
| metfaces_png_13385-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13385-00.png](../../data/public_datasets/flow_matching/metfaces/images/13385-00.png) |
| metfaces_png_13393-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13393-00.png](../../data/public_datasets/flow_matching/metfaces/images/13393-00.png) |
| metfaces_png_13395-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13395-00.png](../../data/public_datasets/flow_matching/metfaces/images/13395-00.png) |
| metfaces_png_13492-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13492-00.png](../../data/public_datasets/flow_matching/metfaces/images/13492-00.png) |
| metfaces_png_13494-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/13494-00.png](../../data/public_datasets/flow_matching/metfaces/images/13494-00.png) |
| metfaces_png_14305-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/14305-00.png](../../data/public_datasets/flow_matching/metfaces/images/14305-00.png) |
| metfaces_png_14310-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/14310-00.png](../../data/public_datasets/flow_matching/metfaces/images/14310-00.png) |
| metfaces_png_14314-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/14314-00.png](../../data/public_datasets/flow_matching/metfaces/images/14314-00.png) |
| metfaces_png_14351-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/14351-00.png](../../data/public_datasets/flow_matching/metfaces/images/14351-00.png) |
| metfaces_png_14356-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/14356-00.png](../../data/public_datasets/flow_matching/metfaces/images/14356-00.png) |
| metfaces_png_14357-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/14357-00.png](../../data/public_datasets/flow_matching/metfaces/images/14357-00.png) |
| metfaces_png_14399-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/14399-00.png](../../data/public_datasets/flow_matching/metfaces/images/14399-00.png) |
| metfaces_png_14400-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/14400-00.png](../../data/public_datasets/flow_matching/metfaces/images/14400-00.png) |
| metfaces_png_14402-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/14402-00.png](../../data/public_datasets/flow_matching/metfaces/images/14402-00.png) |
| metfaces_png_14404-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/14404-00.png](../../data/public_datasets/flow_matching/metfaces/images/14404-00.png) |
| metfaces_png_14419-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/14419-00.png](../../data/public_datasets/flow_matching/metfaces/images/14419-00.png) |
| metfaces_png_14483-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/14483-00.png](../../data/public_datasets/flow_matching/metfaces/images/14483-00.png) |
| metfaces_png_15025-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15025-00.png](../../data/public_datasets/flow_matching/metfaces/images/15025-00.png) |
| metfaces_png_15038-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15038-00.png](../../data/public_datasets/flow_matching/metfaces/images/15038-00.png) |
| metfaces_png_15044-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15044-00.png](../../data/public_datasets/flow_matching/metfaces/images/15044-00.png) |
| metfaces_png_15058-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15058-00.png](../../data/public_datasets/flow_matching/metfaces/images/15058-00.png) |
| metfaces_png_15059-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15059-00.png](../../data/public_datasets/flow_matching/metfaces/images/15059-00.png) |
| metfaces_png_15060-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15060-00.png](../../data/public_datasets/flow_matching/metfaces/images/15060-00.png) |
| metfaces_png_15076-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15076-00.png](../../data/public_datasets/flow_matching/metfaces/images/15076-00.png) |
| metfaces_png_15078-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15078-00.png](../../data/public_datasets/flow_matching/metfaces/images/15078-00.png) |
| metfaces_png_15082-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15082-00.png](../../data/public_datasets/flow_matching/metfaces/images/15082-00.png) |
| metfaces_png_15087-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15087-00.png](../../data/public_datasets/flow_matching/metfaces/images/15087-00.png) |
| metfaces_png_15090-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15090-00.png](../../data/public_datasets/flow_matching/metfaces/images/15090-00.png) |
| metfaces_png_15103-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15103-00.png](../../data/public_datasets/flow_matching/metfaces/images/15103-00.png) |
| metfaces_png_15121-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15121-00.png](../../data/public_datasets/flow_matching/metfaces/images/15121-00.png) |
| metfaces_png_15127-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15127-00.png](../../data/public_datasets/flow_matching/metfaces/images/15127-00.png) |
| metfaces_png_15128-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15128-00.png](../../data/public_datasets/flow_matching/metfaces/images/15128-00.png) |
| metfaces_png_15129-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15129-00.png](../../data/public_datasets/flow_matching/metfaces/images/15129-00.png) |
| metfaces_png_15134-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15134-00.png](../../data/public_datasets/flow_matching/metfaces/images/15134-00.png) |
| metfaces_png_15145-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15145-00.png](../../data/public_datasets/flow_matching/metfaces/images/15145-00.png) |
| metfaces_png_15146-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15146-00.png](../../data/public_datasets/flow_matching/metfaces/images/15146-00.png) |
| metfaces_png_15149-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15149-00.png](../../data/public_datasets/flow_matching/metfaces/images/15149-00.png) |
| metfaces_png_15155-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15155-00.png](../../data/public_datasets/flow_matching/metfaces/images/15155-00.png) |
| metfaces_png_15173-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15173-00.png](../../data/public_datasets/flow_matching/metfaces/images/15173-00.png) |
| metfaces_png_15181-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15181-00.png](../../data/public_datasets/flow_matching/metfaces/images/15181-00.png) |
| metfaces_png_15183-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15183-00.png](../../data/public_datasets/flow_matching/metfaces/images/15183-00.png) |
| metfaces_png_15191-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15191-00.png](../../data/public_datasets/flow_matching/metfaces/images/15191-00.png) |
| metfaces_png_15196-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15196-00.png](../../data/public_datasets/flow_matching/metfaces/images/15196-00.png) |
| metfaces_png_15197-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15197-00.png](../../data/public_datasets/flow_matching/metfaces/images/15197-00.png) |
| metfaces_png_15199-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15199-00.png](../../data/public_datasets/flow_matching/metfaces/images/15199-00.png) |
| metfaces_png_15200-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15200-00.png](../../data/public_datasets/flow_matching/metfaces/images/15200-00.png) |
| metfaces_png_15230-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15230-00.png](../../data/public_datasets/flow_matching/metfaces/images/15230-00.png) |
| metfaces_png_15233-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15233-00.png](../../data/public_datasets/flow_matching/metfaces/images/15233-00.png) |
| metfaces_png_15246-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15246-00.png](../../data/public_datasets/flow_matching/metfaces/images/15246-00.png) |
| metfaces_png_15307-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15307-00.png](../../data/public_datasets/flow_matching/metfaces/images/15307-00.png) |
| metfaces_png_15541-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15541-00.png](../../data/public_datasets/flow_matching/metfaces/images/15541-00.png) |
| metfaces_png_15586-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15586-00.png](../../data/public_datasets/flow_matching/metfaces/images/15586-00.png) |
| metfaces_png_15588-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/15588-00.png](../../data/public_datasets/flow_matching/metfaces/images/15588-00.png) |
| metfaces_png_16588-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/16588-00.png](../../data/public_datasets/flow_matching/metfaces/images/16588-00.png) |
| metfaces_png_16593-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/16593-00.png](../../data/public_datasets/flow_matching/metfaces/images/16593-00.png) |
| metfaces_png_16594-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/16594-00.png](../../data/public_datasets/flow_matching/metfaces/images/16594-00.png) |
| metfaces_png_16687-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/16687-00.png](../../data/public_datasets/flow_matching/metfaces/images/16687-00.png) |
| metfaces_png_16710-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/16710-00.png](../../data/public_datasets/flow_matching/metfaces/images/16710-00.png) |
| metfaces_png_16724-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/16724-00.png](../../data/public_datasets/flow_matching/metfaces/images/16724-00.png) |
| metfaces_png_16859-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/16859-00.png](../../data/public_datasets/flow_matching/metfaces/images/16859-00.png) |
| metfaces_png_16860-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/16860-00.png](../../data/public_datasets/flow_matching/metfaces/images/16860-00.png) |
| metfaces_png_16861-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/16861-00.png](../../data/public_datasets/flow_matching/metfaces/images/16861-00.png) |
| metfaces_png_16892-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/16892-00.png](../../data/public_datasets/flow_matching/metfaces/images/16892-00.png) |
| metfaces_png_16906-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/16906-00.png](../../data/public_datasets/flow_matching/metfaces/images/16906-00.png) |
| metfaces_png_16958-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/16958-00.png](../../data/public_datasets/flow_matching/metfaces/images/16958-00.png) |
| metfaces_png_16959-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/16959-00.png](../../data/public_datasets/flow_matching/metfaces/images/16959-00.png) |
| metfaces_png_17075-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/17075-00.png](../../data/public_datasets/flow_matching/metfaces/images/17075-00.png) |
| metfaces_png_17447-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/17447-00.png](../../data/public_datasets/flow_matching/metfaces/images/17447-00.png) |
| metfaces_png_17514-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/17514-00.png](../../data/public_datasets/flow_matching/metfaces/images/17514-00.png) |
| metfaces_png_17515-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/17515-00.png](../../data/public_datasets/flow_matching/metfaces/images/17515-00.png) |
| metfaces_png_17544-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/17544-00.png](../../data/public_datasets/flow_matching/metfaces/images/17544-00.png) |
| metfaces_png_17564-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/17564-00.png](../../data/public_datasets/flow_matching/metfaces/images/17564-00.png) |
| metfaces_png_17660-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/17660-00.png](../../data/public_datasets/flow_matching/metfaces/images/17660-00.png) |
| metfaces_png_17783-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/17783-00.png](../../data/public_datasets/flow_matching/metfaces/images/17783-00.png) |
| metfaces_png_17901-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/17901-00.png](../../data/public_datasets/flow_matching/metfaces/images/17901-00.png) |
| metfaces_png_18588-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/18588-00.png](../../data/public_datasets/flow_matching/metfaces/images/18588-00.png) |
| metfaces_png_187997-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/187997-00.png](../../data/public_datasets/flow_matching/metfaces/images/187997-00.png) |
| metfaces_png_18811-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/18811-00.png](../../data/public_datasets/flow_matching/metfaces/images/18811-00.png) |
| metfaces_png_18830-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/18830-00.png](../../data/public_datasets/flow_matching/metfaces/images/18830-00.png) |
| metfaces_png_189433-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/189433-00.png](../../data/public_datasets/flow_matching/metfaces/images/189433-00.png) |
| metfaces_png_189434-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/189434-00.png](../../data/public_datasets/flow_matching/metfaces/images/189434-00.png) |
| metfaces_png_19014-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/19014-00.png](../../data/public_datasets/flow_matching/metfaces/images/19014-00.png) |
| metfaces_png_19015-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/19015-00.png](../../data/public_datasets/flow_matching/metfaces/images/19015-00.png) |
| metfaces_png_19022-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/19022-00.png](../../data/public_datasets/flow_matching/metfaces/images/19022-00.png) |
| metfaces_png_19039-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/19039-00.png](../../data/public_datasets/flow_matching/metfaces/images/19039-00.png) |
| metfaces_png_19056-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/19056-00.png](../../data/public_datasets/flow_matching/metfaces/images/19056-00.png) |
| metfaces_png_190710-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/190710-00.png](../../data/public_datasets/flow_matching/metfaces/images/190710-00.png) |
| metfaces_png_190711-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/190711-00.png](../../data/public_datasets/flow_matching/metfaces/images/190711-00.png) |
| metfaces_png_19206-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/19206-00.png](../../data/public_datasets/flow_matching/metfaces/images/19206-00.png) |
| metfaces_png_192770-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/192770-00.png](../../data/public_datasets/flow_matching/metfaces/images/192770-00.png) |
| metfaces_png_19376-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/19376-00.png](../../data/public_datasets/flow_matching/metfaces/images/19376-00.png) |
| metfaces_png_193815-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/193815-00.png](../../data/public_datasets/flow_matching/metfaces/images/193815-00.png) |
| metfaces_png_194002-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/194002-00.png](../../data/public_datasets/flow_matching/metfaces/images/194002-00.png) |
| metfaces_png_194115-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/194115-00.png](../../data/public_datasets/flow_matching/metfaces/images/194115-00.png) |
| metfaces_png_19718-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/19718-00.png](../../data/public_datasets/flow_matching/metfaces/images/19718-00.png) |
| metfaces_png_19732-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/19732-00.png](../../data/public_datasets/flow_matching/metfaces/images/19732-00.png) |
| metfaces_png_19732-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/19732-01.png](../../data/public_datasets/flow_matching/metfaces/images/19732-01.png) |
| metfaces_png_19816-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/19816-00.png](../../data/public_datasets/flow_matching/metfaces/images/19816-00.png) |
| metfaces_png_19826-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/19826-00.png](../../data/public_datasets/flow_matching/metfaces/images/19826-00.png) |
| metfaces_png_198952-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/198952-00.png](../../data/public_datasets/flow_matching/metfaces/images/198952-00.png) |
| metfaces_png_199624-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/199624-00.png](../../data/public_datasets/flow_matching/metfaces/images/199624-00.png) |
| metfaces_png_199671-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/199671-00.png](../../data/public_datasets/flow_matching/metfaces/images/199671-00.png) |
| metfaces_png_199787-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/199787-00.png](../../data/public_datasets/flow_matching/metfaces/images/199787-00.png) |
| metfaces_png_200413-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/200413-00.png](../../data/public_datasets/flow_matching/metfaces/images/200413-00.png) |
| metfaces_png_200656-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/200656-00.png](../../data/public_datasets/flow_matching/metfaces/images/200656-00.png) |
| metfaces_png_200656-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/200656-01.png](../../data/public_datasets/flow_matching/metfaces/images/200656-01.png) |
| metfaces_png_200656-02 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/200656-02.png](../../data/public_datasets/flow_matching/metfaces/images/200656-02.png) |
| metfaces_png_201893-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/201893-00.png](../../data/public_datasets/flow_matching/metfaces/images/201893-00.png) |
| metfaces_png_202718-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/202718-00.png](../../data/public_datasets/flow_matching/metfaces/images/202718-00.png) |
| metfaces_png_203620-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/203620-00.png](../../data/public_datasets/flow_matching/metfaces/images/203620-00.png) |
| metfaces_png_203625-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/203625-00.png](../../data/public_datasets/flow_matching/metfaces/images/203625-00.png) |
| metfaces_png_20517-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/20517-00.png](../../data/public_datasets/flow_matching/metfaces/images/20517-00.png) |
| metfaces_png_205194-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/205194-00.png](../../data/public_datasets/flow_matching/metfaces/images/205194-00.png) |
| metfaces_png_205651-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/205651-00.png](../../data/public_datasets/flow_matching/metfaces/images/205651-00.png) |
| metfaces_png_206321-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/206321-00.png](../../data/public_datasets/flow_matching/metfaces/images/206321-00.png) |
| metfaces_png_206337-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/206337-00.png](../../data/public_datasets/flow_matching/metfaces/images/206337-00.png) |
| metfaces_png_206344-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/206344-00.png](../../data/public_datasets/flow_matching/metfaces/images/206344-00.png) |
| metfaces_png_206345-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/206345-00.png](../../data/public_datasets/flow_matching/metfaces/images/206345-00.png) |
| metfaces_png_206566-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/206566-00.png](../../data/public_datasets/flow_matching/metfaces/images/206566-00.png) |
| metfaces_png_207202-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/207202-00.png](../../data/public_datasets/flow_matching/metfaces/images/207202-00.png) |
| metfaces_png_207365-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/207365-00.png](../../data/public_datasets/flow_matching/metfaces/images/207365-00.png) |
| metfaces_png_207397-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/207397-00.png](../../data/public_datasets/flow_matching/metfaces/images/207397-00.png) |
| metfaces_png_207810-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/207810-00.png](../../data/public_datasets/flow_matching/metfaces/images/207810-00.png) |
| metfaces_png_207814-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/207814-00.png](../../data/public_datasets/flow_matching/metfaces/images/207814-00.png) |
| metfaces_png_208538-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/208538-00.png](../../data/public_datasets/flow_matching/metfaces/images/208538-00.png) |
| metfaces_png_209063-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/209063-00.png](../../data/public_datasets/flow_matching/metfaces/images/209063-00.png) |
| metfaces_png_209314-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/209314-00.png](../../data/public_datasets/flow_matching/metfaces/images/209314-00.png) |
| metfaces_png_209336-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/209336-00.png](../../data/public_datasets/flow_matching/metfaces/images/209336-00.png) |
| metfaces_png_210103-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/210103-00.png](../../data/public_datasets/flow_matching/metfaces/images/210103-00.png) |
| metfaces_png_21126-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/21126-00.png](../../data/public_datasets/flow_matching/metfaces/images/21126-00.png) |
| metfaces_png_211511-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/211511-00.png](../../data/public_datasets/flow_matching/metfaces/images/211511-00.png) |
| metfaces_png_21209-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/21209-00.png](../../data/public_datasets/flow_matching/metfaces/images/21209-00.png) |
| metfaces_png_21473-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/21473-00.png](../../data/public_datasets/flow_matching/metfaces/images/21473-00.png) |
| metfaces_png_226426-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/226426-00.png](../../data/public_datasets/flow_matching/metfaces/images/226426-00.png) |
| metfaces_png_231010-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/231010-00.png](../../data/public_datasets/flow_matching/metfaces/images/231010-00.png) |
| metfaces_png_231051-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/231051-00.png](../../data/public_datasets/flow_matching/metfaces/images/231051-00.png) |
| metfaces_png_231788-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/231788-00.png](../../data/public_datasets/flow_matching/metfaces/images/231788-00.png) |
| metfaces_png_232047-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/232047-00.png](../../data/public_datasets/flow_matching/metfaces/images/232047-00.png) |
| metfaces_png_236026-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/236026-00.png](../../data/public_datasets/flow_matching/metfaces/images/236026-00.png) |
| metfaces_png_237070-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/237070-00.png](../../data/public_datasets/flow_matching/metfaces/images/237070-00.png) |
| metfaces_png_242005-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/242005-00.png](../../data/public_datasets/flow_matching/metfaces/images/242005-00.png) |
| metfaces_png_242069-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/242069-00.png](../../data/public_datasets/flow_matching/metfaces/images/242069-00.png) |
| metfaces_png_242336-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/242336-00.png](../../data/public_datasets/flow_matching/metfaces/images/242336-00.png) |
| metfaces_png_246269-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/246269-00.png](../../data/public_datasets/flow_matching/metfaces/images/246269-00.png) |
| metfaces_png_247009-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/247009-01.png](../../data/public_datasets/flow_matching/metfaces/images/247009-01.png) |
| metfaces_png_247367-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/247367-00.png](../../data/public_datasets/flow_matching/metfaces/images/247367-00.png) |
| metfaces_png_247559-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/247559-00.png](../../data/public_datasets/flow_matching/metfaces/images/247559-00.png) |
| metfaces_png_248501-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/248501-00.png](../../data/public_datasets/flow_matching/metfaces/images/248501-00.png) |
| metfaces_png_248727-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/248727-00.png](../../data/public_datasets/flow_matching/metfaces/images/248727-00.png) |
| metfaces_png_248801-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/248801-00.png](../../data/public_datasets/flow_matching/metfaces/images/248801-00.png) |
| metfaces_png_251089-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/251089-00.png](../../data/public_datasets/flow_matching/metfaces/images/251089-00.png) |
| metfaces_png_252637-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/252637-00.png](../../data/public_datasets/flow_matching/metfaces/images/252637-00.png) |
| metfaces_png_254628-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/254628-01.png](../../data/public_datasets/flow_matching/metfaces/images/254628-01.png) |
| metfaces_png_255045-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/255045-00.png](../../data/public_datasets/flow_matching/metfaces/images/255045-00.png) |
| metfaces_png_256431-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/256431-00.png](../../data/public_datasets/flow_matching/metfaces/images/256431-00.png) |
| metfaces_png_257433-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/257433-00.png](../../data/public_datasets/flow_matching/metfaces/images/257433-00.png) |
| metfaces_png_283182-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/283182-00.png](../../data/public_datasets/flow_matching/metfaces/images/283182-00.png) |
| metfaces_png_285877-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/285877-00.png](../../data/public_datasets/flow_matching/metfaces/images/285877-00.png) |
| metfaces_png_29150-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/29150-00.png](../../data/public_datasets/flow_matching/metfaces/images/29150-00.png) |
| metfaces_png_312237-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/312237-00.png](../../data/public_datasets/flow_matching/metfaces/images/312237-00.png) |
| metfaces_png_322368-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/322368-00.png](../../data/public_datasets/flow_matching/metfaces/images/322368-00.png) |
| metfaces_png_322377-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/322377-00.png](../../data/public_datasets/flow_matching/metfaces/images/322377-00.png) |
| metfaces_png_33284-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/33284-00.png](../../data/public_datasets/flow_matching/metfaces/images/33284-00.png) |
| metfaces_png_334004-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/334004-00.png](../../data/public_datasets/flow_matching/metfaces/images/334004-00.png) |
| metfaces_png_334075-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/334075-00.png](../../data/public_datasets/flow_matching/metfaces/images/334075-00.png) |
| metfaces_png_334259-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/334259-00.png](../../data/public_datasets/flow_matching/metfaces/images/334259-00.png) |
| metfaces_png_334468-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/334468-00.png](../../data/public_datasets/flow_matching/metfaces/images/334468-00.png) |
| metfaces_png_334760-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/334760-00.png](../../data/public_datasets/flow_matching/metfaces/images/334760-00.png) |
| metfaces_png_334861-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/334861-00.png](../../data/public_datasets/flow_matching/metfaces/images/334861-00.png) |
| metfaces_png_335536-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/335536-00.png](../../data/public_datasets/flow_matching/metfaces/images/335536-00.png) |
| metfaces_png_336419-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/336419-00.png](../../data/public_datasets/flow_matching/metfaces/images/336419-00.png) |
| metfaces_png_336461-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/336461-00.png](../../data/public_datasets/flow_matching/metfaces/images/336461-00.png) |
| metfaces_png_336628-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/336628-00.png](../../data/public_datasets/flow_matching/metfaces/images/336628-00.png) |
| metfaces_png_336776-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/336776-00.png](../../data/public_datasets/flow_matching/metfaces/images/336776-00.png) |
| metfaces_png_337096-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/337096-00.png](../../data/public_datasets/flow_matching/metfaces/images/337096-00.png) |
| metfaces_png_337172-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/337172-00.png](../../data/public_datasets/flow_matching/metfaces/images/337172-00.png) |
| metfaces_png_337364-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/337364-00.png](../../data/public_datasets/flow_matching/metfaces/images/337364-00.png) |
| metfaces_png_337431-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/337431-00.png](../../data/public_datasets/flow_matching/metfaces/images/337431-00.png) |
| metfaces_png_337433-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/337433-00.png](../../data/public_datasets/flow_matching/metfaces/images/337433-00.png) |
| metfaces_png_337439-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/337439-00.png](../../data/public_datasets/flow_matching/metfaces/images/337439-00.png) |
| metfaces_png_337473-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/337473-00.png](../../data/public_datasets/flow_matching/metfaces/images/337473-00.png) |
| metfaces_png_337475-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/337475-00.png](../../data/public_datasets/flow_matching/metfaces/images/337475-00.png) |
| metfaces_png_337624-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/337624-00.png](../../data/public_datasets/flow_matching/metfaces/images/337624-00.png) |
| metfaces_png_337666-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/337666-00.png](../../data/public_datasets/flow_matching/metfaces/images/337666-00.png) |
| metfaces_png_338518-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/338518-00.png](../../data/public_datasets/flow_matching/metfaces/images/338518-00.png) |
| metfaces_png_338611-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/338611-00.png](../../data/public_datasets/flow_matching/metfaces/images/338611-00.png) |
| metfaces_png_338722-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/338722-00.png](../../data/public_datasets/flow_matching/metfaces/images/338722-00.png) |
| metfaces_png_34018-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/34018-00.png](../../data/public_datasets/flow_matching/metfaces/images/34018-00.png) |
| metfaces_png_340378-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/340378-00.png](../../data/public_datasets/flow_matching/metfaces/images/340378-00.png) |
| metfaces_png_340852-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/340852-00.png](../../data/public_datasets/flow_matching/metfaces/images/340852-00.png) |
| metfaces_png_340874-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/340874-00.png](../../data/public_datasets/flow_matching/metfaces/images/340874-00.png) |
| metfaces_png_341620-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/341620-00.png](../../data/public_datasets/flow_matching/metfaces/images/341620-00.png) |
| metfaces_png_343419-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/343419-00.png](../../data/public_datasets/flow_matching/metfaces/images/343419-00.png) |
| metfaces_png_35155-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/35155-00.png](../../data/public_datasets/flow_matching/metfaces/images/35155-00.png) |
| metfaces_png_35655-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/35655-00.png](../../data/public_datasets/flow_matching/metfaces/images/35655-00.png) |
| metfaces_png_35705-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/35705-00.png](../../data/public_datasets/flow_matching/metfaces/images/35705-00.png) |
| metfaces_png_36107-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/36107-00.png](../../data/public_datasets/flow_matching/metfaces/images/36107-00.png) |
| metfaces_png_362527-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/362527-00.png](../../data/public_datasets/flow_matching/metfaces/images/362527-00.png) |
| metfaces_png_365524-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/365524-00.png](../../data/public_datasets/flow_matching/metfaces/images/365524-00.png) |
| metfaces_png_365958-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/365958-00.png](../../data/public_datasets/flow_matching/metfaces/images/365958-00.png) |
| metfaces_png_366158-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/366158-00.png](../../data/public_datasets/flow_matching/metfaces/images/366158-00.png) |
| metfaces_png_366342-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/366342-00.png](../../data/public_datasets/flow_matching/metfaces/images/366342-00.png) |
| metfaces_png_366763-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/366763-00.png](../../data/public_datasets/flow_matching/metfaces/images/366763-00.png) |
| metfaces_png_366766-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/366766-00.png](../../data/public_datasets/flow_matching/metfaces/images/366766-00.png) |
| metfaces_png_366767-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/366767-00.png](../../data/public_datasets/flow_matching/metfaces/images/366767-00.png) |
| metfaces_png_373813-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/373813-00.png](../../data/public_datasets/flow_matching/metfaces/images/373813-00.png) |
| metfaces_png_376602-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/376602-00.png](../../data/public_datasets/flow_matching/metfaces/images/376602-00.png) |
| metfaces_png_377272-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/377272-00.png](../../data/public_datasets/flow_matching/metfaces/images/377272-00.png) |
| metfaces_png_384104-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/384104-00.png](../../data/public_datasets/flow_matching/metfaces/images/384104-00.png) |
| metfaces_png_384338-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/384338-00.png](../../data/public_datasets/flow_matching/metfaces/images/384338-00.png) |
| metfaces_png_38777-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/38777-00.png](../../data/public_datasets/flow_matching/metfaces/images/38777-00.png) |
| metfaces_png_387846-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/387846-00.png](../../data/public_datasets/flow_matching/metfaces/images/387846-00.png) |
| metfaces_png_390225-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/390225-00.png](../../data/public_datasets/flow_matching/metfaces/images/390225-00.png) |
| metfaces_png_390242-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/390242-00.png](../../data/public_datasets/flow_matching/metfaces/images/390242-00.png) |
| metfaces_png_39193-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/39193-00.png](../../data/public_datasets/flow_matching/metfaces/images/39193-00.png) |
| metfaces_png_392402-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/392402-00.png](../../data/public_datasets/flow_matching/metfaces/images/392402-00.png) |
| metfaces_png_394418-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/394418-00.png](../../data/public_datasets/flow_matching/metfaces/images/394418-00.png) |
| metfaces_png_394422-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/394422-00.png](../../data/public_datasets/flow_matching/metfaces/images/394422-00.png) |
| metfaces_png_394429-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/394429-00.png](../../data/public_datasets/flow_matching/metfaces/images/394429-00.png) |
| metfaces_png_394431-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/394431-00.png](../../data/public_datasets/flow_matching/metfaces/images/394431-00.png) |
| metfaces_png_394432-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/394432-00.png](../../data/public_datasets/flow_matching/metfaces/images/394432-00.png) |
| metfaces_png_394435-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/394435-00.png](../../data/public_datasets/flow_matching/metfaces/images/394435-00.png) |
| metfaces_png_394540-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/394540-00.png](../../data/public_datasets/flow_matching/metfaces/images/394540-00.png) |
| metfaces_png_394547-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/394547-00.png](../../data/public_datasets/flow_matching/metfaces/images/394547-00.png) |
| metfaces_png_396300-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/396300-00.png](../../data/public_datasets/flow_matching/metfaces/images/396300-00.png) |
| metfaces_png_3970-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/3970-00.png](../../data/public_datasets/flow_matching/metfaces/images/3970-00.png) |
| metfaces_png_3971-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/3971-00.png](../../data/public_datasets/flow_matching/metfaces/images/3971-00.png) |
| metfaces_png_399819-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/399819-00.png](../../data/public_datasets/flow_matching/metfaces/images/399819-00.png) |
| metfaces_png_402823-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/402823-00.png](../../data/public_datasets/flow_matching/metfaces/images/402823-00.png) |
| metfaces_png_404459-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/404459-00.png](../../data/public_datasets/flow_matching/metfaces/images/404459-00.png) |
| metfaces_png_404499-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/404499-00.png](../../data/public_datasets/flow_matching/metfaces/images/404499-00.png) |
| metfaces_png_405971-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/405971-00.png](../../data/public_datasets/flow_matching/metfaces/images/405971-00.png) |
| metfaces_png_408076-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/408076-00.png](../../data/public_datasets/flow_matching/metfaces/images/408076-00.png) |
| metfaces_png_409000-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/409000-00.png](../../data/public_datasets/flow_matching/metfaces/images/409000-00.png) |
| metfaces_png_414244-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/414244-00.png](../../data/public_datasets/flow_matching/metfaces/images/414244-00.png) |
| metfaces_png_415530-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/415530-00.png](../../data/public_datasets/flow_matching/metfaces/images/415530-00.png) |
| metfaces_png_416895-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/416895-00.png](../../data/public_datasets/flow_matching/metfaces/images/416895-00.png) |
| metfaces_png_416900-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/416900-00.png](../../data/public_datasets/flow_matching/metfaces/images/416900-00.png) |
| metfaces_png_416961-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/416961-00.png](../../data/public_datasets/flow_matching/metfaces/images/416961-00.png) |
| metfaces_png_418351-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/418351-00.png](../../data/public_datasets/flow_matching/metfaces/images/418351-00.png) |
| metfaces_png_418353-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/418353-00.png](../../data/public_datasets/flow_matching/metfaces/images/418353-00.png) |
| metfaces_png_421473-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/421473-00.png](../../data/public_datasets/flow_matching/metfaces/images/421473-00.png) |
| metfaces_png_421652-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/421652-00.png](../../data/public_datasets/flow_matching/metfaces/images/421652-00.png) |
| metfaces_png_42488-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/42488-00.png](../../data/public_datasets/flow_matching/metfaces/images/42488-00.png) |
| metfaces_png_42547-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/42547-00.png](../../data/public_datasets/flow_matching/metfaces/images/42547-00.png) |
| metfaces_png_428625-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/428625-00.png](../../data/public_datasets/flow_matching/metfaces/images/428625-00.png) |
| metfaces_png_428755-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/428755-00.png](../../data/public_datasets/flow_matching/metfaces/images/428755-00.png) |
| metfaces_png_435576-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435576-00.png](../../data/public_datasets/flow_matching/metfaces/images/435576-00.png) |
| metfaces_png_435580-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435580-00.png](../../data/public_datasets/flow_matching/metfaces/images/435580-00.png) |
| metfaces_png_435585-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435585-00.png](../../data/public_datasets/flow_matching/metfaces/images/435585-00.png) |
| metfaces_png_435585-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435585-01.png](../../data/public_datasets/flow_matching/metfaces/images/435585-01.png) |
| metfaces_png_435592-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435592-00.png](../../data/public_datasets/flow_matching/metfaces/images/435592-00.png) |
| metfaces_png_435596-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435596-00.png](../../data/public_datasets/flow_matching/metfaces/images/435596-00.png) |
| metfaces_png_435597-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435597-00.png](../../data/public_datasets/flow_matching/metfaces/images/435597-00.png) |
| metfaces_png_435615-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435615-00.png](../../data/public_datasets/flow_matching/metfaces/images/435615-00.png) |
| metfaces_png_435625-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435625-00.png](../../data/public_datasets/flow_matching/metfaces/images/435625-00.png) |
| metfaces_png_435630-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435630-00.png](../../data/public_datasets/flow_matching/metfaces/images/435630-00.png) |
| metfaces_png_435631-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435631-00.png](../../data/public_datasets/flow_matching/metfaces/images/435631-00.png) |
| metfaces_png_435632-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435632-00.png](../../data/public_datasets/flow_matching/metfaces/images/435632-00.png) |
| metfaces_png_435635-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435635-00.png](../../data/public_datasets/flow_matching/metfaces/images/435635-00.png) |
| metfaces_png_435641-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435641-01.png](../../data/public_datasets/flow_matching/metfaces/images/435641-01.png) |
| metfaces_png_435643-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435643-00.png](../../data/public_datasets/flow_matching/metfaces/images/435643-00.png) |
| metfaces_png_435645-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435645-00.png](../../data/public_datasets/flow_matching/metfaces/images/435645-00.png) |
| metfaces_png_435649-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435649-00.png](../../data/public_datasets/flow_matching/metfaces/images/435649-00.png) |
| metfaces_png_435650-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435650-00.png](../../data/public_datasets/flow_matching/metfaces/images/435650-00.png) |
| metfaces_png_435650-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435650-01.png](../../data/public_datasets/flow_matching/metfaces/images/435650-01.png) |
| metfaces_png_435658-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435658-00.png](../../data/public_datasets/flow_matching/metfaces/images/435658-00.png) |
| metfaces_png_435662-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435662-00.png](../../data/public_datasets/flow_matching/metfaces/images/435662-00.png) |
| metfaces_png_435664-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435664-00.png](../../data/public_datasets/flow_matching/metfaces/images/435664-00.png) |
| metfaces_png_435687-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435687-00.png](../../data/public_datasets/flow_matching/metfaces/images/435687-00.png) |
| metfaces_png_435688-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435688-00.png](../../data/public_datasets/flow_matching/metfaces/images/435688-00.png) |
| metfaces_png_435690-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435690-00.png](../../data/public_datasets/flow_matching/metfaces/images/435690-00.png) |
| metfaces_png_435697-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435697-00.png](../../data/public_datasets/flow_matching/metfaces/images/435697-00.png) |
| metfaces_png_435716-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435716-00.png](../../data/public_datasets/flow_matching/metfaces/images/435716-00.png) |
| metfaces_png_435716-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435716-01.png](../../data/public_datasets/flow_matching/metfaces/images/435716-01.png) |
| metfaces_png_435716-02 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435716-02.png](../../data/public_datasets/flow_matching/metfaces/images/435716-02.png) |
| metfaces_png_435721-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435721-00.png](../../data/public_datasets/flow_matching/metfaces/images/435721-00.png) |
| metfaces_png_435722-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435722-00.png](../../data/public_datasets/flow_matching/metfaces/images/435722-00.png) |
| metfaces_png_435739-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435739-00.png](../../data/public_datasets/flow_matching/metfaces/images/435739-00.png) |
| metfaces_png_435757-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435757-00.png](../../data/public_datasets/flow_matching/metfaces/images/435757-00.png) |
| metfaces_png_435758-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435758-00.png](../../data/public_datasets/flow_matching/metfaces/images/435758-00.png) |
| metfaces_png_435761-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435761-00.png](../../data/public_datasets/flow_matching/metfaces/images/435761-00.png) |
| metfaces_png_435763-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435763-00.png](../../data/public_datasets/flow_matching/metfaces/images/435763-00.png) |
| metfaces_png_435763-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435763-01.png](../../data/public_datasets/flow_matching/metfaces/images/435763-01.png) |
| metfaces_png_435765-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435765-00.png](../../data/public_datasets/flow_matching/metfaces/images/435765-00.png) |
| metfaces_png_435802-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435802-00.png](../../data/public_datasets/flow_matching/metfaces/images/435802-00.png) |
| metfaces_png_435803-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435803-00.png](../../data/public_datasets/flow_matching/metfaces/images/435803-00.png) |
| metfaces_png_435804-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435804-00.png](../../data/public_datasets/flow_matching/metfaces/images/435804-00.png) |
| metfaces_png_435807-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435807-00.png](../../data/public_datasets/flow_matching/metfaces/images/435807-00.png) |
| metfaces_png_435818-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435818-00.png](../../data/public_datasets/flow_matching/metfaces/images/435818-00.png) |
| metfaces_png_435819-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435819-00.png](../../data/public_datasets/flow_matching/metfaces/images/435819-00.png) |
| metfaces_png_435820-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435820-00.png](../../data/public_datasets/flow_matching/metfaces/images/435820-00.png) |
| metfaces_png_435829-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435829-00.png](../../data/public_datasets/flow_matching/metfaces/images/435829-00.png) |
| metfaces_png_435830-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435830-00.png](../../data/public_datasets/flow_matching/metfaces/images/435830-00.png) |
| metfaces_png_435834-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435834-00.png](../../data/public_datasets/flow_matching/metfaces/images/435834-00.png) |
| metfaces_png_435835-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435835-00.png](../../data/public_datasets/flow_matching/metfaces/images/435835-00.png) |
| metfaces_png_435837-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435837-00.png](../../data/public_datasets/flow_matching/metfaces/images/435837-00.png) |
| metfaces_png_435844-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435844-00.png](../../data/public_datasets/flow_matching/metfaces/images/435844-00.png) |
| metfaces_png_435844-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435844-01.png](../../data/public_datasets/flow_matching/metfaces/images/435844-01.png) |
| metfaces_png_435856-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435856-00.png](../../data/public_datasets/flow_matching/metfaces/images/435856-00.png) |
| metfaces_png_435864-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435864-00.png](../../data/public_datasets/flow_matching/metfaces/images/435864-00.png) |
| metfaces_png_435870-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435870-00.png](../../data/public_datasets/flow_matching/metfaces/images/435870-00.png) |
| metfaces_png_435875-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435875-00.png](../../data/public_datasets/flow_matching/metfaces/images/435875-00.png) |
| metfaces_png_435876-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435876-00.png](../../data/public_datasets/flow_matching/metfaces/images/435876-00.png) |
| metfaces_png_435886-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435886-00.png](../../data/public_datasets/flow_matching/metfaces/images/435886-00.png) |
| metfaces_png_435892-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435892-00.png](../../data/public_datasets/flow_matching/metfaces/images/435892-00.png) |
| metfaces_png_435895-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435895-00.png](../../data/public_datasets/flow_matching/metfaces/images/435895-00.png) |
| metfaces_png_435896-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435896-00.png](../../data/public_datasets/flow_matching/metfaces/images/435896-00.png) |
| metfaces_png_435897-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435897-00.png](../../data/public_datasets/flow_matching/metfaces/images/435897-00.png) |
| metfaces_png_435912-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435912-00.png](../../data/public_datasets/flow_matching/metfaces/images/435912-00.png) |
| metfaces_png_435921-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435921-00.png](../../data/public_datasets/flow_matching/metfaces/images/435921-00.png) |
| metfaces_png_435941-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435941-00.png](../../data/public_datasets/flow_matching/metfaces/images/435941-00.png) |
| metfaces_png_435944-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435944-00.png](../../data/public_datasets/flow_matching/metfaces/images/435944-00.png) |
| metfaces_png_435945-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435945-00.png](../../data/public_datasets/flow_matching/metfaces/images/435945-00.png) |
| metfaces_png_435946-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435946-00.png](../../data/public_datasets/flow_matching/metfaces/images/435946-00.png) |
| metfaces_png_435947-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435947-00.png](../../data/public_datasets/flow_matching/metfaces/images/435947-00.png) |
| metfaces_png_435948-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435948-00.png](../../data/public_datasets/flow_matching/metfaces/images/435948-00.png) |
| metfaces_png_435949-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435949-00.png](../../data/public_datasets/flow_matching/metfaces/images/435949-00.png) |
| metfaces_png_435950-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435950-00.png](../../data/public_datasets/flow_matching/metfaces/images/435950-00.png) |
| metfaces_png_435951-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435951-00.png](../../data/public_datasets/flow_matching/metfaces/images/435951-00.png) |
| metfaces_png_435952-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435952-00.png](../../data/public_datasets/flow_matching/metfaces/images/435952-00.png) |
| metfaces_png_435953-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435953-00.png](../../data/public_datasets/flow_matching/metfaces/images/435953-00.png) |
| metfaces_png_435954-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435954-00.png](../../data/public_datasets/flow_matching/metfaces/images/435954-00.png) |
| metfaces_png_435955-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435955-00.png](../../data/public_datasets/flow_matching/metfaces/images/435955-00.png) |
| metfaces_png_435959-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435959-00.png](../../data/public_datasets/flow_matching/metfaces/images/435959-00.png) |
| metfaces_png_435960-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435960-00.png](../../data/public_datasets/flow_matching/metfaces/images/435960-00.png) |
| metfaces_png_435961-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435961-00.png](../../data/public_datasets/flow_matching/metfaces/images/435961-00.png) |
| metfaces_png_435980-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435980-00.png](../../data/public_datasets/flow_matching/metfaces/images/435980-00.png) |
| metfaces_png_435984-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435984-00.png](../../data/public_datasets/flow_matching/metfaces/images/435984-00.png) |
| metfaces_png_435998-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/435998-00.png](../../data/public_datasets/flow_matching/metfaces/images/435998-00.png) |
| metfaces_png_436001-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436001-00.png](../../data/public_datasets/flow_matching/metfaces/images/436001-00.png) |
| metfaces_png_436017-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436017-00.png](../../data/public_datasets/flow_matching/metfaces/images/436017-00.png) |
| metfaces_png_436024-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436024-00.png](../../data/public_datasets/flow_matching/metfaces/images/436024-00.png) |
| metfaces_png_436028-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436028-00.png](../../data/public_datasets/flow_matching/metfaces/images/436028-00.png) |
| metfaces_png_436034-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436034-00.png](../../data/public_datasets/flow_matching/metfaces/images/436034-00.png) |
| metfaces_png_436036-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436036-00.png](../../data/public_datasets/flow_matching/metfaces/images/436036-00.png) |
| metfaces_png_436038-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436038-00.png](../../data/public_datasets/flow_matching/metfaces/images/436038-00.png) |
| metfaces_png_436042-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436042-00.png](../../data/public_datasets/flow_matching/metfaces/images/436042-00.png) |
| metfaces_png_436043-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436043-00.png](../../data/public_datasets/flow_matching/metfaces/images/436043-00.png) |
| metfaces_png_436044-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436044-00.png](../../data/public_datasets/flow_matching/metfaces/images/436044-00.png) |
| metfaces_png_436045-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436045-00.png](../../data/public_datasets/flow_matching/metfaces/images/436045-00.png) |
| metfaces_png_436046-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436046-00.png](../../data/public_datasets/flow_matching/metfaces/images/436046-00.png) |
| metfaces_png_436049-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436049-00.png](../../data/public_datasets/flow_matching/metfaces/images/436049-00.png) |
| metfaces_png_436056-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436056-00.png](../../data/public_datasets/flow_matching/metfaces/images/436056-00.png) |
| metfaces_png_436059-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436059-00.png](../../data/public_datasets/flow_matching/metfaces/images/436059-00.png) |
| metfaces_png_436067-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436067-00.png](../../data/public_datasets/flow_matching/metfaces/images/436067-00.png) |
| metfaces_png_436096-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436096-00.png](../../data/public_datasets/flow_matching/metfaces/images/436096-00.png) |
| metfaces_png_436097-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436097-00.png](../../data/public_datasets/flow_matching/metfaces/images/436097-00.png) |
| metfaces_png_436101-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436101-00.png](../../data/public_datasets/flow_matching/metfaces/images/436101-00.png) |
| metfaces_png_436103-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436103-00.png](../../data/public_datasets/flow_matching/metfaces/images/436103-00.png) |
| metfaces_png_436103-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436103-01.png](../../data/public_datasets/flow_matching/metfaces/images/436103-01.png) |
| metfaces_png_436107-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436107-00.png](../../data/public_datasets/flow_matching/metfaces/images/436107-00.png) |
| metfaces_png_436109-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436109-00.png](../../data/public_datasets/flow_matching/metfaces/images/436109-00.png) |
| metfaces_png_436124-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436124-00.png](../../data/public_datasets/flow_matching/metfaces/images/436124-00.png) |
| metfaces_png_436152-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436152-00.png](../../data/public_datasets/flow_matching/metfaces/images/436152-00.png) |
| metfaces_png_436179-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436179-00.png](../../data/public_datasets/flow_matching/metfaces/images/436179-00.png) |
| metfaces_png_436190-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436190-00.png](../../data/public_datasets/flow_matching/metfaces/images/436190-00.png) |
| metfaces_png_436210-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436210-00.png](../../data/public_datasets/flow_matching/metfaces/images/436210-00.png) |
| metfaces_png_436211-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436211-00.png](../../data/public_datasets/flow_matching/metfaces/images/436211-00.png) |
| metfaces_png_436212-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436212-00.png](../../data/public_datasets/flow_matching/metfaces/images/436212-00.png) |
| metfaces_png_436213-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436213-00.png](../../data/public_datasets/flow_matching/metfaces/images/436213-00.png) |
| metfaces_png_436214-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436214-00.png](../../data/public_datasets/flow_matching/metfaces/images/436214-00.png) |
| metfaces_png_436215-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436215-00.png](../../data/public_datasets/flow_matching/metfaces/images/436215-00.png) |
| metfaces_png_436216-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436216-00.png](../../data/public_datasets/flow_matching/metfaces/images/436216-00.png) |
| metfaces_png_436217-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436217-00.png](../../data/public_datasets/flow_matching/metfaces/images/436217-00.png) |
| metfaces_png_436219-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436219-00.png](../../data/public_datasets/flow_matching/metfaces/images/436219-00.png) |
| metfaces_png_436238-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436238-00.png](../../data/public_datasets/flow_matching/metfaces/images/436238-00.png) |
| metfaces_png_436239-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436239-00.png](../../data/public_datasets/flow_matching/metfaces/images/436239-00.png) |
| metfaces_png_436240-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436240-00.png](../../data/public_datasets/flow_matching/metfaces/images/436240-00.png) |
| metfaces_png_436246-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436246-00.png](../../data/public_datasets/flow_matching/metfaces/images/436246-00.png) |
| metfaces_png_436253-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436253-00.png](../../data/public_datasets/flow_matching/metfaces/images/436253-00.png) |
| metfaces_png_436254-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436254-00.png](../../data/public_datasets/flow_matching/metfaces/images/436254-00.png) |
| metfaces_png_436255-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436255-00.png](../../data/public_datasets/flow_matching/metfaces/images/436255-00.png) |
| metfaces_png_436258-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436258-00.png](../../data/public_datasets/flow_matching/metfaces/images/436258-00.png) |
| metfaces_png_436262-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436262-00.png](../../data/public_datasets/flow_matching/metfaces/images/436262-00.png) |
| metfaces_png_436265-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436265-00.png](../../data/public_datasets/flow_matching/metfaces/images/436265-00.png) |
| metfaces_png_436274-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436274-00.png](../../data/public_datasets/flow_matching/metfaces/images/436274-00.png) |
| metfaces_png_436284-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436284-00.png](../../data/public_datasets/flow_matching/metfaces/images/436284-00.png) |
| metfaces_png_436289-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436289-00.png](../../data/public_datasets/flow_matching/metfaces/images/436289-00.png) |
| metfaces_png_436295-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436295-00.png](../../data/public_datasets/flow_matching/metfaces/images/436295-00.png) |
| metfaces_png_436309-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436309-00.png](../../data/public_datasets/flow_matching/metfaces/images/436309-00.png) |
| metfaces_png_436310-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436310-00.png](../../data/public_datasets/flow_matching/metfaces/images/436310-00.png) |
| metfaces_png_436314-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436314-00.png](../../data/public_datasets/flow_matching/metfaces/images/436314-00.png) |
| metfaces_png_436315-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436315-00.png](../../data/public_datasets/flow_matching/metfaces/images/436315-00.png) |
| metfaces_png_436318-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436318-00.png](../../data/public_datasets/flow_matching/metfaces/images/436318-00.png) |
| metfaces_png_436322-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436322-00.png](../../data/public_datasets/flow_matching/metfaces/images/436322-00.png) |
| metfaces_png_436326-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436326-00.png](../../data/public_datasets/flow_matching/metfaces/images/436326-00.png) |
| metfaces_png_436333-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436333-00.png](../../data/public_datasets/flow_matching/metfaces/images/436333-00.png) |
| metfaces_png_436334-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436334-00.png](../../data/public_datasets/flow_matching/metfaces/images/436334-00.png) |
| metfaces_png_436334-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436334-01.png](../../data/public_datasets/flow_matching/metfaces/images/436334-01.png) |
| metfaces_png_436337-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436337-00.png](../../data/public_datasets/flow_matching/metfaces/images/436337-00.png) |
| metfaces_png_436338-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436338-00.png](../../data/public_datasets/flow_matching/metfaces/images/436338-00.png) |
| metfaces_png_436339-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436339-00.png](../../data/public_datasets/flow_matching/metfaces/images/436339-00.png) |
| metfaces_png_436341-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436341-00.png](../../data/public_datasets/flow_matching/metfaces/images/436341-00.png) |
| metfaces_png_436343-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436343-00.png](../../data/public_datasets/flow_matching/metfaces/images/436343-00.png) |
| metfaces_png_436377-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436377-00.png](../../data/public_datasets/flow_matching/metfaces/images/436377-00.png) |
| metfaces_png_436397-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436397-00.png](../../data/public_datasets/flow_matching/metfaces/images/436397-00.png) |
| metfaces_png_436398-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436398-00.png](../../data/public_datasets/flow_matching/metfaces/images/436398-00.png) |
| metfaces_png_436404-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436404-00.png](../../data/public_datasets/flow_matching/metfaces/images/436404-00.png) |
| metfaces_png_436407-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436407-00.png](../../data/public_datasets/flow_matching/metfaces/images/436407-00.png) |
| metfaces_png_436408-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436408-00.png](../../data/public_datasets/flow_matching/metfaces/images/436408-00.png) |
| metfaces_png_436421-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436421-00.png](../../data/public_datasets/flow_matching/metfaces/images/436421-00.png) |
| metfaces_png_436431-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436431-00.png](../../data/public_datasets/flow_matching/metfaces/images/436431-00.png) |
| metfaces_png_436432-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436432-00.png](../../data/public_datasets/flow_matching/metfaces/images/436432-00.png) |
| metfaces_png_436433-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436433-00.png](../../data/public_datasets/flow_matching/metfaces/images/436433-00.png) |
| metfaces_png_436434-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436434-00.png](../../data/public_datasets/flow_matching/metfaces/images/436434-00.png) |
| metfaces_png_436436-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436436-00.png](../../data/public_datasets/flow_matching/metfaces/images/436436-00.png) |
| metfaces_png_436442-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436442-00.png](../../data/public_datasets/flow_matching/metfaces/images/436442-00.png) |
| metfaces_png_436446-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436446-00.png](../../data/public_datasets/flow_matching/metfaces/images/436446-00.png) |
| metfaces_png_436480-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436480-00.png](../../data/public_datasets/flow_matching/metfaces/images/436480-00.png) |
| metfaces_png_436489-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436489-00.png](../../data/public_datasets/flow_matching/metfaces/images/436489-00.png) |
| metfaces_png_436491-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436491-00.png](../../data/public_datasets/flow_matching/metfaces/images/436491-00.png) |
| metfaces_png_436493-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436493-01.png](../../data/public_datasets/flow_matching/metfaces/images/436493-01.png) |
| metfaces_png_436501-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436501-00.png](../../data/public_datasets/flow_matching/metfaces/images/436501-00.png) |
| metfaces_png_436515-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436515-00.png](../../data/public_datasets/flow_matching/metfaces/images/436515-00.png) |
| metfaces_png_436520-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436520-00.png](../../data/public_datasets/flow_matching/metfaces/images/436520-00.png) |
| metfaces_png_436521-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436521-00.png](../../data/public_datasets/flow_matching/metfaces/images/436521-00.png) |
| metfaces_png_436538-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436538-00.png](../../data/public_datasets/flow_matching/metfaces/images/436538-00.png) |
| metfaces_png_436539-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436539-00.png](../../data/public_datasets/flow_matching/metfaces/images/436539-00.png) |
| metfaces_png_436539-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436539-01.png](../../data/public_datasets/flow_matching/metfaces/images/436539-01.png) |
| metfaces_png_436542-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436542-00.png](../../data/public_datasets/flow_matching/metfaces/images/436542-00.png) |
| metfaces_png_436543-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436543-00.png](../../data/public_datasets/flow_matching/metfaces/images/436543-00.png) |
| metfaces_png_436544-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436544-00.png](../../data/public_datasets/flow_matching/metfaces/images/436544-00.png) |
| metfaces_png_436545-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436545-00.png](../../data/public_datasets/flow_matching/metfaces/images/436545-00.png) |
| metfaces_png_436546-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436546-00.png](../../data/public_datasets/flow_matching/metfaces/images/436546-00.png) |
| metfaces_png_436547-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436547-00.png](../../data/public_datasets/flow_matching/metfaces/images/436547-00.png) |
| metfaces_png_436549-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436549-00.png](../../data/public_datasets/flow_matching/metfaces/images/436549-00.png) |
| metfaces_png_436550-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436550-01.png](../../data/public_datasets/flow_matching/metfaces/images/436550-01.png) |
| metfaces_png_436551-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436551-00.png](../../data/public_datasets/flow_matching/metfaces/images/436551-00.png) |
| metfaces_png_436552-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436552-00.png](../../data/public_datasets/flow_matching/metfaces/images/436552-00.png) |
| metfaces_png_436581-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436581-00.png](../../data/public_datasets/flow_matching/metfaces/images/436581-00.png) |
| metfaces_png_436582-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436582-00.png](../../data/public_datasets/flow_matching/metfaces/images/436582-00.png) |
| metfaces_png_436584-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436584-00.png](../../data/public_datasets/flow_matching/metfaces/images/436584-00.png) |
| metfaces_png_436586-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436586-00.png](../../data/public_datasets/flow_matching/metfaces/images/436586-00.png) |
| metfaces_png_436587-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436587-00.png](../../data/public_datasets/flow_matching/metfaces/images/436587-00.png) |
| metfaces_png_436591-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436591-00.png](../../data/public_datasets/flow_matching/metfaces/images/436591-00.png) |
| metfaces_png_436605-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436605-00.png](../../data/public_datasets/flow_matching/metfaces/images/436605-00.png) |
| metfaces_png_436610-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436610-00.png](../../data/public_datasets/flow_matching/metfaces/images/436610-00.png) |
| metfaces_png_436616-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436616-01.png](../../data/public_datasets/flow_matching/metfaces/images/436616-01.png) |
| metfaces_png_436617-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436617-00.png](../../data/public_datasets/flow_matching/metfaces/images/436617-00.png) |
| metfaces_png_436618-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436618-00.png](../../data/public_datasets/flow_matching/metfaces/images/436618-00.png) |
| metfaces_png_436621-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436621-00.png](../../data/public_datasets/flow_matching/metfaces/images/436621-00.png) |
| metfaces_png_436623-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436623-00.png](../../data/public_datasets/flow_matching/metfaces/images/436623-00.png) |
| metfaces_png_436624-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436624-00.png](../../data/public_datasets/flow_matching/metfaces/images/436624-00.png) |
| metfaces_png_436626-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436626-00.png](../../data/public_datasets/flow_matching/metfaces/images/436626-00.png) |
| metfaces_png_436627-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436627-00.png](../../data/public_datasets/flow_matching/metfaces/images/436627-00.png) |
| metfaces_png_436630-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436630-00.png](../../data/public_datasets/flow_matching/metfaces/images/436630-00.png) |
| metfaces_png_436631-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436631-00.png](../../data/public_datasets/flow_matching/metfaces/images/436631-00.png) |
| metfaces_png_436638-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436638-00.png](../../data/public_datasets/flow_matching/metfaces/images/436638-00.png) |
| metfaces_png_436640-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436640-00.png](../../data/public_datasets/flow_matching/metfaces/images/436640-00.png) |
| metfaces_png_436641-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436641-00.png](../../data/public_datasets/flow_matching/metfaces/images/436641-00.png) |
| metfaces_png_436642-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436642-00.png](../../data/public_datasets/flow_matching/metfaces/images/436642-00.png) |
| metfaces_png_436643-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436643-00.png](../../data/public_datasets/flow_matching/metfaces/images/436643-00.png) |
| metfaces_png_436649-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436649-00.png](../../data/public_datasets/flow_matching/metfaces/images/436649-00.png) |
| metfaces_png_436650-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436650-00.png](../../data/public_datasets/flow_matching/metfaces/images/436650-00.png) |
| metfaces_png_436657-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436657-00.png](../../data/public_datasets/flow_matching/metfaces/images/436657-00.png) |
| metfaces_png_436658-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436658-00.png](../../data/public_datasets/flow_matching/metfaces/images/436658-00.png) |
| metfaces_png_436659-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436659-00.png](../../data/public_datasets/flow_matching/metfaces/images/436659-00.png) |
| metfaces_png_436660-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436660-00.png](../../data/public_datasets/flow_matching/metfaces/images/436660-00.png) |
| metfaces_png_436661-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436661-00.png](../../data/public_datasets/flow_matching/metfaces/images/436661-00.png) |
| metfaces_png_436662-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436662-00.png](../../data/public_datasets/flow_matching/metfaces/images/436662-00.png) |
| metfaces_png_436663-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436663-00.png](../../data/public_datasets/flow_matching/metfaces/images/436663-00.png) |
| metfaces_png_436664-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436664-00.png](../../data/public_datasets/flow_matching/metfaces/images/436664-00.png) |
| metfaces_png_436666-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436666-00.png](../../data/public_datasets/flow_matching/metfaces/images/436666-00.png) |
| metfaces_png_436667-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436667-00.png](../../data/public_datasets/flow_matching/metfaces/images/436667-00.png) |
| metfaces_png_436668-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436668-00.png](../../data/public_datasets/flow_matching/metfaces/images/436668-00.png) |
| metfaces_png_436683-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436683-00.png](../../data/public_datasets/flow_matching/metfaces/images/436683-00.png) |
| metfaces_png_436684-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436684-00.png](../../data/public_datasets/flow_matching/metfaces/images/436684-00.png) |
| metfaces_png_436685-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436685-00.png](../../data/public_datasets/flow_matching/metfaces/images/436685-00.png) |
| metfaces_png_436685-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436685-01.png](../../data/public_datasets/flow_matching/metfaces/images/436685-01.png) |
| metfaces_png_436685-02 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436685-02.png](../../data/public_datasets/flow_matching/metfaces/images/436685-02.png) |
| metfaces_png_436686-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436686-00.png](../../data/public_datasets/flow_matching/metfaces/images/436686-00.png) |
| metfaces_png_436688-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436688-00.png](../../data/public_datasets/flow_matching/metfaces/images/436688-00.png) |
| metfaces_png_436691-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436691-00.png](../../data/public_datasets/flow_matching/metfaces/images/436691-00.png) |
| metfaces_png_436692-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436692-01.png](../../data/public_datasets/flow_matching/metfaces/images/436692-01.png) |
| metfaces_png_436694-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436694-00.png](../../data/public_datasets/flow_matching/metfaces/images/436694-00.png) |
| metfaces_png_436695-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436695-00.png](../../data/public_datasets/flow_matching/metfaces/images/436695-00.png) |
| metfaces_png_436696-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436696-00.png](../../data/public_datasets/flow_matching/metfaces/images/436696-00.png) |
| metfaces_png_436703-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436703-00.png](../../data/public_datasets/flow_matching/metfaces/images/436703-00.png) |
| metfaces_png_436705-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436705-00.png](../../data/public_datasets/flow_matching/metfaces/images/436705-00.png) |
| metfaces_png_436706-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436706-00.png](../../data/public_datasets/flow_matching/metfaces/images/436706-00.png) |
| metfaces_png_436709-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436709-00.png](../../data/public_datasets/flow_matching/metfaces/images/436709-00.png) |
| metfaces_png_436711-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436711-00.png](../../data/public_datasets/flow_matching/metfaces/images/436711-00.png) |
| metfaces_png_436714-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436714-00.png](../../data/public_datasets/flow_matching/metfaces/images/436714-00.png) |
| metfaces_png_436715-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436715-00.png](../../data/public_datasets/flow_matching/metfaces/images/436715-00.png) |
| metfaces_png_436717-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436717-00.png](../../data/public_datasets/flow_matching/metfaces/images/436717-00.png) |
| metfaces_png_436721-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436721-00.png](../../data/public_datasets/flow_matching/metfaces/images/436721-00.png) |
| metfaces_png_436722-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436722-00.png](../../data/public_datasets/flow_matching/metfaces/images/436722-00.png) |
| metfaces_png_436738-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436738-00.png](../../data/public_datasets/flow_matching/metfaces/images/436738-00.png) |
| metfaces_png_436738-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436738-01.png](../../data/public_datasets/flow_matching/metfaces/images/436738-01.png) |
| metfaces_png_436739-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436739-00.png](../../data/public_datasets/flow_matching/metfaces/images/436739-00.png) |
| metfaces_png_436741-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436741-00.png](../../data/public_datasets/flow_matching/metfaces/images/436741-00.png) |
| metfaces_png_436747-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436747-00.png](../../data/public_datasets/flow_matching/metfaces/images/436747-00.png) |
| metfaces_png_436769-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436769-00.png](../../data/public_datasets/flow_matching/metfaces/images/436769-00.png) |
| metfaces_png_436771-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436771-00.png](../../data/public_datasets/flow_matching/metfaces/images/436771-00.png) |
| metfaces_png_436789-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436789-00.png](../../data/public_datasets/flow_matching/metfaces/images/436789-00.png) |
| metfaces_png_436790-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436790-00.png](../../data/public_datasets/flow_matching/metfaces/images/436790-00.png) |
| metfaces_png_436796-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436796-00.png](../../data/public_datasets/flow_matching/metfaces/images/436796-00.png) |
| metfaces_png_436797-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436797-00.png](../../data/public_datasets/flow_matching/metfaces/images/436797-00.png) |
| metfaces_png_436797-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436797-01.png](../../data/public_datasets/flow_matching/metfaces/images/436797-01.png) |
| metfaces_png_436798-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436798-00.png](../../data/public_datasets/flow_matching/metfaces/images/436798-00.png) |
| metfaces_png_436819-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436819-00.png](../../data/public_datasets/flow_matching/metfaces/images/436819-00.png) |
| metfaces_png_436821-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436821-00.png](../../data/public_datasets/flow_matching/metfaces/images/436821-00.png) |
| metfaces_png_436823-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436823-00.png](../../data/public_datasets/flow_matching/metfaces/images/436823-00.png) |
| metfaces_png_436824-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436824-00.png](../../data/public_datasets/flow_matching/metfaces/images/436824-00.png) |
| metfaces_png_436825-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436825-00.png](../../data/public_datasets/flow_matching/metfaces/images/436825-00.png) |
| metfaces_png_436834-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436834-00.png](../../data/public_datasets/flow_matching/metfaces/images/436834-00.png) |
| metfaces_png_436838-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436838-00.png](../../data/public_datasets/flow_matching/metfaces/images/436838-00.png) |
| metfaces_png_436838-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436838-01.png](../../data/public_datasets/flow_matching/metfaces/images/436838-01.png) |
| metfaces_png_436846-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436846-00.png](../../data/public_datasets/flow_matching/metfaces/images/436846-00.png) |
| metfaces_png_436846-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436846-01.png](../../data/public_datasets/flow_matching/metfaces/images/436846-01.png) |
| metfaces_png_436848-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436848-00.png](../../data/public_datasets/flow_matching/metfaces/images/436848-00.png) |
| metfaces_png_436850-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436850-00.png](../../data/public_datasets/flow_matching/metfaces/images/436850-00.png) |
| metfaces_png_436852-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436852-00.png](../../data/public_datasets/flow_matching/metfaces/images/436852-00.png) |
| metfaces_png_436853-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436853-00.png](../../data/public_datasets/flow_matching/metfaces/images/436853-00.png) |
| metfaces_png_436855-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436855-00.png](../../data/public_datasets/flow_matching/metfaces/images/436855-00.png) |
| metfaces_png_436862-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436862-00.png](../../data/public_datasets/flow_matching/metfaces/images/436862-00.png) |
| metfaces_png_436871-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436871-00.png](../../data/public_datasets/flow_matching/metfaces/images/436871-00.png) |
| metfaces_png_436872-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436872-00.png](../../data/public_datasets/flow_matching/metfaces/images/436872-00.png) |
| metfaces_png_436873-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436873-00.png](../../data/public_datasets/flow_matching/metfaces/images/436873-00.png) |
| metfaces_png_436873-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436873-01.png](../../data/public_datasets/flow_matching/metfaces/images/436873-01.png) |
| metfaces_png_436887-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436887-00.png](../../data/public_datasets/flow_matching/metfaces/images/436887-00.png) |
| metfaces_png_436895-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436895-00.png](../../data/public_datasets/flow_matching/metfaces/images/436895-00.png) |
| metfaces_png_436910-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436910-00.png](../../data/public_datasets/flow_matching/metfaces/images/436910-00.png) |
| metfaces_png_436911-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436911-00.png](../../data/public_datasets/flow_matching/metfaces/images/436911-00.png) |
| metfaces_png_436928-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436928-00.png](../../data/public_datasets/flow_matching/metfaces/images/436928-00.png) |
| metfaces_png_436928-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436928-01.png](../../data/public_datasets/flow_matching/metfaces/images/436928-01.png) |
| metfaces_png_436930-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436930-00.png](../../data/public_datasets/flow_matching/metfaces/images/436930-00.png) |
| metfaces_png_436933-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436933-00.png](../../data/public_datasets/flow_matching/metfaces/images/436933-00.png) |
| metfaces_png_436941-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436941-00.png](../../data/public_datasets/flow_matching/metfaces/images/436941-00.png) |
| metfaces_png_436944-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436944-00.png](../../data/public_datasets/flow_matching/metfaces/images/436944-00.png) |
| metfaces_png_436948-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436948-00.png](../../data/public_datasets/flow_matching/metfaces/images/436948-00.png) |
| metfaces_png_436955-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436955-00.png](../../data/public_datasets/flow_matching/metfaces/images/436955-00.png) |
| metfaces_png_436956-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436956-00.png](../../data/public_datasets/flow_matching/metfaces/images/436956-00.png) |
| metfaces_png_436984-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436984-01.png](../../data/public_datasets/flow_matching/metfaces/images/436984-01.png) |
| metfaces_png_436984-02 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436984-02.png](../../data/public_datasets/flow_matching/metfaces/images/436984-02.png) |
| metfaces_png_436986-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436986-00.png](../../data/public_datasets/flow_matching/metfaces/images/436986-00.png) |
| metfaces_png_436987-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436987-00.png](../../data/public_datasets/flow_matching/metfaces/images/436987-00.png) |
| metfaces_png_436990-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/436990-00.png](../../data/public_datasets/flow_matching/metfaces/images/436990-00.png) |
| metfaces_png_437004-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437004-00.png](../../data/public_datasets/flow_matching/metfaces/images/437004-00.png) |
| metfaces_png_437021-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437021-00.png](../../data/public_datasets/flow_matching/metfaces/images/437021-00.png) |
| metfaces_png_437030-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437030-00.png](../../data/public_datasets/flow_matching/metfaces/images/437030-00.png) |
| metfaces_png_437038-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437038-01.png](../../data/public_datasets/flow_matching/metfaces/images/437038-01.png) |
| metfaces_png_437038-02 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437038-02.png](../../data/public_datasets/flow_matching/metfaces/images/437038-02.png) |
| metfaces_png_437046-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437046-00.png](../../data/public_datasets/flow_matching/metfaces/images/437046-00.png) |
| metfaces_png_437048-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437048-00.png](../../data/public_datasets/flow_matching/metfaces/images/437048-00.png) |
| metfaces_png_437056-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437056-01.png](../../data/public_datasets/flow_matching/metfaces/images/437056-01.png) |
| metfaces_png_437061-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437061-00.png](../../data/public_datasets/flow_matching/metfaces/images/437061-00.png) |
| metfaces_png_437062-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437062-00.png](../../data/public_datasets/flow_matching/metfaces/images/437062-00.png) |
| metfaces_png_437062-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437062-01.png](../../data/public_datasets/flow_matching/metfaces/images/437062-01.png) |
| metfaces_png_437065-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437065-00.png](../../data/public_datasets/flow_matching/metfaces/images/437065-00.png) |
| metfaces_png_437067-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437067-00.png](../../data/public_datasets/flow_matching/metfaces/images/437067-00.png) |
| metfaces_png_437073-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437073-00.png](../../data/public_datasets/flow_matching/metfaces/images/437073-00.png) |
| metfaces_png_437085-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437085-00.png](../../data/public_datasets/flow_matching/metfaces/images/437085-00.png) |
| metfaces_png_437086-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437086-00.png](../../data/public_datasets/flow_matching/metfaces/images/437086-00.png) |
| metfaces_png_437087-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437087-00.png](../../data/public_datasets/flow_matching/metfaces/images/437087-00.png) |
| metfaces_png_437089-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437089-00.png](../../data/public_datasets/flow_matching/metfaces/images/437089-00.png) |
| metfaces_png_437090-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437090-00.png](../../data/public_datasets/flow_matching/metfaces/images/437090-00.png) |
| metfaces_png_437145-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437145-00.png](../../data/public_datasets/flow_matching/metfaces/images/437145-00.png) |
| metfaces_png_437151-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437151-00.png](../../data/public_datasets/flow_matching/metfaces/images/437151-00.png) |
| metfaces_png_437154-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437154-00.png](../../data/public_datasets/flow_matching/metfaces/images/437154-00.png) |
| metfaces_png_437158-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437158-00.png](../../data/public_datasets/flow_matching/metfaces/images/437158-00.png) |
| metfaces_png_437163-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437163-00.png](../../data/public_datasets/flow_matching/metfaces/images/437163-00.png) |
| metfaces_png_437164-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437164-00.png](../../data/public_datasets/flow_matching/metfaces/images/437164-00.png) |
| metfaces_png_437166-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437166-00.png](../../data/public_datasets/flow_matching/metfaces/images/437166-00.png) |
| metfaces_png_437167-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437167-00.png](../../data/public_datasets/flow_matching/metfaces/images/437167-00.png) |
| metfaces_png_437168-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437168-00.png](../../data/public_datasets/flow_matching/metfaces/images/437168-00.png) |
| metfaces_png_437173-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437173-00.png](../../data/public_datasets/flow_matching/metfaces/images/437173-00.png) |
| metfaces_png_437174-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437174-00.png](../../data/public_datasets/flow_matching/metfaces/images/437174-00.png) |
| metfaces_png_437181-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437181-00.png](../../data/public_datasets/flow_matching/metfaces/images/437181-00.png) |
| metfaces_png_437181-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437181-01.png](../../data/public_datasets/flow_matching/metfaces/images/437181-01.png) |
| metfaces_png_437182-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437182-00.png](../../data/public_datasets/flow_matching/metfaces/images/437182-00.png) |
| metfaces_png_437184-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437184-00.png](../../data/public_datasets/flow_matching/metfaces/images/437184-00.png) |
| metfaces_png_437185-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437185-00.png](../../data/public_datasets/flow_matching/metfaces/images/437185-00.png) |
| metfaces_png_437199-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437199-00.png](../../data/public_datasets/flow_matching/metfaces/images/437199-00.png) |
| metfaces_png_437199-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437199-01.png](../../data/public_datasets/flow_matching/metfaces/images/437199-01.png) |
| metfaces_png_437199-02 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437199-02.png](../../data/public_datasets/flow_matching/metfaces/images/437199-02.png) |
| metfaces_png_437204-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437204-00.png](../../data/public_datasets/flow_matching/metfaces/images/437204-00.png) |
| metfaces_png_437205-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437205-00.png](../../data/public_datasets/flow_matching/metfaces/images/437205-00.png) |
| metfaces_png_437206-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437206-00.png](../../data/public_datasets/flow_matching/metfaces/images/437206-00.png) |
| metfaces_png_437207-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437207-00.png](../../data/public_datasets/flow_matching/metfaces/images/437207-00.png) |
| metfaces_png_437208-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437208-00.png](../../data/public_datasets/flow_matching/metfaces/images/437208-00.png) |
| metfaces_png_437209-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437209-00.png](../../data/public_datasets/flow_matching/metfaces/images/437209-00.png) |
| metfaces_png_437210-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437210-00.png](../../data/public_datasets/flow_matching/metfaces/images/437210-00.png) |
| metfaces_png_437217-05 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437217-05.png](../../data/public_datasets/flow_matching/metfaces/images/437217-05.png) |
| metfaces_png_437232-03 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437232-03.png](../../data/public_datasets/flow_matching/metfaces/images/437232-03.png) |
| metfaces_png_437237-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437237-00.png](../../data/public_datasets/flow_matching/metfaces/images/437237-00.png) |
| metfaces_png_437239-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437239-01.png](../../data/public_datasets/flow_matching/metfaces/images/437239-01.png) |
| metfaces_png_437248-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437248-00.png](../../data/public_datasets/flow_matching/metfaces/images/437248-00.png) |
| metfaces_png_437255-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437255-00.png](../../data/public_datasets/flow_matching/metfaces/images/437255-00.png) |
| metfaces_png_437263-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437263-00.png](../../data/public_datasets/flow_matching/metfaces/images/437263-00.png) |
| metfaces_png_437264-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437264-00.png](../../data/public_datasets/flow_matching/metfaces/images/437264-00.png) |
| metfaces_png_437281-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437281-00.png](../../data/public_datasets/flow_matching/metfaces/images/437281-00.png) |
| metfaces_png_437282-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437282-00.png](../../data/public_datasets/flow_matching/metfaces/images/437282-00.png) |
| metfaces_png_437324-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437324-00.png](../../data/public_datasets/flow_matching/metfaces/images/437324-00.png) |
| metfaces_png_437325-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437325-00.png](../../data/public_datasets/flow_matching/metfaces/images/437325-00.png) |
| metfaces_png_437332-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437332-00.png](../../data/public_datasets/flow_matching/metfaces/images/437332-00.png) |
| metfaces_png_437356-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437356-00.png](../../data/public_datasets/flow_matching/metfaces/images/437356-00.png) |
| metfaces_png_437357-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437357-00.png](../../data/public_datasets/flow_matching/metfaces/images/437357-00.png) |
| metfaces_png_437358-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437358-00.png](../../data/public_datasets/flow_matching/metfaces/images/437358-00.png) |
| metfaces_png_437359-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437359-00.png](../../data/public_datasets/flow_matching/metfaces/images/437359-00.png) |
| metfaces_png_437360-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437360-00.png](../../data/public_datasets/flow_matching/metfaces/images/437360-00.png) |
| metfaces_png_437361-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437361-00.png](../../data/public_datasets/flow_matching/metfaces/images/437361-00.png) |
| metfaces_png_437363-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437363-00.png](../../data/public_datasets/flow_matching/metfaces/images/437363-00.png) |
| metfaces_png_437364-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437364-00.png](../../data/public_datasets/flow_matching/metfaces/images/437364-00.png) |
| metfaces_png_437365-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437365-00.png](../../data/public_datasets/flow_matching/metfaces/images/437365-00.png) |
| metfaces_png_437368-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437368-01.png](../../data/public_datasets/flow_matching/metfaces/images/437368-01.png) |
| metfaces_png_437373-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437373-00.png](../../data/public_datasets/flow_matching/metfaces/images/437373-00.png) |
| metfaces_png_437374-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437374-00.png](../../data/public_datasets/flow_matching/metfaces/images/437374-00.png) |
| metfaces_png_437385-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437385-00.png](../../data/public_datasets/flow_matching/metfaces/images/437385-00.png) |
| metfaces_png_437386-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437386-00.png](../../data/public_datasets/flow_matching/metfaces/images/437386-00.png) |
| metfaces_png_437387-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437387-00.png](../../data/public_datasets/flow_matching/metfaces/images/437387-00.png) |
| metfaces_png_437388-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437388-00.png](../../data/public_datasets/flow_matching/metfaces/images/437388-00.png) |
| metfaces_png_437389-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437389-00.png](../../data/public_datasets/flow_matching/metfaces/images/437389-00.png) |
| metfaces_png_437390-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437390-00.png](../../data/public_datasets/flow_matching/metfaces/images/437390-00.png) |
| metfaces_png_437391-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437391-00.png](../../data/public_datasets/flow_matching/metfaces/images/437391-00.png) |
| metfaces_png_437395-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437395-00.png](../../data/public_datasets/flow_matching/metfaces/images/437395-00.png) |
| metfaces_png_437396-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437396-00.png](../../data/public_datasets/flow_matching/metfaces/images/437396-00.png) |
| metfaces_png_437397-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437397-00.png](../../data/public_datasets/flow_matching/metfaces/images/437397-00.png) |
| metfaces_png_437399-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437399-00.png](../../data/public_datasets/flow_matching/metfaces/images/437399-00.png) |
| metfaces_png_437400-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437400-00.png](../../data/public_datasets/flow_matching/metfaces/images/437400-00.png) |
| metfaces_png_437402-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437402-00.png](../../data/public_datasets/flow_matching/metfaces/images/437402-00.png) |
| metfaces_png_437404-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437404-00.png](../../data/public_datasets/flow_matching/metfaces/images/437404-00.png) |
| metfaces_png_437405-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437405-00.png](../../data/public_datasets/flow_matching/metfaces/images/437405-00.png) |
| metfaces_png_437406-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437406-00.png](../../data/public_datasets/flow_matching/metfaces/images/437406-00.png) |
| metfaces_png_437407-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437407-00.png](../../data/public_datasets/flow_matching/metfaces/images/437407-00.png) |
| metfaces_png_437408-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437408-00.png](../../data/public_datasets/flow_matching/metfaces/images/437408-00.png) |
| metfaces_png_437409-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437409-00.png](../../data/public_datasets/flow_matching/metfaces/images/437409-00.png) |
| metfaces_png_437410-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437410-00.png](../../data/public_datasets/flow_matching/metfaces/images/437410-00.png) |
| metfaces_png_437411-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437411-00.png](../../data/public_datasets/flow_matching/metfaces/images/437411-00.png) |
| metfaces_png_437416-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437416-00.png](../../data/public_datasets/flow_matching/metfaces/images/437416-00.png) |
| metfaces_png_437417-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437417-00.png](../../data/public_datasets/flow_matching/metfaces/images/437417-00.png) |
| metfaces_png_437419-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437419-00.png](../../data/public_datasets/flow_matching/metfaces/images/437419-00.png) |
| metfaces_png_437420-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437420-00.png](../../data/public_datasets/flow_matching/metfaces/images/437420-00.png) |
| metfaces_png_437422-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437422-00.png](../../data/public_datasets/flow_matching/metfaces/images/437422-00.png) |
| metfaces_png_437424-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437424-00.png](../../data/public_datasets/flow_matching/metfaces/images/437424-00.png) |
| metfaces_png_437425-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437425-00.png](../../data/public_datasets/flow_matching/metfaces/images/437425-00.png) |
| metfaces_png_437430-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437430-00.png](../../data/public_datasets/flow_matching/metfaces/images/437430-00.png) |
| metfaces_png_437432-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437432-00.png](../../data/public_datasets/flow_matching/metfaces/images/437432-00.png) |
| metfaces_png_437437-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437437-00.png](../../data/public_datasets/flow_matching/metfaces/images/437437-00.png) |
| metfaces_png_437439-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437439-00.png](../../data/public_datasets/flow_matching/metfaces/images/437439-00.png) |
| metfaces_png_437446-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437446-00.png](../../data/public_datasets/flow_matching/metfaces/images/437446-00.png) |
| metfaces_png_437448-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437448-00.png](../../data/public_datasets/flow_matching/metfaces/images/437448-00.png) |
| metfaces_png_437450-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437450-00.png](../../data/public_datasets/flow_matching/metfaces/images/437450-00.png) |
| metfaces_png_437451-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437451-00.png](../../data/public_datasets/flow_matching/metfaces/images/437451-00.png) |
| metfaces_png_437452-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437452-00.png](../../data/public_datasets/flow_matching/metfaces/images/437452-00.png) |
| metfaces_png_437453-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437453-00.png](../../data/public_datasets/flow_matching/metfaces/images/437453-00.png) |
| metfaces_png_437455-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437455-01.png](../../data/public_datasets/flow_matching/metfaces/images/437455-01.png) |
| metfaces_png_437456-02 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437456-02.png](../../data/public_datasets/flow_matching/metfaces/images/437456-02.png) |
| metfaces_png_437456-03 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437456-03.png](../../data/public_datasets/flow_matching/metfaces/images/437456-03.png) |
| metfaces_png_437456-04 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437456-04.png](../../data/public_datasets/flow_matching/metfaces/images/437456-04.png) |
| metfaces_png_437456-06 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437456-06.png](../../data/public_datasets/flow_matching/metfaces/images/437456-06.png) |
| metfaces_png_437456-07 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437456-07.png](../../data/public_datasets/flow_matching/metfaces/images/437456-07.png) |
| metfaces_png_437459-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437459-00.png](../../data/public_datasets/flow_matching/metfaces/images/437459-00.png) |
| metfaces_png_437463-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437463-00.png](../../data/public_datasets/flow_matching/metfaces/images/437463-00.png) |
| metfaces_png_437465-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437465-00.png](../../data/public_datasets/flow_matching/metfaces/images/437465-00.png) |
| metfaces_png_437488-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437488-00.png](../../data/public_datasets/flow_matching/metfaces/images/437488-00.png) |
| metfaces_png_437499-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437499-00.png](../../data/public_datasets/flow_matching/metfaces/images/437499-00.png) |
| metfaces_png_437500-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437500-00.png](../../data/public_datasets/flow_matching/metfaces/images/437500-00.png) |
| metfaces_png_437501-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437501-00.png](../../data/public_datasets/flow_matching/metfaces/images/437501-00.png) |
| metfaces_png_437502-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437502-00.png](../../data/public_datasets/flow_matching/metfaces/images/437502-00.png) |
| metfaces_png_437503-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437503-00.png](../../data/public_datasets/flow_matching/metfaces/images/437503-00.png) |
| metfaces_png_437504-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437504-00.png](../../data/public_datasets/flow_matching/metfaces/images/437504-00.png) |
| metfaces_png_437505-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437505-00.png](../../data/public_datasets/flow_matching/metfaces/images/437505-00.png) |
| metfaces_png_437509-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437509-00.png](../../data/public_datasets/flow_matching/metfaces/images/437509-00.png) |
| metfaces_png_437530-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437530-00.png](../../data/public_datasets/flow_matching/metfaces/images/437530-00.png) |
| metfaces_png_437531-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437531-00.png](../../data/public_datasets/flow_matching/metfaces/images/437531-00.png) |
| metfaces_png_437532-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437532-00.png](../../data/public_datasets/flow_matching/metfaces/images/437532-00.png) |
| metfaces_png_437540-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437540-00.png](../../data/public_datasets/flow_matching/metfaces/images/437540-00.png) |
| metfaces_png_437579-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437579-00.png](../../data/public_datasets/flow_matching/metfaces/images/437579-00.png) |
| metfaces_png_437580-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437580-00.png](../../data/public_datasets/flow_matching/metfaces/images/437580-00.png) |
| metfaces_png_437580-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437580-01.png](../../data/public_datasets/flow_matching/metfaces/images/437580-01.png) |
| metfaces_png_437581-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437581-00.png](../../data/public_datasets/flow_matching/metfaces/images/437581-00.png) |
| metfaces_png_437582-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437582-00.png](../../data/public_datasets/flow_matching/metfaces/images/437582-00.png) |
| metfaces_png_437583-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437583-00.png](../../data/public_datasets/flow_matching/metfaces/images/437583-00.png) |
| metfaces_png_437584-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437584-00.png](../../data/public_datasets/flow_matching/metfaces/images/437584-00.png) |
| metfaces_png_437594-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437594-00.png](../../data/public_datasets/flow_matching/metfaces/images/437594-00.png) |
| metfaces_png_437595-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437595-00.png](../../data/public_datasets/flow_matching/metfaces/images/437595-00.png) |
| metfaces_png_437598-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437598-00.png](../../data/public_datasets/flow_matching/metfaces/images/437598-00.png) |
| metfaces_png_437599-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437599-00.png](../../data/public_datasets/flow_matching/metfaces/images/437599-00.png) |
| metfaces_png_437600-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437600-00.png](../../data/public_datasets/flow_matching/metfaces/images/437600-00.png) |
| metfaces_png_437608-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437608-00.png](../../data/public_datasets/flow_matching/metfaces/images/437608-00.png) |
| metfaces_png_437609-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437609-00.png](../../data/public_datasets/flow_matching/metfaces/images/437609-00.png) |
| metfaces_png_437610-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437610-00.png](../../data/public_datasets/flow_matching/metfaces/images/437610-00.png) |
| metfaces_png_437640-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437640-00.png](../../data/public_datasets/flow_matching/metfaces/images/437640-00.png) |
| metfaces_png_437644-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437644-00.png](../../data/public_datasets/flow_matching/metfaces/images/437644-00.png) |
| metfaces_png_437645-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437645-00.png](../../data/public_datasets/flow_matching/metfaces/images/437645-00.png) |
| metfaces_png_437646-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437646-00.png](../../data/public_datasets/flow_matching/metfaces/images/437646-00.png) |
| metfaces_png_437649-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437649-00.png](../../data/public_datasets/flow_matching/metfaces/images/437649-00.png) |
| metfaces_png_437662-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437662-00.png](../../data/public_datasets/flow_matching/metfaces/images/437662-00.png) |
| metfaces_png_437677-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437677-00.png](../../data/public_datasets/flow_matching/metfaces/images/437677-00.png) |
| metfaces_png_437679-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437679-00.png](../../data/public_datasets/flow_matching/metfaces/images/437679-00.png) |
| metfaces_png_437687-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437687-00.png](../../data/public_datasets/flow_matching/metfaces/images/437687-00.png) |
| metfaces_png_437688-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437688-00.png](../../data/public_datasets/flow_matching/metfaces/images/437688-00.png) |
| metfaces_png_437689-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437689-00.png](../../data/public_datasets/flow_matching/metfaces/images/437689-00.png) |
| metfaces_png_437698-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437698-00.png](../../data/public_datasets/flow_matching/metfaces/images/437698-00.png) |
| metfaces_png_437699-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437699-01.png](../../data/public_datasets/flow_matching/metfaces/images/437699-01.png) |
| metfaces_png_437728-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437728-00.png](../../data/public_datasets/flow_matching/metfaces/images/437728-00.png) |
| metfaces_png_437734-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437734-00.png](../../data/public_datasets/flow_matching/metfaces/images/437734-00.png) |
| metfaces_png_437744-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437744-00.png](../../data/public_datasets/flow_matching/metfaces/images/437744-00.png) |
| metfaces_png_437745-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437745-00.png](../../data/public_datasets/flow_matching/metfaces/images/437745-00.png) |
| metfaces_png_437759-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437759-00.png](../../data/public_datasets/flow_matching/metfaces/images/437759-00.png) |
| metfaces_png_437766-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437766-00.png](../../data/public_datasets/flow_matching/metfaces/images/437766-00.png) |
| metfaces_png_437768-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437768-00.png](../../data/public_datasets/flow_matching/metfaces/images/437768-00.png) |
| metfaces_png_437769-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437769-00.png](../../data/public_datasets/flow_matching/metfaces/images/437769-00.png) |
| metfaces_png_437769-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437769-01.png](../../data/public_datasets/flow_matching/metfaces/images/437769-01.png) |
| metfaces_png_437778-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437778-01.png](../../data/public_datasets/flow_matching/metfaces/images/437778-01.png) |
| metfaces_png_437822-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437822-00.png](../../data/public_datasets/flow_matching/metfaces/images/437822-00.png) |
| metfaces_png_437825-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437825-00.png](../../data/public_datasets/flow_matching/metfaces/images/437825-00.png) |
| metfaces_png_437831-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437831-00.png](../../data/public_datasets/flow_matching/metfaces/images/437831-00.png) |
| metfaces_png_437837-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437837-00.png](../../data/public_datasets/flow_matching/metfaces/images/437837-00.png) |
| metfaces_png_437838-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437838-00.png](../../data/public_datasets/flow_matching/metfaces/images/437838-00.png) |
| metfaces_png_437843-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437843-00.png](../../data/public_datasets/flow_matching/metfaces/images/437843-00.png) |
| metfaces_png_437861-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437861-00.png](../../data/public_datasets/flow_matching/metfaces/images/437861-00.png) |
| metfaces_png_437862-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437862-00.png](../../data/public_datasets/flow_matching/metfaces/images/437862-00.png) |
| metfaces_png_437869-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437869-00.png](../../data/public_datasets/flow_matching/metfaces/images/437869-00.png) |
| metfaces_png_437870-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437870-00.png](../../data/public_datasets/flow_matching/metfaces/images/437870-00.png) |
| metfaces_png_437872-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437872-00.png](../../data/public_datasets/flow_matching/metfaces/images/437872-00.png) |
| metfaces_png_437873-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437873-00.png](../../data/public_datasets/flow_matching/metfaces/images/437873-00.png) |
| metfaces_png_437875-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437875-00.png](../../data/public_datasets/flow_matching/metfaces/images/437875-00.png) |
| metfaces_png_437878-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437878-00.png](../../data/public_datasets/flow_matching/metfaces/images/437878-00.png) |
| metfaces_png_437879-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437879-00.png](../../data/public_datasets/flow_matching/metfaces/images/437879-00.png) |
| metfaces_png_437884-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437884-00.png](../../data/public_datasets/flow_matching/metfaces/images/437884-00.png) |
| metfaces_png_437887-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437887-00.png](../../data/public_datasets/flow_matching/metfaces/images/437887-00.png) |
| metfaces_png_437889-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437889-00.png](../../data/public_datasets/flow_matching/metfaces/images/437889-00.png) |
| metfaces_png_437890-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437890-00.png](../../data/public_datasets/flow_matching/metfaces/images/437890-00.png) |
| metfaces_png_437893-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437893-00.png](../../data/public_datasets/flow_matching/metfaces/images/437893-00.png) |
| metfaces_png_437894-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437894-00.png](../../data/public_datasets/flow_matching/metfaces/images/437894-00.png) |
| metfaces_png_437897-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437897-01.png](../../data/public_datasets/flow_matching/metfaces/images/437897-01.png) |
| metfaces_png_437898-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437898-00.png](../../data/public_datasets/flow_matching/metfaces/images/437898-00.png) |
| metfaces_png_437900-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437900-00.png](../../data/public_datasets/flow_matching/metfaces/images/437900-00.png) |
| metfaces_png_437904-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437904-00.png](../../data/public_datasets/flow_matching/metfaces/images/437904-00.png) |
| metfaces_png_437905-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437905-00.png](../../data/public_datasets/flow_matching/metfaces/images/437905-00.png) |
| metfaces_png_437917-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437917-00.png](../../data/public_datasets/flow_matching/metfaces/images/437917-00.png) |
| metfaces_png_437918-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437918-00.png](../../data/public_datasets/flow_matching/metfaces/images/437918-00.png) |
| metfaces_png_437921-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437921-00.png](../../data/public_datasets/flow_matching/metfaces/images/437921-00.png) |
| metfaces_png_437943-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437943-00.png](../../data/public_datasets/flow_matching/metfaces/images/437943-00.png) |
| metfaces_png_437954-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437954-00.png](../../data/public_datasets/flow_matching/metfaces/images/437954-00.png) |
| metfaces_png_437957-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437957-00.png](../../data/public_datasets/flow_matching/metfaces/images/437957-00.png) |
| metfaces_png_437962-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437962-00.png](../../data/public_datasets/flow_matching/metfaces/images/437962-00.png) |
| metfaces_png_437970-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437970-00.png](../../data/public_datasets/flow_matching/metfaces/images/437970-00.png) |
| metfaces_png_437971-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437971-00.png](../../data/public_datasets/flow_matching/metfaces/images/437971-00.png) |
| metfaces_png_437978-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437978-00.png](../../data/public_datasets/flow_matching/metfaces/images/437978-00.png) |
| metfaces_png_437978-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437978-01.png](../../data/public_datasets/flow_matching/metfaces/images/437978-01.png) |
| metfaces_png_437990-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/437990-00.png](../../data/public_datasets/flow_matching/metfaces/images/437990-00.png) |
| metfaces_png_438001-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438001-00.png](../../data/public_datasets/flow_matching/metfaces/images/438001-00.png) |
| metfaces_png_438001-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438001-01.png](../../data/public_datasets/flow_matching/metfaces/images/438001-01.png) |
| metfaces_png_438009-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438009-00.png](../../data/public_datasets/flow_matching/metfaces/images/438009-00.png) |
| metfaces_png_438011-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438011-00.png](../../data/public_datasets/flow_matching/metfaces/images/438011-00.png) |
| metfaces_png_438016-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438016-00.png](../../data/public_datasets/flow_matching/metfaces/images/438016-00.png) |
| metfaces_png_438020-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438020-00.png](../../data/public_datasets/flow_matching/metfaces/images/438020-00.png) |
| metfaces_png_438032-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438032-00.png](../../data/public_datasets/flow_matching/metfaces/images/438032-00.png) |
| metfaces_png_438033-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438033-00.png](../../data/public_datasets/flow_matching/metfaces/images/438033-00.png) |
| metfaces_png_438112-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438112-00.png](../../data/public_datasets/flow_matching/metfaces/images/438112-00.png) |
| metfaces_png_438139-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438139-00.png](../../data/public_datasets/flow_matching/metfaces/images/438139-00.png) |
| metfaces_png_438378-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438378-00.png](../../data/public_datasets/flow_matching/metfaces/images/438378-00.png) |
| metfaces_png_438379-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438379-00.png](../../data/public_datasets/flow_matching/metfaces/images/438379-00.png) |
| metfaces_png_438389-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438389-00.png](../../data/public_datasets/flow_matching/metfaces/images/438389-00.png) |
| metfaces_png_438407-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438407-00.png](../../data/public_datasets/flow_matching/metfaces/images/438407-00.png) |
| metfaces_png_438434-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438434-00.png](../../data/public_datasets/flow_matching/metfaces/images/438434-00.png) |
| metfaces_png_438434-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438434-01.png](../../data/public_datasets/flow_matching/metfaces/images/438434-01.png) |
| metfaces_png_438434-02 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438434-02.png](../../data/public_datasets/flow_matching/metfaces/images/438434-02.png) |
| metfaces_png_438544-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438544-00.png](../../data/public_datasets/flow_matching/metfaces/images/438544-00.png) |
| metfaces_png_438585-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438585-00.png](../../data/public_datasets/flow_matching/metfaces/images/438585-00.png) |
| metfaces_png_438590-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438590-00.png](../../data/public_datasets/flow_matching/metfaces/images/438590-00.png) |
| metfaces_png_438603-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438603-00.png](../../data/public_datasets/flow_matching/metfaces/images/438603-00.png) |
| metfaces_png_438615-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438615-00.png](../../data/public_datasets/flow_matching/metfaces/images/438615-00.png) |
| metfaces_png_438737-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438737-00.png](../../data/public_datasets/flow_matching/metfaces/images/438737-00.png) |
| metfaces_png_438740-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438740-00.png](../../data/public_datasets/flow_matching/metfaces/images/438740-00.png) |
| metfaces_png_438776-03 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438776-03.png](../../data/public_datasets/flow_matching/metfaces/images/438776-03.png) |
| metfaces_png_438780-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438780-00.png](../../data/public_datasets/flow_matching/metfaces/images/438780-00.png) |
| metfaces_png_438818-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438818-00.png](../../data/public_datasets/flow_matching/metfaces/images/438818-00.png) |
| metfaces_png_438819-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438819-00.png](../../data/public_datasets/flow_matching/metfaces/images/438819-00.png) |
| metfaces_png_438844-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/438844-00.png](../../data/public_datasets/flow_matching/metfaces/images/438844-00.png) |
| metfaces_png_439273-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/439273-00.png](../../data/public_datasets/flow_matching/metfaces/images/439273-00.png) |
| metfaces_png_439327-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/439327-00.png](../../data/public_datasets/flow_matching/metfaces/images/439327-00.png) |
| metfaces_png_439337-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/439337-00.png](../../data/public_datasets/flow_matching/metfaces/images/439337-00.png) |
| metfaces_png_439405-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/439405-00.png](../../data/public_datasets/flow_matching/metfaces/images/439405-00.png) |
| metfaces_png_439933-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/439933-00.png](../../data/public_datasets/flow_matching/metfaces/images/439933-00.png) |
| metfaces_png_440464-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/440464-00.png](../../data/public_datasets/flow_matching/metfaces/images/440464-00.png) |
| metfaces_png_441111-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/441111-00.png](../../data/public_datasets/flow_matching/metfaces/images/441111-00.png) |
| metfaces_png_441115-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/441115-00.png](../../data/public_datasets/flow_matching/metfaces/images/441115-00.png) |
| metfaces_png_441227-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/441227-00.png](../../data/public_datasets/flow_matching/metfaces/images/441227-00.png) |
| metfaces_png_441227-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/441227-01.png](../../data/public_datasets/flow_matching/metfaces/images/441227-01.png) |
| metfaces_png_441230-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/441230-00.png](../../data/public_datasets/flow_matching/metfaces/images/441230-00.png) |
| metfaces_png_441355-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/441355-00.png](../../data/public_datasets/flow_matching/metfaces/images/441355-00.png) |
| metfaces_png_441363-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/441363-00.png](../../data/public_datasets/flow_matching/metfaces/images/441363-00.png) |
| metfaces_png_441365-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/441365-00.png](../../data/public_datasets/flow_matching/metfaces/images/441365-00.png) |
| metfaces_png_442749-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/442749-00.png](../../data/public_datasets/flow_matching/metfaces/images/442749-00.png) |
| metfaces_png_44613-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/44613-00.png](../../data/public_datasets/flow_matching/metfaces/images/44613-00.png) |
| metfaces_png_447125-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/447125-00.png](../../data/public_datasets/flow_matching/metfaces/images/447125-00.png) |
| metfaces_png_447125-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/447125-01.png](../../data/public_datasets/flow_matching/metfaces/images/447125-01.png) |
| metfaces_png_44833-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/44833-00.png](../../data/public_datasets/flow_matching/metfaces/images/44833-00.png) |
| metfaces_png_450749-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/450749-00.png](../../data/public_datasets/flow_matching/metfaces/images/450749-00.png) |
| metfaces_png_454621-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/454621-00.png](../../data/public_datasets/flow_matching/metfaces/images/454621-00.png) |
| metfaces_png_457723-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/457723-00.png](../../data/public_datasets/flow_matching/metfaces/images/457723-00.png) |
| metfaces_png_457781-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/457781-00.png](../../data/public_datasets/flow_matching/metfaces/images/457781-00.png) |
| metfaces_png_458953-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/458953-00.png](../../data/public_datasets/flow_matching/metfaces/images/458953-00.png) |
| metfaces_png_458972-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/458972-00.png](../../data/public_datasets/flow_matching/metfaces/images/458972-00.png) |
| metfaces_png_458982-03 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/458982-03.png](../../data/public_datasets/flow_matching/metfaces/images/458982-03.png) |
| metfaces_png_458983-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/458983-00.png](../../data/public_datasets/flow_matching/metfaces/images/458983-00.png) |
| metfaces_png_459015-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459015-00.png](../../data/public_datasets/flow_matching/metfaces/images/459015-00.png) |
| metfaces_png_459015-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459015-01.png](../../data/public_datasets/flow_matching/metfaces/images/459015-01.png) |
| metfaces_png_459039-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459039-00.png](../../data/public_datasets/flow_matching/metfaces/images/459039-00.png) |
| metfaces_png_459039-02 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459039-02.png](../../data/public_datasets/flow_matching/metfaces/images/459039-02.png) |
| metfaces_png_459052-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459052-01.png](../../data/public_datasets/flow_matching/metfaces/images/459052-01.png) |
| metfaces_png_459052-02 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459052-02.png](../../data/public_datasets/flow_matching/metfaces/images/459052-02.png) |
| metfaces_png_459053-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459053-00.png](../../data/public_datasets/flow_matching/metfaces/images/459053-00.png) |
| metfaces_png_459054-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459054-00.png](../../data/public_datasets/flow_matching/metfaces/images/459054-00.png) |
| metfaces_png_459059-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459059-00.png](../../data/public_datasets/flow_matching/metfaces/images/459059-00.png) |
| metfaces_png_459059-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459059-01.png](../../data/public_datasets/flow_matching/metfaces/images/459059-01.png) |
| metfaces_png_459071-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459071-00.png](../../data/public_datasets/flow_matching/metfaces/images/459071-00.png) |
| metfaces_png_459073-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459073-00.png](../../data/public_datasets/flow_matching/metfaces/images/459073-00.png) |
| metfaces_png_459074-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459074-00.png](../../data/public_datasets/flow_matching/metfaces/images/459074-00.png) |
| metfaces_png_459081-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459081-00.png](../../data/public_datasets/flow_matching/metfaces/images/459081-00.png) |
| metfaces_png_459082-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459082-00.png](../../data/public_datasets/flow_matching/metfaces/images/459082-00.png) |
| metfaces_png_459087-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459087-00.png](../../data/public_datasets/flow_matching/metfaces/images/459087-00.png) |
| metfaces_png_459088-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459088-00.png](../../data/public_datasets/flow_matching/metfaces/images/459088-00.png) |
| metfaces_png_459089-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459089-00.png](../../data/public_datasets/flow_matching/metfaces/images/459089-00.png) |
| metfaces_png_459090-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459090-01.png](../../data/public_datasets/flow_matching/metfaces/images/459090-01.png) |
| metfaces_png_459106-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459106-00.png](../../data/public_datasets/flow_matching/metfaces/images/459106-00.png) |
| metfaces_png_459126-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459126-00.png](../../data/public_datasets/flow_matching/metfaces/images/459126-00.png) |
| metfaces_png_459127-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459127-00.png](../../data/public_datasets/flow_matching/metfaces/images/459127-00.png) |
| metfaces_png_459129-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459129-00.png](../../data/public_datasets/flow_matching/metfaces/images/459129-00.png) |
| metfaces_png_459213-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459213-00.png](../../data/public_datasets/flow_matching/metfaces/images/459213-00.png) |
| metfaces_png_459253-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459253-00.png](../../data/public_datasets/flow_matching/metfaces/images/459253-00.png) |
| metfaces_png_459254-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459254-00.png](../../data/public_datasets/flow_matching/metfaces/images/459254-00.png) |
| metfaces_png_459425-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459425-00.png](../../data/public_datasets/flow_matching/metfaces/images/459425-00.png) |
| metfaces_png_459965-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/459965-00.png](../../data/public_datasets/flow_matching/metfaces/images/459965-00.png) |
| metfaces_png_463607-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/463607-00.png](../../data/public_datasets/flow_matching/metfaces/images/463607-00.png) |
| metfaces_png_463798-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/463798-00.png](../../data/public_datasets/flow_matching/metfaces/images/463798-00.png) |
| metfaces_png_464127-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/464127-00.png](../../data/public_datasets/flow_matching/metfaces/images/464127-00.png) |
| metfaces_png_464128-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/464128-00.png](../../data/public_datasets/flow_matching/metfaces/images/464128-00.png) |
| metfaces_png_464596-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/464596-01.png](../../data/public_datasets/flow_matching/metfaces/images/464596-01.png) |
| metfaces_png_466084-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/466084-00.png](../../data/public_datasets/flow_matching/metfaces/images/466084-00.png) |
| metfaces_png_467786-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/467786-00.png](../../data/public_datasets/flow_matching/metfaces/images/467786-00.png) |
| metfaces_png_467997-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/467997-00.png](../../data/public_datasets/flow_matching/metfaces/images/467997-00.png) |
| metfaces_png_468720-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/468720-00.png](../../data/public_datasets/flow_matching/metfaces/images/468720-00.png) |
| metfaces_png_469927-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/469927-00.png](../../data/public_datasets/flow_matching/metfaces/images/469927-00.png) |
| metfaces_png_470265-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/470265-00.png](../../data/public_datasets/flow_matching/metfaces/images/470265-00.png) |
| metfaces_png_470266-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/470266-00.png](../../data/public_datasets/flow_matching/metfaces/images/470266-00.png) |
| metfaces_png_470273-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/470273-00.png](../../data/public_datasets/flow_matching/metfaces/images/470273-00.png) |
| metfaces_png_470315-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/470315-00.png](../../data/public_datasets/flow_matching/metfaces/images/470315-00.png) |
| metfaces_png_471036-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/471036-00.png](../../data/public_datasets/flow_matching/metfaces/images/471036-00.png) |
| metfaces_png_471203-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/471203-00.png](../../data/public_datasets/flow_matching/metfaces/images/471203-00.png) |
| metfaces_png_471432-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/471432-00.png](../../data/public_datasets/flow_matching/metfaces/images/471432-00.png) |
| metfaces_png_471460-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/471460-00.png](../../data/public_datasets/flow_matching/metfaces/images/471460-00.png) |
| metfaces_png_471844-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/471844-00.png](../../data/public_datasets/flow_matching/metfaces/images/471844-00.png) |
| metfaces_png_471845-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/471845-00.png](../../data/public_datasets/flow_matching/metfaces/images/471845-00.png) |
| metfaces_png_471853-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/471853-00.png](../../data/public_datasets/flow_matching/metfaces/images/471853-00.png) |
| metfaces_png_471853-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/471853-01.png](../../data/public_datasets/flow_matching/metfaces/images/471853-01.png) |
| metfaces_png_471908-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/471908-00.png](../../data/public_datasets/flow_matching/metfaces/images/471908-00.png) |
| metfaces_png_471942-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/471942-00.png](../../data/public_datasets/flow_matching/metfaces/images/471942-00.png) |
| metfaces_png_471947-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/471947-00.png](../../data/public_datasets/flow_matching/metfaces/images/471947-00.png) |
| metfaces_png_471996-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/471996-00.png](../../data/public_datasets/flow_matching/metfaces/images/471996-00.png) |
| metfaces_png_472342-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/472342-00.png](../../data/public_datasets/flow_matching/metfaces/images/472342-00.png) |
| metfaces_png_473849-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/473849-00.png](../../data/public_datasets/flow_matching/metfaces/images/473849-00.png) |
| metfaces_png_473877-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/473877-00.png](../../data/public_datasets/flow_matching/metfaces/images/473877-00.png) |
| metfaces_png_477317-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/477317-00.png](../../data/public_datasets/flow_matching/metfaces/images/477317-00.png) |
| metfaces_png_47916-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/47916-00.png](../../data/public_datasets/flow_matching/metfaces/images/47916-00.png) |
| metfaces_png_479673-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/479673-00.png](../../data/public_datasets/flow_matching/metfaces/images/479673-00.png) |
| metfaces_png_483-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/483-00.png](../../data/public_datasets/flow_matching/metfaces/images/483-00.png) |
| metfaces_png_485-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/485-00.png](../../data/public_datasets/flow_matching/metfaces/images/485-00.png) |
| metfaces_png_485551-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/485551-00.png](../../data/public_datasets/flow_matching/metfaces/images/485551-00.png) |
| metfaces_png_486-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/486-00.png](../../data/public_datasets/flow_matching/metfaces/images/486-00.png) |
| metfaces_png_49122-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/49122-00.png](../../data/public_datasets/flow_matching/metfaces/images/49122-00.png) |
| metfaces_png_49252-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/49252-00.png](../../data/public_datasets/flow_matching/metfaces/images/49252-00.png) |
| metfaces_png_49345-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/49345-00.png](../../data/public_datasets/flow_matching/metfaces/images/49345-00.png) |
| metfaces_png_49567-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/49567-00.png](../../data/public_datasets/flow_matching/metfaces/images/49567-00.png) |
| metfaces_png_49892-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/49892-00.png](../../data/public_datasets/flow_matching/metfaces/images/49892-00.png) |
| metfaces_png_500-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/500-00.png](../../data/public_datasets/flow_matching/metfaces/images/500-00.png) |
| metfaces_png_50339-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/50339-00.png](../../data/public_datasets/flow_matching/metfaces/images/50339-00.png) |
| metfaces_png_505722-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/505722-00.png](../../data/public_datasets/flow_matching/metfaces/images/505722-00.png) |
| metfaces_png_506078-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/506078-00.png](../../data/public_datasets/flow_matching/metfaces/images/506078-00.png) |
| metfaces_png_506080-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/506080-00.png](../../data/public_datasets/flow_matching/metfaces/images/506080-00.png) |
| metfaces_png_506089-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/506089-00.png](../../data/public_datasets/flow_matching/metfaces/images/506089-00.png) |
| metfaces_png_506090-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/506090-00.png](../../data/public_datasets/flow_matching/metfaces/images/506090-00.png) |
| metfaces_png_50959-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/50959-00.png](../../data/public_datasets/flow_matching/metfaces/images/50959-00.png) |
| metfaces_png_51823-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/51823-00.png](../../data/public_datasets/flow_matching/metfaces/images/51823-00.png) |
| metfaces_png_52701-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/52701-00.png](../../data/public_datasets/flow_matching/metfaces/images/52701-00.png) |
| metfaces_png_53156-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/53156-00.png](../../data/public_datasets/flow_matching/metfaces/images/53156-00.png) |
| metfaces_png_53531-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/53531-00.png](../../data/public_datasets/flow_matching/metfaces/images/53531-00.png) |
| metfaces_png_54077-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/54077-00.png](../../data/public_datasets/flow_matching/metfaces/images/54077-00.png) |
| metfaces_png_54080-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/54080-00.png](../../data/public_datasets/flow_matching/metfaces/images/54080-00.png) |
| metfaces_png_543872-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/543872-00.png](../../data/public_datasets/flow_matching/metfaces/images/543872-00.png) |
| metfaces_png_543900-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/543900-00.png](../../data/public_datasets/flow_matching/metfaces/images/543900-00.png) |
| metfaces_png_543951-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/543951-00.png](../../data/public_datasets/flow_matching/metfaces/images/543951-00.png) |
| metfaces_png_544176-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/544176-00.png](../../data/public_datasets/flow_matching/metfaces/images/544176-00.png) |
| metfaces_png_544334-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/544334-00.png](../../data/public_datasets/flow_matching/metfaces/images/544334-00.png) |
| metfaces_png_544336-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/544336-00.png](../../data/public_datasets/flow_matching/metfaces/images/544336-00.png) |
| metfaces_png_544340-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/544340-00.png](../../data/public_datasets/flow_matching/metfaces/images/544340-00.png) |
| metfaces_png_544454-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/544454-00.png](../../data/public_datasets/flow_matching/metfaces/images/544454-00.png) |
| metfaces_png_544685-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/544685-00.png](../../data/public_datasets/flow_matching/metfaces/images/544685-00.png) |
| metfaces_png_544690-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/544690-00.png](../../data/public_datasets/flow_matching/metfaces/images/544690-00.png) |
| metfaces_png_544704-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/544704-00.png](../../data/public_datasets/flow_matching/metfaces/images/544704-00.png) |
| metfaces_png_544709-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/544709-00.png](../../data/public_datasets/flow_matching/metfaces/images/544709-00.png) |
| metfaces_png_544723-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/544723-00.png](../../data/public_datasets/flow_matching/metfaces/images/544723-00.png) |
| metfaces_png_544752-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/544752-00.png](../../data/public_datasets/flow_matching/metfaces/images/544752-00.png) |
| metfaces_png_544824-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/544824-00.png](../../data/public_datasets/flow_matching/metfaces/images/544824-00.png) |
| metfaces_png_545147-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/545147-00.png](../../data/public_datasets/flow_matching/metfaces/images/545147-00.png) |
| metfaces_png_545477-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/545477-00.png](../../data/public_datasets/flow_matching/metfaces/images/545477-00.png) |
| metfaces_png_54593-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/54593-00.png](../../data/public_datasets/flow_matching/metfaces/images/54593-00.png) |
| metfaces_png_546289-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/546289-00.png](../../data/public_datasets/flow_matching/metfaces/images/546289-00.png) |
| metfaces_png_546290-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/546290-00.png](../../data/public_datasets/flow_matching/metfaces/images/546290-00.png) |
| metfaces_png_546291-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/546291-00.png](../../data/public_datasets/flow_matching/metfaces/images/546291-00.png) |
| metfaces_png_546304-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/546304-00.png](../../data/public_datasets/flow_matching/metfaces/images/546304-00.png) |
| metfaces_png_547257-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/547257-00.png](../../data/public_datasets/flow_matching/metfaces/images/547257-00.png) |
| metfaces_png_547565-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/547565-00.png](../../data/public_datasets/flow_matching/metfaces/images/547565-00.png) |
| metfaces_png_547677-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/547677-00.png](../../data/public_datasets/flow_matching/metfaces/images/547677-00.png) |
| metfaces_png_547698-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/547698-00.png](../../data/public_datasets/flow_matching/metfaces/images/547698-00.png) |
| metfaces_png_547716-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/547716-00.png](../../data/public_datasets/flow_matching/metfaces/images/547716-00.png) |
| metfaces_png_547822-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/547822-00.png](../../data/public_datasets/flow_matching/metfaces/images/547822-00.png) |
| metfaces_png_547826-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/547826-00.png](../../data/public_datasets/flow_matching/metfaces/images/547826-00.png) |
| metfaces_png_547830-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/547830-00.png](../../data/public_datasets/flow_matching/metfaces/images/547830-00.png) |
| metfaces_png_547836-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/547836-00.png](../../data/public_datasets/flow_matching/metfaces/images/547836-00.png) |
| metfaces_png_547844-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/547844-00.png](../../data/public_datasets/flow_matching/metfaces/images/547844-00.png) |
| metfaces_png_547849-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/547849-00.png](../../data/public_datasets/flow_matching/metfaces/images/547849-00.png) |
| metfaces_png_547850-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/547850-00.png](../../data/public_datasets/flow_matching/metfaces/images/547850-00.png) |
| metfaces_png_547851-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/547851-00.png](../../data/public_datasets/flow_matching/metfaces/images/547851-00.png) |
| metfaces_png_547852-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/547852-00.png](../../data/public_datasets/flow_matching/metfaces/images/547852-00.png) |
| metfaces_png_547860-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/547860-00.png](../../data/public_datasets/flow_matching/metfaces/images/547860-00.png) |
| metfaces_png_547861-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/547861-00.png](../../data/public_datasets/flow_matching/metfaces/images/547861-00.png) |
| metfaces_png_548277-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/548277-00.png](../../data/public_datasets/flow_matching/metfaces/images/548277-00.png) |
| metfaces_png_548585-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/548585-00.png](../../data/public_datasets/flow_matching/metfaces/images/548585-00.png) |
| metfaces_png_549228-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/549228-00.png](../../data/public_datasets/flow_matching/metfaces/images/549228-00.png) |
| metfaces_png_550773-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/550773-00.png](../../data/public_datasets/flow_matching/metfaces/images/550773-00.png) |
| metfaces_png_550798-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/550798-00.png](../../data/public_datasets/flow_matching/metfaces/images/550798-00.png) |
| metfaces_png_551104-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/551104-00.png](../../data/public_datasets/flow_matching/metfaces/images/551104-00.png) |
| metfaces_png_551144-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/551144-00.png](../../data/public_datasets/flow_matching/metfaces/images/551144-00.png) |
| metfaces_png_551291-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/551291-00.png](../../data/public_datasets/flow_matching/metfaces/images/551291-00.png) |
| metfaces_png_553763-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/553763-00.png](../../data/public_datasets/flow_matching/metfaces/images/553763-00.png) |
| metfaces_png_553922-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/553922-00.png](../../data/public_datasets/flow_matching/metfaces/images/553922-00.png) |
| metfaces_png_558169-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/558169-00.png](../../data/public_datasets/flow_matching/metfaces/images/558169-00.png) |
| metfaces_png_57612-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/57612-00.png](../../data/public_datasets/flow_matching/metfaces/images/57612-00.png) |
| metfaces_png_61719-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/61719-00.png](../../data/public_datasets/flow_matching/metfaces/images/61719-00.png) |
| metfaces_png_627328-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/627328-00.png](../../data/public_datasets/flow_matching/metfaces/images/627328-00.png) |
| metfaces_png_631063-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/631063-00.png](../../data/public_datasets/flow_matching/metfaces/images/631063-00.png) |
| metfaces_png_632807-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/632807-00.png](../../data/public_datasets/flow_matching/metfaces/images/632807-00.png) |
| metfaces_png_635819-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/635819-00.png](../../data/public_datasets/flow_matching/metfaces/images/635819-00.png) |
| metfaces_png_638084-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/638084-00.png](../../data/public_datasets/flow_matching/metfaces/images/638084-00.png) |
| metfaces_png_639240-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/639240-00.png](../../data/public_datasets/flow_matching/metfaces/images/639240-00.png) |
| metfaces_png_639853-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/639853-00.png](../../data/public_datasets/flow_matching/metfaces/images/639853-00.png) |
| metfaces_png_639909-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/639909-00.png](../../data/public_datasets/flow_matching/metfaces/images/639909-00.png) |
| metfaces_png_640565-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/640565-00.png](../../data/public_datasets/flow_matching/metfaces/images/640565-00.png) |
| metfaces_png_641257-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/641257-00.png](../../data/public_datasets/flow_matching/metfaces/images/641257-00.png) |
| metfaces_png_641594-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/641594-00.png](../../data/public_datasets/flow_matching/metfaces/images/641594-00.png) |
| metfaces_png_641684-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/641684-00.png](../../data/public_datasets/flow_matching/metfaces/images/641684-00.png) |
| metfaces_png_641721-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/641721-00.png](../../data/public_datasets/flow_matching/metfaces/images/641721-00.png) |
| metfaces_png_641729-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/641729-00.png](../../data/public_datasets/flow_matching/metfaces/images/641729-00.png) |
| metfaces_png_643303-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/643303-00.png](../../data/public_datasets/flow_matching/metfaces/images/643303-00.png) |
| metfaces_png_643540-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/643540-00.png](../../data/public_datasets/flow_matching/metfaces/images/643540-00.png) |
| metfaces_png_643553-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/643553-00.png](../../data/public_datasets/flow_matching/metfaces/images/643553-00.png) |
| metfaces_png_647704-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/647704-00.png](../../data/public_datasets/flow_matching/metfaces/images/647704-00.png) |
| metfaces_png_647705-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/647705-00.png](../../data/public_datasets/flow_matching/metfaces/images/647705-00.png) |
| metfaces_png_64897-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/64897-00.png](../../data/public_datasets/flow_matching/metfaces/images/64897-00.png) |
| metfaces_png_654327-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/654327-00.png](../../data/public_datasets/flow_matching/metfaces/images/654327-00.png) |
| metfaces_png_655420-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/655420-00.png](../../data/public_datasets/flow_matching/metfaces/images/655420-00.png) |
| metfaces_png_655421-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/655421-00.png](../../data/public_datasets/flow_matching/metfaces/images/655421-00.png) |
| metfaces_png_655744-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/655744-00.png](../../data/public_datasets/flow_matching/metfaces/images/655744-00.png) |
| metfaces_png_667926-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/667926-00.png](../../data/public_datasets/flow_matching/metfaces/images/667926-00.png) |
| metfaces_png_667985-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/667985-00.png](../../data/public_datasets/flow_matching/metfaces/images/667985-00.png) |
| metfaces_png_668082-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/668082-00.png](../../data/public_datasets/flow_matching/metfaces/images/668082-00.png) |
| metfaces_png_669033-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/669033-00.png](../../data/public_datasets/flow_matching/metfaces/images/669033-00.png) |
| metfaces_png_670765-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/670765-00.png](../../data/public_datasets/flow_matching/metfaces/images/670765-00.png) |
| metfaces_png_671454-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/671454-00.png](../../data/public_datasets/flow_matching/metfaces/images/671454-00.png) |
| metfaces_png_671454-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/671454-01.png](../../data/public_datasets/flow_matching/metfaces/images/671454-01.png) |
| metfaces_png_687513-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/687513-00.png](../../data/public_datasets/flow_matching/metfaces/images/687513-00.png) |
| metfaces_png_698625-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/698625-00.png](../../data/public_datasets/flow_matching/metfaces/images/698625-00.png) |
| metfaces_png_700948-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/700948-00.png](../../data/public_datasets/flow_matching/metfaces/images/700948-00.png) |
| metfaces_png_701989-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/701989-00.png](../../data/public_datasets/flow_matching/metfaces/images/701989-00.png) |
| metfaces_png_703199-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/703199-00.png](../../data/public_datasets/flow_matching/metfaces/images/703199-00.png) |
| metfaces_png_705481-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/705481-00.png](../../data/public_datasets/flow_matching/metfaces/images/705481-00.png) |
| metfaces_png_705777-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/705777-00.png](../../data/public_datasets/flow_matching/metfaces/images/705777-00.png) |
| metfaces_png_707805-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/707805-00.png](../../data/public_datasets/flow_matching/metfaces/images/707805-00.png) |
| metfaces_png_707898-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/707898-00.png](../../data/public_datasets/flow_matching/metfaces/images/707898-00.png) |
| metfaces_png_707898-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/707898-01.png](../../data/public_datasets/flow_matching/metfaces/images/707898-01.png) |
| metfaces_png_708301-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/708301-00.png](../../data/public_datasets/flow_matching/metfaces/images/708301-00.png) |
| metfaces_png_708308-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/708308-00.png](../../data/public_datasets/flow_matching/metfaces/images/708308-00.png) |
| metfaces_png_708341-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/708341-00.png](../../data/public_datasets/flow_matching/metfaces/images/708341-00.png) |
| metfaces_png_708350-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/708350-00.png](../../data/public_datasets/flow_matching/metfaces/images/708350-00.png) |
| metfaces_png_708353-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/708353-00.png](../../data/public_datasets/flow_matching/metfaces/images/708353-00.png) |
| metfaces_png_708365-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/708365-00.png](../../data/public_datasets/flow_matching/metfaces/images/708365-00.png) |
| metfaces_png_712539-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/712539-00.png](../../data/public_datasets/flow_matching/metfaces/images/712539-00.png) |
| metfaces_png_726543-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/726543-00.png](../../data/public_datasets/flow_matching/metfaces/images/726543-00.png) |
| metfaces_png_735054-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/735054-00.png](../../data/public_datasets/flow_matching/metfaces/images/735054-00.png) |
| metfaces_png_735091-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/735091-00.png](../../data/public_datasets/flow_matching/metfaces/images/735091-00.png) |
| metfaces_png_735098-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/735098-00.png](../../data/public_datasets/flow_matching/metfaces/images/735098-00.png) |
| metfaces_png_738456-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/738456-00.png](../../data/public_datasets/flow_matching/metfaces/images/738456-00.png) |
| metfaces_png_742415-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/742415-00.png](../../data/public_datasets/flow_matching/metfaces/images/742415-00.png) |
| metfaces_png_742432-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/742432-00.png](../../data/public_datasets/flow_matching/metfaces/images/742432-00.png) |
| metfaces_png_742434-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/742434-00.png](../../data/public_datasets/flow_matching/metfaces/images/742434-00.png) |
| metfaces_png_746938-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/746938-01.png](../../data/public_datasets/flow_matching/metfaces/images/746938-01.png) |
| metfaces_png_747607-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/747607-00.png](../../data/public_datasets/flow_matching/metfaces/images/747607-00.png) |
| metfaces_png_748947-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/748947-00.png](../../data/public_datasets/flow_matching/metfaces/images/748947-00.png) |
| metfaces_png_75914-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/75914-00.png](../../data/public_datasets/flow_matching/metfaces/images/75914-00.png) |
| metfaces_png_761570-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/761570-00.png](../../data/public_datasets/flow_matching/metfaces/images/761570-00.png) |
| metfaces_png_767842-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/767842-00.png](../../data/public_datasets/flow_matching/metfaces/images/767842-00.png) |
| metfaces_png_771936-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/771936-00.png](../../data/public_datasets/flow_matching/metfaces/images/771936-00.png) |
| metfaces_png_773251-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/773251-00.png](../../data/public_datasets/flow_matching/metfaces/images/773251-00.png) |
| metfaces_png_775309-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/775309-00.png](../../data/public_datasets/flow_matching/metfaces/images/775309-00.png) |
| metfaces_png_775312-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/775312-00.png](../../data/public_datasets/flow_matching/metfaces/images/775312-00.png) |
| metfaces_png_78434-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/78434-00.png](../../data/public_datasets/flow_matching/metfaces/images/78434-00.png) |
| metfaces_png_78569-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/78569-00.png](../../data/public_datasets/flow_matching/metfaces/images/78569-00.png) |
| metfaces_png_78666-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/78666-00.png](../../data/public_datasets/flow_matching/metfaces/images/78666-00.png) |
| metfaces_png_813585-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/813585-00.png](../../data/public_datasets/flow_matching/metfaces/images/813585-00.png) |
| metfaces_png_813594-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/813594-00.png](../../data/public_datasets/flow_matching/metfaces/images/813594-00.png) |
| metfaces_png_815550-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/815550-00.png](../../data/public_datasets/flow_matching/metfaces/images/815550-00.png) |
| metfaces_png_815550-01 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/815550-01.png](../../data/public_datasets/flow_matching/metfaces/images/815550-01.png) |
| metfaces_png_816873-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/816873-00.png](../../data/public_datasets/flow_matching/metfaces/images/816873-00.png) |
| metfaces_png_817504-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/817504-00.png](../../data/public_datasets/flow_matching/metfaces/images/817504-00.png) |
| metfaces_png_822597-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/822597-00.png](../../data/public_datasets/flow_matching/metfaces/images/822597-00.png) |
| metfaces_png_822737-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/822737-00.png](../../data/public_datasets/flow_matching/metfaces/images/822737-00.png) |
| metfaces_png_822756-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/822756-00.png](../../data/public_datasets/flow_matching/metfaces/images/822756-00.png) |
| metfaces_png_823401-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/823401-00.png](../../data/public_datasets/flow_matching/metfaces/images/823401-00.png) |
| metfaces_png_826758-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/826758-00.png](../../data/public_datasets/flow_matching/metfaces/images/826758-00.png) |
| metfaces_png_827515-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/827515-00.png](../../data/public_datasets/flow_matching/metfaces/images/827515-00.png) |
| metfaces_png_828499-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/828499-00.png](../../data/public_datasets/flow_matching/metfaces/images/828499-00.png) |
| metfaces_png_828503-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/828503-00.png](../../data/public_datasets/flow_matching/metfaces/images/828503-00.png) |
| metfaces_png_828517-00 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/metfaces/images/828517-00.png](../../data/public_datasets/flow_matching/metfaces/images/828517-00.png) |
| metfaces_metadata | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/metfaces/metfaces-dataset.json](../../data/public_datasets/flow_matching/metfaces/metfaces-dataset.json) |

### Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport

PDF：本地可用；当前文件存在且尺寸一致；摘要校验依据下载 manifest。 [papers/literature/flow_matching/pdfs/ot_cfm.pdf](../../papers/literature/flow_matching/pdfs/ot_cfm.pdf)

实验来源：[原论文/官方证据](https://arxiv.org/html/2302.00482)。

论文列明实验数据集：cifar10；celeba_original；eb_processed_trajectorynet；eb_raw；neurips2022_cite_seq；neurips2022_multiome；ot_cfm_toy_distributions；ot_cfm_sb_ground_truth；ot_cfm_funnel_10d。

当前可用资料分类：原始/基础公开数据或作者数据包 7 项。

| 来源 ID | 材料类型 | 现场状态 | 路径 |
|---|---|---|---|
| cifar10 | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/cifar10/cifar-10-python.tar.gz](../../data/public_datasets/flow_matching/cifar10/cifar-10-python.tar.gz) |
| celeba_original | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/celeba/img_align_celeba.manual](../../data/public_datasets/flow_matching/celeba/img_align_celeba.manual) |
| eb_processed_trajectorynet | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/embryoid_body/eb_velocity_v5.npz](../../data/public_datasets/flow_matching/embryoid_body/eb_velocity_v5.npz) |
| eb_raw | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/embryoid_body/scRNAseq.manual](../../data/public_datasets/flow_matching/embryoid_body/scRNAseq.manual) |
| neurips2022_cite_seq | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/single_cell/cite_seq.manual](../../data/public_datasets/flow_matching/single_cell/cite_seq.manual) |
| neurips2022_multiome | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/single_cell/multiome.manual](../../data/public_datasets/flow_matching/single_cell/multiome.manual) |
| ot_cfm_toy_distributions | 生成配方/源码 | 无下载记录 | [data/public_datasets/flow_matching/synthetic/ot_cfm_toy.generated](../../data/public_datasets/flow_matching/synthetic/ot_cfm_toy.generated) |
| ot_cfm_sb_ground_truth | 生成配方/源码 | 无下载记录 | [data/public_datasets/flow_matching/synthetic/schrodinger_bridge.generated](../../data/public_datasets/flow_matching/synthetic/schrodinger_bridge.generated) |
| ot_cfm_funnel_10d | 生成配方/源码 | 无下载记录 | [data/public_datasets/flow_matching/synthetic/funnel_10d.generated](../../data/public_datasets/flow_matching/synthetic/funnel_10d.generated) |
| neurips2022_metadata_csv | 受限/未公开 | 无下载记录 | [data/public_datasets/flow_matching/single_cell/neurips2022/metadata.csv](../../data/public_datasets/flow_matching/single_cell/neurips2022/metadata.csv) |
| neurips2022_train_cite_inputs_h5 | 受限/未公开 | 无下载记录 | [data/public_datasets/flow_matching/single_cell/neurips2022/train_cite_inputs.h5](../../data/public_datasets/flow_matching/single_cell/neurips2022/train_cite_inputs.h5) |
| neurips2022_train_cite_targets_h5 | 受限/未公开 | 无下载记录 | [data/public_datasets/flow_matching/single_cell/neurips2022/train_cite_targets.h5](../../data/public_datasets/flow_matching/single_cell/neurips2022/train_cite_targets.h5) |
| neurips2022_test_cite_inputs_h5 | 受限/未公开 | 无下载记录 | [data/public_datasets/flow_matching/single_cell/neurips2022/test_cite_inputs.h5](../../data/public_datasets/flow_matching/single_cell/neurips2022/test_cite_inputs.h5) |
| neurips2022_train_multi_inputs_h5 | 受限/未公开 | 无下载记录 | [data/public_datasets/flow_matching/single_cell/neurips2022/train_multi_inputs.h5](../../data/public_datasets/flow_matching/single_cell/neurips2022/train_multi_inputs.h5) |
| neurips2022_train_multi_targets_h5 | 受限/未公开 | 无下载记录 | [data/public_datasets/flow_matching/single_cell/neurips2022/train_multi_targets.h5](../../data/public_datasets/flow_matching/single_cell/neurips2022/train_multi_targets.h5) |
| neurips2022_test_multi_inputs_h5 | 受限/未公开 | 无下载记录 | [data/public_datasets/flow_matching/single_cell/neurips2022/test_multi_inputs.h5](../../data/public_datasets/flow_matching/single_cell/neurips2022/test_multi_inputs.h5) |
| neurips2022_public_cite_dataset_metadata_mod1_yaml | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/single_cell/neurips2022_public/cite/log_cp10k/dataset_metadata_mod1.yaml](../../data/public_datasets/flow_matching/single_cell/neurips2022_public/cite/log_cp10k/dataset_metadata_mod1.yaml) |
| neurips2022_public_cite_dataset_metadata_mod2_yaml | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/single_cell/neurips2022_public/cite/log_cp10k/dataset_metadata_mod2.yaml](../../data/public_datasets/flow_matching/single_cell/neurips2022_public/cite/log_cp10k/dataset_metadata_mod2.yaml) |
| neurips2022_public_cite_dataset_mod1_h5ad | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/single_cell/neurips2022_public/cite/log_cp10k/dataset_mod1.h5ad](../../data/public_datasets/flow_matching/single_cell/neurips2022_public/cite/log_cp10k/dataset_mod1.h5ad) |
| neurips2022_public_cite_dataset_mod2_h5ad | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/single_cell/neurips2022_public/cite/log_cp10k/dataset_mod2.h5ad](../../data/public_datasets/flow_matching/single_cell/neurips2022_public/cite/log_cp10k/dataset_mod2.h5ad) |
| neurips2022_public_cite_state_yaml | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/single_cell/neurips2022_public/cite/log_cp10k/state.yaml](../../data/public_datasets/flow_matching/single_cell/neurips2022_public/cite/log_cp10k/state.yaml) |
| neurips2022_public_multiome_dataset_metadata_mod1_yaml | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/single_cell/neurips2022_public/multiome/log_cp10k/dataset_metadata_mod1.yaml](../../data/public_datasets/flow_matching/single_cell/neurips2022_public/multiome/log_cp10k/dataset_metadata_mod1.yaml) |
| neurips2022_public_multiome_dataset_metadata_mod2_yaml | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/single_cell/neurips2022_public/multiome/log_cp10k/dataset_metadata_mod2.yaml](../../data/public_datasets/flow_matching/single_cell/neurips2022_public/multiome/log_cp10k/dataset_metadata_mod2.yaml) |
| neurips2022_public_multiome_dataset_mod1_h5ad | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/single_cell/neurips2022_public/multiome/log_cp10k/dataset_mod1.h5ad](../../data/public_datasets/flow_matching/single_cell/neurips2022_public/multiome/log_cp10k/dataset_mod1.h5ad) |
| neurips2022_public_multiome_dataset_mod2_h5ad | 原始/基础公开数据或作者数据包 | 无下载记录 | [data/public_datasets/flow_matching/single_cell/neurips2022_public/multiome/log_cp10k/dataset_mod2.h5ad](../../data/public_datasets/flow_matching/single_cell/neurips2022_public/multiome/log_cp10k/dataset_mod2.h5ad) |
| neurips2022_public_multiome_state_yaml | 原始/基础公开数据或作者数据包 | 可用 | [data/public_datasets/flow_matching/single_cell/neurips2022_public/multiome/log_cp10k/state.yaml](../../data/public_datasets/flow_matching/single_cell/neurips2022_public/multiome/log_cp10k/state.yaml) |

## 所有尚未可用来源与原因

| 来源 ID | manifest 状态 | 原因/门禁 | 官方证据 |
|---|---|---|---|
| jfi_tep_26003087 | failed | https://ndownloader.figshare.com/files/26003087?download=1&cachebust=1790850549810505300: HTTPSConnectionPool(host='ndownloader.figshare.com', port=443): Max retries exceeded with url: /files/26003087?download=1&cachebust=1790850549810505300 (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1010)'))); aria2c through the configured proxy returned HTTP403 even with one connection; requests Range streaming verified HTTP206 and HDF5 magic on the same public endpoint. | [证据](https://api.figshare.com/v2/articles/13385936/versions/1) |
| jfi_tep_26003162 | failed | https://ndownloader.figshare.com/files/26003162?download=1&cachebust=1790850549813843400: HTTPSConnectionPool(host='ndownloader.figshare.com', port=443): Max retries exceeded with url: /files/26003162?download=1&cachebust=1790850549813843400 (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1010)'))); aria2c through the configured proxy returned HTTP403 even with one connection; requests Range streaming verified HTTP206 and HDF5 magic on the same public endpoint. | [证据](https://api.figshare.com/v2/articles/13385936/versions/1) |
| jfi_tep_26003492 | failed | https://ndownloader.figshare.com/files/26003492?download=1&cachebust=1790850549812294400: HTTPSConnectionPool(host='s3q.ait.dtu.dk', port=9000): Max retries exceeded with url: /figshare/26003492/TEP_Mode3.h5?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=00000005001330512eb1/20261001/oha/s3/aws4_request&X-Amz-Date=20261001T102912Z&X-Amz-Expires=10&X-Amz-SignedHeaders=host&X-Amz-Signature=9665065290cf3607f6221d219fbeed2b38902aa2a21ec69cac467e656ae1a4cd (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1010)'))); aria2c through the configured proxy returned HTTP403 even with one connection; requests Range streaming verified HTTP206 and HDF5 magic on the same public endpoint. | [证据](https://api.figshare.com/v2/articles/13385936/versions/1) |
| jfi_tep_26024015 | failed | https://ndownloader.figshare.com/files/26024015?download=1&cachebust=1790850549819016400: HTTPSConnectionPool(host='ndownloader.figshare.com', port=443): Max retries exceeded with url: /files/26024015?download=1&cachebust=1790850549819016400 (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1010)'))); aria2c through the configured proxy returned HTTP403 even with one connection; requests Range streaming verified HTTP206 and HDF5 magic on the same public endpoint. | [证据](https://api.figshare.com/v2/articles/13385936/versions/1) |
| jfi_tep_26028068 | failed | https://ndownloader.figshare.com/files/26028068?download=1&cachebust=1790850549819016400: HTTPSConnectionPool(host='ndownloader.figshare.com', port=443): Max retries exceeded with url: /files/26028068?download=1&cachebust=1790850549819016400 (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1010)'))); aria2c through the configured proxy returned HTTP403 even with one connection; requests Range streaming verified HTTP206 and HDF5 magic on the same public endpoint. | [证据](https://api.figshare.com/v2/articles/13385936/versions/1) |
| jfi_tep_26034659 | failed | https://ndownloader.figshare.com/files/26034659?download=1&cachebust=1790850549819016400: HTTPSConnectionPool(host='ndownloader.figshare.com', port=443): Max retries exceeded with url: /files/26034659?download=1&cachebust=1790850549819016400 (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1010)'))); aria2c through the configured proxy returned HTTP403 even with one connection; requests Range streaming verified HTTP206 and HDF5 magic on the same public endpoint. | [证据](https://api.figshare.com/v2/articles/13385936/versions/1) |
| jfi_tep_26775368 | failed | https://ndownloader.figshare.com/files/26775368?download=1&cachebust=1790850549820197500: HTTPSConnectionPool(host='s3q.ait.dtu.dk', port=9000): Max retries exceeded with url: /figshare/26775368/Readme.html?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=00000005001330512eb1/20261001/oha/s3/aws4_request&X-Amz-Date=20261001T102912Z&X-Amz-Expires=10&X-Amz-SignedHeaders=host&X-Amz-Signature=b493280c5f1891c9030cf30ba42edd69fbc1677b81a9982daf69516fbca6b80e (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1010)'))); aria2c through the configured proxy returned HTTP403 even with one connection; requests Range streaming verified HTTP206 and HDF5 magic on the same public endpoint. | [证据](https://api.figshare.com/v2/articles/13385936/versions/1) |
| mtsbench_cicids_cicids_6_test.csv | failed | https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/cicids/cicids_6_test.csv:                                                                                  [#3dc9a1 0B/0B CN:1 DL:0B]                                                                                 [#3dc9a1 0B/0B CN:1 DL:0B]                                                                                 [#3dc9a1 0B/0B CN:1 DL:0B] 10/01 18:29:13 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/cicids/cicids_6_test.csv Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/cicids/cicids_6_test.csv   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 3dc9a1\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/mtsbench/cicids/cicids_6_test.csv.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details. ; Official mTSBench publisher release of related dataset family; not asserted identical to CATCH/DCdetector authors preprocessing or experiment splits. Reuse fm_data path; columns include timestamp/is_anomaly. | [证据](https://huggingface.co/datasets/PLAN-Lab/mTSBench/tree/b95a65320720154430e1e8e91d105ba06c7903a5/cicids) |
| mtsbench_cicids_cicids_7_test.csv | failed | https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/cicids/cicids_7_test.csv:                                                                                  [#aa79fe 0B/0B CN:1 DL:0B]                                                                                 [#aa79fe 0B/0B CN:1 DL:0B] 10/01 18:29:16 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/cicids/cicids_7_test.csv Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/cicids/cicids_7_test.csv   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= aa79fe\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/mtsbench/cicids/cicids_7_test.csv.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details. ; Official mTSBench publisher release of related dataset family; not asserted identical to CATCH/DCdetector authors preprocessing or experiment splits. Reuse fm_data path; columns include timestamp/is_anomaly. | [证据](https://huggingface.co/datasets/PLAN-Lab/mTSBench/tree/b95a65320720154430e1e8e91d105ba06c7903a5/cicids) |
| mtsbench_swan_swan_sf_test.csv | failed | https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_test.csv:                                                                                  [#a88e2c 0B/0B CN:1 DL:0B]                                                                                 [#a88e2c 0B/0B CN:1 DL:0B] 10/01 18:29:16 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_test.csv Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_test.csv   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= a88e2c\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/mtsbench/swan/swan_sf_test.csv.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details. ; Official mTSBench publisher release of related dataset family; not asserted identical to CATCH/DCdetector authors preprocessing or experiment splits. Reuse fm_data path; columns include timestamp/is_anomaly. | [证据](https://huggingface.co/datasets/PLAN-Lab/mTSBench/tree/b95a65320720154430e1e8e91d105ba06c7903a5/swan) |
| mtsbench_swan_swan_sf_train.csv | failed | https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_train.csv:                                                                                  [#ed89a6 0B/0B CN:1 DL:0B]                                                                                 [#ed89a6 0B/0B CN:1 DL:0B] 10/01 18:29:16 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_train.csv Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_train.csv   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= ed89a6\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/mtsbench/swan/swan_sf_train.csv.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details. ; Official mTSBench publisher release of related dataset family; not asserted identical to CATCH/DCdetector authors preprocessing or experiment splits. Reuse fm_data path; columns include timestamp/is_anomaly. | [证据](https://huggingface.co/datasets/PLAN-Lab/mTSBench/tree/b95a65320720154430e1e8e91d105ba06c7903a5/swan) |
| mtsbench_swan_swan_sf_val.csv | failed | https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_val.csv:                                                                                  [#eb07e9 0B/0B CN:1 DL:0B]                                                                                 [#eb07e9 0B/0B CN:1 DL:0B] 10/01 18:29:16 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_val.csv Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/swan/swan_sf_val.csv   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= eb07e9\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/mtsbench/swan/swan_sf_val.csv.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details. ; Official mTSBench publisher release of related dataset family; not asserted identical to CATCH/DCdetector authors preprocessing or experiment splits. Reuse fm_data path; columns include timestamp/is_anomaly. | [证据](https://huggingface.co/datasets/PLAN-Lab/mTSBench/tree/b95a65320720154430e1e8e91d105ba06c7903a5/swan) |
| cfmi_data_physionet_set-a.tar.gz | failed | https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/physionet/set-a.tar.gz:                                                                                  [#fee9c0 0B/0B CN:1 DL:0B]                                                                                 [#fee9c0 0B/0B CN:1 DL:0B] 10/01 18:29:16 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/physionet/set-a.tar.gz Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/physionet/set-a.tar.gz   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= fee9c0\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/cfmi/physionet/set-a.tar.gz.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [证据](https://github.com/vsimkus/cfmi/tree/e743f4e21c33dde9581421f08b42fa70b7df7449/data) |
| cfmi_data_pm25_data_Code_STMVL_SampleData_pm25_missing.txt | failed | https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/pm25/data/Code/STMVL/SampleData/pm25_missing.txt:                                                                                  [#beb084 0B/0B CN:1 DL:0B]                                                                                 [#beb084 0B/0B CN:1 DL:0B] 10/01 18:29:16 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/pm25/data/Code/STMVL/SampleData/pm25_missing.txt Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/pm25/data/Code/STMVL/SampleData/pm25_missing.txt   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= beb084\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/cfmi/pm25/data/Code/STMVL/SampleData/pm25_missing.txt.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [证据](https://github.com/vsimkus/cfmi/tree/e743f4e21c33dde9581421f08b42fa70b7df7449/data) |
| cfmi_data_pm25_data_STMVL-Readme.pdf | failed | https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/pm25/data/STMVL-Readme.pdf:                                                                                  [#4bed72 0B/0B CN:1 DL:0B]                                                                                 [#4bed72 0B/0B CN:1 DL:0B] 10/01 18:29:17 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/pm25/data/STMVL-Readme.pdf Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/pm25/data/STMVL-Readme.pdf   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 4bed72\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/cfmi/pm25/data/STMVL-Readme.pdf.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [证据](https://github.com/vsimkus/cfmi/tree/e743f4e21c33dde9581421f08b42fa70b7df7449/data) |
| cfmi_data_synthetic_banana_val.npz | failed | https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/banana/val.npz:                                                                                  [#7d0a31 0B/0B CN:1 DL:0B]                                                                                 [#7d0a31 0B/0B CN:1 DL:0B] 10/01 18:29:18 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/banana/val.npz Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/banana/val.npz   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 7d0a31\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/cfmi/synthetic/banana/val.npz.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [证据](https://github.com/vsimkus/cfmi/tree/e743f4e21c33dde9581421f08b42fa70b7df7449/data) |
| cfmi_data_synthetic_funnel_train.npz | failed | https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/funnel/train.npz:                                                                                  [#583c15 0B/0B CN:1 DL:0B]                                                                                 [#583c15 0B/0B CN:1 DL:0B] 10/01 18:29:19 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/funnel/train.npz Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/funnel/train.npz   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 583c15\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/cfmi/synthetic/funnel/train.npz.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [证据](https://github.com/vsimkus/cfmi/tree/e743f4e21c33dde9581421f08b42fa70b7df7449/data) |
| cfmi_data_synthetic_mring_test.npz | failed | https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/mring/test.npz:                                                                                  [#6aa481 0B/0B CN:1 DL:0B]                                                                                 [#6aa481 0B/0B CN:1 DL:0B] 10/01 18:29:19 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/mring/test.npz Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/mring/test.npz   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 6aa481\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/cfmi/synthetic/mring/test.npz.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [证据](https://github.com/vsimkus/cfmi/tree/e743f4e21c33dde9581421f08b42fa70b7df7449/data) |
| cfmi_data_synthetic_ring_test.npz | failed | https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/ring/test.npz:                                                                                  [#130e42 0B/0B CN:1 DL:0B]                                                                                 [#130e42 0B/0B CN:1 DL:0B] 10/01 18:29:20 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/ring/test.npz Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/synthetic/ring/test.npz   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 130e42\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/cfmi/synthetic/ring/test.npz.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [证据](https://github.com/vsimkus/cfmi/tree/e743f4e21c33dde9581421f08b42fa70b7df7449/data) |
| cfmi_data_uci_yeast_yeast.data | failed | https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/uci/yeast/yeast.data:                                                                                  [#1b1599 0B/0B CN:1 DL:0B]                                                                                 [#1b1599 0B/0B CN:1 DL:0B] 10/01 18:29:22 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/uci/yeast/yeast.data Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/vsimkus/cfmi/e743f4e21c33dde9581421f08b42fa70b7df7449/data/uci/yeast/yeast.data   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 1b1599\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/cfmi/uci/yeast/yeast.data.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [证据](https://github.com/vsimkus/cfmi/tree/e743f4e21c33dde9581421f08b42fa70b7df7449/data) |
| tsflow_electricity_nips | failed | https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/mv_release/datasets/electricity_nips.tar.gz:                                                 [#345fe5 9.3MiB/42MiB(21%) CN:2 DL:3.6MiB ETA:9s]                                                                                 [#345fe5 12MiB/42MiB(29%) CN:2 DL:3.5MiB ETA:8s]                                                                                 [#345fe5 16MiB/42MiB(38%) CN:2 DL:3.6MiB ETA:7s]                                                                                 [#345fe5 18MiB/42MiB(44%) CN:2 DL:3.4MiB ETA:6s]                                                                                 [#345fe5 21MiB/42MiB(50%) CN:2 DL:3.3MiB ETA:6s]                                                                                 [#345fe5 23MiB/42MiB(56%) CN:2 DL:3.1MiB ETA:5s]                                                                                 [#345fe5 26MiB/42MiB(61%) CN:2 DL:3.0MiB ETA:5s]                                                                                 [#345fe5 27MiB/42MiB(65%) CN:2 DL:2.9MiB ETA:5s]                                                                                 [#345fe5 31MiB/42MiB(73%) CN:2 DL:2.8MiB ETA:4s]                                                                                 [#345fe5 33MiB/42MiB(79%) CN:2 DL:2.7MiB ETA:3s]                                                                                 [#345fe5 34MiB/42MiB(81%) CN:1 DL:2.2MiB ETA:3s]                                                                                 [#345fe5 34MiB/42MiB(81%) CN:1 DL:1.8MiB ETA:4s]                                                                                 [#345fe5 34MiB/42MiB(81%) CN:1 DL:1.5MiB ETA:5s] 10/01 18:29:35 [[1;31mERROR[0m] CUID#8 - Download aborted. URI=https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/mv_release/datasets/electricity_nips.tar.gz Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/mv_release/datasets/electricity_nips.tar.gz   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 345fe5\|ERR \|   2.2MiB/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/gluonts/electricity_nips.tar.gz.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [证据](https://raw.githubusercontent.com/awslabs/gluonts/dev/src/gluonts/dataset/repository/_gp_copula_2019.py) |
| tsflow_wiki2000 | failed | https://github.com/awslabs/gluonts/raw/b89f203595183340651411a41eeb0ee60570a4d9/datasets/wiki2000_nips.tar.gz:                                                                                  [#d487b1 0B/0B CN:1 DL:0B]                                                                                 [#d487b1 0B/0B CN:1 DL:0B] 10/01 18:29:24 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://github.com/awslabs/gluonts/raw/b89f203595183340651411a41eeb0ee60570a4d9/datasets/wiki2000_nips.tar.gz Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/awslabs/gluonts/b89f203595183340651411a41eeb0ee60570a4d9/datasets/wiki2000_nips.tar.gz   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= d487b1\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/gluonts/wiki2000_nips.tar.gz.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details.  | [证据](https://raw.githubusercontent.com/marcelkollovieh/TSFlow/main/tsflow/dataset.py) |
| taxi_30min | failed | https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/mv_release/datasets/taxi_30min.tar.gz:                                                                                  [#8c1875 23MiB/43MiB(53%) CN:1 DL:0B]                                                                                 [#8c1875 23MiB/43MiB(53%) CN:1 DL:0B] 10/01 18:30:00 [[1;31mERROR[0m] CUID#8 - Download aborted. URI=https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/mv_release/datasets/taxi_30min.tar.gz Exception: [AbstractCommand.cc:351] errorCode=1 URI=https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/mv_release/datasets/taxi_30min.tar.gz   -> [SocketCore.cc:1019] errorCode=1 SSL/TLS handshake failure: Error: 操作成功完成。  (0)  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= 8c1875\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/forecasting/taxi_30min.tar.gz.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details. ; CSDI Appendix E.3; preserve provided forecasting train/test partitions. | [证据](https://github.com/awslabs/gluonts/blob/dev/src/gluonts/dataset/repository/_gp_copula_2019.py) |
| ptbxl_1_0_1 | 无记录 | 无下载记录; access=public; Use 1.0.1 for paper fidelity; large waveform archive. Official SHA256SUMS.txt is available under /files/ptb-xl/1.0.1/. | [证据](https://github.com/AI4HealthUOL/SSSD/blob/main/docs/instructions/PTB-XL/README.md) |
| diffusion_ts_eeg_extension | failed | https://drive.google.com/uc?export=download&id=1IqwE0wbCT1orVdZpul2xFiNkGnYs4t89: File is not a zip file; 2025 repository extension, not an original 2024 paper experiment. | [证据](https://github.com/Y-debug-sys/Diffusion-TS#dataset-preparation) |
| grin_public_bundle | failed | https://mega.nz/folder/qwwG3Qba#c6qFTeT7apmZKKyEunCzSg: HTML returned instead of dataset; AQI-437, AQI-36, METR-LA, PEMS-BAY; use MEGA client or API, URL is not a binary file. | [证据](https://github.com/Graph-Machine-Learning-Group/grin#datasets) |
| sssd_processed_bundle | failed | https://mega.nz/folder/kT91jYpI#97GyTkVVUk97fzs1Oy4nBQ: HTML returned instead of dataset; 6.28 GB author-preprocessed bundle; includes Electricity, ETTm1, MuJoCo, PTB-XL. src/get_data.py gives individual MEGA file URLs. | [证据](https://github.com/AI4HealthUOL/SSSD/blob/main/src/README.md) |
| dcrnn_traffic_bundle | failed | https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX: HTML returned instead of dataset; Official DCRNN folder contains metr-la.h5 and pems-bay.h5. | [证据](https://github.com/liyaguang/DCRNN#data-preparation) |
| cer_e | failed | https://www.ucd.ie/issda/data/commissionforenergyregulationcer/:  10/01 18:30:34 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://www.ucd.ie/issda/data/commissionforenergyregulationcer/ Exception: [AbstractCommand.cc:351] errorCode=22 URI=https://www.ucd.ie/issda/data/commissionforenergyregulationcer/   -> [HttpSkipResponseCommand.cc:239] errorCode=22 The response status is not successful. status=403  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= dfa5f4\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/cer_e/pending.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details. ; ISSDA application and data terms; SME subset of 485 smart meters, not a public download. | [证据](https://github.com/Graph-Machine-Learning-Group/grin#datasets) |
| ganf_pmu | proprietary_not_released | proprietary_not_released; Official repository explicitly says proprietary and cannot be offered publicly. | [证据](https://github.com/EnyanDai/GANF#datasets) |
| catch_bundle | failed | https://1drv.ms/u/c/801ce36c4ff3f93b/EVTDLHyvegpEn_Oxa6ZiuFIBjTsKk6m9JldUqWDqvrVCnQ?e=P2T3Vc:                                                                                  [#c08a9d 0B/0B CN:1 DL:0B] 10/01 18:30:36 [[1;31mERROR[0m] CUID#7 - Download aborted. URI=https://1drv.ms/u/c/801ce36c4ff3f93b/EVTDLHyvegpEn_Oxa6ZiuFIBjTsKk6m9JldUqWDqvrVCnQ?e=P2T3Vc Exception: [AbstractCommand.cc:351] errorCode=22 URI=https://onedrive.live.com/:u:/g/personal/801CE36C4FF3F93B/EVTDLHyvegpEn_Oxa6ZiuFIBjTsKk6m9JldUqWDqvrVCnQ?resid=801CE36C4FF3F93B!s7c2cc3547aaf440a9ff3b16ba662b852&e=P2T3Vc&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvYy84MDFjZTM2YzRmZjNmOTNiL0VWVERMSHl2ZWdwRW5fT3hhNlppdUZJQmpUc0trNm05SmxkVXFXRHF2clZDblE_ZT1QMlQzVmM   -> [HttpSkipResponseCommand.cc:239] errorCode=22 The response status is not successful. status=403  Download Results: gid   \|stat\|avg speed  \|path/URI ======+====+===========+======================================================= c08a9d\|ERR \|       0B/s\|D:/aicoding/IIA_benchmark/data/public_datasets/flow_matching/catch/author_bundle.part  Status Legend: (ERR):error occurred.  aria2 will resume download if the transfer is restarted. If there are any errors, then see the log file. See '-l' option in help/man page for details. ; 10 real-world families plus 12 TODS synthetic sets; alternative Baidu URL https://pan.baidu.com/s/1W7UoAWKZjoukSZ74FTipYA?pwd=2255 . Author data splits differ from conventional five-dataset MTSAD splits. | [证据](https://github.com/decisionintelligence/CATCH#data-preparation) |
| dcdetector_bundle | failed | https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR: HTML returned instead of dataset | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| existing_mtsad_smd | failed | missing or empty payload; Existing acquisition audit reports valid payloads. Do not overwrite; author-specific splits require separate verification. | configs/datasets/public_sources.json |
| existing_mtsad_msl | failed | missing or empty payload; Existing acquisition audit reports valid payloads. Do not overwrite; author-specific splits require separate verification. | configs/datasets/public_sources.json |
| existing_mtsad_psm | failed | missing or empty payload; Existing acquisition audit reports valid payloads. Do not overwrite; author-specific splits require separate verification. | configs/datasets/public_sources.json |
| existing_mtsad_smap | failed | missing or empty payload; Existing acquisition audit reports valid payloads. Do not overwrite; author-specific splits require separate verification. | configs/datasets/public_sources.json |
| existing_mtsad_swat | failed | missing or empty payload; Existing acquisition audit reports valid payloads. Do not overwrite; author-specific splits require separate verification. | configs/datasets/public_sources.json |
| existing_ucr | failed | missing or empty payload; UCR is present inside audited TranAD repository; check archive inventory before asserting all DCdetector subsets match. | data/public_datasets/audit.json |
| tab_author_dataset_zip | 无记录 | 无下载记录; access=public; Official CATCH/TAB authors data archive; contains exact expected ASD/NYC assets if inventory matches author scripts. Google Drive presents a standard public large-file virus-scan confirmation form. The provided form action is a permitted public download, not access control bypass. | [证据](https://github.com/decisionintelligence/TAB#quickstart) |
| cifar10 | 无记录 | 无下载记录; access=public_direct | [证据](https://cave.cs.toronto.edu/kriz/cifar.html) |
| imagenet_32 | login_terms_required | login_terms_required | [证据](https://arxiv.org/html/2210.02747#S6) |
| imagenet_64 | login_terms_required | login_terms_required | [证据](https://arxiv.org/html/2210.02747#S6) |
| imagenet_128 | login_terms_required | login_terms_required | [证据](https://arxiv.org/html/2210.02747#S6) |
| lsun_bedroom_train | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/fyu/lsun/blob/master/download.py) |
| lsun_church_outdoor_train | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/fyu/lsun/blob/master/download.py) |
| afhq_original | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/clovaai/stargan-v2/blob/master/download.sh) |
| celeba_hq_stargan_release | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/clovaai/stargan-v2/blob/master/download.sh) |
| celeba_hq_original_reconstruction | manual_reconstruction_required | manual_reconstruction_required | [证据](https://github.com/tkarras/progressive_growing_of_gans) |
| metfaces_aligned | manual_drive_folder | manual_drive_folder | [证据](https://github.com/NVlabs/metfaces-dataset) |
| domainnet_infograph | 无记录 | 无下载记录; access=public_direct | [证据](https://ai.bu.edu/M3SDA/) |
| domainnet_painting | 无记录 | 无下载记录; access=public_direct | [证据](https://ai.bu.edu/M3SDA/) |
| domainnet_quickdraw | 无记录 | 无下载记录; access=public_direct | [证据](https://ai.bu.edu/M3SDA/) |
| domainnet_real | 无记录 | 无下载记录; access=public_direct | [证据](https://ai.bu.edu/M3SDA/) |
| domainnet_sketch | 无记录 | 无下载记录; access=public_direct | [证据](https://ai.bu.edu/M3SDA/) |
| officehome | 无记录 | 无下载记录; access=public_direct | [证据](https://www.hemanthdv.org/officeHomeDataset.html) |
| celeba_original | 无记录 | 无下载记录; access=public_drive_quota_exceeded | [证据](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| eb_processed_trajectorynet | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/KrishnaswamyLab/TrajectoryNet/tree/master/data) |
| eb_raw | 无记录 | 无下载记录; access=manual_repository_file_resolution | [证据](https://github.com/atong01/conditional-flow-matching/blob/main/examples/2D_tutorials/preprocessing/1.0-embryoid_body_data_to_h5ad.ipynb) |
| neurips2022_cite_seq | 无记录 | 无下载记录; access=public_organizer_processed_available_original_kaggle_gated | [证据](https://www.kaggle.com/competitions/open-problems-multimodal/data) |
| neurips2022_multiome | 无记录 | 无下载记录; access=public_organizer_processed_available_original_kaggle_gated | [证据](https://www.kaggle.com/competitions/open-problems-multimodal/data) |
| ot_cfm_toy_distributions | 无记录 | 无下载记录; access=generated_no_download | [证据](https://arxiv.org/html/2302.00482#S5.SS1) |
| ot_cfm_sb_ground_truth | 无记录 | 无下载记录; access=generated_no_download | [证据](https://arxiv.org/html/2302.00482#A5.SS4) |
| ot_cfm_funnel_10d | 无记录 | 无下载记录; access=generated_no_download | [证据](https://arxiv.org/html/2302.00482#A5.SS6) |
| rectified_flow_toy | 无记录 | 无下载记录; access=generated_no_download | [证据](https://arxiv.org/html/2209.03003#S5.SS1) |
| rectified_flow_reflow_pairs | 无记录 | 无下载记录; access=generated_no_download | [证据](https://github.com/gnobitab/RectifiedFlow) |
| domainnet_clipart_train_index | 无记录 | 无下载记录; access=public_direct | [证据](https://ai.bu.edu/M3SDA/) |
| domainnet_clipart_test_index | 无记录 | 无下载记录; access=public_direct | [证据](https://ai.bu.edu/M3SDA/) |
| domainnet_infograph_train_index | 无记录 | 无下载记录; access=public_direct | [证据](https://ai.bu.edu/M3SDA/) |
| domainnet_infograph_test_index | 无记录 | 无下载记录; access=public_direct | [证据](https://ai.bu.edu/M3SDA/) |
| domainnet_painting_train_index | 无记录 | 无下载记录; access=public_direct | [证据](https://ai.bu.edu/M3SDA/) |
| domainnet_painting_test_index | 无记录 | 无下载记录; access=public_direct | [证据](https://ai.bu.edu/M3SDA/) |
| domainnet_quickdraw_train_index | 无记录 | 无下载记录; access=public_direct | [证据](https://ai.bu.edu/M3SDA/) |
| domainnet_quickdraw_test_index | 无记录 | 无下载记录; access=public_direct | [证据](https://ai.bu.edu/M3SDA/) |
| domainnet_real_train_index | 无记录 | 无下载记录; access=public_direct | [证据](https://ai.bu.edu/M3SDA/) |
| domainnet_real_test_index | 无记录 | 无下载记录; access=public_direct | [证据](https://ai.bu.edu/M3SDA/) |
| domainnet_sketch_train_index | 无记录 | 无下载记录; access=public_direct | [证据](https://ai.bu.edu/M3SDA/) |
| domainnet_sketch_test_index | 无记录 | 无下载记录; access=public_direct | [证据](https://ai.bu.edu/M3SDA/) |
| metfaces_png_10165-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_10176-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_10923-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_11212-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_11213-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_11244-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_11249-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_11267-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_11274-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_11274-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_11284-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_12940-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_12980-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_12994-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_12997-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13000-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13044-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13045-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13049-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13051-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13053-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13057-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13077-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13093-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13094-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13096-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13097-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13098-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13099-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13100-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13102-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13104-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13105-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13187-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13196-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13204-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13209-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13320-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13324-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13325-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13327-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13334-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13335-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13336-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13337-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13338-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13341-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13342-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13346-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13385-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13393-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13395-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13492-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_13494-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_14305-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_14310-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_14314-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_14351-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_14356-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_14357-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_14399-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_14400-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_14402-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_14404-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_14419-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_14483-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15025-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15038-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15044-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15058-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15059-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15060-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15076-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15078-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15082-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15087-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15090-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15103-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15121-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15127-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15128-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15129-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15134-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15145-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15146-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15149-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15155-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15173-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15181-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15183-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15191-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15196-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15197-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15199-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15200-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15230-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15233-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15246-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15307-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15541-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15586-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_15588-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_16588-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_16593-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_16594-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_16687-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_16710-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_16724-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_16859-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_16860-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_16861-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_16892-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_16906-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_16958-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_16959-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_17075-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_17447-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_17514-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_17515-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_17544-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_17564-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_17660-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_17783-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_17901-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_18588-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_187997-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_18811-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_18830-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_189433-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_189434-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_19014-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_19015-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_19022-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_19039-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_19056-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_190710-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_190711-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_19206-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_192770-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_19376-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_193815-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_194002-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_194115-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_19718-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_19732-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_19732-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_19816-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_19826-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_198952-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_199624-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_199671-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_199787-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_200413-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_200656-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_200656-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_200656-02 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_201893-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_202718-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_203620-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_203625-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_20517-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_205194-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_205651-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_206321-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_206337-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_206344-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_206345-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_206566-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_207202-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_207365-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_207397-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_207810-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_207814-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_208538-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_209063-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_209314-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_209336-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_210103-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_21126-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_211511-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_21209-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_21473-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_226426-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_231010-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_231051-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_231788-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_232047-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_236026-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_237070-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_242005-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_242069-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_242336-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_246269-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_247009-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_247367-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_247559-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_248501-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_248727-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_248801-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_251089-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_252637-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_254628-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_255045-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_256431-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_257433-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_283182-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_285877-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_29150-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_312237-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_322368-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_322377-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_33284-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_334004-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_334075-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_334259-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_334468-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_334760-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_334861-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_335536-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_336419-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_336461-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_336628-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_336776-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_337096-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_337172-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_337364-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_337431-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_337433-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_337439-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_337473-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_337475-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_337624-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_337666-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_338518-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_338611-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_338722-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_34018-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_340378-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_340852-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_340874-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_341620-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_343419-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_35155-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_35655-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_35705-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_36107-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_362527-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_365524-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_365958-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_366158-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_366342-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_366763-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_366766-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_366767-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_373813-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_376602-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_377272-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_384104-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_384338-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_38777-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_387846-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_390225-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_390242-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_39193-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_392402-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_394418-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_394422-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_394429-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_394431-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_394432-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_394435-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_394540-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_394547-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_396300-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_3970-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_3971-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_399819-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_402823-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_404459-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_404499-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_405971-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_408076-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_409000-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_414244-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_415530-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_416895-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_416900-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_416961-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_418351-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_418353-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_421473-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_421652-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_42488-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_42547-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_428625-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_428755-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435576-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435580-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435585-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435585-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435592-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435596-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435597-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435615-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435625-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435630-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435631-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435632-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435635-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435641-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435643-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435645-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435649-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435650-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435650-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435658-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435662-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435664-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435687-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435688-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435690-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435697-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435716-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435716-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435716-02 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435721-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435722-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435739-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435757-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435758-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435761-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435763-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435763-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435765-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435802-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435803-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435804-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435807-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435818-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435819-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435820-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435829-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435830-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435834-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435835-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435837-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435844-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435844-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435856-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435864-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435870-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435875-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435876-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435886-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435892-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435895-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435896-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435897-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435912-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435921-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435941-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435944-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435945-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435946-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435947-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435948-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435949-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435950-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435951-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435952-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435953-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435954-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435955-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435959-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435960-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435961-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435980-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435984-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_435998-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436001-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436017-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436024-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436028-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436034-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436036-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436038-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436042-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436043-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436044-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436045-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436046-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436049-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436056-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436059-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436067-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436096-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436097-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436101-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436103-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436103-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436107-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436109-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436124-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436152-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436179-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436190-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436210-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436211-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436212-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436213-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436214-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436215-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436216-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436217-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436219-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436238-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436239-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436240-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436246-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436253-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436254-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436255-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436258-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436262-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436265-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436274-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436284-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436289-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436295-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436309-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436310-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436314-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436315-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436318-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436322-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436326-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436333-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436334-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436334-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436337-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436338-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436339-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436341-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436343-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436377-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436397-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436398-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436404-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436407-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436408-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436421-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436431-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436432-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436433-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436434-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436436-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436442-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436446-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436480-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436489-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436491-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436493-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436501-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436515-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436520-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436521-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436538-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436539-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436539-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436542-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436543-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436544-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436545-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436546-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436547-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436549-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436550-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436551-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436552-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436581-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436582-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436584-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436586-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436587-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436591-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436605-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436610-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436616-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436617-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436618-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436621-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436623-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436624-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436626-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436627-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436630-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436631-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436638-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436640-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436641-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436642-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436643-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436649-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436650-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436657-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436658-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436659-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436660-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436661-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436662-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436663-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436664-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436666-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436667-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436668-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436683-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436684-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436685-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436685-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436685-02 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436686-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436688-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436691-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436692-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436694-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436695-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436696-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436703-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436705-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436706-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436709-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436711-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436714-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436715-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436717-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436721-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436722-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436738-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436738-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436739-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436741-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436747-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436769-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436771-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436789-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436790-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436796-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436797-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436797-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436798-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436819-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436821-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436823-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436824-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436825-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436834-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436838-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436838-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436846-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436846-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436848-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436850-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436852-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436853-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436855-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436862-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436871-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436872-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436873-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436873-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436887-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436895-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436910-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436911-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436928-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436928-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436930-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436933-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436941-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436944-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436948-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436955-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436956-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436984-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436984-02 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436986-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436987-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_436990-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437004-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437021-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437030-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437038-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437038-02 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437046-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437048-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437056-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437061-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437062-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437062-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437065-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437067-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437073-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437085-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437086-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437087-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437089-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437090-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437145-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437151-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437154-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437158-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437163-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437164-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437166-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437167-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437168-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437173-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437174-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437181-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437181-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437182-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437184-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437185-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437199-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437199-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437199-02 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437204-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437205-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437206-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437207-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437208-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437209-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437210-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437217-05 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437232-03 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437237-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437239-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437248-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437255-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437263-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437264-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437281-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437282-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437324-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437325-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437332-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437356-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437357-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437358-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437359-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437360-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437361-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437363-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437364-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437365-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437368-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437373-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437374-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437385-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437386-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437387-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437388-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437389-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437390-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437391-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437395-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437396-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437397-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437399-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437400-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437402-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437404-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437405-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437406-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437407-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437408-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437409-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437410-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437411-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437416-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437417-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437419-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437420-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437422-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437424-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437425-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437430-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437432-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437437-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437439-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437446-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437448-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437450-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437451-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437452-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437453-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437455-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437456-02 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437456-03 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437456-04 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437456-06 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437456-07 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437459-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437463-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437465-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437488-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437499-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437500-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437501-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437502-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437503-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437504-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437505-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437509-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437530-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437531-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437532-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437540-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437579-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437580-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437580-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437581-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437582-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437583-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437584-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437594-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437595-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437598-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437599-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437600-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437608-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437609-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437610-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437640-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437644-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437645-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437646-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437649-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437662-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437677-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437679-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437687-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437688-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437689-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437698-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437699-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437728-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437734-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437744-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437745-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437759-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437766-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437768-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437769-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437769-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437778-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437822-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437825-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437831-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437837-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437838-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437843-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437861-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437862-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437869-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437870-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437872-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437873-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437875-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437878-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437879-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437884-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437887-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437889-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437890-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437893-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437894-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437897-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437898-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437900-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437904-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437905-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437917-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437918-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437921-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437943-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437954-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437957-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437962-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437970-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437971-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437978-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437978-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_437990-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438001-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438001-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438009-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438011-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438016-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438020-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438032-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438033-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438112-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438139-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438378-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438379-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438389-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438407-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438434-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438434-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438434-02 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438544-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438585-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438590-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438603-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438615-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438737-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438740-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438776-03 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438780-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438818-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438819-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_438844-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_439273-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_439327-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_439337-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_439405-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_439933-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_440464-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_441111-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_441115-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_441227-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_441227-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_441230-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_441355-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_441363-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_441365-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_442749-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_44613-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_447125-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_447125-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_44833-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_450749-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_454621-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_457723-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_457781-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_458953-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_458972-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_458982-03 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_458983-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459015-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459015-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459039-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459039-02 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459052-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459052-02 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459053-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459054-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459059-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459059-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459071-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459073-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459074-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459081-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459082-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459087-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459088-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459089-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459090-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459106-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459126-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459127-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459129-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459213-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459253-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459254-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459425-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_459965-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_463607-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_463798-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_464127-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_464128-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_464596-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_466084-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_467786-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_467997-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_468720-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_469927-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_470265-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_470266-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_470273-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_470315-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_471036-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_471203-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_471432-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_471460-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_471844-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_471845-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_471853-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_471853-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_471908-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_471942-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_471947-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_471996-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_472342-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_473849-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_473877-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_477317-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_47916-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_479673-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_483-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_485-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_485551-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_486-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_49122-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_49252-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_49345-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_49567-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_49892-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_500-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_50339-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_505722-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_506078-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_506080-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_506089-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_506090-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_50959-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_51823-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_52701-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_53156-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_53531-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_54077-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_54080-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_543872-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_543900-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_543951-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_544176-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_544334-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_544336-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_544340-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_544454-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_544685-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_544690-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_544704-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_544709-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_544723-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_544752-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_544824-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_545147-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_545477-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_54593-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_546289-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_546290-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_546291-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_546304-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_547257-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_547565-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_547677-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_547698-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_547716-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_547822-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_547826-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_547830-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_547836-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_547844-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_547849-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_547850-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_547851-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_547852-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_547860-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_547861-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_548277-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_548585-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_549228-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_550773-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_550798-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_551104-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_551144-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_551291-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_553763-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_553922-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_558169-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_57612-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_61719-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_627328-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_631063-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_632807-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_635819-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_638084-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_639240-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_639853-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_639909-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_640565-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_641257-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_641594-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_641684-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_641721-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_641729-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_643303-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_643540-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_643553-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_647704-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_647705-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_64897-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_654327-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_655420-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_655421-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_655744-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_667926-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_667985-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_668082-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_669033-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_670765-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_671454-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_671454-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_687513-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_698625-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_700948-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_701989-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_703199-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_705481-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_705777-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_707805-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_707898-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_707898-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_708301-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_708308-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_708341-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_708350-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_708353-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_708365-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_712539-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_726543-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_735054-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_735091-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_735098-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_738456-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_742415-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_742432-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_742434-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_746938-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_747607-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_748947-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_75914-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_761570-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_767842-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_771936-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_773251-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_775309-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_775312-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_78434-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_78569-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_78666-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_813585-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_813594-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_815550-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_815550-01 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_816873-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_817504-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_822597-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_822737-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_822756-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_823401-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_826758-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_827515-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_828499-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_828503-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| metfaces_png_828517-00 | 无记录 | 无下载记录; access=public_direct | [证据](https://github.com/NVlabs/metfaces-dataset) |
| neurips2022_metadata_csv | 无记录 | 无下载记录; access=kaggle_authentication_required | [证据](https://github.com/openproblems-bio/neurips_2022_saturn_notebooks/blob/main/notebooks/getting-started-data-loading.ipynb) |
| neurips2022_train_cite_inputs_h5 | 无记录 | 无下载记录; access=kaggle_authentication_required | [证据](https://github.com/openproblems-bio/neurips_2022_saturn_notebooks/blob/main/notebooks/getting-started-data-loading.ipynb) |
| neurips2022_train_cite_targets_h5 | 无记录 | 无下载记录; access=kaggle_authentication_required | [证据](https://github.com/openproblems-bio/neurips_2022_saturn_notebooks/blob/main/notebooks/getting-started-data-loading.ipynb) |
| neurips2022_test_cite_inputs_h5 | 无记录 | 无下载记录; access=kaggle_authentication_required | [证据](https://github.com/openproblems-bio/neurips_2022_saturn_notebooks/blob/main/notebooks/getting-started-data-loading.ipynb) |
| neurips2022_train_multi_inputs_h5 | 无记录 | 无下载记录; access=kaggle_authentication_required | [证据](https://github.com/openproblems-bio/neurips_2022_saturn_notebooks/blob/main/notebooks/getting-started-data-loading.ipynb) |
| neurips2022_train_multi_targets_h5 | 无记录 | 无下载记录; access=kaggle_authentication_required | [证据](https://github.com/openproblems-bio/neurips_2022_saturn_notebooks/blob/main/notebooks/getting-started-data-loading.ipynb) |
| neurips2022_test_multi_inputs_h5 | 无记录 | 无下载记录; access=kaggle_authentication_required | [证据](https://github.com/openproblems-bio/neurips_2022_saturn_notebooks/blob/main/notebooks/getting-started-data-loading.ipynb) |
| neurips2022_public_cite_dataset_mod1_h5ad | 无记录 | 无下载记录; access=public_direct | [证据](https://openproblems-data.s3.amazonaws.com/?list-type=2&prefix=resources/datasets/openproblems_neurips2022/&max-keys=1000) |
| neurips2022_public_multiome_dataset_mod1_h5ad | 无记录 | 无下载记录; access=public_direct | [证据](https://openproblems-data.s3.amazonaws.com/?list-type=2&prefix=resources/datasets/openproblems_neurips2022/&max-keys=1000) |
| neurips2022_public_multiome_dataset_mod2_h5ad | 无记录 | 无下载记录; access=public_direct | [证据](https://openproblems-data.s3.amazonaws.com/?list-type=2&prefix=resources/datasets/openproblems_neurips2022/&max-keys=1000) |
| drive_1OcNc0YQsOMw9jQIIHgiOXVG03wjXbEiM | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1PMzjODVFblVnwq8xo7pKHrdbczPxdqTa | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1-QvPvJFzBiUTrtHycQaw_XYlDaFjQbk_ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1mmaVfK_Ukp35z58YV9S4h64nJyDkM5N_ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1nGyc0xiUGvLyU6ztyRLqqhhKms0QO2ZV | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_10-r-Zm0nfQJp0i-mVg3iXs6x0u9Ua25a | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1uYiXqmK3Cgyxk4U6-LgUni7JddQnlggs | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1e_JhpIURDLluw4IcHJF-dgtjjJXsPEKE | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_162gZw7v7XBEhrsLv5Bq5r12M3bGCdvwr | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1NHwCJXUDXNXWq9gO3mopJbTGJfC6A7b4 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ETJkCImUSk11p-p9B5CAnduszoNOnSiF | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1RQH7igHhm_0GAgXyVpkJk6TenDl9rd53 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1SYgcRt0DH--byFbvkKTkezJKU5ZENZhw | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1d3tAbYTj0CZLhB7z3IDTfTRg3E7qj_tw | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1SO_j0apQAh0lvh4-t0HOQnwuYiAVyGNh | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1TVppX8PzyEtRbJ45ooWic4HeSXHAQ5kj | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC | 无记录 | 无下载记录; access=public | [证据](https://github.com/liyaguang/DCRNN#data-preparation) |
| drive_1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq | 无记录 | 无下载记录; access=public | [证据](https://github.com/liyaguang/DCRNN#data-preparation) |
| drive_1_Z7VEXY1vlzmj8uk-U8JOlWalyRZXzda | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1my20t6D-BG6hEa-LhxFopQAShDAG5HBy | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1tOgE3yMsthn3uHIsgQRB-3rB3ojJldcf | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_10wJBgNJgKDzrWE3qznHJrdyQ0ANJm80L | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1vSQU4vonAOYpZ70k30SEOPKKQMEvTiPY | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1XCt31gFbB7iIAwLBD_gJLjZJLZjYQLLg | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1kmhQZNYYuKO5uI4tSPaU9QLOWsvMaBru | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_19ZVhVAfrsb6pnHg4Pn1zJipztevKhvpQ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Yi7Ssy8WcF1aM_ZEJE__1LHuYhTl1yVv | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1JmwOOKdshlCUEbxmAHr_Nh37LKWfO04g | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1A4sA9TIGZK2mpvxirfPmzcYQIlRGgeRI | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1BHi82HxZ5A6lnDApqRH2KOnjOFSbvnj1 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_14N_MSTZJiD85IS5mlo0zBu3GKcuGYtIf | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Lkxkz78n1VcF-hJpmi7B7mBqqpE1p1OP | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1UFgAjAY1XYNRu3OpcMEJ8-svOWAvjv-u | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1SS_Iz0uvAFRP-naXBXfCE50YjKaDO_7e | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_10nfevHuV6vjm1A1L1h9NJPCoOaqGM5lA | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1CxRO-PySrzbxGfAo-JSPMCGYN7gz7Tnw | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1KfGIDMJE50HJTqtzzMNetj8ZevTRiNqe | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1VHEsYzMfHL-bFTXxWRKeonmQ0TqwhsoQ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1MYT46_yZRY7ZTz2SFYC7WTs4zFzh00so | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1MLY28wEK7gFWo4WotJb9KgUzZQt8IcoO | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1OrInecE3qNwLhCIqllNAdUY_rmhQ2bnV | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1PQndrNteb_2VDfjGseQ5M_g4AGftr-5L | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1XzGScG93qTY_IUkE8GxK7WgsYv68AsvE | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1mrhvGgVOdC77SlymwNJSODtiYeTjhTbQ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1fTYZhnden18RbJ9WsARwl9HuCyuxD0KG | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1aUCK0tEJBxdhH3sS4SSph23DNTSQvzG2 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1jorGzYQPzA6rk4MH-GBaC-BjyJXxgazg | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1FhrUAySwv-PS-P94pedlgjOcCHSu4a67 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1gAVqw8WL5SgRwL_xrgWl41G2XCMNUXoB | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Zjcy5vBgVW10SD_uMMtZ6MRsqVMtMXUE | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_19iMcYpaLh4W02QSFWVkc7_VcGyr96gYK | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1VNFt2ABjzaOAa_uTKx54ujaVgVv_qG74 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1XxyKVux0cSPGkPpmweog7LAV-_4ziLW8 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1lWB1jC3Rgt3VNfyNBCHtSUfCPQiD8qn6 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1lSrkT1uKpRxScRsm2WNgzvuUWa6uiIkr | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1bg1-Cv3PWSiLCXWd3Q9YZ-B19Cy3ZkUB | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1rP0CvsRWyOv6jtEaN-onAYFZeUCPw4V_ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1JLuCxNV8rUio8mRyOsw9kB2XCkNc5r-L | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1PuMS2uZyWeoOYoH1Eya-3Dc0IHBbFCGC | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1UN91-zGMyVvSvikHSiimsA37Ci7fFwAT | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1TbIdPjKianlVOg4kjB15hvbuRGvq3DNe | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1NuUiBp8_-jYVeuhZ39fMzXeqq_oYC87C | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1EFivVitAxHensOwpz3ZUUVCtLw-PWNiO | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_12n43A9F_5FwgNeSrnrT8NcbCA9kuRr0_ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1iOy-sUdI7CcrflhwhWmsKgc7-W5HrM6M | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1NRug5Cc9EL_bGdRYgJcsjnXSFxlVX-bv | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1u5fs1CbJyPshUtfNCF_bW-n6TA3TAmRl | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_12qlnHdybFZsDL2zqO9KY7XiWhTwTqOp4 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Bx_gfKxob4gdlU37FQ7oqIp6mIp4PSIz | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1hQ1-Uw5I6SXQ9WXcpU6DrzpFhzGz2o9o | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1nnVl-uwVURvRV9KVRNhnplafgxBlZn9E | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1tZ92yomgNo9B16yOrlEO1N7L7U40dfTD | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1erLCw61wRjq9DvJByDs3XQyFpFikhgm3 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1gzNDXk7X1HVOGn3VsFMBI1m7vyeHoZCH | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1AR3ujRKWALXyq2n-566IqpTc-U5LIukM | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1qbNx81LQ-X3VhD980A63vzWIBBwimluf | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Sz83_lklI3pY3ZbJTABBYa26doCLktPU | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1zv2APEKaWVeDmcq6WjFY0WsJ2f9ntz8D | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1COMlrlTZQ2mZAa7dvst-wa9-7ildSeVm | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1vFUvNIxJaefG-5_-MyU4Qv1Z5_cBgpsn | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1hrb-Zd_kZdkh-HRsPocWGmWIUpbYZ69B | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_10-XK0QBgx8MERS1iiJDMrxca7r1Hmg8l | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_10WS_xWE2UdrmdM4ZfrukdW9E_Z49L61X | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1q1j5EmSlrvToAWOm5iO49lSUPVgefQDB | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1CETwYuY4bNAKJljQMdU1tFPvR_6vPWXd | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Csd2ieZqGn2lcRZX8CX93oVEGPnx-hm5 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1dFEjsQ4_Z8v2McdmQrC0aZ9wNpLSAq6A | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1z1SEXsNQXafk_A_UcHUdHACWdal3B2ic | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1n8RlN7BBN9DQB5pEa9m1PqFmt8vD7PLD | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1jaTQ6HIixkFSTmU_Rz2QmcepT3ZlxbvK | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1cbpmJBrnm8_1dGFLASVu6-YjPjlSMl9C | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_10blY__V4MzB4D_UhS29sNPZC559FS9ss | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1u-tsHIWVeBkziFot8aQrhIaqVYIvg88k | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1u7ioLCqHaZYlx3YbLXbN3gE3tDvcHJ85 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1X-hINQk3738DMNoYKEwBwF2_xwLtCJwo | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1P-6TtPijKWFpQBOJxzz2FnnwZIJXx5Is | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1vkhoIsf6tbfkcCOfoMSGCHdUtzZ7pX-n | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1HuTAjmZMqzsr5JvPcHOBQgLM5VKJ1E6m | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1pxeaU5VjR0kEdo4uwX95mc2C7CbYwlwN | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1oJCD-Cpz0ret_dzld99q-O0V35uNvPl9 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1bSRF7cvtXtirjf5s06kANNz9tTDo1-uU | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1YfFVqAjYmXuLUsnoD9CJbnPQgyCCeG34 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_17f14nmRm5lT3B_LyVnSM4RaKrejgrHIL | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1wJ-H7hEUROeadoLtyFpqqaKXLBRjREM5 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1rSJZUjEX1Gb3rLddr1VSYoftk-W9si1t | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1-yzeof4LEjBik1fTBvsr5SP8VJrcFuku | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1spJTalICYxBiQ-gCBr2XfWuCODaB9dNF | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1qlWdDce9IaLu9LCq7xf9Bmd85bjhozCL | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_19NB76gExrV2KwI_8bmoGgmo3UAtVykSD | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1OpSCbPYEzp8yixMbG06JcYKBHlGYrvkg | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1qUX06ZS-bzdEf1uwqoTE-miODBbK2M7k | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1OogtABH4WAp3e3X7dTgqaH7OYp3H5xuo | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_17x7Qs3TAryQpXL9j00NJhIyUXnjcbQ0r | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Azu4LnJyuwT5b3jnJ8ZsryKwQNliOncv | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1upAyfdBWRyNZiRDyZ4ym5QVSqQZxTYZZ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1NNEwGMhN5NpE4E7yvthpn17RP-zgvOeU | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Uui0FmIvtjpZ6xWj-4jiYMMWUoO3a_lh | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1mLs9MEihLoHN6mE2oGZMYvAaO-H5eXHs | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1mK5_6dcvIayCuuqCS2mU8igK_rLOXKRt | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1C2PgR5KQkoouufuvKzkPMIaHC7s_GtQ3 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1a_BiwUQISZM95xOzGM5T6NHUkZ1bX2QR | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Bwb3jY06Lqjr3auycOqweLZ89r0rLLmV | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ruf8qTArjpVlrExunyDCDj59Ud3zXZMq | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Kb5YtTPfkoLGf5ZJiVRdGS1tWc9ZmO02 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_116AJam56k2mTHSKUzDF5_oGYnrN2DsMH | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Exl-QSnadIDvWTw1CIUoEW4RmmvDeZMA | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1g9fpeqZjYAbAuBXAnr6bl4xQG8WvkprK | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1fhJ_RWkkTHlBYMWfedHZ29ZoagpncP0d | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1DwaDLBgjrPhmfGot8f9f5rHr2cX5_m2Q | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1G4BtXoEWKK8iaVo3Ovmga6aqkJaBvS_J | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1wedZKAK64-OzQ48kW93w-ODTlNkmSQdh | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_17z2YY6n7302ymTfUydh6zWHa9aDBGx-f | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_15liHJMrxn7c2ElE4IURRALoO2KnfSJ0r | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1tCLJgcx2Kv_nUu_t8TqvDK5rXQ59XbDw | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1SLVPaWl5dj0I2sLduu29VqNhv_BbzLY4 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_163iyOWQe9QKYvIsrT9kGtU0x4ty1oCga | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_16S95mZ0Lx_VT-cKZdZNUqkffiYasDEUS | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1G2mRIVVQYye0v81IkRp7GEYJ1371_CFM | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1-vKaRCrBPTshSvCao4QChO_bMpBoMBFp | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1kEydXT_-WakFPHRJaI7MMo-rAInHQKYC | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1h1OgZ6NYk6x-xo57vLYnnmcoyA4pkoGt | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1jhzIVXecqH7d4l8d-lNgUtwJukP77LVf | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_14p9rgfJrBwNeVbaPQdKmgj87cIp_nJ2y | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ZzCvfY2MUWcC_hTvit9GzaPawKoiDXkk | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Mgc_d-6XuYIM6KS1VC7P5lp9ewa7UV10 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1oA8oGwogXLDNmJE2YZRQrBP6Cq3HCTNI | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_15AwoqFM-uQ4aPwef5ddHAliKfgNLHEnb | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ylm-qujDC8JpLD0QsZa-YlzSwq-jduVH | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Az-a-e4BSSGodWmELO55xseTbAX3A4dG | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1TyKSmaV-d26XhYRQHuigiLawn2cSDdPE | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1VxxxDEv48EY2BWGMKQhlWn6yWkFB11CP | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1PCGWKETqSkrO6tDPSoEfNosmNOE7U8bL | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1OEEW7Gg1OBgCPif_N1kUp2SykHlT0KCg | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1yELigKSZEx4z7Ph7htFwnEDVKJZXoj4z | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_12c4vTL0w9dQoMCzF1-ZNtXLVAa2h9Oux | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1xC8ofV_6iXkrwhkTmMWSrfFlWNRTQHpW | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1rpNBPZSEKr8VtdloWIK_9ZZiSyON4RkI | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1mJUsgR4sUu4clVgLbkxoRuF6Z_Wi3xP7 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1GzK1HJ_t3-Dj3_272Nk8rg1RKRpWycuU | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1tPgHSrcZvmpWaSaZLhjN_724YZI6kNHF | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1X_mUCKu7EK8_ZHz7-ROu-_FGzXYJesno | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1bWuzmmtGwlM54I9XlK5HHDy8IX3eXDoG | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1XXk3h73W8RRJ4-cD7sf0jHchObW5ZBkX | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ix4oTmJhzlNa0-hUGun45_Qv5Qm9LDl5 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1_y9hfacxWplkQx9ALGxym5IBUYFzmrCS | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1LpnmGCHNimW8ylYINKqMiC_XGKKqpg0G | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1eYpEhdXsnoqPReaKsIMgR077TCBqo6GQ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1vVknS1Npsuy9GQ0xKtk15Xh_6Hc9oLAH | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_18pMmBOUegfru7lJWcMcQxqoLMWiiOT7O | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1tT6kqUD9TwMyY1G6hQrmA5Lw8VpBaEmS | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_16FNYDbuuh3LzzkYLmHKmo-b-3wKiyKc8 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1nAmEIP2VX-wjciriNOeMV4UMTuh5Gvyk | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1P_QhnA7Zcv0zPDSJqee-O3jlQnX7bNQf | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1RcKF7V4WguqtV4dQ0yNVMAutbNSNDgD_ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1uqZ3Wen1jsJ7-DRwlhWxKawv2S4dHeIn | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1C4MMsocbxcevvC4GiObKLEc2Dy86vSlA | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_11zYZhtbKMECPWE9PCacyCe-AaqUJub16 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_18AwOijgGJdrNKmTLTIR7him9ENorS3Ux | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1XwA-LVrl3BR9eNWUIq90VLXdGAKN798b | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1XN6Yr0JO4E85p7YpcDBcHozFHqOwxPyv | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1-VO_7LJgf0RuAZ2TvoffhlB3uV02ZHQJ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1PcRuIC7TPXrhvL-hcY86CcbjSEg0TaXo | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1BYuZXnrLlqa9jVhY3RhOJLTf2y85DYSa | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_155JinbeR_z7D6IAl2QiT5EUJEnb9uTDQ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1jkb1eRtx25-n_ndf5BcEEpZyyV_c82cy | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1pfpt6Z3vFc3E5ZhuIJ2LrH1VxpcQ-qCk | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Vo9pDbY6cZY_XWIxcKi6tZrfGdwjflec | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1lcTYRQCd9uR_0f-vF76ST1sY9CaDsWVq | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1LSldrkKMQCh_BKZktrNX8ou7Y1iZDwqA | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1uSehq69CHwmwE1K4r-l5xhH9W0qYDJ72 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_19lUCeWvNIjN3LEdoQHvuwxCxbSSvrytp | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1JwkUMUvMAlk-s5D91i1XfK_vUAqPUWnt | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_14D1putfiem6Ioe_-aWvDNW-A6x2lIZmb | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1TxEkNgNbugS_JswDP0Urse8bVf7IwJSF | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1pltTxxtlvGjigxdAFxu4NBqgbCoWSxPL | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1SuxVXnVdkGCrQxjqBqIMtSJtms56rxEA | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1pXLYjmhc2y95p6bYmHDczjFL2A-hpNSJ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1QdQPGEs1wrmpt_7zellhWoaYTYosTXxn | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1fGxsUDO0Guq22qaRgxjJXwDBhTWGwTe8 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Y_lKaJ5zcEZeI5JGVsZxT738leetCL_3 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_13_hKYaJNSQ04U72-gWm1mEfeg2sPi4Qg | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1cj50OqVnIUtOVhM6cWnM3YAtx7yfkOC7 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1FH_P8OC9zzBOH89kTTNKdmRRnoklxd6U | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1yzjFWCXNE7WaAYjUEiQKny_mdZpNKMuZ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1TTzPumR3BCXO0TRWyR3Eyu3bBh6B-2jX | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1o6n_Wx537ITw9ZLw11vvuZqx5Qvtx-Wd | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1_ER7QsHheQmBuSYagmb63KGDUlB78xL- | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1WKixx2JNRimzJxQFFEX6CccXjYmocM6P | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_11UWZL4Ikt5tmrgnZJhYkatXkQDhfVckV | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Lsh893Z36pXqSqPqPhPijJGSdbKg14lh | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1sc5HJ9QTyVaNEaK2RBmbNVLuqBdIJn7r | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_16alqaIO3-QXhPlHGrl9YngN4PUReiXFM | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1XEHScaZ_xxJ7EvCMRdN0k1CeRvol6m0d | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1oyHLbEEp5bRcd02k_MqXU4WIye2wq2dM | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1gxhAinUBXi5ZKkuh_8JQYxdKSwsoGrvc | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1MWonEXJensrAvWQmWGpBsNepQw6FMxFa | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1vLVt7KiVQNQiX16BcSLkWiMWTrHUEhl8 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1fZaoqmafavWH7ucTNyviNyQX57jo9Tld | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1qvp2A8D67DAL-f7rNtbpPZVYlShga57U | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1RjtFjX4j5CWDe5C7y-LWTs1GkK0ty9E3 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1qLGbMAFiXO5yid_Nbo1Y6x9bcMowyr-C | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1vMjEhIBQHwZyb6qm7yaQyjIrs0ia4_n- | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1DPW-edvF5gEEVmfqo_kiQu46KFMSzPAn | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Z-feRkBQKYj3GHfxtFa4Zs37BmVkqemD | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1yoeSahySlmujTmZ68JoFAdsrxZ2cyvnw | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Gwb5INec1456yO1vO8kMNdF6MTE9QG7t | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1OYNOdqXsxPYq14vlLoy5bfupHhNuFAQi | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1W6qceK1ouYSqsih2wkYJneJv1KgduPtj | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1yQvE2SBFJDdRGy89irrI5Ea5uUXAEY4F | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Q6z6NR7fBDYqnjusQJShT7k3EW9Y1frZ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ejx7Zwx0pycITgd3V6Ps4wyyBte36lY2 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1EgRpq06FmEk5NmLzNPa36QOnmzi-vI0M | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1WXQ6mT-UTrRHOzDBxe7pVzAXzL2-Bpv3 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1yptQeeQkvkVttVuXW3s03_f4AonxdLUU | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1z0vfadRxtQjDzkkLq71Xx6dQkMDiseOE | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_10W0GtL8nFRUlEGFO0xOEK0eTfcoPrazY | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_118NoUEQqlt2pjp2igbik7tcr23INzO7v | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_18iDupYhTrha1pawmpd_DP4vFPVe-zQnu | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Vt-NWApDK82Z55WK3pwJE7ND7erx-9jp | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1qvSJDYl_ZQBoFd0hijW-x5TzlFcGlmcR | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1djCvOv8nzYpkSKS7xBiglLzA2yoAU50t | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1iAggcFuqLJeG3QQrY_gehUNoobAF6LlR | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1S4nA7hoo4pVMXr5QbTZEfh_gkNwQPYf0 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1sMh9nSqxZ3OnVUzhpMF5aaBbtVT9rc4X | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_11qc1oArgXgDOhT6VnO3H_St0Hya0Iwja | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1HXRNFQL50nsh_mIoaPH1S3j-bi6bFkSW | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1AWvfrtqXaxrYIf8t7KG9NK8nn0xUprKZ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_12USxtEAImiv9wm0PEUUaIEiqx2rKzTgA | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1jY2QWJwpwCOiQ8ofU8bXuziLutdLEncm | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1TiPlb7LbszpaQ4IvdDJJ2Hpinw_Rz83g | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1tv2Hjy0ZSKm8emv5LSGZVuvHNtt0daIQ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1SXr7BAwTDEiEUk99DRjnLu0At89hhQEp | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1xS5HmqbrmKdLkxUUs8oOAUN2qzR9lBsr | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1-9QeKETBW4uZeFf-Fsj_Rij-LY0kCDED | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1XBG0f-V8efInP06qkk_fOHFrRpZsf6e- | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_14qaRLlALo1Nd0d7zs-LOWqV7aLuSoLLu | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Y6qiOs2bihNXbfSF3WAH8iiP6C1og_G3 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1M8PimBOd4KkCL9CsbcfllZ0qlaODcyzK | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_15smwVGc6YFP7IYGPmopdqVjaNS3rwu3r | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Gv2JnxoMrayQHQrExoRNxlrEjvPRpMVw | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1pcoUhfePZ-6PJ5QvuaVZmV_W2-LDI183 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1P_MPwskLacgH9Lu8p-8qCBdUpJc6NslX | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1mY5qUd8MJWgJFokq_vH0qcOV-1lg0q6Z | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1vrUhTY-x_Ca9uk69DXVfU--RZPDwn1hI | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1AZNYfPw5VBB-hjXmEXAT_QE68bRLyUVg | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1sa2WoevBO9CnPWNUJ9sFEGfA4G4aZ6HN | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_16-_7zotm6qmu-95arEa-gYwqDszGGi0b | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_16it3bY0FUkpmZ4Tlp1o-rxoSjdeZUYDx | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1V89Zvwa2UeAxYEm-13ovUKhYTAYXH6E7 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1za1mK1WW_jIU9vxEbmtu7WGczH2eSshN | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1xjsqMVJK2NDdBxtw6p9k1GiNEsfK8dMX | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ataJrznpkgxs_Dlr0LpXHGPppYNHxtGJ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_16LQbYtfbLNvTsgijv_0gO0TOXl_5l9zh | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1NouZnb6g_OK8yX67bFGNvfTUn892dr05 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1j_58OiVtE20Fce1ORKAKpqkKr5rdqpk5 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ot742yTQDq5tMl7MFwCOrO3R1EQMvx81 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_14SrpuZebvPP8JvUxp0xJsrfwjqEwg1zj | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1H_Gfrqr9GneGOTFTerUvIrhIg8FOol3R | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1GjIKk6jwg1gQ2_Z2b1fGX3NJoRsN4xRZ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1puZrO3tfVCFFoM9XA-YJSuUx0G9KfFjP | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1vy_0RMBxnnu3a58SmqC_mYvItV4a8S8m | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1M0Mes3CqksZFO50lA9aDDAXgGqthHdjS | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1nD0tWHBv-VaCN0wnE_Ym0rJvS-Ed2dmO | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ZEjEUB0jVnrJyJ-lialMnRGv2n82kgmq | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1oPa_z5XvJ0_rOa7oYTVauZhY6YX135h8 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1EAWrbra-jU4aAH2rOXl8WnovGxrzkfZj | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1r78RF1m2GlB5TsRwb7LvKzzzyafmPhhZ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1VqtnwPbHiDbN0Q6jA02G3YGUlx51v-LN | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_13a4-mPk0HDnDb2BQ5_eWOjfAPUZ4NdEL | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1tsuLtmRjl1JhRSkMe7ggn2KiW9SlnYX9 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ircxNm9Qy_hb81tYeMUsbCn9oCuaMAnr | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1hWf6f8-1OLbu-0b9HiecWFRfKpovwAwg | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1lC1AlVJ5NDeWJGTkBzFz6_8Q9wABzPLJ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1NuqXOhW-ndaStRWRTZ9czLQ9DHNnrlfY | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1RVPVOr5Z0fBQ1CZEXMcsVSTLnl2QZOBz | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1843RAREYoILT-7jJI7VzVu8UYyUY44P4 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1GE62zCYTnNKERzF2TfBciGbJwoD-6TKb | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1IqK_wkxzkJP4p0MjfttbficCyApEvEI5 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1HiNNn6XtL7Ff3V2K6Rt2KEa3TpBDdHiH | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1P7NeIysFLTYYAd2zcSbFFD4PZRi8Pq3I | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_19IgNjclJRVuL17KdTtmwZbULUC_tpBQ4 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1AmqYZ0mYL_FtO-dYqU86jxh8zwTxgBKS | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_14tUSGytztrVkywTYymMRNtCH6cOZUSln | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1saq9MHgurE0kj7hv-R-dhoczsAQDtONA | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1aRwbnppCTrYbH4nCXbaiY6tdITmj8v5J | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1p4Sc42wJp7gl3LmaVCRHuEWhCGKqQtB3 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1LqjQn1E7lUI2E7eQBfdPAfxrhJxowwnQ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ZrevW7KRmE3kZFEn6nHybfN-B1NcyG-F | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1fcFrXA2xebGS1kxCSvwyA1-0Hj73MD1Z | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1VEG_pH2Fu2wDXgeKo0_rLurRr2PA6gcZ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1FuJDfoy1HCIPQ1ZNpgJfRAKt4dizwHt6 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1w-Mlw2QKMoAtyoXheBXwwVvYJwMBGB0o | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1xUBiGkAur2Kf1QaECTq9BE1JA3X7OplH | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1KtuXxZiZGOulj2IwONGOEmmg82JED-FN | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1sn04wby3ufmYCeFDhegowG7M2yvj1v8_ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1qHfRLHniqrADeuPh-OVXC8Vg8mGnj53b | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1kg5euSTrTbMBV7hfXUtbeYTioxTYZaRa | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1pCZyeLCyLl4HmT7IpognS3VN_x0DG0aX | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1uzfTKmHtFZz20BBUyWgIlCGrRNLjrMoy | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1wZQFP38S7J7_FhecGTGOC4VcGUWDiVZ8 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1qZuC34u1Q_1mUvbEgJaneUR7NSMVloEA | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_15L563IoaGTIzBzLExHCTmnA2_P6kMBlz | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1rmcPzFD-0hN0Yo9VhCIzq9K8rtNOGJAz | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1j_Xrz879h_EFZWPUkwAgNtjrD7GgEnqU | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1E_ufZZ16k6j-nzuNAEHIIkzGUZ85LCUa | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1KTpkKnRHPjnJh1dRgU-tt8xq8ElDalHX | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_13fBKWes03kd03tCeZlwNIQqexs8EfGZZ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1AG83lInMAi4pnvvKKGdQd4er2ug7zPpc | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_14brIJ6B8v-BiwM3JvfgADWOO4y0hyvqq | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1TuQgLJeY2S-1LN9A4mTIdV2rptxP4yBV | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_16LG4IXQ53yK9yEH49816oMe8SJM9DLiu | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Ib0JLrzDGM13Bzf238aA8HLxTN4VGHKE | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1LQS92mDewq0sfAn6fq3adnqjBP-9sq5u | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_18jSNtnleyo-RtndtP3kyD5_Q15axuU3n | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1tKqoO8qX-6p-ddjsIoNBtUvce2oF9n2z | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1fuDJ8Q8YOjj6Tzn_Bs4B7axv15vuZhzp | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_12eYwsguOzYK9kcHtBncwnTrJiABKnTjt | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1FezRSA5aqw8IcvJO0B76VvTkjxW7C3Or | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1nrPJx0snm-GL9n2VCMLZQ4P9GGL0c-OQ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1jZTduPvUGqOKak56KT_AwsBEgJHwEVr7 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_17x86ua6QMwxWIH9n6ek1cJza4Hfnjr7B | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1kzZMXnUCS1MZyZ0DakRx1whIjC6YncoO | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_17unFEEklMZDVRS89q9KW0Sem02-lbnM0 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Lzs80CiluWL5BQ3G2QzU50zuRYVmeQir | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_198BnQuQL2g8-ZRhm7VfOUKUZXBMaYdoO | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Y61qsy8Ebb5Op4AyvX__dljUFbFfeH8V | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1efs41XVquwXWUKcjoIxEgC9oMcm93wEJ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1jmrxNY2LK_5c5WVfnLvNj19pD3e_X_bS | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1mBJsNCxdQ8HsnwGvI3W_Y9qj1YBl-bxj | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1dH8hKrdq2fm_LpxJCX5VLpXRsZ28d9H4 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1g9zDd3-3HK5d-GqPYtp4GZ_Tiaj4_0yQ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1QRBPrROHc_iAHIdRzpWoAbVbzF93KsSa | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1_vSn4ewkT7y1FrZ_auHu_Wv3cDuzzVJi | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1LZfZ2CAWpQI0Xlk4gP7iBR0jyHpF1oLa | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1AhOL2g8a4aIxBxc2reojV-LzVFT4B694 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1fATuS2qHTdyyAZ9BCKCv0_yDcBqt16Yj | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1p-h1z-CUx8fEefFwrQT0xxKemNfyZO0S | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1XI1it2S0gvDWFLsyJdNwpuqPDnpSf2dM | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Lq8LB6Tg-Vp_5NcfW2H4Cg_EbXVr1ycq | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_18GSrLfyjvKEULdr8BhRkz8JZg1qvVmxA | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ecXAaD7ETb5As03UABnaFmoy11gmZ3jK | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1tDzAAlD8ZIzK7UjPw-sxLxDVgSfYZL1y | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ptb3fyn-QxOziuLfZfVM2htridZJUXHa | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1y2JylqTYb2_a0JRIuucafkHaaW9Mz4Vs | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_13d7lzlUF7emAjLR6S0cyYFGH_YttcHcL | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Y2JiLv_3I08mK9vfYFnf9WyjKyDXc12Q | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_18vpSgKRJFI9BhsSpJ4YSPhKpAMih0ooj | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_10FYVrvxRnDL2rhFiXAuOIGGoqWWSUDIO | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1KlGDSxJDGnQH6Zu3Cx3v8OdzwwDArlRZ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1wmzqHMSbKbZ2fBx1EQj_owHuZad1S5xW | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1htU9IBTF5tL0fVNBCauP5_s2oVZC34Zm | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1WtjLlmxrHxgkJhQN4laOAN3f64-I27mS | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1B7YbZ0LhLB1-dRN9UpUu9gZ_P1iYETxz | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1GUs2iDcU8eGvznXKF4zhf6wZ1PFt7NO4 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_19upP_piqJcaUf-sie3nLuAnXMzCzra7g | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1vHJcQTifONWqlNpWyeYZcToJ2YrnfRx6 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_16C-lyb969Z_09W5I22xU_e95sMJF5H_5 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1KwJNat5D49QhuLQr6PQvmZPZWVL7gU5W | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1c_vpGkuZGIK_w-5KEF4oa0Gu73cv6Ptg | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1CbpIbU5KlkIF6DMwLscLqpE6TF8MtLTR | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1gFsXPiJlsvfi7sqhW5JCi6i97gdzuO9e | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1xmVsK0oKKof1E4kjIY3PbJC8LDAFaRim | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1vUEveVREubnCxaq5bagamFPEoyZrM9xa | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1o8a9n56E5EBHbErZQlEEOZXUetgtTrSv | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1usGJF4cXQBtOm95qnUd_MnXkApXfat8h | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1AU1nwg1_GiIs0kxdV7RdEwWKcjwy6S2e | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Vmh4gBCD4MUMpUDEiMZ-6Jd8YFhYplJB | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_19qanvzff0SU_F1r5qN7AVQXMjMqG60Bh | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ZSJuijpSKHji7TK4A0vAc_5Ar-yLEJ9O | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_14_dGspfTK_BwTmTYMU4PDjOPhbuj8vP_ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1yypRZIeKBgDs4dnm140atPufL9Z1zwUs | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ZCM9IAbPo6NwB86qkcDDHd2t1kZfG0hF | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_14wtyNJC5WQL9KVIZ0T8-_OgpfwVsjRNq | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1JDlNEDhzBDO0fVoBUyHhnbfOqzmt3qp8 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1OajcTD6IR4XuiBfyIy9Eg-DO_qJcn5RF | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_16IoRe6fQ018lX_r5QsSUq3Vkom34Ajg5 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1C5qw9dTjMDIEt0WoZ2VDIyTI48JBstrp | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1DOH1C8zfMhWjBXPnFkYK_jgQaN7EbsW4 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_11P1OfGqTLvjYuidScUmA0dyUQVlxYaAl | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1bE5phUvmrZ56eZ3QDBZBuRn982vmvPi3 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1b_Vq_hOMszKwiEh9eSdFdjuT0dOWTC1_ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1JjenaRRj8F_XwHNZX8Xn2nzqfpeyMEn_ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1SHipJ67Ar5Za3SD92FHBuIZd3JXYulAq | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_13s_UJP43MJ7cZxLCnCq1DcXGVMy8-AiB | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1zn3yJlOhIiougqM21APCCK9O_ziTgaGd | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1BueXl4Rms9vLHmESjlcl6PmtDMJEWA7_ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ZzCfvaPRw7ahC5O9Ie8XZca0M_klj_Qb | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1EGklFFwWWRNJjiisYEu_qnGiwqYK-WjR | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1MSHsdqW1TdQttkfrvwC54IkeMLsvILgh | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ZJAvmyXbt6cC1B8Jkdg4qwgbiqsgBaIM | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1FBxIcL5jYNv9HxTpDKr8FZT0316y2XCW | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1jOAfu_0lGLeLd5xyekCwm8ZQiQ_5RbFF | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_14DEqyp5ym7MCZImgF7Dy7QGbqEfNsI7p | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1xL9GP6Y0kgAmxpujk972g309x1Dt40pO | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1qpslV2aavaN6MD4N9YXOIG6UkjfDL2tJ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_18Jb9K4aaHPVUwIvU7lwJ1zlILUvAD0wi | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1uLX2bKLSJl5edEYNe4DkijuglI-zKLqC | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1QmNeKLIQfsskry2Grn2CPsZOkvbfmLM0 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1p8vU8i5ZyCbcnXSO-W_y2HsomZN_t4c_ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_16yP8bEtbPHRkdNHpRCtSlUHki0fkUG_3 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ch6GAUnPMv3UjUokKJ-8Wj7o8iSMLyFP | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1e-aA0GtztS39sjLEbhzwjn-akKjHnh4v | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_17hnFAtfuaZLC-gKbh2OOmxO3SSVSQDyh | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1oCvI-JcWbQ8R2r45xkTIHB_OLr9jXw10 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1WAaAcnu0HARssxit0tYHoN7z3HAVsd5N | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1-zOr-kzmcxE5FKXnpC7bS4DeXSdnLYwx | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_13yYgJEKU8UHvEdCjJ4XKEWIC9UitessI | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_15DMcGLsaY2UNpJG1KvzT9YKMeNp7_T0I | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1zPGS-RjZOJE9six0HeU90f6WEc_NfO6T | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1qdy_dWlVhurjM44sd2_Zj3h7g1G0vwg9 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1SLjgGI_JTI_vI7PSnkNs0GlvwmH-R9Be | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1i8x7-Y4ff6pzaAJbZFD5kUFWU97zuPnn | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1UOk5KN1pL-a9hs9-0vTztrM9XtxQ6QRh | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1dZUFLG107b6EPND3G_S5_bG71fw0xRU6 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1VTrBnU6T4_IgsCHQTaRQHv3EgMLNN1OW | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1YBy133LC1kLsdaoZroPQNwReeA7buIIl | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ivkijdZh063-WX_FieLp-4ZetUqqBygV | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ZMCPVHeqy1nRqquZ4qp2stNnsYGpbJkt | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1u2YZstVFCSnm2X3qeFUg6ySI3AHWWTHN | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_133rOwXxDY_iGddoknuAahHU0JJQ-5htH | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1IJ8OK3V8-C40zSPyzFdrN-BVY5bLadkG | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1fzFJLqbKT2BrXBgZvd-ytB1jM082fyO5 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_19DXmfZMUR_m05yKJ6YC20akRm2EJj1Zq | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_197NvCDCT_z-FX4BVANyvRyiRJSr_8HSu | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1relGjpVASbPQoOOq6pb_BFmYse4KXXp_ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_12zsnQZ2bLo51Dh1jA4DGNC3n2ie55n1B | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ew1I1CR3KIuBNmERjZn3yTss21PU16u- | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1d470xTiA1JWVFh1SXMaoSBDFcmQAM-qO | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1f4PAXh9F0RZTf0R6_YPDBV0nmX5yMfta | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1mh5MrWVUr7B-CDn2gBfcw8iURGKwrBVy | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1HeCaqkkv4m88EV1XBi2V6Z9pnDBhVZZl | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1m8XldeYRRNfQE1zdJ8F5Xdp8CKADJAf0 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Ld7vz_zKGnD8QV8no-o6QX-1BUQ6wI4v | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_15oVL2lQZz9PnrSeTHzUmAfffU2vJ6RBQ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1f8b7x-PX5azFvyXmUpEdfOKUyRef2mox | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ZzsRAyZMkb5cn9VvbL3L4BI7_B_3cmzw | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1gDp30lvieN1MaBx5rPc5-QKOMmgDRsj0 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1sMAU-k3b9GBWGa_qgg3nq1XgQc92jtbD | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1oY8G_ihH0TvMQbfRdktkPI5TPrj9yBbs | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1I6ZjqoIYboDnG6ZH3pAxRi5n64a-KTPs | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1SNvSfnxobmiAaXAyzbRbfS2KQ5WZ3fbf | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_15bw1TrmO54DnU_BKMLW1-921cXRp4_p7 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1IV5aJv8LDVvjDGkK1-KvqIWyBgTEx-mR | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1y_ItdLMHc8bb4NCOFIwWeP0w1Srv2F7F | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_14PyuDUXFhkAX4iq-RKLWErdB9ID3YbAJ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1u6TW19IZ498oil5ZesCdgs4QQNvPjkyL | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1lQ6yuAVBu7i7ODAEnOsjcu1sTJoQQhr3 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1sW4WYliD-xr_nCYThaagSKCSbldHnGRc | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1kirVYu_c6Pu0nt1rX4n6UFFWn5vJLbsc | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1XlWIXX488akj5VQU-Ao7Bv0K9_R4Z1b- | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1nsdXI6Tl_JSq9IKIdnB07HKN3EMDsSpD | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1GnUlf-yhjP3_60ROY4U-ncDVeX0-89VE | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1MmlmOhsft4DFtUgpCq_7PBkVZHglxA7T | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_15l2M5g09RSAiACSyAVIAqQdvkFrWbC0S | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1aseJRq81CaKWIEfZAEKx_IOHtmq6s5Kd | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1H7MibZI3uw0Gfm929Rgvf7X2c6YJOCur | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ALzzMVdgIgyAisEs7c2XdHtRmYVPOHGB | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1xHH3ObxaCz0u5fAtU6s-u8_uNPQnLXWv | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1fLpQX71qllZNJCLbNgn1C0uOmg-4iXq9 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1XbjfuvexBT-9H4iBJ9I2sSC1_PgCmpQd | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1tPHnZLk4dlP2IqJSj7McRJm4haRrihuF | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1QS9wxSM_JFdGYeR5_xIsGshTZy_Wx5RT | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1iMT9gHt1lGY_WGEYCFGBz5xJvxrz3-nf | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ypssyM_WsgG6fx2qz1ZOhmRm9Tu2RhwI | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_19b-c0pNzlxVvUIUqKSUk6GgRsHuc2vRO | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1iyujZ1x-d18XvjtcFVSbV_KaJTk0ulDb | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ojUG4RMoNhEO-qCaIwg5lQmW6AO0O-Vr | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1dN5mo0_Q8bMCieky2y-n6LYrdBMDr5cM | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1qoh_Kdfzctlzmu6OP0l5pMGeiEclBNX5 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1RLjVYmoZNABn_TQMZp8ZA0gnt7ws_AWu | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_14r2z3Ka6l99P3a8IoF6dnui18ZBzNI9C | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1jc0wX0CekV6W4W_260-sBsNpDhEyNs-e | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_14ofAKMgMYx1syRHqxFNQ3YcKFNh4Yg4Q | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_18I0szuN7KG8aj6x7UB3EIaNos6NYNVIt | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1CEeJdlsZc_17o_6wJ7Ct2iRq4AipYqIR | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1YcjvZFdOgRuo8PfZ5Nmwup1GYj_BLQRw | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1YdIHPyjP498TlTQJKC23mrqPJHa19tUZ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1LRHzpCeTCuXPHyHaJ52L_SEmOFhU-oKt | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1p1ROdd2vSOs9-hpWPp0C4AF2k990H1M5 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1uVeJP_PoHL-Rr7tHdyoqr-1Cu18YDdMZ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_17q8dsroJEJuy9Wd_f7xz7MJgkBiRCiMl | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1PxvmtZtgblT-rbuxCVoNHc0qFEzO28z5 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_15QVYoUJGc9PwPxr4RIc240MYWLJ-WXhX | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ggcNcqYOqZw31GuC74UnKePPRiSsnWmD | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1U5WHtRNypvkA1X4a4yq_dNwfFzwN-pMh | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1o5WmA9DjGxR0pXGVWKapzM1Twl3braxQ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1HYgpzneyRnw9q0P8zcOYRY10V0JVqUK8 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1wP2QzEVOFg3rNo2C5PT5lplvCdSv5E0o | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1PT12c2QVLKdc162NROjAy0NyHGgOTQq0 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1MyynOgDCTwGOQbeIfoOj88IY1S4M7u0N | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1lNnon2hIk19Xj4S5QLkTMhnQF1GyNoVP | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1JV8epRxJZv2cUkOIzgFSCbpsiWkE4dOa | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1VlafMs4AiNpcJs5mKk9wI-avHBXg0gzJ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1wrhn4SiqyiSk4YXM6xNmkvqnafmNQqdQ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1olwBuw4-bM4_3u4L2c3wJiyWMkD9zLUm | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1XnsAvtVUGC1Aai_oCCBqQH_F0ON3wKar | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1-7mI3AGVL1Jv4eY7k2sf11l1lkMisvyS | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_19Ui7-wYOTVat_8DFv0AwYFGQIswSYM6P | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ofI_wZ5gWBMXUpSE19JgkRrPCSiNCEo_ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1tUWfW2fT9WUCtO8m62v2EhXnxj8OVbBG | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1TcDzhpMWMUodBw3lNFDJtMSJBHAA_yIG | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Bsfrmtkl1wQtNgWw7_iPcziF9yXLxerJ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1EBy-9Dmg3fpUEt1Ah3XjwB3Ubwn4orbz | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1OzmYY8L-exBZOEUk674NT97UIybc1O4F | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1bI_6L_ADFBx9Q0fatqdq72VjIAlhzvKf | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1BwVRogdDJgRGvUSgpcNIFPHZAJa6Pzsc | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1T8685JMJxtbw76k_cmc4gfeVjgpZs4Zr | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ISkXt4EzfolVbogUiGKmb_Xdn_WaVW6V | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1xcYIMOkigW--eA2fJFHfDMvjOK3JHX1o | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Srimg8SnTDD-vdArJfPGqWZpVHxEN9eR | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1V86awzl-ntjMHz3fW2bxsQ4USUG8y7q1 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_11m9POizww8kUVsYbML0ZW7NA_0jrUYSh | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1uqaLDyYkzqpkfwmBgLDBZEb2g5Rnlieh | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1bqyUF2Wb2dF-QWXfikyEuvTowTPJzhAB | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1CxKApX3vtk4qcc6BtDLZd9QLBnmmnSAX | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1jKrmoTkkQmlr4m7x-N9OhWCxA74WqiAh | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1vadd-L44mdSLcf89yiNnPk5Br5TVdvDt | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_17wQQux_b3YfcZnIHQQa8l9z8wm47dTKA | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1N3ftapKRmx4cEOv1rt9XHMzAO7zwwWUe | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1IbMfiI9BJpTYRZQzER37h3HrjW6C9-Ot | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1CTiHANnB4prCx_-HWOuKh2Lu08ASXnbV | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1w-y2izen_-l89zv3dX4cQxwSiw2wlFkY | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1sT00dDJkKbD02eMkz8-nNawd1V_eo8S_ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1vqYz6sBsmn6v93m2_0R5CGt_O8OYFdWo | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1r_75aybrdLi5JC_vWONyW69s9kDWpT2R | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1mam4KVchGm1Ac0lerFybXHKyc6aVw54k | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1QCDQq0GUs_1vo54N0Jz5h-u-8sfBeT69 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1rwd6pV3uahozkdMktJI1W70IxzPulU_W | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1VYCmqJHCeywIMjh-2JP6aDCUNnsX5rEk | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_16rmQspt5v3dejB9i2TyXna_ao7HieOmd | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1jutwn7zOpqo02jq7d2epUG4PWXJDQk4B | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1cS7uYM39p21BUwMOPb0eF1kEpCPyJ6wq | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1CNDecv66DVAj-UcDDcBGyG8UyBpKniSu | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_17B2vmLIt1rC9gBzJrwbrl3t-MJx6RCee | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1mUqvFsd-eElebhGam46c08Ztc2dmLQ7g | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1mCWx6SkMp93ZmCo9NFTPPo9fxh3srWz7 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ee-7ku3M0nGKt27248VajRaXg7gTIuDq | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1vbJNqMrM8h212W3801xFGrATVALRiFSj | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ZCwXd-44IUlkYVurVw-BpjexfbOIbOMb | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1d0LUtjN38O-KPKFw4R9coIFbrs27uy2j | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1L7c6Yw-8EvuhFT_q88Y-0yZDIjTfaBNX | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_14pcB_CSZWflVAd1JngmCZIpw0G2f5qGw | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1fPYxrlbirH6j11Sgp5MXdsOWKBaMcsYf | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1PGefZmdhu07Lx8a9rLTQXBUcvhs1EZ2X | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ql5SA2Hav_A113aczhizHQOo05loTjTO | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1SCvXvrmdfxPXjX-e7zhom8t_OIZXNS7l | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1V_pz0QNn20X9QrPhoYqpja6w70L-m_Qi | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ylvK2nWhxTMkmlKsTGf9hngSP0_dLgOy | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1RhF4ayp89y0AFmnc42GdC6vjJ0OPNplW | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_13xVG-V9QVyJyIC4cN6RSqmuKBSyguEpd | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1AIMB_1mwomxIxdlUkRltN8MbgSy53p3j | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1VtJT1Sgjo3rU6YdURT35pwxGVzM3VoF7 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1O7wYO5gWt6CVNkHRo2ZanHOEyCjodt6q | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_13FM1M7HdYk11j808H9q7KL296z8NCGvK | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_19VEjCLxwuDiMvz6FcDTps1dftg7RK00Q | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1XFWTmiFpaRBv4ZXJBYvtHXib7HntaF67 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1dOEEWE6N_Rj3aivKT5p7ygujIIVFmbwc | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1wVzscBkLvtcfn2kekp04luIx5eZElWf5 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1igEkiEZjSgmTXqSD0BZ2Yq4qqDVGLFlp | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1cQ47WOhuzIhwvltPul6-q-f1RuFRHhsX | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1DdeoPc5hXOo-RUQkaWqEaPpTUdafzvVM | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Ls4AREyXaQjVoJhjNfinuYJrPLzQueVY | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Ssi6yHLq7EHyRNuVG-MFpRhcvw4yTiV2 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1zsN3Y38etRbMQZ-c-1nkoa7lJ6a0ag38 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ECPUVmUWZYXgW5hbIDvMjEmEjyr9MNl0 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1bbA7B048_bFX5EZz1qyq5N_-wLZXbq-z | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1cKXHAge95XfXp8TJMEYX6CDBoFcfBidp | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_14m8c3GE4DeI5RZt_2eV_TkEZiTGiU0TH | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1V0cIN60TPk-I5fhqIWPZdUm9POnvxHQf | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Lr5OD5odSCWp6nkDE7lEy4N5DFLvBhnQ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1bTQwMtJ3rpRKOEl6SJ-nB7w1bsbt0yiQ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1X4yEAhvPqQ13yNOLrViTktD_ty-toOjf | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1h4OS7vWNoFhHHk2yFIe5tr2fVMK3Tqib | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1cShKu8nXK7yupnyrRQjAKp1I5WkDEeTG | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1fVFfd7OBghkdsJ6xNZQudz6cAJZXEpB0 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_16M1lw1St9AHpAj0oIYOMObcSgCf6pjp1 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Vp3dNI4BysKsyV_MMNYfc_6jYd3HvtaT | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_17wNIKF_DR1bioaE0euJ_G8ioig9xYnaJ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_19d5jxbWsdQ5gszMOZoaGZN3Ep66PKe5o | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1PrK9snsyskU7r2gP08CFc_TfWMzdp0Hv | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1kWkkBDN9Ul4VhphzlHaijaRiZlKL0nDT | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1odiuWwczJxkKV3efcliyjG5-JFPG9MrG | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1obcBhP_86sMrppD_uCyW6IQZmF4UDvUa | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1f3llt34ph5xbGzt20_kdykAPD0zz92NN | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_10tu2fJb8j2dXYA622_K3EdTBhZC8ZcOS | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1XNWWeYqDGU3DFM22XNaH0ornZxMsjUu2 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1l5l-TYwGHNp0LWferqb9up_KeVrOUmkH | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Vgpuitak8Z9r9Hr6bNxz9bTClQe-JLCB | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1-cWRUDG8Vixh2AtWNKN03zAJbpS5Gx4R | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1s9XRpAab2cmqSg8plmVZlxQcNjmAbP90 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1qNIJp8kiKHShHoNqAivWM-wh3cTCLu7U | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1qGQ5X44S9_RIf90mw0lt5e01IsA3zwsX | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1IY6sI7ZRWQUBkIjhpcoJgUqEdCd8kng8 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1eLvFjCq2U3VFjaz10siz8777Vlf3syj5 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_13pe8AyCW-wvs5eB78qZRgX-jFZ5S1ZwL | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1mE_co4khCMvtS4-6wc8Bj66bmy0tkyyk | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1fTCpIHM8kIhD4SppYyHPgw5pUoBT1pTc | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Dlu2IGLK3ljrpj9rjbRals0AAV5z2JH0 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_17qPwKB9MJgq_yEvCNwGSC2VHK-QJt-ik | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1HdTML8FLmrzZPhDKo9I4pWpocHL9MpRy | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_11DOrHbuJlDE0GK54Z98FtI8KBVnkrTGM | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1bl-kHMbhHpei5a_WJ-cZfjktbVuEayPg | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1j5UaPKMxJZFPRFnPDDmfCii4nMy0g5FG | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_17pBKYSHybL5NfbWiLoJ9aWlnN9vK0a5H | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1cGGrmHgHqsE_rulm6Bvw34rFxwICI5_l | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1NjvjYkYv3Z-_JxT9VZoGV9mGScWZ6v69 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1xCVJZodyG62hjMC_r8HwMQPd2gitVX1c | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1csUjRFBMYx7snOjl6j3BiSBSM3I2HWpD | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_18o1CMZGubntmL_MkSKKt8Ndc6CExZeWW | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1x6fJh2R76qHJgEdZf1JkuSFEJWjDKF7p | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1zCmF7EfCMmcX3dq6SyaoE-pIeJW7IQUO | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1KmZLlMtFfWPH_aK7TiS7fC6yD46YMkhY | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Xmec1B1T2tXSh2y5TxlXs3OJFfmoD81g | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1sd7SiCtVD41jxUDr4nn6mqJj8hixA15O | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1-oSVr0BFmtFCcYlfqm0xMJJ5IPlza0V4 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ShwcFpnRGUAPKQa1dzucgz7hO6QfOF7D | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1OVKLfnYCQ11t-H1xSI7b9h9g_OHSVyec | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1o4boxpQEm_BgqXpTKsbEXKcdBdX9RqGf | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1B3gmv6VNCIA5bipj3V6MU0fDScT11QTf | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1R5q6DzK7qIj5Y_KwXICQ0XvXxIZ2XPqQ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1UfYO7R01RAPmqDqOR2MlcneejwP9TlLj | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1bGqaoAR2T19KqNgdUsfJumhACIfM4JRh | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1x7vNRC91gvIbU580aGpmiFw5Kqzoqpne | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1q3IFFXRnbs7eplEhyLqBekwxKP_9oCj7 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1EQH1f5M5iO_fBncado1p2LmS0XIOAMgc | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1WO_s5HuEf4dhNd6qvd8Pgdeie4DHnM7M | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ChNvDHgACNr7MANKnK5SGTPmQuyoAhzN | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1gVLJHjIAy658l4Mk3Zuc4Q7Rl4F1EbXe | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1gbtfnq60q81tSyadi60BM68oUl7YM_7b | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1hrg_pY1LqRrn1xusvPBDccpbHEy_YiaV | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1m0KYuJcT2d9zWBggOYgGOW0OFjHUav3b | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1YdAXzycfbqw2YW1Q0ay7yrnEXDQK7vow | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1u_8DLJzwwJTc4tZiwJegh7cy0G0I1KVx | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1A1gheTj2WhqE9j6z2-wlES6koqYNCB0c | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1oi6fvFXQapD3VdX1KBT6nq1eSH9BHVkE | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1x1hFe6fl9N08mj4-HtYgqMV8XfdPm9sG | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1St8gAkGVC7uKXOf-m8Nm310JEY69K-NL | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1lVkql_p-Re5rQ5UGhzP4nKdimsIdv2QH | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_10pOkKF-vfU4OsQ-EygG4QNAUZJMcrSTL | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1cJeHKXXEC5S5dwwWcrMdZ6_1c7fbd0Me | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1b5W9JWT6rvcbD-E6Xe2azfmFjIAtd1b4 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1mErDZ54Gl__G_HihUa550f2PqIVTV_sA | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1uTObsbpYC3x1bnjyrosxOyerFcK2rrJd | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_15U2WHomNnjRjQgcaH_ExK1nJRlMssmw7 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1L0o9ddt967SF1uwmWCGnVwi0bOKuD1eD | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_17riRcMd6WjDmURk4HD9zi8XYVSTawRVs | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_16aygIkNMJKEnKVvhDLThhZx2P8u8WTO6 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1kBgl6Kgf6JVPGrAztCdnt4pll6c_ANhI | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1KPyJ2TNjroE0tbU3PfNifB1SmCGaY04n | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1fGcVgQd-U-T7TUGgoKygI_d2hBVrAzBq | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1B_aB4IV5NAv_UnpgRDS6s4BxyJgnGo48 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1xBifxBAqsEN1WDeBoiTHzKoVE05x1vxm | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1CRBpUozIEgGgp2FDVyYdVsyAPN9mokip | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_14Dv3jZrhW4jG2YnTB0Jhmes7OC3hXdaf | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1MZeiF0LeGJ9BfaNlhBS3Jv44cqk2xT6a | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1-pXyWqWYkeVEECqSITsXUe3jTFS9NBeR | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1d0I0Ge7Wyc_cGkClaY7lUGBGNah1Mt1E | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1-e-7dQvGAV3TwjCv4i-orFlBYOQwtWVu | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Rx56KevrTs7KF1B9maWDEwtLjapI8rWo | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1B32ArYW0gp51KWaWo-oqrqNQUg30heHM | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_197GS0tBd7xI9j3AlFJxRXPQTMp4_Pr24 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1xMlCT3eleQjXMzVd42ycp4LFvLmZC-0i | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_16VAd_oeqriX2jSRXSzUWlmF2PSXjmfUq | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1VFz3nTDBpIuwZKZ_BctuHzxRgnolQCOC | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1vAdYCZ7Vb38XkeKTTOGlpkcZOlSjOeUW | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ruMwcg8-y2KJPlue2fIHtSKYbzKqjnit | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_19Iq7SWja31YdU8dvNAzkfqe75HAcGv3C | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1j_kGSWbqHFCEvFSo83V_dwWuE0bYmZAX | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_19f16ERaGbvFFgOUxz6UjoFyIkvialSLD | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1fBJ-1xTNRLjH85JuXQT_viHNkf-sX7M8 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1gv2BwJrp6BZX3ZJQrw39t6KuBSwLL2ju | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_10WMdtVm57MhBSC1bVILtzjaIBUow4xVM | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1gsrM62MtsMKvQbD3mdLbb9LKCQCtYb7z | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1T5rn2yFcxaiXVOvWKo7wmlJETnsXfWFR | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1VrRme1KIfbOf1_I4jgQGKgzgjSDihKwg | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_12AaHG3XBY95AEbaV4mzwSSLqd8-xn-9s | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1UySZfID42AFyjPSXuLi_H8wOoBmCppSP | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_10OCvA6l349U0oiOgqf9uVUVJWPJJ4kpv | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1rNvKX8uoMJRzIj3G-4CZculNVtyMh--5 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1_a7q_R5gXKh_sb-6U5Z3ksgySO13vbYK | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ema4WKwJALPjrQ_vcHJ5TcUojt_4E4bg | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1lU00YDWl1MGxN5laFsUmmAj2Z2gNXxsd | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1JuoUtdUM6P9Wkjiq0gaosyv8VGmkplsB | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1GxyXFJMD-5tSIBEOk-zyfgPnuhRPvUqV | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1OyGx5zrSLVWPmDLgqMZTInxyV00BXQyN | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1kRoImuIPee5dE-m9MGU3q2E-wNMwPkoF | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1J7RgppGGyJuA4ycpa7duX-Iea-WfgDfS | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1MPK04BIc9pZyjGrog_Wrzgjig7f6vet5 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1SpJ0WBwjkUyG9Z4d6eNMf38fjDE7uCkN | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1u1lpDYNwR65gXlm2eV_N1n8r7J2z2OlX | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1FRzu36xZpLx6iWr26wo92jl7Kdx1Qycd | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1LBackPqEIOvePYdVLdkjfGwY5SaWZb2r | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1TqBbVlHxi8HsdjiJgU_q5uUD1IwCzTA3 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1D_Mnmc2bqd2H_meN5Lf5fGmuN7KEsd4s | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1tqLF0yPo2qAz6E6QNMrqPsEbUGZgGLXm | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1tdJfkDTDjqu8F-ZXOEinDOr5pmBSbvsw | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1YrJYJnzpl4D6Mey2t2zvRIOSksAr58jL | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1_UPPfYNUFFQu8ZqJyuK9h-oQ9llRRGys | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1FSXFS9YdHQMRj5IIwkNRY7gOB3RO9POD | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1NcXi9L97yMdlV77O0AWrbn27PYXzBHJw | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Tpl-1i_4KGGAauKoatoFhSbx4aoUgh5K | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_18jyLiDJVz8MTasMW-_LBqypoIPWc6R0B | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1sGuW7F_n6qFOBdmhCrlGFoq9BDHq9-e7 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1faWLqFsCpBxFUwGQZj9tNZPyrK6TgY80 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1UDe3T7YbVzh8l-Jdp9ZqPcattWS1Mmh_ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1e18EJUFgA8clmf3Bw-TMmJLAP09mP1JD | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1F2aTA-PnG6sylFxfTXbr4jlOIIbk31pG | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1wQO-dhb9sd-Ra8x4kP-5EXDRxCaA1fPq | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1OjoOT8Uphq4XOJvdPOdurvbzMpdT-RS- | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1dYzeyZZgjLNH7eTYFdusq7ye7pvgvAlv | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_10_2cmGxur7NJLEzaTG-r8E3E4-bo9kk9 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1kQh-WZse0wCJ9G-0pR4hTVS82_YQQzAX | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1WCOA7268KQ6A9AnbUZLeZu7rtUgh1aMZ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_11yXrHk164boyU2HggFdRzHcv-gdb3mFt | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1pT-AftNlCmCCcz0TSZbjMUpyEvw0cctA | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1VEV7iTQG-HSDmTC4I5XKmE8e2Mr8AIUd | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_17O9C9VMdkuVWlfG4S4Yr2vZbxDP7Toh6 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1LLTCHqaFGmBIVHSF-5plrac8zec2n3Nt | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1DIQrMB1IeWMf-XuLAUz9PEFfTQdG9STL | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Jt967hGbf1nKkI68tjkYvZ60viKpCrit | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1O3sMNkYl_EMDY-rEjOe_pB0a-iUUriUb | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1bD67UOBQ3GHmzYsoPXq5vsTmM-iRAYZ4 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1KbHDG-FOHgF1TQmQcw1ztIerVqSWJQoM | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1d_GTydrLdOU1xxB_VQxC_cCLIlTUKIT7 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_17l_MUj9IlqGoURgnXAFYMNw_H6rIfjsI | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1g7rvC4q4T07_5Xz81KIIVqDzv0Nl0sEB | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1m55nsDVlgcx_taJ0MSXDuvRbMC1-WiAA | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1jDIFslzYbeb9wUbduLFrWWWAWZvlEnET | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1pQmSvJkFnEyeA_NjTiGLxIq1a5qhGmSr | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1ztb1o0PQjcL5Mbz4zo7j19OmNQEZ16QR | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1S9fAAGY8eKy3HzTrXVuiVTwHpf4lqJ-c | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1wHmGZBf6m_lVFpzldf01xqBDRJILrz_F | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1YD_yyrXoRIBqMyF3ovZiI--Wb7mWCY7b | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1nCH3UJQmNTf1W5id5n1PYXG2q35GpakA | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1TMmJZJBDtSiU_7fUtM3kmhVbYEgdibFx | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1vipHkn2Xa1MMCpnsiq5YFe12vD0mZYzX | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Zy13pqI2cuMpvis7FheDIhtSilgl8n-L | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1gTOF8lWDNqXGj9fLllWsurLTmD6eBpVP | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Lw18ClwGbcDOMGiGG1RvWFJ1JkvbbKYZ | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1u6RFErfFWnAgUUtugqdiAc169j9fSUq4 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1iDIgCPaHEgxGlFal7yAQmCMpxjGBMR74 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1-35aitCESvhPO9bU5lEQLZSUy5Dz7krB | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1RaOl3AP44jhEx8YveQnovluwUViqoRhm | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1QMgWWmu_mnBS4RPgFSdEgWgCjX7CK0n9 | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1uiGe9ggsK3yhnin04leI2MKx-QcWXomb | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1fcWa13XwFEQQbEaUW5rWNtT742CDRMIh | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_17thH_49POiYU3AjbdPwnwjLxHBc80nBp | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| drive_1Z6f5sn4S3XWIQ7-Fcum-5Nim3Pq--URx | 无记录 | 无下载记录; access=public | [证据](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) |
| sssd_individual_1 | 无记录 | 无下载记录; access=public_mega | [证据](https://raw.githubusercontent.com/AI4HealthUOL/SSSD/main/src/get_data.py) |
| sssd_individual_2 | 无记录 | 无下载记录; access=public_mega | [证据](https://raw.githubusercontent.com/AI4HealthUOL/SSSD/main/src/get_data.py) |
| sssd_individual_3 | 无记录 | 无下载记录; access=public_mega | [证据](https://raw.githubusercontent.com/AI4HealthUOL/SSSD/main/src/get_data.py) |
| sssd_individual_4 | 无记录 | 无下载记录; access=public_mega | [证据](https://raw.githubusercontent.com/AI4HealthUOL/SSSD/main/src/get_data.py) |
| sssd_individual_5 | 无记录 | 无下载记录; access=public_mega | [证据](https://raw.githubusercontent.com/AI4HealthUOL/SSSD/main/src/get_data.py) |
| sssd_individual_6 | 无记录 | 无下载记录; access=public_mega | [证据](https://raw.githubusercontent.com/AI4HealthUOL/SSSD/main/src/get_data.py) |
| sssd_individual_7 | 无记录 | 无下载记录; access=public_mega | [证据](https://raw.githubusercontent.com/AI4HealthUOL/SSSD/main/src/get_data.py) |
| sssd_individual_8 | 无记录 | 无下载记录; access=public_mega | [证据](https://raw.githubusercontent.com/AI4HealthUOL/SSSD/main/src/get_data.py) |
| sssd_individual_9 | 无记录 | 无下载记录; access=public_mega | [证据](https://raw.githubusercontent.com/AI4HealthUOL/SSSD/main/src/get_data.py) |
| sssd_individual_10 | 无记录 | 无下载记录; access=public_mega | [证据](https://raw.githubusercontent.com/AI4HealthUOL/SSSD/main/src/get_data.py) |
| sssd_individual_11 | 无记录 | 无下载记录; access=public_mega | [证据](https://raw.githubusercontent.com/AI4HealthUOL/SSSD/main/src/get_data.py) |
| sssd_individual_12 | 无记录 | 无下载记录; access=public_mega | [证据](https://raw.githubusercontent.com/AI4HealthUOL/SSSD/main/src/get_data.py) |

配置附注：["No data downloads or generation performed by source-verification agent.", "Archive extraction and synthetic generation must create new paths and preserve raw downloads.", "Source checksum absent means record local SHA256 after download; do not treat self-hash as publisher authenticity verification."]

配置附注：mTSBench is an official release by its own authors, not a CATCH author mirror. It omits ASD and NYC and uses different series/splits in some shared families. Exact CATCH data remains the author TAB/OneDrive/Baidu release.

配置附注：["megatools official builds server503 maintenance at verification; use anonymous public MEGA API.", "No existing raw files overwritten; quota/error results retained."]
