# 完整检查点重推断与LS4原始不规则数据准备

本轮实际完成：2／13个当前完整本地GRASP检查点的重推断；LS4 PhysioNet原始A/B两组全部8000名患者的原生解析，以及两套发布配置共375个完整训练／测试批次的加载核验。相关34项测试通过。原论文全部训练、其他数据集和消融实验仍未完成。

## GRASP完整模型重推断

范围固定为当前13个完整训练检查点、31套原始评分配方，保留原来9120个实体任务／480个原始种子组的全部义务。重推断不增加训练种子或替代其余未训练任务。本地GRASP为论文描述还原实现，原作者实现等价性未证明。

截至本快照，ER和GNN的SWAN SF种子1103已通过。每个检查点均为1500轮、93000次更新；重新加载实际模型参数、图谱路径缓冲区、邻接矩阵和训练期scaler，对全部3935个验证点、72000个测试点重推断。使用原stride=1、batch256、5个来源样本、10个流时间点和原有重叠窗口汇总方式。全部原先评分配方均已登记，队列继续执行其他11个模型。

CPU重推断GPU训练的权重，数值误差按启动前冻结的容差核验，不宣称逐位相同。ER完整重推断的测试分数最大相对差约1.32e-7，验证阈值分类差异为0；其验证阈值无PA F1为0.504954536，回顾性测试标签最优F1为0.511194867。后者不是严格泛化排名。来源、完整窗口评估次数和各指标差值见完整快照文件及[完成清单](completed_checkpoint_replays.csv)。

[冻结执行配置](../../../../../configs/reproducibility/fm_grasp_full_checkpoint_replay.2026-10-11_v1.json)、[实际进程核验](runtime_observations.json)、[验证与SHA](validation.json)。大数组保存在`F:/aicoding/IIA_Data/experiments/grasp_full_checkpoint_replay_v1`，原始检查点、分数和失败历史保持不变。

## LS4原始PhysioNet预处理

调用下载的LS4作者原始`PhysioNet`解析器，使用已有且校验通过的A/B压缩包和Outcomes-a。仅处理Windows不合法的`?download`显式文件名，以及离线原始文件的读取路由；作者代码文件未改写。完整保留41个变量、0.016小时量化与原始观测重复值归并。派生文件在`F:/aicoding/IIA_Data/public_datasets/flow_matching/ls4_native_physionet_v1`，原始压缩包和其他已下载数据未覆盖。

初次接口诊断对两套配置均按batch64组批，因此其外推部分属于额外批量诊断，不是发布外推配方的完整加载证明。该诊断及代码保留。后继v2实际调用原始`parse_datasets`，依据每个发布YAML读取批量，并核验全部原始患者及冻结划分顺序：

| 发布任务 | 患者训练／测试 | batch | 完整训练／测试批次 | 原始epoch | 原配方待执行更新数 |
|---|---:|---:|---:|---:|---:|
| 插值 | 6400／1600 | 64 | 100／25 | 500 | 50000 |
| 外推 | 6400／1600 | 32 | 200／50 | 500 | 100000 |

[全部原始解析证据](../ls4_native_physionet_preprocessing_v1/native_preprocessing_audit.json)、[原批量完整加载证据](../ls4_native_physionet_loader_v2/original_loader_audit.json)、[加载配置](../../../../../configs/reproducibility/fm_ls4_native_physionet_loader.2026-10-11_v2.json)。这些是全数据接口证据，不是已完成500轮训练或论文性能。

源码口径仍有明确限制：归一化统计来自全部8000患者，包括测试患者；使用`(data-min)/max`；插值发布配置没有额外抽取的留出掩码；组批后按数据和是否为零筛除时间点。患者枚举顺序按本地原始加载器冻结，历史作者顺序未知。源码另保存了最后一行outcome辅助向量；A组每患者死亡标签仍正确保留在患者元组，B组无标签。当前仅核验`classify=False`两套配置，不能据此宣称分类流程已修复或作者等价。

原始来源：[LS4作者仓库](https://github.com/alexzhou907/ls4)、[PhysioNet2012](https://physionet.org/content/challenge-2012/1.0.0/)。全部源代码、环境、原始文件与派生产物均有SHA绑定。
