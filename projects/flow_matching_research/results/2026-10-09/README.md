# 未完成实验执行进度

快照时间：2026-10-08T17:50:39.885177+00:00。目标仍为全部已知未完成实验；尚未完成。

42 个已可训练窗口方法/消融配置 × 7 个完整本地数据集 × 5 个种子 = 1,470 个基础实验；另有 105 个 SB/SF2M 数值修复重跑任务。
另加入 175 个窗口基线任务：6 个已有方法及 USAD 有符号损失对照，使用相同的完整输入数组和验证段。
保留本地模型配置的训练轮数与容量；非重叠训练窗口和尾部覆盖规则已冻结，这不证明匹配原论文的更新次数、数据划分或架构。
关键训练预算差异：非重叠窗口比原作者 stride=1 的重叠训练少很多梯度更新。相同 epoch 数不能证明训练预算等同；原 stride=1 作者轨仍须独立完成，不能用这里的低分断言原方法无效。
当前任务记录：{'completed': 102, 'partial_or_failed': 60, 'running': 3, 'pending': 1585}。包含数值重试，不能解释为独立方法数或全部基础实验完成数。

CPU 与数值修复进程已核验存活；GPU 续跑等待现有插补链完成并取得共同锁。进程/状态是此快照的观察，后续以当前操作系统进程及结果哈希为准。

## 已完成的本地严格协议结果

验证段校准 1% 报警分位数，无 PA；每个原测试时间点一个有限分数。以下均为本地重建/适配，非作者等效认证。

| 模型配置 | 数据集 | 已完成/预定种子 | 点级 F1 均值 | AUROC 均值 | AP 均值 |
|---|---|---|---:|---:|---:|
| fm_objective_independent | psm | 5/5 | 0.032333 | 0.735128 | 0.550857 |
| fm_objective_ot | psm | 5/5 | 0.030904 | 0.733514 | 0.553258 |
| fm_objective_rectified | psm | 5/5 | 0.032333 | 0.735128 | 0.550857 |
| fm_objective_target | psm | 5/5 | 0.032726 | 0.735262 | 0.550569 |
| fm_objective_vp | psm | 5/5 | 0.029989 | 0.712565 | 0.551002 |
| fm_objective_independent | smap | 5/5 | 0.235262 | 0.536360 | 0.146863 |
| fm_objective_ot | smap | 5/5 | 0.233811 | 0.539347 | 0.147495 |
| fm_objective_rectified | smap | 5/5 | 0.235262 | 0.536360 | 0.146863 |
| fm_objective_target | smap | 5/5 | 0.235143 | 0.535192 | 0.146747 |
| fm_objective_vp | smap | 5/5 | 0.230372 | 0.549575 | 0.149446 |
| fm_objective_independent | msl | 5/5 | 0.072689 | 0.545220 | 0.115527 |
| fm_objective_ot | msl | 5/5 | 0.073644 | 0.545315 | 0.115515 |
| fm_objective_rectified | msl | 5/5 | 0.072689 | 0.545220 | 0.115527 |
| fm_objective_target | msl | 5/5 | 0.072819 | 0.544558 | 0.115526 |
| fm_objective_vp | msl | 5/5 | 0.077124 | 0.541408 | 0.117051 |
| fm_objective_independent | smd | 5/5 | 0.410668 | 0.767886 | 0.337081 |
| fm_objective_ot | smd | 5/5 | 0.412128 | 0.767478 | 0.337461 |
| fm_objective_rectified | smd | 5/5 | 0.410668 | 0.767886 | 0.337081 |
| fm_objective_target | smd | 3/5 | 0.410632 | 0.767544 | 0.334653 |
| fm_objective_sb_stable_v2 | psm | 5/5 | 0.033416 | 0.717722 | 0.545647 |
| fm_strict_timesnet__psm__registered | psm | 4/5 | 0.052776 | 0.598452 | 0.396600 |

逐种子结果、标准误/种子区间、时间块或实体区间、检查点及分数哈希均保存在 [execution_snapshot.json](execution_snapshot.json)。单种子块区间不替代跨算法配对检验。

独立 CFM 与此处单阶段 Rectified 适配器在 sigma=0、相同种子/骨干下使用同一条直线路径，数值相同是预期行为；不应视为两个独立算法的证据。迭代 reflow 尚待另行实现和实验。

## 已补算的固定版本 TAB 范围指标

已核验 78 个完整分数文件的 VUS 与 Affiliation。VUS保留原250阈值、全部整数缓冲长度、inclusive ties与积分公式；完整PSM87841点与原代码执行差异为约1e-16。

| 模型配置 | 数据集 | VUS 已完成种子 | VUS ROC 均值 | VUS PR 均值 | 严格阈值 Affiliation F 均值 |
|---|---|---:|---:|---:|---:|
| fm_objective_independent | msl | 5 | 0.705197 | 0.278498 | 0.725564 |
| fm_objective_independent | psm | 5 | 0.698167 | 0.501818 | 0.585950 |
| fm_objective_independent | smap | 5 | 0.690551 | 0.294041 | 0.537718 |
| fm_objective_independent | smd | 3 | 0.814596 | 0.435263 | 0.801388 |
| fm_objective_ot | msl | 5 | 0.706013 | 0.277971 | 0.728107 |
| fm_objective_ot | psm | 5 | 0.697567 | 0.502244 | 0.578248 |
| fm_objective_ot | smap | 5 | 0.691844 | 0.294302 | 0.536096 |
| fm_objective_rectified | msl | 5 | 0.705197 | 0.278498 | 0.725564 |
| fm_objective_rectified | psm | 5 | 0.698167 | 0.501818 | 0.585950 |
| fm_objective_rectified | smap | 5 | 0.690551 | 0.294041 | 0.537718 |
| fm_objective_target | msl | 5 | 0.704934 | 0.279624 | 0.724266 |
| fm_objective_target | psm | 5 | 0.701574 | 0.502289 | 0.581274 |
| fm_objective_target | smap | 5 | 0.690140 | 0.294203 | 0.537754 |
| fm_objective_vp | msl | 5 | 0.702600 | 0.284205 | 0.733116 |
| fm_objective_vp | psm | 5 | 0.671146 | 0.492057 | 0.522960 |
| fm_objective_vp | smap | 5 | 0.691672 | 0.289387 | 0.533601 |

每个实体独立评价；macro仅平均原参考函数有定义的结果，未定义实体/种子数保留，不填零。Affiliation F、点级 F1 和 PA-F1 是不同指标，不能直接混比。VUS缓冲按测试标签事件长度产生，是已披露的评价依赖；严格阈值仍只来自验证段。完整TAB训练流程未等效认证。

## 历史记录补充与范围更正

先前 2026-10-08 表只覆盖 FM 注册队列，未纳入旧 mtsad_reproduction 目录的 137 份运行记录。TimesNet、Anomaly Transformer、DCdetector 等历史数值确实存在，因此不能据前表声称整个项目没有异常检测成绩。

旧记录可能重复种子、预算或协议；缺少新严格协议要求的冻结原始数据、完整实体/时间覆盖或检查点链，不能直接晋升为新队列已完成项。全部历史数值及协议见 [historical_mtsad_metrics.csv](historical_mtsad_metrics.csv)。未改写历史文件。

## 未完成范围仍然保留

45 篇 ARA 的原始数据集、681 条已审方法/消融记录以及每篇复现缺口，完整保留于 execution_snapshot.json。启动 1,470 个本地任务并未替代这些义务。

- 原有 540 个插补/迁移任务继续执行；原数据和活跃队列未重启。
- 原作者 MaelNet RL、Pi 期刊版本等效、CrossAD、MOMENT 权重及其他 source-only 适配器仍需完善。
- GiFlow、预测、生成、连续时间、图像、单细胞、表格原始实验及工业异常检测迁移仍未全部完成。
- TAB 核心指标/比例诊断单列：合并训练/测试分数校准且按测试标签选最佳比例。剩余 VUS/Affiliation 和完整 TAB 原作者训练流程继续执行。
- 原 Sinkhorn 不收敛日志和部分产物保留；修正版只改求解器续接与迭代上限，最终正则、边缘容差和训练轮数不变。修正版已有实际完整 PSM 成功结果，其余种子/数据集继续核验。

数据身份、通道、实体边界、训练/验证隔离、缺失修复和训练段缩放统计见 [data_manifest.json](data_manifest.json)。SMAP_P-7 仅提供训练文件，明确登记后不纳入 51 个有测试文件的实体评价。
