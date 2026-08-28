# PRONTO 报警表示与标签错配消融报告

日期：2026-08-28  
适用范围：CTFH、Structured HDAM、ConE-AFC、Cross-Conformal AFC 的工程化真实数据代理验证

## 结论

PRONTO 上的退化不能通过把持续报警状态简单改成上升沿事件解决。状态窗与上升沿窗均未使 CTFH 产生有效 consensus hash，也未使以 CTFH 为基分类器的 conformal 方法缩小预测集合。HDAM 的上升沿 balanced accuracy 从 0.1010 升至 0.25，但其所有测试样本均被预测为 `Air blockage`，属于四分类多数类塌缩，不是有效提升。

因此下一步采用双轨策略：

1. 保留 PRONTO，继续用于“状态表示 vs 上升沿表示”的负对照、T1/T2 和过程—报警联合分析。
2. 不在 PRONTO 测试集上反复调参；T4 分类主评测迁移到具有事件级载荷和洪泛/根因类别的 TEP Alarm、NPP 或 FCC 数据。
3. 新增开放事件数据补充 sequence、density、forecasting 和跨设备验证，但没有专家洪泛标签的数据不得替代 TEP Alarm 主榜。

## 协议

- 数据：PRONTO 官方 aligned/labelled Testday2–4。
- 标签：四类非 Normal 故障工况；明确不是专家确认的报警洪泛类别。
- 样本：300 秒非重叠窗口，按连续故障段前 60% 训练、至少一个完整窗口 purge、后 40% 测试。
- 规模：60 个训练窗、38 个测试窗、13 个报警 tag。
- 表示 A：每秒持续报警状态 `state`。
- 表示 B：先在完整测试日时间线上计算 0→1 上升沿，再进行故障段切分和窗口化，避免在边界重复制造激活。
- 前缀：60/120/180/240/300 秒。
- HDAM 本地验证参数：5 秒 bin、binary aggregation、60 秒模板；这些不是论文超参数。
- ConE/Cross-Conformal 基分类器：CTFH；训练数据内部完成 fit/calibration 或 cross-fold calibration，测试集不参与阈值选择。

## 全窗口结果

| 方法 | 状态表示 | 上升沿表示 | 判定 |
|---|---|---|---|
| CTFH | Accuracy 0.2895；BA 0.2500；Macro-F1 0.1122；四类 consensus hash 均为 0 | 完全相同；全部预测为 `Air blockage` | 表示变化无效，指纹空间未形成 |
| Structured HDAM | Accuracy 0.1053；BA 0.1010；Macro-F1 0.0720 | Accuracy 0.2895；BA 0.2500；Macro-F1 0.1122 | 上升沿只造成多数类塌缩，非有效改善 |
| ConE-AFC + CTFH | Coverage 1.0；平均集合大小 4.0；singleton rate 0 | 完全相同 | 覆盖率由全类别集合取得，无判别效率 |
| Cross-Conformal + CTFH | Coverage 1.0；平均集合大小 4.0；singleton rate 0 | 完全相同 | 同上；不是有效的不确定性收缩 |

HDAM 状态窗在 120 秒前缀出现一次 BA 0.1913、Macro-F1 0.1900 的局部峰值，但 60/180/240/300 秒 BA 均为 0.1010 左右，且完整窗口有两个类别召回为 0，不能解释为稳定早期分类能力。上升沿 HDAM 的所有前缀均固定为 BA 0.25、Macro-F1 0.1122。

## 诊断

1. **载荷错配**：CTFH/HDAM 面向事件次序、重复模式和类别特定洪泛结构；PRONTO aligned alarm 列是 1 Hz 持续状态。
2. **标签错配**：PRONTO 的 `Air blockage` 等是过程故障工况，不是由专家划定的洪泛 episode/root-cause 类别。同一工况可包含长时间平稳报警，也可根本不构成洪泛。
3. **信息不足**：上升沿使持续状态变为极稀疏激活，但不能创造缺失的报警消息次序、优先级、ack/reset、事件文本或专家类别。
4. **Conformal 退化是诚实反映**：CTFH 基分类器对类别不提供区分，预测集合覆盖全部四类是符合其输入证据的保守输出；不能只看 coverage=1.0。

## 新匹配数据的结构审计

| 数据 | 已下载画像 | 可立即支持 | 仍不能支持 |
|---|---|---|---|
| CoMoPI | 150,650 个十分钟 bin、8 台设备、123 类报警、194,974 次报警；AL_53/54 阳性 bin 仅 23/18 | 报警密度、bad actor、稀有故障报警预测 | bin 内事件次序、专家洪泛类别 |
| SMD10TOWFGR | 10 台风机、230,618 条日志、167 个全日志 code；其中 Alarm 1,002、Warning 688，99 个报警/警告 code | 事件序列、密度/洪泛候选检测、设备留一验证 | 直接可用的洪泛/root-cause 类标签 |
| EnAS | 219,893 行、33 列、两种产品变体 | 离散状态变化、异常/产品状态序列 | 工业报警洪泛类别 |
| iMAKS | 211,200 条带标注传感器记录、22 个 sensor；1,460 条 WARNING/CRITICAL | 合成异常、报警阈值、因果与适配器 smoke | 真实工业性能或 SOTA 榜单 |

本轮四个载荷均已登记、下载并通过 MD5。结构画像写入 `data/public_datasets/profile.json`；下载审计写入 `data/public_datasets/audit.json`。

SMD10TOWFGR 的首个事件级 Criterion C 实验也已执行：把 Alarm/Warning 日志按 1 分钟 occurrence bin 表示，在 1,820 个 turbine-day 上使用 10 分钟 attention window、30 分钟 long-standing window、阈值 10。共得到 4 个候选区间、分布于 3 个 turbine-day，最大 attention cardinality 为 11，candidate exposure 为 `7.30e-6`。这些是待复核候选，不是真阳性；但它证明书第 5.1 节检测器已经在与算法输入更匹配的真实事件日志上落地，而不是继续强迫分类器使用持续状态窗。

## 进入正式 T4 榜单的门槛

- 原始报警事件至少包含 timestamp、tag/code、activation/clear 或消息次序。
- 提供专家 episode 边界和 fault/root-cause class，或提供可审计的官方映射。
- 按 simulation run、设备或工厂分组切分，禁止同一 episode 跨 train/test。
- 同时报告点分类的 balanced accuracy/macro-F1，以及 conformal coverage、平均集合大小、empty/singleton rate。
- 报告 10%–100% 前缀曲线和 missing/spurious/jitter/delay 鲁棒性。
- 所有超参数仅在训练/验证部分冻结；测试集不得用于选择表示或阈值。

当前最高优先级仍是 [TEP Alarm DataPort](https://ieee-dataport.org/open-access/tennessee-eastman-process-alarm-management-dataset) 主载荷。开放补充数据来源为 [CoMoPI](https://zenodo.org/records/7572501)、[SMD10TOWFGR](https://zenodo.org/records/14546480)、[EnAS](https://zenodo.org/records/4742256) 和 [iMAKS](https://zenodo.org/records/20075430)。
