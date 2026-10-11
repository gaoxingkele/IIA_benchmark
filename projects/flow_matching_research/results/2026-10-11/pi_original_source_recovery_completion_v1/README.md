# Pi-Transformer：原镜像 PSM seed42 完整训练与独立复算

原始任务 `pi_author__source_runtime__psm__full__seed42` 的恢复运行已完成，完整预算和全部保存分数通过独立审计。完整检查点重新推断已经注册并启动控制器；当前等待原有内存门槛，尚未通过。因此本结果未并入 v38 技术指标总表，不能称期刊结果或严格 TAB 复现完成。

| 同一作者阈值下的指标 | 未做 PA | 作者 PA 后 |
|---|---:|---:|
| Precision | 29.95% | 97.32% |
| Recall | 1.16% | 98.76% |
| F1 | 2.24% | 98.04% |
| 实际／调整后真阳性点数 | 283 | 24,045 |
| 假阳性点数 | 662 | 662 |

原始连续分数 AUROC 为 0.481281，AP 为 0.272631。两列均保留作者在测试集上早停，以及合并训练／测试分数设阈值的流程；未做 PA 的一列同样不能直接称为严格 TAB 成绩。以上为一个训练种子，尚未完成原定六种子，也未证明期刊正文与历史镜像等价。以前完成的 PSM seed1103、seed1104 属于 `causal_axis_repaired/full` 分支，不能与本次 `source_runtime/full` 合并为三种子均值。

完整训练保留 132,481×25 原始训练数据、132,382 个 stride=1 窗口、batch256、512 维模型、8 个头、1,930,289 个参数、CPU2。冻结镜像配置的预算是 **3 轮，每轮 518 个优化批次，共 1,554 批次**，三轮均完整执行。配置 e_layers=2 在原镜像中产生一个实际注意力层，该行为没有改写。原测试数据 87,841 点中覆盖 87,800 点，原推断丢弃的 41 个尾部点被显式保留为覆盖缺口。

独立检查包括：原始队列与恢复任务逐字段绑定；全部论文、源码、原始数据和产物 SHA256；原数据形状与完整标签；全部训练／测试分数的形状和有限性；训练分数中位数与 IQR；原作者合并阈值；保留 index0 边界的独立事件 PA；混淆矩阵直接复算；全部 15 个温度／分数流诊断控制。资源守卫没有终止本次完整训练。

事件诊断包含全部 **72 个异常事件，其中只有 27 个被原始逐点预测命中**。事件长度中位数为 5 点，最长 8,861 点；命中的长事件经 PA 填充后增加 23,762 个真阳性点，真阳性点数扩大约 84.96 倍。这解释了本次单种子结果为什么在逐点 F1 很低时，PA 后仍能达到 98%：指标高度受事件长度和真实标签扩展规则影响。事件行只用于描述，不能当作独立训练重复或跨实体泛化证据。

- [完整预算、标签与分数复算证据](independent_full_recovery_completion.json)
- [严格重载完整检查点及原随机顺序的验证](independent_original_sampler_rng_probe.json)
- [全部原始异常事件 CSV](../pi_original_source_PA_event_diagnostics_v1/all_original_anomaly_events.csv)
- [事件诊断汇总](../pi_original_source_PA_event_diagnostics_v1/all_original_event_diagnostics.json)
- [完整检查点重新推断配置](https://github.com/gaoxingkele/IIA_benchmark/blob/main/configs/reproducibility/fm_pi_original_source_full_checkpoint_replay.2026-10-11_v1.json)

随机顺序验证已独立重建原初始化、DFA 遍历入口／出口 RNG 哈希及原抽样索引哈希，再完整消耗三轮训练／测试验证的采样器随机调用。全部原始权重已严格加载。完整重新推断将保留 batch256、CPU2、同一 PyTorch 环境、全部重叠训练窗口和原始测试覆盖，逐个检查全部保存分数流；不能把当前 RNG 验证称为已完成模型重新推断。

已启动的重新推断控制器继续使用原恢复任务的 24 GiB 可用物理内存、32 GiB commit 余量准入、16 GiB 紧急余量、10 GiB RSS／12 GiB private 上限。当前只等待准入，不持有共享工作锁。其他原始训练和失败产物均保留。
