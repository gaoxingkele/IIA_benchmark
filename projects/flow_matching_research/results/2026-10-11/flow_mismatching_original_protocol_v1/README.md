# Flow Mismatching 原始配置、消融目标与自动预处理

本轮核对原文实验设置和相关附录，补齐独立的原文配置和附录 C.2 公式组件，并实际启动两项等待完整原始数据审计的预处理任务。**没有新增本地模型性能结果，完整论文复现仍未完成。**

- 原文默认：每个数据集训练一个带类别条件的模型；2000 epochs、batch32、StableAdamW/AMSGrad、lr1e-4、weight decay1e-4、betas(0.9,0.99)、eps1e-6、update RMS threshold1.0、bf16、TF32、DDP；推理 K10/T10。原文页8/21；旧配置8/8仍是开发配置，保持原有依赖绑定。
- 原文消融：Table6 的30组完整 K/T 设置、Table9 的100至2000每100epoch检查点、Table7 的两数据集各5种聚合器、Table8 的条件/混合类别均值、Table2 的6种同骨干 FM 比较。520个论文报告值写入 `paper_reported_targets.csv`，全部标为本地测量待完成；其中 F1-max 使用测试标签选择阈值，不构成严格 TAB 指标。
- 补齐 Reflect 污染/训练/推理、TCCM 训练/推理、WT-Flow 仿射/逆向训练/能量推理、ReconFlow 重建热图，共9个可调用公式组件。TCCM 原始比较只有50epochs；D-Flow 原始 FlowGrad和优化预算明确缺失。WT原始特征空间方法与本论文图像域适配须区分。原生掩码、WT统计和ODE由调用者明确提供，不能假定已完整复现。
- MVTec/VisA 模型配置明确绑定15/12类。自动预处理等待完整下载、全图像/掩码审计通过，再逐文件重验SHA；只用正常训练图像统计，流式写入F盘新派生目录，拒绝覆盖，保留原始文件。
- 预处理冻结的RGB值域、双线性图像缩放、最近邻掩码缩放、全部正常训练图像的每RGB总体方差，均是原文未明确处的**本地选择**。需作者代码或进一步原文证据才能认证一致性，不以此宣称原论文完全一致。
- 25项针对性回归测试通过；此前42项相关测试通过，两组选项有重叠，不能合计为67个独立测试。测试使用小型人工数组/图像，只验证公式方向、范数、标签、泄漏防护、校验和与原始文件保留。

实际训练与下载进程证据见 `actual_continuing_training_download_and_preprocessing_handles.json`。登记时 TimeGAN、RCGAN、Pi因果修复分支、CrossAD-PSM、BRITS-seed29、CFM-NODE 六个模型继续计算；两下载仍未结束，新增两预处理控制器处于等待完整审计状态。Pi原模型完整检查点推理仍按原资源门限等待。

未完成项包括全部原始数据下载/派生数据审计、StableAdamW实现身份与AMSGrad/RMS交互、DDP批量解释和环境、未报告的seed/数据加载/scheduler/EMA、完整训练、原始FlowGrad/ODE和掩码生成、AUPRO评估实现、全部逐类别与效率/玩具/定性结果。45篇/681方法记录/273数据任务范围和既有v38指标保留，未撤销任何原实验义务。

论文来源：`papers/literature/flow_matching/project_additions/flow_mismatching_2605.23070v1.pdf`；SHA256 `6f6791875149663cfc8703ae87f744858c7ef80d17b1b58e05512fa830e4f8b5`。页21/28/33已渲染核对。
