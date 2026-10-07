# FM 方法族的工程对应

基础训练目标通常是条件速度回归：`E ||v_theta(t,x_t,condition)-u_t||²`。这里的时间 `t` 是输运时间，不能直接当成原始序列时间。不同路径、耦合与条件化会改变目标速度及评分含义。

| 参考 | 核心变化 | 必须冻结的实现因素 | 检测适配边界 |
|---|---|---|---|
| Flow Matching、OT-CFM | 条件概率路径；小批 OT 端点耦合 | path/noise schedule、耦合、时间采样、ODE/NFE | 库的损失类尚不是检测器 |
| Rectified Flow、RF-OT、InstaFlow | 直线路径、reflow、快速蒸馏 | 端点配对、reflow预算、蒸馏教师和步数 | 成本或一步生成优势不推出检测 F1 |
| Stochastic Interpolants、SF2M | 插值噪声及 score/flow 联合学习 | 扩散噪声、桥耦合、score权重、SDEsolver | score、velocity及likelihood须分名比较 |
| DFM | 前/反向向量场与双射一致性 | 双场参数、逆变换及密度积分 | 原文 SMAP 检测可进入原协议复现 |
| GRASP | 图谱平滑输运与速度失配 | 图构造、谱截断、路径及失配积分 | mTSBench 与经典五版本分开 |
| GiFlow、TG-MSFM、JFI、CFMI | 条件先验、图/多尺度结构与联合插补目标 | 缺失掩码、可见条件、观测投影和下游监督 | 原插补先复现，检测适配另立实验 |
| CFM-TS、TSFlow | 轨迹 GP/边际动态、高斯过程先验 | 原时间与输运时间、导数后验、条件化 | 动力学/预测误差不能作为检测分数 |
| PrismFlow、Spectral Mean Flow | 残差专家或分布谱算子 | 路由、张量分解、频谱预算 | 两者机制不同，不按同名词并类 |
| Flow Mismatching | 正常速度与到测试样本的几何速度失配 | 路径采样、时间/路径聚合、正常训练 | 原图像任务，时序适配仍需验证 |

集合中的45篇包含基础、竞争方法和数据/协议来源，不是45个已经训练完成的 FM 变种。每篇的原任务、代码状态和限制以各自 metadata 与 implementation_mapping 为准。

共同评分消融须至少分开：速度失配、重建误差、密度/score、条件插补残差；使用同一训练划分和验证校准，以 NFE、参数量和耗时报告计算成本。完整比较见 comparison_plan.md。
