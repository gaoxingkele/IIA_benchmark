# DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection

原作者阈值与 point adjustment 和严格验证阈值分轨；作为 MaelNet 组件不能等同整个集成。

冻结要求：作者版本与代码提交、数据实体/划分、训练专属归一化、掩码/窗口及回填、种子和预算、积分器/步数、分数聚合、阈值来源、点调整、指标尺度与不确定性。

可直接运行的本地模型配置及当前状态见 `src/code/implementation_mapping.json`；没有 entrypoint 的配置表示待适配，不能宣称方法已实现。
