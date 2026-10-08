# DFM: Interpolant-free Dual Flow Matching

代码、主消融或组件已按原文还原；仍缺：author U-Net and learnable prior；Dopri5；MLE-CNF and exact published FM/I-CFM U-Net baselines；author SMAP entity/feature protocol；threshold and adjustment protocol；objective degeneracy/bijectivity needs mathematical audit

冻结要求：作者版本与代码提交、数据实体/划分、训练专属归一化、掩码/窗口及回填、种子和预算、积分器/步数、分数聚合、阈值来源、点调整、指标尺度与不确定性。

可直接运行的本地模型配置及当前状态见 `src/code/implementation_mapping.json`；没有 entrypoint 的配置表示待适配，不能宣称方法已实现。
