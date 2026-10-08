# Flow Mismatching: Unsupervised Anomaly Detection via Velocity Discrepancies in Flow Matching Models

代码、主消融或组件已按原文还原；仍缺：No Flow Mismatching author repository located; architecture assembled from related cited Meta U-Net.；Exact StableAdamW with update RMS clipping training pipeline, MVTec/VisA splits, category training and checkpoint reproduction remain pending.；Compared Reflect, WT-Flow, TCCM, D-Flow, ReconFlow implementations from Appendix C are not represented by this detector primitive.；Time weighting switch is a local diagnostic supported by toy analysis; it is not a claim of all paper ablations.；K/T sweep list is a development grid; all paper Table 6 rows still require exact transcription.；Original task is image detection; do not report as TSAD performance.

冻结要求：作者版本与代码提交、数据实体/划分、训练专属归一化、掩码/窗口及回填、种子和预算、积分器/步数、分数聚合、阈值来源、点调整、指标尺度与不确定性。

可直接运行的本地模型配置及当前状态见 `src/code/implementation_mapping.json`；没有 entrypoint 的配置表示待适配，不能宣称方法已实现。
