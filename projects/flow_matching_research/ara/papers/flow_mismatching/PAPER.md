# Flow Mismatching: Unsupervised Anomaly Detection via Velocity Discrepancies in Flow Matching Models

研究任务：`image_anomaly_detection`；方法族：`velocity_mismatch_ad`。

本工程将原文主张、代码/数据准备和本地实验状态分开登记。全文校验通过，性能复现由独立实验提供。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

主要复现问题：代码、主消融或组件已按原文还原；仍缺：No Flow Mismatching author repository located; architecture assembled from related cited Meta U-Net.；Exact StableAdamW with update RMS clipping training pipeline, MVTec/VisA splits, category training and checkpoint reproduction remain pending.；Compared Reflect, WT-Flow, TCCM, D-Flow, ReconFlow implementations from Appendix C are not represented by this detector primitive.；Time weighting switch is a local diagnostic supported by toy analysis; it is not a claim of all paper ablations.；K/T sweep list is a development grid; all paper Table 6 rows still require exact transcription.；Original task is image detection; do not report as TSAD performance.
