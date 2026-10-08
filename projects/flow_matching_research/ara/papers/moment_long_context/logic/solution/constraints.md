# Towards Long-Context Time Series Foundation Models With A Handful Of Additional Parameters

代码、主消融或组件已按原文还原；仍缺：MOMENT patching/embedding/RevIN/head reused; T5-efficient-tiny sized random encoder prepared. Original MOMENT-Tiny/ICM checkpoints and exact pretraining not available here.；All context expansion baselines, 26 UEA SVM pipelines, one-epoch Time Series Pile and original forecasting datasets have not been run.；Original T5 relative bias/dropout retained, while compressed-memory aggregation excludes padded keys. Exact unpublished author ICM insertion semantics not independently verified.；Forecast and reconstruction/feature modes available; original SVM classification experiment remains separate.

冻结要求：作者版本与代码提交、数据实体/划分、训练专属归一化、掩码/窗口及回填、种子和预算、积分器/步数、分数聚合、阈值来源、点调整、指标尺度与不确定性。

可直接运行的本地模型配置及当前状态见 `src/code/implementation_mapping.json`；没有 entrypoint 的配置表示待适配，不能宣称方法已实现。
