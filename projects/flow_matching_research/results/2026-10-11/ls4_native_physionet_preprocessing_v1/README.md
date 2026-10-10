# LS4 PhysioNet完整原始解析与batch64诊断

作者原生解析器处理了A/B两组全部8000名患者、41个变量、0.016小时量化，原始文件校验通过且没有覆盖。完整派生患者记录、原始固定42的6400／1600分组划分及归一化参数保存在F盘。

此版加载诊断对插值、外推均按batch64执行250个批次；外推发布YAML实际为batch32。因此该外推诊断不是原批量加载证明。保留原诊断，正确的375个发布配置批次全部由[后继v2](../ls4_native_physionet_loader_v2/README.md)实际调用原始`parse_datasets`另行核验。预处理通过不代表训练完成或作者指标已复现。

[完整原始患者、派生文件SHA及诊断](native_preprocessing_audit.json)。原代码联合训练／测试归一化、无额外插值留出掩码、辅助outcome文件和枚举顺序限制见[本轮说明](../grasp_checkpoint_replay_resume_v1/README.md)。
