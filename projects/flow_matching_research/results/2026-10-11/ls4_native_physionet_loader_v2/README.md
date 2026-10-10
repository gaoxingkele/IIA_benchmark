# LS4两套原始PhysioNet加载流程核验

下载的作者`parse_datasets`在原始环境内实际执行，完整8000名患者／41个变量、固定42的6400／1600患者划分顺序核验通过。插值batch64的100个训练及25个测试批次、外推batch32的200个训练及50个测试批次全部核验，共375个批次。

[全部实际批次证据](original_loader_audit.json)、[SHA](validation.json)。派生数据只读复用完整原始解析，未覆盖任何原始或派生文件。已核验训练预算为每种子50000／100000次更新，尚未执行这里的500轮模型训练；不作为论文性能。完整协议限制及保留的旧batch64诊断见[本轮说明](../grasp_checkpoint_replay_resume_v1/README.md)。
