Spectral Flow长序列原始实验：配置解析依赖恢复
===========================================

原Table3完整GPU预检在导入原utils.utils_args时因缺少OmegaConf退出，尚未构造模型或开始训练。旧失败、原始源代码、原数据、旧环境与所有旧尝试保留。

本轮创建独立环境.venv/spectral-table3-author-py310-v2，离线复用已经登记并核验SHA256的OmegaConf2.2.0和ANTLR4.8安装包，没有下载替代数据或修改其他正在运行的环境。所有97项继承包版本与旧冻结环境一致，新环境共99项冻结包。pip依赖检查通过，配置解析包确实来自新环境自身。首次冻结缺少项目src元数据搜索路径的失败日志也保留；随后使用原队列的PYTHONPATH配置重新冻结通过。原始发布代码没有提供历史OmegaConf版本，这仍是作者环境对齐缺口。

实际调用原发布配置解析器核验全部四项配置：Spectral Flow和ImagenTime分别在FRED-MD与NN5 Daily上运行。所有原YAML设置逐项一致；均保留1000轮、批大小32、四个数据工作进程、训练种子42与评估种子0。完整模型、优化器、采样器、评估器及数据模块共七项原生模块导入通过，未初始化CUDA上下文。

另外在新环境重新执行完整原数据核验：FRED-MD为107×728×1，原划分85/22；NN5 Daily为111×791×1，原划分88/23。所有原数据哈希、整条轨迹标准化与种子42/0的训练/独立评估划分均与旧核验一致，NN5未裁剪到论文文字中的728长度。重建数组与旧数组逐位一致。

26项相关测试通过。确认旧控制器仅在等待资源且没有模型子进程后，终止该已知依赖失败的旧控制器，切换到新的四槽队列。没有中断任何训练，也没有增加训练种子、改变资源保护或缩减模型、数据、采样和评估预算。

新控制器已启动，等待完整GPU内存资源。实际完整GPU模型容量预检、四项1000轮训练、每次完整生成与十次100轮S4分类/预测评估、保存模型重评估仍未完成。本次依赖恢复、模块导入与CPU数据核验均不作为性能成绩，完整结果表仍为v22的1793项实测指标、5490项缺结果。其余45篇论文、681方法基线消融、273原始任务及采用基线继续保留。

证据：offline_installation_and_freeze_receipt.json、full_original_config_parser_and_imports.json、full_original_data_audit.json、independent_data_protocol_comparison.json、previous_controller_retirement_receipt.json、launch.json、verified_iteration_observation.json、validation_command.json。登记配置位于configs/acquisition、configs/reproducibility、configs/models、configs/experiments和configs/runtime。
