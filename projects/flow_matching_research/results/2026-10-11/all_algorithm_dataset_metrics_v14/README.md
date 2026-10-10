# 算法—数据集完整指标与已知缺口

发布：2026-10-10T20:52:10.757805+00:00。主表7257条指标记录：1786条有实测值，5471条缺结果。指标行数不是训练次数；全部原论文实验仍未完成。

[可搜索完整表](results.html)；[完整CSV](all_algorithm_dataset_metrics.csv)；[完整文字表](all_metrics.md)；[异常检测全部指标](strict_algorithm_dataset.md)。

继承v12全部7117条历史记录与各自采集时间，数值逐字节保留；增加120条USHCN、12条原始PhysioNet、8条GT-GAN待完成指标。所有新槽位均为缺结果，不把数据审计、容量诊断或重推断计为新训练性能。

GiFlow/Air36/seed393实际保存的EMA权重已经通过全1747窗口、14批次、268613个目标、20步Euler采样的完整重推断，累计280次模型调用。原生MAE/MSE/MAPE分别为9.654690/359.798492/21.516833%；CPU重算均满足冻结容差。回执中两项列表别名导致的批次数统计已独立修正为9批保留、5批新算，原回执与全部预测保留。[修正与全数组核验](../giflow_native_checkpoint_receipt_audit_v3/corrected_full_native_replay_audit.json)。仍为1/5训练种子，不新增STD/CI，不获作者等价认证。

GT-GAN作者完整245个Git文件已下载并逐一核验；原始Stocks3685行、6变量、全部3662个24步窗口在0/30/50/70%缺失设置下完成原生预处理和样条核验。原生训练和评估尚未完成；30/50%使用明确标注的源程序参数扩展，历史表行等价未证明。

LS4原始PhysioNet插值/外推完整一轮GPU容量检查都通过，分别100/200次更新及25/50个完整测试批次。第一项完整500轮训练已启动；尚未形成500轮性能。USHCN100项500轮任务仍需完成。

[本轮完整执行证据](../gtgan_native_and_giflow_execution_v1/README.md)；[已完成GRASP τ=0.25完整模型核验](../grasp_tau025_checkpoint_replay_v1/checkpoint_replay_audit.json)。其他轨道按原快照保留，不把本次发布日期当作它们重新核验的时间。
