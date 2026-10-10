# Spectral原始完整容量预检与训练队列续跑

上一轮为progress：完成7项GRASP完整结果独立审计；本轮为progress：补上Spectral尚缺的完整容量预检，接入原始30项Table2训练队列，并核验两个新控制器实际存活。

Spectral三原始数据集（sine/stock/mujoco）：保留原始完整网络、全批量、forward/backward/optimizer调用，以及全部三个八工作进程数据加载器。SDFormer已有完整独立容量证明，保持原证明及其SHA不变。

上次混合方法预检因提交余量不足触发保护，失败工作目录及日志不覆盖。新预检采用完整GPU任务准入门限32GiB物理/42GiB提交/14GiB显存，保留16GiB提交/4GiB显存紧急保护及原GPU互斥锁。只更改诊断方法调度、独立输出路径及资源准入；不更改科学配置或训练预算。

预检尚未通过。两个新控制器当前等待GPU锁和完整资源余量；15项Spectral原始训练等待完整容量证明，15项SDFormer任务保留待运行状态。预检不是生成性能结果。

GiFlow Air36和CFM NODE400真实模型进程仍存活并有CPU时间增长；GiFlow原配置上限300轮、作者patience40早停保持不变。原始45篇论文/681方法与消融库存/273数据任务范围保持不变，目标仍未完成。

24项相关测试通过；所有新配置的来源SHA已核验。旧Table2控制器在等待且无模型子进程的状态下交接，未中断活跃模型。

[实际进程与状态](runtime_observations.json)；[调度交接证据](scheduler_transition.json)；[上次失败保护记录](previous_failed_capacity/resource_receipt.json)。
