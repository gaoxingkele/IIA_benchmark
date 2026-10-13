# 两个 BB400 完整重跑的训练锁准入

两个未保存的BB400权限故障已注册完整400 epoch／6400 updates重跑，但旧控制器每10秒尝试非阻塞锁；原CFM队列连续完成GP400种子并进入下一任务，旧控制器仍未派发任何模型。原调度代码在任务间没有额外让出时间，因此一次轮询可以错过两个原任务之间的短暂释放窗口。

本轮只替换了已经核验空闲的重跑父控制器：旧PID28840无原生训练子进程，PID／创建时间／完整命令及旧状态中的队列、运行配置SHA256均核验后，再次复查才停止。原CFM训练父控制器35452及其原生工作进程保持运行；BRITS、MaelNet、Pi及其他训练未停止。交接证明见 `handoff.json`。

新控制器仍使用同一个 `fm_cfm_ts_original_cpu.lock` 文件的第0字节、同一长度1、独占锁。Windows同步 `LockFileEx` 请求在内核等待该锁释放，不再依靠10秒间隔重新竞争；等待时不拥有家族锁。只有原22 GiB提交准入已满足才提出请求，取得锁后再检查同一原资源条件；资源不够立即释放后等待内存。原16 GiB应急保护及全部数值工作进程、模型／数据／种子／预算／容差保持原实现。

实际Windows文件锁测试验证了：与原msvcrt字节锁互斥、释放边界上已排队请求取得锁时原轮询方无法立即抢回、释放后原锁可继续取得，以及取得锁后资源不足／测量异常均释放所有权。测试只运行临时文件锁和调度逻辑，不训练模型，也不提供基准性能。

新父控制器PID、创建时间和命令见 `actual_launched_queued_controller.json`；原生训练与排队状态见 `actual_after_handoff_queued_source_and_untouched_training.json`。取证时新控制器已进入内核锁队列，原GP400／seed1105继续训练。两个新400 epoch重跑尚未完成，不计入成功指标。自动原生模型重放及独立全网格复核控制器保持原配置。

12项关联验证通过，见 `boundary_and_full_scope_validation.xml`。原285项CFM、45篇论文、681项方法／基线／消融记录及273项原始数据任务范围全部保留。此修改仅涉及现有项目文件锁，不需要管理员权限，Windows分页设置未修改。

Microsoft文档：[LockFileEx同步等待、独占范围与解锁行为](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-lockfileex)。CFM文献：[OpenReview Hqn4Aj7xrQ](https://openreview.net/forum?id=Hqn4Aj7xrQ)。
