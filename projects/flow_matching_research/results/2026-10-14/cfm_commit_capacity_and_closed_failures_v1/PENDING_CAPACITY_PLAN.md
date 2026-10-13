# 提交内存容量方案：等待管理员操作

只读检查时，Windows提交上限为67.86315155029297 GiB，空余提交量为28.414073944091797 GiB；当前C盘分页文件实际分配4096 MiB，启用自动管理。当前进程不是管理员。原配置要求CFM完整重放30 GiB、8项完整重跑及Spectral全向量审计40 GiB、SHCL完整413项队列56 GiB提交余量，当前均未满足这些提交门限。

可审阅的拟议配置是：F盘根目录 `F:\pagefile.sys`，初始32768 MiB、上限65536 MiB；C盘保留系统管理。F盘当前空余约484 GiB。该方案额外占用F盘32至64 GiB磁盘空间；保持其他提交使用量不变时，新增初始32 GiB backing容量后，预计提交余量为60.414073944091797 GiB，达到本次列出的最高56 GiB门限。实际启用后仍须重新测量提交上限及各原准入条件，不按估计直接宣布任务已可运行。

系统管理分页文件可能在提交使用达到限额90%左右时增长，而本项目的16 GiB应急余量在当前限额下对应约76.4%使用率。这个区间差异解释了为什么不能把自动扩容当作排队任务的可靠准入机制；它不证明Windows永远不会因其他负载或系统因素增长分页文件。

管理员可打开高级系统设置中的性能设置、虚拟内存配置，将C盘保持系统管理，并为F盘设置上述初始和最大值。分页配置可能延后到系统启动生效；需要在全部当前训练得到安全处理后由用户安排生效时间。本轮只读检查和方案文件没有改动Windows、注册表、分页文件或原资源保护，也没有重启机器或停止训练。

此方案增加提交backing容量，物理RAM、GPU显存、GPU锁及全部实验预算保持原要求；它不提供新的基准性能分数。数据目录 `F:\aicoding\IIA_Data` 保持原位。

Microsoft文档：[分页文件参数及所需特权](https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-pagefilesetting)、[提交上限及系统管理分页文件增长规则](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/how-to-determine-the-appropriate-page-file-size-for-64-bit-versions-of-windows)。具体测量及门限来源见 `read_only_commit_capacity_and_pending_plan.json`。
