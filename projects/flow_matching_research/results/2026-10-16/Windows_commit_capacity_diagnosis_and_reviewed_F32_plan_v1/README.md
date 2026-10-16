# Windows 提交容量：实测瓶颈与尚未应用的 F 盘扩容方案

两套 Windows API 独立实测一致：物理总量约63.86GiB，当前提交上限67.863GiB，可用提交量19.751GiB。
实际只有 C:\pagefile.sys，当前分配4096MiB；全局自动管理开启，注册表配置为 `?:\pagefile.sys`。
可用物理内存和提交余量是两种约束。4条原始配置的32/40GiB提交准入未满足；配置和真实读数、哈希及完整参数见JSON/CSV。
队列因此等待，不能据此判定训练算法出错，也不能改低准入或紧急保护门槛。

新增配置脚本 `benchmark/scripts/flow_matching/configure_F_pagefile_reviewed_v1.ps1` 默认只读，不请求UAC、不自行提权。
经核对的拟写入配置为 C:\pagefile.sys 0 0 和 F:\pagefile.sys 32768 32768：保留C盘按卷系统管理，在F盘增加固定32GiB。
应用时全局 AutomaticManagedPagefile 会从 true 改为 false，改用上述逐卷配置；不能称全局策略不变。
脚本要求F盘固定NTFS且剩余至少96GiB，拒绝覆盖已有F分页文件，核验计划和当前配置一致，先备份再写入，失败尝试恢复原配置。
脚本没有训练进程终止、重启、阈值修改或制造内存压力操作。

19项测试通过，包括规划边界、原配置保留、物理门槛不被提交投影代替、PowerShell解析，以及模拟CIM/注册表的成功写入与故障回滚。
在真实机器上只执行了默认只读模式和非管理员 `-Apply` 拒绝路径；前后注册表与自动管理标志一致，F分页文件未创建。
管理员实际应用分支尚未在真实系统执行，不能把模拟测试写成真实扩容成功。

管理员可在复核计划后执行：

```powershell
powershell.exe -NoLogo -NoProfile -NonInteractive -File "D:\aicoding\IIA_benchmark\scripts\flow_matching\configure_F_pagefile_reviewed_v1.ps1" -PlanFile "F:\aicoding\IIA_Data\environments\windows_commit_capacity\read_only_diagnosis_20261016_v1\reviewable_pagefile_plan.json" -Apply
```

此命令未执行。原始计划如已过期，脚本会拒绝覆盖，需重新只读采集并复核。
分页配置可能到后续系统启动才生效；现有训练期间不重启。只有运行时API提交上限与余量确实增加，且所有物理/GPU/锁/稳定采样门槛满足，才能承认实际准入。
表中“增加32GiB”仅假设C盘当前分配和总提交用量不变；后续启动或其他负载会改变这些条件，不承诺全部队列同时可跑，也不把虚拟提交容量等同于实际RAM。

本包新增完整实验数为0、优化器更新为0。现有45篇/681方法基线消融/273原始数据任务/702论文标量目标保持，整体未完成。

依据：[Microsoft 分页文件配置与运行状态](https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-pagefilesetting)，
[Microsoft 提交上限与系统管理分页文件](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/how-to-determine-the-appropriate-page-file-size-for-64-bit-versions-of-windows)。
