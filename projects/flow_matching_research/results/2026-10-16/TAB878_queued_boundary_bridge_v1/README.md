# 为原 TAB878 接续控制器提供排队式安全边界窗口

原 TAB 在实体完成后立即重新申请审计锁，续接控制器的一秒轮询已实际错过一次释放边界。新增辅助进程用 Windows 原生异步 LockFileEx 排队等待同一文件的字节零锁，收到释放事件后短暂留出元数据窗口，最多持有两秒；观察原父进程因锁等待而进入原十秒重试周期后释放，既有一秒续接者可以尝试接管。若其他合法复核先拿到锁，不打断它，而在下次释放时再次排队。此处不宣称 Windows 各存储实现有普遍严格 FIFO 保证。

原始指标实体和其他复核运行时均持有同一锁，辅助进程拿不到锁，不会暂停、终止或启动任何计算进程，也不重启等待超时的实验。它只排队自己的文件请求，取消时只取消自身请求并等待完成、释放自身句柄。锁文件必须已经存在，原生排队实现不写文件字节。

保留全部 878 项/2057 组标签、全部指标函数与网格、源工作进程、参数、16/28/16 GiB 保护线、3/6 GiB 上限及同一审计锁。辅助进程不派发科学工作，不改变既有成功、失败或部分结果，不生成生产科学守卫。接管依旧由已验证的原续接控制器完成其两次身份/子进程/终止守卫检查。

19 项测试通过，包括真实子进程中原 msvcrt 锁与新 LockFileEx 锁互操作、排队者在释放时取得锁而旧非阻塞锁立即重抢失败、取消排队不解锁其他所有者、不写原锁文件字节，以及异常/有界窗口结束时始终释放自身锁。元数据模拟只使用独立测试目录，不计数据集实验成绩。

实际是否已接管见 actual_bridge_and_existing_fair_successor.json。若 handoff_completed=false，辅助进程仍在实际系统锁队列中等待，原计算保持运行，不能声称 SAITS 已完成或调度问题已经解决。全部原 45 篇论文、681 项方法/基线/消融、273 项数据任务、702 项论文标量目标及作者队列、原生/标签复核、完整 TAB、五训练种子统计继续保留。

接口依据：[Microsoft LockFileEx](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-lockfileex)、[OVERLAPPED](https://learn.microsoft.com/en-us/windows/win32/api/minwinbase/ns-minwinbase-overlapped)、[CancelIoEx](https://learn.microsoft.com/en-us/windows/win32/api/ioapiset/nf-ioapiset-cancelioex)。
