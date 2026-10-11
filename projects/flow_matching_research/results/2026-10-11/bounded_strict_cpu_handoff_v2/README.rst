完整CPU基线训练队列继续执行
==========================

原CPU基线资源适配器在等待训练共享互斥锁，下一任务为Anomaly Transformer／SMD registered配置seed1104。
本轮把整条原定175项CPU基线队列迁入独立的串行训练通道，实际原训练命令已启动；其他训练进程没有停止。

完整范围与保真
--------------

* 原队列configs/experiments/fm_strict_baseline_queue.v1.json及其SHA保持不变，全部175项任务、原种子和模型配置保留。
* 原队列CPU线程数仍为2，原原生工作模块仍为scripts.flow_matching.run_tsad_experiment。
* 当前SMD／Anomaly Transformer配置仍为window100、d_model32、8头、3层、batch256及10轮，不把这一registered本地实现宣称为论文作者训练框架等价。
* 原完整结果验证、原队列cpu_baselines.lock、原日志及所有旧结果/失败尝试保持；新通道资源收据使用独立目录以避免覆盖。
* 可用物理内存门槛从12 GiB提高到16 GiB，commit余量仍至少32 GiB，紧急余量仍为16 GiB。增加工作进程树RSS上限10 GiB、private上限12 GiB。
* 保留原资源保护、原轮询与任务间让步逻辑；原GuardedChild类唯一AST修改为给原commit紧急保护条件添加两项内存峰值上限条件。

实际切换证据
------------

handoff_receipt.json核验旧适配器PID17052、创建时间、完整命令、原登记及队列SHA、等待状态、未持有共享锁和没有原生工作子进程。
旧状态中的active_job仍显示前一个已完成任务；只有其原资源收据证明exit0、未触发保护、完整预算保留，且模型结果/权重/分数独立通过完整性验证后，才允许该陈旧字段通过闲置核验。
原状态逐字节保存在preserved_idle_cpu_status.json，旧控制器真实终止后才启动新控制器。
新控制器PID25448，初始原训练工作进程PID6184；创建时间、实际命令和实时CPU/内存观测保存在actual_execution_observations.json。
以后判断是否仍运行应重新核验实际进程及创建时间，不能单凭本快照。

本轮同时发现共享GPU互斥锁由正在运行BRITS／Electricity seed27的原控制器持有，而非空闲持锁。
未退出该控制器或训练进程，未改变GPU序列化策略。该原训练的实际宿主工作进程PID17540及其创建时间也纳入观测。
CFM、RCGAN、TimeGAN和MaelNet的原进程继续运行。

验证与失败保留
--------------

validation.json及validation.txt记录16项通过，覆盖全部原门槛不下降、GuardedChild AST保真、实际资源保护路径中的RSS/private/commit条件、收据与锁只释放一次、PID复用、活跃工作进程、错误命令/配置/待执行任务拒绝及旧资源适配器验证。
probe_receipt.json证明加载整条175项队列，控制器未导入NumPy或Torch。

首次单元测试有4项失败，原因是测试用内存信息缺少vms字段，而原保护代码会计算getattr的后备表达式；真实psutil内存信息含该字段。
修正测试夹具后16项通过；没有改变原保护实现。failed_initial_test_fixture.json及failed_initial_test_fixture_source.py保留首次失败，当时未退出任何控制器或启动原生工作进程。

解释边界
--------

本轮完成的是原定完整训练队列恢复和实际训练启动，不是训练完成或新增性能指标。
最新全局数值表仍为all_algorithm_dataset_metrics_v26；当前工作进程完成后仍须核验模型、全量分数、数据与评价协议才能更新指标。
原45篇论文、681项方法/基线/消融记录、273项原始数据任务及采用基线范围不变，全部实验尚未完成，论文作者等价及完整TAB训练框架认证未取得。
