完整保存分数评价队列恢复
======================

四条原定VUS与Affiliation评价队列此前均等待训练共用的CPU互斥锁。
本次只将这四条只读评价队列迁入独立的串行CPU互斥锁，已确认实际原生评价开始并产出完整结果。
原训练进程没有停止；原科学配置、模型权重、分数、划分、实体、测试时间点、评价工作模块均未改变。

范围与保真
----------

* 四条队列共登记3,362个原定评价任务：主队列1,750、Reflow 140、工业1,400、MOMENT 72。
* 切换登记时有967个模型结果文件。这只表示文件存在，不表示967项均已通过完整性核验。
* 保留原来的1个CPU线程、250个阈值、从0至两倍截断中位异常事件长度的每一个整数缓冲区，以及全量精确稀疏Affiliation计算。
* 原控制器main函数经AST核验，唯一改动为互斥锁路径常量；原生评价仍由evaluate_tab_ranges_sparse_job执行。
* 提高资源门槛：可用物理内存至少16 GiB、commit余量至少28 GiB、紧急余量16 GiB；增加进程树RSS上限8 GiB和private上限10 GiB。
* 四个旧控制器均核验实际PID、创建时间、命令、原配置SHA、等待状态及无实际工作子进程后退出。保留旧状态快照，确认终止后才启动替代控制器。

完整结果证据
------------

completed_evaluations_and_live_execution.json在固定采集时刻独立核验了切换后新生成的7项完整评价，共评价2,985,083个测试点（跨实验累计，包含不同种子）。
完整原生evaluation.json、evaluation.sha256和resource_usage.json的字节副本保存在full_completed_evaluations下。

* Reflow两阶段：CICIDS seed1107。
* Reflow三阶段：PSM seed1103、1104、1105。
* MOMENT-small release512：CICIDS、PRONTO Day2、PRONTO Day3，固定发布权重seed42。

核验包括冻结队列任务身份、模型结果/权重/分数SHA、全部原实体测试点和有限分数、每个实体的完整整数缓冲区数量、250个阈值及VUS实体宏平均重算。
7项资源记录均未触发资源保护中止。CICIDS中最长缓冲区仍为12,610，逐一评价12,611个缓冲区，没有截断。
该采集还直接核验四个新控制器的实际进程，并确认CFM、RCGAN、TimeGAN、MaelNet原训练进程仍在运行。

验证与失败保留
--------------

validation.json和validation.txt记录一次完整相关测试调用：59项通过。
覆盖原控制器AST仅更换锁、全部四条冻结队列、资源门槛、PID复用/错误命令/工作子进程/配置变更拒绝、原TAB范围数学和稀疏Affiliation差分验证。

最初误登记2个线程被原线程一致性检查拦截，未退出任何旧控制器，未启动评价工作进程。
failed_cpu2_preflight.json、failed_cpu2_registration.json和failed_registration_test_source.py保留该失败。
configs/runtime/fm_readonly_tab_range_queues.v1.json是失败登记，不能作为当前运行配置；当前使用冻结的v2.json及对应handoff登记。

解释边界
--------

这7项是对已有完整模型分数的新增评价，不是新增训练种子，也不是重新训练MOMENT。
验证阈值Affiliation与依赖测试标签的ratio诊断保留为不同字段；VUS缓冲区使用测试事件长度属于原评价定义，不用于模型校准。
原始TAB训练框架及论文作者等价认证仍未取得。没有新增全局指标发布，最新全局汇总保持v24；其数值快照时间不因本次切换而更新。
原范围仍为45篇论文、681项方法/基线/消融记录、273项原始数据任务及已采用基线；所有实验尚未完成。

当前进程状态应重新核验实际PID及创建时间；本目录中的状态均为注明采集时刻的证据，不能单独证明未来仍存活。
