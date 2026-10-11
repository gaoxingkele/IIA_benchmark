完整预算的事故恢复与新完成 SMD 实验 v4
====================================

本轮保留旧恢复队列的全部 11 个原始实验槽，补充两个原 Pi-Transformer/PSM/source_runtime seed1104、seed1105 的停止且未完成尝试，形成 13 个唯一原始实验槽。旧队列的 5 项完整结果已经独立验证身份和模型/分数或作者 artifact SHA，保留原结果。其余 8 项需完成，13 不能解释为本轮新增完成数量。

恢复范围包括工业异常检测、Rectified FM 60轮控制、TimesNet、Pi-Transformer 两分支，以及 MaelNet 原多阶段实验。旧 Rectified/CICIDS seed1103 和 Pi/source_runtime/PSM seed42 恢复尝试均明确被内存保护中止，本轮登记新输出目录完整重跑；旧的 4 项未启动尝试保留原任务身份与输出路径。新增 Pi seed1104 的终止原因不从检查点推断，seed1105 的原内存中止回执保留。没有改变原模型、超参数、输入、原始种子、作者阶段、线程、训练预算或评价配方。

旧控制器 PID12828 已从实际进程表确认不存在；旧状态文件仍写 waiting_for_heavy_lock，不能据此认为它仍在工作。保留原状态快照，启动新的实际控制器 PID33668/create_time1791688953.664792。启动前检查所有登记原任务和尝试的实际/分离模型进程均不存在。原其他模型进程未停止。初次实际新状态为 waiting_for_heavy_lock，尚无新恢复模型子进程；等待同一个现有 CPU 串行资源锁，不增加模型并发。

运行控制复用原 run_resource_recovery.main 的完整 AST，唯一改动为资源锁路径；原 guarded_wait 所有语句、返回、原紧急条件和轮询保持，只追加进程树 RSS/private 上限。原恢复队列 24 GiB 物理内存启动线保留，提交内存启动线由16加强为32 GiB，紧急保留线由8加强为16 GiB，RSS/private 上限10/12 GiB。使用与已运行 CPU 基线、熵正则 FM、Pi 和 MOMENT 相同的真实串行锁。原 run_resource_recovery_job 调用原冻结实现，没有新的替代模型。

55 项相关检查通过（18.61秒）。检查覆盖全部原实验绑定、原设置及评价、只改尝试ID/输出、单种子不重复、原完整控制器AST、只加强原资源保护条件，以及实际guard函数的RSS/private/commit四种边界。实际冷探针覆盖全部13项，没有导入numpy/torch。保存探针、测试结果、原交接/失败/完整结果证据及源 SHA。

另外，原 Anomaly Transformer 注册本地适配/SMD seed1104 在运行期间完成，原控制器已自动启动 seed1105。seed1104完整10轮、2线程、窗口100、22,750参数；完整注册数据18实体、256,589测试点。原资源回执exit0、没有内存保护中止，模型和分数SHA通过；重新核验全部训练/验证/测试分数长度、有限性、测试标签与冻结输入一致，并按原0.1/1/5三个验证阈值完全重算实体、micro和macro指标，与原result逐项完全一致。主阈值无PA的单种子F1为0.031758369885070234（3.1758369885070234%），不是完整五种子结论或原论文结果。

证据在 new_completed_smd_seed1104/independent_full_score_verification.json、原result及资源回执副本；没有在这里声称重新运行模型推理。这里的18实体是原本地注册数据范围，不能替代作者完整SMD28实体认证。此新完成实验尚未更新冻结的 all_algorithm_dataset_metrics_v26 全表，原论文与其它协议数值未改。

其它训练实际进程继续运行；actual_execution_observations.json保留第一时刻的五个旧训练句柄，next_full_smd_seed1105_actual.json保留新SMD种子的实际句柄。原45篇论文、681条方法/基线/消融、273条原始数据任务及采用基线全部范围仍待完成，作者等价和完整TAB训练尚未认证。注册恢复、通过探针或控制器存活均不等于实验完成。
