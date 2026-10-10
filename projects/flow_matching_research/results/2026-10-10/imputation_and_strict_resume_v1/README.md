# 原始插补队列恢复与TimesNet原始种子补跑

2026-10-10T15:45:18.809454+00:00

恢复GRIN全部10项原始实验，以及CFMI/CSDI全部270项工业插补迁移实验。完整冻结模型命令、数据、种子与训练/采样预算不变；保留原始依赖、原有插补队列锁、共享GPU锁和完整资源保护。

GRIN当前等待200项SAITS/BRITS原始实验；工业迁移等待原始实验范围审查完成。这些等待项不计为完成。

TimesNet在SMD上核验了4个原始种子的完整结果、检查点和分数文件身份。seed1106被已有空目录阻挡；新登记同一完整科学任务，只更改尝试ID和输出路径，保留旧目录。补跑不增加种子，不挑选测试成绩。

原始运行器结果完成后，仍须独立核验指标和训练产物，再更新正式指标表。本报告没有发布新算法性能。

[实际进程、完整队列、来源校验与剩余工作](runtime_observations.json)。

| 控制器 | PID | 当前进程身份核验 | Python子进程数 |
|---|---:|---|---:|
| cfm_original | 18116 | True | 2 |
| cfm_exact_recovery | 11508 | True | 0 |
| grasp_full | 1756 | True | 0 |
| giflow_full | 21984 | True | 0 |
| spectral_table1_full | 29248 | True | 2 |
| strict_cpu | 17052 | True | 0 |
| entropic_cpu | 25148 | True | 0 |
| pi_cpu | 2400 | True | 2 |
| moment_cpu | 22036 | True | 0 |
| crossad | 18488 | True | 0 |
| maelnet_stage_resume | 31656 | True | 0 |
| saits_original_resume | 32196 | True | 0 |
| sdformer_full_gpu_preflight | 10740 | True | 0 |
| table2_all30_method_gated | 15648 | True | 0 |
| range_main | 21668 | True | 0 |
| range_reflow | 19576 | True | 0 |
| range_industrial | 10508 | True | 0 |
| range_moment | 31184 | True | 0 |
| grasp_new_full_audit | 31832 | True | 0 |
| industrial_empty_output_recovery | 29224 | True | 0 |
| tsad_gpu_full | 30764 | True | 0 |
| industrial_gpu_full | 32616 | True | 0 |
| grin_original_native_resume | 32760 | True | 0 |
| transfer_native_resume | 3456 | True | 0 |
| strict_empty_output_recovery | 16088 | True | 0 |

相关测试：34 passed。测试验证调度和完整性约束，不是性能结果。

完整目标仍包含45篇论文、681条方法/基线/消融记录和273条原始数据/任务记录；全范围实验尚未完成。
