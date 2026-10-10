# 未完成实验恢复与完整预算登记

核验时间：2026-10-10T14:44:36.436665+00:00。机器启动时间：2026-10-09T05:30:36.276416+00:00。

保留完整目标：45篇论文及全部登记的方法、基线和消融，尚未完成。

本次恢复CFM-TS、GRASP、GiFlow、Spectral Table1、严格基线、SB/SF2M、Pi-Transformer、MOMENT及CrossAD调度。四条CPU队列共1492项，采用逐任务共享CPU资源锁；完整参数、数据、种子、预算和原结果校验器均保留。

CFM新增两个原槽位的精确重跑：NODE/Pendulum/seed1103因重启中断，seed1104因提交内存保护退出。已先完整校验此前三个恢复结果，避免重复种子；全部285个原始槽位仍保留。

Spectral Table2登记6条原始方法/数据集管线×5种子，共30项：Spectral Flow及SDFormer-AR，Sine、Stock、MuJoCo。CPU全数据检查和原TensorFlow判别器2000次更新通过；GPU完整批量检查因提交内存不足被保护终止，峰值进程提交约40.3GB。保留失败工作区和日志，未减少批量、8进程加载、更新预算或采样步数；无正式Table2成绩。

[实际进程身份、资源保护与完整剩余工作](runtime_observations.json)；[全部30项逐阶段完整更新预算](table2_full_stage_budgets.csv)。

| 队列/控制器 | PID | 核验时存活 | Python子进程数 |
|---|---:|---|---:|
| cfm_original | 18116 | True | 2 |
| cfm_exact_recovery | 11508 | True | 0 |
| grasp_full | 31300 | True | 1 |
| giflow_full | 21984 | True | 0 |
| spectral_table1_full | 29248 | True | 0 |
| strict_cpu | 17052 | True | 0 |
| entropic_cpu | 25148 | True | 1 |
| pi_cpu | 2400 | True | 0 |
| moment_cpu | 22036 | True | 0 |
| crossad | 18488 | True | 0 |

尚需继续处理：

- 完成所有恢复队列的原预算训练、全数据评价和独立审计；不以进程启动视为实验完成。
- Spectral Table2原批量GPU预检仍未通过；应逐方法验证完整接口，保留全部30项任务，解决资源限制后继续完整管线。
- 恢复MaelNet、SAITS/BRITS及其他尚未恢复的工业、迁移、范围评价控制器，并登记重启中断的原种子精确重跑。
- CrossAD剩余原批量预检案例尚未全部通过；修复具体环境或运行错误后继续完整案例和消融。
- 继续登记及完成严格基线、Pi-Transformer、MOMENT、GRASP等全部未完整原种子的精确重跑；既有完整结果优先，不能按成绩挑选。
- 继续45篇论文完整原始任务范围：其余生成、预测、图像、连续动力学和消融均保留未完成义务。

运行中和预检通过均不表示已复现论文成绩；正式成绩仍须完整训练、全数据评价、独立数组及检查点审计。
