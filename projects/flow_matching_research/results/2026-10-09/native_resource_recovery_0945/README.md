# 原生实验恢复与串行资源调度

核验快照：2026-10-09T01:47:16.245352+00:00（UTC）。

HBOS/COPOD原登记80个槽，已核验{'completed': 80}；其中2个通过来源与预算一致的恢复任务补齐。恢复不是新增独立重复。全部实体、原始测试点、标签、有限分数、原数据及产物SHA、实体宏平均已核验。

| 方法 | 数据集 | 指标 | 均值 | 完成/预定 |
|---|---|---|---:|---:|
| COPOD | cicids | Best_F1 | 0.373292 | 10/10 |
| COPOD | cicids | PRC | 0.221469 | 10/10 |
| COPOD | cicids | ROC | 0.556690 | 10/10 |
| COPOD | smap | Best_F1 | 0.279133 | 10/10 |
| COPOD | smap | PRC | 0.191397 | 10/10 |
| COPOD | smap | ROC | 0.556677 | 10/10 |
| COPOD | smd | Best_F1 | 0.295090 | 10/10 |
| COPOD | smd | PRC | 0.213850 | 10/10 |
| COPOD | smd | ROC | 0.719338 | 10/10 |
| COPOD | swan | Best_F1 | 0.441567 | 10/10 |
| COPOD | swan | PRC | 0.305447 | 10/10 |
| COPOD | swan | ROC | 0.506458 | 10/10 |
| HBOS | cicids | Best_F1 | 0.371214 | 10/10 |
| HBOS | cicids | PRC | 0.241775 | 10/10 |
| HBOS | cicids | ROC | 0.586479 | 10/10 |
| HBOS | smap | Best_F1 | 0.268565 | 10/10 |
| HBOS | smap | PRC | 0.189749 | 10/10 |
| HBOS | smap | ROC | 0.564434 | 10/10 |
| HBOS | smd | Best_F1 | 0.344017 | 10/10 |
| HBOS | smd | PRC | 0.265833 | 10/10 |
| HBOS | smd | ROC | 0.741925 | 10/10 |
| HBOS | swan | Best_F1 | 0.436479 | 10/10 |
| HBOS | swan | PRC | 0.292443 | 10/10 |
| HBOS | swan | ROC | 0.445162 | 10/10 |

统计方法使用测试特征拟合，Best-F1使用测试标签选择阈值，无PA；不得混入严格验证阈值主表。确定性重新拟合不报告随机训练置信区间。

并行GiFlow全batch预检出现超过11GiB的实际Windows私有内存峰值，并与原生任务失败同时发生。CUDA非法访问的根因尚未确认。并行预检已结束，原始失败、部分训练历史及诊断日志均保留。

GRASP主方法和Transformer的两个失败种子已登记同数据、同模型、同种子、完整1500轮的恢复任务，等待共享GPU锁。当前其他GRASP任务继续执行，不重启活跃模型。

GiFlow正式140项和串行预检共用原GRASP/GiFlow GPU互斥锁。新门限：提交内存启动25GiB/紧急12GiB，显存启动20GiB/紧急4GiB。两套全batch预检及实际峰值余量校验通过后才允许正式完整原生任务。batch128/max300epochs/原patience40/完整数据保持不变。诊断不计入benchmark。

[全部80个原始槽与来源](canonical_statistical_slots.csv)；[逐实体指标](statistical_per_entity.csv)；[完整审计和已核验进程](execution_audit.json)；[串行调度交接记录](giflow_serial_transition.json)。

All45 original papers and681 reviewed method/baseline/ablation records remain required. This audit resolves original native statistical seed slots and records native GPU recovery scheduling. It does not claim the overall goal or GRASP/GiFlow experiments are complete.
