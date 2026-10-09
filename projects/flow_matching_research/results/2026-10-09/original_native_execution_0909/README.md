# 原生实验执行与论文数值核验

核验区间：2026-10-09T01:12:19.393536+00:00 至 2026-10-09T01:12:54.211143+00:00（UTC）。

HBOS/COPOD 完整原生任务状态：{'completed': 65, 'running': 1, 'pending': 14}，原登记共80项。所有已完成实体、原测试点数、标签数、有限分数、原数据和模型/分数哈希以及实体宏平均已核验。

GRASP 完整训练状态：{'completed': 1, 'running': 1, 'pending': 9118}，原登记9120个实体任务、480个全实体组；已完成1组。

完整GRASP产物另外核验保存的1500轮检查点、实际更新数、全验证/测试行身份与标签，并从保存的分数独立重算指标。单种子不能证明与论文十种子均值一致。

| 方法 | 数据集 | 指标 | 本地均值 | 论文均值 | 差值 | 完成/预定重复 |
|---|---|---|---:|---:|---:|---:|
| COPOD | CICIDS | Best_F1 | 0.373292 | 0.3733 | -0.000008 | 2/10 |
| COPOD | CICIDS | PRC | 0.221469 | 0.2214 | +0.000069 | 2/10 |
| COPOD | CICIDS | ROC | 0.556690 | 0.5566 | +0.000090 | 2/10 |
| COPOD | SMAP | Best_F1 | 0.279133 | 0.2841 | -0.004967 | 10/10 |
| COPOD | SMAP | PRC | 0.191397 | 0.1927 | -0.001303 | 10/10 |
| COPOD | SMAP | ROC | 0.556677 | 0.5918 | -0.035123 | 10/10 |
| COPOD | SMD | Best_F1 | 0.295090 | 0.2951 | -0.000010 | 10/10 |
| COPOD | SMD | PRC | 0.213850 | 0.2139 | -0.000050 | 10/10 |
| COPOD | SMD | ROC | 0.719338 | 0.7193 | +0.000038 | 10/10 |
| COPOD | SWAN | Best_F1 | 0.441567 | 0.4365 | +0.005067 | 10/10 |
| COPOD | SWAN | PRC | 0.305447 | 0.2792 | +0.026247 | 10/10 |
| COPOD | SWAN | ROC | 0.506458 | 0.5000 | +0.006458 | 10/10 |
| HBOS | CICIDS | Best_F1 | 0.371214 | 0.3852 | -0.013986 | 3/10 |
| HBOS | CICIDS | PRC | 0.241775 | 0.2603 | -0.018525 | 3/10 |
| HBOS | CICIDS | ROC | 0.586479 | 0.5586 | +0.027879 | 3/10 |
| HBOS | SMAP | Best_F1 | 0.268565 | 0.2664 | +0.002165 | 10/10 |
| HBOS | SMAP | PRC | 0.189749 | 0.1897 | +0.000049 | 10/10 |
| HBOS | SMAP | ROC | 0.564434 | 0.5650 | -0.000566 | 10/10 |
| HBOS | SMD | Best_F1 | 0.344017 | 0.3556 | -0.011583 | 10/10 |
| HBOS | SMD | PRC | 0.265833 | 0.2678 | -0.001967 | 10/10 |
| HBOS | SMD | ROC | 0.741925 | 0.7378 | +0.004125 | 10/10 |
| HBOS | SWAN | Best_F1 | 0.436479 | 0.4365 | -0.000021 | 10/10 |
| HBOS | SWAN | PRC | 0.292443 | 0.2792 | +0.013243 | 10/10 |
| HBOS | SWAN | ROC | 0.445162 | 0.5000 | -0.054838 | 10/10 |
| GNN | SWAN | PRC | 0.412057 | 0.5306 | -0.118543 | 1/10 |
| GNN | SWAN | ROC | 0.604945 | 0.6570 | -0.052055 | 1/10 |
| GNN | SWAN | Best_F1 | 0.465510 | 0.5188 | -0.053290 | 1/10 |

测试特征拟合的统计方法、测试最优 Best-F1 和附加验证阈值对照分别解释。论文数字来自同一已核验PDF的表1（第8页）及架构表7（第25页），保留全部204个表值和来源哈希；四位小数吻合不是作者协议等价认证。

GiFlow 的140个任务、原输出路径与训练预算保持不变。原等待控制器经过PID/创建时间/命令/无模型子进程核验后切换到显卡资源受控队列，与完整GRASP共用CUDA训练锁。原SAITS/GRIN任务和活跃模型未停止。

GiFlow控制器已核验存活：True；当前状态：waiting_for_heavy_lock。资源达到原CPU/commit门限且取得共享GPU锁后执行原始完整任务，不用缩小batch或训练轮数。

[完整执行审计](execution_audit.json)；[完整论文表值](paper_references.csv)；[逐项数值比较](paper_comparison.csv)；[统计方法逐实体指标](statistical_per_entity.csv)。

Audit complete native results only. Paper decimal agreement is not protocol certification. GRASP single-seed results do not establish agreement with a ten-seed paper mean. All 45 papers, 681 reviewed records and other original task/transfer obligations remain required.
