完整算法—数据集指标 v24
==========================

7283项指标；1793项有实测值；5490项缺结果。本次只补齐GiFlow完整模型核验状态，全部数值表及行采集时间逐字节保留v23。

GiFlow已有的三份规范模型现在全部纳入同一独立证据链：released393、released394和reviewed_corrected393。逐一核验原始训练早停历史、实际优化更新、评估权重、源文件、掩码、图、缩放器和划分，重新校验全部42批原始与重推断数组。每份1747个测试窗口、14批、20步Euler，共840次已有原生前向；seed394有264837个目标，其余两份各268613个目标。所有完整分数及原生无权重批均值指标均在原冻结容差内通过，非逐点位相同预测。

只核验已完成模型，没有新增训练或种子：released仍为2/5种子，reviewed_corrected仍为1/5种子；GiFlow原始140项任务中完成3项。released测试集进入验证、每训练批重置EMA及原生重叠窗口指标限制继续保留。平方根批均值MSE不是全局目标加权RMSE。

CSV中的旧核验描述为原采集快照；当前核验状态以本README、publication_validation.json及完整模型证据为准。全部45篇论文、681项方法基线消融、273项原始数据任务及采用基线仍未完成，作者等价与严格TAB认证未取得。

证据：../giflow_all_completed_models_verified_v1/independent_all3_full_replay_union.json。完整表：results.html、all_algorithm_dataset_metrics.csv、all_metrics.md。
