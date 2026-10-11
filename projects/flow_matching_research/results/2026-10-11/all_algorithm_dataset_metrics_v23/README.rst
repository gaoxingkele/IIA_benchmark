完整算法—数据集指标 v23
==========================

7283项指标；1793项有实测值；5490项缺结果。本次只更新九项既有指标。

| 算法 | 指标 | 完成／预定 | 均值 | STD | 95%下界 | 95%上界 | 协议 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CFM-TS/node | MSE | 5/5 | 5.761706 | 7.429543 | -3.463291 | 14.986702 | appendix_d |
| CFM-TS/node | MAE | 5/5 | 1.762039 | 1.092781 | 0.405172 | 3.118906 | appendix_d |
| CFM-TS/node | RMSE | 5/5 | 2.112989 | 1.273275 | 0.532009 | 3.693969 | appendix_d |
| GRASP/er | AP | 2/10 | 0.531214 | 0.011938 | 0.423958 | 0.638469 | paper_test_label_oracle_no_PA |
| GRASP/er | ROC | 2/10 | 0.712358 | 0.021971 | 0.514959 | 0.909757 | paper_test_label_oracle_no_PA |
| GRASP/er | Best_F1 | 2/10 | 0.526300 | 0.021361 | 0.334375 | 0.718224 | paper_test_label_oracle_no_PA |
| GRASP/er | precision | 2/10 | 0.378579 | 0.003958 | 0.343020 | 0.414138 | validation_only_1_percent_no_PA_control |
| GRASP/er | recall | 2/10 | 0.792687 | 0.032858 | 0.497473 | 1.087900 | validation_only_1_percent_no_PA_control |
| GRASP/er | f1 | 2/10 | 0.512374 | 0.010493 | 0.418101 | 0.606646 | validation_only_1_percent_no_PA_control |

CFM-TS全部85份已完成模型均通过完整原始测试网格重推断，逐点完全一致；Pendulum/NODE/附录D/10轮补齐5/5种子，17组五种子完整。全部285项原始任务仍须完成。

GRASP/ER/SWAN为2/10原始训练种子，每份1500轮、93000次更新、72000测试点、3935验证点。完整CPU重推断在冻结容差内通过，验证阈值分类无差异；保存原始训练分数并独立重算指标。测试标签选阈值与验证集第99百分位阈值、无PA的对照分别列出。

其余7274项指标及采集时间保留v22。45篇论文、681项方法基线消融、273项原始数据任务及采用基线范围不变，尚未全部完成；作者实现等价和严格TAB认证仍未取得。

完整表：results.html、all_algorithm_dataset_metrics.csv、all_metrics.md。
