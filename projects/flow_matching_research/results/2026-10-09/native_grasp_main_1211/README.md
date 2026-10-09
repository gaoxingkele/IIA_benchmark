# GRASP 主模型：完整重跑与独立核验

SWAN，种子1103；完整1500轮、93,000次更新、10套完整推断配方。72,000个测试点与3,935个验证点均按原始行身份核验，保存的检查点、标签、分数、归因数组及指标复算通过。

同一原始槽位由首个完整核验的精确预算重跑解决；原失败产物保留。这里只完成1/10训练种子，不能据此宣称与论文十种子均值一致。

| 协议 | 源样本数 | 流评估数 | 指标 | 本地值 | 论文值 | 本地减论文 |
|---|---:|---:|---|---:|---:|---:|
| paper_test_label_oracle_no_PA | 5 | 10 | AP | 0.505820 | 0.7556 | -0.249780 |
| paper_test_label_oracle_no_PA | 5 | 10 | ROC | 0.678885 | 0.8758 | -0.196915 |
| paper_test_label_oracle_no_PA | 5 | 10 | Best_F1 | 0.495332 | 0.6886 | -0.193268 |
| validation_only_1_percent_no_PA_control | 5 | 10 | precision | 0.361534 | — | — |
| validation_only_1_percent_no_PA_control | 5 | 10 | recall | 0.780050 | — | — |
| validation_only_1_percent_no_PA_control | 5 | 10 | f1 | 0.494076 | — | — |
| paper_test_label_oracle_no_PA | 5 | 1 | AP | 0.494359 | — | — |
| paper_test_label_oracle_no_PA | 5 | 1 | ROC | 0.659596 | — | — |
| paper_test_label_oracle_no_PA | 5 | 1 | Best_F1 | 0.483771 | — | — |
| validation_only_1_percent_no_PA_control | 5 | 1 | precision | 0.370090 | — | — |
| validation_only_1_percent_no_PA_control | 5 | 1 | recall | 0.682438 | — | — |
| validation_only_1_percent_no_PA_control | 5 | 1 | f1 | 0.479917 | — | — |
| paper_test_label_oracle_no_PA | 5 | 2 | AP | 0.497137 | — | — |
| paper_test_label_oracle_no_PA | 5 | 2 | ROC | 0.664299 | — | — |
| paper_test_label_oracle_no_PA | 5 | 2 | Best_F1 | 0.485251 | — | — |
| validation_only_1_percent_no_PA_control | 5 | 2 | precision | 0.359569 | — | — |
| validation_only_1_percent_no_PA_control | 5 | 2 | recall | 0.740945 | — | — |
| validation_only_1_percent_no_PA_control | 5 | 2 | f1 | 0.484176 | — | — |
| paper_test_label_oracle_no_PA | 5 | 5 | AP | 0.502903 | — | — |
| paper_test_label_oracle_no_PA | 5 | 5 | ROC | 0.674497 | — | — |
| paper_test_label_oracle_no_PA | 5 | 5 | Best_F1 | 0.492689 | — | — |
| validation_only_1_percent_no_PA_control | 5 | 5 | precision | 0.360663 | — | — |
| validation_only_1_percent_no_PA_control | 5 | 5 | recall | 0.770348 | — | — |
| validation_only_1_percent_no_PA_control | 5 | 5 | f1 | 0.491306 | — | — |
| paper_test_label_oracle_no_PA | 5 | 20 | AP | 0.507273 | — | — |
| paper_test_label_oracle_no_PA | 5 | 20 | ROC | 0.680988 | — | — |
| paper_test_label_oracle_no_PA | 5 | 20 | Best_F1 | 0.497001 | — | — |
| validation_only_1_percent_no_PA_control | 5 | 20 | precision | 0.362259 | — | — |
| validation_only_1_percent_no_PA_control | 5 | 20 | recall | 0.784876 | — | — |
| validation_only_1_percent_no_PA_control | 5 | 20 | f1 | 0.495719 | — | — |
| paper_test_label_oracle_no_PA | 5 | 50 | AP | 0.508153 | — | — |
| paper_test_label_oracle_no_PA | 5 | 50 | ROC | 0.682209 | — | — |
| paper_test_label_oracle_no_PA | 5 | 50 | Best_F1 | 0.497729 | — | — |
| validation_only_1_percent_no_PA_control | 5 | 50 | precision | 0.362694 | — | — |
| validation_only_1_percent_no_PA_control | 5 | 50 | recall | 0.786866 | — | — |
| validation_only_1_percent_no_PA_control | 5 | 50 | f1 | 0.496523 | — | — |
| paper_test_label_oracle_no_PA | 1 | 10 | AP | 0.505892 | — | — |
| paper_test_label_oracle_no_PA | 1 | 10 | ROC | 0.678960 | — | — |
| paper_test_label_oracle_no_PA | 1 | 10 | Best_F1 | 0.495446 | — | — |
| validation_only_1_percent_no_PA_control | 1 | 10 | precision | 0.362139 | — | — |
| validation_only_1_percent_no_PA_control | 1 | 10 | recall | 0.777662 | — | — |
| validation_only_1_percent_no_PA_control | 1 | 10 | f1 | 0.494159 | — | — |
| paper_test_label_oracle_no_PA | 2 | 10 | AP | 0.505779 | — | — |
| paper_test_label_oracle_no_PA | 2 | 10 | ROC | 0.678822 | — | — |
| paper_test_label_oracle_no_PA | 2 | 10 | Best_F1 | 0.495542 | — | — |
| validation_only_1_percent_no_PA_control | 2 | 10 | precision | 0.361805 | — | — |
| validation_only_1_percent_no_PA_control | 2 | 10 | recall | 0.779303 | — | — |
| validation_only_1_percent_no_PA_control | 2 | 10 | f1 | 0.494179 | — | — |
| paper_test_label_oracle_no_PA | 10 | 10 | AP | 0.505831 | — | — |
| paper_test_label_oracle_no_PA | 10 | 10 | ROC | 0.678905 | — | — |
| paper_test_label_oracle_no_PA | 10 | 10 | Best_F1 | 0.495406 | — | — |
| validation_only_1_percent_no_PA_control | 10 | 10 | precision | 0.361202 | — | — |
| validation_only_1_percent_no_PA_control | 10 | 10 | recall | 0.781294 | — | — |
| validation_only_1_percent_no_PA_control | 10 | 10 | f1 | 0.494015 | — | — |
| paper_test_label_oracle_no_PA | 20 | 10 | AP | 0.505848 | — | — |
| paper_test_label_oracle_no_PA | 20 | 10 | ROC | 0.678938 | — | — |
| paper_test_label_oracle_no_PA | 20 | 10 | Best_F1 | 0.495383 | — | — |
| validation_only_1_percent_no_PA_control | 20 | 10 | precision | 0.361168 | — | — |
| validation_only_1_percent_no_PA_control | 20 | 10 | recall | 0.780896 | — | — |
| validation_only_1_percent_no_PA_control | 20 | 10 | f1 | 0.493903 | — | — |

论文表7中的TSMixer在SWAN上的AP/ROC/Best-F1分别为0.7556/0.8758/0.6886；当前完整本地单种子数值明显偏低。原作者代码未公开，图估计/投影、归一化、训练选择和窗口聚合仍有本地解释；当前原始文件38个特征，而论文文字记39个变量。原因需要受控消融和十种子复验，不能只归因于评价阈值。

论文配方的Best-F1使用测试标签搜索阈值；验证控制仅使用验证分数第99百分位，均无PA，两个协议分开。

[全部60条指标](algorithm_dataset_metrics.csv)；[完整独立数组审计](independent_array_audit.json)；[原始结果快照](result_snapshot.json)；[论文数值比较](paper_comparison.json)；[受控资源记录](audit_resource_receipt.json)。

完整45篇论文和681条方法/基线/消融的目标保持不变；本记录不代表整个目标或GRASP全部实验完成。

??????[??????????](attribution_array_audit.json)????/???????????????????????????????????????????????????????
