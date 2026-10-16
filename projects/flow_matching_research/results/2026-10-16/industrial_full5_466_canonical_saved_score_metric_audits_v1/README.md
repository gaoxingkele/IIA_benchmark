# 五个工业划分：466个原始条件的完整分数与独立指标核验

463项原目录完整结果全部通过；另3项SKAB资源恢复结果仅id和输出目录不同，数据、种子、模型、全部参数/窗口和原评估设置逐字段一致，经单独受保护核验后映射回原条件，各计一次。
合计466/550个CPU原条件有完整分数与独立指标证据；剩余84项为75个原始熵运输失败记录和9个稳定熵运输完整条件，原失败保留。GPU850和完整全局范围未缩减。
这不是新增466次训练或检查点神经重放；检查点只核对文件哈希，历史源执行不追认当前32GiB等资源门槛。3个恢复结果的原始resource_usage完整保留。

| 数据划分 | 已核验原条件 | 原需条件 | 原测试点数/项 | 异常点数/项 |
|---|---:|---:|---:|---:|
| tep_classic | 95 | 110 | 20160 | 16800 |
| skab | 95 | 110 | 36656 | 12879 |
| pronto_day4 | 89 | 110 | 11700 | 4843 |
| pronto_day2 | 95 | 110 | 12420 | 11698 |
| pronto_day3 | 92 | 110 | 21300 | 16980 |

全部31715个评分和元数据数组、完整标签/时间戳/行号/故障字符串均核对；1398组固定验证0.1%/1%/5%无PA阈值、微观/逐实体/事件指标通过独立公式复算。
保持完整337数组独立原始预处理证据、冻结代码及模型/分数/原始文件哈希；独立复算不调用源指标函数，不导入NumPy/Torch/sklearn。未独立重跑bootstrap、VUS或Affiliation。

110个原模型—数据集五种子组中89组完整、6组部分、15组无完整结果；只对89完整组提供267个固定阈值预算下的均值、样本STD、min/max和df4的95% t区间，部分组不统计、不补零。
种子区间只描述初始化变异。PRONTO三个轮换共用原日期，不能当三个独立数据源；正常训练/验证段使用原标签选择，属于标签辅助协议。SKAB验证有异常，验证1%不是正常总体或跨分布测试误报率保证。
各方法容量、原始20/40/60轮预算和复现状态分别保存；PSM配置迁移和本地FM评分适配不等于原论文复现，不能声称相同种子就是计算量完全匹配。

验证1%下的完整五种子结果（只做已冻结设置的描述性比较，不据测试结果选超参数）：

| 划分 | 配置 | F1均值 | 样本STD | AUROC均值 | AP均值 |
|---|---|---:|---:|---:|---:|
| tep_classic | fm_strict_usad__psm__registered | 0.832105 | 0.001750 | 0.859262 | 0.965419 |
| tep_classic | fm_strict_usad__psm__signed_paper_loss | 0.831361 | 0.002091 | 0.857314 | 0.965089 |
| tep_classic | fm_strict_tranad__psm__registered | 0.761331 | 0.002994 | 0.887078 | 0.977274 |
| tep_classic | fm_objective_target | 0.732081 | 0.003420 | 0.888648 | 0.977200 |
| tep_classic | fm_objective_ot | 0.731832 | 0.003062 | 0.890985 | 0.977679 |
| tep_classic | fm_objective_sb_stable_v2 | 0.731479 | 0.004900 | 0.887587 | 0.976978 |
| tep_classic | fm_objective_sf2m_flow_only_stable_v2 | 0.731479 | 0.004900 | 0.887587 | 0.976978 |
| tep_classic | fm_rectified_60epoch_budget_control | 0.731432 | 0.002449 | 0.890682 | 0.977676 |
| tep_classic | fm_rectified_40epoch_budget_control | 0.731061 | 0.002012 | 0.888723 | 0.977260 |
| tep_classic | fm_objective_independent | 0.730604 | 0.002643 | 0.889980 | 0.977490 |
| tep_classic | fm_objective_rectified | 0.730604 | 0.002643 | 0.889980 | 0.977490 |
| tep_classic | fm_objective_sf2m_stable_v2 | 0.730588 | 0.004335 | 0.888192 | 0.977092 |
| tep_classic | fm_reflow_3stage | 0.730526 | 0.003174 | 0.891304 | 0.977736 |
| tep_classic | fm_reflow_2stage | 0.730444 | 0.002947 | 0.890811 | 0.977647 |
| tep_classic | fm_objective_vp | 0.729787 | 0.002138 | 0.891685 | 0.977797 |
| tep_classic | fm_strict_timesnet__psm__registered | 0.612447 | 0.010601 | 0.732020 | 0.936110 |
| tep_classic | fm_strict_itransformer__psm__registered | 0.557398 | 0.006692 | 0.723174 | 0.934172 |
| tep_classic | fm_strict_dcdetector__psm__registered | 0.028155 | 0.004178 | 0.404660 | 0.809885 |
| tep_classic | fm_strict_anomaly_transformer__psm__registered | 0.026373 | 0.001736 | 0.468803 | 0.835523 |
| skab | fm_strict_usad__psm__registered | 0.516005 | 0.001776 | 0.674037 | 0.633656 |
| skab | fm_objective_vp | 0.510206 | 0.000391 | 0.658022 | 0.591310 |
| skab | fm_objective_sf2m_stable_v2 | 0.509810 | 0.000562 | 0.646947 | 0.568899 |
| skab | fm_strict_tranad__psm__registered | 0.509802 | 0.000359 | 0.677208 | 0.637654 |
| skab | fm_strict_usad__psm__signed_paper_loss | 0.509769 | 0.004188 | 0.679032 | 0.641558 |
| skab | fm_objective_ot | 0.509655 | 0.000482 | 0.638719 | 0.550659 |
| skab | fm_objective_sb_stable_v2 | 0.509601 | 0.000368 | 0.644456 | 0.563593 |
| skab | fm_objective_sf2m_flow_only_stable_v2 | 0.509601 | 0.000368 | 0.644456 | 0.563593 |
| skab | fm_objective_target | 0.509466 | 0.000711 | 0.637567 | 0.550253 |
| skab | fm_objective_independent | 0.509209 | 0.000646 | 0.637441 | 0.549567 |
| skab | fm_objective_rectified | 0.509209 | 0.000646 | 0.637441 | 0.549567 |
| skab | fm_reflow_3stage | 0.509112 | 0.000664 | 0.641799 | 0.557373 |
| skab | fm_reflow_2stage | 0.509045 | 0.000547 | 0.641226 | 0.556431 |
| skab | fm_rectified_40epoch_budget_control | 0.507581 | 0.000905 | 0.624892 | 0.527667 |
| skab | fm_rectified_60epoch_budget_control | 0.507144 | 0.000872 | 0.619909 | 0.521104 |
| skab | fm_strict_timesnet__psm__registered | 0.198290 | 0.005218 | 0.554765 | 0.430283 |
| skab | fm_strict_itransformer__psm__registered | 0.195811 | 0.009043 | 0.555937 | 0.431904 |
| skab | fm_strict_dcdetector__psm__registered | 0.022592 | 0.004360 | 0.472803 | 0.338942 |
| skab | fm_strict_anomaly_transformer__psm__registered | 0.019442 | 0.001269 | 0.427430 | 0.320380 |
| pronto_day4 | fm_strict_anomaly_transformer__psm__registered | 0.049669 | 0.001718 | 0.480756 | 0.419808 |
| pronto_day4 | fm_strict_dcdetector__psm__registered | 0.020039 | 0.003476 | 0.468322 | 0.397537 |
| pronto_day4 | fm_strict_itransformer__psm__registered | 0.013923 | 0.002093 | 0.539136 | 0.462087 |
| pronto_day4 | fm_strict_timesnet__psm__registered | 0.004447 | 0.001521 | 0.558825 | 0.475870 |
| pronto_day4 | fm_objective_ot | 0.004286 | 0.000368 | 0.462867 | 0.377187 |
| pronto_day4 | fm_reflow_3stage | 0.004203 | 0.000184 | 0.400798 | 0.344741 |
| pronto_day4 | fm_objective_independent | 0.004121 | 0.000000 | 0.463218 | 0.377897 |
| pronto_day4 | fm_objective_rectified | 0.004121 | 0.000000 | 0.463218 | 0.377897 |
| pronto_day4 | fm_objective_vp | 0.004121 | 0.000000 | 0.426152 | 0.356261 |
| pronto_day4 | fm_reflow_2stage | 0.004121 | 0.000000 | 0.418756 | 0.352676 |
| pronto_day4 | fm_strict_tranad__psm__registered | 0.004121 | 0.000000 | 0.434710 | 0.360376 |
| pronto_day4 | fm_rectified_40epoch_budget_control | 0.004121 | 0.000291 | 0.471913 | 0.385524 |
| pronto_day4 | fm_rectified_60epoch_budget_control | 0.004039 | 0.000184 | 0.472890 | 0.386122 |
| pronto_day4 | fm_objective_target | 0.003957 | 0.000469 | 0.463794 | 0.378149 |
| pronto_day4 | fm_strict_usad__psm__registered | 0.000000 | 0.000000 | 0.456466 | 0.371131 |
| pronto_day4 | fm_strict_usad__psm__signed_paper_loss | 0.000000 | 0.000000 | 0.409522 | 0.353194 |
| pronto_day2 | fm_rectified_60epoch_budget_control | 0.166946 | 0.000240 | 0.740475 | 0.978277 |
| pronto_day2 | fm_rectified_40epoch_budget_control | 0.166917 | 0.000187 | 0.747698 | 0.978831 |
| pronto_day2 | fm_strict_usad__psm__registered | 0.166143 | 0.007885 | 0.681805 | 0.970660 |
| pronto_day2 | fm_objective_independent | 0.164760 | 0.000639 | 0.717108 | 0.975313 |
| pronto_day2 | fm_objective_rectified | 0.164760 | 0.000639 | 0.717108 | 0.975313 |
| pronto_day2 | fm_objective_target | 0.164299 | 0.001633 | 0.718803 | 0.975536 |
| pronto_day2 | fm_objective_ot | 0.163174 | 0.002019 | 0.710470 | 0.974522 |
| pronto_day2 | fm_objective_sb_stable_v2 | 0.155779 | 0.009831 | 0.710824 | 0.974961 |
| pronto_day2 | fm_objective_sf2m_flow_only_stable_v2 | 0.155779 | 0.009831 | 0.710824 | 0.974961 |
| pronto_day2 | fm_objective_sf2m_stable_v2 | 0.154316 | 0.006831 | 0.708870 | 0.974728 |
| pronto_day2 | fm_objective_vp | 0.110471 | 0.005297 | 0.710927 | 0.974623 |
| pronto_day2 | fm_reflow_2stage | 0.086194 | 0.007324 | 0.726795 | 0.976459 |
| pronto_day2 | fm_strict_tranad__psm__registered | 0.082245 | 0.010895 | 0.675399 | 0.969625 |
| pronto_day2 | fm_strict_usad__psm__signed_paper_loss | 0.034891 | 0.068935 | 0.709400 | 0.976639 |
| pronto_day2 | fm_strict_dcdetector__psm__registered | 0.019390 | 0.002236 | 0.506659 | 0.941669 |
| pronto_day2 | fm_strict_timesnet__psm__registered | 0.013506 | 0.005239 | 0.555308 | 0.953766 |
| pronto_day2 | fm_reflow_3stage | 0.012962 | 0.006153 | 0.733609 | 0.977437 |
| pronto_day2 | fm_strict_itransformer__psm__registered | 0.011388 | 0.001923 | 0.551696 | 0.952608 |
| pronto_day2 | fm_strict_anomaly_transformer__psm__registered | 0.008578 | 0.001970 | 0.523648 | 0.950468 |
| pronto_day3 | fm_strict_usad__psm__registered | 0.252246 | 0.000000 | 0.354380 | 0.724316 |
| pronto_day3 | fm_strict_tranad__psm__registered | 0.234463 | 0.010436 | 0.395507 | 0.735216 |
| pronto_day3 | fm_rectified_40epoch_budget_control | 0.203669 | 0.015031 | 0.426199 | 0.741105 |
| pronto_day3 | fm_objective_ot | 0.200215 | 0.007576 | 0.381933 | 0.728847 |
| pronto_day3 | fm_rectified_60epoch_budget_control | 0.199669 | 0.008059 | 0.441464 | 0.745232 |
| pronto_day3 | fm_objective_target | 0.194646 | 0.006754 | 0.380926 | 0.728535 |
| pronto_day3 | fm_objective_independent | 0.191145 | 0.006740 | 0.377849 | 0.727954 |
| pronto_day3 | fm_objective_rectified | 0.191145 | 0.006740 | 0.377849 | 0.727954 |
| pronto_day3 | fm_objective_vp | 0.179236 | 0.015686 | 0.379484 | 0.727503 |
| pronto_day3 | fm_reflow_2stage | 0.174145 | 0.028827 | 0.385813 | 0.729001 |
| pronto_day3 | fm_reflow_3stage | 0.130011 | 0.018682 | 0.385244 | 0.728009 |
| pronto_day3 | fm_strict_usad__psm__signed_paper_loss | 0.071241 | 0.092169 | 0.338023 | 0.706692 |
| pronto_day3 | fm_strict_timesnet__psm__registered | 0.039603 | 0.003105 | 0.315425 | 0.703740 |
| pronto_day3 | fm_strict_itransformer__psm__registered | 0.035969 | 0.001148 | 0.341709 | 0.714053 |
| pronto_day3 | fm_strict_dcdetector__psm__registered | 0.021622 | 0.004055 | 0.598332 | 0.828335 |
| pronto_day3 | fm_strict_anomaly_transformer__psm__registered | 0.013840 | 0.004102 | 0.566628 | 0.813593 |

TEP95的旧报告/守卫哈希和指标保持；源目录缺口87中3项有完整独立恢复，现在仅计原条件一次，因此为466个不同原条件、84个仍待完整执行。
38项恢复身份/直接核验相关测试及60项共享指标/恢复相关测试通过。小型夹具只验证流程，不算性能；现有数值任务没有停止，所有原数据与失败均保留。
本核验没有新增训练、神经推理或训练种子，不认证历史作者、完整TAB或全部论文效果一致；整体目标未完成。

已核验的性能差异与边界
----------------------

TEP验证1%下，FM F1=0.730604、USAD F1=0.832105；FM AUROC=0.889980、USAD AUROC=0.859262。正常点误报率分别0.4048%和19.6429%。F1排序、排序指标与实际误报代价在这里并不一致。
SKAB验证集本身含188个异常点，并且各完整实验运行之间分布不同；原验证1%规则下多种方法的测试正常点误报率约62%，不能把名义验证尾部比例写成测试正常误报率。此处报告分布事实，不据此证明唯一因果机制。
PRONTO day4的FM点级F1约0.0041、AUROC约0.4632，day3的AUROC约0.3778。退化同时出现在多种方法，独立指标和原始预处理核对一致，当前证据不支持把差异解释成一次指标公式计算错误；也不能据此排除其他建模问题。
PRONTO day2异常先验约94.19%，常数排序AP即约0.9419，一直报警F1约0.9701；FM AP约0.9753仍需结合AUROC约0.7171及固定阈值F1约0.1648解释。先验控制仅作诊断，不是相同报警预算的对手。
给出FM在预先固定0.1%、1%、5%验证预算下全部15组完整五种子控制，以及与USAD同划分同种子的25组差值和5个完整配对种子汇总；没有按测试成绩重新选阈值、方向、模型或超参数。
这些是本地评分适配和PSM配置迁移结果。原论文原始评分函数、作者训练流程、PA或测试集校准策略要依据各自单独复现证据核对，不能把本表差异直接归因于所有作者代码错误或指标泄漏。
配对保留原参数量与训练轮数，但结构、窗口和实际更新预算可能不同，配对区间描述这五次初始化条件，不认证容量匹配的组件因果效应。
