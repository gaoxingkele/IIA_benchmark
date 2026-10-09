# 未完成实验执行进度

快照时间：2026-10-09T02:24:49.180719+00:00。目标仍为全部已知未完成实验；尚未完成。

42 个已可训练窗口方法/消融配置 × 7 个完整本地数据集 × 5 个种子 = 1,470 个基础实验；另有 105 个 SB/SF2M 数值修复重跑任务。
另加入 175 个窗口基线任务：6 个已有方法及 USAD 有符号损失对照，使用相同的完整输入数组和验证段。
另加入 140 个两/三阶段 reflow 及40/60轮总训练轮数对照；保存每阶段教师、端点配对与实际轮数。对照不抵消reflow额外的ODE生成开销。
独立作者/镜像完整流程：maelnet 150个任务、600个阶段，{'failed_or_partial_preserved': 2, 'running': 1, 'pending': 147}；pi_transformer_historical_mirror 1140个任务、2280个阶段，{'failed_or_partial_preserved': 3, 'completed': 1, 'running': 1, 'pending': 1135}；crossad_complete 139个任务、556个阶段，{'completed': 2, 'failed_or_partial_preserved': 2, 'pending': 134, 'running': 1}。原协议单列，不计入严格无PA成绩。
另有工业异常检测任务 1400 项，状态 {'completed': 466, 'partial_or_failed': 84, 'pending': 850}。TEP整运行、SKAB整实验和PRONTO整日角色隔离；不是插补结果。
保留本地模型配置的训练轮数与容量；非重叠训练窗口和尾部覆盖规则已冻结，这不证明匹配原论文的更新次数、数据划分或架构。
关键训练预算差异：非重叠窗口比原作者 stride=1 的重叠训练少很多梯度更新。相同 epoch 数不能证明训练预算等同；原 stride=1 作者轨仍须独立完成，不能用这里的低分断言原方法无效。
当前任务记录：{'completed': 958, 'partial_or_failed': 195, 'pending': 2206, 'running': 3}。包含数值重试，不能解释为独立方法数或全部基础实验完成数。

CPU 与数值修复进程已核验存活；GPU 续跑等待现有插补链完成并取得共同锁。进程/状态是此快照的观察，后续以当前操作系统进程及结果哈希为准。

## 已完成的本地严格协议结果

验证段校准 1% 报警分位数，无 PA；每个原测试时间点一个有限分数。以下均为本地重建/适配，非作者等效认证。

| 模型配置 | 数据集 | 已完成/预定种子 | 点级 F1 均值 | AUROC 均值 | AP 均值 |
|---|---|---|---:|---:|---:|
| fm_objective_independent | psm | 5/5 | 0.032333 | 0.735128 | 0.550857 |
| fm_objective_ot | psm | 5/5 | 0.030904 | 0.733514 | 0.553258 |
| fm_objective_rectified | psm | 5/5 | 0.032333 | 0.735128 | 0.550857 |
| fm_objective_target | psm | 5/5 | 0.032726 | 0.735262 | 0.550569 |
| fm_objective_vp | psm | 5/5 | 0.029989 | 0.712565 | 0.551002 |
| fm_objective_independent | smap | 5/5 | 0.235262 | 0.536360 | 0.146863 |
| fm_objective_ot | smap | 5/5 | 0.233811 | 0.539347 | 0.147495 |
| fm_objective_rectified | smap | 5/5 | 0.235262 | 0.536360 | 0.146863 |
| fm_objective_target | smap | 5/5 | 0.235143 | 0.535192 | 0.146747 |
| fm_objective_vp | smap | 5/5 | 0.230372 | 0.549575 | 0.149446 |
| fm_objective_independent | msl | 5/5 | 0.072689 | 0.545220 | 0.115527 |
| fm_objective_ot | msl | 5/5 | 0.073644 | 0.545315 | 0.115515 |
| fm_objective_rectified | msl | 5/5 | 0.072689 | 0.545220 | 0.115527 |
| fm_objective_target | msl | 5/5 | 0.072819 | 0.544558 | 0.115526 |
| fm_objective_vp | msl | 5/5 | 0.077124 | 0.541408 | 0.117051 |
| fm_objective_independent | smd | 5/5 | 0.410668 | 0.767886 | 0.337081 |
| fm_objective_ot | smd | 5/5 | 0.412128 | 0.767478 | 0.337461 |
| fm_objective_rectified | smd | 5/5 | 0.410668 | 0.767886 | 0.337081 |
| fm_objective_target | smd | 5/5 | 0.409530 | 0.767068 | 0.334713 |
| fm_objective_vp | smd | 5/5 | 0.392979 | 0.756467 | 0.324163 |
| fm_objective_independent | swat | 5/5 | 0.289642 | 0.822661 | 0.727248 |
| fm_objective_ot | swat | 5/5 | 0.289527 | 0.821859 | 0.724678 |
| fm_objective_rectified | swat | 5/5 | 0.289642 | 0.822661 | 0.727248 |
| fm_objective_target | swat | 5/5 | 0.289505 | 0.823257 | 0.726629 |
| fm_objective_vp | swat | 5/5 | 0.300797 | 0.821427 | 0.728332 |
| fm_objective_independent | swan | 5/5 | 0.555188 | 0.807976 | 0.674500 |
| fm_objective_ot | swan | 5/5 | 0.555968 | 0.810315 | 0.677442 |
| fm_objective_rectified | swan | 5/5 | 0.555188 | 0.807976 | 0.674500 |
| fm_objective_target | swan | 5/5 | 0.554533 | 0.806938 | 0.673441 |
| fm_objective_vp | swan | 5/5 | 0.579502 | 0.836403 | 0.707246 |
| fm_objective_independent | cicids | 5/5 | 0.048710 | 0.620327 | 0.392767 |
| fm_objective_ot | cicids | 5/5 | 0.048663 | 0.626055 | 0.393674 |
| fm_objective_rectified | cicids | 5/5 | 0.048710 | 0.620327 | 0.392767 |
| fm_objective_target | cicids | 5/5 | 0.048064 | 0.619822 | 0.392276 |
| fm_objective_vp | cicids | 5/5 | 0.057745 | 0.634149 | 0.397584 |
| fm_objective_sb_stable_v2 | psm | 5/5 | 0.033416 | 0.717722 | 0.545647 |
| fm_objective_sf2m_stable_v2 | psm | 5/5 | 0.032784 | 0.718805 | 0.547731 |
| fm_objective_sf2m_flow_only_stable_v2 | psm | 5/5 | 0.033416 | 0.717722 | 0.545647 |
| fm_objective_sb_stable_v2 | smap | 5/5 | 0.233867 | 0.540958 | 0.147982 |
| fm_objective_sf2m_stable_v2 | smap | 5/5 | 0.232735 | 0.541786 | 0.148027 |
| fm_objective_sf2m_flow_only_stable_v2 | smap | 5/5 | 0.233867 | 0.540958 | 0.147982 |
| fm_objective_sb_stable_v2 | msl | 5/5 | 0.076244 | 0.534781 | 0.113941 |
| fm_objective_sf2m_stable_v2 | msl | 5/5 | 0.075694 | 0.533194 | 0.115100 |
| fm_objective_sf2m_flow_only_stable_v2 | msl | 5/5 | 0.076244 | 0.534781 | 0.113941 |
| fm_objective_sb_stable_v2 | smd | 5/5 | 0.408927 | 0.762697 | 0.332341 |
| fm_objective_sf2m_stable_v2 | smd | 2/5 | 0.399169 | 0.761568 | 0.326040 |
| fm_strict_timesnet__psm__registered | psm | 5/5 | 0.053941 | 0.598985 | 0.396761 |
| fm_strict_anomaly_transformer__psm__registered | psm | 5/5 | 0.014575 | 0.513439 | 0.295091 |
| fm_strict_dcdetector__psm__registered | psm | 5/5 | 0.017267 | 0.501093 | 0.278458 |
| fm_strict_tranad__psm__registered | psm | 5/5 | 0.027646 | 0.640820 | 0.478071 |
| fm_strict_usad__psm__registered | psm | 5/5 | 0.028775 | 0.720846 | 0.543615 |
| fm_strict_usad__psm__signed_paper_loss | psm | 5/5 | 0.058838 | 0.519146 | 0.304404 |
| fm_strict_itransformer__psm__registered | psm | 5/5 | 0.048369 | 0.593232 | 0.385533 |
| fm_strict_timesnet__smap__registered | smap | 5/5 | 0.038918 | 0.411246 | 0.104513 |
| fm_strict_anomaly_transformer__smap__registered | smap | 5/5 | 0.040822 | 0.547435 | 0.144792 |
| fm_strict_dcdetector__smap__registered | smap | 5/5 | 0.008984 | 0.559623 | 0.138280 |
| fm_strict_usad__smap__registered | smap | 5/5 | 0.249936 | 0.463074 | 0.135130 |
| fm_strict_usad__smap__signed_paper_loss | smap | 5/5 | 0.062489 | 0.550997 | 0.132300 |
| fm_strict_itransformer__smap__registered | smap | 5/5 | 0.045223 | 0.421727 | 0.107702 |
| fm_strict_timesnet__msl__registered | msl | 5/5 | 0.076810 | 0.566504 | 0.136118 |
| fm_strict_anomaly_transformer__msl__registered | msl | 5/5 | 0.020377 | 0.499501 | 0.103149 |
| fm_strict_dcdetector__msl__registered | msl | 5/5 | 0.018754 | 0.510673 | 0.108932 |
| fm_strict_tranad__msl__registered | msl | 5/5 | 0.081264 | 0.564902 | 0.123400 |
| fm_strict_usad__msl__registered | msl | 5/5 | 0.065173 | 0.588159 | 0.130747 |
| fm_strict_usad__msl__signed_paper_loss | msl | 5/5 | 0.137823 | 0.600799 | 0.133992 |
| fm_strict_itransformer__msl__registered | msl | 5/5 | 0.079931 | 0.579678 | 0.138575 |
| fm_strict_timesnet__smd__registered | smd | 1/5 | 0.247426 | 0.848478 | 0.364956 |
| fm_reflow_2stage | psm | 5/5 | 0.029949 | 0.725765 | 0.552078 |
| fm_reflow_2stage | smap | 5/5 | 0.230333 | 0.547532 | 0.148794 |
| fm_reflow_2stage | msl | 5/5 | 0.075546 | 0.539035 | 0.117010 |
| fm_reflow_2stage | smd | 5/5 | 0.390249 | 0.753078 | 0.322908 |
| fm_reflow_2stage | swat | 5/5 | 0.296223 | 0.820484 | 0.725875 |
| fm_reflow_2stage | swan | 5/5 | 0.577738 | 0.830896 | 0.703194 |
| fm_reflow_2stage | cicids | 5/5 | 0.058868 | 0.621817 | 0.390273 |
| fm_reflow_3stage | psm | 5/5 | 0.030127 | 0.720384 | 0.550238 |
| fm_reflow_3stage | smap | 5/5 | 0.230133 | 0.553047 | 0.149801 |
| fm_reflow_3stage | msl | 5/5 | 0.075017 | 0.538887 | 0.116631 |
| fm_reflow_3stage | smd | 5/5 | 0.382961 | 0.749359 | 0.319022 |
| fm_reflow_3stage | swat | 5/5 | 0.300950 | 0.820895 | 0.726619 |
| fm_reflow_3stage | swan | 5/5 | 0.588824 | 0.842724 | 0.716953 |
| fm_reflow_3stage | cicids | 5/5 | 0.061199 | 0.624536 | 0.391916 |
| fm_rectified_40epoch_budget_control | psm | 5/5 | 0.033572 | 0.737732 | 0.550026 |
| fm_rectified_40epoch_budget_control | smap | 5/5 | 0.232921 | 0.535617 | 0.146476 |
| fm_rectified_40epoch_budget_control | msl | 5/5 | 0.072524 | 0.542974 | 0.114567 |
| fm_rectified_40epoch_budget_control | smd | 5/5 | 0.418723 | 0.772170 | 0.340303 |
| fm_rectified_40epoch_budget_control | swat | 5/5 | 0.288095 | 0.822282 | 0.723544 |
| fm_rectified_40epoch_budget_control | swan | 5/5 | 0.555437 | 0.804995 | 0.672827 |
| fm_rectified_40epoch_budget_control | cicids | 5/5 | 0.048674 | 0.619370 | 0.394101 |
| fm_rectified_60epoch_budget_control | psm | 5/5 | 0.033730 | 0.736344 | 0.551276 |
| fm_rectified_60epoch_budget_control | smap | 5/5 | 0.231837 | 0.535460 | 0.146316 |
| fm_rectified_60epoch_budget_control | msl | 5/5 | 0.072324 | 0.543456 | 0.114749 |
| fm_rectified_60epoch_budget_control | smd | 5/5 | 0.413955 | 0.774513 | 0.341945 |
| fm_rectified_60epoch_budget_control | swat | 5/5 | 0.287477 | 0.821908 | 0.723513 |
| fm_rectified_60epoch_budget_control | swan | 5/5 | 0.558325 | 0.806741 | 0.675242 |
| fm_rectified_60epoch_budget_control | cicids | 4/5 | 0.048355 | 0.615679 | 0.392124 |
| fm_moment_pretrained_small__release512 | psm | 1/1 | 0.036639 | 0.555556 | 0.337271 |
| fm_moment_pretrained_small__release512 | smap | 1/1 | 0.105609 | 0.454654 | 0.126343 |
| fm_moment_pretrained_small__release512 | msl | 1/1 | 0.086996 | 0.559740 | 0.132951 |
| fm_moment_pretrained_small__release512 | smd | 1/1 | 0.174824 | 0.813581 | 0.260849 |
| fm_moment_pretrained_small__release512 | swat | 1/1 | 0.083518 | 0.262044 | 0.113999 |
| fm_moment_pretrained_small__release512 | swan | 1/1 | 0.426205 | 0.684565 | 0.495399 |
| fm_moment_pretrained_small__release512 | cicids | 1/1 | 0.028173 | 0.420767 | 0.285714 |
| fm_moment_pretrained_small__release512 | tep_classic | 1/1 | 0.581320 | 0.562498 | 0.852026 |
| fm_moment_pretrained_small__release512 | skab | 1/1 | 0.240866 | 0.594941 | 0.424080 |
| fm_moment_pretrained_small__release512 | pronto_day2 | 1/1 | 0.046752 | 0.544039 | 0.953758 |
| fm_moment_pretrained_small__release512 | pronto_day3 | 1/1 | 0.097875 | 0.375337 | 0.721401 |
| fm_moment_pretrained_small__release512 | pronto_day4 | 1/1 | 0.003298 | 0.586470 | 0.473271 |
| fm_moment_pretrained_small__TAB100 | psm | 1/1 | 0.051095 | 0.585120 | 0.383768 |
| fm_moment_pretrained_small__TAB100 | smap | 1/1 | 0.047672 | 0.434559 | 0.111922 |
| fm_moment_pretrained_small__TAB100 | msl | 1/1 | 0.077288 | 0.560163 | 0.135149 |
| fm_moment_pretrained_small__TAB100 | smd | 1/1 | 0.268146 | 0.828068 | 0.363400 |
| fm_moment_pretrained_small__TAB100 | swat | 1/1 | 0.026401 | 0.241222 | 0.085623 |
| fm_moment_pretrained_small__TAB100 | swan | 1/1 | 0.363260 | 0.594773 | 0.434081 |
| fm_moment_pretrained_small__TAB100 | cicids | 1/1 | 0.038031 | 0.404400 | 0.276631 |
| fm_moment_pretrained_small__TAB100 | tep_classic | 1/1 | 0.529663 | 0.709197 | 0.929416 |
| fm_moment_pretrained_small__TAB100 | skab | 1/1 | 0.197611 | 0.556066 | 0.431646 |
| fm_moment_pretrained_small__TAB100 | pronto_day2 | 1/1 | 0.010375 | 0.558382 | 0.953429 |
| fm_moment_pretrained_small__TAB100 | pronto_day3 | 1/1 | 0.034931 | 0.329034 | 0.708615 |
| fm_moment_pretrained_small__TAB100 | pronto_day4 | 1/1 | 0.008225 | 0.550899 | 0.470046 |
| fm_moment_pretrained_base__release512 | psm | 1/1 | 0.039181 | 0.554868 | 0.336232 |
| fm_objective_independent | tep_classic | 5/5 | 0.730604 | 0.889980 | 0.977490 |
| fm_objective_ot | tep_classic | 5/5 | 0.731832 | 0.890985 | 0.977679 |
| fm_objective_rectified | tep_classic | 5/5 | 0.730604 | 0.889980 | 0.977490 |
| fm_objective_target | tep_classic | 5/5 | 0.732081 | 0.888648 | 0.977200 |
| fm_objective_vp | tep_classic | 5/5 | 0.729787 | 0.891685 | 0.977797 |
| fm_objective_sb_stable_v2 | tep_classic | 5/5 | 0.731479 | 0.887587 | 0.976978 |
| fm_objective_sf2m_stable_v2 | tep_classic | 5/5 | 0.730588 | 0.888192 | 0.977092 |
| fm_objective_sf2m_flow_only_stable_v2 | tep_classic | 5/5 | 0.731479 | 0.887587 | 0.976978 |
| fm_reflow_2stage | tep_classic | 5/5 | 0.730444 | 0.890811 | 0.977647 |
| fm_reflow_3stage | tep_classic | 5/5 | 0.730526 | 0.891304 | 0.977736 |
| fm_rectified_40epoch_budget_control | tep_classic | 5/5 | 0.731061 | 0.888723 | 0.977260 |
| fm_rectified_60epoch_budget_control | tep_classic | 5/5 | 0.731432 | 0.890682 | 0.977676 |
| fm_strict_timesnet__psm__registered | tep_classic | 5/5 | 0.612447 | 0.732020 | 0.936110 |
| fm_strict_anomaly_transformer__psm__registered | tep_classic | 5/5 | 0.026373 | 0.468803 | 0.835523 |
| fm_strict_dcdetector__psm__registered | tep_classic | 5/5 | 0.028155 | 0.404660 | 0.809885 |
| fm_strict_tranad__psm__registered | tep_classic | 5/5 | 0.761331 | 0.887078 | 0.977274 |
| fm_strict_usad__psm__registered | tep_classic | 5/5 | 0.832105 | 0.859262 | 0.965419 |
| fm_strict_itransformer__psm__registered | tep_classic | 5/5 | 0.557398 | 0.723174 | 0.934172 |
| fm_strict_usad__psm__signed_paper_loss | tep_classic | 5/5 | 0.831361 | 0.857314 | 0.965089 |
| fm_objective_independent | skab | 5/5 | 0.509209 | 0.637441 | 0.549567 |
| fm_objective_ot | skab | 5/5 | 0.509655 | 0.638719 | 0.550659 |
| fm_objective_rectified | skab | 5/5 | 0.509209 | 0.637441 | 0.549567 |
| fm_objective_target | skab | 5/5 | 0.509466 | 0.637567 | 0.550253 |
| fm_objective_vp | skab | 5/5 | 0.510206 | 0.658022 | 0.591310 |
| fm_objective_sb_stable_v2 | skab | 5/5 | 0.509601 | 0.644456 | 0.563593 |
| fm_objective_sf2m_stable_v2 | skab | 5/5 | 0.509810 | 0.646947 | 0.568899 |
| fm_objective_sf2m_flow_only_stable_v2 | skab | 5/5 | 0.509601 | 0.644456 | 0.563593 |
| fm_reflow_2stage | skab | 5/5 | 0.509045 | 0.641226 | 0.556431 |
| fm_reflow_3stage | skab | 5/5 | 0.509112 | 0.641799 | 0.557373 |
| fm_rectified_40epoch_budget_control | skab | 5/5 | 0.507581 | 0.624892 | 0.527667 |
| fm_rectified_60epoch_budget_control | skab | 5/5 | 0.507144 | 0.619909 | 0.521104 |
| fm_strict_timesnet__psm__registered | skab | 5/5 | 0.198290 | 0.554765 | 0.430283 |
| fm_strict_anomaly_transformer__psm__registered | skab | 5/5 | 0.019442 | 0.427430 | 0.320380 |
| fm_strict_dcdetector__psm__registered | skab | 5/5 | 0.022592 | 0.472803 | 0.338942 |
| fm_strict_tranad__psm__registered | skab | 5/5 | 0.509802 | 0.677208 | 0.637654 |
| fm_strict_usad__psm__registered | skab | 5/5 | 0.516005 | 0.674037 | 0.633656 |
| fm_strict_itransformer__psm__registered | skab | 5/5 | 0.195811 | 0.555937 | 0.431904 |
| fm_strict_usad__psm__signed_paper_loss | skab | 5/5 | 0.509769 | 0.679032 | 0.641558 |
| fm_objective_independent | pronto_day4 | 5/5 | 0.004121 | 0.463218 | 0.377897 |
| fm_objective_ot | pronto_day4 | 5/5 | 0.004286 | 0.462867 | 0.377187 |
| fm_objective_rectified | pronto_day4 | 5/5 | 0.004121 | 0.463218 | 0.377897 |
| fm_objective_target | pronto_day4 | 5/5 | 0.003957 | 0.463794 | 0.378149 |
| fm_objective_vp | pronto_day4 | 5/5 | 0.004121 | 0.426152 | 0.356261 |
| fm_objective_sb_stable_v2 | pronto_day4 | 3/5 | 0.004121 | 0.451857 | 0.369249 |
| fm_objective_sf2m_stable_v2 | pronto_day4 | 3/5 | 0.003984 | 0.452381 | 0.369946 |
| fm_objective_sf2m_flow_only_stable_v2 | pronto_day4 | 3/5 | 0.004121 | 0.451857 | 0.369249 |
| fm_reflow_2stage | pronto_day4 | 5/5 | 0.004121 | 0.418756 | 0.352676 |
| fm_reflow_3stage | pronto_day4 | 5/5 | 0.004203 | 0.400798 | 0.344741 |
| fm_rectified_40epoch_budget_control | pronto_day4 | 5/5 | 0.004121 | 0.471913 | 0.385524 |
| fm_rectified_60epoch_budget_control | pronto_day4 | 5/5 | 0.004039 | 0.472890 | 0.386122 |
| fm_strict_timesnet__psm__registered | pronto_day4 | 5/5 | 0.004447 | 0.558825 | 0.475870 |
| fm_strict_anomaly_transformer__psm__registered | pronto_day4 | 5/5 | 0.049669 | 0.480756 | 0.419808 |
| fm_strict_dcdetector__psm__registered | pronto_day4 | 5/5 | 0.020039 | 0.468322 | 0.397537 |
| fm_strict_tranad__psm__registered | pronto_day4 | 5/5 | 0.004121 | 0.434710 | 0.360376 |
| fm_strict_usad__psm__registered | pronto_day4 | 5/5 | 0.000000 | 0.456466 | 0.371131 |
| fm_strict_itransformer__psm__registered | pronto_day4 | 5/5 | 0.013923 | 0.539136 | 0.462087 |
| fm_strict_usad__psm__signed_paper_loss | pronto_day4 | 5/5 | 0.000000 | 0.409522 | 0.353194 |
| fm_objective_independent | pronto_day2 | 5/5 | 0.164760 | 0.717108 | 0.975313 |
| fm_objective_ot | pronto_day2 | 5/5 | 0.163174 | 0.710470 | 0.974522 |
| fm_objective_rectified | pronto_day2 | 5/5 | 0.164760 | 0.717108 | 0.975313 |
| fm_objective_target | pronto_day2 | 5/5 | 0.164299 | 0.718803 | 0.975536 |
| fm_objective_vp | pronto_day2 | 5/5 | 0.110471 | 0.710927 | 0.974623 |
| fm_objective_sb_stable_v2 | pronto_day2 | 5/5 | 0.155779 | 0.710824 | 0.974961 |
| fm_objective_sf2m_stable_v2 | pronto_day2 | 5/5 | 0.154316 | 0.708870 | 0.974728 |
| fm_objective_sf2m_flow_only_stable_v2 | pronto_day2 | 5/5 | 0.155779 | 0.710824 | 0.974961 |
| fm_reflow_2stage | pronto_day2 | 5/5 | 0.086194 | 0.726795 | 0.976459 |
| fm_reflow_3stage | pronto_day2 | 5/5 | 0.012962 | 0.733609 | 0.977437 |
| fm_rectified_40epoch_budget_control | pronto_day2 | 5/5 | 0.166917 | 0.747698 | 0.978831 |
| fm_rectified_60epoch_budget_control | pronto_day2 | 5/5 | 0.166946 | 0.740475 | 0.978277 |
| fm_strict_timesnet__psm__registered | pronto_day2 | 5/5 | 0.013506 | 0.555308 | 0.953766 |
| fm_strict_anomaly_transformer__psm__registered | pronto_day2 | 5/5 | 0.008578 | 0.523648 | 0.950468 |
| fm_strict_dcdetector__psm__registered | pronto_day2 | 5/5 | 0.019390 | 0.506659 | 0.941669 |
| fm_strict_tranad__psm__registered | pronto_day2 | 5/5 | 0.082245 | 0.675399 | 0.969625 |
| fm_strict_usad__psm__registered | pronto_day2 | 5/5 | 0.166143 | 0.681805 | 0.970660 |
| fm_strict_itransformer__psm__registered | pronto_day2 | 5/5 | 0.011388 | 0.551696 | 0.952608 |
| fm_strict_usad__psm__signed_paper_loss | pronto_day2 | 5/5 | 0.034891 | 0.709400 | 0.976639 |
| fm_objective_independent | pronto_day3 | 5/5 | 0.191145 | 0.377849 | 0.727954 |
| fm_objective_ot | pronto_day3 | 5/5 | 0.200215 | 0.381933 | 0.728847 |
| fm_objective_rectified | pronto_day3 | 5/5 | 0.191145 | 0.377849 | 0.727954 |
| fm_objective_target | pronto_day3 | 5/5 | 0.194646 | 0.380926 | 0.728535 |
| fm_objective_vp | pronto_day3 | 5/5 | 0.179236 | 0.379484 | 0.727503 |
| fm_objective_sb_stable_v2 | pronto_day3 | 4/5 | 0.173649 | 0.396407 | 0.733941 |
| fm_objective_sf2m_stable_v2 | pronto_day3 | 4/5 | 0.178907 | 0.393857 | 0.733293 |
| fm_objective_sf2m_flow_only_stable_v2 | pronto_day3 | 4/5 | 0.173649 | 0.396407 | 0.733941 |
| fm_reflow_2stage | pronto_day3 | 5/5 | 0.174145 | 0.385813 | 0.729001 |
| fm_reflow_3stage | pronto_day3 | 5/5 | 0.130011 | 0.385244 | 0.728009 |
| fm_rectified_40epoch_budget_control | pronto_day3 | 5/5 | 0.203669 | 0.426199 | 0.741105 |
| fm_rectified_60epoch_budget_control | pronto_day3 | 5/5 | 0.199669 | 0.441464 | 0.745232 |
| fm_strict_timesnet__psm__registered | pronto_day3 | 5/5 | 0.039603 | 0.315425 | 0.703740 |
| fm_strict_anomaly_transformer__psm__registered | pronto_day3 | 5/5 | 0.013840 | 0.566628 | 0.813593 |
| fm_strict_dcdetector__psm__registered | pronto_day3 | 5/5 | 0.021622 | 0.598332 | 0.828335 |
| fm_strict_tranad__psm__registered | pronto_day3 | 5/5 | 0.234463 | 0.395507 | 0.735216 |
| fm_strict_usad__psm__registered | pronto_day3 | 5/5 | 0.252246 | 0.354380 | 0.724316 |
| fm_strict_itransformer__psm__registered | pronto_day3 | 5/5 | 0.035969 | 0.341709 | 0.714053 |
| fm_strict_usad__psm__signed_paper_loss | pronto_day3 | 5/5 | 0.071241 | 0.338023 | 0.706692 |

逐种子结果、标准误/种子区间、时间块或实体区间、检查点及分数哈希均保存在 [execution_snapshot.json](execution_snapshot.json)。单种子块区间不替代跨算法配对检验。

独立 CFM 与单阶段 Rectified 在 sigma=0 时使用同一条直线路径，数值相同是预期行为。新增两/三阶段 reflow 真正生成前一流的端点配对，并在配对上重新训练；其本地异常分数仍不是原图像实验等价证明。

## 已补算的固定版本 TAB 范围指标

已核验 633 个完整分数文件的 VUS 与 Affiliation。VUS保留原250阈值、全部整数缓冲长度、inclusive ties与积分公式；完整PSM87841点与原代码执行差异为约1e-16。

| 模型配置 | 数据集 | VUS 已完成种子 | VUS ROC 均值 | VUS PR 均值 | 严格阈值 Affiliation F 均值 |
|---|---|---:|---:|---:|---:|
| fm_objective_independent | cicids | 2 | 0.698929 | 0.342803 | 0.345049 |
| fm_objective_independent | msl | 5 | 0.705197 | 0.278498 | 0.725564 |
| fm_objective_independent | psm | 5 | 0.698167 | 0.501818 | 0.585950 |
| fm_objective_independent | smap | 5 | 0.690551 | 0.294041 | 0.537718 |
| fm_objective_independent | smd | 5 | 0.814265 | 0.433968 | 0.799890 |
| fm_objective_independent | swan | 5 | 0.715689 | 0.585292 | 0.656444 |
| fm_objective_independent | swat | 5 | 0.678608 | 0.482889 | 0.717671 |
| fm_objective_ot | msl | 5 | 0.706013 | 0.277971 | 0.728107 |
| fm_objective_ot | psm | 5 | 0.697567 | 0.502244 | 0.578248 |
| fm_objective_ot | smap | 5 | 0.691844 | 0.294302 | 0.536096 |
| fm_objective_ot | smd | 5 | 0.814468 | 0.436098 | 0.799317 |
| fm_objective_ot | swan | 5 | 0.720770 | 0.589988 | 0.660748 |
| fm_objective_ot | swat | 5 | 0.679549 | 0.485110 | 0.715820 |
| fm_objective_rectified | msl | 5 | 0.705197 | 0.278498 | 0.725564 |
| fm_objective_rectified | psm | 5 | 0.698167 | 0.501818 | 0.585950 |
| fm_objective_rectified | smap | 5 | 0.690551 | 0.294041 | 0.537718 |
| fm_objective_rectified | smd | 5 | 0.814265 | 0.433968 | 0.799890 |
| fm_objective_rectified | swan | 5 | 0.715689 | 0.585292 | 0.656444 |
| fm_objective_rectified | swat | 5 | 0.678608 | 0.482889 | 0.717671 |
| fm_objective_sb_stable_v2 | psm | 5 | 0.687497 | 0.497809 | 0.587712 |
| fm_objective_sf2m_stable_v2 | psm | 4 | 0.687794 | 0.499966 | 0.578675 |
| fm_objective_target | msl | 5 | 0.704934 | 0.279624 | 0.724266 |
| fm_objective_target | psm | 5 | 0.701574 | 0.502289 | 0.581274 |
| fm_objective_target | smap | 5 | 0.690140 | 0.294203 | 0.537754 |
| fm_objective_target | smd | 5 | 0.813133 | 0.433818 | 0.800311 |
| fm_objective_target | swan | 5 | 0.714948 | 0.584402 | 0.656056 |
| fm_objective_target | swat | 5 | 0.680010 | 0.485990 | 0.718252 |
| fm_objective_vp | msl | 5 | 0.702600 | 0.284205 | 0.733116 |
| fm_objective_vp | psm | 5 | 0.671146 | 0.492057 | 0.522960 |
| fm_objective_vp | smap | 5 | 0.691672 | 0.289387 | 0.533601 |
| fm_objective_vp | smd | 5 | 0.804581 | 0.431433 | 0.798092 |
| fm_objective_vp | swan | 5 | 0.747169 | 0.617598 | 0.671753 |
| fm_objective_vp | swat | 5 | 0.682084 | 0.487826 | 0.703106 |
| fm_strict_anomaly_transformer__psm__registered | psm | 5 | 0.432373 | 0.291488 | 0.635304 |
| fm_strict_dcdetector__psm__registered | psm | 2 | 0.439577 | 0.280859 | 0.642914 |
| fm_strict_timesnet__psm__registered | psm | 5 | 0.597688 | 0.398548 | 0.767881 |
| fm_reflow_2stage | cicids | 4 | 0.697292 | 0.345104 | 0.348105 |
| fm_reflow_2stage | msl | 5 | 0.700867 | 0.285871 | 0.732293 |
| fm_reflow_2stage | psm | 5 | 0.685073 | 0.497841 | 0.547103 |
| fm_reflow_2stage | smap | 5 | 0.692927 | 0.291242 | 0.535977 |
| fm_reflow_2stage | smd | 5 | 0.795336 | 0.420790 | 0.799751 |
| fm_reflow_2stage | swan | 5 | 0.745971 | 0.616491 | 0.675825 |
| fm_reflow_2stage | swat | 5 | 0.680466 | 0.487292 | 0.706191 |
| fm_objective_independent | pronto_day2 | 5 | 0.976135 | 0.998642 | 0.557584 |
| fm_objective_independent | pronto_day3 | 5 | 0.904335 | 0.971246 | 0.666297 |
| fm_objective_independent | pronto_day4 | 5 | 0.646314 | 0.629969 | 0.390340 |
| fm_objective_independent | skab | 5 | 0.797160 | 0.736057 | 0.774492 |
| fm_objective_independent | tep_classic | 5 | 0.976290 | 0.995223 | 0.942948 |
| fm_objective_ot | pronto_day2 | 5 | 0.975452 | 0.998594 | 0.557396 |
| fm_objective_ot | pronto_day3 | 5 | 0.904840 | 0.971282 | 0.671914 |
| fm_objective_ot | pronto_day4 | 5 | 0.645902 | 0.628389 | 0.390393 |
| fm_objective_ot | skab | 5 | 0.798622 | 0.737487 | 0.773416 |
| fm_objective_ot | tep_classic | 5 | 0.976236 | 0.995193 | 0.945853 |
| fm_objective_rectified | pronto_day2 | 5 | 0.976135 | 0.998642 | 0.557584 |
| fm_objective_rectified | pronto_day3 | 5 | 0.904335 | 0.971246 | 0.666297 |
| fm_objective_rectified | pronto_day4 | 5 | 0.646314 | 0.629969 | 0.390340 |
| fm_objective_rectified | skab | 5 | 0.797160 | 0.736057 | 0.774492 |
| fm_objective_rectified | tep_classic | 5 | 0.976290 | 0.995223 | 0.942948 |
| fm_objective_sb_stable_v2 | pronto_day2 | 5 | 0.976188 | 0.998645 | 0.557018 |
| fm_objective_sb_stable_v2 | pronto_day3 | 4 | 0.907950 | 0.972265 | 0.687009 |
| fm_objective_sb_stable_v2 | pronto_day4 | 3 | 0.636532 | 0.616044 | 0.390422 |
| fm_objective_sb_stable_v2 | skab | 5 | 0.807162 | 0.750735 | 0.773940 |
| fm_objective_sb_stable_v2 | tep_classic | 5 | 0.975428 | 0.995032 | 0.948819 |
| fm_objective_sf2m_stable_v2 | pronto_day2 | 5 | 0.975919 | 0.998628 | 0.557062 |
| fm_objective_sf2m_stable_v2 | pronto_day3 | 4 | 0.907328 | 0.972101 | 0.681135 |
| fm_objective_sf2m_stable_v2 | pronto_day4 | 3 | 0.637129 | 0.618050 | 0.390340 |
| fm_objective_sf2m_stable_v2 | skab | 5 | 0.811590 | 0.756376 | 0.774487 |
| fm_objective_sf2m_stable_v2 | tep_classic | 5 | 0.975403 | 0.995018 | 0.944569 |
| fm_objective_sf2m_flow_only_stable_v2 | pronto_day2 | 5 | 0.976188 | 0.998645 | 0.557018 |
| fm_objective_sf2m_flow_only_stable_v2 | pronto_day3 | 4 | 0.907950 | 0.972265 | 0.687009 |
| fm_objective_sf2m_flow_only_stable_v2 | pronto_day4 | 3 | 0.636532 | 0.616044 | 0.390422 |
| fm_objective_sf2m_flow_only_stable_v2 | skab | 5 | 0.807162 | 0.750735 | 0.773940 |
| fm_objective_sf2m_flow_only_stable_v2 | tep_classic | 5 | 0.975428 | 0.995032 | 0.948819 |
| fm_objective_target | pronto_day2 | 5 | 0.976382 | 0.998662 | 0.557522 |
| fm_objective_target | pronto_day3 | 5 | 0.904979 | 0.971362 | 0.663851 |
| fm_objective_target | pronto_day4 | 5 | 0.646883 | 0.630595 | 0.390339 |
| fm_objective_target | skab | 5 | 0.797759 | 0.737317 | 0.772947 |
| fm_objective_target | tep_classic | 5 | 0.976239 | 0.995212 | 0.946545 |
| fm_objective_vp | pronto_day2 | 5 | 0.975270 | 0.998589 | 0.555089 |
| fm_objective_vp | pronto_day3 | 5 | 0.903143 | 0.970629 | 0.650789 |
| fm_objective_vp | pronto_day4 | 5 | 0.616521 | 0.600420 | 0.390340 |
| fm_objective_vp | skab | 5 | 0.831504 | 0.783663 | 0.776964 |
| fm_objective_vp | tep_classic | 5 | 0.976332 | 0.995198 | 0.938951 |
| fm_rectified_40epoch_budget_control | pronto_day2 | 5 | 0.979421 | 0.998867 | 0.557791 |
| fm_rectified_40epoch_budget_control | pronto_day4 | 5 | 0.653947 | 0.641289 | 0.390389 |
| fm_rectified_40epoch_budget_control | skab | 5 | 0.780206 | 0.712742 | 0.773067 |
| fm_rectified_40epoch_budget_control | tep_classic | 5 | 0.976414 | 0.995308 | 0.944949 |
| fm_rectified_60epoch_budget_control | pronto_day2 | 5 | 0.978899 | 0.998840 | 0.557825 |
| fm_rectified_60epoch_budget_control | pronto_day4 | 5 | 0.655000 | 0.643326 | 0.390340 |
| fm_rectified_60epoch_budget_control | skab | 5 | 0.774063 | 0.706324 | 0.775239 |
| fm_rectified_60epoch_budget_control | tep_classic | 5 | 0.976417 | 0.995317 | 0.941918 |
| fm_reflow_2stage | pronto_day2 | 5 | 0.977087 | 0.998712 | 0.554312 |
| fm_reflow_2stage | pronto_day3 | 5 | 0.904377 | 0.970878 | 0.653527 |
| fm_reflow_2stage | pronto_day4 | 5 | 0.611032 | 0.595473 | 0.390340 |
| fm_reflow_2stage | skab | 5 | 0.803590 | 0.745465 | 0.774031 |
| fm_reflow_2stage | tep_classic | 5 | 0.976223 | 0.995191 | 0.939546 |
| fm_reflow_3stage | pronto_day2 | 5 | 0.977928 | 0.998776 | 0.494028 |
| fm_reflow_3stage | pronto_day3 | 3 | 0.904168 | 0.970648 | 0.645587 |
| fm_reflow_3stage | pronto_day4 | 5 | 0.595280 | 0.579246 | 0.390389 |
| fm_reflow_3stage | skab | 5 | 0.804914 | 0.747594 | 0.777073 |
| fm_reflow_3stage | tep_classic | 5 | 0.976181 | 0.995186 | 0.939388 |
| fm_strict_anomaly_transformer__psm__registered | pronto_day2 | 5 | 0.967297 | 0.997646 | 0.905712 |
| fm_strict_anomaly_transformer__psm__registered | pronto_day4 | 5 | 0.717926 | 0.694036 | 0.790997 |
| fm_strict_anomaly_transformer__psm__registered | skab | 5 | 0.658215 | 0.587740 | 0.750049 |
| fm_strict_anomaly_transformer__psm__registered | tep_classic | 5 | 0.890125 | 0.972942 | 0.904345 |
| fm_strict_dcdetector__psm__registered | pronto_day2 | 5 | 0.973202 | 0.997794 | 0.967182 |
| fm_strict_dcdetector__psm__registered | pronto_day4 | 5 | 0.702741 | 0.675540 | 0.762463 |
| fm_strict_dcdetector__psm__registered | skab | 4 | 0.685294 | 0.595164 | 0.711856 |
| fm_strict_dcdetector__psm__registered | tep_classic | 5 | 0.876082 | 0.969965 | 0.884707 |
| fm_strict_itransformer__psm__registered | pronto_day2 | 5 | 0.984508 | 0.998937 | 0.763014 |
| fm_strict_itransformer__psm__registered | pronto_day4 | 5 | 0.745938 | 0.724188 | 0.649683 |
| fm_strict_itransformer__psm__registered | skab | 5 | 0.747270 | 0.673080 | 0.721763 |
| fm_strict_itransformer__psm__registered | tep_classic | 5 | 0.950915 | 0.989562 | 0.914512 |
| fm_strict_timesnet__psm__registered | pronto_day2 | 5 | 0.984372 | 0.998968 | 0.784650 |
| fm_strict_timesnet__psm__registered | pronto_day4 | 5 | 0.761214 | 0.735129 | 0.545347 |
| fm_strict_timesnet__psm__registered | skab | 5 | 0.748035 | 0.674873 | 0.711315 |
| fm_strict_timesnet__psm__registered | tep_classic | 5 | 0.955354 | 0.990342 | 0.921253 |
| fm_strict_tranad__psm__registered | pronto_day2 | 5 | 0.971351 | 0.998274 | 0.553244 |
| fm_strict_tranad__psm__registered | pronto_day4 | 5 | 0.617609 | 0.596592 | 0.390576 |
| fm_strict_tranad__psm__registered | skab | 5 | 0.881295 | 0.862251 | 0.765349 |
| fm_strict_tranad__psm__registered | tep_classic | 5 | 0.974274 | 0.994929 | 0.916555 |
| fm_strict_usad__psm__registered | pronto_day2 | 5 | 0.972063 | 0.998345 | 0.557191 |
| fm_strict_usad__psm__registered | pronto_day4 | 5 | 0.640990 | 0.627183 | — |
| fm_strict_usad__psm__registered | skab | 5 | 0.890857 | 0.879593 | 0.757419 |
| fm_strict_usad__psm__registered | tep_classic | 5 | 0.980942 | 0.996367 | 0.935752 |
| fm_strict_usad__psm__signed_paper_loss | pronto_day2 | 5 | 0.979420 | 0.998868 | 0.423748 |
| fm_strict_usad__psm__signed_paper_loss | pronto_day4 | 5 | 0.596755 | 0.587822 | — |
| fm_strict_usad__psm__signed_paper_loss | skab | 5 | 0.898785 | 0.891228 | 0.749148 |
| fm_strict_usad__psm__signed_paper_loss | tep_classic | 5 | 0.980764 | 0.996333 | 0.934338 |
| fm_moment_pretrained_small__release512 | msl | 1 | 0.720217 | 0.309355 | 0.696811 |
| fm_moment_pretrained_small__release512 | psm | 1 | 0.547813 | 0.333540 | 0.658476 |
| fm_moment_pretrained_small__release512 | skab | 1 | 0.777686 | 0.694011 | 0.835865 |
| fm_moment_pretrained_small__release512 | smap | 1 | 0.674969 | 0.251157 | 0.545930 |
| fm_moment_pretrained_small__release512 | smd | 1 | 0.810018 | 0.326123 | 0.543465 |
| fm_moment_pretrained_small__release512 | swan | 1 | 0.597565 | 0.445366 | 0.593244 |
| fm_moment_pretrained_small__release512 | swat | 1 | 0.369082 | 0.162870 | 0.704033 |
| fm_moment_pretrained_small__release512 | tep_classic | 1 | 0.918699 | 0.980888 | 0.913304 |

每个实体独立评价；macro仅平均原参考函数有定义的结果，未定义实体/种子数保留，不填零。Affiliation F、点级 F1 和 PA-F1 是不同指标，不能直接混比。VUS缓冲按测试标签事件长度产生，是已披露的评价依赖；严格阈值仍只来自验证段。完整TAB训练流程未等效认证。

## 历史记录补充与范围更正

先前 2026-10-08 表只覆盖 FM 注册队列，未纳入旧 mtsad_reproduction 目录的 137 份运行记录。TimesNet、Anomaly Transformer、DCdetector 等历史数值确实存在，因此不能据前表声称整个项目没有异常检测成绩。

旧记录可能重复种子、预算或协议；缺少新严格协议要求的冻结原始数据、完整实体/时间覆盖或检查点链，不能直接晋升为新队列已完成项。全部历史数值及协议见 [historical_mtsad_metrics.csv](historical_mtsad_metrics.csv)。未改写历史文件。

## 未完成范围仍然保留

45 篇 ARA 的原始数据集、681 条已审方法/消融记录以及每篇复现缺口，完整保留于 execution_snapshot.json。启动 1,470 个本地任务并未替代这些义务。

- 原有 540 个插补/迁移任务继续执行；原数据和活跃队列未重启。
- 原作者 MaelNet RL、Pi 期刊版本等效、CrossAD、MOMENT 权重及其他 source-only 适配器仍需完善。
- GiFlow、预测、生成、连续时间、图像、单细胞、表格原始实验及工业异常检测迁移仍未全部完成。
- TAB 核心指标/比例诊断单列：合并训练/测试分数校准且按测试标签选最佳比例。剩余 VUS/Affiliation 和完整 TAB 原作者训练流程继续执行。
- 原 Sinkhorn 不收敛日志和部分产物保留；修正版只改求解器续接与迭代上限，最终正则、边缘容差和训练轮数不变。修正版已有实际完整 PSM 成功结果，其余种子/数据集继续核验。

数据身份、通道、实体边界、训练/验证隔离、缺失修复和训练段缩放统计见 [data_manifest.json](data_manifest.json)。SMAP_P-7 仅提供训练文件，明确登记后不纳入 51 个有测试文件的实体评价。
