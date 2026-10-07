# 流匹配参考文献 ARA 工程

本集合含 45 篇独立参考论文，按逻辑、证据、实现及追溯四层建档。每篇有来源哈希/版本/页码、方法与约束、可证伪主张、实验设计、代码/数据映射及明确的实验状态。

配置真源：benchmark 仓库 configs/reproducibility/flow_matching_ara.v1.json。集合覆盖已注册 FM 及基础、八个指定竞争方法、后续版本、用户补充的数据/方法来源、TAB 与 Anomaly Transformer；并非递归的全部引用网络。

本轮只做素材核验和工程建档。原作者结果与本地实验结果分开；当前完整复现与统一异常检测对比仍需实际运行证据。未逐项转录的表格有明确状态，不填造数字。

[补件核对](archive_reconciliation.md) · [方法族映射](method_families.md) · [比较与失配分析](comparison_plan.md) · [结构化索引](index.json)

| 参考 | 原任务 | 方法族 | ARA入口 |
|---|---|---|---|
| grasp | time_series_anomaly_detection | graph_spectral_fm | [工程](papers/grasp/PAPER.md) |
| giflow | imputation | graph_informed_cfm | [工程](papers/giflow/PAPER.md) |
| cfmi | imputation | conditional_fm | [工程](papers/cfmi/PAPER.md) |
| tsflow | generation_and_forecasting | gp_conditional_fm | [工程](papers/tsflow/PAPER.md) |
| dfm | time_series_anomaly_detection | dual_fm | [工程](papers/dfm/PAPER.md) |
| prismflow | time_series_generation | residual_expert_fm | [工程](papers/prismflow/PAPER.md) |
| csdi | imputation | conditional_diffusion | [工程](papers/csdi/PAPER.md) |
| sssd | imputation_and_forecasting | state_space_diffusion | [工程](papers/sssd/PAPER.md) |
| brits | imputation | bidirectional_rnn | [工程](papers/brits/PAPER.md) |
| saits | imputation | attention_imputation | [工程](papers/saits/PAPER.md) |
| grin | imputation | graph_recurrent | [工程](papers/grin/PAPER.md) |
| diffusion_ts | time_series_generation | interpretable_diffusion | [工程](papers/diffusion_ts/PAPER.md) |
| ganf | time_series_anomaly_detection | graph_normalizing_flow | [工程](papers/ganf/PAPER.md) |
| catch | time_series_anomaly_detection | frequency_channel_attention | [工程](papers/catch/PAPER.md) |
| dcdetector | time_series_anomaly_detection | contrastive_dual_attention | [工程](papers/dcdetector/PAPER.md) |
| flow_matching | generative_foundation | conditional_vector_field | [工程](papers/flow_matching/PAPER.md) |
| rectified_flow | generative_foundation | straight_path_reflow | [工程](papers/rectified_flow/PAPER.md) |
| ot_cfm | generative_foundation | minibatch_ot_cfm | [工程](papers/ot_cfm/PAPER.md) |
| tg_msfm | imputation | time_gated_multiscale_fm | [工程](papers/tg_msfm/PAPER.md) |
| jfi | online_imputation_and_supervised_downstream | joint_flow_imputation | [工程](papers/jfi/PAPER.md) |
| sf2m | generative_foundation | score_and_flow_matching | [工程](papers/sf2m/PAPER.md) |
| building_stochastic_interpolants | generative_foundation | stochastic_interpolants | [工程](papers/building_stochastic_interpolants/PAPER.md) |
| stochastic_interpolants_framework | generative_foundation | flow_diffusion_unification | [工程](papers/stochastic_interpolants_framework/PAPER.md) |
| spectral_mean_flow | sequence_modeling | spectral_mean_flow | [工程](papers/spectral_mean_flow/PAPER.md) |
| cfm_ts | continuous_time_dynamics | trajectory_conditional_fm | [工程](papers/cfm_ts/PAPER.md) |
| rectified_flow_ot_theory | generative_foundation | rectified_transport_theory | [工程](papers/rectified_flow_ot_theory/PAPER.md) |
| instaflow | text_to_image_generation | one_step_rectified_flow | [工程](papers/instaflow/PAPER.md) |
| flow_mismatching | image_anomaly_detection | velocity_mismatch_ad | [工程](papers/flow_mismatching/PAPER.md) |
| flow_matching_guide | foundation_and_code_guide | fm_reference | [工程](papers/flow_matching_guide/PAPER.md) |
| maelnet | time_series_anomaly_detection | ensemble_dual_transformer | [工程](papers/maelnet/PAPER.md) |
| pi_transformer | time_series_anomaly_detection | prior_dual_attention | [工程](papers/pi_transformer/PAPER.md) |
| dt_la | time_series_anomaly_detection | dual_transformer_latent_amplification | [工程](papers/dt_la/PAPER.md) |
| shcl | time_series_anomaly_detection | subsequence_contrastive | [工程](papers/shcl/PAPER.md) |
| kgl | time_series_anomaly_detection | kan_graph_lstm | [工程](papers/kgl/PAPER.md) |
| crossad | time_series_anomaly_detection | cross_scale_cross_window | [工程](papers/crossad/PAPER.md) |
| timesnet | general_time_series | periodic_2d_model | [工程](papers/timesnet/PAPER.md) |
| moment | time_series_foundation | masked_pretraining | [工程](papers/moment/PAPER.md) |
| moment_long_context | forecasting_and_classification | compressive_channel_memory | [工程](papers/moment_long_context/PAPER.md) |
| psm | dataset_and_anomaly_localization | spectral_autoencoder | [工程](papers/psm/PAPER.md) |
| omnianomaly | time_series_anomaly_detection | stochastic_recurrent_flow | [工程](papers/omnianomaly/PAPER.md) |
| sensitivehue | time_series_anomaly_detection | heteroscedastic_sensitivity | [工程](papers/sensitivehue/PAPER.md) |
| usad | time_series_anomaly_detection | adversarial_autoencoder | [工程](papers/usad/PAPER.md) |
| swat_dataset | dataset_provenance | industrial_testbed | [工程](papers/swat_dataset/PAPER.md) |
| tab | evaluation_protocol | tsad_benchmark | [工程](papers/tab/PAPER.md) |
| anomaly_transformer | time_series_anomaly_detection | association_discrepancy | [工程](papers/anomaly_transformer/PAPER.md) |

更新与验证（从 benchmark 根运行）：

```powershell
python -m scripts.literature.import_flow_matching_archives
python -m scripts.literature.build_flow_matching_ara
python -m scripts.literature.verify_flow_matching_ara
```
