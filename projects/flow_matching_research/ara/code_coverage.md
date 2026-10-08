# 全方法与消融代码准备情况

**结论：未全部准备好。**论文全文齐全和代码/消融齐全是不同状态。

核对45篇ARA参考，发现71个独立源码资源目录，关联35篇参考（含共享库、镜像和部分实现）。25篇有可静态定位的本地运行/模型入口；AST定位不代替环境/运行验证。

原先13项纯代码空缺已补入作者源码、核心还原或组件。仍有完整架构、任务集成、历史环境及论文等价性缺口；不能由此称全部主方法已完整实现。

没有任何一篇被本次审计认证为“全文全部消融均已准备并验收”。已有DCdetector消融脚本、SF2M教程和USAD开关等记录为部分资源；不能以文件名、库类或预检推断全部变体完整。

| 参考 | 主方法代码状态 | 本地入口 | 消融完整性 |
|---|---|---|---|
| [grasp](papers/grasp/PAPER.md) | 核心及部分消融已还原，原版等价性待核 | 符号在场 | 未逐项验收完整 |
| [giflow](papers/giflow/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [cfmi](papers/cfmi/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [tsflow](papers/tsflow/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [dfm](papers/dfm/PAPER.md) | 核心及部分消融已还原，原版等价性待核 | 符号在场 | 未逐项验收完整 |
| [prismflow](papers/prismflow/PAPER.md) | 核心及部分消融已还原，原版等价性待核 | 符号在场 | 未逐项验收完整 |
| [csdi](papers/csdi/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [sssd](papers/sssd/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [brits](papers/brits/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [saits](papers/saits/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [grin](papers/grin/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [diffusion_ts](papers/diffusion_ts/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [ganf](papers/ganf/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [catch](papers/catch/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [dcdetector](papers/dcdetector/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [flow_matching](papers/flow_matching/PAPER.md) | 只有相关库/组件 | 符号在场 | 未逐项验收完整 |
| [rectified_flow](papers/rectified_flow/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [ot_cfm](papers/ot_cfm/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [tg_msfm](papers/tg_msfm/PAPER.md) | 官方补充源码在场，等价性待核 | 尚无已定位统一入口 | 未逐项验收完整 |
| [jfi](papers/jfi/PAPER.md) | 核心及部分消融已还原，原版等价性待核 | 符号在场 | 未逐项验收完整 |
| [sf2m](papers/sf2m/PAPER.md) | 组件与教程在场 | 符号在场 | 未逐项验收完整 |
| [building_stochastic_interpolants](papers/building_stochastic_interpolants/PAPER.md) | 作者相关库在场，原版本待对齐 | 尚无已定位统一入口 | 未逐项验收完整 |
| [stochastic_interpolants_framework](papers/stochastic_interpolants_framework/PAPER.md) | 作者相关库在场，原版本待对齐 | 尚无已定位统一入口 | 未逐项验收完整 |
| [spectral_mean_flow](papers/spectral_mean_flow/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [cfm_ts](papers/cfm_ts/PAPER.md) | 核心及部分消融已还原，原版等价性待核 | 符号在场 | 未逐项验收完整 |
| [rectified_flow_ot_theory](papers/rectified_flow_ot_theory/PAPER.md) | 只有相关库/组件 | 尚无已定位统一入口 | 未逐项验收完整 |
| [instaflow](papers/instaflow/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [flow_mismatching](papers/flow_mismatching/PAPER.md) | 论文组件已还原，完整集成待做 | 符号在场 | 未逐项验收完整 |
| [flow_matching_guide](papers/flow_matching_guide/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [maelnet](papers/maelnet/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [pi_transformer](papers/pi_transformer/PAPER.md) | 第三方镜像在场，版本待核 | 尚无已定位统一入口 | 未逐项验收完整 |
| [dt_la](papers/dt_la/PAPER.md) | 核心及部分消融已还原，原版等价性待核 | 符号在场 | 未逐项验收完整 |
| [shcl](papers/shcl/PAPER.md) | 作者部分源码及本地补充分支在场 | 符号在场 | 未逐项验收完整 |
| [kgl](papers/kgl/PAPER.md) | 核心及部分消融已还原，原版等价性待核 | 符号在场 | 未逐项验收完整 |
| [crossad](papers/crossad/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [timesnet](papers/timesnet/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [moment](papers/moment/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [moment_long_context](papers/moment_long_context/PAPER.md) | 论文组件已还原，完整集成待做 | 符号在场 | 未逐项验收完整 |
| [psm](papers/psm/PAPER.md) | 作者源码在场，历史环境待验收 | 尚无已定位统一入口 | 未逐项验收完整 |
| [omnianomaly](papers/omnianomaly/PAPER.md) | 作者源码在场，历史环境待验收 | 尚无已定位统一入口 | 未逐项验收完整 |
| [sensitivehue](papers/sensitivehue/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [usad](papers/usad/PAPER.md) | 本地还原存在明确消融偏离 | 符号在场 | 未逐项验收完整 |
| [swat_dataset](papers/swat_dataset/PAPER.md) | 数据来源章节 | 尚无已定位统一入口 | 不适用 |
| [tab](papers/tab/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [anomaly_transformer](papers/anomaly_transformer/PAPER.md) | 本地转写在场 | 符号在场 | 未逐项验收完整 |

## 已定位的部分变体代码

- **grasp**：fm_native_grasp_er（`src/iia_benchmark/models/fm_native.py`，selected_code_markers_present）；fm_native_grasp_grasp（`src/iia_benchmark/models/fm_native.py`，selected_code_markers_present）；fm_native_grasp_lin（`src/iia_benchmark/models/fm_native.py`，selected_code_markers_present）；fm_native_grasp_mean（`src/iia_benchmark/models/fm_native.py`，selected_code_markers_present）
- **dfm**：fm_native_dfm_exact（`src/iia_benchmark/models/fm_native.py`，selected_code_markers_present）；fm_native_dfm_hutchinson（`src/iia_benchmark/models/fm_native.py`，selected_code_markers_present）
- **prismflow**：fm_native_prismflow_beta_zero（`src/iia_benchmark/models/fm_native.py`，selected_code_markers_present）；fm_native_prismflow_full（`src/iia_benchmark/models/fm_native.py`，selected_code_markers_present）；fm_native_prismflow_gamma_zero（`src/iia_benchmark/models/fm_native.py`，selected_code_markers_present）；fm_native_prismflow_no_balance（`src/iia_benchmark/models/fm_native.py`，selected_code_markers_present）；fm_native_prismflow_no_wta（`src/iia_benchmark/models/fm_native.py`，selected_code_markers_present）；fm_native_prismflow_vanilla_fm（`src/iia_benchmark/models/fm_native.py`，selected_code_markers_present）
- **dcdetector**：Ablation_Window_Size.sh（`experiments/runs/flow_matching_campaign/sources/dcdetector/original/scripts/Ablation_Window_Size.sh`，selected_code_markers_present）；Ablation_Multiscale.sh（`experiments/runs/flow_matching_campaign/sources/dcdetector/original/scripts/Ablation_Multiscale.sh`，selected_code_markers_present）；Ablation_encoder_layer.sh（`experiments/runs/flow_matching_campaign/sources/dcdetector/original/scripts/Ablation_encoder_layer.sh`，selected_code_markers_present）；Ablation_attention_head.sh（`experiments/runs/flow_matching_campaign/sources/dcdetector/original/scripts/Ablation_attention_head.sh`，selected_code_markers_present）
- **flow_matching**：fm_objective_target（`src/iia_benchmark/models/fm_objective_detectors.py`，selected_code_markers_present）
- **rectified_flow**：fm_objective_rectified（`src/iia_benchmark/models/fm_objective_detectors.py`，selected_code_markers_present）
- **ot_cfm**：ConditionalFlowMatcher（`experiments/runs/flow_matching_campaign/sources/ot_cfm/original/torchcfm/conditional_flow_matching.py`，selected_code_markers_present）；ExactOptimalTransportConditionalFlowMatcher（`experiments/runs/flow_matching_campaign/sources/ot_cfm/original/torchcfm/conditional_flow_matching.py`，selected_code_markers_present）；TargetConditionalFlowMatcher（`experiments/runs/flow_matching_campaign/sources/ot_cfm/original/torchcfm/conditional_flow_matching.py`，selected_code_markers_present）；SchrodingerBridgeConditionalFlowMatcher（`experiments/runs/flow_matching_campaign/sources/ot_cfm/original/torchcfm/conditional_flow_matching.py`，selected_code_markers_present）；VariancePreservingConditionalFlowMatcher（`experiments/runs/flow_matching_campaign/sources/ot_cfm/original/torchcfm/conditional_flow_matching.py`，selected_code_markers_present）；fm_objective_independent（`src/iia_benchmark/models/fm_objective_detectors.py`，selected_code_markers_present）；fm_objective_ot（`src/iia_benchmark/models/fm_objective_detectors.py`，selected_code_markers_present）；fm_objective_sb（`src/iia_benchmark/models/fm_objective_detectors.py`，selected_code_markers_present）；fm_objective_vp（`src/iia_benchmark/models/fm_objective_detectors.py`，selected_code_markers_present）
- **jfi**：fm_supplemental_jfi（`src/iia_benchmark/models/fm_supplemental.py`，selected_code_markers_present）；fm_supplemental_jfi__nfe_1（`src/iia_benchmark/models/fm_supplemental.py`，selected_code_markers_present）；fm_supplemental_jfi__nfe_10（`src/iia_benchmark/models/fm_supplemental.py`，selected_code_markers_present）；fm_supplemental_jfi__nfe_50（`src/iia_benchmark/models/fm_supplemental.py`，selected_code_markers_present）；fm_supplemental_jfi__without_jfol（`src/iia_benchmark/models/fm_supplemental.py`，selected_code_markers_present）；fm_supplemental_jfi__without_lpd（`src/iia_benchmark/models/fm_supplemental.py`，selected_code_markers_present）；fm_supplemental_jfi__without_mcb_ca（`src/iia_benchmark/models/fm_supplemental.py`，selected_code_markers_present）
- **sf2m**：SF2M joint flow/score tutorial（`experiments/runs/flow_matching_campaign/sources/ot_cfm/original/examples/2D_tutorials/SF2M_tutorial.ipynb`，selected_code_markers_present）；fm_objective_sf2m（`src/iia_benchmark/models/fm_objective_detectors.py`，selected_code_markers_present）；fm_objective_sf2m_flow_only（`src/iia_benchmark/models/fm_objective_detectors.py`，selected_code_markers_present）
- **cfm_ts**：fm_native_cfm_ts_bb_gaussian_consistent（`src/iia_benchmark/models/fm_native_trajectory.py`，selected_code_markers_present）；fm_native_cfm_ts_bb_paper_literal（`src/iia_benchmark/models/fm_native_trajectory.py`，selected_code_markers_present）；fm_native_cfm_ts_gp_gaussian_conditional（`src/iia_benchmark/models/fm_native_trajectory.py`，selected_code_markers_present）；fm_native_cfm_ts_gp_paper_literal（`src/iia_benchmark/models/fm_native_trajectory.py`，selected_code_markers_present）
- **flow_mismatching**：fm_supplemental_flow_mismatching（`src/iia_benchmark/models/fm_supplemental_author.py`，selected_code_markers_present）；fm_supplemental_flow_mismatching__class_conditional（`src/iia_benchmark/models/fm_supplemental_author.py`，selected_code_markers_present）；fm_supplemental_flow_mismatching__class_conditional__visa（`src/iia_benchmark/models/fm_supplemental_author.py`，selected_code_markers_present）；fm_supplemental_flow_mismatching__path_mean（`src/iia_benchmark/models/fm_supplemental_author.py`，selected_code_markers_present）；fm_supplemental_flow_mismatching__path_percentile_10（`src/iia_benchmark/models/fm_supplemental_author.py`，selected_code_markers_present）；fm_supplemental_flow_mismatching__path_percentile_20（`src/iia_benchmark/models/fm_supplemental_author.py`，selected_code_markers_present）；fm_supplemental_flow_mismatching__path_percentile_30（`src/iia_benchmark/models/fm_supplemental_author.py`，selected_code_markers_present）；fm_supplemental_flow_mismatching__pooled_categories（`src/iia_benchmark/models/fm_supplemental_author.py`，selected_code_markers_present）；fm_supplemental_flow_mismatching__unweighted_time（`src/iia_benchmark/models/fm_supplemental_author.py`，selected_code_markers_present）
- **maelnet**：fm_comparator_maelnet（`src/models/MaelNet.py`，missing_file）；fm_comparator_maelnet_fn_1p5_fp_0p6（`src/models/MaelNet.py`，missing_file）；fm_comparator_maelnet_fn_1p5_fp_0p8（`src/models/MaelNet.py`，missing_file）；fm_comparator_maelnet_fn_2p0_fp_0p6（`src/models/MaelNet.py`，missing_file）；fm_comparator_maelnet_fn_2p0_fp_0p8（`src/models/MaelNet.py`，missing_file）；fm_comparator_maelnet_without_slow_learner（`src/models/MaelNet.py`，missing_file）
- **pi_transformer**：fm_comparator_pi_transformer（`src/model/PiTransformer.py`，missing_file）；fm_comparator_pi_transformer_single_head（`src/model/PiTransformer.py`，missing_file）
- **dt_la**：fm_comparator_dt_la（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_dt_la_l2_no_sparse（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_dt_la_l2_sparse（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_dt_la_lambda_0p1（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_dt_la_lambda_0p3（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_dt_la_lambda_0p7（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_dt_la_lambda_1p0（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_dt_la_mrh_no_sparse（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_dt_la_score_input（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_dt_la_score_latent（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_dt_la_score_softmax（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）
- **shcl**：SHCL-VAE released branch（`experiments/runs/mtsad_protocol_audit/sources/shcl/original/model/SHCLVae.py`，selected_code_markers_present）；fm_comparator_shcl_transformer（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_shcl_transformer_backbone_cnn（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_shcl_transformer_backbone_lstm（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_shcl_transformer_backbone_rnn（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_shcl_transformer_backbone_vae（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_shcl_transformer_backbone_vae_only_pair（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_shcl_transformer_backbone_vae_only_point（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_shcl_transformer_backbone_vae_random_mask（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_shcl_transformer_backbone_vae_reconstruction（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_shcl_transformer_only_pair（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_shcl_transformer_only_point（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_shcl_transformer_random_mask（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_shcl_transformer_reconstruction（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）
- **kgl**：fm_comparator_kgl（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_kgl_without_gat（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_kgl_without_kan（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）；fm_comparator_kgl_without_lstm（`src/iia_benchmark/models/fm_comparator_reconstructions.py`，selected_code_markers_present）
- **moment_long_context**：fm_supplemental_moment_long_context（`src/iia_benchmark/models/fm_supplemental.py`，selected_code_markers_present）；fm_supplemental_moment_long_context__concatenation_relative_bias（`src/iia_benchmark/models/fm_supplemental.py`，selected_code_markers_present）；fm_supplemental_moment_long_context__finetune_beta（`src/iia_benchmark/models/fm_supplemental.py`，selected_code_markers_present）；fm_supplemental_moment_long_context__freeze_beta（`src/iia_benchmark/models/fm_supplemental.py`，selected_code_markers_present）；fm_supplemental_moment_long_context__icm_static（`src/iia_benchmark/models/fm_supplemental.py`，selected_code_markers_present）；fm_supplemental_moment_long_context__independent（`src/iia_benchmark/models/fm_supplemental.py`，selected_code_markers_present）；fm_supplemental_moment_long_context_t5（`src/iia_benchmark/models/fm_supplemental_moment.py`，selected_code_markers_present）；fm_supplemental_moment_long_context_t5__features（`src/iia_benchmark/models/fm_supplemental_moment.py`，selected_code_markers_present）；fm_supplemental_moment_long_context_t5__finetune_beta_head（`src/iia_benchmark/models/fm_supplemental_moment.py`，selected_code_markers_present）；fm_supplemental_moment_long_context_t5__freeze_beta（`src/iia_benchmark/models/fm_supplemental_moment.py`，selected_code_markers_present）；fm_supplemental_moment_long_context_t5__freeze_beta_head（`src/iia_benchmark/models/fm_supplemental_moment.py`，selected_code_markers_present）；fm_supplemental_moment_long_context_t5__icm_static（`src/iia_benchmark/models/fm_supplemental_moment.py`，selected_code_markers_present）；fm_supplemental_moment_long_context_t5__independent（`src/iia_benchmark/models/fm_supplemental_moment.py`，selected_code_markers_present）；fm_supplemental_moment_long_context_t5__reconstruction（`src/iia_benchmark/models/fm_supplemental_moment.py`，selected_code_markers_present）
- **sensitivehue**：fm_supplemental_sensitivehue（`src/iia_benchmark/models/fm_supplemental_author.py`，selected_code_markers_present）；fm_supplemental_sensitivehue__mse（`src/iia_benchmark/models/fm_supplemental_author.py`，selected_code_markers_present）；fm_supplemental_sensitivehue__nll（`src/iia_benchmark/models/fm_supplemental_author.py`，selected_code_markers_present）；fm_supplemental_sensitivehue__no_sfr（`src/iia_benchmark/models/fm_supplemental_author.py`，selected_code_markers_present）；fm_supplemental_sensitivehue__paper_sfr（`src/iia_benchmark/models/fm_supplemental_variants.py`，selected_code_markers_present）；fm_supplemental_sensitivehue__table4_beta_nll_05（`src/iia_benchmark/models/fm_supplemental_variants.py`，selected_code_markers_present）；fm_supplemental_sensitivehue__table4_beta_nll_1（`src/iia_benchmark/models/fm_supplemental_variants.py`，selected_code_markers_present）；fm_supplemental_sensitivehue__table4_compression（`src/iia_benchmark/models/fm_supplemental_variants.py`，selected_code_markers_present）；fm_supplemental_sensitivehue__table4_faithful（`src/iia_benchmark/models/fm_supplemental_variants.py`，selected_code_markers_present）；fm_supplemental_sensitivehue__table4_mask（`src/iia_benchmark/models/fm_supplemental_variants.py`，selected_code_markers_present）；fm_supplemental_sensitivehue__table4_natural（`src/iia_benchmark/models/fm_supplemental_variants.py`，selected_code_markers_present）；fm_supplemental_sensitivehue__table4_nll（`src/iia_benchmark/models/fm_supplemental_variants.py`，selected_code_markers_present）；fm_supplemental_sensitivehue__table4_revin（`src/iia_benchmark/models/fm_supplemental_variants.py`，selected_code_markers_present）
- **usad**：USAD adversarial switch and local reconstruction（`src/iia_benchmark/models/mtsad_detectors.py`，selected_code_markers_present）；Default non-adversarial ablation config（`configs/models/mtsad_usad.json`，selected_code_markers_present）

这些定位只说明相应代码片段在场，不说明论文整套消融配置和实验已经完成。

## 特殊边界

- **grasp**：代码、主消融或组件已按原文还原；仍缺：author-verified graph estimator/projection convention；exact mTSBench split/window-to-point aggregation；validation schedule/early stopping；MLP/CNN velocity architecture comparison；all sensitivity experiments and 10-seed datasets

- **dfm**：代码、主消融或组件已按原文还原；仍缺：author U-Net and learnable prior；Dopri5；MLE-CNF and exact published FM/I-CFM U-Net baselines；author SMAP entity/feature protocol；threshold and adjustment protocol；objective degeneracy/bijectivity needs mathematical audit

- **prismflow**：代码、主消融或组件已按原文还原；仍缺：author backbone/projection/schedules/hyperparameters；conditional guidance；all metric evaluator and baseline pipelines；low-data/forecasting/imputation tests

- **flow_matching**：Meta当前官方库可用；未定位2022原论文完整图像训练管线，亦没有完成时序异常适配。

- **tg_msfm**：ICLR官方补充包在场，目录名MSPF-main，含多尺度门、Heun等；实现/损失/观测投影与论文逐项等价性尚未核验。

- **jfi**：代码、主消融或组件已按原文还原；仍缺：Exact author code not located; branch reduction/separability and time embedding wiring are local choices.；Algorithm 1 line 6 omits factor t on z1; implementation follows explicit Eq.7 rather than that inconsistent line.；TEP simulation IDs, HP private data, pre-trained DCNN/SOFTS teacher pipelines, mask generator, preprocessing and exact paper hyperparameters not completed.；Online windows must contain only data available at deployment timestamp; LPD interpolates within supplied observed window.；Required frozen downstream teacher enforced when joint distillation weight > 0.；sigma/prior variance/loss weights here are explicit development defaults, not certified paper-best values.

- **sf2m**：TorchCFM包含SF2M教程及联合flow/score损失；单细胞、图像等原论文完整实验配置尚未逐项核验。

- **building_stochastic_interpolants**：两套作者相关插值库在场，原ICLR论文实验版本及全部设置尚未对齐。

- **stochastic_interpolants_framework**：作者插值库和JAX相关库在场，不等于JMLR全文各实验均已配齐。

- **cfm_ts**：代码、主消融或组件已按原文还原；仍缺：author formula interpretation；exact simulation split/count version；author network and dopri5 adjoint NODE baseline；full five-seed MSE experiments

- **rectified_flow_ot_theory**：普通RectifiedFlow源码在场；该篇固定凸成本的单目标OT变体未独立核实。

- **flow_mismatching**：代码、主消融或组件已按原文还原；仍缺：No Flow Mismatching author repository located; architecture assembled from related cited Meta U-Net.；Exact StableAdamW with update RMS clipping training pipeline, MVTec/VisA splits, category training and checkpoint reproduction remain pending.；Compared Reflect, WT-Flow, TCCM, D-Flow, ReconFlow implementations from Appendix C are not represented by this detector primitive.；Time weighting switch is a local diagnostic supported by toy analysis; it is not a claim of all paper ablations.；K/T sweep list is a development grid; all paper Table 6 rows still require exact transcription.；Original task is image detection; do not report as TSAD performance.

- **pi_transformer**：镜像源码的CPU构造、phase reshape、causal mask与prior支持四项修复及single-head配置已物化；作者现仓库仍仅README/LICENSE，完整期刊等价性待核。

- **dt_la**：MRH、双编码器、注意力熵、scaled-softmax及fit/score和显式消融已还原；未公开架构/归约/验证搜索细节待核。

- **shcl**：作者VAE源码保留，五骨干Transformer/VAE/CNN/RNN/LSTM及THM/连续性/高斯KL/评分和部分消融已补；adaptive masking及精确架构仍待对齐。

- **kgl**：GAT、cubic B-spline KAN、LSTM与去模块消融已还原；论文未公开训练目标/输出头/图邻域，本地选择显式登记，非完整原版。

- **moment_long_context**：代码、主消融或组件已按原文还原；仍缺：MOMENT patching/embedding/RevIN/head reused; T5-efficient-tiny sized random encoder prepared. Original MOMENT-Tiny/ICM checkpoints and exact pretraining not available here.；All context expansion baselines, 26 UEA SVM pipelines, one-epoch Time Series Pile and original forecasting datasets have not been run.；Original T5 relative bias/dropout retained, while compressed-memory aggregation excludes padded keys. Exact unpublished author ICM insertion semantics not independently verified.；Forecast and reconstruction/feature modes available; original SVM classification experiment remains separate.

- **psm**：官方RANSynCoders源码及同步开关在场；历史TensorFlow环境、私有生产模型与BKPI、全消融及TAB适配待验收。

- **omnianomaly**：官方OmniAnomaly及ZhuSuan/tfsnippet源码在场；TensorFlow1.12/TFP0.5历史环境与精确版本/消融对齐待验收。

- **usad**：本地含adversarial开关及两解码器；默认配置adversarial=false，已运行版本是非对抗消融，原算法长期稳定性/等价性未通过。 官方原始源码现已补入，但本地实现与原版的完整等价性仍待核验。

- **anomaly_transformer**：本地逐组件转写实现及测试在场；算法和阈值口径须与作者原版本对齐。 官方原始源码现已补入，但本地实现与原版的完整等价性仍待核验。

## 已锚定的消融工作项

- **grasp**：线性路径GRASP-lin、随机图GRASP-ER、均匀评分权重GRASP-mean（PDF9页）
- **giflow**：高斯先验FM-Gauss、仅空间先验GFM、仅时间先验TFM（PDF8页）
- **prismflow**：Vanilla FM、去WTA损失、去balance损失、训练beta=0、采样gamma=0（PDF8页）
- **tg_msfm**：时间门/多尺度头、可见性掩蔽注意力、Heun及data-consistency；具体表格变体待核（PDF1页）
- **jfi**：去联合目标JFOL、去局部先验LPD、去混合卷积及通道注意力MCB/CA（PDF10页）
- **maelnet**：去慢学习器、快/慢学习器组合及选择；其余变体待逐表核验（PDF5页）
- **pi_transformer**：无先验、单头先验、先验组件删除/固定替换、评分流、深度/宽度及训练参数（PDF12页）
- **dt_la**：潜空间重建损失替换、稀疏注意力开关、MRH阈值delta和正则lambda敏感性（PDF9页）
- **shcl**：THM与随机/自适应/点/对掩码、重建与SH异常准则、Transformer与VAE骨干（PDF10页）
- **kgl**：去GAT、KAN替换ReLU、去LSTM（PDF7页）
- **crossad**：多尺度建模、跨尺度重建、子序列表征、全局上下文（PDF9页）
- **timesnet**：二维表示、Inception替换其他块、独立/共享参数、自适应聚合（PDF15页）

上述是选定原文段落中的消融轴，并非全部附录、超参数组合或全部比较基线。各项仍须绑定代码、冻结配置、输出差异与行为测试；不是已执行任务。

新增[逐篇方法清单](paper_method_inventory.json)记录45篇主项、515条对比基线提及和121条变体/组件轴，共681条原文页码锚点。304个基线名称标签含别名，不是304个独立算法。候选源码不代表作者等效流程；尚未穷尽全部表和附录。

当前Python语法检查发现部分历史代码仍使用Python 2语法；本地BRITS运行入口采用SAITS发布版中的实现，两者算法/版本等价性尚未验收。其他源码通过语法解析也不代表依赖、CUDA扩展、checkpoint或完整实验已经可用。

既有CPU接口测试记录与[本轮代码补全和验证](code_completion.md)分别保存。合成输入及PSM训练集小片段预检仅验证实现行为和接线，不是论文或TAB成绩。已有插补预检和原结果比较单独引用，未改变运行队列；重复次数不完整的结果不标记复现完成。

完成门槛：论文算法/表行与原文版本 → 源码符号和哈希 → 每个变体冻结配置与训练/评分入口 → 数据/依赖/掩码协议 → 行为与恢复测试 → 原数据多种子运行。

[结构化总表](code_coverage.json)；每篇 `src/code/code_coverage.json` 提供其来源、入口、消融页码、补丁哈希与缺口。配置真源为benchmark的 `configs/reproducibility/fm_code_coverage.v1.json`。
