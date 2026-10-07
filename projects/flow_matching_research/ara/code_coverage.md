# 全方法与消融代码准备情况

**结论：未全部准备好。**论文全文齐全和代码/消融齐全是不同状态。

核对45篇ARA参考，发现26个独立源码资源目录，关联29篇参考（含共享库、镜像和部分实现）。10篇有可静态定位的本地运行/模型入口；AST定位不代替环境/运行验证。

未定位已登记主方法实现：grasp, dfm, prismflow, jfi, cfm_ts, flow_mismatching, maelnet, dt_la, kgl, moment_long_context, psm, omnianomaly, sensitivehue。这是本项目已登记/已搜索本地资源范围的结论，未推断网上不存在代码。

没有任何一篇被本次审计认证为“全文全部消融均已准备并验收”。已有DCdetector消融脚本、SF2M教程和USAD开关等记录为部分资源；不能以文件名、库类或预检推断全部变体完整。

| 参考 | 主方法代码状态 | 本地入口 | 消融完整性 |
|---|---|---|---|
| [grasp](papers/grasp/PAPER.md) | 主方法未定位本地登记实现 | 尚无已定位统一入口 | 未逐项验收完整 |
| [giflow](papers/giflow/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [cfmi](papers/cfmi/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [tsflow](papers/tsflow/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [dfm](papers/dfm/PAPER.md) | 主方法未定位本地登记实现 | 尚无已定位统一入口 | 未逐项验收完整 |
| [prismflow](papers/prismflow/PAPER.md) | 主方法未定位本地登记实现 | 尚无已定位统一入口 | 未逐项验收完整 |
| [csdi](papers/csdi/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [sssd](papers/sssd/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [brits](papers/brits/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [saits](papers/saits/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [grin](papers/grin/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [diffusion_ts](papers/diffusion_ts/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [ganf](papers/ganf/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [catch](papers/catch/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [dcdetector](papers/dcdetector/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [flow_matching](papers/flow_matching/PAPER.md) | 只有相关库/组件 | 尚无已定位统一入口 | 未逐项验收完整 |
| [rectified_flow](papers/rectified_flow/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [ot_cfm](papers/ot_cfm/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [tg_msfm](papers/tg_msfm/PAPER.md) | 官方补充源码在场，等价性待核 | 尚无已定位统一入口 | 未逐项验收完整 |
| [jfi](papers/jfi/PAPER.md) | 主方法未定位本地登记实现 | 尚无已定位统一入口 | 未逐项验收完整 |
| [sf2m](papers/sf2m/PAPER.md) | 组件与教程在场 | 尚无已定位统一入口 | 未逐项验收完整 |
| [building_stochastic_interpolants](papers/building_stochastic_interpolants/PAPER.md) | 作者相关库在场，原版本待对齐 | 尚无已定位统一入口 | 未逐项验收完整 |
| [stochastic_interpolants_framework](papers/stochastic_interpolants_framework/PAPER.md) | 作者相关库在场，原版本待对齐 | 尚无已定位统一入口 | 未逐项验收完整 |
| [spectral_mean_flow](papers/spectral_mean_flow/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [cfm_ts](papers/cfm_ts/PAPER.md) | 主方法未定位本地登记实现 | 尚无已定位统一入口 | 未逐项验收完整 |
| [rectified_flow_ot_theory](papers/rectified_flow_ot_theory/PAPER.md) | 只有相关库/组件 | 尚无已定位统一入口 | 未逐项验收完整 |
| [instaflow](papers/instaflow/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [flow_mismatching](papers/flow_mismatching/PAPER.md) | 主方法未定位本地登记实现 | 尚无已定位统一入口 | 未逐项验收完整 |
| [flow_matching_guide](papers/flow_matching_guide/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [maelnet](papers/maelnet/PAPER.md) | 主方法未定位本地登记实现 | 尚无已定位统一入口 | 未逐项验收完整 |
| [pi_transformer](papers/pi_transformer/PAPER.md) | 第三方镜像在场，版本待核 | 尚无已定位统一入口 | 未逐项验收完整 |
| [dt_la](papers/dt_la/PAPER.md) | 主方法未定位本地登记实现 | 尚无已定位统一入口 | 未逐项验收完整 |
| [shcl](papers/shcl/PAPER.md) | 论文方法仅部分发布 | 尚无已定位统一入口 | 未逐项验收完整 |
| [kgl](papers/kgl/PAPER.md) | 主方法未定位本地登记实现 | 尚无已定位统一入口 | 未逐项验收完整 |
| [crossad](papers/crossad/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [timesnet](papers/timesnet/PAPER.md) | 作者源码在场，完整实验未验证 | 符号在场 | 未逐项验收完整 |
| [moment](papers/moment/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [moment_long_context](papers/moment_long_context/PAPER.md) | 主方法未定位本地登记实现 | 尚无已定位统一入口 | 未逐项验收完整 |
| [psm](papers/psm/PAPER.md) | 主方法未定位本地登记实现 | 尚无已定位统一入口 | 未逐项验收完整 |
| [omnianomaly](papers/omnianomaly/PAPER.md) | 主方法未定位本地登记实现 | 尚无已定位统一入口 | 未逐项验收完整 |
| [sensitivehue](papers/sensitivehue/PAPER.md) | 主方法未定位本地登记实现 | 尚无已定位统一入口 | 未逐项验收完整 |
| [usad](papers/usad/PAPER.md) | 本地还原存在明确消融偏离 | 符号在场 | 未逐项验收完整 |
| [swat_dataset](papers/swat_dataset/PAPER.md) | 数据来源章节 | 尚无已定位统一入口 | 不适用 |
| [tab](papers/tab/PAPER.md) | 作者源码在场，完整实验未验证 | 尚无已定位统一入口 | 未逐项验收完整 |
| [anomaly_transformer](papers/anomaly_transformer/PAPER.md) | 本地转写在场 | 符号在场 | 未逐项验收完整 |

## 已定位的部分变体代码

- **dcdetector**：Ablation_Window_Size.sh（`experiments/runs/flow_matching_campaign/sources/dcdetector/original/scripts/Ablation_Window_Size.sh`，selected_code_markers_present）；Ablation_Multiscale.sh（`experiments/runs/flow_matching_campaign/sources/dcdetector/original/scripts/Ablation_Multiscale.sh`，selected_code_markers_present）；Ablation_encoder_layer.sh（`experiments/runs/flow_matching_campaign/sources/dcdetector/original/scripts/Ablation_encoder_layer.sh`，selected_code_markers_present）；Ablation_attention_head.sh（`experiments/runs/flow_matching_campaign/sources/dcdetector/original/scripts/Ablation_attention_head.sh`，selected_code_markers_present）
- **ot_cfm**：ConditionalFlowMatcher（`experiments/runs/flow_matching_campaign/sources/ot_cfm/original/torchcfm/conditional_flow_matching.py`，selected_code_markers_present）；ExactOptimalTransportConditionalFlowMatcher（`experiments/runs/flow_matching_campaign/sources/ot_cfm/original/torchcfm/conditional_flow_matching.py`，selected_code_markers_present）；TargetConditionalFlowMatcher（`experiments/runs/flow_matching_campaign/sources/ot_cfm/original/torchcfm/conditional_flow_matching.py`，selected_code_markers_present）；SchrodingerBridgeConditionalFlowMatcher（`experiments/runs/flow_matching_campaign/sources/ot_cfm/original/torchcfm/conditional_flow_matching.py`，selected_code_markers_present）；VariancePreservingConditionalFlowMatcher（`experiments/runs/flow_matching_campaign/sources/ot_cfm/original/torchcfm/conditional_flow_matching.py`，selected_code_markers_present）
- **sf2m**：SF2M joint flow/score tutorial（`experiments/runs/flow_matching_campaign/sources/ot_cfm/original/examples/2D_tutorials/SF2M_tutorial.ipynb`，selected_code_markers_present）
- **shcl**：SHCL-VAE released branch（`experiments/runs/mtsad_protocol_audit/sources/shcl/original/model/SHCLVae.py`，selected_code_markers_present）
- **usad**：USAD adversarial switch and local reconstruction（`src/iia_benchmark/models/mtsad_detectors.py`，selected_code_markers_present）；Default non-adversarial ablation config（`configs/models/mtsad_usad.json`，selected_code_markers_present）

这些定位只说明相应代码片段在场，不说明论文整套消融配置和实验已经完成。

## 特殊边界

- **flow_matching**：Meta当前官方库可用；未定位2022原论文完整图像训练管线，亦没有完成时序异常适配。

- **tg_msfm**：ICLR官方补充包在场，目录名MSPF-main，含多尺度门、Heun等；实现/损失/观测投影与论文逐项等价性尚未核验。

- **sf2m**：TorchCFM包含SF2M教程及联合flow/score损失；单细胞、图像等原论文完整实验配置尚未逐项核验。

- **building_stochastic_interpolants**：两套作者相关插值库在场，原ICLR论文实验版本及全部设置尚未对齐。

- **stochastic_interpolants_framework**：作者插值库和JAX相关库在场，不等于JMLR全文各实验均已配齐。

- **rectified_flow_ot_theory**：普通RectifiedFlow源码在场；该篇固定凸成本的单目标OT变体未独立核实。

- **pi_transformer**：原论文链接仓库404；现有第三方镜像保留作者关联历史，未获现任作者确认，期刊版本和消融对应待核。

- **shcl**：作者发布SHCL-VAE；未定位所需SHCL-Transformer完整分支，不能替代其Table2/5结果。

- **usad**：本地含adversarial开关及两解码器；默认配置adversarial=false，已运行版本是非对抗消融，原算法长期稳定性/等价性未通过。

- **anomaly_transformer**：本地逐组件转写实现及测试在场；算法和阈值口径须与作者原版本对齐。

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

论文中引用/比较的全部基线尚未逐表建立完整注册。TAB和TSLib存在大量模型及封装，缺依赖的封装、普通层类和第三方移植不能按完整基线计数。结构化报告保存类/文件定位候选与边界。

当前Python语法检查发现原始BRITS的main.py仍使用Python 2 print语句；本地BRITS运行入口采用SAITS发布版中的实现，两者算法/版本等价性尚未验收。其他源码通过语法解析也不代表依赖、CUDA扩展、checkpoint或完整实验已经可用。

本地CPU接口测试10项通过；合成输入仅验证可调用接口。已有插补预检和原结果比较单独引用，未改变运行队列；重复次数不完整的结果不标记复现完成。

完成门槛：论文算法/表行与原文版本 → 源码符号和哈希 → 每个变体冻结配置与训练/评分入口 → 数据/依赖/掩码协议 → 行为与恢复测试 → 原数据多种子运行。

[结构化总表](code_coverage.json)；每篇 `src/code/code_coverage.json` 提供其来源、入口、消融页码、补丁哈希与缺口。配置真源为benchmark的 `configs/reproducibility/fm_code_coverage.v1.json`。
