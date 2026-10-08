# 代码补全与未闭合项（2026-10-08）

已扩大可执行实现及作者源码准备范围，仍未宣称全部方法、全部消融或论文成绩已复现。

| 项目 | 本轮结果 | 验收边界 |
|---|---|---|
| 本地源码资源 | 71个目录，本轮新增45个 | 目录可重复涉及同一算法；包括作者、维护者、第三方和组件 |
| 新增方法/变体配置 | 110个；95个声明本地入口、15个作者/历史环境配置 | 配置数不是算法数；入口与完整任务流程分别核验 |
| 原文方法记录 | 45个主项、515条基线提及、121条变体/组件轴 | 681条有主版本哈希和物理PDF页码；尚未穷尽所有表/附录 |
| 基线候选代码映射 | 298/515条有候选；217条尚未映射 | 候选不等于作者版本或运行验收；304名称标签包含别名 |
| 测试 | 156项通过，1项既有TranAD提示 | CPU行为与接线测试，不是 benchmark |
| PSM训练集小片段预检 | 21个配置训练和评分通过 | 所有小分段来自train.csv；无原测试集、标签或F1 |

## 已补的实现

- GRASP 图谱路径及 mean/ER/lin；DFM 双余弦目标、逆向密度积分和 exact/Hutchinson trace；PrismFlow 专家/WTA/平衡及主消融；CFM-TS BB/GP，字面公式与一致数学版本分别登记。
- DT-LA 双编码器、MRH、注意力熵与评分消融；KGL GAT+B-spline KAN+LSTM 与去模块；SHCL Transformer/VAE/CNN/RNN/LSTM 五骨干及点/成对等分支。
- JFI LPD/JFOL/U-Net及消融；SensitiveHUE 作者模型、论文公式和Table4九行变体；ICM嵌入完整Tiny MOMENT/T5网络的实现（随机初始化，不是原预训练权重）；FlowMismatching图像组件。
- independent、OT、target、SB、VP、SF2M、rectified七种目标及SF2M去score消融，共用可调用异常评分适配；评分为本地设计，不能当成生成论文原有TSAD方法。
- MaelNet官方代码、奖励/去慢学习器变体及CPU前后向检查；Pi镜像四处修复；PSM RANSynCoders、OmniAnomaly及依赖原始快照。

## 对比算法源码

新增或固定了Merlion、TODS、PyOD、USAD、AnomalyTransformer、GDN、MTAD-GAT，BeatGAN、InterFusion、MEMTO、TFMAE、CARLA、Donut、MAD-GAN、DeepSVDD、TranAD、TimeSeAD、VAE-LSTM及DAEMON相关版本；以及FFJORD、OT-Flow、ScoreSDE、ScoreFlow、TrajectoryNet、DSB、Glow、nflows，GP-VAE、GAIN、HyperImpute、PriSTI、TimeGAN、TimeVAE、TimeGrad、Cot-GAN、DiffWave。

具体身份和缺口见[逐篇方法清单](paper_method_inventory.json)、[代码覆盖](code_coverage.json)和benchmark下 `configs/acquisition/fm_*baseline_sources_2026-10-08.json`。
TimeSeAD中的THOC/Park2018 LSTM-VAE是第三方实现；SHCL的Lin2020 LSTM-VAE按原文ref17单独绑定。
DAEMON所取为WSDM2023后续源，ICDE2021等效性待核；DiffWave是官网指向的复现源；DiffTime官方补充包只有PDF，不能当作者代码；其CSDI comparator组件单独标记。

## 仍须解决

1. 217条对比记录未定位候选代码，已在结构化清单中保留名称、原文页码和缺口；它们不等于同样数量的独立算法。
2. 主方法的精确架构、未公开训练目标、完整消融与作者版本还需对齐；KGL等当前是明确记录本地选择的还原。DFM字面余弦目标存在退化解；CFM-TS公式差异没有被静默修正。
3. MaelNet完整RL流程、PSM/OmniAnomaly历史环境、Pi期刊版本对应、ICM原权重和私有数据未闭合。
4. HyperImpute当前GAIN插件忽略传入构造参数并在transform时训练；不能据插件参数宣称CFMI所述25000次训练已完成。
5. 论文原协议、TAB与部署协议仍需在原数据、多种子、冻结预算和分组划分下分别跑完。当前代码预检没有产生论文复现成绩。

SHCL更正：原PDF第12页Table7确有LSTM在PSM上96.42%，此前仅检查Table2遗漏了这一行。该作者报告值已补进ARA；不代表TAB原始F1或本地成绩。

## 核查入口

从benchmark根目录执行：

```powershell
python -m scripts.flow_matching.register_code_completion
python -m scripts.flow_matching.refresh_paper_method_inventory
python -m scripts.literature.build_flow_matching_ara
python -m scripts.flow_matching.audit_code_coverage
python -m scripts.flow_matching.verify_paper_method_inventory
python -m scripts.literature.verify_flow_matching_ara
```

配置调用入口是 `scripts/flow_matching/model_registry.py`；原始窗口评分执行器是 `scripts/flow_matching/run_configured_windows.py`。
完整预检结果位于benchmark的 `docs/reports/fm_real_train_code_preflight_2026-10-08.json`，本轮结构化汇总为 `docs/reports/flow_matching_code_completion_2026-10-08.json`。
原始论文、源码和数据保留本地，Git保存实现、配置、版本/哈希和审计证据。

素材检查追加验证：10项检查器测试通过；16个有明确空文件SHA256的Python包标记通过验证，空数据文件仍拒绝。5611个登记素材检查无新增失败，18项授权/私有等既有缺口继续保留。
