# Sequence Modeling with Spectral Mean Flows

这里的 mean flow 是谱均值嵌入方法，不能与其他同名平均速度一步生成法混同。

代码来源、提交、模型 entrypoint 和状态见 implementation_mapping.json。原始代码保留；后续修正存独立副本、补丁和行为检验，不能覆盖作者快照。

单独更新的[代码及消融覆盖审计](code_coverage.json)记录源码树哈希、已定位入口、消融原文页码和未完成项。[本篇方法及对比记录](method_inventory.json)绑定原文页码、代码候选与配置；由 scripts.flow_matching.refresh_paper_method_inventory 生成。先运行 scripts.flow_matching.audit_code_coverage 更新审计；不能由源码存在推断全部方法已实现。

LS4/SaShiMi 支持文献的 S4 朴素 Cauchy 回退发现遗漏共轭项。[真实 FRED 成对诊断及完整续跑](../../../../../results/2026-10-11/ls4_cauchy_corrected_resume_v3/README.md)绑定官方 S4 修正，保留下载原文源码和 40 个完整 LS4 原始槽位；修正后的训练结果与历史作者等价仍待验证。

[SaShiMi 高斯自回归重建](../../../../../results/2026-10-11/sashimi_monash_gaussian_resume_v2/README.md)补齐四个原始 Monash 数据集的 40 个完整任务；两种完整结构及原生 100 epoch 评价已通过真实 FRED 流程预检。专用基线配置未公开，作者等价未认证，本地结构敏感性对照不标为论文消融。
