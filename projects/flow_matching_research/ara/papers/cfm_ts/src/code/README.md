# Conditional Flow Matching for Time Series Modelling

原始ODE数据生成、64/258网络解释、dopri5、adjoint NODE及正文/附录完整预算均已实现并冻结；首组Pendulum BB原文公式400轮已完成5种子。原作者代码/随机数据未公开，公式/网络/网格解释仍有差异，285项尚未全部完成。

代码来源、提交、模型 entrypoint 和状态见 implementation_mapping.json。原始代码保留；后续修正存独立副本、补丁和行为检验，不能覆盖作者快照。

单独更新的[代码及消融覆盖审计](code_coverage.json)记录源码树哈希、已定位入口、消融原文页码和未完成项。[本篇方法及对比记录](method_inventory.json)绑定原文页码、代码候选与配置；由 scripts.flow_matching.refresh_paper_method_inventory 生成。先运行 scripts.flow_matching.audit_code_coverage 更新审计；不能由源码存在推断全部方法已实现。

原始任务数据、网络、完整预算、实际执行状态和剩余差异见 [original_task_execution.v1.json](original_task_execution.v1.json)。正文与附录计数冲突保留为独立协议，BB/GP原文与修正公式分别登记；NODE使用dopri5及adjoint梯度。
