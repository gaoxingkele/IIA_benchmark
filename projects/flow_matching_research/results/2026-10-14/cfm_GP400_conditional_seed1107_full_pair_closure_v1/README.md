# GP conditional400／Sine／seed1107 完整原生与独立证明

原400epoch／250000updates／80×500×1模型已正常完成训练、评分、全模型重放及guarded独立每坐标核验；资源未中止，全部当前模型、数据、结果、数组及证明SHA256再次验证。重放逐位相同、最大预测差0。

同数据、同原种子、同预算、仅target_mode不同的配对：paper-literal MSE=0.0009982016461484702；Gaussian-conditional MSE=0.005085361313724507；conditional减literal差=0.004087159667576038。条件于这两个已训练模型、以完整轨迹为单位的5000次配对bootstrap区间[0.0031535907232972966,0.005050878099000664]。此种子conditional更差，不删除该结果。

至此主队列140/285；加入两个新BB规范证明后唯一联合数量142/285。原五种子两目标完整对照已经单独完成，不由这一种子的区间推断五种子优劣；完整五种子区间包含零，见相邻 `cfm_actual142_closed_union_v1/`。全45篇／681方法消融／273数据任务、作者等价及严格TAB仍未完成或未认证。

配置入口：`configs/reproducibility/fm_cfm_GP400_original_target_pair_seed1107.2026-10-14_v1.json`；全模型证明见 `new_complete_native_and_independent_certificate/`。本次额外模型推理、训练更新均为零。
