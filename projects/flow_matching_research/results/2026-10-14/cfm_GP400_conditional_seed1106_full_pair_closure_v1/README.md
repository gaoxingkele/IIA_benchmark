# GP conditional400／Sine／seed1106 最终完整核验

附录原400epoch／250000 updates／80×500×1模型已完成原生模型重放及guarded独立全网格核验。全部模型、原始数据、结果、重放数组、两层证明和资源凭证SHA256重新检查，模型重放逐位相同，最大预测差0。原源与两次核验均正常结束、资源未中止；原参数和预算未改变。

同一原数据、同一划分、同一训练种子、相同模型参数及完整预算，与paper-literal目标配对，只有预先登记的`target_mode`不同：

| 目标 | 完整原网格MSE |
| --- | --- |
| paper-literal | 0.013290877221230776 |
| Gaussian-conditional | 0.0004647279883639597 |

conditional减literal的配对MSE差为-0.012826149232866818；固定seed20261014、5000次完整轨迹配对bootstrap区间为[-0.018566965362009438,-0.00772450151598762]。该区间条件于这一对已训练模型，不把500个时间点当独立组，不代表完整五训练／数据种子区间。

主规范强核验快照为139/285，两个目标的完整种子配对已有1103–1106四组。原seed1107继续训练，十模型完整五种子自动比较仍需要其原生与独立证明，未提前输出五种子结论。作者等价、严格TAB和整个45篇／681方法消融／273数据任务完成均未认证。

完整证明见 `new_complete_native_and_independent_certificate/`，配对统计见 `complete_original_target_seed_pair.json`，配置入口为 `configs/reproducibility/fm_cfm_GP400_original_target_pair_seed1106.2026-10-14_v1.json`。本次仅读取已完成证明和全部数组，额外训练与模型推理均为零。

文献：[Conditional Flow Matching for Time Series Modelling](https://openreview.net/forum?id=Hqn4Aj7xrQ)。
