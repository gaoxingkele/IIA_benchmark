# GP400 原始目标变种的第二个完整种子配对

新增完成附录 GP Gaussian-conditional／Sine／seed1104：400 epoch、250000 次优化更新、全部 80 条测试轨迹及每条 500 个坐标。原始训练资源凭证正常结束；保存模型完整重放与原始预测逐位一致（最大差 0），独立全数组复核通过。本轮重新检查模型、数据、结果、重放数组和全部证明链的 SHA256，归档于 `new_closed_GP400_gaussian_seed1104/`。冻结时 CFM 已完成原生与独立核验 137/285 个原注册模型。

配对使用同一原始数据文件、同一划分、同一训练种子、相同完整预算和模型参数；两种配置只有训练前已注册的 `target_mode` 不同。

| 训练种子 | paper-literal MSE | Gaussian-conditional MSE | conditional − literal | 完整轨迹配对 bootstrap 95% 区间 |
| --- | --- | --- | --- | --- |
| 1103 | 0.000506740136 | 0.008315532802 | +0.007808792666 | [0.005948692302, 0.009932501611] |
| 1104 | 0.018872643797 | 0.001895292496 | −0.016977351300 | [−0.019804730602, −0.014284012756] |

seed1103 的 conditional 误差较小轨迹为 1/80；seed1104 为 79/80。两个已完成种子的优劣方向相反，因而单个完整、可逐位重放的结果仍不足以确定整体变种排名。尚未确定这种反转的训练机制，也不能据此归因于作者代码错误。继续执行原注册的 seed1105、1106、1107，不根据测试表现选择目标公式或改变预算。

每个区间进行 5000 次完整轨迹配对重采样，bootstrap seed 为 20261014，条件于该对已训练模型。500 个时间坐标不作为独立样本；两个种子的轨迹也不合并为独立训练重复。未报告训练种子置信区间，完整五种子比较尚未完成。本轮仅读取已有结果和全部数组，未追加模型推理或优化更新。

这项 CFM-TS 生成轨迹结果不能作为严格 TAB 异常检测结果，也未证明论文作者指标等价。研究范围保持 45 篇论文、681 项方法／基线／消融记录、273 项原始数据任务及全部 285 个 CFM 注册模型。其他实际训练进程与原数据均未修改或中断。

复核入口：`configs/reproducibility/fm_cfm_GP400_original_target_pair_seed1104.2026-10-14_v1.json`；结果见 `complete_seed1104_paired_original_target_variants.json` 和 `two_complete_seed_pairs_with_uncertainty_boundary.json`。相关验证结果见 `validation.xml`。

文献：[CFM-TS，OpenReview Hqn4Aj7xrQ](https://openreview.net/forum?id=Hqn4Aj7xrQ)。
