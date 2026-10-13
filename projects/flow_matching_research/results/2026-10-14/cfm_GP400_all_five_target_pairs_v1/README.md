# GP400 两种目标解释的完整五种子自动比较

此任务固定原附录 Sine 数据的 seed1103–1107，配对 paper-literal 和 Gaussian-conditional 共十个模型。每个模型要求完整 400 epoch、250000 optimizer updates、80 条原始测试轨迹及每条 500 个坐标。每对原数据 SHA256、模型 overrides 与预算相同，模型参数仅 `target_mode` 不同。保留全部 285 个原 CFM 规范槽位，未新增训练种子或替换数据。

注册时已有 8/10 个完整模型证明，仍缺 conditional seed1106、1107。seed1106 实际在原科学进程中继续训练；seed1107 待原队列运行。新自动核验进程 PID1944、create_time1791935817.2184107，等待全部十个原生模型重放与独立全网格证明。等待阶段不导入 NumPy 或 Torch，不执行模型训练、推理或调参。当前没有完整五种子比较结果，不能把本次注册计为模型完成。

十个证明齐备后，自动重新校验完整证明链、模型／结果／数据／重放数组 SHA256，以及全部 400000 个预测坐标；每对测试轨迹身份、时间和真实值必须逐位相同。任一缺失、重复、非有限、预算不完整或证明不一致的模型都不能被成功子集替代。

随后报告两种目标的五个逐种子 MSE、均值、种子 STD/SE、逐种子配对差，以及固定 bootstrap seed20261014 的 10000 次分层配对 bootstrap 区间。外层单位为完整的原始数据／训练种子配对，内层为该配对中的完整测试轨迹；500 个时间坐标不作为独立样本。计算采用有界批次，保存模型无需额外推理。

原始种子同时影响数据生成、划分和模型训练，因此外层变化描述联合的数据与训练随机性，不是孤立的训练 RNG 效应。仅有五个外层组，区间属于描述性不确定性，不能据此认证作者等价、总体 SOTA 或严格 TAB 异常检测性能。完整五种子比较即使完成，也只完成这一已注册对照，整个 45 篇／681 方法基线消融／273 数据任务研究仍未完成。

21 项相关验证通过，覆盖全部五个种子要求、缺失与重复拒绝、轨迹分组、外层种子变化不能被内层恒定误差抹掉、时间单元扁平化拒绝、原始预算与范围，以及既有全数组与模型证明链校验；见 `validation.xml`。这些测试只验证统计和核验流程，不作为科学模型性能。

配置入口：`configs/reproducibility/fm_cfm_GP400_all_five_original_target_pairs.2026-10-14_v1.json`。过程状态：`experiments/runs/fm_cfm_gp400_five_target_pairs_v1/status.json`。成功后才生成 `all_five_full_original_target_pair_comparison.json` 和 `frozen_all_ten_complete_model_snapshot.json`；当前缺口在实际注册与进程状态证据中明确保留。

文献：[CFM-TS，OpenReview Hqn4Aj7xrQ](https://openreview.net/forum?id=Hqn4Aj7xrQ)。
