Algorithm-dataset metrics v52
============================

SAITS Electricity20%已完整核验原始seed26/27，当前2/5个训练种子。仅三项指标更新两种子均值、样本标准差和Student-t区间；其余8623条记录保持原样。n=2的区间较宽，独立正态种子误差假设尚未验证；月份bootstrap只描述单种子的时间敏感性。论文历史条件对齐与严格TAB等价性未认证；45论文/681方法与消融/273原始数据任务的完整目标仍待完成。

Two original training seeds26/27 on one frozen split; sampleSTD(ddof1),SE and existing two-sided95%Student-t CI(df1). Normal independent seed-error assumption is unverified with n2; wide untruncated intervals are not five-seed or cross-dataset robustness. Ten-month bootstrap is a separate within-seed temporal sensitivity; windows/epochs/replays are not training seeds.
