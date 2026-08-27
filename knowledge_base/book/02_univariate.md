# Chapter 2 — Univariate alarm design

## 数学与方法

IID 过程的 delay timer 可用 Markov chain 计算误报率（FAR）、漏报率（MAR）与平均报警延迟（AAD）。真实序列先通过递归 Pettitt 均值变点和 t-test 划分正常/异常段，再用 KDE 估计分布；threshold、delay、deadband 以归一化加权损失网格搜索。

非 IID 情况用报警持续时间分布和 Bayesian 估计描述 FAR/MAR 的不确定性；deadband 从持续时间与幅值差异选择。Alarm Probability Plot 把相邻样本离散为 Markov 状态，计算概率/时间统计量 `Pka/Tka/Pam/Tam`，以有判别力且样本量可靠的区域选阈值。

## 算法抽取

```text
for threshold, delay, deadband in candidate_grid:
    alarm = stateful_threshold(x, threshold, delay, deadband)
    far, mar, aad = score(normal/abnormal segments, alarm)
    loss = sum(weight * [far, mar, aad] / target)
return candidate with minimum frozen-validation loss
```

`ThresholdDelayDeadband`、`evaluate_alarm_design` 与 `design_alarm` 已实现。首版未实现完整 Markov/Bayesian posterior，登记为下一步基线，不把 plug-in estimate 与 Bayesian interval 混报。

## 数据与测试

PRONTO/TEP/SKAB 可提供连续过程变量；必须存在正常段、异常开始和采样周期。合成 step-fault smoke 覆盖 threshold/delay/deadband 状态机。指标：FAR、MAR、AAD、报警率、chattering transitions。

证据：`02_univariate_design.txt`，PDF physical pages 60–138。
