# Chapter 3 — Multivariate alarm design

## 数学与方法

书中以 Normal Operating Zone（NOZ）替代逐变量固定阈值。凸 NOZ：归一化正常点，围绕目标 FAR `β` 选择代表点，Quickhull 得到多面体超平面；固定其他变量后解每个超平面对目标维的约束，可产生动态上下界；越界点可投影到凸区得到最近正常解释。非凸 NOZ 用 search cone 表达多工况边界。

Variation-direction 方法通过自适应时间尺度/遗忘因子估计 time gradient 与 volatility，再用规则矩阵禁止危险的联合变化方向。设备案例包括：泵的 time-varying regression、Bayesian/ridge update、累计误差和 binomial 假设更新；冷凝器把物理模型、参数不确定性、search-cone operating zone 与 Bayesian FAR/MAR bound 结合。

## 算法抽取

`MahalanobisAlarm` 是经典统计基线；`ConvexHullNOZAlarm` 实现归一化、稳健中心裁剪、凸包 membership、signed violation 与 conditional dynamic bounds。它是书中凸 NOZ 的透明近似，不等同作者完整的 fitness/quick-hull 搜索。

## 数据与测试

TEP、PRONTO、SKAB 支持多变量关系；切分必须覆盖不同工况且不得把异常点放进 NOZ 拟合。报告 overall 与 per-operating-mode 指标，并记录 hull facets/训练点数。高维凸包退化时必须回退 Mahalanobis/PCA 或显式报错。

证据：`03_multivariate_design.txt`，PDF physical pages 139–230。
