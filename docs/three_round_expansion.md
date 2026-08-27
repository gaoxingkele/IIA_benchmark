# 三轮扩展闭环

本文件把每一轮都固定成 `数据集 → 引用文献 → 方法论提取 → 可支配算法/复现位 → 相似或近期论文 → 下一批数据集`。日期截点为 2026-08-27；“SOTA candidate”仅表示当前可检索的近期强基线，不代表经过本仓库复测后的冠军。

## Round 1：从报警设计理论到过程/报警联合数据

**起点数据**：经典 Tennessee Eastman Process（TEP）仿真与 PRONTO multiphase flow。TEP 提供带故障的连续过程变量；PRONTO 明确同时包含异构过程测量和报警记录。

**文献链**：Downs & Vogel (1993) TEP → 专著第 2–5 章 → TEP Alarm Management Dataset（DataPort DOI `10.21227/326k-qr90`）。

**提取的方法**：IID/non-IID 阈值、delay timer、deadband；FAR/MAR/AAD 优化；凸/非凸 normal operating zone；NTE/NDTE 根因图；洪泛检测、局部序列对齐、频繁闭模式和 next-alarm 最大熵思路。

**算法落地**：`ThresholdDelayDeadband`、`design_alarm`、`MahalanobisAlarm`、`ConvexHullNOZAlarm`、`TransferEntropyRanker`、`smith_waterman_similarity`、`EmpiricalNextAlarmPredictor`。这些是透明经典基线，不宣称逐行复现书中全部加速技巧。

**扩展终点**：TEP alarm 专用数据具备 100 个仿真 run、300 个异常实例/29 个变体、170 个阈值及报警激活记录的论文描述，但归档文件目前需 DataPort 登录，故只登记元数据；PRONTO 完整压缩包 1.72 GB 默认跳过，可显式下载。

## Round 2：从洪泛序列到早期、开集与不确定性

**起点数据**：TEP alarm 元信息和 PIADE。PIADE 来自五台包装机，提供原始 interval/alarm 表及小时级序列特征；本仓库已下载并提供 `load_piade_alarm_events`。

**文献链**：CASIM/DCE 2024（DOI `10.1017/dce.2024.22`）→ Code Ocean artifact `10.24433/CO.4874993.v1` → ConE-AFC/IEEE Access 2024（DOI `10.1109/ACCESS.2024.3492348`，artifact `10.24433/CO.5512337.v2`）。

**提取的方法**：CASIM 以多种卷积 kernel 特征（MultiRocket）、ridge ensemble 与局部离群概率处理完整/早期序列和未知类；ConE-AFC 在前缀分类外加入 conformal prediction sets，用覆盖率与集合大小表达不确定性。

**算法/协议落地**：当前 A 级包含序列对齐和 coverage/set-size 指标；MultiRocket/ridge/LoOP 与 conformal calibration 为 B 级复现位，必须从官方 artifact 锁定环境后才能比较。

**扩展终点**：Nuclear Power Plant Alarm Dataset（DOI `10.21227/g2fa-9y43`）和 FCC alarm dataset（DOI `10.60517/2v23vv393`）进入第三轮跨装置/跨域验证候选；受许可页约束时只保存 landing metadata。

## Round 3：从跨域数据到现实扰动鲁棒性

**起点数据**：NPP/FCC 元数据、PIADE 真实设备日志、SKAB 35 个带异常标注的水循环实验。

**文献链**：2025 IFAC “predicting uncertainty reduction” 路线（random forest + jackknife+，DOI `10.1016/j.ifacol.2025.11.935`）→ 2026 AFC-RobustBench 预印本（DOI `10.2139/ssrn.6999280`）。

**提取的方法**：分类不只给点标签，还评估随报警前缀增长的不确定性下降；鲁棒性分别注入 missing、spurious、timing 和 detector-delay 扰动，比较 clean/perturbed 性能与最坏组。

**算法落地**：`perturb_alarm_episode` 与 `robustness_degradation` 为 A 级；jackknife+/bifurcation 预测和 RobustBench 完整模型为 B/C 级。2026 工作是预印本 SOTA candidate，需等数据/代码可得性审计，不在 README 宣称领先结果。

**本轮终点/下一循环**：形成 `TEP/PRONTO → PIADE → SKAB/NPP/FCC` 的跨数据矩阵。下一轮应优先取得 DataPort 授权归档、接入 CASIM/ConE-AFC 官方 artifact，并新增跨工厂 leave-one-site-out 与标签漂移协议。
