# Chapter 5 — Alarm floods

## 数学与方法

洪泛检测可按窗口原始事件数 `A`、active tags `B`，或书中强调的新出现 tag 数 `C`。`C` 结合短期状态继承、排除长期驻留 alarm，比原始 event count 更抗 chattering。

相似度路线包括 Smith–Waterman 局部对齐，以及带 alarm priority、pre-match、seed/extend/backtrack 与时间歧义容差的 BLAST-like 加速对齐。模式挖掘使用 CHARM 找 closed frequent alarm patterns，再按代表模式做 `δ` 聚类。预测部分用当前全部 alarms、历史 characteristic functions 和时间距离 penalty 建模下一报警的 maximum-entropy distribution。

## 算法抽取

- `detect_alarm_floods`：滚动窗内 unique new activation tags。
- `smith_waterman_similarity`：归一化局部对齐经典基线。
- `EmpiricalNextAlarmPredictor`：全上下文、距离衰减的透明近似。
- `perturb_alarm_episode`：missing/spurious/timing 扰动。

加速 BLAST-like、CHARM 和严格最大熵优化是 B 级；不能把经验 predictor 标成作者算法的逐式复现。

## 数据与测试

TEP alarm/NPP/FCC 适合有根因的洪泛分类，PIADE 适合真实设备 sequence/bad-actor/forecast，PRONTO 可做过程与报警联合分析。分 episode/run 切分，另报 prefix、open-set 和 robustness。

证据：`05_alarm_floods.txt`，PDF physical pages 312–388。
