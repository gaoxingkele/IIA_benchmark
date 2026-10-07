# Simulation-free Schrödinger bridges via score and flow matching

## E01

**Verifies**: C01

**Setup**: 本地原文完整性和章节核验

**Procedure**: 读取登记 PDF、重新计算 SHA256、解析指定范围；保存来源清单。

**Expected outcome**: 与原始登记匹配，书籍仅按章节范围建立证据。

**Evidence**: `evidence/source/source_manifest.json`

## E02

**Verifies**: C02, C03

**Setup**: 文献机制与导航依据

**Procedure**: 核验配置中标记确在对应页；人工注释范围与全文自动索引区分；有数字的表格单独标记转录来源。

**Expected outcome**: 锚点可解析，不推断原文性能成立。

**Evidence**: `evidence/source/source_manifest.json`

## E03

**Verifies**: C03

**Setup**: 原作者协议复现，状态待执行/另行登记

**Procedure**: 锁定代码、数据划分和表格版本；先接口检查，再按原预算多种子运行；以预先冻结的等效区间比较。

**Expected outcome**: 取得实际运行日志、预测及配对差异；当前无本轮运行结果。

**Evidence**: `evidence/runs/local_validation.json`

## E04

**Verifies**: C04

**Setup**: 共同异常检测协议与工业迁移，状态 planned

**Procedure**: 依 comparison config 分原作者、TAB、严格验证、工业迁移四轨；报告 AUROC/AP、无PA点F1、实体宏平均、误报/延迟及配对不确定性。

**Expected outcome**: 检验性能差异来自评分协议、数据/训练预算还是模型机制；适配不合格者退出排行榜。

**Evidence**: `evidence/runs/local_validation.json`
