# 单变量新数据集接入说明

## 目标与边界

新数据集通过 JSON 配置接入，原始 CSV 始终只读。接入器生成分布审计、校准期适用性判断、自动选择的静态或 block/recency ECDF 报警器、逐点指标、事件指标、事件块 Beta 后验和 moving-block bootstrap 区间。

适用性路由只允许读取 `normal_train` 和 `abnormal_calibration`。`normal_evaluation` 与 `abnormal_evaluation` 仅在模型冻结后用于评价。如果校准期最差时间块 AUC 或方向一致性未通过，结果为 `denied_univariate`，不会强行给出排行榜分数。

当前通用入口支持预先注册的 CSV 单变量分区。TEP 空白分隔格式仍由原有 TEP loader 和 Chapter 2 harness 负责。

## 必需分区

| 分区 | 允许用途 |
|---|---|
| `normal_train` | ECDF、稳健统计、内部时间切分、阈值及 delay 校准 |
| `abnormal_calibration` | 变量方向、时间块可分性与适用性门禁 |
| `normal_evaluation` | 冻结后的 FAR、虚警事件和分布漂移诊断 |
| `abnormal_evaluation` | 冻结后的 MAR、事件召回、延迟和异常阶段漂移诊断 |

每个分区描述必须提供 `path`、`loader: csv`、`value_column`、`filters`、过滤后的 `row_start/row_stop` 和 `group_id`；非逗号 CSV 通过 `delimiter` 登记，例如 SKAB 使用分号。同一文件、相同过滤条件下的行区间不能重叠。`leaderboard_eligible: true` 时四个分区的 `group_id` 必须互异。

## 运行实例

仓库提供实际 SKAB 示例：

```powershell
python scripts/run_univariate_adapter.py configs/experiments/skab_univariate_adapter_onboarding.json
```

输出位置由配置中的 `output` 决定。示例输出为：

```text
experiments/reports/skab_univariate_onboarding_validation.json
```

新数据集接入步骤：

1. 复制 SKAB 示例配置并更换数据路径、列名、过滤条件和时间切分。
2. 填写数据集论文 DOI/URL、采样周期和真实 group ID。
3. 首次以 `leaderboard_eligible: false` 运行并检查分布审计。
4. 确认设备、批次或事件分组完全隔离后，再申请 leaderboard 资格。
5. 检查 `calibration_applicability`；被拒绝时转入多变量模型，不修改测试集或降低门禁。
6. 固定配置、数据 SHA-256、种子和代码 revision 后才发布结果。

## 结果解释

- `static_ecdf`：校准期未发现需要适配的明显漂移或时间依赖。
- `block_calibrated_ecdf`：内部正常分布或时间依赖触发了 recent-window 与 delay 校准。
- `denied_univariate`：单变量最差时间块可分性或方向稳定性不足，应转入 PCA/DPCA、Mahalanobis、动态残差或其他多变量方法。

F1 不能单独决定可部署性。至少同时检查 FAR/hour、虚警事件持续时间、事件召回、检测延迟、事件块 Beta 后验和 block-bootstrap 区间。`event_prior_alpha/event_prior_beta` 显式记录先验；默认均为 1.0，零事件时也不会隐式回退到伪造的点值。
