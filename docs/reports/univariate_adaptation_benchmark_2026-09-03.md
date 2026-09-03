# 单变量跨数据集适配基准报告（2026-09-03）

## 结论

本轮在不修改书籍原算法的前提下，完成了独立适配层、校准期适用性门禁、三数据集三种子消融、Page-Hinkley 变点事件检测、物理时间 delay、事件块 Beta 后验、moving-block bootstrap 和新数据集配置化接入。

没有一个适配器同时满足三个数据集的低 FAR/低 MAR 要求。当前可确认的是：

- TEP 适合静态或双侧 ECDF；双侧 ECDF 的平均 F1 为 0.9502。
- SKAB 需要 recent-window 与 block delay；B4 将 FAR 从 0.5949 降到 0.1446、MAR 从 0.0125 降到 0.0042，但仍有约 55.1 次虚警事件/小时，尚不可称为工业部署方案。
- PRONTO 的 B5 将 MAR 从 0.7905 降到 0.2824，但 FAR 上升到 0.4313，未通过 FAR≤0.15 门禁。未来异常阶段漂移无法由校准期完全预知。
- TEP d11 在三个种子上均被自动拒绝单变量评分，避免用测试集调参制造高分。
- 安全滚动 ECDF（B6）在大幅基线漂移后冻结，SKAB FAR 退化到 0.8441，因此不晋级。

这些结果属于 M2/P1 工程迁移证据，不是书籍私有工业数据或论文原表的 P3 精确复现。

## 方法结构

书籍 Chapter 2 的四个实现保持在 `src/iia_benchmark/models/univariate_book.py`。新增模块只作为前置归一化、选择、校准和拒绝包装器：

```text
normal_train + abnormal_calibration
  → 分布/时序审计
  → 稳定变量与方向
  → static / block-recent / deny 路由
  → 书籍报警状态机
  → frozen evaluation
  → point + event + event-posterior + block-bootstrap evidence
```

主要实现：

- `src/iia_benchmark/evaluation/distribution_audit.py`
- `src/iia_benchmark/evaluation/event_metrics.py`
- `src/iia_benchmark/models/adaptive_univariate.py`
- `src/iia_benchmark/data/univariate_partition.py`
- `src/iia_benchmark/adaptation/univariate_pipeline.py`

`PageHinkleyChangeAlarm` 提供稳健尺度上的双侧变化事件，`TimeBasedEmpiricalCDFAlarm` 将秒/分钟持续时间按采样周期向上取整为 delay。二者已具备配置引用和单元测试，但尚未加入 B0–B7 排行，因此不能把测试通过误称为跨数据集性能提升。

## 数据与机制

| 数据集 | 选中变量 | 正常 KS 中位数 | 异常阶段 KS 中位数 | 测试 AUC 中位数 | 主要问题 |
|---|---|---:|---:|---:|---|
| TEP | XMEAS_01、XMV_10 | 0.0547 | 0.0990 | 1.0000（最低 0.5127） | d11 单变量不可分 |
| PRONTO | Water.Density、Water.T、Water.level 等 | 0.3649 | 0.8876 | 0.5343 | 异常阶段漂移、强自相关 |
| SKAB | Accelerometer2RMS | 0.4777 | 0.1440 | 1.0000 | 正常基线漂移、类别不平衡 |

三种子重复相同物理 episode，只改变 bootstrap 变量选择；不能把 9 个记录当作 9 个独立工业事件做传统 IID 显著性检验。

## B0–B7 消融结果

| 变体 | 定义 |
|---|---|
| B0 | 冻结的书籍 IID 基线 |
| B1 | 单侧 ECDF |
| B2 | 双侧 ECDF |
| B3 | 稳定变量选择 + ECDF |
| B4 | 单侧 block + recent-window ECDF |
| B5 | 双侧 block + recent-window ECDF |
| B6 | 带污染保护的 rolling ECDF |
| B7 | 自动路由与显式拒绝 |

### F1

| 数据集 | B0 | B1 | B2 | B3 | B4 | B5 | B6 | B7* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TEP | 0.8678 | 0.8246 | **0.9502** | 0.6612 | 0.8163 | 0.8972 | 0.7681 | 0.9501 |
| PRONTO | 0.3011 | 0.1243 | 0.2413 | 0.1504 | 0.5012 | **0.6738** | 0.6222 | 0.6552 |
| SKAB | 0.1448 | 0.2415 | 0.3301 | 0.2415 | **0.4124** | 0.2618 | 0.1066 | **0.4124** |

`B7*` 为 selective performance。TEP 的 3 个 d11 episode-seed 单元被拒绝，不能将其与全覆盖方法直接比较。

### FAR / MAR

| 数据集/候选 | FAR | MAR | 判定 |
|---|---:|---:|---|
| TEP B2 | 0.0271 | 0.0458 | 通过性能保持门禁 |
| PRONTO B5 | 0.4313 | 0.2824 | MAR 改善，但 FAR 门禁失败 |
| PRONTO B7 | 0.2935 | 0.3653 | 仍超过 FAR≤0.15 |
| SKAB B4/B7 | 0.1446 | 0.0042 | 相对基线显著改善，绝对虚警仍高 |

## 书籍四算法在适配分数空间的验证

| 数据集 | IID F1 | non-IID F1 | deadband F1 | APP F1 |
|---|---:|---:|---:|---:|
| TEP | 0.9868 | 0.9255 | 0.9408 | 0.5541 |
| PRONTO | 0.4546 | 0.6863 | 0.2264 | 0.7235 |
| SKAB | 0.2214 | 0.2214 | 0.3785 | 0.1021 |

APP 在 PRONTO 的 F1 较高但 FAR 为 0.5689，不能只按 F1 晋级。SKAB APP FAR 为 0.8975。离散 ECDF 支持集不足、deadband 正常报警 episode 少于两个时，工程会保留明确的退化/拒绝状态，不伪造书籍机制激活。

## 不确定性与事件指标

每个 episode 使用正常分区、异常分区分别进行 moving-block bootstrap，默认 block size 60、200 次抽样；SKAB 单独接入示例使用 500 次。报告逐项保存 FAR、MAR、precision、recall、F1 的点估计、标准误和 95% 区间。

配置化接入还把每个非重叠时间块是否出现报警建模为 Bernoulli 事件，并保存 Beta(1,1) 后验。它在零事件时仍给出有限区间，取代“零事件就回退到常数”的隐式处理；这不是 IID 样本置信区间，也不替代 moving-block bootstrap。

SKAB valve1/0 端到端接入结果：

- F1 0.4134，95% block-bootstrap 区间 `[0.3283, 0.5322]`；
- FAR 0.1446，区间 `[0.0901, 0.2097]`；
- MAR 0.0041；
- 72 个正常段虚警事件，约 55.1 次/小时；
- 最长连续虚警 214 个采样点；
- 异常事件召回 1.0，检测延迟 1 秒。
- 60 秒正常块报警事件后验均值 0.4074，95% 区间 `[0.3035, 0.5156]`；异常检测块后验均值 0.8571，区间 `[0.5407, 0.9958]`。

因此“逐点 FAR 相对下降”与“现场报警负担可接受”是不同结论。

## 新数据集接入

配置驱动入口：

```powershell
python scripts/run_univariate_adapter.py configs/experiments/skab_univariate_adapter_onboarding.json
```

完整字段和门禁说明见 `docs/univariate_dataset_onboarding.md`。接入器会：

1. 只读加载四个 CSV 分区；
2. 校验过滤后行区间没有重叠；
3. leaderboard 配置检查 group ID 隔离；
4. 仅用训练/校准分区做路由；
5. 记录输入文件 SHA-256；
6. 输出适配或 `denied_univariate`；
7. 生成逐点、事件、事件块后验和 block-bootstrap 结果。

当前 SKAB 示例故意标为 `leaderboard_eligible: false`，因为正常文件和 fault run 是按时间再次切分，不能代表独立设备/事件组泛化。

## 复现实验

```powershell
python experiments/paper_harness/univariate_adaptation/experiment.py --out_dir experiments/paper_harness/univariate_adaptation/run_1
python experiments/paper_harness/univariate_adaptation/experiment.py --out_dir experiments/paper_harness/univariate_adaptation/run_2
python experiments/paper_harness/univariate_adaptation/experiment.py --out_dir experiments/paper_harness/univariate_adaptation/run_3
python experiments/paper_harness/univariate_adaptation/plot.py
python -m pytest -q
```

机器可读汇总：`experiments/reports/univariate_distribution_audit_validation.json`。图表位于 `experiments/paper_harness/univariate_adaptation/Figure_1.png` 至 `Figure_4.png`。

## 未闭合项

- PRONTO 尚无同时满足低 FAR 和低 MAR 的单变量方案；Page-Hinkley 组件虽已落地，但仍需真实工况标签完成跨 episode 阈值验证，失败时转状态条件模型或多变量回退。
- SKAB 尚未达到低事件虚警负担；需要独立正常工况组验证，而不是继续使用同一 anomaly-free 文件调参。
- TEP d11 需要动态残差或多变量方法。
- 当前公开数据是系统故障/异常标签，不是专家验证的逐变量报警真值。
- 路由阈值属于多数据集工程 beacon，不是普适统计常数。

## 仓库

项目地址：<https://github.com/gaoxingkele/IIA_benchmark>

本轮所有迭代均通过测试后单独提交并推送 `main`，没有覆盖原始书籍或数据文件。
