# FCC Alarm 官方数据获取与完整性审计（2026-08-29）

## 结论

真正的 Fluid Catalytic Cracking (FCC) Alarm Dataset 已从 Ruhr University
Bochum 的公开 ReSeeD 记录直接取得。来源 DOI 为 `10.60517/2v23vv393`，记录许可为
CC BY 4.0。官方记录包含 4 个附件，共约 18.9 MB；本地文件均位于
`data/public_datasets/fcc_alarm/`，由 `.gitignore` 排除，不把原始载荷提交到 Git。

这次获取闭合了 `fcc_alarm` 的原始数据缺口，但尚未闭合 adapter、grouped split、
算法运行和论文参考分数。此前阿里云盘内同名 ZIP 的内容仍被判定为 CO2 吸附数据，
继续以 `fcc_alarm_candidate_rejected` 保留在质量审计中，不得混入本数据集。

## 官方来源与附件

- 官方记录：<https://reseed.ruhr-uni-bochum.de/concern/datasets/19217e41-e9ad-4c0e-bb51-97e7072813f7>
- DOI：<https://doi.org/10.60517/2v23vv393>
- 创建者：Gianluca Manca、Franz Christopher Kunze
- 许可：CC BY 4.0
- 官方记录发布日期：2025-05-12；当前附件上传/记录更新日期：2026-05-28

| 文件 | 字节数 | SHA-256 | 校验 |
|---|---:|---|---|
| `alarmseriesdata.zip` | 828,104 | `c62157d28e643dabd2a72d695e0df84aa39ad40b68b140d13db07a2243e2d28d` | ZIP 全量 CRC 通过 |
| `timeseriesdata.zip` | 18,732,740 | `8a67269306c4c52898400d977473689b8a0b9133145d395d329bf44f998f6669` | ZIP 全量 CRC 通过 |
| `FCC_Diagramm.pdf` | 151,306 | `bdbfb50e92e23cf10052addf58f924c7bf6922bcbcd4d43113a311a15cc3901d` | 文件完成并登记哈希 |
| `FCC_Dataset_Appendix_Alarm_Thresholds_and_Disturbance_Parameters.pdf` | 94,501 | `b224aadc359e97603911ddd291e9a417a4d621ddf9a17d6e10f062ce541df544` | 文件完成并登记哈希 |

官方对象存储在连续传输时出现低速断流。过程时序 ZIP 使用 HTTP Range 分块恢复，
完成后严格核对服务器声明总字节数，并以 ZIP 全量 CRC 和最终 SHA-256 作为有效性判据。

## 结构审计

报警载荷包含 1,600 个 CSV，按 16 类异常目录组织，每类 100 个独立仿真 run：

- `catalyst_deactivation`
- `cyclone_damage`
- `preheater_shutdown`
- `preheater_temp_increase`
- `V2/V3/V4/V6/V7/V8` 各自的 `high` 与 `low`

每个报警 CSV 有 60 行采样、57 个二值报警变量。配套过程载荷含 4,800 个 CSV，
即每个 run 对应一个 `process`、`valves` 和 `disturbances` 文件：

| 文件类型 | 文件数 | 每个文件采样行 | 列数（含 `Time`） |
|---|---:|---:|---:|
| alarm | 1,600 | 60 | 57 |
| process | 1,600 | 60 | 25 |
| valves | 1,600 | 60 | 11 |
| disturbances | 1,600 | 60 | 15 |

目录名可作为异常类别真值；`disturbances` 列提供注入扰动和环境条件，过程与阀位表
提供根因/动态响应证据。所有 run 均为高保真仿真，不能表述为真实工厂采集数据。

## 适用任务与评测边界

- T4 洪泛分类：直接使用 57 位报警状态序列；按 run 分组，禁止把同一 run 的窗口拆到训练和测试。
- open-set：完整留出一种或多种异常族，而不是随机留出窗口。
- T3 根因排序：联合 alarm、disturbance、process 和 valve 表，依据注入异常族构造根因真值。
- 鲁棒性：对报警位实施 missing/spurious/timing-jitter/delay 扰动，并报告相对 clean 退化。
- 跨域：与 TEP/NPP 使用统一 `AlarmEvent`/`AlarmEpisode` 合同，但必须单列“FCC simulated”域。

在 adapter、冻结 split、多 seed 和参考方法同协议运行完成前，本次结果只表示“官方原始载荷已取得并通过结构校验”，不表示形成了 leaderboard 成绩。

## 下一步

1. 实现只读 ZIP adapter，按 `scenario/run` 生成稳定样本 ID，并校验四类 CSV 对齐。
2. 建立 run-grouped closed-set、scenario-held-out open-set 和跨数据集 split。
3. 首批运行 CTFH、HDAM、CASIM、ConE-AFC 与 Cross-Conformal，验证 PRONTO 上的退化是否由表示/标签错配导致。
4. 将两份 PDF 中的报警阈值与扰动参数映射到数据字典和根因真值，而不从文件名臆测语义。
