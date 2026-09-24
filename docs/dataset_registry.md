# Dataset registry

| ID | 状态 | 内容/适用任务 | 获取与引用 |
|---|---|---|---|
| `tep_classic` | 已下载 Git，revision 见 audit | 过程变量、故障；T1/T2/T3 | [GitHub](https://github.com/jkitchin/tennessee-eastman-profbraatz), Downs & Vogel DOI `10.1016/0098-1354(93)80018-I` |
| `pronto_*` | 1.72 GB payload 已下载，MD5 与全量 ZIP CRC 已通过；官方 aligned/labelled 子集已安全提取 | 过程变量+报警；T1–T5（T4 为故障窗代理，非专家洪泛标签） | [Zenodo 1341583](https://zenodo.org/records/1341583), CC BY 4.0 |
| `piade_*` | 两个 CSV 已下载且 MD5 通过 | 五台包装机 interval/alarm 与小时序列；T4/T5/T6 | [Zenodo 7071747](https://zenodo.org/records/7071747) |
| `skab` | 已下载 Git | 35 个水循环异常实验；T1/T2/鲁棒性 | [官方仓库](https://github.com/waico/SKAB), DOI `10.1007/s41060-022-00355-4` |
| `tep_alarm_dataport` | 16,983,510,811 字节 RAR 已取得并审计；五类 ZIP 的只读 adapter、G0、seeded 60/20/20 split 与首批 6 个 SOTA 实验已完成；100-run/异常变体 adapter 待补 | 100 Tests run × Original/Filter/Deadband、1,000 条五类报警序列、18 个异常场景变体；T3/T4/open-set/序列鲁棒性 | [IEEE DataPort](https://ieee-dataport.org/open-access/tennessee-eastman-process-alarm-management-dataset), DOI `10.21227/326k-qr90` |
| `npp_alarm_dataport` | 199,576,419 字节 RAR 已取得，SHA-256 与 UnRAR 全量 CRC 通过；alpha=0.50 只读 adapter、G0、去重 grouped split 与首批 6 个实验已完成 | 101 个阈值层，每层 1,212 个 run CSV；12 类事故/扰动 + Normal，192 个二值报警位、10 秒采样；T3/T4/跨域/阈值鲁棒性 | DOI `10.21227/g2fa-9y43` |
| `fcc_alarm` | ReSeeD 官方 4 个附件已下载并校验；alarm/process adapter、G0、完整 run split 与首批 9 个实验已完成；PLR 适用性 G0 已拒绝时间贡献评分 | 1,600 个仿真 run、16 类异常、每类 100 run；每个 run 有 60 个一分钟采样、57 个报警位，并配套过程/阀位/扰动序列；扰动列在单 run 内恒定，不可用于 PLR 时滞恢复；T3/T4/open-set/鲁棒性 | [ReSeeD](https://reseed.ruhr-uni-bochum.de/concern/datasets/19217e41-e9ad-4c0e-bb51-97e7072813f7)，FCC DOI `10.60517/2v23vv393`，CC BY 4.0；误命名候选包实际 DOI `10.60517/19027803-fec9-41f2-8a02-408cc176554e` |
| `comopi` | 报警 CSV 已下载，42,677,268 字节，MD5 通过 | 8 台包装设备、123 类十分钟报警计数；T5/T6、bad actor；无 bin 内次序/洪泛标签 | [Zenodo 7572501](https://zenodo.org/records/7572501)，DOI `10.5281/zenodo.7572501` |
| `smd10towfgr` | 180,707,378 字节 XLSX 已下载，MD5 通过 | 10 台风机的 SCADA 与 230,618 条事件/报警日志；T4 序列/密度、T5/T6；缺专家洪泛类别 | [Zenodo 14546480](https://zenodo.org/records/14546480)，CC BY 4.0 |
| `enas` | 20,010,388 字节 CSV 已下载，MD5 通过；只读 adapter 与 G0 已完成 | 219,893 条数字传感器/执行器状态变化和 160 个 ME/HE/UE 人工错误标记；递归 BN 的原始脉冲与五行持久化配对验证已完成；T3/T5 | [Zenodo 4742256](https://zenodo.org/records/4742256)，CC BY 4.0 |
| `imaks` | 19,797,994 字节 ZIP 已下载，MD5 通过；sensor/KG adapter 已完成；synthetic | 211,200 行 MQTT/传感器注释与因果真值；IGDTE/BN/PLR 的 T3 诊断 smoke，负结果保留，不得进入真实工业榜单 | [Zenodo 20075430](https://zenodo.org/records/20075430) |
| `mtsad` | 15 个载荷 + 2 个 SWaT 基准切分已下载并逐文件 SHA-256 固定；adapter、四个协议矩阵与首批复现实验已完成 | SWaT/MSL/SMAP/PSM/SMD 多变量时间序列异常检测基准族：SMD 708405×38 / MSL 58317×55 / SMAP 135183×25 / PSM 132481×25 / SWaT 495000×51（训练）；异常率 4.16% / 10.53% / 12.79% / 27.76% / 12.14% | [Time-Series-Library 镜像](https://huggingface.co/datasets/thuml/Time-Series-Library)；原始来源见 `papers/literature/mtsad_registry.json` |

## MTSAD 族说明（多变量异常检测基准）

`mtsad` 是新增的多变量时间序列异常检测基准族，用于回答“论文报告分数是否可复现”。
其载荷取自 `thuml/Time-Series-Library` 的公开镜像（CC BY 4.0 声明），并由
`configs/datasets/public_sources.json` 第 4 轮条目逐文件绑定 SHA-256；SWaT 的规范发布源
（iTrust/SUTD）仍需签署申请，镜像替代这一事实记录在
`configs/datasets/mtsad_swat.json` 的 `access_note`。

复现协议矩阵定义在 `configs/experiments/mtsad_reproduction.json`：窗口到时间戳的展开方式、
阈值规则（分位数 / 最优 F1）与是否点调整（point adjustment）各自独立，因此同一检测器的
不同分数可以归因到协议而非模型。协议出处是 Time-Series-Library 的
`exp/exp_anomaly_detection.py` 与 `utils/tools.py:adjustment`。

已知冲突：TranAD 论文的数据集表记 SWaT 训练集为 496800 行，本项目固定的基准载荷为
495000 行，两者相差 1800 行；该差异不做静默裁剪，而是随复现结果一起报告（见
`ara_mtsad/evidence/tables/swat_split_audit.md`）。

### MTSAD 族已登记的额外数据集（论文实际用到的全部数据集）

`ara_mtsad` 登记的论文不止使用五个基准数据集：TranAD 用 9 个（NAB、UCR、MBA、
SMAP、MSL、SWaT、WADI、SMD、MSDS），Anomaly Transformer 用 6 个（五者 + NeurIPS-TS）。
本轮把这些数据集一并纳入，逐项状态如下（登记见 `configs/datasets/public_sources.json`
第 5 轮）：

| 数据集 | 来源 | 状态 |
|---|---|---|
| SMAP / MSL | `imperial-qore/TranAD`（Telemanom 原始 npy 格式，166 文件） | 已下载；与 TSLib 基准载荷互为对照 |
| SMD | 同上（OmniAnomaly 原始 per-machine 格式，114 文件） | 已下载 |
| NAB（realKnownCause 子集） | 同上；上游 `numenta/NAB`（MIT） | 已下载，7 条序列 + labels.json |
| UCR（自然源子集） | 同上；上游 SIGKDD 2021 多数据集竞赛 | 已下载，4 条序列（InternalBleeding 16–19） |
| MBA（MIT-BIH 心律） | 同上；上游 PhysioNet `mitdb` | 已下载（train/test/labels xlsx） |
| WADI | iTrust/SUTD，需申请 | 未下载；仅保留攻击标签与说明，登记为 metadata |
| MSDS | Zenodo 3549604，需向作者申请 | 未下载；仅保留记录页与标签，登记为 metadata |

仓库另有一份 TranAD 论文数据的完整清单：`data/public_datasets/mtsad_paper_repos/tranad/`
（git 克隆，commit 记录在 `data/public_datasets/audit.json` 的 revision 字段）。

本次本地 profile：FCC Alarm 1,600 个仿真 run、16 类异常、57 个报警位并配套 4,800 个过程/阀位/扰动 CSV；PIADE raw 429,394 行、5 台设备、92,084 个非 `A_000` 报警区间；CoMoPI 150,650 个十分钟 bin、194,974 次报警；SMD10TOWFGR 230,618 条日志、167 个事件 code；EnAS 219,893 行；iMAKS annotated sensor 表 211,200 行；SKAB 35 个 CSV 实验；经典 TEP 44 个 run 文件（2 normal、42 fault）、52 个变量。数值来自 `profile_public_datasets.py`，可在数据更新后重算，不作为上游数据集的永久版本声明。

`configs/datasets/public_sources.json` 是机器可读登记；`data/public_datasets/audit.json` 是本机实际状态。下载器默认只抓取体量适中且无需交互授权的资源。显式下载 PRONTO 完整包：

```powershell
python scripts/data_acquisition/download_public_datasets.py --dataset pronto_full

# 显式下载 180.7 MB 的 SMD10TOWFGR 工作簿
python scripts/data_acquisition/download_public_datasets.py --dataset smd10towfgr

# FCC 官方报警序列、过程序列和两份说明附件（均为公开下载）
python scripts/data_acquisition/download_public_datasets.py --dataset fcc_alarm_series --dataset fcc_alarm_timeseries --dataset fcc_alarm_diagram --dataset fcc_alarm_appendix
```

DataPort 的二进制归档不能通过绕过登录/条款的方式抓取。当前 TEP/NPP 原始载荷已通过认证传输取得并补齐 checksum；FCC Alarm 已从公开的 ReSeeD 官方记录直接取得。TEP/FCC 当前仅达到 E2/P1-P2 工程验证；在多 seed、论文 exact protocol 和参考分数跑通前，算法成熟度仍不得提升为 A。
