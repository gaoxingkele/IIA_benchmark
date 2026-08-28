# Dataset registry

| ID | 状态 | 内容/适用任务 | 获取与引用 |
|---|---|---|---|
| `tep_classic` | 已下载 Git，revision 见 audit | 过程变量、故障；T1/T2/T3 | [GitHub](https://github.com/jkitchin/tennessee-eastman-profbraatz), Downs & Vogel DOI `10.1016/0098-1354(93)80018-I` |
| `pronto_*` | 1.72 GB payload 已下载，MD5 与全量 ZIP CRC 已通过；官方 aligned/labelled 子集已安全提取 | 过程变量+报警；T1–T5（T4 为故障窗代理，非专家洪泛标签） | [Zenodo 1341583](https://zenodo.org/records/1341583), CC BY 4.0 |
| `piade_*` | 两个 CSV 已下载且 MD5 通过 | 五台包装机 interval/alarm 与小时序列；T4/T5/T6 | [Zenodo 7071747](https://zenodo.org/records/7071747) |
| `skab` | 已下载 Git | 35 个水循环异常实验；T1/T2/鲁棒性 | [官方仓库](https://github.com/waico/SKAB), DOI `10.1007/s41060-022-00355-4` |
| `tep_alarm_dataport` | 16,983,510,811 字节 RAR 已经认证传输取得；SHA-256、外层 RAR5 CRC、两项小型内层 ZIP 全量 CRC 与两项大型 ZIP 目录审计通过；adapter/实验未闭环 | 100 Tests run × Original/Filter/Deadband、1,000 条五类报警序列、18 个异常场景变体；T3/T4/open-set/序列鲁棒性 | [IEEE DataPort](https://ieee-dataport.org/open-access/tennessee-eastman-process-alarm-management-dataset), DOI `10.21227/326k-qr90` |
| `npp_alarm_dataport` | 199,576,419 字节 RAR 已取得，SHA-256 与 UnRAR 全量 CRC 通过；adapter 未实现 | 101 个顶层 run/组、12 类事故/扰动 + Normal、122,510 个 CSV（含 101 个阈值文件）；T3/T4/跨域 | DOI `10.21227/g2fa-9y43` |
| `fcc_alarm` | 真正 FCC payload 仍缺；同名候选 ZIP 已证实为 CO2 吸附数据并隔离 | FCC 报警跨域 T3/T4 仍为缺口；误命名包仅用于数据质量审计 | FCC DOI `10.60517/2v23vv393`；误命名包实际 DOI `10.60517/19027803-fec9-41f2-8a02-408cc176554e` |
| `comopi` | 报警 CSV 已下载，42,677,268 字节，MD5 通过 | 8 台包装设备、123 类十分钟报警计数；T5/T6、bad actor；无 bin 内次序/洪泛标签 | [Zenodo 7572501](https://zenodo.org/records/7572501)，DOI `10.5281/zenodo.7572501` |
| `smd10towfgr` | 180,707,378 字节 XLSX 已下载，MD5 通过 | 10 台风机的 SCADA 与 230,618 条事件/报警日志；T4 序列/密度、T5/T6；缺专家洪泛类别 | [Zenodo 14546480](https://zenodo.org/records/14546480)，CC BY 4.0 |
| `enas` | 20,010,388 字节 CSV 已下载，MD5 通过 | 219,893 条数字传感器/执行器状态变化和人工错误状态；T3/T5 | [Zenodo 4742256](https://zenodo.org/records/4742256)，CC BY 4.0 |
| `imaks` | 19,797,994 字节 ZIP 已下载，MD5 通过；synthetic | MQTT/传感器报警和因果真值；T3/鲁棒性 smoke，不得进入真实工业榜单 | [Zenodo 20075430](https://zenodo.org/records/20075430) |

本次本地 profile：PIADE raw 429,394 行、5 台设备、92,084 个非 `A_000` 报警区间；CoMoPI 150,650 个十分钟 bin、194,974 次报警；SMD10TOWFGR 230,618 条日志、167 个事件 code；EnAS 219,893 行；iMAKS annotated sensor 表 211,200 行；SKAB 35 个 CSV 实验；经典 TEP 44 个 run 文件（2 normal、42 fault）、52 个变量。数值来自 `profile_public_datasets.py`，可在数据更新后重算，不作为上游数据集的永久版本声明。

`configs/datasets/public_sources.json` 是机器可读登记；`data/public_datasets/audit.json` 是本机实际状态。下载器默认只抓取体量适中且无需交互授权的资源。显式下载 PRONTO 完整包：

```powershell
python scripts/data_acquisition/download_public_datasets.py --dataset pronto_full

# 显式下载 180.7 MB 的 SMD10TOWFGR 工作簿
python scripts/data_acquisition/download_public_datasets.py --dataset smd10towfgr
```

DataPort 的二进制归档不能通过绕过登录/条款的方式抓取。当前 TEP/NPP 原始载荷已通过认证传输取得并补齐 checksum，但在 adapter、grouped split 和论文参考分数跑通前，算法成熟度仍不得提升为 A。真正 FCC alarm payload 仍需从稳定且许可明确的来源取得。
