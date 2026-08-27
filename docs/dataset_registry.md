# Dataset registry

| ID | 状态 | 内容/适用任务 | 获取与引用 |
|---|---|---|---|
| `tep_classic` | 已下载 Git，revision 见 audit | 过程变量、故障；T1/T2/T3 | [GitHub](https://github.com/jkitchin/tennessee-eastman-profbraatz), Downs & Vogel DOI `10.1016/0098-1354(93)80018-I` |
| `pronto_*` | README/技术报告已下载；1.72 GB payload 可选 | 过程变量+报警；T1–T5 | [Zenodo 1341583](https://zenodo.org/records/1341583), CC BY 4.0 |
| `piade_*` | 两个 CSV 已下载且 MD5 通过 | 五台包装机 interval/alarm 与小时序列；T4/T5/T6 | [Zenodo 7071747](https://zenodo.org/records/7071747) |
| `skab` | 已下载 Git | 35 个水循环异常实验；T1/T2/鲁棒性 | [官方仓库](https://github.com/waico/SKAB), DOI `10.1007/s41060-022-00355-4` |
| `tep_alarm_dataport` | landing page 已下载；payload 受限 | 100-run 工业报警洪泛；T3/T4/open-set | [IEEE DataPort](https://ieee-dataport.org/open-access/tennessee-eastman-process-alarm-management-dataset), DOI `10.21227/326k-qr90` |
| `npp_alarm_dataport` | landing page 已下载；payload 可能需登录 | 核电报警；T3/T4 | DOI `10.21227/g2fa-9y43` |
| `fcc_alarm` | 已登记；站点重定向不稳定 | FCC 报警；跨域 T3/T4 | DOI `10.60517/2v23vv393` |

本次本地 profile：PIADE raw 429,394 行、5 台设备、92,084 个非 `A_000` 报警区间；PIADE 小时表 23,376 行/164 列；SKAB 35 个 CSV 实验；经典 TEP 44 个 run 文件（2 normal、42 fault）、52 个变量。数值来自 `profile_public_datasets.py`，可在数据更新后重算，不作为上游数据集的永久版本声明。

`configs/datasets/public_sources.json` 是机器可读登记；`data/public_datasets/audit.json` 是本机实际状态。下载器默认只抓取体量适中且无需交互授权的资源。显式下载 PRONTO 完整包：

```powershell
python scripts/data_acquisition/download_public_datasets.py --dataset pronto_full
```

DataPort 的二进制归档不能通过绕过登录/条款的方式抓取。取得授权后，将文件放入登记目录、补充 checksum 与 adapter，再把成熟度从 C/B 提升为 A。
