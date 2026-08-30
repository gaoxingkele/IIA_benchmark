# P0 Paper-Exact 暂停检查点

暂停时间：2026-08-30 14:32:59（Asia/Shanghai）  
基线提交：`81d8ba6bdafd1e2934b7f1ce10327f320f12e889`

## 已安全停止

CASIM、ConE-AFC 与 BiP-AFC 的实验进程均已停止；停止后进程审计为 0。正式结果与迁移基线没有被改写，部分结果不会被标记为 P3 或完整论文复现。

| 实验 | 有效断点 | 恢复语义 |
|---|---:|---|
| CASIM 开放集 | 48/70 tasks | 直接读取 `seed_results.jsonl`，只计算剩余 22 个任务 |
| ConE 完整网格 | 0/50 splits | 6 个在途 split 未完成、没有可用输出；下次从 split 0 开始 |
| ConE 冒烟 | 1 个 WDI-1NN split | 仅证明执行路径可用，文件已改名，不能冒充完整网格 |
| BiP 作者默认 | 2/8 dataset-model groups 的 CSV | TEP/MBW-LR 与 TEP/EAC-1NN 已归档；缺 stdout 计数，不能生成 Tables 1–4 |

CASIM 断点 SHA-256 为 `2dfd73afa41176296f06ee9b13778e3be03e655ad3ba7b93348b57541900f827`。BiP 的 16 个部分 CSV 共 34,570 bytes，目录清单哈希为 `24ccd43ede38497e75b5dbb4350455e8d5e3f9496bdcab58d05af611cbbb7fdb`。机器重启或临时目录清理后，仓库中的 checkpoint 仍可审计。

## 下次恢复

全部按顺序恢复：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\resume_p0_checkpoint.ps1 -Target all
```

也可以分别使用 `-Target casim`、`-Target cone` 或 `-Target bip`。脚本会先进行 Capsule 全哈希检查和 Paper-Exact 测试。

注意：BiP 官方 `main.py` 没有 dataset/model 级 resume 参数，作者原样复跑只能从头开始。现有部分文件保存在 `run_3/checkpoint_results`，正式完整结果仍应写入 `run_3/author_results`。后续可在不修改作者模型源码的前提下增加逐模型包装器，以获得真正的细粒度恢复能力。
