# IIA paper harness

本目录把书籍算法、SOTA 方法、数据集和论文实验条目组织为可审计的实验状态机。`E1/E2 transfer_result` 回答跨域迁移问题，不能替代 `paper_exact_result`；只有同数据、预处理、切分、超参数、指标和目标表格全部闭合，才允许标记 P3。

## 核心文件

- `protocol_freeze.v1.json`：评测内核、匹配度、证据等级、门禁、种子和资源预算。
- `experiment_matrix.v1.json`：算法到数据集与任务的目标映射。
- `reference_experiments.v1.json`：书籍章节和论文实验复现 backlog。
- `paper_exact/*.v1.json`：逐论文协议卡、目标表格、容差、缺失信息和双轨结果路径。
- `gate_ledger.seed.jsonl`：已执行波次与负结果账本。
- `scripts/paper_harness.py`：全局矩阵只读检查。
- `scripts/paper_exact.py`：P0 Capsule 校验、作者代码运行与结果摘要。

## 使用

```powershell
python scripts/paper_harness.py check
python scripts/paper_harness.py status
python scripts/paper_harness.py plan

python scripts/paper_exact.py check --require-local --full-hash
python scripts/paper_exact.py status
python scripts/paper_exact.py run-author --paper-id faulwasser2024_casim
python scripts/paper_exact.py summarize --paper-id faulwasser2024_casim

python -m pytest -q tests/test_paper_harness.py tests/test_paper_exact.py
```

## 证据边界

- `adapter_status=runnable` 只表示存在读取路径，不保证模型已接入统一 runner。
- `E2` 表示真实数据工程验证，不等于论文分数复现。
- `M1/P0` 只用于失配诊断，不参与 M2/M3 排名。
- `P2_author_capsule_default` 表示作者工件的默认子协议成功运行；若默认脚本少于论文实验网格，不能升级 P3。
- 每次正式波次必须使用新输出目录，记录 Git revision、协议哈希、数据哈希和环境信息，并单独提交。
