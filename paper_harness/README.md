# IIA paper harness

本目录把书籍算法、SOTA 方法、数据集和论文实验条目组织为一个可审计的实验状态机。
当前版本是规划冻结草案：矩阵已经闭合，但只有完成 Wave 0 的数据先验、适配器和 split
审计后，才能把 `protocol_freeze.v1.json` 的状态改成正式冻结。

## 文件

- `protocol_freeze.v1.json`：冻结评测内核 K、可变实验面 X、证据等级、门禁、种子和资源预算。
- `experiment_matrix.v1.json`：30 个算法到 11 个数据集族的目标映射；展开后为 121 个目标。
- `reference_experiments.v1.json`：5 个书籍章节组和 28 篇论文的实验条目复现 backlog。
- `gate_ledger.seed.jsonl`：已有 PRONTO 负结果的种子账本，确保退化证据不会被覆盖。
- `semantic_gene_bank.md`：从失败中提取的 WHERE × WHY 可复用规则。
- `scripts/paper_harness.py`：只读检查、状态和执行顺序输出。

## 使用

在仓库根目录运行：

```powershell
python scripts/paper_harness.py check
python scripts/paper_harness.py status
python scripts/paper_harness.py plan
python -m pytest -q tests/test_paper_harness.py
```

`check` 验证 30 个算法、11 个数据集族和 28 篇论文是否全部闭合，并拒绝未知 ID、
重复映射、少于三个有效数据集的算法、M1 非哨兵结果以及 lane/task 不一致。
这些命令不启动正式训练，也不接触冻结测试集。

## 当前边界

- `adapter_status=runnable` 只说明仓库已有数据读取路径，不保证对应算法已经接入统一 runner。
- `current_evidence=E2` 表示存在至少一个真实数据工程验证，不等于论文分数复现。
- `M1/P0` 只作为失配诊断；不能与 M2/M3 汇总排名。
- `P2` 表示同一数据族的重建协议，只有完整对齐数据、split、预处理、超参数、指标和条目时才是 `P3`。
- 每次正式波次必须使用新输出目录、记录 Git revision 和协议哈希，并单独提交。
