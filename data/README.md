# Data policy

`public_datasets/` 是本机缓存，不提交 Git；机器可读来源与 checksum 位于 `configs/datasets/public_sources.json`。运行 `audit_public_datasets.py` 会生成轻量 `audit.json`，用于记录 presence、bytes 和 Git revision。

任何新数据集先登记 URL/DOI/license/checksum/task，再下载，再实现 adapter 和 data card。没有根因 ground truth 的日志禁止用于 root-cause accuracy；没有 clearance event 的 activation-only 日志禁止直接计算 standing-alarm duration。
