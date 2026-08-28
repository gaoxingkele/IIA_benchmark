# Data policy

`public_datasets/` 是本机缓存，不提交 Git；机器可读来源与 checksum 位于 `configs/datasets/public_sources.json`。运行 `audit_public_datasets.py` 会生成轻量 `audit.json`，用于记录 presence、bytes 和 Git revision。

任何新数据集先登记 URL/DOI/license/checksum/task，再下载，再实现 adapter 和 data card。没有根因 ground truth 的日志禁止用于 root-cause accuracy；没有 clearance event 的 activation-only 日志禁止直接计算 standing-alarm duration。

认证下载的原始归档保存在 `public_datasets/alipan_anomaly_archives/`，同样不提交 Git。其 SHA-256、字节数、内部清单与适用边界登记在 `configs/datasets/public_sources.json` 和 `docs/reports/alipan_alarm_archive_acquisition_2026-08-28.md`。大包只在 `tmp/` 中短暂展开做目录审计；原始归档不得截断、重打包或覆盖。
