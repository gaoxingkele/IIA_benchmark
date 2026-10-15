"""Archive source counter evidence and the unexecuted exact-epoch adapter."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from scripts.flow_matching.audit_moment_icm_epoch_contract_v1 import CONFIG, ROOT, read, sha, verify, write

config = read(ROOT / CONFIG)
verify(config)
report_path = Path(config['audit_output'])
report = read(report_path)
assert report['passed'] and report['configuration_sha256'] == sha(ROOT / CONFIG)
assert report['actual_model_updates'] == report['neural_pretraining_epochs_executed'] == 0
assert report['only_stopping_condition_changed'] and report['source_file_unchanged']
assert not report['integrated_full_scientific_pretraining_runner_available']
assert not report['actual_full_corpus_steps_or_tokens_certified']
xml = ROOT / 'tmp/MOMENT_ICM_epoch_contract_validation_20261015_v1.xml'
suites = list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 18
assert all(int(s.get(k, 0)) == 0 for s in suites for k in ['failures', 'errors', 'skipped'])
folder = ROOT / 'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_one_full_epoch_contract_v1'
folder.mkdir(parents=True, exist_ok=False)
copies = {
    'frozen_epoch_contract_configuration.json': ROOT / CONFIG,
    'actual_source_counter_and_adapter_audit.json': report_path,
    'epoch_contract_validation.xml': xml,
    'moment_icm_epoch_contract_v1.py': ROOT / 'scripts/flow_matching/moment_icm_epoch_contract_v1.py',
    'audit_moment_icm_epoch_contract_v1.py': ROOT / 'scripts/flow_matching/audit_moment_icm_epoch_contract_v1.py',
    'test_fm_moment_icm_epoch_contract_v1.py': ROOT / 'tests/test_fm_moment_icm_epoch_contract_v1.py',
}
for name, p in copies.items():
    shutil.copyfile(p, folder / name)
    assert sha(p) == sha(folder / name)
originals = {}
for pin in config['source_receipts']:
    if '/moment_research/original/' not in pin['path']:
        continue
    value = (ROOT / pin['path']).read_bytes().decode('utf-8')
    assert hashlib.sha256(value.encode('utf-8')).hexdigest() == pin['sha256']
    originals[pin['path']] = dict(sha256=pin['sha256'], original_utf8=value)
write(folder / 'related_original_training_source_snapshot.json', dict(
    related_source_commit=config['related_source_commit'], files=originals,
    original_bytes_recoverable_by_UTF8_encoding=True))
(folder / 'README.md').write_text('''# MOMENT/ICM 一整 epoch 的停止条件对齐

实际读取并按 SHA256 固定的相关 MOMENT 代码使用：

```python
while opt_steps < self.args.max_opt_steps or cur_epoch < self.args.max_epoch:
    # 完整遍历 train_dataloader
    # 每批更新一次 opt_steps
    cur_epoch = cur_epoch + 1
```

默认配置和 pretraining 配置都写明 `max_opt_steps=5000000`、`max_epoch=50`。该循环在两个最低条件均达到后才退出，且没有批内的提前结束判断。它可能是来源训练方案的有意设置，不能仅因它与另一论文的预算不同就称原算法错误。但只把 max_epoch 改为 1 并不能落实 ICM 论文的一整 epoch。

已实现可调用的隔离适配器：对固定原 `Pretraining.train` 做私有 AST 副本，只把 while 判断替换为 `cur_epoch < self.args.max_epoch`。绑定时要求任务的 max_epoch 为 1，以保持调度器参数一致；要求完整、非空训练加载器保留最后不足一批的数据。绑定前校验任务的原 train 函数与固定源码 AST 相同。绑定返回独立方法，不替换任务原方法，也不修改来源文件或现有进程。

对恢复原条件后的整段 AST 做完全相等检查。原训练循环体（前向、掩码损失、AMP、优化器、梯度裁剪、日志、调度器及原有回调）保持不变。适配器不表示这些回调已纳入完整资源控制；完整训练入口还必须负责模型/环境/数据、资源保护、原回调行为及最终检查点的保存与复核。

18 项检查通过：从实际源码抽取计数初始化、循环条件和增量执行计数投影，验证 OR 的两个最低预算、整 epoch 的边界越过、隔离后在不同步数设置下准确完成一个 epoch、原训练体不变、绑定隔离、最后部分批保留及拒绝错误来源哈希/预算。它们没有运行神经网络。假设每 epoch 有 2,048 批，仅修改 max_epoch=1 时原条件会执行 2,442 个 epoch、5,001,216 次批更新；隔离后的目标为 1 个 epoch、2,048 次。这是控制流例子，2,048 不是当前真实语料测得的批数。

本轮完成来源控制流审计和可调用的 epoch 适配器；实际神经网络更新、预训练 epoch、新增性能结果均为 0。完整 Tiny/ICM 预训练入口尚未整合。全来源多通道数据加载、采样/分组、精确步数与 token、模型环境、最终权重落盘、原生独立复核、全部预测、117 数据集 SVM、beta 和严格预处理对照仍需完成。当前 33/47 与单独时间长度对照 28/47 的 Monash 输入范围也必须分开报告，不能把不同语料预算当作模型结构消融。

论文：[2409.13530v1](https://arxiv.org/abs/2409.13530v1)。来源 MOMENT 提交 `3ab637e413f35f2c317573c0ace280d825c558de` 与 ICM 历史流水线的等价性仍未认证。训练源码和两份来源配置原字节已经封装，按 SHA256 可恢复检查。已有 45 篇论文/681 条方法记录/273 条数据任务的最低范围及后续发现的实验均保留，全部研究实验尚未完成。
''', encoding='utf-8')
(folder / '.gitattributes').write_text('*.py -text\n*.json -text\n', encoding='utf-8')
shutil.copyfile(__file__, folder / 'publication_source.py')
write(folder / 'archive_validation.json', dict(
    captured_utc=datetime.now(timezone.utc).isoformat(), checks_passed=18,
    actual_neural_updates=0, actual_pretraining_epochs=0, new_performance_results=0,
    only_source_loop_condition_replaced_in_private_AST=True,
    original_code_and_workers_modified=False, integrated_full_pretraining_complete=False,
    files_sha256={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
    all_original_experiments_complete=False))
ara = ROOT / 'projects/flow_matching_research/ara/execution/moment_ICM_one_full_epoch_contract.2026-10-15_v1.json'
write(ara, dict(
    paper_id='moment_long_context', configuration=CONFIG, evidence=folder.relative_to(ROOT).as_posix(),
    source_training_stop_policy_audited=True, private_paper_epoch_adapter_callable=True,
    actual_pretraining_epochs_executed=0, model_performance_results=0,
    integrated_full_pretraining_entry_available=False,
    previous_goal_turn_classification='progress_complete_2441_file_and47_profile_audit_published',
    current_goal_turn_classification='progress_resolved_source_paper_epoch_stopping_mismatch',
    full_original_scope_retained=True, all_original_experiments_complete=False))
targets = [CONFIG, folder.relative_to(ROOT).as_posix(), ara.relative_to(ROOT).as_posix(),
    'scripts/flow_matching/moment_icm_epoch_contract_v1.py',
    'scripts/flow_matching/audit_moment_icm_epoch_contract_v1.py',
    'tests/test_fm_moment_icm_epoch_contract_v1.py']
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    paths.update(subprocess.check_output(['git', *args, '--', *targets], cwd=ROOT, text=True).splitlines())
git = lambda cwd, *args: subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
assert paths and not git(ROOT, 'diff', '--cached', '--name-only')
write(ROOT / 'tmp/MOMENT_ICM_epoch_contract_publication_manifest_20261015_v1.json', dict(
    previous_root=git(ROOT, 'rev-parse', 'HEAD'),
    previous_parent=git(ROOT.parent / 'flow-matching-research', 'rev-parse', 'HEAD'), paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths), checks_passed=18, neural_updates=0,
                     private_epoch_adapter_prepared=True, full_pretraining_complete=False)))
