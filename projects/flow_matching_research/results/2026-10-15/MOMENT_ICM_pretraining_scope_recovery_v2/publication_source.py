"""Archive terminal full-corpus input audit, preserving the failed eager audit."""
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from scripts.flow_matching import audit_moment_icm_pretraining_corpus_v2 as audit
from scripts.flow_matching.register_moment_icm_pretraining_corpus_audit_v2 import CONFIG, EVIDENCE

ROOT, read, sha, write = audit.ROOT, audit.read, audit.sha, audit.write
config = read(ROOT / CONFIG)
audit.verify(config)
base = ROOT / config['state_root']
folder = ROOT / EVIDENCE
assert folder.is_dir() and not (folder / 'archive_validation.json').exists()
status, guard = read(base / 'status.json'), read(base / 'guard_receipt.json')
report_path = Path(config['artifact_root']) / 'full_corpus_and_temporal_gate_audit.json'
report = read(report_path)
assert status['active_pid'] is None and status['state'] == 'complete_snapshot_and47profiles_verified'
assert guard['passed'] and guard['exit_code'] == 0 and not guard['resource_guard_aborted']
assert guard['proof_sha256'] == sha(report_path)
assert report['passed'] and report['configuration_sha256'] == sha(ROOT / CONFIG)
assert report['snapshot_file_count'] == 2441 and report['snapshot_bytes'] == 8705622732
assert report['original_pretraining_epochs_executed'] == report['neural_fit_inference_or_SVM'] == 0
assert not report['historical_ICM_pretraining_corpus_equivalence_certified']
assert report['environment'] == config['environment']
rows = report['Monash_profiles']
assert len(rows) == 47 and not report['profile_failures_preserved']
assert [r['path'] for r in rows] == config['all_source_getter_paths']['short_forecast']
registry = {r['path']: r for r in read(ROOT / config['integrity'])['records']}
raw = report['raw_snapshot_checks']
assert len(raw) == len({r['path'] for r in raw}) == len(registry) == 2441
for r in raw:
    expected = registry[r['path']]
    assert expected['status'] == 'verified' and r['bytes'] == expected['size_bytes']
    assert all(r[k] == expected[k] for k in ['sha256', 'git_blob_sha1'])
for r in rows:
    assert r['source_sha256'] == registry[r['path']]['sha256']
    assert r['streaming_all_numeric_tokens_validated'] and not r['eager_value_frame_constructed']
    assert not r['neural_dataset_sample_getitem_called']
    assert not r['source_preprocessing_or_model_equivalence_certified']
    assert r['actual_training_rows'] == int(.6 * r['source_length_field'])
    assert r['original_source_field_selected'] == (r['source_length_field'] >= 256)
    assert r['observed_training_time_selected'] == ((r['training_temporal_max'] or 0) >= 256)
for key, flag in [('source_count_selected', 'original_source_field_selected'),
                  ('observed_training_time_selected', 'observed_training_time_selected')]:
    assert report[key] == [r['path'] for r in rows if r[flag]]
assert report['source_selected_training_items'] == sum(r['actual_training_rows'] for r in rows if r['original_source_field_selected'])
assert report['temporal_selected_training_items'] == sum(r['actual_training_rows'] for r in rows if r['observed_training_time_selected'])
old_config = read(ROOT / config['previous_configuration'])
assert config['settings'] == old_config['settings']
assert config['all_source_getter_paths'] == old_config['all_source_getter_paths']
assert sha(folder / 'preserved_v1_original_guard.json') == config['previous_failed_guard_sha256']
old_guard = read(folder / 'preserved_v1_original_guard.json')
assert old_guard['resource_guard_aborted'] and not old_guard['passed'] and old_guard['proof_sha256'] is None
assert old_guard['peak_tree_private_bytes'] > old_config['settings']['maximum_worker_private_bytes']
assert old_guard['abort_snapshot']['commit_available_bytes'] > old_config['settings']['emergency_commit_headroom_bytes']
assert read(folder / 'preserved_v1_progress.json')['profiles'] == 13

checks = 0
for name, path in [
    ('source_metadata_validation.xml', 'tmp/MOMENT_ICM_pretraining_inputs_validation_20261015_v3.xml'),
    ('streaming_metadata_validation.xml', 'tmp/MOMENT_ICM_streaming_scope_validation_20261015_v1.xml')]:
    suites = list(ET.parse(ROOT / path).getroot().iter('testsuite'))
    assert sum(int(s.get('tests', 0)) for s in suites) == 14
    assert all(int(s.get(k, 0)) == 0 for s in suites for k in ['failures', 'errors', 'skipped'])
    checks += 14
    shutil.copyfile(ROOT / path, folder / name)

source_names = [
    'moment_icm_pretraining_inputs_v1.py', 'audit_moment_icm_pretraining_corpus_v1.py',
    'register_moment_icm_pretraining_corpus_audit_v1.py', 'moment_icm_streaming_scope_v2.py',
    'audit_moment_icm_pretraining_corpus_v2.py', 'register_moment_icm_pretraining_corpus_audit_v2.py']
copies = {name: ROOT / 'scripts/flow_matching' / name for name in source_names}
copies.update({
    'test_fm_moment_icm_pretraining_inputs_v1.py': ROOT / 'tests/test_fm_moment_icm_pretraining_inputs_v1.py',
    'test_fm_moment_icm_streaming_scope_v2.py': ROOT / 'tests/test_fm_moment_icm_streaming_scope_v2.py',
    'frozen_streaming_v2_configuration.json': ROOT / CONFIG,
    'actual_v2_controller_launch.json': base / 'controller_launch.json',
    'actual_v2_worker_launch.json': base / 'worker_launch.json',
    'actual_v2_terminal_status.json': base / 'status.json',
    'actual_v2_guard.json': base / 'guard_receipt.json',
    'actual_full_corpus_and_temporal_gate_audit.json': report_path,
    'actual_v2_final_progress.json': Path(config['artifact_root']) / 'progress.json',
})
for name, source in copies.items():
    assert not (folder / name).exists(), name
    shutil.copyfile(source, folder / name)
    assert sha(source) == sha(folder / name)

original_sources = {}
for pin in config['source_receipts']:
    if '/moment_research/original/' not in pin['path']:
        continue
    p = ROOT / pin['path']
    value = p.read_bytes().decode('utf-8')
    assert hashlib.sha256(value.encode('utf-8')).hexdigest() == pin['sha256']
    original_sources[pin['path']] = dict(sha256=pin['sha256'], original_utf8=value)
write(folder / 'related_original_source_snapshot.json', dict(
    source_commit=config['related_source_commit'], files=original_sources,
    original_bytes_recoverable_by_UTF8_encoding=True, historical_ICM_pipeline_certified=False))

buffer = io.StringIO(newline='')
fields = ['dataset', 'source_length_field', 'actual_training_rows', 'training_temporal_min',
          'training_temporal_max', 'training_temporal_total_points', 'training_rows_length_ge_window',
          'original_source_field_selected', 'observed_training_time_selected',
          'raw_values_parsed_and_discarded', 'raw_missing_tokens', 'raw_nonfinite_numeric_tokens',
          'original_special_splitter_tail_omitted_points', 'source_sha256']
writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
writer.writeheader()
for r in rows:
    writer.writerow(dict(dataset=Path(r['path']).stem, **{k: r[k] for k in fields[1:]}))
(folder / 'all47_monash_source_vs_temporal_input_scope.csv').write_text(buffer.getvalue(), encoding='utf-8', newline='')

summary = dict(
    captured_utc=datetime.now(timezone.utc).isoformat(), snapshot_files=2441, snapshot_bytes=8705622732,
    source_getter_candidate_counts={k: len(v) for k, v in config['all_source_getter_paths'].items()},
    Monash_files_profiled=47, Monash_source_count_selected=len(report['source_count_selected']),
    Monash_temporal_control_selected=len(report['observed_training_time_selected']),
    source_selected_training_items=report['source_selected_training_items'],
    temporal_selected_training_items=report['temporal_selected_training_items'],
    added_by_temporal_control=report['added_by_temporal_control'],
    excluded_by_temporal_control=report['excluded_by_temporal_control'],
    v1_failed_original_guard_preserved=True, v2_complete_guarded_audit_passed=True,
    v1_peak_private_bytes=old_guard['peak_tree_private_bytes'],
    v2_peak_private_bytes=guard['peak_tree_private_bytes'], v2_peak_rss_bytes=guard['peak_tree_rss_bytes'],
    tests_passed=checks, original_neural_pretraining_epochs_executed=0, new_model_performance_results=0,
    temporal_gate_is_a_separate_metadata_control=True,
    changed_input_scope_is_not_an_isolated_architecture_ablation=True,
    full_corpus_optimizer_step_and_token_budget_certified=False,
    historical_ICM_corpus_and_pipeline_equivalence_certified=False,
    optional_FRED_absence_proves_missing_required_paper_data=False,
    all_original_experiments_complete=False)
write(folder / 'complete_input_scope_summary.json', summary)
added = ', '.join(Path(p).stem for p in report['added_by_temporal_control']) or '(none)'
excluded = ', '.join(Path(p).stem for p in report['excluded_by_temporal_control']) or '(none)'
(folder / 'README.md').write_text(f'''# MOMENT/ICM：完整输入范围审计与流式内存修复

全部 2,441 个当前登记的 PILE 原始文件已重新校验 SHA256、Git blob SHA1、大小及读取前后状态，总计 8,705,622,732 字节；全部 47 个 Monash 文件均已逐值解析并完成训练长度统计。原数据没有覆盖。这里完成的是数据范围审计，神经网络预训练、推理、SVM及新增性能结果均为 0。

先前 v1 元数据任务在完成文件校验、处理 13 个 Monash 文件后，构建大型来源数据表时超过原定 6 GiB 进程树上限，被资源保护终止，没有最终成功证明。原失败回执和进度完整保留。v2 在新配置和新进程中逐个验证全部数值后丢弃数值对象，只保留长度；固定 pandas/NumPy 的 seed=13 行采样、训练比例 0.6、上下文 256，以及来源特殊长序列的分块和尾部丢弃规则。v2 未减少文件、样本或预算，准入与保护上限也没有降低；相关模型和正在运行的训练未修改。原数值对象的精确内存分配根因尚未单独证明。

两套共 28 项检查通过，其中 8 个真实小型来源文件的原解析器/数据类与流式实现的全部原有统计字段相同。最终 47 个文件均完成验证；大型文件没有再次构建完整数值表，因此不据此声称所有原数值表逐坐标重放。

相关 MOMENT 来源代码把 Monash 的 `length_timeseries` 设为拆分前的数据表行数，即序列条数。原筛选用该字段是否大于等于 256，选中 {len(report['source_count_selected'])}/47 个文件、{report['source_selected_training_items']:,} 个训练条目。单独登记的时间长度对照判断训练序列最大长度是否大于等于 256，选中 {len(report['observed_training_time_selected'])}/47 个文件、{report['temporal_selected_training_items']:,} 个训练条目。它不要求每个条目长度均合格，也不等于原作者的已证实修复方案。完整逐文件数字在 CSV 中。

时间长度对照新增：{added}。

时间长度对照排除：{excluded}。

例如 NN5 daily：来源字段为 111 条序列，训练有 66 条、各 791 个时间点；来源字段筛选将其排除，时间长度对照将其保留。两套筛选改变数据组成与一个 epoch 的数据预算，应分别报告，不能作为仅改变模型结构的消融。这里的训练条目只覆盖 Monash 分支，尚未证明整个多来源预训练集的优化器步数或 token 总预算。

来源 getter 在当前快照返回分类 157、长预测 9、短预测 47、可选 FRED 0、异常检测 1,949，共 2,162 个筛选前候选引用。这不是 2,162 个已认证可训练数据集。Windows 路径兼容层保留来源对 CharacterTrajectories 的明确排除，原生 Windows 差异另行记录。可选 `fred/preprocessed` 为空不能证明缺失论文必需素材；Monash FRED-MD 已包含。公开镜像的当前内容、README 描述和历史 ICM 预训练语料不应直接等同。

论文：[2409.13530v1](https://arxiv.org/abs/2409.13530v1)。相关来源提交 `{config['related_source_commit']}` 的关键原文代码按原 UTF8 字节封装归档，可以按 SHA256 恢复核对。它是相关 MOMENT 代码，尚未认证为 ICM 原作者完整流水线。原论文 Tiny/ICM 一整 epoch 预训练、全部预测、117 数据集 SVM、beta 消融及严格训练集预处理对照仍需执行。此前完成的 117 分类数据准备和单独的 72+6 异常检测复核保持独立。

研究的最低范围仍包含原登记的 45 篇论文、681 条方法/基线/消融和 273 条数据任务引用，以及后续原文发现的实验。本审计不代表全部研究实验已完成。
''', encoding='utf-8')
(folder / '.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n', encoding='utf-8')
shutil.copyfile(__file__, folder / 'publication_source.py')
write(folder / 'archive_validation.json', dict(
    complete_snapshot_files=2441, complete_monash_profiles=47, tests_passed=28,
    original_failure_preserved=True, source_and_temporal_controls_separate=True,
    new_model_results=0, all_original_experiments_complete=False,
    files_sha256={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}))
ara = ROOT / 'projects/flow_matching_research/ara/execution/moment_ICM_complete_pretraining_input_scope.2026-10-15_v2.json'
write(ara, dict(
    paper_id='moment_long_context', configuration=CONFIG, evidence=EVIDENCE,
    entire_registered_snapshot_integrity_audit_passed=True, files=2441, Monash_profiles=47,
    preserved_failed_metadata_v1=True, diagnosed_streaming_metadata_v2_passed=True,
    original_pretraining_epochs_executed=0, new_performance_results=0,
    historical_ICM_pipeline_certified=False, full_pretraining_step_budget_certified=False,
    previous_goal_turn_classification='no_progress_for_experiments_user_hook_troubleshooting_only',
    current_goal_turn_classification='progress_completed_full_corpus_input_audit',
    required_remaining='Original full pretraining,forecasting,context expansion,SVM,beta and strict controls;all other registered unfinished families remain.',
    all_original_experiments_complete=False))
targets = [EVIDENCE, ara.relative_to(ROOT).as_posix(), CONFIG, config['previous_configuration']]
targets += ['scripts/flow_matching/' + name for name in source_names]
targets += ['tests/test_fm_moment_icm_pretraining_inputs_v1.py', 'tests/test_fm_moment_icm_streaming_scope_v2.py']
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    paths.update(subprocess.check_output(['git', *args, '--', *targets], cwd=ROOT, text=True).splitlines())
git = lambda cwd, *args: subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
assert paths and not git(ROOT, 'diff', '--cached', '--name-only')
write(ROOT / 'tmp/MOMENT_ICM_pretraining_scope_publication_manifest_20261015_v2.json', dict(
    previous_root=git(ROOT, 'rev-parse', 'HEAD'),
    previous_parent=git(ROOT.parent / 'flow-matching-research', 'rev-parse', 'HEAD'), paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths), checks=28, complete_raw_files=2441, Monash=47,
    source_selected=summary['Monash_source_count_selected'], temporal_selected=summary['Monash_temporal_control_selected'], new_model_results=0)))
