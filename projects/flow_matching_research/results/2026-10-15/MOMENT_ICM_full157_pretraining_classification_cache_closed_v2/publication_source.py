"""Archive full recovered preparation and independent157cache closure/index."""
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from scripts.flow_matching import verify_moment_icm_PILE_classification_v2 as verifier

ROOT, read, sha, write = verifier.ROOT, verifier.read, verifier.sha, verifier.write
config = read(ROOT / verifier.CONFIG)
previous, recovered, jobs = verifier.verify(config)
base = ROOT / config['state_root']; status = read(base / 'status.json')
assert status['state'] == 'full157_independent_bounded_cache_closure_passed' and status['active_pid'] is None
guard = read(base / 'guard_receipt.json')
proof_path = Path(config['artifact_root']) / 'full157_classification_cache_closure.json'; proof = read(proof_path)
assert guard['passed'] and guard['exit_code'] == 0 and not guard['resource_guard_aborted']
assert guard['proof_sha256'] == sha(proof_path) and proof['configuration_sha256'] == sha(ROOT / verifier.CONFIG)
assert proof['passed'] and proof['datasets'] == len(proof['records']) == 157 and proof['array_files'] == 471
assert proof['original_source_successes'] == 156 and proof['recovered_source_successes'] == 1
assert proof['original_failed_outcomes_preserved'] == 1 and proof['original_TRAIN_cases'] == 115112
assert proof['full_signal_coordinates'] == 1649646336 and proof['original_raw_length_gate_selected'] == 95
assert proof['neural_training_inference_SVM'] == 0 and not proof['strict_train_only_preprocessing']
assert not proof['historical_ICM_pipeline_certified'] and not proof['full_PILE_loader_integrated']
recovery_base = ROOT / recovered['state_root']; recovery_guard = read(recovery_base / 'guard_receipt.json')
recovery_proof = Path(recovered['artifact_root']) / 'InsectWingbeat/pretraining_preparation.json'
r = read(recovery_proof)
assert recovery_guard['passed'] and recovery_guard['proof_sha256'] == sha(recovery_proof)
assert r['record']['instances'] == 25000 and r['record']['channels'] == 200
assert r['record']['source_coordinates_copied_and_checked'] == 1280000000
assert r['original_complete_prefix_cases_compared'] == 5414
assert not r['original_raw_length_selected'] and r['source_pre_resampling_temporal_length'] == 22
assert r['original_failed_outcome_not_overwritten'] and not r['full_original_failed_export_equivalence_certified']
folder = ROOT / 'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_full157_pretraining_classification_cache_closed_v2'
folder.mkdir(parents=True, exist_ok=False)
for name, source in {
    'frozen_full157_bounded_closure_configuration.json': ROOT / verifier.CONFIG,
    'verify_moment_icm_PILE_classification_v2.py': ROOT / 'scripts/flow_matching/verify_moment_icm_PILE_classification_v2.py',
    'test_fm_moment_icm_PILE_classification_closure_v2.py': ROOT / 'tests/test_fm_moment_icm_PILE_classification_closure_v2.py',
    'actual_full157_independent_cache_closure.json': proof_path,
    'actual_full157_closure_guard.json': base / 'guard_receipt.json',
    'actual_closure_controller_launch.json': base / 'controller_launch.json',
    'actual_closure_worker_launch.json': base / 'worker_launch.json',
    'actual_closure_terminal_status.json': base / 'status.json',
    'actual_full_InsectWingbeat_recovered_preparation.json': recovery_proof,
    'actual_full_InsectWingbeat_recovery_guard.json': recovery_base / 'guard_receipt.json',
    'actual_InsectWingbeat_recovery_terminal_status.json': recovery_base / 'status.json',
    'actual_InsectWingbeat_final_block_progress.json': Path(recovered['artifact_root']) / 'InsectWingbeat/chunked_export_progress.json',
    'preserved_InsectWingbeat_original_failed_guard.json': Path(recovered['previous_failed_guard']),
}.items():
    shutil.copyfile(source, folder / name); assert sha(source) == sha(folder / name)
xml = ROOT / 'tmp/MOMENT_ICM_PILE_classification_closure_validation_20261015_v2.xml'
suites = list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 8
assert all(int(s.get(k, 0)) == 0 for s in suites for k in ['failures', 'errors', 'skipped'])
shutil.copyfile(xml, folder / 'bounded_closure_validation.xml')

entries = []
for row, job in zip(proof['records'], jobs):
    assert row['dataset'] == job['dataset']
    record = row['record']
    assert record['all_signal_coordinates_finite'] and record['all_padding_masks_binary']
    assert record['complete_raw_case_channel_and_row_reconciliation'] and record['label_arrays_absent']
    assert record['whole_signal_array_never_mapped']
    assert record['complete_read_mapping_scans']['signal']['checked_coordinates'] == record['coordinates']
    assert record['complete_read_mapping_scans']['mask']['checked_coordinates'] == record['instances'] * 256
    assert record['complete_read_mapping_scans']['rows']['checked_coordinates'] == record['instances']
    entries.append(dict(dataset=row['dataset'], source_track=row['source_track'],
        source_configuration=row['source_configuration'], source_directory=row['source_directory'],
        source_proof_sha256=row['source_proof_sha256'], source_guard_sha256=row['source_guard_sha256'],
        raw_TRAIN=job['TRAIN'], cases=record['instances'], channels=record['channels'], length=256,
        values_dtype=record['values_dtype'], cache_files_sha256=record['file_sha256'],
        source_pre_resampling_temporal_length=row['source_pre_resampling_temporal_length'],
        selected_by_original_raw_length_gate=row['original_raw_length_selected'],
        disclosed_paired_casewise_split=row['disclosed_paired_casewise_split']))
index_path = 'configs/reproducibility/fm_moment_ICM_classification_pretraining_cache_index.2026-10-15_v1.json'
assert not (ROOT / index_path).exists()
index = dict(schema_version=1, paper_id='moment_long_context', paper_citation=config['paper_citation'],
    cache_closure_configuration=verifier.CONFIG, cache_closure_configuration_sha256=sha(ROOT / verifier.CONFIG),
    complete_cache_closure_proof=folder.relative_to(ROOT).as_posix() + '/actual_full157_independent_cache_closure.json',
    complete_cache_closure_proof_sha256=sha(proof_path), complete_candidates=157,
    original_raw_length_gate='source_pre_resampling_temporal_length_greater_equal256',
    selected_candidates=95, retained_unselected_candidates=62,
    cache_instances=115112, cache_coordinates=1649646336, entries=entries,
    raw_short_candidates_retained_but_not_silently_selected=True,
    per_model_training_input_selection_and_budget_not_executed=True,
    grouped_casewise_split_is_disclosed_reconstruction=True,
    future_grouped_independent_and_ICM_models_must_share_split_and_batch_manifests=True,
    strict_train_only_preprocessing=False, source_fullTRAIN_standardizer_precedes_validation_split=True,
    historical_ICM_pipeline_certified=False, complete_PILE_training_loader_integrated=False,
    neural_model_performance_results=0, all_original_experiments_complete=False)
write(ROOT / index_path, index)
shutil.copyfile(ROOT / index_path, folder / 'frozen_classification_pretraining_cache_index.json')
buffer = io.StringIO(newline='')
fields = ['dataset', 'source_track', 'cases', 'channels', 'length', 'source_pre_resampling_temporal_length',
          'selected_by_original_raw_length_gate', 'values_dtype', 'source_directory']
writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n'); writer.writeheader()
for row in entries: writer.writerow({k: row[k] for k in fields})
(folder / 'all157_classification_cache_and_source_eligibility.csv').write_text(buffer.getvalue(), encoding='utf-8', newline='')
write(folder / 'complete_classification_cache_summary.json', dict(
    complete_candidates=157, source_original_successes=156, diagnosed_recovered_successes=1,
    original_failed_outcomes_preserved=1, full_cached_arrays=471, original_TRAIN_cases=115112,
    full_signal_coordinates=1649646336, original_raw_length_selected=95, retained_unselected=62,
    recovery_peak_RSS_bytes=recovery_guard['peak_tree_rss_bytes'],
    recovery_peak_private_bytes=recovery_guard['peak_tree_private_bytes'],
    closure_peak_RSS_bytes=guard['peak_tree_rss_bytes'], closure_peak_private_bytes=guard['peak_tree_private_bytes'],
    independent_full_cache_byte_shape_row_finite_mask_raw_cardinality_audit_passed=True,
    old5414case_complete_prefix_comparison_independently_repeated=True,
    checks_passed=8, historical_ICM_pipeline_certified=False, strict_train_only_preprocessing=False,
    new_model_performance_results=0, full_PILE_pretraining_complete=False, all_original_experiments_complete=False))
(folder / 'README.md').write_text('''# 完整 157 个 PILE 分类缓存已完成独立核验

当前全部 157 个来源可见分类候选的数据准备已经完成并核验，证明链明确区分 156 个原始成功结果与 1 个 InsectWingbeat 分块恢复结果。原始 InsectWingbeat 的 RSS 失败保留，不能表述为首次 157/157 成功。471 个完整数组、115,112 个官方 TRAIN 实例、1,649,646,336 个信号坐标均通过独立缓存检查，原始数据和旧部分产物未覆盖。

InsectWingbeat 恢复实际完整写出 25,000 个实例×200 通道×256 时间点，共 1,280,000,000 个坐标，退出码 0，保护通过。恢复峰值 RSS 2,009,829,376 字节、private 1,975,607,296 字节，仍在原 3 GiB 进程树上限内。来源类、原数据、标准化/插值/填充、精度、实例与通道范围均未变，仅释放输出映射的频率改变。固定旧部分产物的前 5,414 个完整实例与新产物前缀哈希相同；独立核验进程再次重读新旧前缀并验证相同。对原失败后未写出的实例，不能声称有完整旧产物可作逐字节对照。

独立检查逐数据集验证准确的配置、来源命令、证明哈希和资源回执；恢复结果还核对来源数值配置及原失败绑定。重读原始 TRAIN 的 SHA256和独立实例/通道计数，重新计算全部 471 个数组哈希，核对完整 shape、原生 dtype、每个原实例行号，以及全部信号有限性和全部 mask 二值性。每个读取映射仅覆盖至多 16 MiB 信号块，检查后释放；没有把 10.24 GB 数值数组整体映射并逐页保留。真实独立核验退出码 0、峰值 RSS 68,395,008 字节（约 65 MiB）、private 35,516,416 字节，资源保护通过。

8 项检查覆盖完整分块访问、最后一块非有限值和非二值 mask、行序、精度改变、文件额外尾字节、额外标签数组、缺失实例及恢复命令/配置/原失败链错误。模型训练、推理、SVM 和新增性能结果为 0。

来源原始时间长度筛选选中 95/157 个分类候选，其余 62 个完整缓存仍保留并标记未选中，不能因为变换后都是长度 256 就悄悄纳入。特别是 InsectWingbeat 原时间长度为 22，尽管有 200 通道并生成大缓存，仍不满足来源的原始长度门槛。新的配置索引记录全部 157 个目录、哈希、原始长度、筛选标志，以及原先展平通道切分和披露的完整实例 6/7 切分差异，供下一步训练入口从 configs 读取。

这一步没有独立重写来源标准化和插值数值。来源完整 TRAIN 拟合标准化后才切验证的规则仍包含未来验证统计，不是严格训练集专属预处理；没有使用 TEST 信号，也没有导出预训练标签。分组前按完整实例切分是披露的重建选择，后续独立通道与 ICM 必须共用它，不能声称等同原来源的展平边界或历史 ICM 完整流水线。

这是 PILE 分类分支的完整缓存准备与核验。长预测、Monash、异常分支的完整加载/分组/采样、整个语料的训练步数与 token、Tiny/ICM 一整 epoch、权重保存与原生独立复核、下游预测/117 数据集 SVM/beta/严格预处理对照仍需继续；本缓存不会代替整个预训练语料。至少 45 篇论文/681 方法记录/273 数据任务及后来发现的原文实验范围保持不变，全部实验尚未完成。

论文：[2409.13530v1](https://arxiv.org/abs/2409.13530v1)。大数组保留在配置指定的 F 盘，Git 只包含代码、配置、表格和真实证据。
''', encoding='utf-8')
(folder / '.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n', encoding='utf-8')
shutil.copyfile(__file__, folder / 'publication_source.py')
write(folder / 'archive_validation.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    datasets=157, files=471, source_original=156, source_recovered=1, original_failure_preserved=1,
    complete_signal_coordinates=1649646336, tests_passed=8, new_model_results=0,
    files_sha256={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}, all_original_experiments_complete=False))
ara = ROOT / 'projects/flow_matching_research/ara/execution/moment_ICM_full157_pretraining_cache_closed.2026-10-15_v2.json'
write(ara, dict(paper_id='moment_long_context', configuration=verifier.CONFIG, cache_index=index_path,
    evidence=folder.relative_to(ROOT).as_posix(), complete_classification_candidates=157,
    original_source_successes=156, recovered_source_successes=1, original_failed_outcomes_preserved=1,
    full_cached_arrays=471, original_TRAIN_cases=115112, full_signal_coordinates=1649646336,
    complete_independent_cache_audit_passed=True, source_gate_selected_classification_candidates=95,
    strict_train_only_preprocessing=False, full_PILE_loader_integrated=False, actual_pretraining_epochs=0,
    new_performance_results=0, previous_goal_turn_classification='progress_all157attempted_and_diagnosed_recovery_started',
    current_goal_turn_classification='progress_full_recovery_and_all157_independent_cache_closure_passed',
    all_original_experiments_complete=False))
targets = [verifier.CONFIG, index_path, folder.relative_to(ROOT).as_posix(), ara.relative_to(ROOT).as_posix(),
    'scripts/flow_matching/verify_moment_icm_PILE_classification_v2.py',
    'tests/test_fm_moment_icm_PILE_classification_closure_v2.py']
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    paths.update(subprocess.check_output(['git', *args, '--', *targets], cwd=ROOT, text=True).splitlines())
git = lambda cwd, *args: subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
assert paths and not git(ROOT, 'diff', '--cached', '--name-only')
write(ROOT / 'tmp/MOMENT_ICM_full157_cache_closure_publication_manifest_20261015_v2.json', dict(
    previous_root=git(ROOT, 'rev-parse', 'HEAD'),
    previous_parent=git(ROOT.parent / 'flow-matching-research', 'rev-parse', 'HEAD'), paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths), classification_candidates=157, independent_files=471,
    TRAIN_instances=115112, coordinates=1649646336, tests=8, new_model_results=0)))
