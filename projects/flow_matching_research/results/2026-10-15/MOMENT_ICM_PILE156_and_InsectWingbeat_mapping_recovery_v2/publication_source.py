"""Preserve all156 original successes, the failed157th and live diagnosed recovery."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil

from scripts.flow_matching import recover_moment_icm_insect_chunked_export_v2 as recovery
from scripts.flow_matching import verify_moment_icm_PILE_classification_v1 as closure

ROOT, read, sha, write = recovery.ROOT, recovery.read, recovery.sha, recovery.write
config = read(ROOT / recovery.CONFIG); recovery.verify(config)
previous = read(ROOT / recovery.original.CONFIG); recovery.original.verify(previous)
prepared_closure = read(ROOT / closure.CONFIG); closure.verify(prepared_closure)
source_base = ROOT / previous['state_root']; source_status = read(source_base / 'status.json')
assert source_status['state'] == 'all157_classification_preparation_attempts_terminal'
assert source_status['completed_count'] == len(source_status['completed_datasets']) == 156
assert set(source_status['failed_preserved']) == {'InsectWingbeat'}
failure = read(config['previous_failed_guard'])
assert not failure['passed'] and failure['resource_guard_aborted'] and failure['proof_sha256'] is None
assert failure['peak_tree_rss_bytes'] > previous['settings']['maximum_worker_rss_bytes']
assert failure['peak_tree_private_bytes'] < previous['settings']['maximum_worker_private_bytes']
base = ROOT / config['state_root']; launch = read(base / 'controller_launch.json')
p = psutil.Process(launch['pid'])
assert p.create_time() == launch['create_time'] and p.cmdline() == launch['command']
assert launch['configuration_sha256'] == sha(ROOT / recovery.CONFIG)
live_status_bytes = (base / 'status.json').read_bytes(); live_status = json.loads(live_status_bytes)
assert not live_status.get('resource_guard_aborted')
folder = ROOT / 'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_PILE156_and_InsectWingbeat_mapping_recovery_v2'
folder.mkdir(parents=True, exist_ok=False)
for name, source in {
    'original_all157_terminal_status.json': source_base / 'status.json',
    'original_all157_controller_launch.json': source_base / 'controller_launch.json',
    'preserved_InsectWingbeat_original_failed_guard.json': Path(config['previous_failed_guard']),
    'frozen_InsectWingbeat_chunked_recovery_configuration.json': ROOT / recovery.CONFIG,
    'frozen_prepared_original_only_closure_configuration.json': ROOT / closure.CONFIG,
    'recover_moment_icm_insect_chunked_export_v2.py': ROOT / 'scripts/flow_matching/recover_moment_icm_insect_chunked_export_v2.py',
    'verify_moment_icm_PILE_classification_v1.py': ROOT / 'scripts/flow_matching/verify_moment_icm_PILE_classification_v1.py',
    'test_fm_moment_icm_chunked_export_v2.py': ROOT / 'tests/test_fm_moment_icm_chunked_export_v2.py',
    'test_fm_moment_icm_PILE_classification_closure_v1.py': ROOT / 'tests/test_fm_moment_icm_PILE_classification_closure_v1.py',
    'actual_recovery_controller_launch.json': base / 'controller_launch.json',
    'actual_recovery_worker_launch.json': base / 'worker_launch.json',
}.items():
    shutil.copyfile(source, folder / name); assert sha(source) == sha(folder / name)
(folder / 'actual_recovery_live_status_snapshot.json').write_bytes(live_status_bytes)
progress_path = Path(config['artifact_root']) / 'InsectWingbeat/chunked_export_progress.json'
progress_bytes = progress_path.read_bytes(); progress = json.loads(progress_bytes)
assert progress['completed_cases'] > 0 and progress['required_cases'] == 25000 and progress['channels'] == 200
(folder / 'actual_recovery_completed_block_snapshot.json').write_bytes(progress_bytes)
tests = 0
for name, path, count in [
    ('stored_cache_closure_validation.xml', 'tmp/MOMENT_ICM_PILE_classification_closure_validation_20261015_v1.xml', 8),
    ('chunked_output_validation.xml', 'tmp/MOMENT_ICM_chunked_export_validation_20261015_v2.xml', 4)]:
    suites = list(ET.parse(ROOT / path).getroot().iter('testsuite'))
    assert sum(int(s.get('tests', 0)) for s in suites) == count
    assert all(int(s.get(k, 0)) == 0 for s in suites for k in ['failures', 'errors', 'skipped'])
    tests += count; shutil.copyfile(ROOT / path, folder / name)
originals = {}; cases = coordinates = selected = 0
for name in source_status['completed_datasets']:
    proof_path = Path(previous['artifact_root']) / name / 'pretraining_preparation.json'
    guard_path = source_base / 'attempts' / name / 'guard_receipt.json'
    proof, guard = read(proof_path), read(guard_path)
    assert proof['passed'] and proof['configuration_sha256'] == sha(ROOT / recovery.original.CONFIG)
    assert guard['passed'] and guard['exit_code'] == 0 and not guard['resource_guard_aborted']
    assert guard['proof_sha256'] == sha(proof_path)
    assert proof['neural_model_fit_inference_SVM'] == 0 and not proof['TEST_signal_values_processed']
    cases += proof['record']['instances']; coordinates += proof['record']['source_coordinates_copied_and_checked']
    selected += int(proof['original_raw_length_selected'])
    originals[name] = dict(source_sha256=sha(proof_path), guard_sha256=sha(guard_path),
        original_source_utf8=proof_path.read_bytes().decode('utf-8'),
        original_guard_utf8=guard_path.read_bytes().decode('utf-8'))
write(folder / 'all156_original_success_proofs_and_guards.json', dict(
    original_UTF8_bytes_recoverable=True, completed=156, required=157,
    source_failures_preserved=['InsectWingbeat'], records=originals))
summary = dict(captured_utc=datetime.now(timezone.utc).isoformat(), required_classification_candidates=157,
    original_attempts_terminal=157, original_successes=156, original_failures_preserved=1,
    original_success_TRAIN_cases=cases, original_success_transformed_coordinates=coordinates,
    original_raw_length_gate_selected_among_successes=selected,
    preserved_failed_RSS_bytes=failure['peak_tree_rss_bytes'], preserved_failed_private_bytes=failure['peak_tree_private_bytes'],
    original_partial_values_file_bytes=config['preserved_partial_outputs']['values.npy']['bytes'],
    conservative_original_complete_prefix_cases=5414, diagnosed_recovery_required_cases=25000,
    diagnosed_recovery_required_channels=200, recovery_completed_cases_at_snapshot=progress['completed_cases'],
    recovery_controller_exact_live=True, recovery_complete_claim=False, original_caps_unchanged=True,
    recovery_peak_RSS_at_snapshot=live_status.get('peak_tree_rss_bytes'),
    prepared_closure_checks=8, chunked_output_checks=4, new_model_performance_results=0,
    full157_cache_independent_closure_not_executed=True, full_PILE_pretraining_not_complete=True,
    all_original_experiments_complete=False)
write(folder / 'terminal_source_and_live_recovery_summary.json', summary)
(folder / 'README.md').write_text(f'''# 原始分类分支：156 成功、1 次 RSS 失败与独立分块恢复

全 157 个当前来源分类候选的首次准备均已实际运行至终态，156 个成功、1 个失败（InsectWingbeat）。成功部分共 {cases:,} 个完整 TRAIN 实例、{coordinates:,} 个变换后坐标；全部 156 份来源证明和保护回执按原 UTF8 字节归档。没有删去失败候选或把它写成成功。当前这仍是分类分支的数据准备，不是神经网络预训练。

InsectWingbeat 原失败回执：进程树 RSS 为 3,266,428,928 字节，超过原 3 GiB 上限；private 为 1,978,892,288 字节，低于该上限；系统 commit 余量仍高于紧急线。stderr 没有 Python 例外，子进程由 RSS 保护终止，无最终成功证明。原 partial values.npy 已创建为 25,000×200×256 的 float64 数组（10,240,000,128 字节），掩码末时间点为 1 的连续前缀有 5,415 个实例。证明已进入全文件映射写出阶段，末个实例可能只写完部分通道，故仅保守采用此前 5,414 个完整实例作旧输出对照。映射驻留页随写出积累是本轮针对的内存问题，精确分配器细节未独立剖析。

新 v2 使用独立配置、目录及进程，保留同一来源类、原训练文件、全量 25,000 个实例/200 通道、相同标准化/插值/填充、原生精度及原 16/21/16 GiB 数据资源门槛和 3 GiB 进程树上限。只改变输出映射的生存期：每写出最多 64 MiB 的信号块后 flush/unmap，再打开下一块；每个输出坐标仍与原来源样本逐字节核对。原始文件、旧部分产物、失败回执和原先完成的 156 个数据集均未改写。

4 项分块导出检查通过，其中 3 个真实来源数据集的新旧导出文件哈希完全相同（强制极小块以覆盖多块切换）。恢复任务还将把新产物前 5,414 个完整实例与固定旧文件前缀逐字节哈希比较；该比较须等新产物完整写出后完成，不能提前称通过。本证据包捕获时，恢复控制器及命令身份已核验，已完成 {progress['completed_cases']:,}/25,000 个实例的块写出，尚未宣称整个恢复成功。实际状态和进度的时间点见快照，作业继续运行；任何新失败仍保留而不自动重试。

同时准备了独立缓存检查器，8 项检查覆盖完整形状/通道/行序、全部有限信号、二值掩码、哈希变更、额外标签数组和语义损坏。它的原始版本要求全部 157 个首次来源任务成功；当前有 1 个保留失败，因此未启动或宣称整套闭合通过。下一步必须在恢复结束后显式核对 156 原来源 + 1 分块来源的不同命令/配置/证明链，保留原失败，并让大型数组的只读检查同样使用分块映射以避免重复 RSS 问题。不能改写旧保护回执或假造原始 157/157 成功。

预训练默认 padding 与下游 117 分类缓存的插值例外保持分开；来源原始时间长度筛选与后续重建选择必须显式登记。来源完整 TRAIN 拟合标准化后才切验证的规则仍是来源对照，不能称严格训练集专属预处理。全 PILE 的预测/Monash/异常分支、整个预训练加载器、实际步数和 token、Tiny/ICM 一整 epoch、最终权重和原生复核、下游预测/SVM/beta/严格对照仍需完成。此轮新增神经网络性能结果为 0；至少 45 篇论文/681 方法记录/273 数据任务及后续新增实验范围不变。

论文：[2409.13530v1](https://arxiv.org/abs/2409.13530v1)。所有大数组仍在配置指定的 F 盘目录。来源数值等价证据不等于历史 ICM 原作者完整流水线认证。
''', encoding='utf-8')
(folder / '.gitattributes').write_text('*.py -text\n*.json -text\n', encoding='utf-8')
shutil.copyfile(__file__, folder / 'publication_source.py')
write(folder / 'archive_validation.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    original_terminal_attempts=157, original_successes=156, preserved_original_failures=1,
    recovery_started_complete_unproven=True, tests_passed=12, new_model_results=0,
    files_sha256={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}, all_original_experiments_complete=False))
ara = ROOT / 'projects/flow_matching_research/ara/execution/moment_ICM_PILE156_and_InsectWingbeat_recovery.2026-10-15_v2.json'
write(ara, dict(paper_id='moment_long_context', source_configuration=recovery.original.CONFIG,
    recovery_configuration=recovery.CONFIG, prepared_original_only_closure_configuration=closure.CONFIG,
    evidence=folder.relative_to(ROOT).as_posix(), original_attempts=157, original_successes=156,
    original_failed_outcomes_preserved=1, recovery_live_handle=launch, recovery_complete=False,
    full157_independent_cache_closure_complete=False, actual_pretraining_epochs=0, new_performance_results=0,
    previous_goal_turn_classification='progress_started_all157_real_preparation_and_published',
    current_goal_turn_classification='progress_all157_attempts_terminal_diagnosed_RSS_failure_started_isolated_full_recovery',
    all_original_experiments_complete=False))
targets = [recovery.CONFIG, closure.CONFIG, folder.relative_to(ROOT).as_posix(), ara.relative_to(ROOT).as_posix(),
    'scripts/flow_matching/recover_moment_icm_insect_chunked_export_v2.py',
    'scripts/flow_matching/verify_moment_icm_PILE_classification_v1.py',
    'tests/test_fm_moment_icm_chunked_export_v2.py', 'tests/test_fm_moment_icm_PILE_classification_closure_v1.py']
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    paths.update(subprocess.check_output(['git', *args, '--', *targets], cwd=ROOT, text=True).splitlines())
git = lambda cwd, *args: subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
assert paths and not git(ROOT, 'diff', '--cached', '--name-only')
write(ROOT / 'tmp/MOMENT_ICM_classification_memory_recovery_publication_manifest_20261015_v2.json', dict(
    previous_root=git(ROOT, 'rev-parse', 'HEAD'),
    previous_parent=git(ROOT.parent / 'flow-matching-research', 'rev-parse', 'HEAD'), paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths), source_successes=156, preserved_failure=1,
    recovered_cases_at_snapshot=progress['completed_cases'], recovery_total_cases=25000, tests=12, new_model_results=0)))
