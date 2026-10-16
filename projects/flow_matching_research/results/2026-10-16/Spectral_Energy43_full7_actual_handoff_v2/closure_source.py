from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import math
import shutil
import statistics
import subprocess
import numpy as np
import psutil
import xml.etree.ElementTree as ET
from scripts.flow_matching import continue_spectral70_queued_handoff_v2 as handoff_module
from scripts.flow_matching import reexecute_spectral_full70_generation_evaluators_v1 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

c = read(ROOT / s.CONFIG)
parent, cases = s.verify(c)
prior = ROOT / 'projects/flow_matching_research/results/2026-10-16/Spectral_Energy42_full_reexecution6_v1'
catalog_path = prior / 'all6_complete_generator_reexecution_catalog.json'
catalog = read(catalog_path)
assert len(catalog['records']) == 6
identity = 'spectral_original__diffusion_ts__energy__released_author_data__seed43'
sids = sorted(set(catalog['records']) | {identity})
hc = read(ROOT / handoff_module.CONFIG)
assert handoff_module.verify(hc)[0] == c
handoff_base = ROOT / hc['state_root']
state_path = handoff_base / 'status.json'
state = read(state_path)
assert set(sids).issubset(state['verified_ids']) and not state['failed_preserved']
owner = psutil.Process(state['pid'])
assert owner.create_time() == state['create_time'] and owner.is_running()
reports, bindings, rows, repeats, checks = {}, {}, [], [], {}
for sid in sids:
    job = cases[sid]['job']
    out = Path(c['artifact_root']) / sid
    path = out / 'full_generation_and_evaluator_reexecution.json'
    guard_path = ROOT / c['state_root'] / 'attempts' / sid / 'guard_receipt.json'
    report, guard = read(path), read(guard_path)
    assert guard['passed'] and guard['exit_code'] == 0 and not guard['resource_guard_aborted']
    assert guard['report_sha256'] == sha(path)
    assert report['passed'] and report['canonical_id'] == sid
    assert report['separate_sampling_seed'] == job['seed'] + 1000000
    assert report['full_original_sampling_calls'] == 2 and report['original_evaluation_repeats'] == 5
    assert report['actual_new_generator_training_updates'] == 0
    assert not report['generation_field_chunking'] and not report['historical_sampling_RNG_recovered']
    assert not report['historical_original_generated_bytes_equivalence_claim']
    for key in ['original_checkpoint_unchanged', 'original_source_and_frozen_dataset_verified',
                'every_model_EMA_optimizer_state_unchanged', 'all_generated_payload_bytes_replayed',
                'complete_RNG_restoration_and_equal_consumption', 'original_sampler_grid_and_solver_unchanged',
                'all_four_original_evaluators_reexecuted', 'all_learned_evaluators_independently_retrained']:
        assert report[key], (sid, key)
    proof, proof_path = s.ready_proof(parent, cases[sid], sid)
    source_path = ROOT / job['output_directory'] / 'result.json'
    source = read(source_path)
    assert report['original_result_sha256'] == sha(source_path)
    assert report['original_artifact_proof_sha256'] == sha(proof_path)
    checkpoint = Path(next(x['path'] for x in source['artifacts'] if x['role'] == 'checkpoint'))
    assert sha(checkpoint) == report['original_checkpoint_sha256']
    for artifact in report['artifacts']:
        assert sha(artifact['path']) == artifact['sha256']
    first = np.load(out / 'first_full_generated.npy', mmap_mode='r', allow_pickle=False)
    second = np.load(out / 'second_full_generated.npy', mmap_mode='r', allow_pickle=False)
    reference = np.load(out / 'reference_data.npy', mmap_mode='r', allow_pickle=False)
    assert list(first.shape) == report['full_generation_shape'] == proof['complete_generated_shape']
    assert list(reference.shape) == proof['complete_reference_shape']
    assert first.dtype == second.dtype and np.array_equal(first, second)
    assert np.isfinite(first).all() and np.isfinite(second).all() and np.isfinite(reference).all()
    assert sha(out / 'first_full_generated.npy') == sha(out / 'second_full_generated.npy')
    assert sha(out / 'reference_data.npy') == job['reference_sha256']
    if sid in catalog['records']:
        entry = catalog['records'][sid]
        assert sha(path) == entry['report_sha256'] and sha(guard_path) == entry['guard_sha256']
        bindings[sid] = entry
    else:
        bindings[sid] = dict(report_path=str(path), report_sha256=sha(path),
            guard_path=str(guard_path), guard_sha256=sha(guard_path),
            original_source_result_sha256=sha(source_path), original_artifact_proof_sha256=sha(proof_path),
            complete_generated_shape=list(first.shape), complete_reference_shape=list(reference.shape),
            all_new_generated_file_bytes_equal=True)
    assert set(report['metrics']) == {'context_fid', 'correlational', 'discriminative', 'predictive'}
    values = {}
    for metric, item in report['metrics'].items():
        raw = item['raw_repeats']
        assert len(raw) == 5 and all(math.isfinite(x) and x >= 0 for x in raw)
        mean, sd = statistics.mean(raw), statistics.stdev(raw)
        se = sd / math.sqrt(5)
        half = 2.7764451051977987 * se
        stats = dict(mean=mean, std=sd, se=se, ci95_low=mean-half,
                     ci95_high=mean+half, author_display_halfwidth=half)
        for key, value in stats.items():
            assert math.isclose(item[key], value, rel_tol=1e-12, abs_tol=1e-12)
        assert item['independent_training_models'] == 1 and item['between_training_seed_CI95'] is None
        difference = report['original_vs_new_generation'][metric]
        assert difference['original_mean'] == proof['metrics'][metric]['mean']
        assert difference['new_generation_mean'] == mean
        assert difference['difference'] == mean - difference['original_mean']
        values[metric] = dict(raw_repeats=raw, **stats,
            original_source_mean=difference['original_mean'], difference=difference['difference'])
        rows.append(dict(canonical_id=sid, training_seed=job['seed'], sampling_seed=report['separate_sampling_seed'],
            metric=metric, source_original_mean=difference['original_mean'], new_generation_mean=mean,
            new_minus_original=difference['difference'], within_model_evaluator_SD=sd,
            evaluator_repeats=5, independent_trained_generators=1))
        for index, value in enumerate(raw):
            repeats.append(dict(canonical_id=sid, training_seed=job['seed'], metric=metric,
                                evaluator_repeat=index, value=value))
    if sid in catalog['statistics']:
        assert values == catalog['statistics'][sid]
    checks[sid], reports[sid] = values, report
assert {k: bindings[k] for k in catalog['records']} == catalog['records']
cohorts = s.full_cohort_comparisons(cases, reports)
assert cohorts['complete_cohorts'] == 1
assert cohorts['cohorts']['diffusion_ts/stocks/released_author_data']['complete']
energy = cohorts['cohorts']['diffusion_ts/energy/released_author_data']
assert not energy['complete'] and energy['verified_training_seeds'] == [42,43]
live_cohorts = read(Path(c['artifact_root']) / 'full14_cohort_comparisons.json')
assert live_cohorts == cohorts
with (prior / 'all6_generators_24_metric_comparisons.csv').open(encoding='utf-8-sig', newline='') as f:
    old_rows = list(csv.DictReader(f))
mapped = {(r['canonical_id'], r['metric']): {k: str(v) for k, v in r.items()} for r in rows}
assert len(old_rows) == 24
assert all(mapped[(r['canonical_id'], r['metric'])] == r for r in old_rows)
old_cohorts = read(prior / 'all14_five_seed_cohort_completion.json')
normal = lambda x: json.loads(json.dumps(x))
assert normal(cohorts['cohorts']['diffusion_ts/stocks/released_author_data']) == old_cohorts['cohorts']['diffusion_ts/stocks/released_author_data']

# Verify the completed handoff, without acquiring a live lock or changing actors.
h = read(handoff_base / 'handoff.json')
hi = read(handoff_base / 'handoff_identity.json')
restored = read(handoff_base / 'restored_verified_report_bindings.json')
old_path = handoff_base / 'previous_status_original_bytes.json'
old = read(old_path)
assert h['previous_owner'] == hc['previous_owner'] == hi['previous_owner']
assert h['previous_observer'] == hc['previous_observer'] == hi['previous_observer']
assert h['successor_pid'] == state['pid'] == owner.pid
assert h['only_two_proven_idle_metadata_parents_retired'] and not h['numerical_child_stopped']
assert h['verified_conditions_preserved'] == len(sids) == 7 and h['failures_preserved'] == 0
for field in ['actual_original_GPU_byte_lock_owned', 'exact_identities_checked_twice',
              'terminal_child_guard_verified', 'all_attempts_have_preserved_terminal_outcomes']:
    assert hi[field]
assert not hi['numerical_child_stopped'] and not hi['numerical_configuration_changed']
for children_key in ['owner_descendants', 'observer_descendants']:
    assert all(handoff_module.previous.fair.allowed_console_child(p['name'], p['exe']) for p in hi[children_key])
# The inherited controller intentionally mirrors its new state to the original
# mutable status path. Compare the retained historical snapshot semantically,
# then verify the current mirrored path belongs to the live successor.
assert old['pid'] == hc['previous_owner']['pid'] and old['create_time'] == hc['previous_owner']['create_time']
mirrored = read(ROOT / c['state_root'] / 'status.json')
assert mirrored['pid'] == owner.pid and mirrored['create_time'] == owner.create_time()
assert mirrored['scheduler_configuration'] == handoff_module.CONFIG
assert set(sids).issubset(mirrored['verified_ids']) and not mirrored['failed_preserved']
observer_base = ROOT / read(ROOT / hc['previous_configuration'])['state_root']
assert (handoff_base / 'previous_observer_status_original_bytes.json').read_bytes() == (observer_base / 'status.json').read_bytes()
assert set(old['verified_ids']) == set(restored['records']) == set(sids)
assert old['failed_preserved'] == restored['failed_preserved'] == state['failed_preserved'] == {}
assert restored['new_numerical_completions'] == 0
for sid, entry in restored['records'].items():
    assert sha(entry['report_path']) == entry['report_sha256'] == bindings[sid]['report_sha256']
    assert sha(entry['guard_path']) == entry['guard_sha256'] == bindings[sid]['guard_sha256']
retired = []
for expected in [hc['previous_owner'], hc['previous_observer']]:
    try:
        proc = psutil.Process(expected['pid'])
        assert proc.create_time() != expected['create_time']
    except psutil.NoSuchProcess:
        pass
    retired.append(expected)
last_guard = read(ROOT / c['state_root'] / 'attempts' / identity / 'guard_receipt.json')
assert last_guard['exact_child_handle'] == old['exact_child_handle']
test_xml = Path('F:/aicoding/IIA_Data/experiments/Spectral_Energy43_full7_handoff_20261016_v1_tests.xml')
suites = list(ET.parse(test_xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests', 0)) for x in suites) == 23
assert all(int(x.get('failures', 0)) == int(x.get('errors', 0)) == int(x.get('skipped', 0)) == 0 for x in suites)
packet = ROOT / 'projects/flow_matching_research/results/2026-10-16/Spectral_Energy43_full7_actual_handoff_v2'
packet.mkdir(exist_ok=False)
(packet / '.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n', encoding='utf-8')
out = Path(c['artifact_root']) / identity
attempt = ROOT / c['state_root'] / 'attempts' / identity
for name, path in dict(energy43_report=out/'full_generation_and_evaluator_reexecution.json',
        energy43_guard=attempt/'guard_receipt.json', actual_successor_state=state_path,
        handoff=handoff_base/'handoff.json', handoff_identity=handoff_base/'handoff_identity.json',
        restored_verified_report_bindings=handoff_base/'restored_verified_report_bindings.json',
        previous_status_original_bytes=old_path,
        previous_observer_status_original_bytes=handoff_base/'previous_observer_status_original_bytes.json').items():
    shutil.copyfile(path, packet / (name + '.json'))
shutil.copyfile(test_xml, packet / 'relevant_validation.xml')
write(packet / 'all7_complete_generator_reexecution_catalog.json', dict(records=bindings, statistics=checks,
    prior6_catalog_path=catalog_path.relative_to(ROOT).as_posix(), prior6_catalog_sha256=sha(catalog_path),
    prior6_report_and_guard_hashes_and_all24_metric_rows_unchanged=True, full_reexecutions=7, required_conditions=70))
write(packet / 'all14_five_seed_cohort_completion.json', cohorts)
for name, data in [('all7_generators_28_metric_comparisons.csv', rows), ('all7_generators_140_actual_evaluator_values.csv', repeats)]:
    with (packet / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
assert (packet / 'previous_status_original_bytes.json').read_bytes() == old_path.read_bytes()
write(packet / 'closure_preflight_correction.json', dict(previous_unpublished_closure='tmp/close_Spectral_Energy43_full7_handoff_20261016_v1.py',
    failed_assertion='Historical state snapshot equals current mutable original status path',
    observed_reason='Verified continuation write callback intentionally updates both current status paths; historical snapshot is retained separately',
    numerical_execution_or_failure_created=False, previous_v1_source_preserved=True))
write(packet / 'actual_handoff_independent_verification.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    historical_original_state_snapshot_copied_byte_identically=True, historical_snapshot_sha256=sha(old_path),
    current_mirrored_state_owned_by_live_successor=True, all7_full_reports_artifacts_and_guards_reverified=True,
    actual_GPU_lock_ownership_at_boundary=hi['actual_original_GPU_byte_lock_owned'],
    retired_exact_idle_metadata_identities=retired, no_numerical_process_stopped=True,
    successor=dict(pid=owner.pid, create_time=owner.create_time(), command=owner.cmdline()),
    successor_current_state=state['state'], new_generator_training_updates=0,
    original_numerical_source_command_budget_resources_unchanged=True,
    global_FIFO_or_future_GRASP_admission_proven=False, all_original_experiments_complete=False))
write(packet / 'raw_log_bindings.json', dict(files=[dict(path=str(attempt/name), sha256=sha(attempt/name)) for name in ['stdout.log', 'stderr.log']]))
write(packet / 'iteration_status.json', dict(original_scope=c['original_scope'], full_generation_evaluator_reexecutions=7,
    required_conditions=70, complete_five_training_seed_cohorts=1, required_cohorts=14,
    Energy_verified_training_seeds=[42,43], new_full_reexecution_conditions=1,
    actual_metadata_handoff_completed=True, new_generator_training_updates=0, new_training_seed_slots=0,
    original_numerical_workers_stopped=False, all_original_experiments_complete=False))
lines = ['# Energy seed43 全量生成评估完成，调度交接已核验', '',
    'Diffusion-TS/Energy/released_author_data/seed43完成两次完整原始Trainer.sample，以及四类原始评估器各5次运行。',
    f"生成与完整参考形状均为{bindings[identity]['complete_generated_shape']}；两次完整采样数组字节和结束随机状态一致。原始训练模型/EMA/优化器状态、原冻结数据、求解器及评估预算均保持，生成器新增训练更新为0。", '',
    '| 指标 | 原生成均值 | 本次完整采样均值 | 新减原 |', '|---|---:|---:|---:|---:|']
for metric, item in checks[identity].items():
    lines.append(f"| {metric} | {item['original_source_mean']:.9f} | {item['mean']:.9f} | {item['difference']:.9f} |")
lines += ['',
    'Energy现完成42、43两个训练种子（2/5），五次评估重复不能算作五个训练模型；不报告该不完整组的训练种子均值、区间或配对显著性。历史采样RNG未保存，本次单独采样种子1000043，不声称历史生成字节等价。',
    '全部7个条件的完整数组、RNG、原始源产物和守卫重新核验；前6条报告/守卫哈希、24行指标及Stocks完整五训练种子汇总保持不变。现为7/70条件、1/14完整五训练种子组。原源产物完成数不与重跑数相加。',
    '18:38:44的交接记录证实：继任器实际取得同一GPU文件第0字节锁；两次核对旧父进程和旧观察器的PID、创建时间与完整命令，确认仅有合法控制台子进程，最后数值子任务守卫已终结。只退出两个空闲元数据进程，没有停止数值子进程。7条原成功记录与0条失败原样恢复，历史状态快照原字节保存，所有报告哈希再次核验。继任器按原续跑逻辑更新两个当前状态路径，不把当前可变状态误当历史快照。',
    '继任器继续保留全部70条件与14组，使用原数值模块、命令、预算与资源门槛，在条件之间释放共享锁并让出35秒。本证据证明安全交接已发生；不保证系统全局FIFO，也不把其他家族尚未获得资源写成已开跑。',
    '相关23项测试通过，实际全数组/统计/交接检查通过；这些验证不抵扣实验完成。45篇/681方法记录/273数据任务/702原始指标及原论文、TAB、新数据轨道的全部未完成范围保留，整体目标未完成。']
(packet / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
shutil.copyfile(__file__, packet / 'closure_source.py')
write(packet / 'packet_integrity.json', dict(files=[dict(path=p.relative_to(packet).as_posix(), sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara = ROOT / 'projects/flow_matching_research/ara/execution/Spectral_Energy43_full7_actual_handoff.2026-10-16_v2.json'
assert not ara.exists()
write(ara, dict(result_packet=packet.relative_to(ROOT).as_posix(), full_generation_evaluator_reexecutions=7,
    required_conditions=70, complete_five_training_seed_cohorts=1, required_cohorts=14,
    metadata_handoff_completed=True, numerical_process_stopped=False, all_original_experiments_complete=False))
head = lambda p: subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD']).decode().strip()
write(ROOT/'tmp/Spectral_Energy43_full7_actual_handoff_manifest_20261016_v2.json', dict(previous_root=head(ROOT),
    previous_parent=head(ROOT.parent/'flow-matching-research'), paths=sorted([ara.relative_to(ROOT).as_posix()]
    + [p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()])))
print(json.dumps(dict(packet=str(packet), full_reexecutions=7, complete_cohorts=1,
    actual_metadata_handoff_completed=True, metrics=checks[identity], all_original_experiments_complete=False)))
