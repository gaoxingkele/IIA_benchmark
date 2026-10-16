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
from scripts.flow_matching import reexecute_spectral_full70_generation_evaluators_v1 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

c = read(ROOT / s.CONFIG)
parent, cases = s.verify(c)
prior = ROOT / 'projects/flow_matching_research/results/2026-10-16/Spectral_Stocks5_full_reexecution_cohort_v1'
catalog_path = prior / 'all5_complete_generator_reexecution_catalog.json'
catalog = read(catalog_path)
assert len(catalog['records']) == 5
identity = 'spectral_original__diffusion_ts__energy__released_author_data__seed42'
sids = sorted(set(catalog['records']) | {identity})
state_path = ROOT / c['state_root'] / 'status.json'
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
assert not energy['complete'] and energy['verified_training_seeds'] == [42]
assert read(Path(c['artifact_root']) / 'full14_cohort_comparisons.json') == cohorts
with (prior / 'all5_generators_20_metric_comparisons.csv').open(encoding='utf-8-sig', newline='') as f:
    old_rows = list(csv.DictReader(f))
mapped = {(r['canonical_id'], r['metric']): {k: str(v) for k, v in r.items()} for r in rows}
assert len(old_rows) == 20
assert all(mapped[(r['canonical_id'], r['metric'])] == r for r in old_rows)
old_cohorts = read(prior / 'all14_five_seed_cohort_completion.json')
normal = lambda x: json.loads(json.dumps(x))
assert normal(cohorts['cohorts']['diffusion_ts/stocks/released_author_data']) == old_cohorts['cohorts']['diffusion_ts/stocks/released_author_data']
packet = ROOT / 'projects/flow_matching_research/results/2026-10-16/Spectral_Energy42_full_reexecution6_v1'
packet.mkdir(exist_ok=False)
(packet / '.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n', encoding='utf-8')
out = Path(c['artifact_root']) / identity
attempt = ROOT / c['state_root'] / 'attempts' / identity
for name, path in dict(energy42_report=out/'full_generation_and_evaluator_reexecution.json',
        energy42_guard=attempt/'guard_receipt.json', actual_source_queue_state=state_path).items():
    shutil.copyfile(path, packet / (name+'.json'))
write(packet / 'all6_complete_generator_reexecution_catalog.json', dict(records=bindings, statistics=checks,
    prior5_catalog_path=catalog_path.relative_to(ROOT).as_posix(), prior5_catalog_sha256=sha(catalog_path),
    prior5_report_and_guard_hashes_and_all20_metric_rows_unchanged=True,
    full_reexecutions=6, required_conditions=70))
write(packet / 'all14_five_seed_cohort_completion.json', cohorts)
for name, data in [('all6_generators_24_metric_comparisons.csv', rows),
                   ('all6_generators_120_actual_evaluator_values.csv', repeats)]:
    with (packet / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
write(packet / 'actual_numeric_continuation_handle.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    owner=dict(pid=owner.pid, create_time=owner.create_time(), command=owner.cmdline()),
    numerical_descendants=[dict(pid=p.pid, create_time=p.create_time(), command=p.cmdline())
        for p in owner.children(recursive=True) if p.name().lower() == 'python.exe'],
    no_numeric_process_stopped=True))
write(packet / 'raw_log_bindings.json', dict(files=[dict(path=str(attempt/name), sha256=sha(attempt/name))
    for name in ['stdout.log', 'stderr.log']]))
write(packet / 'iteration_status.json', dict(original_scope=c['original_scope'],
    full_generation_evaluator_reexecutions=6, required_conditions=70,
    complete_five_training_seed_cohorts=1, required_cohorts=14,
    Energy_verified_training_seeds=[42], new_full_reexecution_conditions=1,
    new_generator_training_updates=0, new_training_seed_slots=0,
    prior5_unchanged=True, original_numerical_workers_stopped=False,
    all_original_experiments_complete=False))
lines = ['# Energy seed42：完整生成与评估重跑', '',
    'Diffusion-TS/Energy/released_author_data/seed42完成两次完整原始采样，以及四类原评估器各5次实际运行。',
    f"生成数组{bindings[identity]['complete_generated_shape']}，完整参考{bindings[identity]['complete_reference_shape']}；两次采样完整数组字节与结束随机状态一致。原检查点、模型/EMA/优化器、冻结数据、求解器和全部评估预算保持，生成器新增训练更新为0。", '',
    '| 指标 | 原生成均值 | 新采样均值 | 新减原 |', '|---|---:|---:|---:|']
for metric, item in checks[identity].items():
    lines.append(f"| {metric} | {item['original_source_mean']:.9f} | {item['mean']:.9f} | {item['difference']:.9f} |")
lines += ['',
    'Energy只有一个已训练模型的重跑结果（1/5），不能把5次评估器拟合当成5个训练种子，不报告Energy训练种子均值/SE/CI或五种子配对结论。新采样种子1000042，历史采样RNG未恢复，不能声称历史生成字节重现。',
    '此前Stocks五条完整报告/守卫哈希、20行指标以及完整五训练种子汇总逐项保持。当前生成与评估完整重跑6/70、完整五训练种子组1/14。原始源产物完成数与本重跑数不相加。',
    '精确原进程身份核验后，队列已自行推进Energy seed43。公平继任观察器尚未接管，不声称已经完成GPU队列公平切换。没有停止数值进程或改动预算、精度、线程、采样数量及资源门槛。',
    '本轮没有实现更改；原全条件守卫、全部生成/参考/RNG文件和源产物校验、四评估器20个原始值及统计重算均通过。',
    '全部45篇/681方法记录/273数据任务/702既有指标及原论文、TAB、新数据未完成范围保留，整体目标未完成。']
(packet / 'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
shutil.copyfile(__file__, packet/'closure_source.py')
write(packet / 'packet_integrity.json', dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p))
    for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/Spectral_Energy42_full_reexecution6.2026-10-16_v1.json'
assert not ara.exists()
write(ara, dict(result_packet=packet.relative_to(ROOT).as_posix(), full_generation_evaluator_reexecutions=6,
    required_conditions=70, complete_five_training_seed_cohorts=1, required_cohorts=14,
    all_original_experiments_complete=False))
head=lambda p: subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD']).decode().strip()
paths=[ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
write(ROOT/'tmp/Spectral_Energy42_full6_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),
    previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet), files=len(paths), metrics=checks[identity],
    full_reexecutions=6, complete_cohorts=1)))
