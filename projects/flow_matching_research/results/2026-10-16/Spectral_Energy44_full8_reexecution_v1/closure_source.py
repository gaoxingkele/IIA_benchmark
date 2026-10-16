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
prior = ROOT / 'projects/flow_matching_research/results/2026-10-16/Spectral_Energy43_full7_actual_handoff_v2'
catalog_path = prior / 'all7_complete_generator_reexecution_catalog.json'
catalog = read(catalog_path)
assert len(catalog['records']) == 7
identity = 'spectral_original__diffusion_ts__energy__released_author_data__seed44'
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
assert not energy['complete'] and energy['verified_training_seeds'] == [42,43,44]
live_cohorts = read(Path(c['artifact_root']) / 'full14_cohort_comparisons.json')
assert live_cohorts == cohorts
with (prior / 'all7_generators_28_metric_comparisons.csv').open(encoding='utf-8-sig', newline='') as f:
    old_rows = list(csv.DictReader(f))
mapped = {(r['canonical_id'], r['metric']): {k: str(v) for k, v in r.items()} for r in rows}
assert len(old_rows) == 28
assert all(mapped[(r['canonical_id'], r['metric'])] == r for r in old_rows)
old_cohorts = read(prior / 'all14_five_seed_cohort_completion.json')
normal = lambda x: json.loads(json.dumps(x))
assert normal(cohorts['cohorts']['diffusion_ts/stocks/released_author_data']) == old_cohorts['cohorts']['diffusion_ts/stocks/released_author_data']

# This closure observes the existing live scheduler; it performs no handoff.
for key,previous in old_cohorts['cohorts'].items():
    if key!='diffusion_ts/energy/released_author_data':assert normal(cohorts['cohorts'][key])==previous
assert len(sids)==8 and len(rows)==32 and len(repeats)==160
assert not energy.get('metrics')
test_xml=Path('F:/aicoding/IIA_Data/experiments/Spectral_Energy44_full8_20261016_v1_tests.xml')
suites=list(ET.parse(test_xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==23
assert all(int(x.get('failures',0))==int(x.get('errors',0))==int(x.get('skipped',0))==0 for x in suites)
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/Spectral_Energy44_full8_reexecution_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
out=Path(c['artifact_root'])/identity;attempt=ROOT/c['state_root']/'attempts'/identity
for name,path in dict(energy44_report=out/'full_generation_and_evaluator_reexecution.json',
    energy44_guard=attempt/'guard_receipt.json',actual_live_scheduler_state=state_path).items():
    shutil.copyfile(path,packet/(name+'.json'))
shutil.copyfile(test_xml,packet/'relevant23_validation.xml')
write(packet/'all8_complete_generator_reexecution_catalog.json',dict(records=bindings,statistics=checks,
    prior7_catalog_path=catalog_path.relative_to(ROOT).as_posix(),prior7_catalog_sha256=sha(catalog_path),
    prior7_report_guard_statistics_and_all28_metric_rows_unchanged=True,full_reexecutions=8,required_conditions=70))
write(packet/'all14_five_seed_cohort_completion.json',cohorts)
for name,data in [('all8_generators_32_metric_comparisons.csv',rows),('all8_generators_160_actual_evaluator_values.csv',repeats)]:
    with (packet/name).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
write(packet/'full_case_raw_log_bindings.json',dict(files=[dict(path=str(attempt/name),sha256=sha(attempt/name)) for name in ['stdout.log','stderr.log']]))
write(packet/'iteration_status.json',dict(original_scope=c['original_scope'],full_generation_evaluator_reexecutions=8,
    required_conditions=70,complete_five_training_seed_cohorts=1,required_cohorts=14,Energy_verified_training_seeds=[42,43,44],
    new_full_reexecution_conditions=1,new_generator_training_updates=0,new_training_seed_slots=0,
    actual_live_scheduler=dict(pid=owner.pid,create_time=owner.create_time(),command=owner.cmdline()),
    scientific_workers_stopped=False,all_original_experiments_complete=False))
lines=['# Energy seed44：第8个完整生成与评估重跑条件','',
    '原始Diffusion-TS/Energy/released_author_data/seed44检查点完成两次完整Trainer.sample，四类原始评估器分别重跑5次，共20次。',
    f"完整生成形状{bindings[identity]['complete_generated_shape']}，完整参考形状{bindings[identity]['complete_reference_shape']}；保留作者采样器产生的全部额外行。两次生成数组每字节一致，完整随机状态恢复及消耗相同。冻结原始模型/EMA/优化器及检查点字节保持，生成器新增优化更新0。",'',
    '| 指标 | 原生成均值 | 本次新采样均值 | 新减原 |','|---|---:|---:|---:|']
for metric,value in checks[identity].items():
    lines.append(f"| {metric} | {value['original_source_mean']:.9f} | {value['mean']:.9f} | {value['difference']:.9f} |")
lines += ['',
    'Energy现完成42、43、44三个训练模型的完整重跑，3/5；45、46仍必需，不计算部分模型均值、训练种子区间或完整组排名。每个固定模型内部的五次评估器重复、评估器SD/SE/t95与五个独立训练模型分别标注。历史采样RNG未保存，新采样种子为1000044；本次两次新生成的逐位重现不证明历史生成字节一致。',
    '重新核验全部8个完整条件的源模型、完整生成/参考数组、原始完整随机状态文件、20评估器与实际守卫。原7报告/守卫哈希、全部28行指标、全部已有统计及Stocks完整五训练种子组保持；现为8/70条件、1/14完整五训练种子组。源训练产物34/70是另一层证据，不能与重跑数相加。',
    '相关23项检查本轮重新通过。既有70条件调度器仍运行，原采样批量、模型和数据尺寸、求解器、两个完整采样、评估次数、资源门槛和完整范围保持；本归档没有新训练、推理或调度交接，也没有停止数值进程。',
    '本结果只说明冻结模型在新采样随机状态下的完整执行与重放，不认证历史作者性能或严格TAB。全部45篇/681方法记录/273数据任务/702原始指标及原论文、TAB、新数据轨道仍保留，整体目标未完成。']
(packet/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
shutil.copyfile(__file__,packet/'closure_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/Spectral_Energy44_full8_reexecution.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),full_reexecutions=8,required_conditions=70,
    complete_five_training_seed_cohorts=1,required_cohorts=14,Energy_complete_seeds=[42,43,44],all_original_experiments_complete=False))
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD']).decode().strip()
manifest=ROOT/'tmp/Spectral_Energy44_full8_manifest_20261016_v1.json';assert not manifest.exists()
write(manifest,dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),
    paths=sorted([ara.relative_to(ROOT).as_posix(),'projects/flow_matching_research/README.rst']+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()])))
print(json.dumps(dict(packet=str(packet),full_reexecutions=8,complete_cohorts=1,metrics=checks[identity])))
