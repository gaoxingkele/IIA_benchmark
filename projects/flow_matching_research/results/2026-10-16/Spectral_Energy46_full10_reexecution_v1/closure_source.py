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
prior = ROOT / 'projects/flow_matching_research/results/2026-10-16/Spectral_Energy45_full9_reexecution_v1'
catalog_path = prior / 'all9_complete_generator_reexecution_catalog.json'
catalog = read(catalog_path)
assert len(catalog['records']) == 9
identity = 'spectral_original__diffusion_ts__energy__released_author_data__seed46'
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
assert cohorts['complete_cohorts'] == 2
assert cohorts['cohorts']['diffusion_ts/stocks/released_author_data']['complete']
energy = cohorts['cohorts']['diffusion_ts/energy/released_author_data']
assert energy['complete'] and energy['verified_training_seeds'] == [42,43,44,45,46]
live_cohorts = read(Path(c['artifact_root']) / 'full14_cohort_comparisons.json')
assert live_cohorts == cohorts
with (prior / 'all9_generators_36_metric_comparisons.csv').open(encoding='utf-8-sig', newline='') as f:
    old_rows = list(csv.DictReader(f))
mapped = {(r['canonical_id'], r['metric']): {k: str(v) for k, v in r.items()} for r in rows}
assert len(old_rows) == 36
assert all(mapped[(r['canonical_id'], r['metric'])] == r for r in old_rows)
old_cohorts = read(prior / 'all14_five_seed_cohort_completion.json')
normal = lambda x: json.loads(json.dumps(x))
assert normal(cohorts['cohorts']['diffusion_ts/stocks/released_author_data']) == old_cohorts['cohorts']['diffusion_ts/stocks/released_author_data']

# This closure observes the existing live scheduler; it performs no handoff.
for key,previous in old_cohorts['cohorts'].items():
    if key!='diffusion_ts/energy/released_author_data':assert normal(cohorts['cohorts'][key])==previous
assert len(sids)==10 and len(rows)==40 and len(repeats)==200
assert energy['paired_training_seed_metric_differences']
test_xml=ROOT/'tmp/Spectral_Energy46_full10_validation_20261016_v1.xml'
suites=list(ET.parse(test_xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==23
assert all(int(x.get('failures',0))==int(x.get('errors',0))==int(x.get('skipped',0))==0 for x in suites)
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/Spectral_Energy46_full10_reexecution_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
out=Path(c['artifact_root'])/identity;attempt=ROOT/c['state_root']/'attempts'/identity
for name,path in dict(energy46_report=out/'full_generation_and_evaluator_reexecution.json',
    energy46_guard=attempt/'guard_receipt.json',actual_live_scheduler_state=state_path).items():shutil.copyfile(path,packet/(name+'.json'))
shutil.copyfile(test_xml,packet/'relevant23_validation.xml')
write(packet/'all10_complete_generator_reexecution_catalog.json',dict(records=bindings,statistics=checks,
    prior9_catalog_path=catalog_path.relative_to(ROOT).as_posix(),prior9_catalog_sha256=sha(catalog_path),
    prior9_report_guard_statistics_and_all36_metric_rows_unchanged=True,full_reexecutions=10,required_conditions=70))
write(packet/'all14_five_seed_cohort_completion.json',cohorts)
for name,data in [('all10_generators_40_metric_comparisons.csv',rows),('all10_generators_200_actual_evaluator_values.csv',repeats)]:
    with (packet/name).open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
full_stats=[]
for metric in ['context_fid','correlational','discriminative','predictive']:
    ids=[f'spectral_original__diffusion_ts__energy__released_author_data__seed{seed}' for seed in [42,43,44,45,46]]
    new=[checks[sid][metric]['mean'] for sid in ids];old=[checks[sid][metric]['original_source_mean'] for sid in ids]
    delta=[a-b for a,b in zip(new,old)];item=dict(metric=metric,independent_trained_models=5,evaluator_repeats_per_fixed_model=5)
    for prefix,values in [('new_generation',new),('original_generation',old),('paired_new_minus_original',delta)]:
        mean=statistics.mean(values);sd=statistics.stdev(values);se=sd/math.sqrt(5);half=2.7764451051977987*se
        item.update({prefix+'_mean':mean,prefix+'_sample_SD':sd,prefix+'_SE':se,prefix+'_t95_low':mean-half,prefix+'_t95_high':mean+half})
    target=energy['paired_training_seed_metric_differences'][metric]
    assert math.isclose(item['paired_new_minus_original_mean'],target['mean'],rel_tol=1e-13,abs_tol=1e-13)
    assert math.isclose(item['paired_new_minus_original_sample_SD'],target['sample_SD'],rel_tol=1e-13,abs_tol=1e-13)
    assert all(math.isclose(item[key],value,rel_tol=1e-13,abs_tol=1e-13) for key,value in zip(
        ['paired_new_minus_original_t95_low','paired_new_minus_original_t95_high'],target['t95_interval']))
    full_stats.append(item)
with (packet/'Energy_all5_complete_training_seed_statistics.csv').open('x',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(full_stats[0]));w.writeheader();w.writerows(full_stats)
write(packet/'full_case_raw_log_bindings.json',dict(files=[dict(path=str(attempt/name),sha256=sha(attempt/name)) for name in ['stdout.log','stderr.log']]))
write(packet/'iteration_status.json',dict(original_scope=c['original_scope'],full_generation_evaluator_reexecutions=10,
    required_conditions=70,complete_five_training_seed_cohorts=2,required_cohorts=14,Energy_verified_training_seeds=[42,43,44,45,46],
    new_full_reexecution_conditions=1,new_generator_training_updates=0,new_training_seed_slots=0,
    actual_live_scheduler=dict(pid=owner.pid,create_time=owner.create_time(),command=owner.cmdline()),
    scientific_workers_stopped=False,all_original_experiments_complete=False))
lines=['# Energy seed46：第10个完整生成重执行条件、第二个完整五模型组','',
    '原Diffusion-TS/Energy/released_author_data/seed46检查点完成两次完整原Trainer.sample；四类原评估器各完整重跑五次，共20次。',
    f"完整生成形状{bindings[identity]['complete_generated_shape']}，完整参考形状{bindings[identity]['complete_reference_shape']}。保留原采样器生成的额外行；两次新生成数组每字节一致，完整随机状态恢复和消耗相同，模型/EMA/优化器/检查点保持，生成器新训练更新为0。",'',
    '| 指标 | 本次固定模型评估器五次均值 | 完整五训练模型均值 | 五训练模型SD |',
    '|---|---:|---:|---:|']
for row in full_stats:lines.append(f"| {row['metric']} | {checks[identity][row['metric']]['mean']:.9f} | {row['new_generation_mean']:.9f} | {row['new_generation_sample_SD']:.9f} |")
lines+=['',
    'Energy的42..46五个训练模型现已全部完成重执行，可计算完整五模型均值、样本标准差、SE及不截断df4 Student-t区间，详见Energy_all5_complete_training_seed_statistics.csv。每个模型先汇总内部五次评估器重复，再以五个模型均值计算训练种子统计；不能把25次评估器输出当作25个独立训练模型。原生成与新生成按相同训练种子成对比较。',
    '成对差异同时包含新采样随机性与重新训练评估器的随机性，不能只归因生成采样；df4区间不是历史性能等价检验或未见数据泛化界。历史采样RNG未保存，新采样种子为1000046；两次新生成重放一致不证明历史生成数组字节等价。',
    '此前9项完整报告/守卫/统计与36行指标逐项保持，Stocks完整五模型组和其他未完成组保持。现10/70条件、2/14完整五模型组，40行指标、200个真实评估器值。相关23项测试通过，未修改数值任务、原模型/数据/批量/求解器或资源门槛。本归档没有新增训练、推理或调度交接。',
    '全部45篇/681方法记录/273数据任务/702论文指标以及原论文/TAB/新数据范围保持；完整研究任务尚未完成，不认证历史作者性能或严格TAB成绩。']
(packet/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
shutil.copyfile(__file__,packet/'closure_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.iterdir()) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/Spectral_Energy46_full10_reexecution.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(packet=packet.relative_to(ROOT).as_posix(),full_reexecutions=10,required_conditions=70,
    complete_five_training_seed_cohorts=2,required_cohorts=14,Energy_complete_seeds=[42,43,44,45,46],all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8')
text+='\n`Spectral Energy46完成，累计10/70、完整五模型组2/14 <results/2026-10-16/'+packet.name+'/README.md>`_\n'
text+='两次完整生成每字节一致，四个原评估器各五次；Energy42..46现全5/5，完整五训练模型均值/SD/SE/t95及与原生成的成对差异独立核验，评估器内部重复单列。此前9项证明与36行指标保持，Stocks完整组保持；现40行/200真实评估器值。23项测试通过，生成器新训练更新0；历史采样RNG/作者性能等价未认证，整体未完成。\n'
page.write_text(text,encoding='utf-8',newline='\n')
head=lambda d:subprocess.check_output(['git','-C',str(d),'rev-parse','HEAD'],text=True).strip()
paths=[ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in packet.iterdir() if p.is_file()]
write(ROOT/'tmp/Spectral_Energy46_full10_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=paths))
print(dict(packet=str(packet),complete_conditions=10,complete_five_model_cohorts=2,Energy_full_stats=full_stats))
