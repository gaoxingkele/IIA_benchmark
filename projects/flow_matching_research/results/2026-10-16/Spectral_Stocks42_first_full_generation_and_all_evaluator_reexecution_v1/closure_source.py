from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import math
import shutil
import statistics
import subprocess
import xml.etree.ElementTree as ET
import psutil
import numpy as np
from scripts.flow_matching import reexecute_spectral_full70_generation_evaluators_v1 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/s.CONFIG);parent,cases=s.verify(c)
identity='spectral_original__diffusion_ts__stocks__released_author_data__seed42'
state_path=ROOT/c['state_root']/'status.json';state=read(state_path)
controller=psutil.Process(state['pid']);assert controller.is_running() and controller.create_time()==state['create_time']
if identity not in state['verified_ids']:
    if identity in state['failed_preserved']:raise ValueError('Actual numerical failure preserved:'+state['failed_preserved'][identity])
    child=psutil.Process(state['active_pid']);handle=state['exact_child_handle']
    assert child.is_running() and child.create_time()==handle['create_time'] and child.cmdline()==handle['command']
    print(json.dumps(dict(pending=True,verified_live_pid=child.pid,create_time=child.create_time(),
        actual_progress=read(Path(c['artifact_root'])/identity/'progress.json'))));raise SystemExit(0)
assert not state['failed_preserved']
case=cases[identity];job=case['job'];out=Path(c['artifact_root'])/identity
report_path=out/'full_generation_and_evaluator_reexecution.json';r=read(report_path)
guard_path=ROOT/c['state_root']/'attempts'/identity/'guard_receipt.json';guard=read(guard_path)
assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['report_sha256']==sha(report_path)
assert r['passed'] and r['canonical_id']==identity and r['seed']==42 and r['separate_sampling_seed']==1000042
for key in ['original_checkpoint_unchanged','original_source_and_frozen_dataset_verified',
    'every_model_EMA_optimizer_state_unchanged','all_generated_payload_bytes_replayed',
    'complete_RNG_restoration_and_equal_consumption','original_sampler_grid_and_solver_unchanged',
    'all_four_original_evaluators_reexecuted','all_learned_evaluators_independently_retrained']:assert r[key]
assert r['actual_new_generator_training_updates']==0 and r['full_original_sampling_calls']==2
assert r['original_evaluation_repeats']==5 and not r['generation_field_chunking']
assert not r['historical_sampling_RNG_recovered'] and not r['historical_original_generated_bytes_equivalence_claim']
proof,proof_path=s.ready_proof(parent,case,identity)
assert sha(proof_path)==r['original_artifact_proof_sha256']
source_path=ROOT/job['output_directory']/'result.json';source=read(source_path)
assert sha(source_path)==r['original_result_sha256']==proof['original_result_sha256']
checkpoint=Path(next(x['path'] for x in source['artifacts'] if x['role']=='checkpoint'))
assert sha(checkpoint)==r['original_checkpoint_sha256']
for item in r['artifacts']:assert sha(item['path'])==item['sha256']
first=np.load(out/'first_full_generated.npy',mmap_mode='r',allow_pickle=False)
second=np.load(out/'second_full_generated.npy',mmap_mode='r',allow_pickle=False)
reference=np.load(out/'reference_data.npy',mmap_mode='r',allow_pickle=False)
assert list(first.shape)==r['full_generation_shape']==proof['complete_generated_shape']
assert first.shape==second.shape and first.dtype==second.dtype and np.array_equal(first,second)
assert np.isfinite(first).all() and np.isfinite(second).all() and np.isfinite(reference).all()
assert list(reference.shape)==proof['complete_reference_shape']
assert sha(out/'reference_data.npy')==job['reference_sha256']
assert sha(out/'first_full_generated.npy')==sha(out/'second_full_generated.npy')
checks={};rows=[]
for key,values in r['metrics'].items():
    raw=values['raw_repeats'];assert len(raw)==5 and all(math.isfinite(v) and v>=0 for v in raw)
    mean=statistics.mean(raw);sd=statistics.stdev(raw);se=sd/math.sqrt(5);half=2.7764451051977987*se
    computed=dict(mean=mean,std=sd,se=se,ci95_low=mean-half,ci95_high=mean+half,author_display_halfwidth=half)
    for k,v in computed.items():assert math.isclose(v,values[k],rel_tol=1e-12,abs_tol=1e-12)
    assert values['independent_training_models']==1 and values['between_training_seed_CI95'] is None
    delta=r['original_vs_new_generation'][key]
    assert delta['original_mean']==proof['metrics'][key]['mean'] and delta['new_generation_mean']==mean
    assert delta['difference']==mean-delta['original_mean']
    checks[key]=dict(raw_repeats=raw,**computed,original_source_mean=delta['original_mean'],difference=delta['difference'])
    for i,value in enumerate(raw):rows.append(dict(canonical_id=identity,training_seed=42,sampling_seed=1000042,
        metric=key,evaluator_repeat=i,evaluator_value=value,new_mean=mean,original_source_mean=delta['original_mean'],
        new_minus_original_mean=delta['difference'],independent_training_models=1))
assert set(checks)=={'context_fid','correlational','discriminative','predictive'}
cohorts=s.full_cohort_comparisons(cases,{identity:r})
assert cohorts['complete_cohorts']==0 and len(cohorts['cohorts'])==14
tests=Path('F:/aicoding/IIA_Data/experiments/Spectral_full70_generation_evaluators_and_source_regression_20261016_v1_tests.xml')
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==48
assert not any(int(x.get(k,0)) for x in suites for k in ['failures','errors','skipped'])
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/Spectral_Stocks42_first_full_generation_and_all_evaluator_reexecution_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.xml -text -whitespace\n*.py -text\n*.csv -text\n*.log -text -whitespace\n',encoding='utf-8')
for label,path in dict(actual_full_reexecution_report=report_path,actual_process_guard=guard_path,
    original_full_source_result=source_path,original_full_artifact_proof=proof_path,actual_queue_state=state_path).items():
    shutil.copyfile(path,packet/(label+'.json'))
attempt=ROOT/c['state_root']/'attempts'/identity
for name in ['stdout.log','stderr.log']:shutil.copyfile(attempt/name,packet/('actual_worker_'+name))
write(packet/'independent_complete_array_and_evaluator_statistic_checks.json',dict(passed=True,
    full_generated_shape=list(first.shape),full_reference_shape=list(reference.shape),all_generated_bytes_equal=True,
    dtype=str(first.dtype),all_original_artifact_hashes_rechecked=True,all_four_by_five_statistics_recomputed=checks,
    RNG_files_reloaded_by_this_cold_archive=False,RNG_files_native_restore_and_comparison_passed=r['complete_RNG_restoration_and_equal_consumption'],
    new_generator_training_updates=0,all_original_experiments_complete=False))
write(packet/'all14_five_training_seed_cohort_completion.json',cohorts)
with (packet/'all20_actual_retrained_evaluator_values.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
shutil.copyfile(tests,packet/'prior_frozen_source_48_native_environment_checks.xml')
shutil.copyfile(__file__,packet/'closure_source.py')
write(packet/'iteration_status.json',dict(original_scope=c['original_scope'],actual_full_generative_reexecution_conditions=1,
    required_original_conditions=70,complete_five_training_seed_cohorts=0,required_cohorts=14,
    previously_completed_source_training_models_not_credited_again=True,new_generator_training_updates=0,
    all_original_experiments_complete=False))
summary='\n'.join(f'- {k}: new mean={v["mean"]:.9f}; source mean={v["original_source_mean"]:.9f}; difference={v["difference"]:.9f}' for k,v in checks.items())
(packet/'README.md').write_text(f'''# Spectral/Diffusion-TS Stocks seed42：首个真正完整生成与评估器重跑

原始完整训练、采样、全部评估和产物审计已先行通过。本轮重新加载相同生成模型、EMA、优化器及其完成检查点，禁止生成器更新，执行原始完整采样两次。
训练种子42不变；因历史采样前RNG未保留，明确采用独立采样种子1000042。恢复落盘Python/NumPy/Torch CPU/CUDA状态后，完整生成数组{list(first.shape)}的所有样本、特征及额外尾部行逐字节相同，结束RNG消耗相等。
原完整参考{list(reference.shape)}和检查点SHA256不变，所有网络/EMA/优化器状态已在原生进程中核对。
随后实际重新训练全部学习式评估器，执行Context-FID、Correlational、Discriminative、Predictive四类各5次完整原始评估；原始训练预算、网格、采样批量、求解器及评估代码未缩减。
原生进程退出0，CPU/GPU资源守卫通过、无资源中止。归档再次检查所有输出哈希、完整数组有限性/逐字节相等和4×5次评估的统计量。

{summary}

这些差异是固定已训练模型在明确新采样随机流下的结果，不是历史采样随机流恢复，也不声称与原历史生成数组逐字节相同。
每个指标的5次评估只来自1个已训练生成模型，不能当作5个训练种子；Stocks种子43..46及其他69条件仍需各自完整重跑。当前完成1/70、完整五训练种子组0/14，不计算五种子均值或算法优越性。
此前33/70的源训练加产物审计与这里1/70的真实生成/评估器重跑是两种阶段，不能相加或互相抵扣。
冻结前48项原生环境实现检查已通过，本次另有真实端到端数值执行及全部产物/统计核验，不把旧单元测试当作新的模型完成数。
不新增生成模型训练种子槽，不改写原始源实验数据或结果。完整45/681/273/702研究范围和其他未完成实验保留，整体任务未完成。
''',encoding='utf-8')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/Spectral_Stocks42_true_generation_reexecution.2026-10-16_v1.json';assert not ara.exists()
write(ara,dict(configuration=s.CONFIG,result_packet=packet.relative_to(ROOT).as_posix(),actual_full_reexecution_conditions=1,
    required_conditions=70,new_generator_training_updates=0,historical_RNG_recovered=False,all_original_experiments_complete=False))
def head(path):return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
paths=[ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
write(ROOT/'tmp/Spectral_first_full_generation_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),full_generated_shape=list(first.shape),metrics=checks)))
