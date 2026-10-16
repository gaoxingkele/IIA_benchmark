from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import math
import shutil
import statistics
import subprocess
import psutil
import numpy as np
from scripts.flow_matching import reexecute_spectral_full70_generation_evaluators_v1 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/s.CONFIG);parent,cases=s.verify(c)
prior_root=ROOT/'projects/flow_matching_research/results/2026-10-16/Spectral_Stocks45_full_reexecution4_v1'
prior_path=prior_root/'all4_complete_generator_reexecution_catalog.json';prior=read(prior_path)
assert len(prior['records'])==4
identity='spectral_original__diffusion_ts__stocks__released_author_data__seed46'
sids=[f'spectral_original__diffusion_ts__stocks__released_author_data__seed{i}' for i in [42,43,44,45,46]]
state_path=ROOT/c['state_root']/'status.json';state=read(state_path)
assert set(sids).issubset(state['verified_ids']) and not state['failed_preserved']
owner=psutil.Process(state['pid']);assert owner.is_running() and owner.create_time()==state['create_time']
reports={};bindings=dict(prior['records']);checks={};rows=[];repeat_rows=[]
assert len({cases[x]['job']['reference_sha256'] for x in sids})==1
assert len({sha(ROOT/cases[x]['job']['frozen_samples']) for x in sids})==1
for sid in sids:
    job=cases[sid]['job'];out=Path(c['artifact_root'])/sid
    path=out/'full_generation_and_evaluator_reexecution.json';r=read(path)
    gp=ROOT/c['state_root']/'attempts'/sid/'guard_receipt.json';g=read(gp)
    assert g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted'] and g['report_sha256']==sha(path)
    assert r['passed'] and r['canonical_id']==sid and r['separate_sampling_seed']==1000000+job['seed']
    for key in ['original_checkpoint_unchanged','original_source_and_frozen_dataset_verified',
        'every_model_EMA_optimizer_state_unchanged','all_generated_payload_bytes_replayed',
        'complete_RNG_restoration_and_equal_consumption','original_sampler_grid_and_solver_unchanged',
        'all_four_original_evaluators_reexecuted','all_learned_evaluators_independently_retrained']:assert r[key]
    assert r['actual_new_generator_training_updates']==0 and r['full_original_sampling_calls']==2 and r['original_evaluation_repeats']==5
    assert not r['generation_field_chunking'] and not r['historical_sampling_RNG_recovered']
    assert not r['historical_original_generated_bytes_equivalence_claim']
    proof,proof_path=s.ready_proof(parent,cases[sid],sid)
    source_path=ROOT/job['output_directory']/'result.json';source=read(source_path)
    assert r['original_result_sha256']==sha(source_path) and r['original_artifact_proof_sha256']==sha(proof_path)
    cp=Path(next(x['path'] for x in source['artifacts'] if x['role']=='checkpoint'))
    assert sha(cp)==r['original_checkpoint_sha256']
    for item in r['artifacts']:assert sha(item['path'])==item['sha256']
    first=np.load(out/'first_full_generated.npy',mmap_mode='r',allow_pickle=False)
    second=np.load(out/'second_full_generated.npy',mmap_mode='r',allow_pickle=False)
    ref=np.load(out/'reference_data.npy',mmap_mode='r',allow_pickle=False)
    assert list(first.shape)==r['full_generation_shape']==proof['complete_generated_shape']==[4002,24,6]
    assert first.dtype==second.dtype and np.array_equal(first,second) and np.isfinite(first).all() and np.isfinite(second).all()
    assert list(ref.shape)==proof['complete_reference_shape']==[3662,24,6] and np.isfinite(ref).all()
    assert sha(out/'first_full_generated.npy')==sha(out/'second_full_generated.npy')
    assert sha(out/'reference_data.npy')==job['reference_sha256']
    if sid in prior['records']:
        assert sha(path)==prior['records'][sid]['report_sha256'] and sha(gp)==prior['records'][sid]['guard_sha256']
    else:
        bindings[sid]=dict(report_path=str(path),report_sha256=sha(path),guard_path=str(gp),guard_sha256=sha(gp),
            original_source_result_sha256=sha(source_path),original_artifact_proof_sha256=sha(proof_path),
            complete_generated_shape=list(first.shape),complete_reference_shape=list(ref.shape),all_new_generated_file_bytes_equal=True)
    values={}
    for metric,item in r['metrics'].items():
        raw=item['raw_repeats'];assert len(raw)==5 and all(math.isfinite(v) and v>=0 for v in raw)
        mean=statistics.mean(raw);sd=statistics.stdev(raw);se=sd/math.sqrt(5);half=2.7764451051977987*se
        stats=dict(mean=mean,std=sd,se=se,ci95_low=mean-half,ci95_high=mean+half,author_display_halfwidth=half)
        for name,value in stats.items():assert math.isclose(item[name],value,rel_tol=1e-12,abs_tol=1e-12)
        assert item['independent_training_models']==1 and item['between_training_seed_CI95'] is None
        diff=r['original_vs_new_generation'][metric]
        assert diff['original_mean']==proof['metrics'][metric]['mean'] and diff['new_generation_mean']==mean
        assert diff['difference']==mean-diff['original_mean']
        values[metric]=dict(raw_repeats=raw,**stats,original_source_mean=diff['original_mean'],difference=diff['difference'])
        rows.append(dict(canonical_id=sid,training_seed=job['seed'],sampling_seed=r['separate_sampling_seed'],metric=metric,
            source_original_mean=diff['original_mean'],new_generation_mean=mean,new_minus_original=diff['difference'],
            within_model_evaluator_SD=sd,evaluator_repeats=5,independent_trained_generators=1))
        for i,value in enumerate(raw):repeat_rows.append(dict(canonical_id=sid,training_seed=job['seed'],metric=metric,evaluator_repeat=i,value=value))
    assert set(values)=={'context_fid','correlational','discriminative','predictive'}
    if sid in prior['statistics']:assert values==prior['statistics'][sid]
    checks[sid]=values;reports[sid]=r

assert {k:bindings[k] for k in prior['records']}==prior['records']
cohorts=s.full_cohort_comparisons(cases,reports)
stock=cohorts['cohorts']['diffusion_ts/stocks/released_author_data']
assert cohorts['complete_cohorts']==1 and stock['complete'] and stock['verified_training_seeds']==[42,43,44,45,46]
def interval(v):
    m=statistics.mean(v);sd=statistics.stdev(v);half=2.7764451051977987*sd/math.sqrt(5)
    return dict(mean=m,sample_SD=sd,t95_low=m-half,t95_high=m+half,n_independent_training_models=5,df=4)
summary={};summary_rows=[]
for metric in ['context_fid','correlational','discriminative','predictive']:
    old=[checks[x][metric]['original_source_mean'] for x in sids]
    new=[checks[x][metric]['mean'] for x in sids]
    dif=[b-a for a,b in zip(old,new)]
    stat=interval(dif);actual=stock['paired_training_seed_metric_differences'][metric]
    assert stat['mean']==actual['mean'] and stat['sample_SD']==actual['sample_SD']
    assert [stat['t95_low'],stat['t95_high']]==actual['t95_interval']
    summary[metric]=dict(original=interval(old),new_sampling=interval(new),paired_difference=stat,
        original_model_means=old,new_sampling_model_means=new,paired_model_differences=dif)
    summary_rows.append(dict(metric=metric,original_mean=statistics.mean(old),new_sampling_mean=statistics.mean(new),
        paired_difference=stat['mean'],paired_difference_SD=stat['sample_SD'],paired_t95_low=stat['t95_low'],paired_t95_high=stat['t95_high'],training_models=5,evaluator_runs_per_model=5))
runtime_cohorts=read(Path(c['artifact_root'])/'full14_cohort_comparisons.json')
assert runtime_cohorts==cohorts
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/Spectral_Stocks5_full_reexecution_cohort_v1'
packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.log -text -whitespace\n',encoding='utf-8')
out=Path(c['artifact_root'])/identity;attempt=ROOT/c['state_root']/'attempts'/identity
for label,path in dict(seed46_actual_reexecution_report=out/'full_generation_and_evaluator_reexecution.json',
    seed46_process_guard=attempt/'guard_receipt.json',actual_source_queue_state=state_path).items():
    shutil.copyfile(path,packet/(label+'.json'))
for name in ['stdout.log','stderr.log']:shutil.copyfile(attempt/name,packet/('seed46_'+name))
write(packet/'all5_complete_generator_reexecution_catalog.json',dict(records=bindings,statistics=checks,
    original_first4_catalog_path=prior_path.relative_to(ROOT).as_posix(),original_first4_catalog_sha256=sha(prior_path),
    first4_entries_and_report_hashes_unchanged=True,full_reexecutions=5,required_conditions=70))
write(packet/'all14_five_seed_cohort_completion.json',cohorts)
write(packet/'five_training_model_statistics.json',dict(metrics=summary,
    training_seeds=[42,43,44,45,46],within_model_evaluator_runs=5,
    scope='Descriptive five trained-generator means on one frozen Stocks reference; paired new sampling versus original generation; no historical RNG recovery or generalization claim',
    historical_generated_byte_equivalence_claim=False,all_original_experiments_complete=False))
for name,values in [('all5_generators_20_metric_comparisons.csv',rows),('all5_generators_100_actual_evaluator_values.csv',repeat_rows),('five_model_summary.csv',summary_rows)]:
    with (packet/name).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
write(packet/'iteration_status.json',dict(original_scope=c['original_scope'],full_generation_evaluator_reexecutions=5,
    required_conditions=70,complete_five_training_seed_cohorts=1,required_cohorts=14,
    new_full_reexecution_conditions=1,new_generator_training_updates=0,new_training_seed_slots=0,
    prior4_unchanged=True,original_numerical_workers_stopped=False,all_original_experiments_complete=False))
lines=['# Stocks：首个完整五训练种子生成与评估重跑组', '',
    'Diffusion-TS/Stocks/released_author_data 的 seed42–46 全部完成，每个模型执行两次完整原采样并恢复全部随机状态，随后四类原评估器各实际重跑5次。',
    '每份生成数组4002×24×6，参考3662×24×6；两次生成完整字节及结束随机状态一致，原检查点、模型/EMA/优化器与冻结输入未变。预算与求解器未缩减。',
    '此前四条目录记录及报告哈希逐项保持。新增仅seed46，不重复计入生成器训练完成数，生成器新增训练更新为0。', '',
    '| 指标 | 原生成均值 | 新采样均值 | 配对差值 | 配对差值95%区间 |',
    '|---|---:|---:|---:|---|']
for metric,item in summary.items():
    d=item['paired_difference']
    lines.append(f"| {metric} | {item['original']['mean']:.9f} | {item['new_sampling']['mean']:.9f} | {d['mean']:.9f} | [{d['t95_low']:.9f}, {d['t95_high']:.9f}] |")
lines += ['', '以上以五个独立训练模型为统计单位，每个模型的五次评估先取均值；区间为df=4的描述性t区间。四项配对区间均包含0，不能据此证明统计等价或论文指标复现成功。',
    '新采样种子为训练种子+1000000；历史采样随机状态未知，不声称历史生成字节重现。CSV保留20行模型指标比较与100个实际评估值。',
    '当前完整生成与评估重跑5/70、完整五种子组1/14。原始45篇/681方法记录/273数据任务/702指标及原论文、TAB、新数据轨道的未完成项全部保留，整体目标未完成。']
(packet/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
shutil.copyfile(__file__,packet/'closure_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/Spectral_Stocks5_full_reexecution_cohort.2026-10-16_v1.json'
assert not ara.exists()
write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),full_generation_evaluator_reexecutions=5,
    required_conditions=70,complete_five_training_seed_cohorts=1,required_cohorts=14,all_original_experiments_complete=False))
def head(path):return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
paths=[ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
write(ROOT/'tmp/Spectral_Stocks5_cohort_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),full_reexecutions=5,complete_cohorts=1,metrics=summary),ensure_ascii=False))
