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
prior_root=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row1_full8_Spectral_reexecution3_v1'
prior_path=prior_root/'Spectral_all3_source_reexecution_bindings.json';prior=read(prior_path)
assert len(prior['records'])==3
identity='spectral_original__diffusion_ts__stocks__released_author_data__seed45'
sids=[f'spectral_original__diffusion_ts__stocks__released_author_data__seed{i}' for i in [42,43,44,45]]
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
cohorts=s.full_cohort_comparisons(cases,reports);stock=cohorts['cohorts']['diffusion_ts/stocks/released_author_data']
assert cohorts['complete_cohorts']==0 and stock['verified_training_seeds']==[42,43,44,45]
assert not stock['complete'] and 'paired_training_seed_metric_differences' not in stock
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/Spectral_Stocks45_full_reexecution4_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.log -text -whitespace\n',encoding='utf-8')
case_out=Path(c['artifact_root'])/identity;attempt=ROOT/c['state_root']/'attempts'/identity
for label,path in dict(new_seed45_actual_reexecution_report=case_out/'full_generation_and_evaluator_reexecution.json',
    new_seed45_process_guard=attempt/'guard_receipt.json',actual_source_queue_state=state_path).items():shutil.copyfile(path,packet/(label+'.json'))
for name in ['stdout.log','stderr.log']:shutil.copyfile(attempt/name,packet/('new_seed45_'+name))
write(packet/'all4_complete_generator_reexecution_catalog.json',dict(records=bindings,statistics=checks,
    original_first3_catalog_path=prior_path.relative_to(ROOT).as_posix(),original_first3_catalog_sha256=sha(prior_path),
    original_first3_entries_and_reports_unchanged=True,full_reexecution_conditions=4,required_conditions=70,
    all_original_experiments_complete=False))
write(packet/'all14_original_five_seed_cohort_completion.json',cohorts)
for name,values in [('all4_generators_16_metric_comparisons.csv',rows),('all4_generators_80_actual_evaluator_values.csv',repeat_rows)]:
    with (packet/name).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
write(packet/'iteration_status.json',dict(original_scope=c['original_scope'],full_generation_and_evaluator_reexecutions=4,
    required_conditions=70,complete_five_training_seed_cohorts=0,required_cohorts=14,
    Stocks_training_seeds_verified=[42,43,44,45],Stocks_required_remaining_training_seeds=[46],
    new_full_reexecution_conditions=1,new_generator_training_updates=0,original_training_models_credited_again=0,
    original_numerical_workers_stopped=False,all_original_experiments_complete=False))
new=checks[identity]
(packet/'README.md').write_text(f'''# Spectral/Diffusion-TS Stocks seed45：第4个完整生成与评估器重跑

seed45重新加载已完成的原始生成器/EMA/优化器，使用明确的新采样种子1000045，从落盘Python/NumPy/Torch CPU/CUDA状态恢复后执行两次完整原采样。
两份4002×24×6生成数组全部字节及结束RNG消耗相同；原3662×24×6参考、原检查点及模型/EMA/优化器状态不变，生成器新增训练更新为0。
随后全部学习式评估器实际重新训练，四类各5次完整原始评估完成，守卫进程退出0、无资源中止。采样批量/求解器/网格/评估预算均未缩减。
seed45评估均值：Context-FID={new['context_fid']['mean']:.9f}，Correlational={new['correlational']['mean']:.9f}，Discriminative={new['discriminative']['mean']:.9f}，Predictive={new['predictive']['mean']:.9f}。

重新核验此前42/43/44及本次45全部输出文件、检查点、参考与冻结训练输入哈希，所有完整数组有限且两次生成相同，全部4×5评估统计重新计算。
前3条目录记录与原始报告哈希保持，新增仅seed45。CSV包含4模型×4指标的16行比较和80个实际评估值；旧报告通过不可变路径/哈希引用，避免重复拷贝。
4/70条件完成，Stocks尚缺原定seed46，完整五训练种子组仍0/14，不计算4种子均值或五种子区间。
每个指标的5次评估不等于5个训练模型；新采样随机流不是历史采样RNG恢复，不声称历史生成字节或论文数值已认证。
全部原始70条件、14组及45/681/273/702研究范围、其他未完成训练/基线/消融/原论文/TAB/新数据轨道保留，整体任务未完成。
''',encoding='utf-8')
shutil.copyfile(__file__,packet/'closure_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/Spectral_Stocks45_full_reexecution4.2026-10-16_v1.json';assert not ara.exists()
write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),full_generation_evaluator_reexecutions=4,
    required_conditions=70,complete_five_training_seed_cohorts=0,all_original_experiments_complete=False))
def head(path):return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
paths=[ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
write(ROOT/'tmp/Spectral_Stocks45_full4_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),full_reexecutions=4,new_seed45_metrics={k:v['mean'] for k,v in new.items()})))
