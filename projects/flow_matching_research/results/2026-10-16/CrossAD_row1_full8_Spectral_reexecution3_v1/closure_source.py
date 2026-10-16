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
from scripts.flow_matching import collect_crossad_full139_cold_handoff_v1 as cold
from scripts.flow_matching import collect_crossad_CUDA_full139_verified_v1 as groups
from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1 as metrics
from scripts.flow_matching import reexecute_spectral_full70_generation_evaluators_v1 as spectral
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

cc=read(ROOT/cold.CONFIG);original,q=cold.verify(cc)
mc=read(ROOT/metrics.CONFIG);_,upstream=metrics.verify(mc)
cs=read(ROOT/cc['state_root']/'status.json');ms=read(ROOT/mc['state_root']/'status.json')
assert not cs['errors_preserved'] and not ms['failed_preserved']
identity='crossad_ablation__row1__gecco__seed1103'
required=set(cs['verified_original_CUDA_ids']);assert identity in required and len(required)==8
assert required.issubset(ms['verified_ids'])
proofs={};metric_reports={};strict_rows=[]
for cid in sorted(required):
    job=next(j for j in q['jobs'] if j['canonical_CPU_condition_reference']==cid)
    proof_path=Path(original['artifact_root'])/'conditions'/(cid+'.json');proof=read(proof_path)
    assert groups.condition(original,q,job)==proof
    mp=Path(mc['artifact_root'])/'conditions'/(cid+'.json');m=read(mp)
    assert metrics.audit(mc,q,upstream,job)==m
    mg=read(ROOT/mc['state_root']/'guards'/(cid+'.json'))
    assert mg['passed'] and mg['exit_code']==0 and not mg['resource_guard_aborted'] and mg['report_sha256']==sha(mp)
    proofs[cid]=proof;metric_reports[cid]=m
    for ratio,values in m['independent_strict_metrics'].items():
        strict_rows.append(dict(canonical_id=cid,dataset=job['dataset'],training_or_eval_seed=job['seed'],
            execution_mode=job['mode'],reconstructed_ablation=job['reconstructed_ablation'],ablation_row=job.get('ablation_row'),
            validation_alarm_percent=ratio,threshold=values['threshold'],**values['micro']))
cohorts=groups.cohorts(q['jobs'],proofs)
assert cohorts['complete_primary_cohorts']==1 and not cohorts['paired_seed_ablation_differences']
partial=cohorts['primary_cohorts']['GECCO/paper_reconstruction_row1']
assert not partial['complete'] and partial['verified_seed_ids']==[1103] and 'metrics' not in partial
job=next(j for j in q['jobs'] if j['canonical_CPU_condition_reference']==identity)
model=read(ROOT/job['model_config'])
assert job['reconstructed_ablation'] and model['implementation']=='paper_reconstructed_author_harness'
assert model['parameters']['components']==[False]*4
rp=ROOT/job['output_directory']/'result.json';result=read(rp)
assert result['details']['epochs_completed']==20 and result['details']['training_forward_calls']==8640
main_id='crossad_author__fresh_train__gecco__seed1103';a=metric_reports[identity];b=metric_reports[main_id]
assert a['complete_points']==b['complete_points']==dict(train=55408,validation=13852,test=69261)
assert proofs[identity]['raw_sha256']==proofs[main_id]['raw_sha256']
for field in ['labels','source_rows']:
    assert a['all_score_label_and_timestamp_payload_sha256'][field]==b['all_score_label_and_timestamp_payload_sha256'][field]
ablation_comparison=dict(full_model=b['independent_strict_metrics']['1']['micro'],
    reconstructed_row1=a['independent_strict_metrics']['1']['micro'],same_training_seed=1103,
    F1_difference_row1_minus_full=a['independent_strict_metrics']['1']['micro']['f1']-b['independent_strict_metrics']['1']['micro']['f1'],
    one_seed_point_comparison_only=True,full_five_seed_ablation_complete=False,
    exact_unreleased_author_ablation_architecture_certified=False)

sc=read(ROOT/spectral.CONFIG);parent,cases=spectral.verify(sc)
ss=read(ROOT/sc['state_root']/'status.json');assert not ss['failed_preserved']
sids=[f'spectral_original__diffusion_ts__stocks__released_author_data__seed{i}' for i in [42,43,44]]
assert set(sids).issubset(ss['verified_ids'])
spectral_reports={};statistics_checks={};bindings={};raw_rows=[];delta_rows=[]
assert len({cases[i]['job']['reference_sha256'] for i in sids})==1
assert len({sha(ROOT/cases[i]['job']['frozen_samples']) for i in sids})==1
for sid in sids:
    case=cases[sid];j=case['job'];out=Path(sc['artifact_root'])/sid
    report_path=out/'full_generation_and_evaluator_reexecution.json';r=read(report_path)
    gp=ROOT/sc['state_root']/'attempts'/sid/'guard_receipt.json';g=read(gp)
    assert g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted'] and g['report_sha256']==sha(report_path)
    assert r['passed'] and r['canonical_id']==sid and r['separate_sampling_seed']==1000000+j['seed']
    for key in ['original_checkpoint_unchanged','original_source_and_frozen_dataset_verified',
        'every_model_EMA_optimizer_state_unchanged','all_generated_payload_bytes_replayed',
        'complete_RNG_restoration_and_equal_consumption','original_sampler_grid_and_solver_unchanged',
        'all_four_original_evaluators_reexecuted','all_learned_evaluators_independently_retrained']:assert r[key]
    assert r['actual_new_generator_training_updates']==0 and r['full_original_sampling_calls']==2 and r['original_evaluation_repeats']==5
    assert not r['generation_field_chunking'] and not r['historical_sampling_RNG_recovered']
    assert not r['historical_original_generated_bytes_equivalence_claim']
    proof,pp=spectral.ready_proof(parent,case,sid)
    source_path=ROOT/j['output_directory']/'result.json';source=read(source_path)
    assert r['original_result_sha256']==sha(source_path) and r['original_artifact_proof_sha256']==sha(pp)
    cp=Path(next(x['path'] for x in source['artifacts'] if x['role']=='checkpoint'))
    assert sha(cp)==r['original_checkpoint_sha256']
    for item in r['artifacts']:assert sha(item['path'])==item['sha256']
    first=np.load(out/'first_full_generated.npy',mmap_mode='r',allow_pickle=False)
    second=np.load(out/'second_full_generated.npy',mmap_mode='r',allow_pickle=False)
    reference=np.load(out/'reference_data.npy',mmap_mode='r',allow_pickle=False)
    assert list(first.shape)==r['full_generation_shape']==proof['complete_generated_shape']==[4002,24,6]
    assert first.dtype==second.dtype and np.array_equal(first,second) and np.isfinite(first).all() and np.isfinite(second).all()
    assert list(reference.shape)==proof['complete_reference_shape']==[3662,24,6] and np.isfinite(reference).all()
    assert sha(out/'reference_data.npy')==j['reference_sha256']
    assert sha(out/'first_full_generated.npy')==sha(out/'second_full_generated.npy')
    checked={}
    for key,v in r['metrics'].items():
        raw=v['raw_repeats'];assert len(raw)==5 and all(math.isfinite(x) and x>=0 for x in raw)
        mean=statistics.mean(raw);sd=statistics.stdev(raw);se=sd/math.sqrt(5);half=2.7764451051977987*se
        values=dict(mean=mean,std=sd,se=se,ci95_low=mean-half,ci95_high=mean+half,author_display_halfwidth=half)
        for k,x in values.items():assert math.isclose(x,v[k],rel_tol=1e-12,abs_tol=1e-12)
        assert v['independent_training_models']==1 and v['between_training_seed_CI95'] is None
        d=r['original_vs_new_generation'][key]
        assert d['original_mean']==proof['metrics'][key]['mean'] and d['new_generation_mean']==mean and d['difference']==mean-d['original_mean']
        checked[key]=dict(raw_repeats=raw,**values,original_source_mean=d['original_mean'],difference=d['difference'])
        delta_rows.append(dict(canonical_id=sid,training_seed=j['seed'],sampling_seed=r['separate_sampling_seed'],metric=key,
            source_original_mean=d['original_mean'],new_generation_mean=mean,new_minus_original=d['difference'],
            within_model_evaluator_SD=sd,evaluator_repeats=5,independent_trained_generators=1))
        for i,value in enumerate(raw):raw_rows.append(dict(canonical_id=sid,training_seed=j['seed'],metric=key,evaluator_repeat=i,value=value))
    assert set(checked)=={'context_fid','correlational','discriminative','predictive'}
    statistics_checks[sid]=checked;spectral_reports[sid]=r
    bindings[sid]=dict(report_path=str(report_path),report_sha256=sha(report_path),guard_path=str(gp),guard_sha256=sha(gp),
        original_source_result_sha256=sha(source_path),original_artifact_proof_sha256=sha(pp),
        complete_generated_shape=list(first.shape),complete_reference_shape=list(reference.shape),all_new_generated_file_bytes_equal=True)
previous=ROOT/'projects/flow_matching_research/results/2026-10-16/Spectral_Stocks42_first_full_generation_and_all_evaluator_reexecution_v1/actual_full_reexecution_report.json'
assert sha(previous)==bindings[sids[0]]['report_sha256']
spcohorts=spectral.full_cohort_comparisons(cases,spectral_reports)
assert spcohorts['complete_cohorts']==0
stock=spcohorts['cohorts']['diffusion_ts/stocks/released_author_data']
assert stock['verified_training_seeds']==[42,43,44] and not stock['complete']
assert 'paired_training_seed_metric_differences' not in stock

tests=Path('F:/aicoding/IIA_Data/experiments/CrossAD_first_ablation_Spectral3_20261016_v1_tests.xml')
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==18
assert not any(int(x.get(k,0)) for x in suites for k in ['failures','errors','skipped'])
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row1_full8_Spectral_reexecution3_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.xml -text -whitespace\n*.py -text\n*.log -text -whitespace\n',encoding='utf-8')
folder=packet/'CrossAD_new_row1';folder.mkdir()
paths=dict(source_result=rp,training_trace=rp.parent/'training_trace.json',model_configuration=ROOT/job['model_config'],
    native_replay=Path(q['artifact_root'])/identity/'native_replay/report.json',
    full_checkpoint_array_proof=Path(original['artifact_root'])/'conditions'/(identity+'.json'),
    independent_metric_report=Path(mc['artifact_root'])/'conditions'/(identity+'.json'),
    **{p+'_guard':ROOT/q['state_root']/'jobs'/identity/(p+'.guard.json') for p in ['CUDA_probe','CPU_reference','author','replay']})
for label,path in paths.items():shutil.copyfile(path,folder/(label+'.json'))
write(packet/'CrossAD_all22_primary_cohort_completion.json',cohorts)
write(packet/'CrossAD_same_seed_full_vs_reconstructed_row1.json',ablation_comparison)
for name,values in [('CrossAD_all8_conditions_24_strict_metric_rows.csv',strict_rows),
    ('Spectral_all3_generators_12_metric_comparisons.csv',delta_rows),('Spectral_all3_generators_60_actual_evaluator_values.csv',raw_rows)]:
    with (packet/name).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
for sid in sids:
    folder=packet/'Spectral'/('seed'+str(cases[sid]['job']['seed']));folder.mkdir(parents=True)
    for label,key in [('full_generation_and_evaluator_report','report_path'),('full_reexecution_guard','guard_path')]:
        shutil.copyfile(bindings[sid][key],folder/(label+'.json'))
    attempt=ROOT/sc['state_root']/'attempts'/sid
    for name in ['stdout.log','stderr.log']:shutil.copyfile(attempt/name,folder/name)
write(packet/'Spectral_all3_source_reexecution_bindings.json',dict(records=bindings,statistics=statistics_checks,
    same_full_original_reference_and_frozen_training_input_across_all3=True,
    saved_RNG_reloaded_by_this_metadata_archive=False,full_RNG_restore_verified_by_original_native_worker=True))
write(packet/'Spectral_all14_cohort_completion.json',spcohorts)
shutil.copyfile(tests,packet/'validation_18_ablation_and_full_cohort_checks.xml');shutil.copyfile(__file__,packet/'closure_source.py')
handles=[]
for state in [cs,ms,ss]:
    p=psutil.Process(state['pid']);assert p.is_running() and p.create_time()==state['create_time']
    handles.append(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()))
write(packet/'actual_live_source_and_result_controllers.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=handles))
write(packet/'iteration_status.json',dict(original_scope=q['original_scope'],CrossAD_full_CUDA_conditions=8,CrossAD_required_conditions=139,
    new_complete_reconstructed_ablation_conditions=1,CrossAD_complete_primary_cohorts=1,CrossAD_required_primary_cohorts=22,
    Spectral_actual_complete_generation_evaluator_reexecution_conditions=3,Spectral_required_conditions=70,
    new_full_Spectral_reexecution_conditions=2,Spectral_complete_five_seed_cohorts=0,Spectral_required_cohorts=14,
    original_source_Spectral_training_models_credited_again=0,original_CPU_CrossAD_completion_credit=0,
    original_numerical_workers_stopped=False,all_original_experiments_complete=False))
ab=ablation_comparison['reconstructed_row1'];full=ablation_comparison['full_model']
table=['| Stocks训练种子 | Context-FID | Correlational | Discriminative | Predictive |','|---:|---:|---:|---:|---:|']
for sid in sids:
    v=statistics_checks[sid];table.append('| '+str(cases[sid]['job']['seed'])+' | '+' | '.join(f'{v[k]["mean"]:.6f}' for k in ['context_fid','correlational','discriminative','predictive'])+' |')
(packet/'README.md').write_text(f'''# 首个完整CrossAD论文重建消融与Spectral三个真实生成重跑

CrossAD GECCO row1/seed1103完整20轮、8640训练批次、全部原始评估、CPU/GPU初始梯度比较及全量新鲜检查点重放完成，四项过程守卫退出0且无资源中止。
55408训练/13852验证/69261测试点及标签/原始行号/尾部完整；源与重放评分逐字节相同；独立标准库完整重算阈值、TP/FP/FN、F1、ROC、AP等通过。
row1按模型配置关闭multi-scale、cross-scale、subseries与global-context四组件，复用原始encoder/decoder单尺度局部重建。
原作者未公开精确消融架构；始终标为paper_reconstructed_author_harness，不声称未发布作者消融原样复现或历史论文指标已重现。
验证集1%阈值、无PA：row1 F1={ab['f1']:.9f}，ROC={ab['auroc']:.9f}，AP={ab['average_precision']:.9f}；同一原数据/标签/行号和种子1103的完整模型F1={full['f1']:.9f}。
这是一个种子的点值对照，row1主要种子当前仅1/5，不计算配对五种子消融均值/区间。主要完整组仍1/22，完整GPU条件8/139；原CPU实验不抵扣。

Spectral/Diffusion-TS Stocks训练种子42/43/44分别完成完整两次原采样及四类各5次原评估，真实生成+学习式评估器重跑3/70；本轮新增43/44两项，原42的完整报告哈希未变。
三项输入参考和冻结训练数据SHA256相同；每项重新加载原完成模型/EMA/优化器，禁止生成器更新。明确采样种子1000042/1000043/1000044，落盘随机状态恢复后两次完整4002×24×6生成样本逐字节相同，结束RNG消耗相等，原参考3662×24×6完整保留。
全部学习式评估器真实重新训练，四类各5次原始评估预算未变；进程守卫通过无资源中止。归档逐项复查检查点、全部生成/参考/RNG文件哈希、完整数组有限性/字节相等及全部4×5统计量。

'''+ '\n'.join(table)+'''

表中每格是一个已训练生成模型的5次评估均值，不能当作五个独立训练模型。
Stocks完整五训练种子仍缺45/46，14个五种子组目前0个完整，不计算3种子总体均值或五种子区间。45的实际执行另由原队列继续。
历史采样前RNG未保留，因此本次为明确新随机流下的完整重跑，不声称恢复历史生成字节或认证历史论文数值。
原33个完成源训练/产物审计与这里3个真实生成/评估器重跑属于不同阶段，不能相加或互相抵扣。
18项消融实现/完整组/部分组拒绝等相关检查通过；不把单元fixture当作基准分数。本归档不新增生成器训练更新或原始训练种子槽。
全部45/681/273/702范围、其他模型/基线/消融/数据集及原论文/TAB/新数据轨道保持，整体任务未完成。
''',encoding='utf-8')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/CrossAD_first_ablation_Spectral3.2026-10-16_v1.json';assert not ara.exists()
write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),CrossAD_complete_CUDA_conditions=8,
    complete_reconstructed_ablation_conditions=1,Spectral_full_generation_evaluator_reexecutions=3,
    original_scope=q['original_scope'],all_original_experiments_complete=False))
def head(path):return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
files=[ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
write(ROOT/'tmp/CrossAD_first_ablation_Spectral3_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(files)))
print(json.dumps(dict(packet=str(packet),files=len(files),ablation_comparison=ablation_comparison,
    full_Spectral_reexecutions=3,Spectral_metric_means={sid:{k:v['mean'] for k,v in values.items()} for sid,values in statistics_checks.items()})))
