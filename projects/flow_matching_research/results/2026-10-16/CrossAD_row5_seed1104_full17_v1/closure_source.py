"""Close the actual seventeenth full condition using frozen full-array auditors."""
from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import shutil
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import collect_crossad_full139_cold_handoff_v1 as cold
from scripts.flow_matching import collect_crossad_CUDA_full139_verified_v1 as groups
from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1 as metrics
from scripts.flow_matching.audit_crossad_full_native_completion_v1 import training_budget
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

cc=read(ROOT/cold.CONFIG);original,q=cold.verify(cc)
mc=read(ROOT/metrics.CONFIG);_,upstream=metrics.verify(mc)
cs=read(ROOT/cc['state_root']/'status.json');ms=read(ROOT/mc['state_root']/'status.json')
assert not cs['errors_preserved'] and not ms['failed_preserved']
prior=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row4_seed1104_full16_v1'
previous=read(prior/'all16_proof_bindings.json')['records']
identity='crossad_ablation__row5__gecco__seed1104';required=set(previous)|{identity}
assert len(previous)==16 and len(required)==17
assert required.issubset(cs['verified_original_CUDA_ids']) and required.issubset(ms['verified_ids'])
proofs={};metric_reports={};rows=[]
for cid in sorted(required):
    job=next(j for j in q['jobs'] if j['canonical_CPU_condition_reference']==cid)
    pp=Path(original['artifact_root'])/'conditions'/(cid+'.json')
    mp=Path(mc['artifact_root'])/'conditions'/(cid+'.json')
    proof,metric=read(pp),read(mp)
    assert groups.condition(original,q,job)==proof
    assert metrics.audit(mc,q,upstream,job)==metric
    guard=read(ROOT/mc['state_root']/'guards'/(cid+'.json'))
    assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['report_sha256']==sha(mp)
    proofs[cid]=proof;metric_reports[cid]=metric
    if cid in previous:
        assert sha(pp)==previous[cid]['full_array_proof_sha256']
        assert sha(mp)==previous[cid]['independent_metric_report_sha256']
    for ratio,values in metric['independent_strict_metrics'].items():
        rows.append(dict(canonical_id=cid,dataset=job['dataset'],training_or_eval_seed=job['seed'],
            execution_mode=job['mode'],reconstructed_ablation=job['reconstructed_ablation'],
            ablation_row=job.get('ablation_row'),validation_alarm_percent=ratio,threshold=values['threshold'],**values['micro']))
with (prior/'all16_conditions_48_strict_metric_rows.csv').open(encoding='utf-8-sig',newline='') as f:old_rows=list(csv.DictReader(f))
normalize=lambda row:{k:'' if v is None else str(v) for k,v in row.items()}
indexed={(r['canonical_id'],str(r['validation_alarm_percent'])):normalize(r) for r in rows}
assert len(old_rows)==48 and len(rows)==51
assert all(indexed[(r['canonical_id'],r['validation_alarm_percent'])]==r for r in old_rows)
cohorts=groups.cohorts(q['jobs'],proofs)
assert cohorts['complete_primary_cohorts']==1 and not cohorts['paired_seed_ablation_differences']
for row in range(1,6):
    partial=cohorts['primary_cohorts'][f'GECCO/paper_reconstruction_row{row}']
    assert partial['verified_seed_ids']==[1103,1104] and not partial['complete'] and 'metrics' not in partial
assert json.loads(json.dumps(cohorts['primary_cohorts']['GECCO/main']))==read(prior/'all22_primary_cohort_completion.json')['primary_cohorts']['GECCO/main']
job=next(j for j in q['jobs'] if j['canonical_CPU_condition_reference']==identity)
model=read(ROOT/job['model_config'])
assert model['implementation']=='paper_reconstructed_author_harness'
assert model['parameters']['components']==[False,False,True,True]
rp=ROOT/job['output_directory']/'result.json';result=read(rp)
budget=training_budget(job,result['details'],read(rp.parent/'training_trace.json'))
assert budget['original_maximum_epochs']==20 and budget['original_patience']==3
paired=[]
for seed in [1103,1104]:
    cid=f'crossad_ablation__row5__gecco__seed{seed}';main=f'crossad_author__fresh_train__gecco__seed{seed}'
    a,b=metric_reports[cid],metric_reports[main]
    assert a['complete_points']==b['complete_points']==dict(train=55408,validation=13852,test=69261)
    assert proofs[cid]['raw_sha256']==proofs[main]['raw_sha256']
    for field in ['labels','source_rows']:
        assert a['all_score_label_and_timestamp_payload_sha256'][field]==b['all_score_label_and_timestamp_payload_sha256'][field]
    for ratio in ['0.1','1','5']:
        av,bv=a['independent_strict_metrics'][ratio]['micro'],b['independent_strict_metrics'][ratio]['micro']
        paired.append(dict(training_seed=seed,validation_alarm_percent=ratio,row5_F1=av['f1'],full_F1=bv['f1'],
            row5_minus_full_F1=av['f1']-bv['f1'],row5_AUROC=av['auroc'],full_AUROC=bv['auroc'],row5_AP=av['average_precision'],full_AP=bv['average_precision']))
test=Path('F:/aicoding/IIA_Data/experiments/CrossAD_row2_seed1104_full14_20261016_v1_tests.xml')
suites=list(ET.parse(test).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==15
assert all(int(s.get('failures',0))==int(s.get('errors',0))==int(s.get('skipped',0))==0 for s in suites)
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row5_seed1104_full17_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
paths=dict(source_result=rp,training_trace=rp.parent/'training_trace.json',model_configuration=ROOT/job['model_config'],
    native_replay=Path(q['artifact_root'])/identity/'native_replay/report.json',
    full_checkpoint_array_proof=Path(original['artifact_root'])/'conditions'/(identity+'.json'),
    independent_metric_report=Path(mc['artifact_root'])/'conditions'/(identity+'.json'),
    independent_metric_guard=ROOT/mc['state_root']/'guards'/(identity+'.json'),
    **{phase+'_guard':ROOT/q['state_root']/'jobs'/identity/(phase+'.guard.json') for phase in ['CUDA_probe','CPU_reference','author','replay']})
for name,path in paths.items():shutil.copyfile(path,packet/(name+'.json'))
shutil.copyfile(test,packet/'relevant_unchanged_verifier_validation.xml')
write(packet/'actual_original_training_budget_verification.json',budget)
write(packet/'all22_primary_cohort_completion.json',cohorts)
for name,records in [('all17_conditions_51_strict_metric_rows.csv',rows),('same_seed_row5_vs_full_two_seeds.csv',paired)]:
    with (packet/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
write(packet/'all17_proof_bindings.json',dict(records={cid:dict(
    full_array_proof_path=str(Path(original['artifact_root'])/'conditions'/(cid+'.json')),
    full_array_proof_sha256=sha(Path(original['artifact_root'])/'conditions'/(cid+'.json')),
    independent_metric_report_path=str(Path(mc['artifact_root'])/'conditions'/(cid+'.json')),
    independent_metric_report_sha256=sha(Path(mc['artifact_root'])/'conditions'/(cid+'.json'))) for cid in sorted(required)},
    prior16_strict_metric_rows_unchanged=True,prior_main_full_five_seed_summary_unchanged=True))
handles=[]
for state in [cs,ms]:
    proc=psutil.Process(state['pid']);assert proc.is_running() and proc.create_time()==state['create_time']
    handles.append(dict(pid=proc.pid,create_time=proc.create_time(),command=proc.cmdline()))
write(packet/'actual_live_collectors.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=handles))
write(packet/'iteration_status.json',dict(original_scope=q['original_scope'],CrossAD_full_CUDA_conditions=17,
    CrossAD_required_conditions=139,complete_reconstructed_ablation_conditions=10,
    row5_verified_seeds=[1103,1104],row5_required_seeds=groups.SEEDS,complete_primary_cohorts=1,
    required_primary_cohorts=22,original_CPU_completion_credit=0,new_numerical_executions_by_this_closure=0,all_original_experiments_complete=False))
a=metric_reports[identity]['independent_strict_metrics']['1']['micro'];b=metric_reports['crossad_author__fresh_train__gecco__seed1104']['independent_strict_metrics']['1']['micro']
(packet/'README.md').write_text(f'''# CrossAD row5/seed1104：第17个完整GPU条件

完成{budget['actual_completed_epochs']}轮、{budget['full_training_forward_calls']}个完整训练批次；原20轮上限、patience3、batch128及432批次/轮保持。CPU/GPU梯度核对、作者运行、检查点重放四项守卫退出0，无资源中止。55408训练点、13852验证点、69261测试点含13点尾部全部重放一致。

独立标准库从完整数组复算阈值、混淆计数、F1、AUROC和AP。验证集1%阈值、无PA：F1={a['f1']:.12f}，AUROC={a['auroc']:.12f}，AP={a['average_precision']:.12f}；同种子完整模型F1={b['f1']:.12f}，差值={a['f1']-b['f1']:.12f}。

row5启用subseries和global-context，关闭multi-scale和cross-scale；使用源编码/解码与EMA原语重建单尺度查询和全局bank。作者精确消融实现未发布，保留paper_reconstructed_author_harness状态。row1至row5均完成1103、1104两个训练种子，各2/5；两组同种子差值单列，不计算部分五种子均值或区间。完整主要五种子组仍1/22。

累计17/139，包括6个新训练主模型、1个发布权重评估、10个重建消融条件。全部17条件的完整数组和独立指标重新核验；前16条件的产物哈希、48行指标及主模型五种子统计不变。附原15项检查XML，验证器未修改，本次完整数组重算不算新的单元测试。该闭合不证明历史论文或完整TAB等价；原CPU条件不抵扣GPU结果。

全部45篇论文、681方法记录、273数据任务、702原始指标，以及原论文、TAB、新数据轨道继续保留。整体未完成。
''',encoding='utf-8')
shutil.copyfile(__file__,packet/'closure_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/CrossAD_row5_seed1104_full17.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),CrossAD_complete_CUDA_conditions=17,
    required_conditions=139,row5_complete_seeds=2,complete_primary_cohorts=1,all_original_experiments_complete=False))
print(json.dumps(dict(packet=str(packet),one_percent_metrics=a,one_percent_F1_difference=a['f1']-b['f1'],full_conditions=17)))
