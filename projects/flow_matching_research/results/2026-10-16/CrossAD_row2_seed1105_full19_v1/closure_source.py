"""Archive the nineteenth actual complete condition, never a partial epoch."""
from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import shutil
import subprocess
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
prior=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row1_seed1105_full18_v1'
previous=read(prior/'all18_proof_bindings.json')['records']
identity='crossad_ablation__row2__gecco__seed1105';required=set(previous)|{identity}
assert len(previous)==18 and len(required)==19
assert required.issubset(cs['verified_original_CUDA_ids']) and required.issubset(ms['verified_ids'])
proofs={};reports={};rows=[];bindings={}
jobs={j['canonical_CPU_condition_reference']:j for j in q['jobs']}
for cid in sorted(required):
    j=jobs[cid];pp=Path(original['artifact_root'])/'conditions'/(cid+'.json');mp=Path(mc['artifact_root'])/'conditions'/(cid+'.json')
    proof,report=read(pp),read(mp)
    assert groups.condition(original,q,j)==proof and metrics.audit(mc,q,upstream,j)==report
    guard_path=ROOT/mc['state_root']/'guards'/(cid+'.json');guard=read(guard_path)
    assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['report_sha256']==sha(mp)
    bindings[cid]=dict(full_array_proof_path=str(pp),full_array_proof_sha256=sha(pp),independent_metric_report_path=str(mp),independent_metric_report_sha256=sha(mp))
    if cid in previous:assert bindings[cid]==previous[cid]
    proofs[cid]=proof;reports[cid]=report
    for ratio,v in report['independent_strict_metrics'].items():
        rows.append(dict(canonical_id=cid,dataset=j['dataset'],training_or_eval_seed=j['seed'],execution_mode=j['mode'],
            reconstructed_ablation=j['reconstructed_ablation'],ablation_row=j.get('ablation_row'),validation_alarm_percent=ratio,threshold=v['threshold'],**v['micro']))
with (prior/'all18_conditions_54_strict_metric_rows.csv').open(encoding='utf-8-sig',newline='') as f:old_rows=list(csv.DictReader(f))
normalize=lambda r:{k:'' if v is None else str(v) for k,v in r.items()}
indexed={(r['canonical_id'],str(r['validation_alarm_percent'])):normalize(r) for r in rows}
assert len(rows)==57 and len(old_rows)==54 and all(indexed[(r['canonical_id'],r['validation_alarm_percent'])]==r for r in old_rows)
cohorts=groups.cohorts(q['jobs'],proofs)
assert cohorts['complete_primary_cohorts']==1 and not cohorts['paired_seed_ablation_differences']
for row in range(1,6):
    item=cohorts['primary_cohorts'][f'GECCO/paper_reconstruction_row{row}']
    assert item['verified_seed_ids']==([1103,1104,1105] if row in (1,2) else [1103,1104]) and not item['complete'] and 'metrics' not in item
assert json.loads(json.dumps(cohorts['primary_cohorts']['GECCO/main']))==read(prior/'all22_primary_cohort_completion.json')['primary_cohorts']['GECCO/main']
j=jobs[identity];model=read(ROOT/j['model_config'])
assert model['implementation']=='paper_reconstructed_author_harness' and model['parameters']['components']==[True,False,False,False]
rp=ROOT/j['output_directory']/'result.json';result=read(rp)
budget=training_budget(j,result['details'],read(rp.parent/'training_trace.json'))
assert budget['original_maximum_epochs']==20 and budget['original_patience']==3
paired=[];budgets={}
for seed in [1103,1104,1105]:
    aid=f'crossad_ablation__row2__gecco__seed{seed}';bid=f'crossad_author__fresh_train__gecco__seed{seed}'
    a,b=reports[aid],reports[bid]
    assert a['complete_points']==b['complete_points']==dict(train=55408,validation=13852,test=69261)
    assert proofs[aid]['raw_sha256']==proofs[bid]['raw_sha256']
    for field in ['labels','source_rows']:assert a['all_score_label_and_timestamp_payload_sha256'][field]==b['all_score_label_and_timestamp_payload_sha256'][field]
    for cid in [aid,bid]:
        actual=read(ROOT/q['delegated_first_configuration'])['job'] if cid==q['delegated_first_original_condition'] else jobs[cid]
        path=ROOT/actual['output_directory']/'result.json';r=read(path)
        assert sha(path)==proofs[cid]['source_result_sha256']
        actual_budget=training_budget(actual,r['details'],read(path.parent/'training_trace.json'))
        assert actual_budget['original_maximum_epochs']==20 and actual_budget['original_patience']==3
        budgets[cid]=dict(actual_original_budget=actual_budget,result_sha256=sha(path),training_trace_sha256=sha(path.parent/'training_trace.json'))
    for ratio in ['0.1','1','5']:
        av,bv=a['independent_strict_metrics'][ratio]['micro'],b['independent_strict_metrics'][ratio]['micro']
        paired.append(dict(training_seed=seed,validation_alarm_percent=ratio,row2_F1=av['f1'],full_F1=bv['f1'],row2_minus_full_F1=av['f1']-bv['f1'],
            row2_AUROC=av['auroc'],full_AUROC=bv['auroc'],row2_AP=av['average_precision'],full_AP=bv['average_precision'],
            row2_actual_epochs=budgets[aid]['actual_original_budget']['actual_completed_epochs'],full_actual_epochs=budgets[bid]['actual_original_budget']['actual_completed_epochs'],
            row2_actual_updates=budgets[aid]['actual_original_budget']['full_training_forward_calls'],full_actual_updates=budgets[bid]['actual_original_budget']['full_training_forward_calls']))
xml=ROOT/'tmp/CrossAD_row2_seed1105_full19_validation_20261016_v1.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==15 and all(int(s.get('failures',0))==int(s.get('errors',0))==int(s.get('skipped',0))==0 for s in suites)
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row2_seed1105_full19_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
paths=dict(source_result=rp,training_trace=rp.parent/'training_trace.json',model_configuration=ROOT/j['model_config'],
    native_replay=Path(q['artifact_root'])/identity/'native_replay/report.json',full_checkpoint_array_proof=Path(original['artifact_root'])/'conditions'/(identity+'.json'),
    independent_metric_report=Path(mc['artifact_root'])/'conditions'/(identity+'.json'),independent_metric_guard=ROOT/mc['state_root']/'guards'/(identity+'.json'),
    **{phase+'_guard':ROOT/q['state_root']/'jobs'/identity/(phase+'.guard.json') for phase in ['CUDA_probe','CPU_reference','author','replay']})
for name,path in paths.items():shutil.copyfile(path,packet/(name+'.json'))
shutil.copyfile(xml,packet/'relevant_fresh15_full19_validation.xml')
write(packet/'actual_original_training_budget_verification.json',budget)
write(packet/'all22_primary_cohort_completion.json',cohorts)
for name,data in [('all19_conditions_57_strict_metric_rows.csv',rows),('all9_same_seed_row2_comparisons_with_actual_training_budget.csv',paired)]:
    with (packet/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
write(packet/'all19_proof_bindings.json',dict(records=bindings,prior18_strict_metric_rows_unchanged=True,prior_main_full_five_seed_summary_unchanged=True))
write(packet/'all6_paired_original_training_budget_bindings.json',dict(records=budgets,partial_seed_mean_or_CI_computed=False,actual_update_counts_may_differ_under_same_original_stopping_rule=True))
handles=[]
for state in [cs,ms]:
    p=psutil.Process(state['pid']);assert p.is_running() and p.create_time()==state['create_time']
    handles.append(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()))
write(packet/'actual_live_collectors.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=handles))
write(packet/'iteration_status.json',dict(original_scope=q['original_scope'],CrossAD_full_CUDA_conditions=19,CrossAD_required_conditions=139,
    complete_reconstructed_ablation_conditions=12,row1_verified_seeds=[1103,1104,1105],row2_verified_seeds=[1103,1104,1105],
    required_primary_seeds=groups.SEEDS,complete_primary_cohorts=1,required_primary_cohorts=22,
    original_CPU_completion_credit=0,new_numerical_executions_by_this_closure=0,all_original_experiments_complete=False))
a=reports[identity]['independent_strict_metrics']['1']['micro'];b=reports['crossad_author__fresh_train__gecco__seed1105']['independent_strict_metrics']['1']['micro']
(packet/'README.md').write_text(f'''# CrossAD row2 / seed1105：第19个完整GPU条件

完成{budget['actual_completed_epochs']}轮、{budget['full_training_forward_calls']}个完整训练批次；原20轮上限、patience3、batch128及432批次/轮保持。
CPU/GPU梯度核对、作者运行、检查点重放四项守卫均退出0且无资源中止。55408训练点、13852验证点、69261测试点含13点尾部全部重放一致。

标准库从完整数组独立复算验证阈值、混淆计数、F1、AUROC和AP。验证1%阈值、无PA：F1={a['f1']:.12f}，AUROC={a['auroc']:.12f}，AP={a['average_precision']:.12f}。
同种子完整模型F1={b['f1']:.12f}，差值row2-main={a['f1']-b['f1']:.12f}。
row2仅保留multi-scale，关闭cross-scale、subseries和global-context；作者精确消融代码未发布，保持paper_reconstructed_author_harness标记。
row1与row2均为1103..1105三种子、3/5，其他消融各2/5。列出row2的9项同种子阈值配对和全部6模型实际训练预算，不计算部分五种子均值或区间。
原相同早停规则可能产生不同实际更新数，不能称完全匹配计算量的纯组件因果实验。

累计19/139，包括6个新训练主模型、1个发布权重评估、12个重建消融条件；完整主要五种子组仍1/22。
全部19条件的数组和独立指标再次复核，前18条绑定和54行指标及主模型五种子统计不变。相关15项测试本轮全部通过。
该闭合不额外训练或推理，不证明历史作者或完整TAB等价，原CPU条件不抵扣GPU结果。
全部45篇、681方法记录、273数据任务、702论文指标及原论文/TAB/新数据任务保持，整体未完成。
''',encoding='utf-8',newline='\n')
shutil.copyfile(__file__,packet/'closure_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/CrossAD_row2_seed1105_full19.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),CrossAD_complete_CUDA_conditions=19,required_conditions=139,
    row2_complete_seeds=3,complete_primary_cohorts=1,all_original_experiments_complete=False))
head=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
files=[ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
write(ROOT/'tmp/CrossAD_row2_seed1105_full19_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(files)))
print(json.dumps(dict(packet=str(packet),one_percent_metrics=a,one_percent_F1_difference=a['f1']-b['f1'],full_conditions=19)))
