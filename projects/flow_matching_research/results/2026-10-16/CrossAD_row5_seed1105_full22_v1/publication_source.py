"""Close one actual original-budget CUDA condition, preserve all previous20."""
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
prior=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row4_seed1105_full21_v1'
previous=read(prior/'all21_proof_bindings.json')['records']
identity='crossad_ablation__row5__gecco__seed1105';required=set(previous)|{identity}
assert len(previous)==21 and len(required)==22
assert required.issubset(cs['verified_original_CUDA_ids']) and required.issubset(ms['verified_ids'])
jobs={j['canonical_CPU_condition_reference']:j for j in q['jobs']}
proofs={};reports={};bindings={};rows=[]
for cid in sorted(required):
    j=jobs[cid];pp=Path(original['artifact_root'])/'conditions'/(cid+'.json');mp=Path(mc['artifact_root'])/'conditions'/(cid+'.json')
    proof,report=read(pp),read(mp)
    binding=dict(full_array_proof_path=str(pp),full_array_proof_sha256=sha(pp),independent_metric_report_path=str(mp),independent_metric_report_sha256=sha(mp))
    if cid in previous:assert binding==previous[cid]
    else:assert groups.condition(original,q,j)==proof and metrics.audit(mc,q,upstream,j)==report
    guard_path=ROOT/mc['state_root']/'guards'/(cid+'.json');guard=read(guard_path)
    assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['report_sha256']==sha(mp)
    bindings[cid]=binding;proofs[cid]=proof;reports[cid]=report
    for ratio,v in report['independent_strict_metrics'].items():
        rows.append(dict(canonical_id=cid,dataset=j['dataset'],training_or_eval_seed=j['seed'],execution_mode=j['mode'],
            reconstructed_ablation=j['reconstructed_ablation'],ablation_row=j.get('ablation_row'),validation_alarm_percent=ratio,threshold=v['threshold'],**v['micro']))
with (prior/'all21_conditions_63_strict_metric_rows.csv').open(encoding='utf-8-sig',newline='') as f:old_rows=list(csv.DictReader(f))
normal=lambda r:{k:'' if v is None else str(v) for k,v in r.items()}
index={(r['canonical_id'],str(r['validation_alarm_percent'])):normal(r) for r in rows}
assert len(rows)==66 and len(old_rows)==63 and all(index[r['canonical_id'],r['validation_alarm_percent']]==r for r in old_rows)
cohorts=groups.cohorts(q['jobs'],proofs)
assert cohorts['complete_primary_cohorts']==1 and not cohorts['paired_seed_ablation_differences']
for row in range(1,6):
    item=cohorts['primary_cohorts'][f'GECCO/paper_reconstruction_row{row}']
    assert item['verified_seed_ids']==([1103,1104,1105] if row in (1,2,3,4,5) else [1103,1104])
    assert not item['complete'] and 'metrics' not in item
assert json.loads(json.dumps(cohorts['primary_cohorts']['GECCO/main']))==read(prior/'all22_primary_cohort_completion.json')['primary_cohorts']['GECCO/main']
j=jobs[identity];model=read(ROOT/j['model_config'])
assert model['implementation']=='paper_reconstructed_author_harness' and model['parameters']['components']==[False,False,True,True]
rp=ROOT/j['output_directory']/'result.json';result=read(rp)
budget=training_budget(j,result['details'],read(rp.parent/'training_trace.json'))
assert budget['original_maximum_epochs']==20 and budget['original_patience']==3 and budget['actual_completed_epochs']==20
assert budget['full_training_forward_calls']==8640
paired=[];budgets={}
for seed in [1103,1104,1105]:
    aid=f'crossad_ablation__row5__gecco__seed{seed}';bid=f'crossad_author__fresh_train__gecco__seed{seed}';rid=f'crossad_ablation__row1__gecco__seed{seed}'
    for cid in [aid,bid,rid]:
        assert reports[cid]['complete_points']==dict(train=55408,validation=13852,test=69261)
        assert proofs[cid]['raw_sha256']==proofs[aid]['raw_sha256']
        for field in ['labels','source_rows']:
            assert reports[cid]['all_score_label_and_timestamp_payload_sha256'][field]==reports[aid]['all_score_label_and_timestamp_payload_sha256'][field]
        actual=read(ROOT/q['delegated_first_configuration'])['job'] if cid==q['delegated_first_original_condition'] else jobs[cid]
        path=ROOT/actual['output_directory']/'result.json';r=read(path)
        assert sha(path)==proofs[cid]['source_result_sha256']
        actual_budget=training_budget(actual,r['details'],read(path.parent/'training_trace.json'))
        assert actual_budget['original_maximum_epochs']==20 and actual_budget['original_patience']==3
        budgets[cid]=dict(actual_original_budget=actual_budget,result_sha256=sha(path),training_trace_sha256=sha(path.parent/'training_trace.json'))
    for ratio in ['0.1','1','5']:
        av,bv,rv=[reports[cid]['independent_strict_metrics'][ratio]['micro'] for cid in [aid,bid,rid]]
        paired.append(dict(training_seed=seed,validation_alarm_percent=ratio,row5_F1=av['f1'],full_F1=bv['f1'],row5_minus_full_F1=av['f1']-bv['f1'],
            row5_AUROC=av['auroc'],full_AUROC=bv['auroc'],row5_AP=av['average_precision'],full_AP=bv['average_precision'],
            row5_actual_epochs=budgets[aid]['actual_original_budget']['actual_completed_epochs'],full_actual_epochs=budgets[bid]['actual_original_budget']['actual_completed_epochs'],
            row5_actual_updates=budgets[aid]['actual_original_budget']['full_training_forward_calls'],full_actual_updates=budgets[bid]['actual_original_budget']['full_training_forward_calls'],
            nearest_row1_F1=rv['f1'],row5_minus_row1_F1=av['f1']-rv['f1'],nearest_row1_AUROC=rv['auroc'],row5_minus_row1_AUROC=av['auroc']-rv['auroc'],
            nearest_row1_AP=rv['average_precision'],row5_minus_row1_AP=av['average_precision']-rv['average_precision'],
            nearest_row1_actual_epochs=budgets[rid]['actual_original_budget']['actual_completed_epochs'],nearest_row1_actual_updates=budgets[rid]['actual_original_budget']['full_training_forward_calls']))
xml=ROOT/'tmp/CrossAD_row5_seed1105_full22_validation_20261016_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==15 and all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
out=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row5_seed1105_full22_v1';out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8',newline='\n')
paths=dict(source_result=rp,training_trace=rp.parent/'training_trace.json',model_configuration=ROOT/j['model_config'],
    native_replay=Path(q['artifact_root'])/identity/'native_replay/report.json',full_checkpoint_array_proof=Path(original['artifact_root'])/'conditions'/(identity+'.json'),
    independent_metric_report=Path(mc['artifact_root'])/'conditions'/(identity+'.json'),independent_metric_guard=ROOT/mc['state_root']/'guards'/(identity+'.json'),
    **{phase+'_guard':ROOT/q['state_root']/'jobs'/identity/(phase+'.guard.json') for phase in ['CUDA_probe','CPU_reference','author','replay']})
for name,path in paths.items():shutil.copyfile(path,out/(name+'.json'))
shutil.copyfile(xml,out/'relevant_fresh15_full22_validation.xml')
write(out/'actual_original_training_budget_verification.json',budget)
write(out/'all22_primary_cohort_completion.json',cohorts)
for name,records in [('all22_conditions_66_strict_metric_rows.csv',rows),('all9_same_seed_row5_comparisons_with_actual_training_budget.csv',paired)]:
    with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
write(out/'all22_proof_bindings.json',dict(records=bindings,prior21_strict_metric_rows_unchanged=True,prior_main_full_five_seed_summary_unchanged=True))
assert len(budgets)==9
write(out/'all9_paired_original_training_budget_bindings.json',dict(records=budgets,partial_seed_mean_or_CI_computed=False,
    actual_update_counts_may_differ_under_same_original_stopping_rule=True))
handles=[]
for state in [cs,ms]:
    p=psutil.Process(state['pid']);assert p.is_running() and p.create_time()==state['create_time']
    handles.append(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()))
write(out/'actual_live_collectors.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=handles))
write(out/'iteration_status.json',dict(original_scope=q['original_scope'],CrossAD_full_CUDA_conditions=22,CrossAD_required_conditions=139,
    complete_reconstructed_ablation_conditions=15,required_primary_seeds=groups.SEEDS,complete_primary_cohorts=1,required_primary_cohorts=22,
    original_CPU_completion_credit=0,new_numerical_executions_by_this_closure=0,all_original_experiments_complete=False))
a=reports[identity]['independent_strict_metrics']['1']['micro']
b=reports['crossad_author__fresh_train__gecco__seed1105']['independent_strict_metrics']['1']['micro']
r=reports['crossad_ablation__row1__gecco__seed1105']['independent_strict_metrics']['1']['micro']
(out/'README.md').write_text(f'''# CrossAD row5／GECCO／seed1105：第22个完整CUDA条件

原20轮上限、patience3及batch128保持，完成全部20轮、8640个完整训练批次。
CPU/GPU梯度核对、作者运行、检查点重放四项守卫均退出0且无资源中止；完整55408训练、13852验证、69261测试点含13点尾部均通过。

独立公式复算验证0.1%/1%/5%阈值、全部混淆计数、F1、并列AUROC/AP和事件指标。
验证1%无PA：F1={a['f1']:.12f}，Precision={a['precision']:.12f}，Recall={a['recall']:.12f}，AUROC={a['auroc']:.12f}，AP={a['average_precision']:.12f}；TP={a['true_positives']}、FP={a['false_positives']}、FN={a['false_negatives']}。
同seed1105完整模型F1={b['f1']:.12f}，row5-main差={a['f1']-b['f1']:.12f}。
最近结构row1 F1={r['f1']:.12f}，row5-row1差={a['f1']-r['f1']:.12f}；TP差={a['true_positives']-r['true_positives']}，FP差={a['false_positives']-r['false_positives']}。

row5组件为[False,False,True,True]：子序列和全局分支启用，多尺度与跨尺度关闭。
作者没有公布精确消融实现及全部层/头选择，保持paper_reconstructed_author_harness，不声称作者认证等价。
与row1[False,False,False,False]比较对应联合加入子序列/全局分支，与完整模型比较则联合关闭多尺度/跨尺度，均不能当作只改一个组件。
各1103..1105和三种固定阈值共9项配对均列出，保留9模型实际轮数与更新数；相同原早停规则也可能产生不同实际预算，不能称计算量完全匹配的纯组件因果效应。
每个模型阈值由自身验证分数计算，规则相同不代表阈值数值相同。

累计22/139完整CUDA条件：6个新训练主模型、1个发布检查点评估和15个重建消融条件。
row1..row5均各3/5种子；完整主要五种子组仍1/22，不计算部分均值、STD或CI。
原21项证明哈希、63行严格指标和主模型完整五种子汇总均保持；新增3行，完整66行与全部22项绑定已保存。
15项相关验证通过。原CPU条件不获CUDA替代完成记账，不新增训练种子；历史论文和完整TAB等价仍未认证，整体未完成。
''',encoding='utf-8',newline='\n')
shutil.copyfile(__file__,out/'publication_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/CrossAD_row5_seed1105_full22.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(packet=out.relative_to(ROOT).as_posix(),full_CUDA_conditions=22,required_conditions=139,
    row5_seed1105_actual_epochs=20,row5_seed1105_actual_updates=8640,complete_primary_cohorts=1,required_primary_cohorts=22,
    all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8')
text+='''
`CrossAD row5／seed1105完整闭环，累计22/139 <results/2026-10-16/CrossAD_row5_seed1105_full22_v1/README.md>`_
原20轮上限/patience3规则下完整20轮、8640更新，四项守卫和完整检查点/原生数组重放、独立指标通过。
验证1%无PA F1=0.367864693446、AUROC=0.838596641496、AP=0.362568606681；实际预算与row1和完整模型分别配对。
保留子序列/全局分支、关闭多尺度/跨尺度，明确两组联合组件比较和重建选择；row1..row5均各3/5，完整主要组仍1/22。
原21项证明与63行指标保持，新增3行；15项验证通过，原CPU和全局范围保持，整体未完成。
'''
page.write_text(text,encoding='utf-8',newline='\n')
paths=[ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()]
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
write(ROOT/'tmp/CrossAD_row5_seed1105_full22_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(out),files=len(paths),full_CUDA_conditions=22,new_condition_metrics=a,actual_epochs=20,actual_updates=8640)))
