"""Close the actual26th CUDA case without partial-seed aggregates."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import collect_crossad_full139_cold_handoff_v1 as cold
from scripts.flow_matching import collect_crossad_CUDA_full139_verified_v1 as groups
from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1 as metrics
from scripts.flow_matching.audit_crossad_full_native_completion_v1 import training_budget
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

cc=read(ROOT/cold.CONFIG);original,q=cold.verify(cc);mc=read(ROOT/metrics.CONFIG);_,upstream=metrics.verify(mc)
cs=read(ROOT/cc['state_root']/'status.json');ms=read(ROOT/mc['state_root']/'status.json')
assert not cs['errors_preserved'] and not ms['failed_preserved']
prior=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row3_seed1106_full25_v1'
previous=read(prior/'all25_proof_bindings.json')['records'];identity='crossad_ablation__row4__gecco__seed1106'
required=set(previous)|{identity};assert len(required)==26 and len(previous)==25
assert required.issubset(cs['verified_original_CUDA_ids']) and required.issubset(ms['verified_ids'])
jobs={j['canonical_CPU_condition_reference']:j for j in q['jobs']};proofs={};bindings={};rows=[]
for cid in sorted(required):
    j=jobs[cid];pp=Path(original['artifact_root'])/'conditions'/(cid+'.json');mp=Path(mc['artifact_root'])/'conditions'/(cid+'.json')
    proof,report=read(pp),read(mp)
    binding=dict(full_array_proof_path=str(pp),full_array_proof_sha256=sha(pp),independent_metric_report_path=str(mp),independent_metric_report_sha256=sha(mp))
    if cid in previous:assert binding==previous[cid]
    else:assert groups.condition(original,q,j)==proof and metrics.audit(mc,q,upstream,j)==report
    guard=read(ROOT/mc['state_root']/'guards'/(cid+'.json'))
    assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['report_sha256']==sha(mp)
    bindings[cid]=binding;proofs[cid]=proof
    for ratio,v in report['independent_strict_metrics'].items():
        rows.append(dict(canonical_id=cid,dataset=j['dataset'],training_or_eval_seed=j['seed'],execution_mode=j['mode'],
            reconstructed_ablation=j['reconstructed_ablation'],ablation_row=j.get('ablation_row'),validation_alarm_percent=ratio,threshold=v['threshold'],**v['micro']))
with (prior/'all25_conditions_75_strict_metric_rows.csv').open(encoding='utf-8-sig',newline='') as f:old_rows=list(csv.DictReader(f))
normalize=lambda r:{k:'' if v is None else str(v) for k,v in r.items()}
index={(r['canonical_id'],str(r['validation_alarm_percent'])):normalize(r) for r in rows}
assert len(rows)==78 and len(old_rows)==75 and all(index[r['canonical_id'],r['validation_alarm_percent']]==r for r in old_rows)
cohorts=groups.cohorts(q['jobs'],proofs)
assert cohorts['complete_primary_cohorts']==1 and not cohorts['paired_seed_ablation_differences']
for row in range(1,6):
    item=cohorts['primary_cohorts'][f'GECCO/paper_reconstruction_row{row}']
    assert item['verified_seed_ids']==([1103,1104,1105,1106] if row in [1,2,3,4] else [1103,1104,1105]) and not item['complete'] and 'metrics' not in item
assert json.loads(json.dumps(cohorts['primary_cohorts']['GECCO/main']))==read(prior/'all25_primary_cohort_completion.json')['primary_cohorts']['GECCO/main']
j=jobs[identity];model=read(ROOT/j['model_config']);assert model['parameters']['components']==[True,True,False,True]
rp=ROOT/j['output_directory']/'result.json';result=read(rp);budget=training_budget(j,result['details'],read(rp.parent/'training_trace.json'))
assert budget['original_maximum_epochs']==20 and budget['original_patience']==3 and 1<=budget['actual_completed_epochs']<=20 and budget['full_training_forward_calls']==432*budget['actual_completed_epochs']
xml=ROOT/'tmp/CrossAD_row4_seed1106_full26_validation_20261016_v1.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==15 and all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
out=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row4_seed1106_full26_v1';out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8',newline='\n')
for name,path in dict(source_result=rp,training_trace=rp.parent/'training_trace.json',model_configuration=ROOT/j['model_config'],
    native_replay=Path(q['artifact_root'])/identity/'native_replay/report.json',full_checkpoint_array_proof=Path(original['artifact_root'])/'conditions'/(identity+'.json'),
    independent_metric_report=Path(mc['artifact_root'])/'conditions'/(identity+'.json'),independent_metric_guard=ROOT/mc['state_root']/'guards'/(identity+'.json'),
    **{phase+'_guard':ROOT/q['state_root']/'jobs'/identity/(phase+'.guard.json') for phase in ['CUDA_probe','CPU_reference','author','replay']}).items():shutil.copyfile(path,out/(name+'.json'))
shutil.copyfile(xml,out/'relevant_fresh15_full26_validation.xml');write(out/'actual_original_training_budget_verification.json',budget)
write(out/'all26_primary_cohort_completion.json',cohorts)
write(out/'all26_proof_bindings.json',dict(records=bindings,prior25_strict_metric_rows_unchanged=True,main_full_five_seed_summary_unchanged=True))
with (out/'all26_conditions_78_strict_metric_rows.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
handles=[]
for state in [cs,ms]:
    p=psutil.Process(state['pid']);assert p.is_running() and p.create_time()==state['create_time'];handles.append(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()))
write(out/'actual_live_collectors.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=handles))
write(out/'iteration_status.json',dict(original_scope=q['original_scope'],CrossAD_full_CUDA_conditions=26,CrossAD_required_conditions=139,
    complete_reconstructed_ablation_conditions=19,complete_primary_cohorts=1,required_primary_cohorts=22,
    original_CPU_completion_credit=0,new_numerical_executions_by_closure=0,all_original_experiments_complete=False))
a=read(out/'independent_metric_report.json')['independent_strict_metrics']['1']['micro']
for phase in ['CUDA_probe','CPU_reference','author','replay']:
    guard=read(out/(phase+'_guard.json'))
    assert guard['exit_code']==0 and not guard['resource_guard_aborted']
(out/'README.md').write_text(f'''# CrossAD row4／GECCO／seed1106：第26个完整CUDA条件

保留原20轮上限、patience3和batch128。实际完成{budget['actual_completed_epochs']}轮、{budget['full_training_forward_calls']}次训练批次，最佳验证轮为{budget['best_validation_epoch']}；是否结束由原早停逻辑决定，完整训练预算已核验。四项CPU/GPU核对、训练和检查点重放守卫均退出0、无资源中止。全部55408训练、13852验证、69261测试点含13点尾部完整重放；三个固定验证阈值的点级及事件指标独立复算通过。

验证1%无PA：F1={a['f1']:.12f}，Precision={a['precision']:.12f}，Recall={a['recall']:.12f}，AUROC={a['auroc']:.12f}，AP={a['average_precision']:.12f}。TP={a['true_positives']}、FP={a['false_positives']}、FN={a['false_negatives']}。

row4为[True,True,False,True]重建配置，作者精确消融代码及全部层/头选择未发布，保持paper_reconstructed_author_harness，不认证作者精确等价。row1..4各完成1103..1106，即4/5；row5各3/5。完整主要五种子组仍仅1/22，不计算部分组均值、区间或成对差异。不同条件的实际早停轮数应分别列出，不能假设所有条件具有相同实际训练计算量。

累计26/139：6个新训练主模型、1个发布权重评估、19个重建消融条件。此前25条证明、75行严格指标及主模型完整五种子统计保持，现在78行。15项相关验证通过；归档未额外执行训练或推理，原CPU完成记账保持独立。原45篇/681方法记录/273数据任务/702论文指标及全部条件范围保持，整体未完成。
''',encoding='utf-8',newline='\n')
shutil.copyfile(__file__,out/'publication_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/CrossAD_row4_seed1106_full26.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(packet=out.relative_to(ROOT).as_posix(),complete_CUDA_conditions=26,required_conditions=139,
    complete_primary_cohorts=1,required_primary_cohorts=22,actual_source_epochs=budget['actual_completed_epochs'],
    actual_source_training_batches=budget['full_training_forward_calls'],all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8')
text+=f'\n`CrossAD row4／seed1106完整验收，累计26/139 <results/2026-10-16/{out.name}/README.md>`_\n'
text+=f'原20轮上限/patience3保持，实际{budget["actual_completed_epochs"]}轮/{budget["full_training_forward_calls"]}批次、全原生点检查点重放和独立三阈值指标通过；验证1%无PA F1={a["f1"]:.12f}、AUROC={a["auroc"]:.12f}、AP={a["average_precision"]:.12f}。row1..4各4/5、row5为3/5，完整主要组仍1/22，不计算部分组统计。原25证明/75指标行保持，现78行，15项测试通过；作者精确等价未认证，原CPU及全研究范围保持，整体未完成。\n'
page.write_text(text,encoding='utf-8',newline='\n')
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
paths=[ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in out.iterdir() if p.is_file()]
write(ROOT/'tmp/CrossAD_row4_seed1106_full26_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=paths))
print(dict(packet=str(out),complete_conditions=26,actual_training_budget=budget,metrics=a))
