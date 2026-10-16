"""Close the actual23rd CUDA case without partial-seed aggregates."""
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
prior=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row5_seed1105_full22_v1'
previous=read(prior/'all22_proof_bindings.json')['records'];identity='crossad_ablation__row1__gecco__seed1106'
required=set(previous)|{identity};assert len(required)==23 and len(previous)==22
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
with (prior/'all22_conditions_66_strict_metric_rows.csv').open(encoding='utf-8-sig',newline='') as f:old_rows=list(csv.DictReader(f))
normalize=lambda r:{k:'' if v is None else str(v) for k,v in r.items()}
index={(r['canonical_id'],str(r['validation_alarm_percent'])):normalize(r) for r in rows}
assert len(rows)==69 and len(old_rows)==66 and all(index[r['canonical_id'],r['validation_alarm_percent']]==r for r in old_rows)
cohorts=groups.cohorts(q['jobs'],proofs)
assert cohorts['complete_primary_cohorts']==1 and not cohorts['paired_seed_ablation_differences']
for row in range(1,6):
    item=cohorts['primary_cohorts'][f'GECCO/paper_reconstruction_row{row}']
    assert item['verified_seed_ids']==([1103,1104,1105,1106] if row==1 else [1103,1104,1105]) and not item['complete'] and 'metrics' not in item
assert json.loads(json.dumps(cohorts['primary_cohorts']['GECCO/main']))==read(prior/'all22_primary_cohort_completion.json')['primary_cohorts']['GECCO/main']
j=jobs[identity];model=read(ROOT/j['model_config']);assert model['parameters']['components']==[False,False,False,False]
rp=ROOT/j['output_directory']/'result.json';result=read(rp);budget=training_budget(j,result['details'],read(rp.parent/'training_trace.json'))
assert budget['original_maximum_epochs']==20 and budget['original_patience']==3 and budget['actual_completed_epochs']==20 and budget['full_training_forward_calls']==8640
xml=ROOT/'tmp/CrossAD_row1_seed1106_full23_validation_20261016_v1.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==15 and all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
out=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row1_seed1106_full23_v1';out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8',newline='\n')
for name,path in dict(source_result=rp,training_trace=rp.parent/'training_trace.json',model_configuration=ROOT/j['model_config'],
    native_replay=Path(q['artifact_root'])/identity/'native_replay/report.json',full_checkpoint_array_proof=Path(original['artifact_root'])/'conditions'/(identity+'.json'),
    independent_metric_report=Path(mc['artifact_root'])/'conditions'/(identity+'.json'),independent_metric_guard=ROOT/mc['state_root']/'guards'/(identity+'.json'),
    **{phase+'_guard':ROOT/q['state_root']/'jobs'/identity/(phase+'.guard.json') for phase in ['CUDA_probe','CPU_reference','author','replay']}).items():shutil.copyfile(path,out/(name+'.json'))
shutil.copyfile(xml,out/'relevant_fresh15_full23_validation.xml');write(out/'actual_original_training_budget_verification.json',budget)
write(out/'all22_primary_cohort_completion.json',cohorts)
write(out/'all23_proof_bindings.json',dict(records=bindings,prior22_strict_metric_rows_unchanged=True,main_full_five_seed_summary_unchanged=True))
with (out/'all23_conditions_69_strict_metric_rows.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
handles=[]
for state in [cs,ms]:
    p=psutil.Process(state['pid']);assert p.is_running() and p.create_time()==state['create_time'];handles.append(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()))
write(out/'actual_live_collectors.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=handles))
write(out/'iteration_status.json',dict(original_scope=q['original_scope'],CrossAD_full_CUDA_conditions=23,CrossAD_required_conditions=139,
    complete_reconstructed_ablation_conditions=16,complete_primary_cohorts=1,required_primary_cohorts=22,
    original_CPU_completion_credit=0,new_numerical_executions_by_closure=0,all_original_experiments_complete=False))
a=read(out/'independent_metric_report.json')['independent_strict_metrics']['1']['micro']
(out/'README.md').write_text(f'''# CrossAD row1／GECCO／seed1106：第23个完整CUDA条件

完整20轮、8640个训练批次，保持原patience3和batch128。四项实际CPU/GPU核对、源训练和检查点重放守卫均退出0且无资源中止。
55408训练点、13852验证点、69261测试点含13点尾部全部完整重放；三个固定验证阈值的点级指标与事件统计独立复算一致。
验证1%无PA：F1={a['f1']:.12f}，Precision={a['precision']:.12f}，Recall={a['recall']:.12f}，AUROC={a['auroc']:.12f}，AP={a['average_precision']:.12f}。
TP={a['true_positives']}、FP={a['false_positives']}、FN={a['false_negatives']}。

row1为[False,False,False,False]的登记重建配置。作者精确消融源码及全部层/头选择未发布，保持paper_reconstructed_author_harness，不声称作者认证等价。
row1现1103..1106为4/5，其余四个消融各3/5，完整主要五种子组仍1/22；不计算部分均值或区间。
累计23/139条件包括6个新训练主模型、1个发布权重评估及16个重建消融；原22证明、66行指标和主模型完整五种子统计保持，现69行。
15项相关验证通过，不新增或替代原CPU完成记账，全部原条件及45/681/273/702范围保留，整体未完成。
''',encoding='utf-8',newline='\n')
shutil.copyfile(__file__,out/'publication_source.py');write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/CrossAD_row1_seed1106_full23.2026-10-16_v1.json';assert not ara.exists()
write(ara,dict(packet=out.relative_to(ROOT).as_posix(),full_CUDA_conditions=23,required_conditions=139,row1_primary_seeds_complete=4,required_primary_seeds=5,all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8')
text+=f'''
`CrossAD row1／seed1106完整验收，累计23/139 <results/2026-10-16/CrossAD_row1_seed1106_full23_v1/README.md>`_
原完整20轮、8640更新和全部原生数据点重放、独立指标通过；验证1%无PA F1={a['f1']:.12f}、AUROC={a['auroc']:.12f}、AP={a['average_precision']:.12f}。
row1为4/5、其他消融各3/5，完整主要组仍1/22，不计算部分五种子统计；原22项证明和66行指标保持，15项相关验证通过。
原CPU及全部研究范围保持，作者精确消融等价仍未认证，整体未完成。
'''
page.write_text(text,encoding='utf-8',newline='\n')
paths=[ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()]
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
write(ROOT/'tmp/CrossAD_row1_seed1106_full23_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(out),files=len(paths),full_CUDA_conditions=23,metrics=a)))
