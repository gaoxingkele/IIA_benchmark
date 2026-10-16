"""Close the actual29th CUDA case without partial-seed aggregates."""
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
prior=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row1_seed1107_full28_v1'
previous=read(prior/'all28_proof_bindings.json')['records'];identity='crossad_ablation__row2__gecco__seed1107'
required=set(previous)|{identity};assert len(required)==29 and len(previous)==28
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
with (prior/'all28_conditions_84_strict_metric_rows.csv').open(encoding='utf-8-sig',newline='') as f:old_rows=list(csv.DictReader(f))
normalize=lambda r:{k:'' if v is None else str(v) for k,v in r.items()}
index={(r['canonical_id'],str(r['validation_alarm_percent'])):normalize(r) for r in rows}
assert len(rows)==87 and len(old_rows)==84 and all(index[r['canonical_id'],r['validation_alarm_percent']]==r for r in old_rows)
cohorts=groups.cohorts(q['jobs'],proofs)
assert cohorts['complete_primary_cohorts']==3
assert set(cohorts['paired_seed_ablation_differences'])=={f'GECCO/paper_reconstruction_row{r}_minus_main' for r in [1,2]}
for row in range(1,6):
    item=cohorts['primary_cohorts'][f'GECCO/paper_reconstruction_row{row}']
    assert item['verified_seed_ids']==([1103,1104,1105,1106,1107] if row in [1,2] else [1103,1104,1105,1106])
    assert item['complete']==(row in [1,2])
    if row not in [1,2]:assert 'metrics' not in item
prior_cohorts=read(prior/'all28_primary_cohort_completion.json')
for name,item in prior_cohorts['primary_cohorts'].items():
    if item['complete']:assert json.loads(json.dumps(cohorts['primary_cohorts'][name]))==item
for name,item in prior_cohorts['paired_seed_ablation_differences'].items():
    assert json.loads(json.dumps(cohorts['paired_seed_ablation_differences'][name]))==item
primary='validation_false_alarm_percent/1/micro/f1'
main_stats=cohorts['primary_cohorts']['GECCO/main']['metrics'][primary]
row_stats=cohorts['primary_cohorts']['GECCO/paper_reconstruction_row2']['metrics'][primary]
paired_stats=cohorts['paired_seed_ablation_differences']['GECCO/paper_reconstruction_row2_minus_main'][primary]
assert main_stats['n']==row_stats['n']==paired_stats['n']==5
j=jobs[identity];model=read(ROOT/j['model_config']);assert model['parameters']['components']==[True,False,False,False]
rp=ROOT/j['output_directory']/'result.json';result=read(rp);budget=training_budget(j,result['details'],read(rp.parent/'training_trace.json'))
assert budget['original_maximum_epochs']==20 and budget['original_patience']==3 and 1<=budget['actual_completed_epochs']<=20 and budget['full_training_forward_calls']==432*budget['actual_completed_epochs']
xml=ROOT/'tmp/CrossAD_row2_seed1107_full29_validation_20261016_v1.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==15 and all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
out=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row2_seed1107_full29_v1';out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8',newline='\n')
for name,path in dict(source_result=rp,training_trace=rp.parent/'training_trace.json',model_configuration=ROOT/j['model_config'],
    native_replay=Path(q['artifact_root'])/identity/'native_replay/report.json',full_checkpoint_array_proof=Path(original['artifact_root'])/'conditions'/(identity+'.json'),
    independent_metric_report=Path(mc['artifact_root'])/'conditions'/(identity+'.json'),independent_metric_guard=ROOT/mc['state_root']/'guards'/(identity+'.json'),
    **{phase+'_guard':ROOT/q['state_root']/'jobs'/identity/(phase+'.guard.json') for phase in ['CUDA_probe','CPU_reference','author','replay']}).items():shutil.copyfile(path,out/(name+'.json'))
shutil.copyfile(xml,out/'relevant_fresh15_full29_validation.xml');write(out/'actual_original_training_budget_verification.json',budget)
write(out/'all29_primary_cohort_completion.json',cohorts)
write(out/'all29_proof_bindings.json',dict(records=bindings,prior28_strict_metric_rows_unchanged=True,main_full_five_seed_summary_unchanged=True))
with (out/'all29_conditions_87_strict_metric_rows.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
handles=[]
for state in [cs,ms]:
    p=psutil.Process(state['pid']);assert p.is_running() and p.create_time()==state['create_time'];handles.append(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()))
write(out/'actual_live_collectors.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=handles))
write(out/'iteration_status.json',dict(original_scope=q['original_scope'],CrossAD_full_CUDA_conditions=29,CrossAD_required_conditions=139,
    complete_reconstructed_ablation_conditions=22,complete_primary_cohorts=3,required_primary_cohorts=22,
    original_CPU_completion_credit=0,new_numerical_executions_by_closure=0,all_original_experiments_complete=False))
a=read(out/'independent_metric_report.json')['independent_strict_metrics']['1']['micro']
for phase in ['CUDA_probe','CPU_reference','author','replay']:
    guard=read(out/(phase+'_guard.json'))
    assert guard['exit_code']==0 and not guard['resource_guard_aborted']
(out/'README.md').write_text(f'''# CrossAD row2／GECCO／seed1107：第二个完整五种子消融组

新增第29个完整CUDA条件。原20轮上限、patience3及batch128保持；本条件实际{budget['actual_completed_epochs']}轮、{budget['full_training_forward_calls']}个训练批次，最佳验证轮为{budget['best_validation_epoch']}。全部55408训练、13852验证及69261测试点含13点尾部重放，四阶段资源守卫及独立三阈值指标通过。

row2为[True,False,False,False]论文重建配置，原作者精确消融源码及层/头选择未发布，不认证作者精确等价。row2种子1103至1107完整验收。验证1%阈值、无PA F1：row2均值{row_stats['mean']:.12f}，种子SD {row_stats['seed_sample_SD']:.12f}，均值t95区间{row_stats['mean_t95_interval']}；主模型均值{main_stats['mean']:.12f}，种子SD {main_stats['seed_sample_SD']:.12f}，均值t95区间{main_stats['mean_t95_interval']}。

同五个种子的row2减主模型F1差值均值{paired_stats['mean']:.12f}，差值SD {paired_stats['seed_sample_SD']:.12f}，成对均值t95区间{paired_stats['mean_t95_interval']}。区间仅描述固定划分上的训练种子波动，不代表未知实体、历史作者复现误差或多重比较显著性。row1此前完整组及成对差值全部保持，row3至5各4/5，不计算部分统计。所有固定阈值的完整组统计和逐种子向量见all29_primary_cohort_completion.json。

累计29/139条件、3/22完整主要组：6个新训练主模型、1个发布权重评估、22个重建消融条件。此前28条证明、84行严格指标及既有主模型/row1统计保持，现87行严格指标。15项相关验证通过，归档未额外训练或推理。原CPU轨道及45论文/681方法记录/273数据任务/702论文指标全部范围保持，整体未完成。
''',encoding='utf-8',newline='\n')
shutil.copyfile(__file__,out/'publication_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/CrossAD_row2_seed1107_full29.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(packet=out.relative_to(ROOT).as_posix(),complete_CUDA_conditions=29,required_conditions=139,
    complete_primary_cohorts=3,required_primary_cohorts=22,actual_source_epochs=budget['actual_completed_epochs'],
    actual_source_training_batches=budget['full_training_forward_calls'],all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8')
text+=f'\n`CrossAD第二个完整五种子消融组，累计29／139、主要完整组3／22 <results/2026-10-16/{out.name}/README.md>`_\n'
text+=f'row2种子1103至1107完整训练/重放/独立指标通过。验证1%无PA F1均值{row_stats["mean"]:.12f}，主模型均值{main_stats["mean"]:.12f}，同种子差值均值{paired_stats["mean"]:.12f}，t95区间{paired_stats["mean_t95_interval"]}；原主模型/row1统计、成对差值及28证明/84指标行保持，现87行。row3至5各4/5不聚合，作者精确消融等价未认证；15项检查通过，全研究未完成。\n'
page.write_text(text,encoding='utf-8',newline='\n')
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
paths=[ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in out.iterdir() if p.is_file()]
write(ROOT/'tmp/CrossAD_row2_seed1107_full29_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=paths))
print(dict(packet=str(out),complete_conditions=29,actual_training_budget=budget,metrics=a))
