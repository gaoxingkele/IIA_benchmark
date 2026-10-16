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
identity='crossad_ablation__row3__gecco__seed1103'
required=set(cs['verified_original_CUDA_ids']);assert identity in required and len(required)==10
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
partial=cohorts['primary_cohorts']['GECCO/paper_reconstruction_row3']
assert not partial['complete'] and partial['verified_seed_ids']==[1103] and 'metrics' not in partial
job=next(j for j in q['jobs'] if j['canonical_CPU_condition_reference']==identity)
model=read(ROOT/job['model_config'])
assert job['reconstructed_ablation'] and model['implementation']=='paper_reconstructed_author_harness'
assert model['parameters']['components']==[True,True,False,False]
rp=ROOT/job['output_directory']/'result.json';result=read(rp)
from scripts.flow_matching.audit_crossad_full_native_completion_v1 import training_budget
budget=training_budget(job,result['details'],read(rp.parent/'training_trace.json'))
assert budget['original_maximum_epochs']==20 and budget['original_patience']==3
main_id='crossad_author__fresh_train__gecco__seed1103';a=metric_reports[identity];b=metric_reports[main_id]
assert a['complete_points']==b['complete_points']==dict(train=55408,validation=13852,test=69261)
assert proofs[identity]['raw_sha256']==proofs[main_id]['raw_sha256']
for field in ['labels','source_rows']:
    assert a['all_score_label_and_timestamp_payload_sha256'][field]==b['all_score_label_and_timestamp_payload_sha256'][field]
ablation_comparison=dict(full_model=b['independent_strict_metrics']['1']['micro'],
    reconstructed_row3=a['independent_strict_metrics']['1']['micro'],same_training_seed=1103,
    F1_difference_row3_minus_full=a['independent_strict_metrics']['1']['micro']['f1']-b['independent_strict_metrics']['1']['micro']['f1'],
    one_seed_point_comparison_only=True,full_five_seed_ablation_complete=False,
    exact_unreleased_author_ablation_architecture_certified=False)


prior=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row2_full9_v1'
with (prior/'all9_conditions_27_strict_metric_rows.csv').open(encoding='utf-8-sig',newline='') as f: old=list(csv.DictReader(f))
assert len(old)==27 and {x['canonical_id'] for x in old}==required-{identity}
def textrow(row):return {k:'' if v is None else str(v) for k,v in row.items()}
new={ (x['canonical_id'],str(x['validation_alarm_percent'])):textrow(x) for x in strict_rows}
assert all(new[(x['canonical_id'],x['validation_alarm_percent'])]==x for x in old)
previous_cohorts=read(prior/'all22_primary_cohort_completion.json')
normalize=lambda x:json.loads(json.dumps(x))
assert normalize(cohorts['primary_cohorts']['GECCO/main'])==previous_cohorts['primary_cohorts']['GECCO/main']
previous_bindings=read(prior/'all9_proof_bindings.json')['records']
assert set(previous_bindings)==required-{identity}
for cid,entry in previous_bindings.items():
    assert sha(entry['full_array_proof_path'])==entry['full_array_proof_sha256']
    assert sha(entry['independent_metric_report_path'])==entry['independent_metric_report_sha256']
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row3_full10_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n',encoding='utf-8')
paths=dict(source_result=rp,training_trace=rp.parent/'training_trace.json',model_configuration=ROOT/job['model_config'],
    native_replay=Path(q['artifact_root'])/identity/'native_replay/report.json',
    full_checkpoint_array_proof=Path(original['artifact_root'])/'conditions'/(identity+'.json'),
    independent_metric_report=Path(mc['artifact_root'])/'conditions'/(identity+'.json'),
    independent_metric_guard=ROOT/mc['state_root']/'guards'/(identity+'.json'),
    **{p+'_guard':ROOT/q['state_root']/'jobs'/identity/(p+'.guard.json') for p in ['CUDA_probe','CPU_reference','author','replay']})
for label,path in paths.items():shutil.copyfile(path,packet/(label+'.json'))
write(packet/'actual_original_training_budget_verification.json',budget)
write(packet/'all22_primary_cohort_completion.json',cohorts)
write(packet/'same_seed_full_vs_reconstructed_row3.json',ablation_comparison)
with (packet/'all10_conditions_30_strict_metric_rows.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(strict_rows[0]));writer.writeheader();writer.writerows(strict_rows)
write(packet/'all10_proof_bindings.json',dict(records={cid:dict(
    full_array_proof_path=str(Path(original['artifact_root'])/'conditions'/(cid+'.json')),
    full_array_proof_sha256=sha(Path(original['artifact_root'])/'conditions'/(cid+'.json')),
    independent_metric_report_path=str(Path(mc['artifact_root'])/'conditions'/(cid+'.json')),
    independent_metric_report_sha256=sha(Path(mc['artifact_root'])/'conditions'/(cid+'.json'))) for cid in sorted(required)},
    prior9_strict_metric_rows_unchanged=True,prior_full_model_five_seed_summary_unchanged=True))
handles=[]
for state in [cs,ms]:
    p=psutil.Process(state['pid']);assert p.is_running() and p.create_time()==state['create_time']
    handles.append(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()))
write(packet/'actual_live_collectors.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=handles))
write(packet/'iteration_status.json',dict(original_scope=q['original_scope'],CrossAD_full_CUDA_conditions=10,
    CrossAD_required_conditions=139,complete_reconstructed_ablation_conditions=3,complete_primary_cohorts=1,
    required_primary_cohorts=22,original_CPU_completion_credit=0,all_original_experiments_complete=False))
ab=ablation_comparison['reconstructed_row3'];full=ablation_comparison['full_model']
(packet/'README.md').write_text(f'''# CrossAD GECCO 第三行重建消融完整核验

row3/seed1103完成{budget['actual_completed_epochs']}轮、{budget['full_training_forward_calls']}训练批次；原20轮上限与patience3早停规则核验通过。CPU/GPU初始梯度比较、全部原始评估和全量新鲜检查点重放均通过，四项过程守卫退出0且无资源中止。
训练55408、验证13852、测试69261点全部保留，包括13点尾部；源与重放完整评分数组字节相同。独立标准库完整重算各阈值、混淆计数、F1、ROC与AP通过。
row3按配置启用multi-scale与cross-scale；subseries、global-context关闭。原作者未发布精确消融架构，标为paper_reconstructed_author_harness，不称作者认证复现。

验证集1%阈值、无PA：F1={ab['f1']:.9f}，ROC={ab['auroc']:.9f}，AP={ab['average_precision']:.9f}。
同数据、标签、原始行号、训练种子1103的完整模型F1={full['f1']:.9f}，row3减完整模型F1差值={ab['f1']-full['f1']:.9f}。
这是单种子点值对照；row3仅1/5，尚无五种子消融均值或配对区间。所有22个主要组保留，完整组仍1/22。

全部10个完整条件重新核验全量产物与独立指标报告；此前9个条件27行严格指标及主模型五种子汇总保持不变。
累计10/139个GPU条件，包括6个新训练主模型、1个发布权重评估及3个论文重建消融条件。原CPU实验不抵扣，未停止其他数值进程。
本轮无实现改动，使用现有完整条件验证器与独立指标重算完成实际验证；此前18项实现/部分组拒绝测试不重复计为实验结果。
原始45篇/681方法记录/273数据任务/702指标及原论文、TAB、新数据轨道的全部未完成项保留，整体任务未完成。
''',encoding='utf-8')
shutil.copyfile(__file__,packet/'closure_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/CrossAD_row3_full10.2026-10-16_v1.json';assert not ara.exists()
write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),CrossAD_complete_CUDA_conditions=10,
    required_conditions=139,complete_reconstructed_ablation_conditions=3,all_original_experiments_complete=False))
def githead(path):return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
paths=[ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
write(ROOT/'tmp/CrossAD_row3_full10_publication_manifest_20261016_v1.json',dict(previous_root=githead(ROOT),previous_parent=githead(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),comparison=ablation_comparison)))
