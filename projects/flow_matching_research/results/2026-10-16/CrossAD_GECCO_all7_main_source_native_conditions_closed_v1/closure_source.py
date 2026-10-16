from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import shutil
import subprocess
import psutil
from scripts.flow_matching import collect_crossad_full139_cold_handoff_v1 as cold
from scripts.flow_matching import collect_crossad_CUDA_full139_verified_v1 as g
from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1 as m
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/cold.CONFIG);original,q=cold.verify(c)
mc=read(ROOT/m.CONFIG);_,upstream=m.verify(mc)
jobs=[j for j in q['jobs'] if j['dataset']=='GECCO' and not j['reconstructed_ablation']]
assert len(jobs)==7 and {j['mode'] for j in jobs}=={'fresh_train','release_checkpoint'}
assert {j['seed'] for j in jobs if j['mode']=='fresh_train'}=={1103,1104,1105,1106,1107,2025}
states=[read(ROOT/config['state_root']/'status.json') for config in [c,mc]]
required={j['canonical_CPU_condition_reference'] for j in jobs}
assert required.issubset(states[0]['verified_original_CUDA_ids']) and not states[0]['errors_preserved']
assert required.issubset(states[1]['verified_ids']) and not states[1]['failed_preserved']
proofs={};reports={};rows=[]
for job in jobs:
    identity=job['canonical_CPU_condition_reference']
    proof_path=Path(original['artifact_root'])/'conditions'/(identity+'.json')
    proof=read(proof_path);assert proof==g.condition(original,q,job)
    report_path=Path(mc['artifact_root'])/'conditions'/(identity+'.json')
    report=read(report_path);assert report==m.audit(mc,q,upstream,job)
    guard=read(ROOT/mc['state_root']/'guards'/(identity+'.json'))
    assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['report_sha256']==sha(report_path)
    assert report['complete_points']==dict(train=55408,validation=13852,test=69261)
    proofs[identity]=proof;reports[identity]=report
    for ratio,values in report['independent_strict_metrics'].items():
        rows.append(dict(canonical_id=identity,execution_mode=job['mode'],training_or_evaluation_seed=job['seed'],
            primary_five_training_seed=job['mode']=='fresh_train' and job['seed'] in g.SEEDS,
            validation_alarm_percent=ratio,threshold=values['threshold'],**values['micro'],**values['event_metrics']))
prior_path=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_first_complete_five_seed_GECCO_and_same_score_protocol_factorial_v1/all22_original_CUDA_cohorts.json'
summary=g.cohorts(q['jobs'],proofs)
assert json.loads(json.dumps(summary))==read(prior_path) and summary['complete_primary_cohorts']==1
assert summary['historical_seed2025_and_release_checks_excluded_from_primary_means']
new_jobs=[j for j in jobs if j['seed']==2025];assert len(new_jobs)==2
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_GECCO_all7_main_source_native_conditions_closed_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n',encoding='utf-8')
for job in new_jobs:
    identity=job['canonical_CPU_condition_reference'];folder=packet/'new_complete_conditions'/identity;folder.mkdir(parents=True)
    result_path=ROOT/job['output_directory']/'result.json';result=read(result_path)
    if job['mode']=='fresh_train':
        assert result['details']['epochs_completed']==20 and result['details']['training_forward_calls']==8640
        shutil.copyfile(result_path.parent/'training_trace.json',folder/'full_training_trace.json')
    else:
        assert result['details']['epochs_completed']==0 and result['details']['training_forward_calls']==0
    paths=dict(source_result=result_path,native_full_replay=Path(q['artifact_root'])/identity/'native_replay/report.json',
        full_checkpoint_array_proof=Path(original['artifact_root'])/'conditions'/(identity+'.json'),
        independent_metric_report=Path(mc['artifact_root'])/'conditions'/(identity+'.json'),
        independent_metric_guard=ROOT/mc['state_root']/'guards'/(identity+'.json'),
        **{phase+'_guard':ROOT/q['state_root']/'jobs'/identity/(phase+'.guard.json') for phase in ['CUDA_probe','CPU_reference','author','replay']})
    for label,path in paths.items():shutil.copyfile(path,folder/(label+'.json'))
write(packet/'all7_original_native_and_independent_metric_bindings.json',dict(records=[dict(
    canonical_id=j['canonical_CPU_condition_reference'],mode=j['mode'],seed=j['seed'],
    native_full_array_proof_sha256=sha(Path(original['artifact_root'])/'conditions'/(j['canonical_CPU_condition_reference']+'.json')),
    independent_full_metric_report_sha256=sha(Path(mc['artifact_root'])/'conditions'/(j['canonical_CPU_condition_reference']+'.json')),
    complete_payload_sha256=reports[j['canonical_CPU_condition_reference']]['all_score_label_and_timestamp_payload_sha256']) for j in jobs],
    completed_conditions=7,fresh_training_models=6,release_weight_evaluations=1))
write(packet/'all22_primary_cohorts_unchanged_after_control_and_release.json',summary)
with (packet/'all7_conditions_21_full_strict_metric_rows.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
handles=[]
for state in states:
    p=psutil.Process(state['pid']);assert p.is_running() and p.create_time()==state['create_time']
    handles.append(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()))
source_state=read(ROOT/q['state_root']/'status.json')
assert source_state['active_condition']=='crossad_ablation__row1__gecco__seed1103'
handle=source_state['exact_child_handle'];p=psutil.Process(handle['pid'])
assert p.is_running() and p.create_time()==handle['create_time'] and p.cmdline()==handle['command']
handles.append(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()))
write(packet/'actual_live_collectors_and_first_reconstructed_ablation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    handles=handles,actual_source_state=source_state,ablation_result_not_yet_complete=True))
write(packet/'iteration_status.json',dict(original_scope=q['original_scope'],CrossAD_full_original_CUDA_conditions=7,
    required_original_conditions=139,new_fresh_training_models=1,new_release_weight_evaluations=1,
    complete_primary_five_seed_cohorts=1,required_primary_cohorts=22,primary_seed_means_unchanged=True,
    original_CPU_completion_credit=0,all_original_experiments_complete=False))
control=reports['crossad_author__fresh_train__gecco__seed2025']['independent_strict_metrics']['1']['micro']
release=reports['crossad_author__release_checkpoint__gecco__seed2025']['independent_strict_metrics']['1']['micro']
(packet/'README.md').write_text(f'''# CrossAD GECCO全部7个主方法条件闭合

新增两项真实完成：控制训练种子2025完成20轮/8640训练批次及完整原始评估；发布权重使用评估种子2025完成完整测试，训练更新为0，不能当作新训练模型或已知历史训练种子。
两项均通过CUDA初始化梯度、CPU参考、原始执行、完整新鲜检查点重放四项守卫；55408训练、13852验证、69261测试点及全部标签/原始行号/尾部覆盖完整，源与重放评分逐字节一致。
独立标准库实现再次核算三个固定验证阈值及完整指标；全部7条件原始产物、数组、过程报告与证明哈希重新核验。

验证集1%阈值、无PA：
- fresh seed2025: F1={control['f1']:.9f}, ROC-AUC={control['auroc']:.9f}, AP={control['average_precision']:.9f}。
- published weights / evaluation seed2025: F1={release['f1']:.9f}, ROC-AUC={release['auroc']:.9f}, AP={release['average_precision']:.9f}。

全部7条件为6个训练模型加1个发布权重评估。预先固定的主要种子1103..1107保持5个，控制2025及发布权重不混入主要均值。
全22组摘要与此前五种子闭合文件逐字段完全相同，完整五种子组仍1/22；CSV保留全部7条件×3预算的21行完整指标。
发布权重结果不证明历史数据版本/训练过程或论文表格已复现，全部结果不冒充完整TAB。

当前完整GPU条件7/139；原始CPU实验完成数不抵扣。现场核验源控制器已进入GECCO seed1103的论文重建消融row1实际数值阶段，未把启动或预检当作消融结果。
原始45/681/273/702研究范围、全部其他方法/数据集/消融及未完成实验保持，整体任务未完成。
''',encoding='utf-8')
shutil.copyfile(__file__,packet/'closure_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/CrossAD_GECCO_all7_main.2026-10-16_v1.json';assert not ara.exists()
write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),complete_original_CUDA_conditions=7,required_conditions=139,
    primary_five_seed_group_unchanged=True,new_fresh_training_models=1,new_release_weight_evaluations=1,all_original_experiments_complete=False))
def head(path):return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
paths=[ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
write(ROOT/'tmp/CrossAD_GECCO_all7_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),complete_conditions=7,control=control,published_weight_evaluation=release)))
