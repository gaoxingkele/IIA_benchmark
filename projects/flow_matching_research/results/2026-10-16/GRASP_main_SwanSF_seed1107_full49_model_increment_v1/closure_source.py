from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import run_grasp_completion_replay_queue_v1 as g
from scripts.flow_matching.run_grasp_registered_replay_review_pipeline_v2 import binding
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

runtime=read(ROOT/'configs/runtime/fm_grasp_completion_replay_queue.v1.json')
originals,executions,_,_=g.verify(runtime)
identity='grasp_protocol_v3__main__swan__swan_sf__seed1107'
reference=g.completed_pipeline(runtime,identity)
pipeline_path=ROOT/reference['completed_pipeline_configuration'];pipeline=read(pipeline_path)
producer=binding(pipeline)
config,native,receipt=g.current_native(pipeline['producer_configuration'])
assert len(native['records'])==1
record=native['records'][0];assert record['canonical_id']==identity
result_path=ROOT/record['result_path'];result=read(result_path)
g.check_complete_result(originals[identity],executions[identity],result)
assert record['epochs_completed']==1500 and record['optimizer_updates']==93000 and record['parameter_count']==69244
assert sha(result_path)==record['result_sha256'] and sha(record['checkpoint_path'])==record['checkpoint_sha256']
assert len(record['recipes'])==len(originals[identity]['score_recipes'])==10
for artifact in result['artifacts']:assert sha(ROOT/artifact['path'])==artifact['sha256']
for recipe in record['recipes']:
    assert recipe['complete_validation_points']==3935 and recipe['complete_test_points']==72000
    assert recipe['full_recipe_reinference_passed']
    for role in ['original','replay']:assert sha(recipe[role+'_score_path'])==recipe[role+'_score_sha256']
review_path=ROOT/pipeline['review_output_directory']/'independent_full_replay_verification.json'
review=read(review_path)
assert review['passed'] and review['actual_completed_recipe_count']==10
assert review['all_original_forward_counts_and_recipe_budgets_verified'] and review['all_metrics_independently_recomputed_from_full_replay']
assert review['terminal_resource_receipt']==receipt and receipt['exit_code']==0 and not receipt['resource_guard_aborted']
prior_path=ROOT/'projects/flow_matching_research/results/2026-10-16/grasp_gnn_seed1107_full48_model_increment_v1/actual_full48_model_increment_closure.json'
prior=read(prior_path);assert len(set(prior['current_complete_original_ids']))==48
for name,digest in prior['current_source_proof_hashes'].items():assert sha(prior_path.parent/(name+'.json'))==digest
state_path=ROOT/runtime['state_root']/'status.json';state=read(state_path)
assert set(state['verified_original_ids'])==set(prior['current_complete_original_ids'])|{identity}
assert not state['failed_preserved']
owner=psutil.Process(state['pid']);assert owner.is_running() and owner.create_time()==state['create_time']
seeds=sorted(int(x.rsplit('seed',1)[1]) for x in state['verified_original_ids'] if x.startswith('grasp_protocol_v3__main__swan__swan_sf__seed'))
required=sorted(originals[x]['seed'] for x in originals if x.startswith('grasp_protocol_v3__main__swan__swan_sf__seed'))
assert seeds==list(range(1103,1108)) and required==list(range(1103,1113))
tests=Path('F:/aicoding/IIA_Data/experiments/GRASP_main1107_full49_closure_20261016_v1_tests.xml')
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==51
assert not any(int(x.get(k,0)) for x in suites for k in ['failures','errors','skipped'])
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/GRASP_main_SwanSF_seed1107_full49_model_increment_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.xml -text -whitespace\n*.py -text\n',encoding='utf-8')
paths=dict(original_result=result_path,native_configuration=ROOT/pipeline['producer_configuration'],
    native_report=ROOT/producer['output_directory']/'checkpoint_replay_audit.json',native_guarded_receipt=ROOT/producer['state_root']/'receipt.json',
    completed_pipeline_configuration=pipeline_path,completed_pipeline_proof=ROOT/reference['completed_full_replay_proof'],
    independent_configuration=ROOT/pipeline['review_configuration'],independent_full_recipe_proof=review_path,
    independent_guarded_receipt=ROOT/pipeline['state_root']/'receipt.json',
    original_training_resource_usage=ROOT/result['job']['output_directory']/'resource_usage.json',
    frozen_current_full9120_completion_state=state_path)
for label,path in paths.items():shutil.copyfile(path,packet/(label+'.json'))
closure=dict(captured_utc=datetime.now(timezone.utc).isoformat(),canonical_id=identity,
    complete_original_training_and_full_native_replay_and_independent_recipe_review=True,
    epochs=1500,optimizer_updates=93000,parameter_count=69244,recipes=record['recipes'],
    actual_current_unique_complete_models=49,current_complete_original_ids=state['verified_original_ids'],
    prior48_model_closure=prior_path.relative_to(ROOT).as_posix(),prior48_closure_sha256=sha(prior_path),
    current_source_proof_hashes={name:sha(path) for name,path in paths.items()},
    required_original_entity_models=9120,required_original_training_cohorts=480,required_original_training_seeds=10,
    complete_Main_SwanSF_seeds=seeds,pending_Main_SwanSF_seeds=sorted(set(required)-set(seeds)),
    full_ten_seed_Main_SwanSF_cohort_complete=False,additional_model_inference_during_closure=0,additional_optimizer_updates_during_closure=0,
    inherited_native_template_prose_discrepancy='Frozen template prose mentions tau1/seed1104;actual canonical/main profile/seed/checkpoint/source and all10structured recipes prove main seed1107. Frozen template/results preserved.',
    score_tolerance=config['score_tolerance'],bitwise_score_identity_claim=False,
    author_equivalence_certified=False,strict_TAB_result=False,original_scope=runtime['original_scope'],all_original_experiments_complete=False)
write(packet/'actual_full49_model_increment_closure.json',closure)
rows=[]
for recipe in record['recipes']:
    metric=recipe['replayed_metrics'];v=metric['validation_only_control'];p=metric['paper_retrospective']
    rows.append(dict(canonical_id=identity,training_seed=1107,**recipe['recipe'],validation_points=3935,test_points=72000,
        validation_only_noPA_f1=v['f1'],validation_only_noPA_precision=v['precision'],validation_only_noPA_recall=v['recall'],
        validation_only_threshold=v['threshold'],ROC=p['ROC'],AP=p['AP'],test_label_optimized_BestF1=p['Best_F1'],
        BestF1_is_diagnostic_only=True,full_original_recipe_and_native_replay_complete=True))
with (packet/'all10_full_scoring_recipe_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
shutil.copyfile(tests,packet/'validation_51_checks.xml');shutil.copyfile(__file__,packet/'closure_source.py')
active=[]
for pid,ct in [(36292,1792136499.03037),(28836,1792136662.9399016),(34576,1792136313.8145552),
    (1784,1792136313.8267477),(36912,1792097263.1796665),(41992,1791948094.8603492)]:
    p=psutil.Process(pid);assert p.is_running() and p.create_time()==ct
    active.append(dict(pid=pid,create_time=ct,command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2])))
spectral=read(ROOT/'experiments/runs/fm_spectral_full70_generation_evaluator_reexecution_v1/status.json')
assert spectral['active_pid']==36292 and not spectral['failed_preserved']
write(packet/'actual_live_workers_and_Spectral_full_reexecution.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    handles=active,Spectral_actual_state=spectral))
baseline=rows[0]
(packet/'README.md').write_text(f'''# GRASP主模型 SWAN SF seed1107：第49个完整核验模型

原定1500轮训练、93000次更新、69244参数和全部10种原始评分设置完成。
每个设置均覆盖3935个完整验证点和72000个完整测试点；原始检查点重新恢复后完整重推理，独立复核全部数组坐标、前向次数、预算和指标。
原生重放和独立评审均成功退出0，无资源中止。重新核对原结果、检查点、所有源产物及原始/重放评分文件SHA256。
保留原score rtol=0.0002/atol=0.001及全部资源守卫，不声明CPU/GPU评分逐字节相同。
原模板文字继承了tau1/seed1104描述，实际结构化canonical、主模型profile、seed、权重和10项评分网格均明确绑定main/seed1107；不改写冻结原件。

默认5个源样本×10次流评估：验证集99%阈值、无PA的F1={baseline['validation_only_noPA_f1']:.9f}，Precision={baseline['validation_only_noPA_precision']:.9f}，Recall={baseline['validation_only_noPA_recall']:.9f}。
ROC={baseline['ROC']:.9f}，AP={baseline['AP']:.9f}。测试标签选阈值的BestF1={baseline['test_label_optimized_BestF1']:.9f}仅列为事后诊断。
CSV保留全部10项原评分设置，不选取测试表现最佳设置充当固定主结果。

49个完整条件恰好等于此前48个条件加本模型。主模型SWAN SF当前仅完成5/10原始训练种子1103..1107，1108..1112仍必需，不汇报完整十种子均值/区间。
全部9120实体模型、480训练组及10种子范围保留。原始论文等价与完整TAB认证均未声称。
51项相关回归检查通过。归档不产生额外模型推理或优化更新。
Spectral完整生成与学习式评估器重跑当前实际进程36292已执行；其完成数另行等待守卫及完整输出，不能从进程存活计作完成。
原45篇论文/681方法基线消融/273数据任务/702标量目标及其他未完成实验继续保留，整体任务未完成。
''',encoding='utf-8')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/GRASP_main1107_full49.2026-10-16_v1.json';assert not ara.exists()
write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),new_full_original_model=identity,
    full_original_models_closed=49,required_models=9120,required_training_seeds=10,all_original_experiments_complete=False))
def head(path):return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
manifest=ROOT/'tmp/GRASP_main1107_full49_publication_manifest_20261016_v1.json'
files=[ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
write(manifest,dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(files)))
print(json.dumps(dict(packet=str(packet),files=len(files),complete_models=49,recipes=10,default_recipe_metrics=baseline)))
