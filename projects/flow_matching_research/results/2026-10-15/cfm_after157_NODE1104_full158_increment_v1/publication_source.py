"""Publish full canonical union and the completed ten-epoch NODE baseline."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT,read,sha,write

folder=ROOT/'projects/flow_matching_research/results/2026-10-15/cfm_after157_NODE1104_full158_increment_v1'
catalog_path=folder/'actual_current_deduplicated_complete_original_models.json';catalog=read(catalog_path)
assert catalog['full_unique_canonical_models']==len(catalog['deduplicated_full_records'])==158
assert len({r['original_canonical_id'] for r in catalog['deduplicated_full_records']})==158
assert catalog['required_original_models']==285 and catalog['previous157_records_identical']
assert catalog['all_current_native_independent_proof_chains_and_artifacts_rehashed']
assert len(catalog['new_full_models'])==1
name='cfm_ts_original__appendix_d__node_e10__sine__seed1104'
row=catalog['new_full_models'][0]['record'];source=read(ROOT/row['result_path'])
assert row['original_canonical_id']==name and row['execution_id']==name
assert row['epochs']==source['epochs_completed']==10 and row['optimizer_updates']==source['optimizer_updates']==20
assert row['full_replay_shape']==source['test_prediction_shape']==[80,500,1]
assert row['bitwise_identical'] and row['max_absolute_difference']==0.
assert source['parameter_count']==150865 and source['training_seconds']>0 and source['score_seconds']>0
assert len(source['training_losses'])==10 and not source['diagnostic_smoke']
assert source['job']['training_items']==320 and source['test_used_for_training_or_selection'] is False
for artifact in source['artifacts']:assert sha(artifact['path'])==artifact['sha256']
for label,digest in catalog['new_full_models'][0]['current_source_proof_hashes'].items():
    assert sha(folder/name/(label+'.json'))==digest
xml=ROOT/'tmp/CFM_NODE_Sine1104_full158_validation_20261015_v1.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==42
assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
shutil.copyfile(xml,folder/'validation_42_checks.xml')
source_summary=dict(canonical_id=name,registered_task='appendix_d/sine/held_out_complete_trajectories',
    original_training_data_model_seed=1104,training_trajectories=320,epochs=10,optimizer_updates=20,
    model_parameters=150865,test_shape=[80,500,1],full_heldout_test_grid_points=40000,
    metrics=source['metrics'],full_training_loss_history=source['training_losses'],
    source_training_seconds=source['training_seconds'],source_training_hours=source['training_seconds']/3600,
    source_scoring_seconds=source['score_seconds'],CPU_threads=2,
    all_saved_prediction_elements_equal_after_native_reinference=True,native_max_absolute_difference=0.,
    frozen_native_prediction_tolerance=dict(rtol=1e-7,atol=1e-8),independent_complete_original_grids_verified=True,
    checkpoint_persistence=dict(weights=True,training_history=True,preprocessing=True,update_count=True,
        full_optimizer_state=False,training_RNG=False),
    checkpoint_boundary='Original canonical checkpoint supports model replay; it does not contain a full resumable optimizer/RNG training state.',
    source_result_sha256=sha(ROOT/row['result_path']),source_checkpoint_sha256=row['checkpoint_sha256'],
    paired_original_seeds_required=[1103,1104,1105,1106,1107],cohort_ready_for_five_seed_statistics=False,
    trajectory_STD_is_not_between_model_STD=True,source_reference_trajectories_used_for_training_or_selection=False,
    historical_author_equivalence_or_strict_TAB_result_certified=False,all_original_experiments_complete=False)
write(folder/'full_budget_training_cost_and_metrics.json',source_summary)
with (folder/'README.md').open('a',encoding='utf-8') as stream:
    stream.write('''

## 新增NODE1104的预算、耗时与状态边界

原神经ODE/Sine/Appendix D种子1104实际完成320条训练轨迹、10个epoch和20次优化更新，150,865参数模型覆盖80条留出轨迹×500时间点×1观测维，共40,000个原测试网格点。原训练68856.05875秒（约19.12668小时），评分4.53747秒；长耗时发生在训练阶段。未因等待较久重启、减小预算、更改求解器或放松容差。具体运算热点未另做侵入式剖析。

|MSE|MAE|RMSE|80轨迹MSE描述性样本STD|
|---:|---:|---:|---:|
|0.11929626419875178|0.2526033776306255|0.34539291278014345|0.16658710331504847|

CPU原生检查点重推理与原保存预测逐元素相等，最大绝对差0；原先冻结rtol1e-7/atol1e-8保持。第二独立进程复核全部原始网格、轨迹身份、真值、完整指标和当前文件哈希。原训练、原生复核、独立检查均有实际成功退出且无资源中止的保护证明。

原checkpoint保留权重、预处理、10轮loss和更新计数，足以支持这次模型重推理；它没有完整优化器或训练RNG状态，因此未认证训练可从任意中间步骤精确续跑。80轨迹STD描述单个已拟合模型的轨迹差异，不是五个配对数据/模型种子的置信区间。本组1103..1107仍全部要求，不能用目前两项代替五种子汇总。

跨三个重叠快照按canonical ID去重后确认为158/285；单个收集器155不能当作全项目总数，也不能把三份快照相加。原157项保持相同，所有现有证明链、数据及模型/预测产物重新核对。已观察到的GP600失败种子和未定义五种子均值继续保留。42项相关检查通过，大型原产物留在原目录；本次归档未新增训练或重复模型重推理。
''')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=42,
    unique_full_original_models=158,required_original_models=285,new_full_canonical_models=1,
    original_full_predictions_native_bitwise_equal=True,
    files_sha256={f.relative_to(folder).as_posix():sha(f) for f in sorted(folder.rglob('*')) if f.is_file()},all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/CFM_NODE_Sine1104_full158_closed.2026-10-15_v1.json'
write(ara,dict(method='Neural ODE',task='CFM-TS Appendix D Sine',seed=1104,epochs=10,updates=20,
    evidence=folder.relative_to(ROOT).as_posix(),complete_native_and_independent_review=True,
    unique_full_canonical_models=158,required_original_models=285,
    complete_five_paired_seed_cohort=False,full_optimizer_RNG_resume_state_available=False,all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix(),
    f'configs/reproducibility/fm_cfm_completion_replay_queue_v4/{name}.native.json',
    f'configs/reproducibility/fm_cfm_completion_replay_queue_v4/{name}.review.json'];paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/CFM_NODE_Sine1104_full158_publication_manifest_20261015_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(new_canonical_model=name,unique_full_models=158,required_original_models=285,
    tests_passed=42,MSE=source['metrics']['MSE'],training_seconds=source['training_seconds'],exact_files=len(paths))))
