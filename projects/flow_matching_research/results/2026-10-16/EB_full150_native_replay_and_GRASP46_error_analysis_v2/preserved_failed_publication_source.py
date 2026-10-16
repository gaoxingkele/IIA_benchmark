"""Publish actual EB full-model replay and a frozen one-seed anomaly error audit."""
import csv
from datetime import datetime,timezone
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import psutil
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching import audit_otcfm_EB_CITE_full150_replay_v1 as audit
from scripts.flow_matching import run_otcfm_EB_CITE_native150_v2 as execution
from scripts.flow_matching.otcfm_EB_CITE_result_integrity_v1 import inspect,full_metric_names

packet=ROOT/'projects/flow_matching_research/results/2026-10-16/EB_full150_native_replay_and_GRASP46_error_analysis_v1'
config=read(ROOT/audit.CONFIG);q=audit.verify(config);job=q['jobs'][0]
source,provenance=inspect(q,job);key=Path(job['artifact_directory']).name
report_path=Path(config['artifact_root'])/key/'full_checkpoint_replay.json'
guard_path=ROOT/config['state_root']/'attempts'/key/'guard_receipt.json'
r,g=read(report_path),read(guard_path)
assert g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted'] and g['full_replay_report_sha256']==sha(report_path)
assert r['status']=='passed' and r['id']==job['id'] and r['full_native_test_reinference_passed']
assert r['full_training_epochs']==1000 and r['full_training_optimizer_updates']==4000
assert r['selected_checkpoint_epoch']==340 and r['selected_checkpoint_updates']==1360
assert r['source_total_and_selected_optimizer_counters_separately_verified']
assert r['all_learned_tensors_and_optimizer_moments_checked']
assert set(r['native_metrics'])==full_metric_names(5) and len(r['native_metrics'])==44
assert r['native_metrics']==r['original_metrics']==source['test_metrics']
assert r['native_metric_tolerance']==dict(rtol=2e-6,atol=2e-6) and r['extra_model_training_updates']==0
solves=r['full_test_population_distances_independently_recomputed'][0]['solves']
assert len(solves)==6 and all(row['native_LP_result_code']==1 for row in solves)
w1=[row for row in solves if row['power']==1]
assert len(w1)==3 and all(row['independent_LP_result_code']==1 and row['full_costs_independently_recomputed']
    and row['objective_absolute_difference']<=row['LP_objective_perturbation_bound'] for row in w1)
assert r['complete_interior_timepoint_cost_shapes']==[[2656,4374],[4374,3650],[3650,3873]]
for item in r['all_full_test_arrays_and_artifacts']:
    p=Path(item['path']);assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes']
    if p.suffix=='.npy':assert np.isfinite(np.load(p,allow_pickle=False)).all()
grasp=ROOT/'projects/flow_matching_research/results/2026-10-16/grasp_tau4_seed1106_full46_model_increment_v1'
closure=read(grasp/'actual_full46_model_increment_closure.json')
assert closure['actual_current_unique_complete_models']==46 and closure['complete_original_training_and_full_native_replay_and_independent_recipe_review']
assert closure['epochs']==1500 and closure['optimizer_updates']==93000 and closure['parameter_count']==69244
assert len(closure['current_complete_original_ids'])==len(set(closure['current_complete_original_ids']))==46
assert closure['complete_Tau4_Swan_seeds']==[1103,1104,1105,1106] and not closure['full_ten_seed_Tau4_cohort_complete']
diagnostic=read(grasp/'paired_Tau1_Tau4_single_seed_error_diagnostic.json')
assert diagnostic['only_registered_model_parameter_difference']=='tau:1.0 ->4.0'
assert diagnostic['all_test_labels_and_test_and_validation_row_IDs_equal']
for row in diagnostic['records']:
    assert sha(row['replay_score_path'])==row['replay_score_sha256'] and row['test_points']==72000
    assert row['positive_test_points']==20100 and row['validation_points']==3935
    assert sum(row[x] for x in ('TP','FP','TN','FN'))==72000
states={}
for name,c in [('full150_source',q),('full150_replay',config)]:
    s=read(ROOT/c['state_root']/'status.json');p=psutil.Process(s['pid']);assert p.create_time()==s['create_time']
    states[name]=dict(status=s,exact_command=p.cmdline(),actual_python_children=[dict(pid=x.pid,create_time=x.create_time(),command=x.cmdline(),cpu_seconds=sum(x.cpu_times()[:2]))
        for x in p.children(recursive=True) if x.name().lower()=='python.exe'])
tests=[(Path('F:/aicoding/IIA_Data/experiments/otcfm_EB_CITE_full150_native_replay_scope_v1_tests.xml'),14),
       (Path('F:/aicoding/IIA_Data/experiments/GRASP_Tau4_seed1106_full46_closure_v1_tests.xml'),58)]
for path,count in tests:
    suites=list(ET.parse(path).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==count
    assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
packet.mkdir(parents=True,exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.py -text\n*.xml -text\n*.csv -text\n',encoding='utf-8',newline='\n')
for name,path in [('full_EB_native_checkpoint_replay',report_path),('full_EB_native_checkpoint_replay_guard',guard_path),
    ('full_EB_original_training_result',Path(provenance['result_path'])),('full_EB_original_training_guard',Path(provenance['guard_path'])),
    ('full150_replay_registration',ROOT/audit.CONFIG),('full150_replay_launch',ROOT/config['state_root']/'launch.json')]:
    shutil.copyfile(path,packet/(name+'.json'))
write(packet/'actual_live_processes.json',states)
write(packet/'full_EB_complete_replay_closure.json',dict(id=job['id'],full_training_epochs=1000,full_training_updates=4000,
    selected_checkpoint_epoch=340,selected_checkpoint_updates=1360,full_native_metrics_replayed=44,
    all44_source_and_replayed_metrics_exactly_equal=True,full_interior_cost_shapes=r['complete_interior_timepoint_cost_shapes'],
    full_independent_Euclidean_cost_entries=sum(x['full_cost_shape'][0]*x['full_cost_shape'][1] for x in w1),
    independent_full_W1_LP_results=w1,native_LP_result_codes=[x['native_LP_result_code'] for x in solves],
    original_primary_W1=source['primary_heldout_W1'],all_checkpoint_tensor_optimizer_and_source_artifacts_verified=True,
    all_native_replay_array_hashes_and_finiteness_rechecked=True,extra_training_updates=0,
    source_result_sha256=provenance['result_sha256'],replay_report_sha256=sha(report_path),replay_guard_sha256=sha(guard_path),
    source_validation_repair=job['validation_compatibility'],historical_paper_equivalence_certified=False,strict_TAB_result=False,
    all_original_experiments_complete=False))
with (packet/'paired_tau_single_seed_error_rows.csv').open('w',encoding='utf-8',newline='') as stream:
    writer=csv.writer(stream);writer.writerow(['tau','seed','TP','FP','FN','TN','validation_only_F1','native_test_label_optimized_F1','standard_no_epsilon_test_label_optimized_F1','validation_score_fraction_above_threshold','benign_test_fraction_above_validation_threshold'])
    for row in diagnostic['records']:writer.writerow([row[k] for k in ['tau','seed','TP','FP','FN','TN','validation_only_F1','native_test_label_optimized_F1','standard_no_epsilon_test_label_optimized_F1','validation_score_fraction_above_threshold','benign_test_fraction_above_validation_threshold']])
for index,(path,count) in enumerate(tests):shutil.copyfile(path,packet/f'validation_{index}.xml')
paths=[audit.MODULE.replace('.','/')+'.py','tests/test_fm_otcfm_EB_CITE_full150_replay_v1.py',audit.CONFIG]
for index,path in enumerate(paths):shutil.copyfile(ROOT/path,packet/f'source{index:02d}'+Path(path).suffix)
write(packet/'source_manifest.json',dict(files=[dict(path=p,sha256=sha(ROOT/p)) for p in paths],
    preserved_original120_replay=config['independent_source_replay_queue_preserved'],original_scope=q['original_scope']))
(packet/'README.md').write_text('''EB完整检查点复核与GRASP第46项闭环/单种子误差分析

已为全部150个EB/CITE模型登记并启动自动检查点复核，保留现有120个Multiome复核；原方法/两套完整训练预算/每个留出时间点/五训练种子均在。原生复测容差2e-6与所有资源门限未改，EB完整3个内部时间点全部检查44个标量，CITE完整2个内部时间点全部检查36个标量。

首项EB/I-CFM/released YAML/留出1/seed42已实际通过：1000epoch、4000训练更新，验证选择epoch340/1360次更新。完整源检查点、共享学习权重、优化器矩/计数、last检查点及原训练产物全部核验后原生重推理，44项标量与原结果逐项完全相同。三个完整代价矩阵形状为2656×4374、4374×3650、3650×3873，独立重算所有41718894个欧式距离并精确求解W1；6次原生W1/W2与3次独立双精度W1均返回OPTIMAL，目标差异受完整代价扰动界约束。完整数组/hash和退出0资源证据保留；原生留出W1为1.0388561487。EB验证专用266 batch的明确源修复及历史PCA/数据身份缺口不因复测通过而消失。

异常检测GRASP Tau4/Swan/swan_sf/seed1106完成1500epoch、93000更新、69244参数、原5源样本×10流评估的完整配方；3935验证点和72000测试点的源检查点重推理、所有前向次数与独立逐点/指标复核均通过，预测分歧0。源GPU与CPU分数不逐位相等，最大相对差约2.13e-7，所有原容差保留。现46/9120原任务闭环，480个原队列组和10种子要求保留，Tau4/Swan仅4/10种子，不能发布完整十种子结论。该GRASP实现由论文推导，作者代码等价性未认证。

配对seed1106的完整数据/标签/行ID一致，注册模型参数仅tau=1与4不同。固定验证99%阈值、无点调整下，Tau1的F1=0.525354，Tau4=0.508213。Tau4的TP15718/FP26038/FN4382/TN25862，正常测试误报率50.1696%，验证分数超阈值1.0165%；这是校准与正常测试分数分布不匹配的观测，尚不能只凭此归因于原始特征漂移、模型还是预处理约定。对同一固定分数，所有单阈值的标准F1最大值0.513967，原生回顾性指标为0.513963；原生公式分母带1e-5稳定项，约4.5e-6数值差已明确记录，原数字未改。测试标签在此用于事后误差分析及明确标记的回顾性指标，没有重选验证阈值、训练或增加种子。该单种子分析不是严格TAB或历史论文SOTA认证。

本轮14项EB/CITE复核范围测试及58项GRASP复核/流程测试通过。全部45论文/681方法基线消融/273数据任务/702论文标量目标仍未完成；不把本轮闭环、注册队列或部分种子替代原完整目标。
''',encoding='utf-8',newline='\n')
paths += [p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
paths += [p.relative_to(ROOT).as_posix() for p in grasp.rglob('*') if p.is_file()]
write(ROOT/'tmp/EB150_replay_GRASP46_error_analysis_publication_manifest_20261016_v1.json',dict(
    previous_root=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    previous_parent=subprocess.check_output(['git','-C',str(ROOT.parent/'flow-matching-research'),'rev-parse','HEAD'],text=True).strip(),paths=sorted(set(paths))))
print(dict(first_full_EB_replay=job['id'],full_EB_metrics=44,GRASP_closed=46,diagnostic_threshold_unchanged=True,exact_files=len(set(paths))))
