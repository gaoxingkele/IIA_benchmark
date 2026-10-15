"""Archive actual96input comparisons and all8full51entity label views."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import audit_NASA_split_representation_v1 as split
from scripts.flow_matching import audit_timesnet_declared_CSV_label_control_v1 as control

ROOT,read,sha=split.ROOT,split.read,split.sha
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sc=read(ROOT/split.CONFIG);split.verify(sc)
sp=Path(sc['artifact_root'])/split.REPORT;sg=ROOT/sc['state_root']/'guard_receipt.json'
proof,guard=read(sp),read(sg)
assert proof['passed'] and proof['configuration_sha256']==sha(ROOT/split.CONFIG) and proof['declared_comparisons_executed']==96
assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['proof_sha256']==sha(sp)
state=read(ROOT/sc['state_root']/'status.json');assert state['state']=='two_split_NASA_serialization_audit_passed' and state['active_pid'] is None
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/NASA_split_roles_and_actual2_full_label_controls_v1'
folder.mkdir(parents=True,exist_ok=False)
sources=dict(actual96representation_audit=sp,successful_representation_guard=sg,
    representation_terminal_status=ROOT/sc['state_root']/'status.json',frozen_representation_configuration=ROOT/split.CONFIG,
    original_representation_source=ROOT/(split.MODULE.replace('.','/')+'.py'),original_representation_tests=ROOT/split.TEST)
partitions=[];array_hashes=0
for result in proof['results']:
    for f in result['representation_files'].values():assert sha(f['path'])==f['sha256'];array_hashes+=1
    candidates=result['complete_raw_TEST_partitions'];assert len(candidates)==1
    p=candidates[0]
    assert p['profile']==control.PROFILE and p['timestamp_origin']==1 and p['every_original_test_row_once'] and p['overlap_rows']==0
    partitions.append(dict(entity=result['entity'],**p,CSV_role_summaries={k:dict(rows=v['rows'],timestamp_first=v['timestamp_first'],
        timestamp_last=v['timestamp_last'],provided_labels=v['provided_labels'],provided_anomaly_points=v['provided_anomaly_points'],
        exact_matches=[{x:r[x] for x in ['raw_role','timestamp_origin','profile','label_policy_exact_matches']} for r in v['candidates'] if r['every_coordinate_equal']]) for k,v in result['CSV_roles'].items()}))
assert array_hashes==16
cc=read(ROOT/control.CONFIG);jobs=control.verify(cc);private=control.private_module();control.representation_evidence()
sources.update(frozen_label_control_configuration=ROOT/control.CONFIG,
    original_label_control_source=ROOT/(control.MODULE.replace('.','/')+'.py'),original_label_control_tests=ROOT/control.TEST)
owner_state=read(ROOT/cc['state_root']/'status.json');owner=psutil.Process(owner_state['pid'])
assert owner.create_time()==owner_state['create_time'] and owner_state['active_pid'] is None and owner_state['seed']==1105
sources['current_label_control_owner_status']=ROOT/cc['state_root']/'status.json'
sources['actual_label_control_owner_launch']=ROOT/cc['state_root']/'launch.json'
rows=[];endpoints=[];label_hashes=0
for seed in [1103,1104]:
    job=next(j for j in jobs if j['seed']==seed);output=Path(cc['artifact_root'])/job['id'];attempt=ROOT/cc['state_root']/'attempts'/job['id']
    p=output/'full_label_protocol_audit.json';d=read(p);g=read(attempt/'guard_receipt.json')
    assert d['passed'] and d['job']==job and d['configuration_sha256']==sha(ROOT/control.CONFIG)
    assert g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted'] and g['proof_sha256']==sha(p)
    assert d['source_binding']==private.source_native(cc,job) and d['full_entities']==51 and d['full_test_points']==408160
    assert d['all_raw_label_controls_evaluable'] and len(d['label_controls'])==4
    assert all(r['full_mapping_proven'] for r in d['raw_CSV_crosswalk'])
    assert all('source_first_inclusive' in r['original_CSV_label_policy_matches'] for r in d['raw_CSV_crosswalk'])
    g7=next(r for r in d['raw_CSV_crosswalk'] if r['entity']=='SMAP_G-7')
    assert not g7['old_direct_mapping_proven'] and g7['declared_representation_profile']==control.PROFILE
    assert all(r['old_direct_mapping_proven'] for r in d['raw_CSV_crosswalk'] if r['entity']!='SMAP_G-7')
    for policy,v in d['label_controls'].items():
        assert sha(v['labels_path'])==v['labels_sha256'];label_hashes+=1
        assert not v['label_policy_selected_by_performance']
        b=v['protocol_diagnostics']
        rows.append(dict(seed=seed,policy=policy,anomaly_points=b['anomaly_points'],validation_threshold=b['validation_threshold'],
            validation_only_noPA_F1=b['validation_only_noPA']['f1'],same_validation_threshold_entitywise_PA_F1=b['same_validation_threshold_entitywise_PA_diagnostic']['f1'],
            conditional_noPA_entity_CI95=v['metrics']['strict']['1']['f1_ci95'],unchanged_original_score_SHA256=v['same_scores_sha256'],
            targets_file=v['labels_path'],targets_SHA256=v['labels_sha256'],complete_51entity_timestamp_coverage=True))
    a=d['label_controls']['source_first_inclusive']['protocol_diagnostics'];b=d['label_controls']['source_first_exclusive']['protocol_diagnostics']
    assert a['anomaly_points']-b['anomaly_points']==64
    endpoints.append(dict(seed=seed,changed_target_points=64,
        noPA_F1_delta=a['validation_only_noPA']['f1']-b['validation_only_noPA']['f1'],
        PA_F1_delta=a['same_validation_threshold_entitywise_PA_diagnostic']['f1']-b['same_validation_threshold_entitywise_PA_diagnostic']['f1'],
        PA_true_positive_delta=a['same_validation_threshold_entitywise_PA_diagnostic']['true_positives']-b['same_validation_threshold_entitywise_PA_diagnostic']['true_positives']))
    for name,path in [('full_label_control_proof',p),('successful_guard',attempt/'guard_receipt.json'),('exact_worker_launch',attempt/'launch.json')]:sources[f'seed{seed}/'+name]=path
validation=ROOT/'tmp/NASA_split_and_declared_label_control_validation_20261016_v1.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'));assert sum(int(x.get('tests',0)) for x in suites)==23
assert all(int(x.get(k,0))==0 for x in suites for k in ['failures','errors','skipped'])
sources['validation23_checks']=validation
for name,path in sources.items():
    dest=folder/(name+path.suffix);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest);assert sha(path)==sha(dest)
write(folder/'actual_split_partition_summary.json',dict(entities=partitions,actual_all96comparisons_passed=True,
    representation_arrays_rehashed=16,no_source_file_modified=True,no_numerical_tolerance_used=True,
    current_transform_compatibility_proven=True,historical_writer_or_software_version_certified=False))
write(folder/'actual_all8_label_view_results.json',dict(actual_original_seeds=[1103,1104],required_original_seeds=[1103,1104,1105,1106,1107],
    views=rows,endpoint_sensitivity=endpoints,all_original_scores_and_validation_thresholds_unchanged=True,
    existing_CSV_labels_exactly_match_first_inclusive_all51entities=True,old_direct_float64_gate_unchanged=True,
    new_condition_is_named_exact_representation_control=True,historical_generator_certified=False,
    original_required_five_seed_cohort_complete=False,all_original_experiments_complete=False))
(folder/'README.md').write_text('''# 两个 NASA 通道的拆分已逐值还原；两个原模型的四标签视图已完整计算

本次实际执行两通道、六个 train/val/test CSV、四个固定表达转换、两种原始角色及零/一基时间戳，共 96 项比较。原文件未修改，全部 16 个表达数组和本次 8 个标签覆盖文件当前哈希已重新核验，23 项测试通过。数据与指标 worker 均成功退出且无资源中止，没有新增训练或模型推理。

|通道|原始 TEST 点数|现有 val|现有 test|核对结果|
|---|---:|---:|---:|---|
|SMAP G-7|8029|4014|4015|两段按时间先后、不重叠，恰好覆盖原始 TEST|
|MSL T-13|2430|1215|1215|两段按时间先后、不重叠，恰好覆盖原始 TEST|

两者 val/test 数值只与固定的“原始数组经 Pandas CSV 写出，再用默认 C 解析器读入”表达逐坐标精确相等；所有其他声明候选也保留，没有加 epsilon、猜测偏移或按模型成绩挑转换。train 则匹配原始 TRAIN，未发现这两个通道的训练段由原始 TEST 拼成。官方 mTSBench 说明会留出验证序列用于模型选择；当前精确数据分区与这一用途相容，但未知历史生成代码/软件版本仍不认证。

因此，G-7 的约 5.55e-17 差异在当前软件环境下可由一个固定 CSV 读写过程重现。此前“直接原值精确映射”的失败结果和门槛保持不变。本次另立命名表达对照：G-7 只在标签映射时使用上述转换，其他 50 个实体保持直接值；原模型输入、权重、分数、阈值、指标公式均不改。不是把旧检查放宽到近似相等，也不将表达兼容性当作已证明历史流程。

|原种子|标签口径|无 PA F1|同验证阈值的 PA 诊断 F1|
|---|---|---:|---:|
|1103|首条，不含右端点|0.045216452|0.742030396|
|1103|首条，含右端点|0.045308258|0.758657700|
|1103|末条，不含右端点|0.045951307|0.741556330|
|1103|末条，含右端点|0.046042440|0.758233833|
|1104|首条，不含右端点|0.041472566|0.740073928|
|1104|首条，含右端点|0.041533141|0.756800864|
|1104|末条，不含右端点|0.042169112|0.739582846|
|1104|末条，含右端点|0.042228994|0.756360619|

每个视图均覆盖原 51 实体、408160 个测试点。现有 CSV 标签逐点等于“首条、含右端点”；P-2 的末条标注则构成单独对照，不按 F1 选择。仅端点改变 64 个目标点，无 PA F1 差异不足 0.01 个百分点；按同一阈值做 PA 后，差异约 1.66–1.67 个百分点。PA 会放大标签边界影响。完整条件实体簇区间、事件指标和原测试阈值网格也保留，不能当作五个训练种子的区间或严格主榜指标。

这两种子中，不同标签口径造成的无 PA 变化较小；PA 与无 PA 的口径差异明显更大。这里只是当前固定本地子集上的对照，不认证原文 SOTA 或完整 TAB 训练复现。五种子仍只完成 2/5，后续三个种子由既有控制进程在原模型完整重推理后自动评估。完整 109 实体原始数据、873 项本地方法/消融及所有作者专用和其他论文实验仍待相应证据完成。

来源：[mTSBench 官方说明](https://github.com/PLAN-Lab/mTSBench#-model-selection)、[Telemanom](https://github.com/khundman/telemanom)、[TimesNet](https://arxiv.org/abs/2210.02186)。
''',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),actual_representation_comparisons=96,
    actual_full51entity_label_views=8,actual_original_model_seeds=2,required_original_model_seeds=5,
    representation_array_hashes_verified=16,label_array_hashes_verified=label_hashes,tests_passed=23,
    files_sha256={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()},all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/NASA_split_roles_and_actual2_label_controls.2026-10-16_v1.json'
write(ara,dict(actual96comparisons_complete=True,actual_original_models=2,required_original_models=5,
    actual_full51entity_label_views=8,evidence=folder.relative_to(ROOT).as_posix(),all_original_experiments_complete=False))
paths=[split.CONFIG,control.CONFIG,split.MODULE.replace('.','/')+'.py',control.MODULE.replace('.','/')+'.py',split.TEST,control.TEST,ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/NASA_split_and_actual2_label_controls_publication_manifest_20261016_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(actual_comparisons=96,actual_full51entity_label_views=8,original_models=2,tests_passed=23,
    exact_files=len(paths),endpoint_sensitivity=endpoints)))
