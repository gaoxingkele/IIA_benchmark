"""Archive actual fixed-score protocol results and preserve every prior failure."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import audit_timesnet_NASA_label_protocol_v2 as s

ROOT,read,sha=s.ROOT,s.read,s.sha
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
c=read(ROOT/s.CONFIG);jobs=s.verify(c);private=s.private_module()
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/TimesNet_actual2_fixed_score_NASA_protocol_audits_v2'
folder.mkdir(parents=True,exist_ok=False)
xml=ROOT/'tmp/TimesNet_all5_NASA_label_protocol_v2_validation_20261016_v2.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==32
assert all(int(x.get(k,0))==0 for x in suites for k in ['failures','errors','skipped'])
sources=dict(frozen_v1_configuration=ROOT/s.original.CONFIG,frozen_v2_configuration=ROOT/s.CONFIG,
    original_v1_implementation=Path(s.original.__file__),original_v2_implementation=ROOT/(s.MODULE.replace('.','/')+'.py'),
    original_v1_tests=ROOT/s.original.TEST,original_v2_tests=ROOT/s.TEST,validation_32_passed=xml)
state=read(ROOT/c['state_root']/'status.json');owner=psutil.Process(state['pid'])
assert owner.create_time()==state['create_time'] and state['seed']==1105 and state['active_pid'] is None
old=read(ROOT/s.original.CONFIG);old_state=read(ROOT/old['state_root']/'status.json')
old_owner=psutil.Process(old_state['pid']);assert old_owner.create_time()==old_state['create_time']
write(folder/'current_v2_controller_status.json',state);write(folder/'preserved_live_v1_metadata_failure_status.json',old_state)
sources['actual_v2_owner_launch']=ROOT/c['state_root']/'launch.json'
sources['preserved_v1_owner_launch']=ROOT/old['state_root']/'launch.json'
rows=[];mapped=[]
for seed in [1103,1104]:
    job=next(j for j in jobs if j['seed']==seed);gate=s.metadata_failure(c,job)
    assert gate['no_original_worker_or_output_created'] and gate['case_failure']==s.EXPECTED_ERROR
    base=ROOT/c['state_root']/'attempts'/job['id'];output=Path(c['artifact_root'])/job['id']
    p=output/'full_label_protocol_audit.json';d=read(p);g=read(base/'guard_receipt.json')
    assert d['passed'] and d['job']==job and d['configuration_sha256']==sha(ROOT/s.CONFIG)
    assert g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted'] and g['proof_sha256']==sha(p)
    assert d['source_binding']==private.source_native(c,job)
    assert d['original_metrics_exactly_recomputed'] and d['full_entities']==51 and d['full_test_points']==408160
    assert not d['model_training_or_inference_executed'] and not d['all_raw_label_controls_evaluable'] and d['label_controls']=={}
    crosswalk=d['raw_CSV_crosswalk'];exact=[r for r in crosswalk if r['full_mapping_proven']];unmapped=[r for r in crosswalk if not r['full_mapping_proven']]
    assert len(exact)==50 and len(unmapped)==1 and unmapped[0]['entity']=='SMAP_G-7'
    assert unmapped[0]['candidates'][0]['maximum_finite_absolute_difference']==5.551115123125783e-17
    protocol=d['baseline_protocol_diagnostics'];oracle=d['original_test_calibrated_oracle_grid']
    assert protocol['test_points']==408160 and protocol['anomaly_points']==51269
    rows.append(dict(seed=seed,validation_only_noPA_F1=protocol['validation_only_noPA']['f1'],
        same_validation_threshold_entitywise_PA_F1=protocol['same_validation_threshold_entitywise_PA_diagnostic']['f1'],
        test_label_selected_noPA_F1_grid=oracle['best_grid_f_score'],test_label_selected_PA_F1_grid=oracle['best_grid_adjust_f_score'],
        always_anomaly_noPA_F1=protocol['always_anomaly_noPA_F1'],fixed_validation_threshold=protocol['validation_threshold'],
        exact_raw_coordinate_entities=50,required_raw_coordinate_entities=51,full_raw_label_controls_skipped=True,
        paired_training_seed_cohort_complete=False))
    mapped.append(dict(seed=seed,unmapped=unmapped[0],pre_worker_metadata_recovery=gate,
        original_proof_path=str(p),original_proof_sha256=sha(p),guard_receipt_sha256=sha(base/'guard_receipt.json')))
    for label,path in [('full_protocol_and_crosswalk_proof',p),('successful_original_guard',base/'guard_receipt.json'),('exact_worker_launch',base/'launch.json')]:
        sources[f'seed{seed}/'+label]=path
for label,path in sources.items():
    target=folder/(label+path.suffix);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);assert sha(path)==sha(target)
failures=[]
for name,path,expected in [
    ('pre_registration_v1','tmp/TimesNet_all5_NASA_label_protocol_pre_registration_failure_20261016_v1.xml',2),
    ('pre_registration_v2','tmp/TimesNet_all5_NASA_label_protocol_v2_pre_registration_failure_20261016_v1.xml',1)]:
    target=folder/name/'preserved_pre_registration_validation_failure.xml';target.parent.mkdir()
    shutil.copyfile(ROOT/path,target)
    suites=list(ET.parse(target).getroot().iter('testsuite'));assert sum(int(x.get('failures',0)) for x in suites)==expected
    failures.append(target.relative_to(ROOT/'projects/flow_matching_research').as_posix())
write(folder/'actual_protocol_comparison.json',dict(models=rows,full_original_seed_cohort=[1103,1104,1105,1106,1107],
    actually_audited_original_seeds=[1103,1104],required_original_models=5,all5_complete=False,
    all_scores_and_model_weights_unchanged=True,no_threshold_or_label_policy_selected_by_performance=True,
    new_model_inference_executed=False,all_original_experiments_complete=False))
write(folder/'strict_mapping_and_recovery_evidence.json',mapped)
inventory=read(ROOT/'projects/flow_matching_research/results/2026-10-16/original_raw_entity_coverage_inventory_v1/actual_original_entity_coverage.json')
lengths=[]
for dataset in ['SMAP','MSL']:
    raw={r['metadata']['chan_id']:r for r in inventory['original_label_rows'] if r['metadata']['spacecraft']==dataset}
    for csv in inventory['registered_CSV_files']:
        if csv['dataset']!=dataset or csv['role']!='test':continue
        row=raw[csv['entity']];n=row['raw']['test']['shape'][0]
        if n!=csv['physical_data_lines']:
            lengths.append(dict(dataset=dataset,entity=csv['entity'],full_original_test_points=n,
                current_CSV_test_points=csv['physical_data_lines'],point_count_difference=n-csv['physical_data_lines'],
                original_anomaly_sequences=row['metadata']['anomaly_sequences'],
                exact_row_and_label_equivalence_not_claimed=True))
assert {(r['dataset'],r['entity'],r['point_count_difference']) for r in lengths}=={('SMAP','G-7',4014),('MSL','T-13',1215)}
write(folder/'shared_entity_test_length_discrepancies.json',lengths)
(folder/'README.md').write_text('''# TimesNet 两个完整模型：同一分数下的协议差异已实际核验

原 seed1103、1104 均已先完成 51 实体、153 角色的全模型逐字节重推理，本次只复核保存分数、标签映射及指标口径，没有新训练或新推理。两个实际审计 worker 均成功退出，无资源保护中止。原始点、事件、bootstrap 和测试校准网格指标全部精确重算一致。

|原种子|验证集固定阈值，无 PA|同一阈值，按实体 PA 诊断|测试标签选择网格最优，无 PA|测试标签选择网格最优，PA|
|---|---:|---:|---:|---:|
|1103|0.045308258|0.758657700|0.145187237|0.777473435|
|1104|0.041533141|0.756800864|0.137882270|0.776857043|

左边两列使用完全相同的分数与同一个验证阈值，仅切换 PA 就产生约 0.71 的绝对 F1 差异。右边两列使用原已披露的 train/test 分数校准与测试标签挑网格，保留为诊断。PA 不跨实体边界；不把这些诊断当作严格主指标。测试点 408160、异常点 51269，异常占比 12.561%；永远预测异常的无 PA F1 为 0.223185737。该基线仅帮助解释 F1 与异常占比，不代替方法成绩。

这是固定本地子集、现有参数和两个原种子的协议对照；完整五种子仅完成 2/5，不提供部分种子的算法优劣区间，也不认证论文历史端到端等价。

NASA 原始标签替换检查要求每一个数值坐标精确一致且标签映射明确。50/51 个实体通过；G-7 有最大 5.551115123125783e-17 的输入差异。未放宽精度门槛、猜测偏移、丢弃 G-7 或改成 50 实体成绩，因此四个完整原始标签视图仍未计算，不能称这部分已完成。标签对应失败和每个候选映射逐项保留。后续须查明数值转录来源，再另行建立有依据的对照，不修改当前冻结规则。

额外的测试长度差异也已精确定位：G-7 原始 8029 点而当前 CSV 4015 点，相差 4014；MSL T-13 原始 2430 点而 CSV 1215 点，相差 1215。这里只证明文件长度差异，不仅凭长度推断时段或标签完全对应。完整 109 实体准备及 873 个新数据任务继续保留。

修复过程全部留痕：注册前测试发现全局可变策略列表会被测试修改，修成不可变常量后原 19 项通过；v1 运行时只有元数据检查报 complete_result 缺失，未创建任何数值 attempt/output。独立 v2 只在新进程的资源 facade 中补齐固定源码的只读检查函数，数值 worker、映射和指标字节码保持不变；仅允许有此确切失败且无旧数值尝试的病例进入。v2 的补齐函数依赖测试失败也保留；最终 32 项测试全部通过。原 v1 进程及失败记录未覆盖或停止，v2 正等待后续三个原模型完成完整重推理。

当前研究目标仍未完成。所有原论文、作者专用任务、完整数据实验、消融、独立重推理和完整 TAB 范围指标继续逐项执行。

来源：[TimesNet](https://arxiv.org/abs/2210.02186)、[Telemanom](https://github.com/khundman/telemanom)、[TAB](https://github.com/decisionintelligence/TAB)。
''',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\npre_registration_v1/preserved_pre_registration_validation_failure.xml whitespace=-blank-at-eol\npre_registration_v2/preserved_pre_registration_validation_failure.xml whitespace=-blank-at-eol\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),actual_closed_audits=2,
    required_original_models=5,final_tests_passed=32,numerical_worker_retried=False,full_raw_label_controls_complete=False,
    files_sha256={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()},all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/TimesNet_actual2_NASA_fixed_score_protocol_audits.2026-10-16_v2.json'
write(ara,dict(actual_audited_original_models=2,required_original_models=5,evidence=folder.relative_to(ROOT).as_posix(),
    full51raw_label_mapping_complete=False,all_original_experiments_complete=False))
paths=[s.original.CONFIG,s.CONFIG,s.original.TEST,s.TEST,
    Path(s.original.__file__).relative_to(ROOT).as_posix(),s.MODULE.replace('.','/')+'.py',ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/TimesNet_actual2_NASA_protocol_publication_manifest_20261016_v2.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths),raw_evidence_whitespace_exceptions=failures))
print(json.dumps(dict(actual_original_models_audited=2,required=5,protocol_comparison=rows,strict_label_mapping_complete=False,
    actual_controller=state['pid'],tests_passed=32,exact_files=len(paths))))
