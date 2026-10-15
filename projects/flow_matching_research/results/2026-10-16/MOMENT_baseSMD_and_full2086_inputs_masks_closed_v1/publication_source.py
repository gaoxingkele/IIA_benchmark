from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil,subprocess,sys
import xml.etree.ElementTree as ET
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.flow_matching import prepare_original_full_entity_tsad_v2 as prep
from scripts.flow_matching import run_original_full_raw873_experiments_v2 as dispatch
from scripts.flow_matching import run_moment_full_checkpoint_replay_v1 as moment
from scripts.flow_matching import bind_moment_icm_full_training_inputs_v1 as inputs
from scripts.flow_matching import audit_moment_icm_full_training_masks_v1 as masks

ROOT,read,sha=prep.ROOT,prep.read,prep.sha
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def copy(path,dest):
    Path(dest).parent.mkdir(parents=True,exist_ok=True);assert not Path(dest).exists()
    shutil.copyfile(path,dest);assert sha(path)==sha(dest)
def passed(config,report):
    p=Path(config['artifact_root'])/report;g=ROOT/config['state_root']/'guard_receipt.json'
    proof,guard=read(p),read(g)
    assert proof['passed'] and guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted']
    assert guard['proof_sha256']==sha(p)
    return p,g,proof
def checks(path,expected):
    suites=list(ET.parse(path).getroot().iter('testsuite'))
    assert sum(int(s.get('tests','0')) for s in suites)==expected
    assert all(int(s.get(k,'0'))==0 for s in suites for k in ['failures','errors','skipped'])
def observe(handle):
    try:p=psutil.Process(handle['pid'])
    except psutil.NoSuchProcess:return dict(**handle,observed_live=False)
    if abs(p.create_time()-handle['create_time'])>.001:return dict(**handle,observed_live=False,pid_reused=True)
    return dict(**handle,observed_live=True,actual_command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2]),
        actual_children=[dict(pid=c.pid,create_time=c.create_time(),command=c.cmdline(),cpu_seconds=sum(c.cpu_times()[:2])) for c in p.children(recursive=True)])

base=ROOT/'projects/flow_matching_research/results/2026-10-16'
folder=base/'full109_actual_preparation_and_full873_recovery_v2';folder.mkdir(parents=True,exist_ok=False)
pc=read(ROOT/prep.CONFIG);prep.verify(pc);pp,pg,proof=passed(pc,prep.REPORT)
assert proof['configuration_sha256']==sha(ROOT/prep.CONFIG)
assert proof['entities']==109 and proof['CSV_files_compared']==191 and len(proof['cases'])==109
assert proof['full_input_coordinate_reconstruction'] and proof['complete_serialized_arrays_verified']
assert all(c['every_normalized_coordinate_rebuilt'] for c in proof['cases'])
dc=read(ROOT/dispatch.CONFIG);dispatch.verify(dc);binding=dispatch.ready_data(dc);material=dispatch.verify_materialization(dc)
assert binding is not None and material['required_cases']==873 and len(material['queues'])==7
manifest=read(binding['manifest']);assert sha(binding['manifest'])==proof['prepared_manifest_sha256']
launch=read(ROOT/'tmp/original_full109_and873_recovery_v2_launch_receipt.json')
assert all(sha(r['path'])==r['sha256'] for r in launch['previous_failed_attempts_preserved'])
checks(ROOT/'tmp/original_full109_optional_train_label_recovery_v2_registered_validation.xml',60)
selected={
    'actual_all109_full_input_proof.json':pp,'successful_data_guard.json':pg,
    'actual_complete_entity_manifest.json':Path(binding['manifest']),
    'actual_all873_materialization.json':Path(dc['artifact_root'])/'materialization.json',
    'successful_data_terminal_status.json':ROOT/pc['state_root']/'status.json',
    'exact_recovery_launch_and_preservation.json':ROOT/'tmp/original_full109_and873_recovery_v2_launch_receipt.json',
    'validation60.xml':ROOT/'tmp/original_full109_optional_train_label_recovery_v2_registered_validation.xml'}
for name,path in selected.items():copy(path,folder/name)
for module in [prep,dispatch]:
    for kind,path in [('source',module.MODULE.replace('.','/')+'.py'),('tests',module.TEST),('configuration',module.CONFIG)]:
        copy(ROOT/path,folder/(module.MODULE.split('.')[-1]+'_'+kind+Path(path).suffix))
state=read(ROOT/dc['state_root']/'status.json')
assert state['state']=='all7metadata_controllers_launched_actual_training_and_reviews_remain'
families=[]
for h in state['family_handles']:
    entry=observe(h);status=ROOT/dc['state_root']/'families'/h['family']/'status.json'
    if status.exists():entry['current_state_snapshot']=read(status)
    families.append(entry)
assert len(families)==7 and all(h['observed_live'] for h in families)
summary=[]
for d in manifest['datasets']:
    summary.append(dict(dataset=d['id'],entities=len(d['entities']),full_TEST_points=sum(e['test_points'] for e in d['entities']),
        labels_only_controls=list(d['labels_only_controls']),primary_label_policy=d['primary_label_policy'],
        adjusted_short_entities=[dict(entity=e['entity_id'],raw_train_points=e['raw_train_points'],fit=e['fit_interval'],purge=e['purged_interval'],validation=e['validation_interval']) for e in d['entities'] if e['short_series_validation_adjustment']]))
unlabeled=[r for r in proof['CSV_mapping'] if not r['CSV_anomaly_labels_available']]
assert len(unlabeled)==96 and all(r['role']=='train' and r['CSV_labelled_anomaly_points'] is None for r in unlabeled)
assert sum(e['full_TEST_points'] for e in summary)==1217975
write(folder/'actual_scope_and_live_family_observation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    datasets=summary,normalized_input_coordinates=sum(r['input_coordinates'] for r in proof['cases']),
    CSV_files_compared=191,unlabeled_training_CSVs=96,labeled_CSVs=95,
    exact_direct_round_trip_CSV_matches=sum(bool(r['exact_matching_candidates']) for r in proof['CSV_mapping']),
    all_CSV_source_equivalence_claimed=False,all7_actual_live_family_handles=families,
    metadata_materialization_complete=True,all873_training_scoring_complete=False,
    model_independent_native_replay_all_labels_seed_statistics_and_full_TAB_complete=False,
    original_scope=dict(papers=45,methods=681,data_tasks=273,scalar_targets=702),all_original_experiments_complete=False))
failed=ROOT/'tmp/original_full109_optional_train_label_recovery_v2_validation.xml'
dest=Path('F:/aicoding/IIA_Data/experiments/original_full109_optional_label_recovery_validation_v2/pre_registration_dependency_order_failure.xml')
copy(failed,dest)
suites=list(ET.parse(failed).getroot().iter('testsuite'));assert sum(int(s.get('failures','0')) for s in suites)==33
write(folder/'preserved_pre_registration_validation_failure.json',dict(path=str(dest),sha256=sha(dest),failed_tests=33,passed_tests=27,
    cause='New full873 tests invoked before new full109 configuration registration;FileNotFoundError,not numerical-model failures',
    correction='Registered full109 config before dependent tests;all60passed before launching either new task'))
(folder/'README.md').write_text('''# 全量109实体数据准备已完成；873项实验已接入独立修复版本

原V1实际因训练CSV缺少is_anomaly列而报错，资源未中止。旧数据准备、调度器已确认为终止；原873项从未物化队列或启动模型。原源文件、配置、错误与失败守卫已逐文件留存并重查哈希。V2只修正训练CSV标签可选的检查，验证/测试标签仍强制存在；原归一化、全部109实体、7个标签版本、划分及资源策略完全保留。60项验证通过，包括复现旧错误、无标签训练文件的完整坐标检查，以及全部873项模型/预算/随机种子/相位门禁保持不变。

实际完整数据准备及守卫均成功，原始数据未覆盖。SMAP54、MSL27、SMD28个实体，完整测试共1,217,975点；75,057,730个归一化输入坐标已从原始输入重建，保存数组完整回读通过。D-12和T-9沿用预先登记的短序列规则与100点隔离带、100点验证段，不删除实体。P-2冲突标注和端点差异保留为独立标签覆盖；标签变体必须使用同一训练模型与预测。

191个现有CSV实际完成检查，其中96个训练CSV无标签、95个CSV有标签。171个CSV有严格round-trip逐值匹配，其余20个的失败候选均如实保留；不据此声称191个CSV与原始数据全部等价。G-7/T-13另外已完成的表达转换及val/test划分证明在独立结果包，本次不放宽原直接比较规则。

873项、7个算法家族的完整队列已依据实际数据物化，7个家族控制器当前进程身份均已核验。它们按原CPU/GPU资源与先决条件排队；队列建立、控制器存活和数据准备成功均不等于873项训练完成。独立完整模型回放、所有标签指标、五种子统计、完整TAB和历史论文协议一致性仍待逐项完成。NASA原始TEST已经源端缩放的限制仍披露。其他论文、方法与数据集范围继续保留。
''',encoding='utf-8')

mf=base/'MOMENT_baseSMD_and_full2086_inputs_masks_closed_v1';mf.mkdir(parents=True,exist_ok=False)
rc=read(ROOT/'configs/runtime/fm_moment_full78_checkpoint_replay.v1.json');cases=moment.verify(rc)
rs=read(ROOT/rc['state_root']/'status.json');rows=[];models=[]
for size,params in [('small',35341512),('base',109641608)]:
    identity='fm_moment_pretrained_'+size+'__release512__smd__seed42';case=cases[identity]
    source,_,bind=moment.check_source(rc,case);directory=Path(rc['artifact_root'])/identity
    nativep=directory/'native/native_replay.json';reviewp=directory/'independent/review.json'
    native,review=read(nativep),read(reviewp)
    assert identity in rs['verified_ids'] and native['passed'] and review['passed']
    assert native['source_binding']==review['source_binding']==bind and native['job']==review['job']==case['job']
    assert native['runtime_sha256']==sha(ROOT/'configs/runtime/fm_moment_full78_checkpoint_replay.v1.json')
    assert review['native_replay_sha256']==sha(nativep) and native['all_original_groups_reinferred'] and native['all_array_dtype_shape_content_identical']
    assert review['all_original_arrays_independently_checked'] and review['all_original_core_metrics_events_and_200_resample_intervals_recomputed']
    assert native['checkpoint_load_audit']['parameter_count']==params and native['checkpoint_load_audit']['all_keys_strictly_loaded']
    assert native['training_updates_performed']==0 and native['native_coverage']==source['native_coverage']
    totals={role:sum(r['native_points'] for r in native['native_coverage'] if r['role']==role) for role in ['train','validation','test']}
    assert totals==dict(train=347450,validation=87323,test=256589)
    assert len(native['native_coverage'])==54 and len(native['arrays'])==72 and native['inline_full_window_MSE_check']['forward_calls']==1374
    assert sha(native['scores_path'])==native['scores_sha256']==bind['scores_sha256']
    for stage,p in [('native',nativep),('independent',reviewp)]:
        attempt=ROOT/rc['state_root']/'attempts'/identity/stage;g=read(attempt/'guard_receipt.json')
        assert g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted'] and g['proof_sha256']==sha(p)
        assert g['exact_child_handle']['command']==[str(ROOT/case['queue']['python']),'-B','-X','utf8','-u','-m',moment.MODULE,'--config','configs/runtime/fm_moment_full78_checkpoint_replay.v1.json','--stage',stage,'--job-id',identity]
        copy(p,mf/size/(stage+'_proof.json'));copy(attempt/'guard_receipt.json',mf/size/(stage+'_guard.json'))
        copy(attempt/'launch.json',mf/size/(stage+'_launch.json'))
    copy(ROOT/case['job']['output_directory']/'result.json',mf/size/'original_result.json')
    copy(ROOT/case['job']['model_config'],mf/size/'model_configuration.json')
    controls=review['independent_confusion_ranking_validation_threshold_PA_and_ten_ratio_controls']
    for ratio,original in source['metrics']['strict'].items():
        got=controls[ratio];assert got['micro']['f1']==original['micro']['f1'] and got['threshold']==original['calibration']['threshold']
        rows.append(dict(model=size,dataset='SMD',context=512,official_checkpoint_seed=42,validation_quantile_exceedance_percent=float(ratio),
            no_PA_F1=got['micro']['f1'],same_validation_threshold_PA_F1=got['same_validation_threshold_PA_diagnostic']['f1'],
            AUROC=got['micro']['auroc'],AP=got['micro']['average_precision'],precision=got['micro']['precision'],recall=got['micro']['recall'],
            F1_entity_cluster_CI95=original['f1_ci95'],event_metrics=original['event_metrics']))
    models.append(dict(model=size,parameters=params,binding=bind,role_points=totals,registered_entities=18,
        full_window_float64_MSE_cross_check=native['inline_full_window_MSE_check'],new_training_seeds=0))
assert models[0]['binding']['input_sha256']==models[1]['binding']['input_sha256']
checks(ROOT/'tmp/MOMENT_SMD_and_actual2086_binding_mask_validation_20261016_v1.xml',51)
copy(ROOT/'tmp/MOMENT_SMD_and_actual2086_binding_mask_validation_20261016_v1.xml',mf/'validation51.xml')
copy(ROOT/'configs/runtime/fm_moment_full78_checkpoint_replay.v1.json',mf/'frozen_full78_replay_configuration.json')
write(mf/'actual_SMD_metrics_all_fixed_quantiles.json',dict(records=rows,models=models,new_complete_cases=1,reused_previously_closed_small_case=1,
    collector_verified_count=rs['verified_count'],required=78,registered_SMD_subset_is_all28_raw_entities=False,
    training_seed_uncertainty_or_historical_paper_or_full_TAB_certified=False,all_original_experiments_complete=False))
reports=[]
for module in [inputs,masks]:
    c=read(ROOT/module.CONFIG);module.verify(c);p,g,d=passed(c,module.REPORT)
    assert d['configuration_sha256']==sha(ROOT/module.CONFIG)
    copy(p,mf/'pretraining_inputs'/module.REPORT)
    copy(g,mf/'pretraining_inputs'/(module.MODULE.split('.')[-1]+'_guard.json'))
    copy(ROOT/c['state_root']/'launch.json',mf/'pretraining_inputs'/(module.MODULE.split('.')[-1]+'_launch.json'))
    copy(ROOT/module.CONFIG,mf/'pretraining_inputs'/Path(module.CONFIG).name)
    reports.append(d)
bound,mask=reports
assert len(bound['sources'])==len(bound['source_records'])==bound['source_count']==2086 and len(bound['original_anomaly_target_bindings'])==1627
assert mask['sources']==2086 and mask['epoch_items']==80850189 and mask['incompatible_training_items']==0
assert mask['every_case_and_shared_window_mask_analytically_covered'] and mask['every_original_random_mask_draw_has_nonempty_RevIN_inputs']
assert not bound['model_train_or_performance_executed'] and not mask['neural_training_or_Torch_RNG_draws']
write(mf/'actual_pretraining_input_and_mask_summary.json',dict(input_sources=2086,source_families=dict(Counter(r['family'] for r in bound['sources'])),
    all_source_input_file_hashes_checked=sum(len(r['input_files_sha256']) for r in bound['source_records']),
    first_middle_last_geometry_checks=sum(len(r['example_geometry_checks']) for r in bound['source_records']),
    original_raw_anomaly_target_profiles=1627,complete_epoch_items_analytically_mask_covered=80850189,incompatible_training_items=0,
    every_80m_tensor_exhaustively_read_or_trained=False,full_Tiny_and_ICM_training_complete=False,all_original_experiments_complete=False))
primary=[r for r in rows if r['validation_quantile_exceedance_percent']==1.]
lines=['# MOMENT Base/SMD完整回放完成；2086源输入与全量掩码检查完成','','|模型|无PA F1|同一验证阈值PA F1|AUROC|AP|','|---|---:|---:|---:|---:|']
for r in primary:lines.append(f"|{r['model']}|{r['no_PA_F1']:.8f}|{r['same_validation_threshold_PA_F1']:.8f}|{r['AUROC']:.8f}|{r['AP']:.8f}|")
lines+=['','Base为新增关闭的任务；Small为先前完成的同数据对照，不重复计数。两个官方固定权重均覆盖现有18实体、54个角色、256589个TEST点、72个分数/标签/时间数组；各1374次完整窗口前向。重新推理数组dtype、shape和内容完全一致，独立进程重算核心指标、事件、200次实体重采样区间及全部10个诊断比例，两阶段守卫均通过。当前完整回放16/78；没有新增训练种子。SMD源端完整28实体的新增轨道在另一个结果包，不能把本18实体结果声称为其完整成绩。',
    '',f"原float64完整窗口MSE检查保持原2e-6相对/绝对容差；Base最大绝对差为{models[1]['full_window_float64_MSE_cross_check']['maximum_mse_absolute_error']}，并非近零绝对误差。所有原0.1/1/5验证阈值结果完整保留；1%主指标未按TEST选择，不做TEST导向的分数翻转。实体重采样区间仅条件于该模型和阈值，不是训练种子差异。原常数通道预处理与独立constant-fit控制轨道继续区分。",'',
    '另一组实际完成的是Tiny/ICM预训练先决条件：2086源全部7645个输入文件哈希校验、6258个首/中/尾样本形状检查和1627个原异常源的raw-observation目标有效性绑定；完整80,850,189个训练条目的掩码兼容性得到解析覆盖，0个不兼容条目。所有实际数据worker及守卫成功退出，未改原始数据。解析覆盖没有执行80M个Torch随机掩码，也不是逐一物化80M张量或完成NN训练。两模型完整预训练、468分类及54预测仍受原资源与先决条件约束继续排队。',
    '', '51项相关验证通过。完整TAB、历史论文数据/预处理一致性、全量训练和整个研究范围仍未完成。']
(mf/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

for f in [folder,mf]:
    copy(__file__,f/'publication_source.py')
    write(f/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        files_sha256={p.relative_to(f).as_posix():sha(p) for p in sorted(f.rglob('*')) if p.is_file()},all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/full109_actual_recovery_and_MOMENT_actual_closures.2026-10-16_v2.json'
write(ara,dict(packets=[f.relative_to(ROOT).as_posix() for f in [folder,mf]],tests=111,
    actual_completed=dict(full_original_entities_prepared=109,normalized_input_coordinates=75057730,new_complete_MOMENT_replays=1,
        all2086_training_sources_bound=True,full80850189_item_mask_compatibility=True),
    actual_materialized_and_dispatched=dict(method_dataset_seed_cases=873,families=7),
    all_original_experiments_complete=False,scope_preserved=dict(papers=45,methods=681,data_tasks=273,scalar_targets=702)))
paths=[module.MODULE.replace('.','/')+'.py' for module in [prep,dispatch]]+[module.TEST for module in [prep,dispatch]]+[module.CONFIG for module in [prep,dispatch]]
paths+=[p.relative_to(ROOT).as_posix() for f in [folder,mf] for p in f.rglob('*') if p.is_file()]+[ara.relative_to(ROOT).as_posix()]
git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/full109_recovery_and_actual_MOMENT_closures_publication_manifest_20261016_v2.json',dict(
    previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),full109_prepared=True,materialized_jobs=873,live_family_controllers=7,
    additional_MOMENT_native_complete=1,collector_count=rs['verified_count'],input_sources=2086,mask_items=80850189,primary_SMD=primary)))
