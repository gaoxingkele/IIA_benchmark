"""Publish bounded full-input prerequisites and two actual native model closures."""
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import bind_moment_icm_full_training_inputs_v1 as p
from scripts.flow_matching import run_moment_full_checkpoint_replay_v1 as v

ROOT, read, sha, write = p.ROOT, p.read, p.sha, p.write
stamp = datetime.now(timezone.utc).isoformat()


def validation(path, count):
    suites = list(ET.parse(path).getroot().iter('testsuite'))
    assert sum(int(s.get('tests', 0)) for s in suites) == count
    assert all(int(s.get(key, 0)) == 0 for s in suites for key in ('failures', 'errors', 'skipped'))


def snapshot_owner(state_path):
    state = read(state_path)
    proc = psutil.Process(state['pid'])
    assert proc.create_time() == state['create_time']
    return dict(state=state, exact_live_handle_checked_utc=stamp,
                command=proc.cmdline(), cpu_seconds=sum(proc.cpu_times()[:2]))


targets = []
packet = ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_all2086_bounded_training_inputs_registered_v1'
packet.mkdir(parents=True, exist_ok=False)
config = read(ROOT/p.CONFIG)
original, sources = p.verify(config)
test = ROOT/'tmp/MOMENT_ICM_cache_reader_and_bindings_validation_20261015_v3.xml'
validation(test, 20)
shutil.copyfile(test, packet/'validation_20_checks.xml')
shutil.copyfile(ROOT/p.CONFIG, packet/'frozen_input_bindings_configuration.json')
failed = ROOT/'tmp/MOMENT_ICM_unified_reader_validation_20261015_v1.xml'
assert any(int(s.get('failures', 0)) > 0 for s in ET.parse(failed).getroot().iter('testsuite'))
write(packet/'preserved_pre_registration_validation.json', dict(
    path=str(failed), sha256=sha(failed), original_xml_utf8=failed.read_text(encoding='utf-8'),
    diagnosis='Classification NPY files live under TRAIN; reader and independent expected-path fixture corrected before registration.',
    scientific_attempt_started=False,
    additional_review_fix='Preserve a supplied empty target-binding dict; the actual worker adds verified raw masks incrementally.',
    actual_worker_path_regression='One real original anomaly source is fully read, then mandatory full2086/1627 gate rejects partial fixture. No full readiness report written.'))
base = ROOT/config['state_root']
owner = snapshot_owner(base/'status.json')
assert owner['state']['state'] == 'waiting_full322_control_closure_and_full_epoch_reference_plan'
assert not (base/'worker_launch.json').exists()
shutil.copyfile(base/'launch.json', packet/'actual_metadata_launch.json')
write(packet/'actual_metadata_owner_snapshot.json', owner)
upstream = {}
for name in ('fm_moment_icm_anomaly_finite_control_cold_metadata_v4',
             'fm_moment_icm_all322_finite_control_closure_v1', 'fm_moment_icm_full_PILE_epoch_plan_v1'):
    upstream[name] = snapshot_owner(ROOT/'experiments/runs'/name/'status.json')
write(packet/'actual_upstream_owners_snapshot.json', upstream)
write(packet/'scope_and_status.json', dict(
    captured_utc=stamp, status='REGISTERED_WAITING_ACTUAL_FULL_PREREQUISITES',
    selected_sources=len(sources), source_family_counts={family:sum(s['family']==family for s in sources)
        for family in sorted({s['family'] for s in sources})},
    epoch_items=original['count_audit']['total_items'], full_epoch_batch_size=2048,
    raw_anomaly_target_profiles_required=1627, changed_preprocessing_controls_required=322,
    bounded_mapping_cache_bytes=config['header_and_mapping_cache_bytes'], block_bytes=config['mapping_block_bytes'],
    frozen_configuration_sha256=sha(ROOT/p.CONFIG), tests_passed=20,
    registration_is_full_data_readiness=False, full_input_binding_complete=False,
    actual_full80m_tensor_replay_complete=False, full_neural_runner_integrated=False,
    any_new_model_training_or_performance=False, historical_paper_equivalence_certified=False,
    original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
    all_original_experiments_complete=False))
(packet/'README.md').write_text('''# MOMENT Tiny/ICM 全量训练输入：读取器验证并登记

读取组件覆盖四类原始数据：分类、长预测、Monash 与异常序列；异常序列包含未改动的1627源和明确改变预处理的322控制。全2086源、80,850,189个训练项仍全部登记。采用64MiB只读分块映射缓存及1MiB块，原值、dtype、行身份、通道余数抽样及Monash洗牌位置保留。

原始异常源先重新核对原CSV的观测位置；不可从插值后有限值推断目标有效。原始观测且源有限的目标与padding、后续随机掩码共同决定损失。分类/Monash仍使用源缓存padding语义，未声称独立核验全部内部原始缺失；长预测仍保留源全CSV插值，不能称严格因果预处理。全部原数据和旧缓存保持。

20项检查通过，包括真实数据各家族的完整文件路径登记、完整首源哈希、首/中/末读取几何、尾部NaN的实际源控制、动态原始掩码绑定、部分范围拒绝，以及不得提高资源上限或绕过共享锁。一次登记前分类TRAIN路径失败原文、诊断和修正均保存。单源worker回归只验证管线；最终范围门禁明确拒绝它，未生成完整就绪报告。

新任务必须等待真实322源/966划分/6762数组的独立闭合、以及完整80,850,189项索引双射证明及各自成功保护回执。之后核对全源输入哈希，为1627原始异常源逐一建立观测目标绑定，并验证每源首/中/末输入；它不是全8000万张量穷举读取。原16/21/16/3GiB策略及共享资源锁不变。

此包保存的是已启动但等待前置任务的状态，不是数据绑定已经完成。完整神经训练循环仍需接入有效目标掩码，Tiny/ICM一整轮预训练、下游实验、论文指标复现均未据此宣告完成。原45篇/681方法记录/273数据任务和新增实验全部继续。
''', encoding='utf-8')
(packet/'.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n',encoding='utf-8')
write(packet/'archive_validation.json',dict(captured_utc=stamp,files_sha256={f.name:sha(f) for f in sorted(packet.iterdir()) if f.is_file()},
    registered_only=True,training_executed=False,all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/MOMENT_ICM_all2086_training_input_bindings.2026-10-15_v1.json'
write(ara,dict(paper_id='moment_long_context',status='REGISTERED_WAITING_ACTUAL_FULL_PREREQUISITES',
    configuration=p.CONFIG,selected_sources=2086,epoch_items=80850189,evidence=packet.relative_to(ROOT).as_posix(),
    neural_training_complete=False,all_original_experiments_complete=False))
targets += [packet.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix(),p.CONFIG,
    'scripts/flow_matching/moment_icm_pile_cache_reader_v1.py','scripts/flow_matching/bind_moment_icm_full_training_inputs_v1.py',
    'tests/test_fm_moment_icm_pile_cache_reader_v1.py','tests/test_fm_moment_icm_full_training_input_bindings_v1.py']

runtime_path='configs/runtime/fm_moment_full78_checkpoint_replay.v1.json'
runtime=read(ROOT/runtime_path);cases=v.verify(runtime)
state=read(ROOT/runtime['state_root']/'status.json')
assert state['required_jobs']==78 and state['verified_count']>=6
native_validation=ROOT/'tmp/MOMENT_small512_PSM42_native_closure_validation_20261015_v1.xml'
validation(native_validation,15)
packet=ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_small512_SWaT_SWaN42_full_native_closed_v1'
packet.mkdir(parents=True,exist_ok=False)
shutil.copyfile(native_validation,packet/'unchanged_frozen_native_validation_15_checks.xml')
shutil.copyfile(ROOT/runtime_path,packet/'frozen_full78_replay_configuration.json')
write(packet/'complete78_collector_status_snapshot.json',state)
metrics=[];summaries=[]
for dataset in ('swat','swan'):
    name=f'fm_moment_pretrained_small__release512__{dataset}__seed42'
    case=cases[name];source,data,binding=v.check_source(runtime,case)
    directory=Path(runtime['artifact_root'])/name
    native_path=directory/'native/native_replay.json';review_path=directory/'independent/review.json'
    n,r=read(native_path),read(review_path)
    assert name in state['verified_ids'] and n['passed'] and r['passed']
    assert n['job']==r['job']==case['job'] and n['source_binding']==r['source_binding']==binding
    assert n['runtime_sha256']==sha(ROOT/runtime_path) and r['native_replay_sha256']==sha(native_path)
    assert n['all_original_groups_reinferred'] and n['all_array_dtype_shape_content_identical']
    assert r['all_original_arrays_independently_checked'] and r['all_original_core_metrics_events_and_200_resample_intervals_recomputed']
    assert n['checkpoint_load_audit']['all_keys_strictly_loaded'] and n['checkpoint_load_audit']['parameter_count']==35341512
    assert n['training_updates_performed']==0 and n['native_coverage']==source['native_coverage']
    assert len(data['entities'])==1 and len(n['native_coverage'])==3 and len(n['arrays'])==4
    points={role:sum(g['native_points'] for g in n['native_coverage'] if g['role']==role) for role in ('train','validation','test')}
    assert points==({'train':395900,'validation':99000,'test':449919} if dataset=='swat' else {'train':15637,'validation':3935,'test':72000})
    assert sha(n['scores_path'])==n['scores_sha256']==binding['scores_sha256']
    dest=packet/dataset;dest.mkdir()
    files=dict(original_result=ROOT/case['job']['output_directory']/'result.json',native_replay=native_path,independent_review=review_path,
               original_model_configuration=ROOT/case['job']['model_config'])
    for phase,path in [('native',native_path),('independent',review_path)]:
        base=ROOT/runtime['state_root']/'attempts'/name/phase;g=read(base/'guard_receipt.json')
        assert g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted'] and g['proof_sha256']==sha(path)
        assert g['exact_child_handle']['command']==[str(ROOT/case['queue']['python']),'-B','-X','utf8','-u','-m',v.MODULE,
            '--config',runtime_path,'--stage',phase,'--job-id',name]
        files[phase+'_guard']=base/'guard_receipt.json';files[phase+'_launch']=base/'launch.json'
    for label,path in files.items():shutil.copyfile(path,dest/(label+'.json'));assert sha(path)==sha(dest/(label+'.json'))
    write(dest/'actual_registered_dataset_manifest_snapshot.json',data)
    controls=r['independent_confusion_ranking_validation_threshold_PA_and_ten_ratio_controls']
    for ratio,s in source['metrics']['strict'].items():
        control=controls[ratio];ci=s['f1_ci95'];assert control['micro']['f1']==s['micro']['f1']
        metrics.append(dict(model='MOMENT-1-small',context=512,seed=42,dataset=dataset,dataset_track=data['source_track'],
            validation_false_alarm_percent=float(ratio),threshold=s['calibration']['threshold'],micro_no_PA_F1=s['micro']['f1'],
            macro_no_PA_F1=s['macro_f1'],same_validation_threshold_PA_F1=control['same_validation_threshold_PA_diagnostic']['f1'],
            AUROC=s['micro']['auroc'],AP=s['micro']['average_precision'],precision=s['micro']['precision'],recall=s['micro']['recall'],
            circular_block_CI95_low=ci['low'],circular_block_CI95_high=ci['high'],resamples=ci['resamples'],block_points=ci['block_points'],
            test_points=points['test'],events=s['event_metrics']['events'],detected_events=s['event_metrics']['detected_events'],
            event_recall=s['event_metrics']['event_recall'],false_alarms_per1000_normal=s['event_metrics']['false_alarms_per_1000_normal_points']))
    summaries.append(dict(job_id=name,source_binding=binding,points=points,entities=1,arrays=4,
        original_arrays_bitwise_equal=True,all_registered_groups_reinferred=True,independent_complete_core_review_passed=True))
write(packet/'complete_metrics_and_scope.json',dict(captured_utc=stamp,records=metrics,model_summaries=summaries,
    original_pretraining_repeated=False,new_training_seeds=0,full_TAB_harness_certified=False,
    historical_paper_equivalence_certified=False,preprocessing_values_independently_reconstructed=False,
    independent_lowlevel_VUS_affiliation_integrals_recomputed=False,new_guards_retroactively_certify_source_history=False,
    all_original_experiments_complete=False))
buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=list(metrics[0]),lineterminator='\n')
writer.writeheader();writer.writerows(metrics);(packet/'all_six_validation_threshold_metrics.csv').write_text(buf.getvalue(),encoding='utf-8',newline='')
(packet/'README.md').write_text('''# MOMENT small512 / SWaT 与 SWaN：完整原生和独立复核

两例均使用35,341,512参数官方MOMENT-1-small权重、context512、batch1、原两线程和seed42。分别完整重推理原train/validation/test三组与四个原数组；dtype/shape/内容全部字节相等。第二独立进程核对完整数组、核心指标、事件和原200次区间，另行实现混淆矩阵/排序/验证阈值/PA及十比例诊断。两阶段均保存成功退出且没有资源中止的实际保护回执。

|数据集|测试点|固定验证1%未做PA F1|同阈值PA F1|AUROC|AP|固定模型条件F1区间|
|---|---:|---:|---:|---:|---:|---|
|SWaT|449919|0.08351799|0.84116665|0.26204382|0.11399928|[0.04488450, 0.13207219]|
|SWaN|72000|0.42620509|0.73510045|0.68456483|0.49539916|[0.33524099, 0.50572057]|

SWaT训练395900点、验证99000点，35事件检出17；SWaN训练15637点、验证3935点，6615事件检出2888。以上主阈值提前固定，0.1%、1%、5%全部结果均保留，未根据测试F1挑阈值。PA用测试标签仅供诊断，其明显放大不能解释为严格未调整点检测性能。200次circular-block bootstrap、块1024、seed10042，为同一固定模型及固定验证阈值的条件区间，不是多重训种子区间。

SWaT为登记公开benchmark负载的连续序列划分；SWaN为冻结mTSBench实体CSV及全登记通道，未证明恰好匹配原作者子集。所有输入、归一化及实体元数据见原清单快照。原上下文窗口使用未来点，事件延迟不能称因果在线延迟。

复用冻结代码的15项已通过相关验证，未重新执行已完成的模型。未独立重建所有预处理值或VUS/Affiliation积分；原训练执行保护历史缺失仍保留，不由新回执追溯认证。两例是官方预训练模型的零样本完整复核，不是新增训练种子、Tiny/ICM整轮预训练、完整TAB执行器或原论文配方等价复现。

本轮新增闭合两例，78例队列至少6例已完成；72原始与6预处理控制、45篇/681方法记录/273数据任务及新增实验均继续保留，研究目标未完成。
''',encoding='utf-8')
(packet/'.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n',encoding='utf-8')
shutil.copyfile(__file__,packet/'publication_source.py')
write(packet/'archive_validation.json',dict(captured_utc=stamp,new_native_models_closed=2,new_training_seeds=0,tests_passed=15,
    files_sha256={f.relative_to(packet).as_posix():sha(f) for f in sorted(packet.rglob('*')) if f.is_file()},all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/MOMENT_small512_SWaT_SWaN_full_native_closed.2026-10-15_v1.json'
write(ara,dict(method='MOMENT-1-small',context=512,datasets=['SWaT','SWaN'],seed=42,new_closed_cases=2,
    full_native_checkpoint_and_independent_review=True,evidence=packet.relative_to(ROOT).as_posix(),required_cases=78,
    new_training_seed=False,historical_paper_equivalence_certified=False,all_original_experiments_complete=False))
targets += [packet.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix()]

paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
manifest=ROOT/'tmp/MOMENT_full_inputs_SWaT_SWaN_publication_manifest_20261015_v1.json'
write(manifest,dict(previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_publication_files=len(paths),manifest=str(manifest),bound_sources_registered=2086,
    actual_model_closures_added=2,full_input_readiness=False,all_original_experiments_complete=False)))
