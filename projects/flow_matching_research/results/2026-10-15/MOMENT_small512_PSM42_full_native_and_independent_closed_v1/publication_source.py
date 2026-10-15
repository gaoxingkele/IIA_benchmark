"""Publish one actually closed official-weight full MOMENT replay, not a new seed."""
import csv
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import run_moment_full_checkpoint_replay_v1 as v

ROOT,read,write,sha=v.ROOT,v.read,v.write,v.sha
config_path='configs/runtime/fm_moment_full78_checkpoint_replay.v1.json'
runtime=read(ROOT/config_path);cases=v.verify(runtime)
name='fm_moment_pretrained_small__release512__psm__seed42';case=cases[name]
source,dataset,binding=v.check_source(runtime,case)
directory=Path(runtime['artifact_root'])/name
native_path=directory/'native/native_replay.json';review_path=directory/'independent/review.json'
native,review=read(native_path),read(review_path)
assert native['passed'] and review['passed'] and native['job']==review['job']==case['job']
assert native['source_binding']==review['source_binding']==binding
assert review['native_replay_sha256']==sha(native_path) and native['runtime_sha256']==sha(ROOT/config_path)
assert native['all_original_groups_reinferred'] and native['all_array_dtype_shape_content_identical']
assert review['all_original_arrays_independently_checked']
assert review['all_original_core_metrics_events_and_200_resample_intervals_recomputed']
assert native['training_updates_performed']==0 and native['checkpoint_load_audit']['parameter_count']==35341512
assert native['checkpoint_load_audit']['all_keys_strictly_loaded']
assert native['native_coverage']==source['native_coverage']
assert [(r['role'],r['native_points']) for r in native['native_coverage']]==[('train',105884),('validation',26497),('test',87841)]
assert native['inline_full_window_MSE_check']['forward_calls']==431
assert sha(native['scores_path'])==binding['scores_sha256']==native['scores_sha256']
state=read(ROOT/runtime['state_root']/'status.json')
assert name in state['verified_ids'] and state['required_jobs']==78
validation=ROOT/'tmp/MOMENT_small512_PSM42_native_closure_validation_20261015_v1.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'))
checks=sum(int(s.get('tests',0)) for s in suites)
assert checks>10 and all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_small512_PSM42_full_native_and_independent_closed_v1'
folder.mkdir(parents=True,exist_ok=False)
files=dict(frozen_full78_replay_configuration=ROOT/config_path,
    frozen_original_model_configuration=ROOT/case['job']['model_config'],
    original_completed_result=ROOT/case['job']['output_directory']/'result.json',
    native_full_checkpoint_replay=native_path,independent_full_score_review=review_path)
for phase in ['native','independent']:
    base=ROOT/runtime['state_root']/'attempts'/name/phase
    g=read(base/'guard_receipt.json');p=native_path if phase=='native' else review_path
    assert g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted'] and g['proof_sha256']==sha(p)
    expected=[str(ROOT/case['queue']['python']),'-B','-X','utf8','-u','-m',v.MODULE,'--config',config_path,'--stage',phase,'--job-id',name]
    assert g['exact_child_handle']['command']==expected
    files[phase+'_guard']=base/'guard_receipt.json';files[phase+'_launch']=base/'launch.json'
for label,path in files.items():
    shutil.copyfile(path,folder/(label+'.json'));assert sha(path)==sha(folder/(label+'.json'))
shutil.copyfile(validation,folder/'validation.xml')
write(folder/'complete_replay_collector_status_snapshot.json',state)
records=[]
controls=review['independent_confusion_ranking_validation_threshold_PA_and_ten_ratio_controls']
for ratio,value in source['metrics']['strict'].items():
    c=controls[ratio]
    assert c['threshold']==value['calibration']['threshold'] and c['micro']['f1']==value['micro']['f1']
    records.append(dict(model='MOMENT-1-small',dataset='PSM',context=512,seed=42,
        calibration='validation_only',false_alarm_percent=float(ratio),threshold=c['threshold'],
        no_PA_F1=c['micro']['f1'],same_threshold_PA_F1=c['same_validation_threshold_PA_diagnostic']['f1'],
        AUROC=c['micro']['auroc'],average_precision=c['micro']['average_precision'],
        precision=c['micro']['precision'],recall=c['micro']['recall'],
        F1_CI95_low=value['f1_ci95']['low'],F1_CI95_high=value['f1_ci95']['high'],
        bootstrap_unit=value['f1_ci95']['unit'],bootstrap_resamples=value['f1_ci95']['resamples'],
        event_recall=value['event_metrics']['event_recall'],events=value['event_metrics']['events'],
        detected_events=value['event_metrics']['detected_events'],
        false_alarms_per1000_normal=value['event_metrics']['false_alarms_per_1000_normal_points']))
write(folder/'metrics_and_boundaries.json',dict(job_id=name,records=records,model_parameters=35341512,
    native_train_points=105884,native_validation_points=26497,native_test_points=87841,complete_forward_calls=431,
    native_all_scores_dtype_shape_bytes_equal=True,independent_all_arrays_and_core_metrics_review=True,
    bootstrap_intervals_recomputed_with_original_library=True,bootstrap_is_between_model_uncertainty=False,
    numerical_VUS_affiliation_integrals_independently_recomputed=False,preprocessing_values_independently_reconstructed=False,
    offline_window_context_is_causal=False,new_training_seed=False,training_updates_performed=0,
    historical_original_execution_guard_proven=False,new_replay_guard_retroactively_certifies_old_execution=False,
    full_TAB_harness_certified=False,historical_paper_equivalence_certified=False,all_original_experiments_complete=False))
buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=list(records[0]),lineterminator='\n')
writer.writeheader();writer.writerows(records)
(folder/'validation_calibration_metrics.csv').write_text(buffer.getvalue(),encoding='utf-8',newline='')
write(folder/'original_all10_train_plus_test_pooled_threshold_diagnostics.json',source['metrics']['tab_core_diagnostic'])
(folder/'README.md').write_text(f'''# MOMENT small / PSM / context512：完整检查点重推理及独立核验

已实际重新加载官方 MOMENT-1-small 的完整 35,341,512 参数权重，保留原 seed42、batch1、context512、minimal-patch padding、CPU 两线程及全部训练/验证/测试时间点。431 次完整窗口 forward 覆盖训练105,884、验证26,497、测试87,841个原生点。分数、标签及所有原数组的 dtype、shape、内容字节均与原结果一致。不是新训练种子，也不是仅检查已保存分数。

第二独立进程重新核对完整数组和原数据身份，复算所有核心指标、事件与200次分组/时间块区间，并独立核对混淆矩阵、排序指标、验证阈值、PA以及全部10个 train+test pooled 阈值诊断。两次都有成功退出、无资源中止的实际保护回执。原始运行的历史保护证据缺失仍保留，不用新复核回执追溯认证旧运行。

|验证集假警报比例|未做PA F1|同阈值PA F1|AUROC|AP|
|---|---:|---:|---:|---:|
|0.1%|0.0164134|0.841551|0.555556|0.337271|
|1%|0.0366394|0.879989|0.555556|0.337271|
|5%|0.0771489|0.916030|0.555556|0.337271|

1% 固定验证阈值为4.6210474395752，未做PA的F1区间[0.0106610,0.0810290]；原200次循环时间块bootstrap、block1024、seed10042，表达同一预训练模型与固定阈值的条件不确定性，不是模型种子不确定性。1%时72个事件命中33个，事件召回0.458333；每1000个正常点假警报3.387961。窗口有未来上下文，原点延迟不能称因果在线延迟。三个预先登记阈值全部保留，没有用测试F1挑选最佳阈值。

相同分数和阈值下PA带来的变化已完整核验；这不能单独证明论文失实，也不能将当前 PSM/small 零样本设置视为论文 SMD 指标的直接复现。本文原生复核与论文 ICM/Tiny 预训练实验保持不同任务。历史论文流程、完整TAB执行器、VUS/Affiliation数值积分和全部预处理值的独立重建仍未认证。原区间重算使用原库，未声称另写了bootstrap估计器。

{checks}项相关检查通过。源结果、模型/完整78项队列配置、实际原生与独立证明、保护及启动记录、全10项pooled诊断均随包归档，大数组及权重保留原F盘。完整78项队列中本例已闭合，其他任务以实际原生/独立证明为准；至少45篇/681方法记录/273数据任务及新增实验范围不变，全部研究目标尚未完成。
''',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),job_id=name,full_native_replay_and_independent_review=True,
    tests_passed=checks,files_sha256={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()},all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/MOMENT_small512_PSM42_full_native_closed.2026-10-15_v1.json'
write(ara,dict(method='MOMENT-1-small',dataset='PSM',context=512,seed=42,actual_full_checkpoint_replay=True,independent_all_arrays_review=True,
    evidence=folder.relative_to(ROOT).as_posix(),required_native_queue_cases=78,new_training_seed=False,original_source_guard_history_proven=False,
    strict_validation_no_PA_F1=records[1]['no_PA_F1'],same_threshold_PA_F1=records[1]['same_threshold_PA_F1'],
    historical_paper_equivalence_certified=False,full_TAB_harness_certified=False,all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix()];paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/MOMENT_small512_PSM42_full_native_publication_manifest_20261015_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(full_native_model_closed=name,parameters=35341512,test_points=87841,tests_passed=checks,exact_publication_files=len(paths))))
