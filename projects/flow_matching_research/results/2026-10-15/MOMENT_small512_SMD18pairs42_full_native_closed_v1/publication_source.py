"""Archive actual SMD registered18-pair native replay and independent controls."""
import csv
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import run_moment_full_checkpoint_replay_v1 as v

ROOT,read,sha,write=v.ROOT,v.read,v.sha,v.write
config_path='configs/runtime/fm_moment_full78_checkpoint_replay.v1.json';runtime=read(ROOT/config_path);cases=v.verify(runtime)
name='fm_moment_pretrained_small__release512__smd__seed42';case=cases[name]
source,dataset,binding=v.check_source(runtime,case)
directory=Path(runtime['artifact_root'])/name;np=directory/'native/native_replay.json';rp=directory/'independent/review.json'
n,r=read(np),read(rp)
assert n['passed'] and r['passed'] and n['job']==r['job']==case['job'] and n['source_binding']==r['source_binding']==binding
assert n['runtime_sha256']==sha(ROOT/config_path) and r['native_replay_sha256']==sha(np)
assert n['all_original_groups_reinferred'] and n['all_array_dtype_shape_content_identical']
assert r['all_original_arrays_independently_checked'] and r['all_original_core_metrics_events_and_200_resample_intervals_recomputed']
assert n['checkpoint_load_audit']['parameter_count']==35341512 and n['checkpoint_load_audit']['all_keys_strictly_loaded']
assert n['training_updates_performed']==0 and n['native_coverage']==source['native_coverage']
assert len(dataset['entities'])==18 and len(n['native_coverage'])==54 and len(n['arrays'])==72
assert 'mTSBench available entity pairs' in dataset['source_track']
assert {role:sum(g['native_points'] for g in n['native_coverage'] if g['role']==role) for role in ['train','validation','test']}==dict(train=347450,validation=87323,test=256589)
assert sha(n['scores_path'])==n['scores_sha256']==binding['scores_sha256']
state=read(ROOT/runtime['state_root']/'status.json');assert name in state['verified_ids'] and state['required_jobs']==78
validation=ROOT/'tmp/MOMENT_small512_PSM42_native_closure_validation_20261015_v1.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==15
assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_small512_SMD18pairs42_full_native_closed_v1'
folder.mkdir(parents=True,exist_ok=False)
files=dict(frozen_full78_replay_configuration=ROOT/config_path,frozen_original_model_configuration=ROOT/case['job']['model_config'],
    original_result=ROOT/case['job']['output_directory']/'result.json',native_replay=np,independent_full_review=rp)
for phase,path in [('native',np),('independent',rp)]:
    b=ROOT/runtime['state_root']/'attempts'/name/phase;g=read(b/'guard_receipt.json')
    assert g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted'] and g['proof_sha256']==sha(path)
    assert g['exact_child_handle']['command']==[str(ROOT/case['queue']['python']),'-B','-X','utf8','-u','-m',v.MODULE,'--config',config_path,'--stage',phase,'--job-id',name]
    files[phase+'_guard']=b/'guard_receipt.json';files[phase+'_launch']=b/'launch.json'
for label,path in files.items():shutil.copyfile(path,folder/(label+'.json'));assert sha(path)==sha(folder/(label+'.json'))
shutil.copyfile(validation,folder/'validation_15_checks.xml')
write(folder/'actual_registered_dataset_manifest_snapshot.json',dataset)
write(folder/'complete78_collector_status_snapshot.json',state)
rows=[];controls=r['independent_confusion_ranking_validation_threshold_PA_and_ten_ratio_controls']
for ratio,s in source['metrics']['strict'].items():
    c=controls[ratio];ci=s['f1_ci95'];assert c['micro']['f1']==s['micro']['f1']
    rows.append(dict(model='MOMENT-1-small',dataset='SMD',dataset_track='pinned_mTSBench_18_available_entity_pairs',context=512,seed=42,
        validation_false_alarm_percent=float(ratio),threshold=s['calibration']['threshold'],micro_no_PA_F1=s['micro']['f1'],
        macro_no_PA_F1=s['macro_f1'],same_threshold_PA_F1=c['same_validation_threshold_PA_diagnostic']['f1'],
        AUROC=c['micro']['auroc'],AP=c['micro']['average_precision'],precision=c['micro']['precision'],recall=c['micro']['recall'],
        entity_cluster_CI95_low=ci['low'],entity_cluster_CI95_high=ci['high'],resamples=ci['resamples'],
        test_entities=18,test_points=256589,event_recall=s['event_metrics']['event_recall'],events=s['event_metrics']['events'],
        detected_events=s['event_metrics']['detected_events'],false_alarms_per1000_normal=s['event_metrics']['false_alarms_per_1000_normal_points']))
write(folder/'complete_primary_and_calibration_metrics.json',dict(records=rows,dataset_track=dataset['source_track'],
    entity_ids=[g['entity_id'] for g in dataset['entities']],source_binding=binding,
    every_registered_role_group_replayed=True,original_arrays_bitwise_equal=True,independent_complete_core_review_passed=True,
    original_execution_guard_history_proven=binding['source_guard_history']['original_execution_guard_history_proven'],
    new_guard_does_not_retroactively_certify_source_history=True,original_classic_payload_equivalence_certified=False,
    historical_paper_equivalence_certified=False,full_TAB_harness_certified=False,original_pretraining_repeated=False,new_training_seed=False,
    independent_lowlevel_VUS_affiliation_integrals_recomputed=False,preprocessing_values_independently_reconstructed=False,
    all_original_experiments_complete=False))
buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(folder/'all3_validation_threshold_metrics.csv').write_text(buf.getvalue(),encoding='utf-8',newline='')
(folder/'README.md').write_text('''# MOMENT small512 / SMD已登记18配对实体：完整原生及独立复核通过

此例严格使用已冻结的mTSBench可用实体配对输入，18个实体、54个train/val/test角色组、72个原数组。重推理覆盖训练347,450、验证87,323、测试256,589个原生点，完整35,341,512参数官方权重、context512、batch1、两线程、seed42及原padding规则保持。分数和标签等数组的dtype/shape/全部字节与原结果一致；第二独立进程核对全部数组、核心指标/事件、200次条件区间及独立混淆矩阵/排序/阈值/PA诊断，两阶段均有成功退出且无资源中止的实际回执。

|固定验证假警报比例|未做PA micro F1|同阈值PA F1|AUROC|AP|
|---|---:|---:|---:|---:|
|0.1%|0.08604147|0.22726752|0.81358145|0.26084896|
|1%|0.17482408|0.66894743|0.81358145|0.26084896|
|5%|0.36016899|0.68001795|0.81358145|0.26084896|

1%为预先固定主协议，阈值125.51861877441401，micro F1区间[0.08216823,0.28501802]、macro F1为0.13723021。200次entity-cluster bootstrap、seed10042，表达同一预训练模型和固定阈值的条件不确定性，不是重训种子区间。196个事件检出73个，事件召回0.37244898；每1000正常点假警报12.18036。原窗口包含未来上下文，不能把原点检测延迟称为因果在线延迟。

数据清单明确声明“pinned mTSBench available entity pairs; full features except timestamp/label; distinct from flat classic payload”。这是此已登记18配对负载的完整复核，不能称经典完整SMD负载或论文原始指标等价复现。所有实体ID、裁剪/划分/归一化元数据和原始文件清单随包保留；原历史运行保护缺失仍如实保留，不能由新回执追溯认证。

三个登记验证阈值全部保存，未根据测试F1选择；原10个train+test pooled诊断仍在源结果中。PA使用测试标签且只作诊断，不混为严格未做PA排名。原bootstrap库用于区间重算；未声称独立实现VUS/Affiliation积分、全部预处理值、完整TAB执行器或历史论文完整配方。

15项相关验证在冻结源码未变后复用；没有重复执行已完成真实模型。本例闭合后全量队列至少4/78通过（72原始及6预处理控制仍全部在范围）。官方零样本MOMENT复核不同于Tiny/ICM论文的一整轮预训练，未新增训练种子。至少45篇/681方法记录/273数据任务及新增实验保持，研究目标未完成。
''',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),full_native_models_closed=1,entities=18,arrays=72,
    test_points=256589,tests_passed=15,new_training_seeds=0,
    files_sha256={f.name:sha(f) for f in sorted(folder.iterdir()) if f.is_file()},all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/MOMENT_small512_SMD18pairs_full_native_closed.2026-10-15_v1.json'
write(ara,dict(method='MOMENT-1-small',context=512,dataset='SMD',dataset_track=dataset['source_track'],entities=18,
    seed=42,full_native_checkpoint_and_independent_review=True,evidence=folder.relative_to(ROOT).as_posix(),
    required_native_cases=78,new_training_seed=False,classic_payload_or_historical_paper_equivalence_certified=False,
    all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix()];paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/MOMENT_small512_SMD18pairs_full_native_publication_manifest_20261015_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(full_native_model=name,entities=18,arrays=72,no_PA_F1=rows[1]['micro_no_PA_F1'],same_threshold_PA_F1=rows[1]['same_threshold_PA_F1'],exact_publication_files=len(paths))))
