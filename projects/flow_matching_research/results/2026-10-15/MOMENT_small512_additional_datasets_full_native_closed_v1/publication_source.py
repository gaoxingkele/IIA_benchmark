"""Archive complete native zero-shot cases and independent full-array reviews."""
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
runtime_path='configs/runtime/fm_moment_full78_checkpoint_replay.v1.json'
runtime=read(ROOT/runtime_path);cases=v.verify(runtime)
state=read(ROOT/runtime['state_root']/'status.json')
datasets=['cicids','tep_classic','skab','pronto_day2','pronto_day3','pronto_day4']
names=[f'fm_moment_pretrained_small__release512__{d}__seed42' for d in datasets]
selected=[name for name in names if name in state['verified_ids']]
assert len(selected)>=5 and state['required_jobs']==78
validation=ROOT/'tmp/MOMENT_small512_PSM42_native_closure_validation_20261015_v1.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==15
assert all(int(s.get(key,0))==0 for s in suites for key in ['failures','errors','skipped'])
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_small512_additional_datasets_full_native_closed_v1'
assert not folder.exists()
snapshots={};rows=[];summaries=[]
for name in selected:
    case=cases[name];source,data,binding=v.check_source(runtime,case);dataset=case['job']['dataset']
    directory=Path(runtime['artifact_root'])/name
    native_path=directory/'native/native_replay.json';review_path=directory/'independent/review.json'
    n,r=read(native_path),read(review_path)
    assert n['passed'] and r['passed'] and n['job']==r['job']==case['job']
    assert n['source_binding']==r['source_binding']==binding and n['runtime_sha256']==sha(ROOT/runtime_path)
    assert r['native_replay_sha256']==sha(native_path)
    assert n['all_original_groups_reinferred'] and n['all_array_dtype_shape_content_identical']
    assert r['all_original_arrays_independently_checked'] and r['all_original_core_metrics_events_and_200_resample_intervals_recomputed']
    assert n['checkpoint_load_audit']['all_keys_strictly_loaded'] and n['checkpoint_load_audit']['parameter_count']==35341512
    assert n['training_updates_performed']==0 and n['native_coverage']==source['native_coverage']
    assert sha(n['scores_path'])==n['scores_sha256']==binding['scores_sha256']
    points={role:sum(g['native_points'] for g in n['native_coverage'] if g['role']==role) for role in ['train','validation','test']}
    files=dict(original_result=ROOT/case['job']['output_directory']/'result.json',native_replay=native_path,independent_review=review_path,
        frozen_original_model_configuration=ROOT/case['job']['model_config'])
    for stage,proof in [('native',native_path),('independent',review_path)]:
        base=ROOT/runtime['state_root']/'attempts'/name/stage;g=read(base/'guard_receipt.json')
        assert g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted'] and g['proof_sha256']==sha(proof)
        assert g['exact_child_handle']['command']==[str(ROOT/case['queue']['python']),'-B','-X','utf8','-u','-m',v.MODULE,
            '--config',runtime_path,'--stage',stage,'--job-id',name]
        files[stage+'_guard']=base/'guard_receipt.json';files[stage+'_launch']=base/'launch.json'
    snapshots[dataset]=dict(files=files,data=data)
    controls=r['independent_confusion_ranking_validation_threshold_PA_and_ten_ratio_controls']
    for ratio,s in source['metrics']['strict'].items():
        c=controls[ratio];ci=s['f1_ci95'];assert c['micro']['f1']==s['micro']['f1']
        prevalence=(s['micro']['true_positives']+s['micro']['false_negatives'])/points['test']
        rows.append(dict(model='MOMENT-1-small',context=512,pretrained_seed=42,dataset=dataset,
            validation_score_quantile_exceedance_percent=float(ratio),threshold=s['calibration']['threshold'],
            micro_no_PA_F1=s['micro']['f1'],macro_no_PA_F1=s['macro_f1'],same_validation_threshold_PA_F1=c['same_validation_threshold_PA_diagnostic']['f1'],
            AUROC=s['micro']['auroc'],AP=s['micro']['average_precision'],test_anomaly_fraction=prevalence,
            precision=s['micro']['precision'],recall=s['micro']['recall'],F1_CI95_low=ci['low'],F1_CI95_high=ci['high'],
            CI_unit=ci['unit'],CI_block_points=ci['block_points'],CI_resamples=ci['resamples'],CI_seed=ci['seed'],
            test_entities=len(data['entities']),test_points=points['test'],events=s['event_metrics']['events'],
            detected_events=s['event_metrics']['detected_events'],event_recall=s['event_metrics']['event_recall'],
            false_alarms_per1000_normal=s['event_metrics']['false_alarms_per_1000_normal_points'],source_track=data['source_track']))
    summaries.append(dict(job_id=name,dataset=dataset,points=points,entities=len(data['entities']),
        role_groups=len(n['native_coverage']),native_array_count=len(n['arrays']),source_binding=binding,
        source_track=data['source_track'],every_original_group_reinferred=True,all_original_arrays_bitwise_equal=True,
        independent_complete_core_review_passed=True,new_training_seed=False,
        historical_author_subset_or_preprocessing_equivalence_certified=False,
        original_execution_guard_history_proven=binding['source_guard_history']['original_execution_guard_history_proven']))

folder.mkdir(parents=True,exist_ok=False)
for dataset,snapshot in snapshots.items():
    out=folder/dataset;out.mkdir()
    for label,path in snapshot['files'].items():
        shutil.copyfile(path,out/(label+'.json'));assert sha(path)==sha(out/(label+'.json'))
    write(out/'actual_registered_dataset_manifest_snapshot.json',snapshot['data'])
shutil.copyfile(ROOT/runtime_path,folder/'frozen_full78_replay_configuration.json')
shutil.copyfile(validation,folder/'unchanged_frozen_native_validation_15_checks.xml')
write(folder/'collector_status_snapshot.json',state)
write(folder/'complete_primary_calibration_and_scope_metrics.json',dict(records=rows,models=summaries,
    additional_complete_checkpoint_dataset_cases=len(selected),official_pretrained_weight_sets=1,new_training_seeds=0,
    complete_all_three_frozen_validation_quantiles_preserved=True,test_F1_threshold_selection=False,
    PA_uses_test_labels_and_is_diagnostic=True,calibration_quantile_not_automatically_normal_only_FAR=True,
    bootstrap_is_fixed_model_fixed_threshold_conditional_uncertainty=True,
    full_TAB_harness_certified=False,historical_paper_equivalence_certified=False,
    preprocessing_values_independently_reconstructed=False,independent_lowlevel_VUS_or_affiliation_integrals_recomputed=False,
    new_guards_retroactively_certify_source_history=False,all_original_experiments_complete=False))
buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(folder/'all_validation_quantile_metrics.csv').write_text(buffer.getvalue(),encoding='utf-8',newline='')
primary=[row for row in rows if row['validation_score_quantile_exceedance_percent']==1.]
lines=['# MOMENT small512：新增数据集的完整原生重推理及独立复核','',
    '全部使用同一35,341,512参数官方预训练权重、context512、原batch1/线程/seed42/补齐规则。每例覆盖全部登记train/validation/test角色组；全部原数组dtype/shape/内容字节相等。第二独立进程核对完整数组、核心指标/事件/原200次区间，并另行计算混淆矩阵、排序、验证分位数阈值、PA和10个pooled诊断。原生及独立阶段均有真实成功退出且无资源中止回执。','',
    '|数据集|测试点|异常点比例|固定验证1%分位未做PA F1|同阈值PA F1|AUROC|AP|','|---|---:|---:|---:|---:|---:|---:|']
for row in primary:
    lines.append(f"|{row['dataset']}|{row['test_points']}|{row['test_anomaly_fraction']:.6f}|{row['micro_no_PA_F1']:.8f}|{row['same_validation_threshold_PA_F1']:.8f}|{row['AUROC']:.8f}|{row['AP']:.8f}|")
lines += ['',
    '0.1%、1%、5%三个预先登记验证阈值全部保存，未从测试F1择优。PA用测试标签，只作诊断；高异常比例本身会提高AP基线，不能把高PA F1或高AP脱离比例解释成强逐点区分能力。比如PRONTO day2同阈值PA可达1.0，未做PA的F1却仅约0.04675。','',
    'CICIDS为冻结mTSBench实体CSV及全部登记通道，不证明原论文相同子集。经典TEP保留整段d00训练、独立d00_te校准及21个d01_te..d21_te故障测试，固定故障起点160；它不等于多模态HDF5版本。SKAB以完整正常训练、事先指定other/1.csv校准及其余全标注实验为测试，校准含原生异常，因此分位数不代表normal-only假警报率。PRONTO使用17个共同过程通道和整日分离角色；训练/校准日的全部连续正常段保留，正常筛选用到标签，不能称完全无标签。各实际文件、实体和选择/归一化边界见每例清单快照。','',
    '200次条件bootstrap、seed10042按多实体cluster或单序列circular block1024进行，每例单位和区间随CSV/JSON保留。它不等于重训种子不确定性或独立算法比较。窗口使用未来上下文，事件延迟不能称在线因果延迟。未独立重建全部预处理值或VUS/Affiliation底层积分；新执行保护不能追溯补证缺失的原运行历史。官方零样本模型完整复核不同于Tiny/ICM整轮训练、完整TAB执行器或历史论文精确复现。','',
    f'本轮新增{len(selected)}个完整权重/数据集复核案例，新增训练种子数为0；复用未变冻结源码的15项相关验证，未重跑已完成模型。快照时主队列已通过{state["verified_count"]}/78（72原始+6控制），全部剩余案例及45篇/681方法记录/273数据任务和新增实验继续保留，研究目标未完成。']
(folder/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),new_complete_checkpoint_dataset_cases=len(selected),
    reused_frozen_validation_tests=15,new_training_seeds=0,files_sha256={f.relative_to(folder).as_posix():sha(f) for f in sorted(folder.rglob('*')) if f.is_file()},
    all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/MOMENT_small512_additional_datasets_full_native_closed.2026-10-15_v1.json'
write(ara,dict(method='MOMENT-1-small',context=512,official_pretrained_seed=42,
    evidence=folder.relative_to(ROOT).as_posix(),complete_native_and_independent_cases=selected,
    new_training_seeds=0,complete_native_queue_required=78,full_TAB_or_historical_paper_equivalence_certified=False,all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix()];paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/MOMENT_small512_additional_datasets_publication_manifest_20261015_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(new_native_cases=len(selected),datasets=[x['dataset'] for x in primary],collector_closed=state['verified_count'],required=78,
    exact_files=len(paths),new_training_seeds=0)))
