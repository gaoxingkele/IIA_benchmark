"""Curate all three actual full native/independent official small512 completions."""
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
state=read(ROOT/runtime['state_root']/'status.json')
names=[f'fm_moment_pretrained_small__release512__{dataset}__seed42' for dataset in ['psm','smap','msl']]
assert set(names).issubset(state['verified_ids']) and state['required_jobs']==78
validation=ROOT/'tmp/MOMENT_small512_PSM42_native_closure_validation_20261015_v1.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==15
assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_small512_first3_full_native_replays_v1'
assert not folder.exists();records=[];originals={};summary=[]
for name in names:
    case=cases[name];source,dataset,binding=v.check_source(runtime,case)
    directory=Path(runtime['artifact_root'])/name;npath=directory/'native/native_replay.json';rpath=directory/'independent/review.json'
    native,review=read(npath),read(rpath)
    assert native['passed'] and review['passed'] and native['job']==review['job']==case['job']
    assert native['source_binding']==review['source_binding']==binding
    assert native['runtime_sha256']==sha(ROOT/config_path) and review['native_replay_sha256']==sha(npath)
    assert native['all_original_groups_reinferred'] and native['all_array_dtype_shape_content_identical']
    assert review['all_original_arrays_independently_checked'] and review['all_original_core_metrics_events_and_200_resample_intervals_recomputed']
    assert native['checkpoint_load_audit']['all_keys_strictly_loaded'] and native['checkpoint_load_audit']['parameter_count']==35341512
    assert native['native_coverage']==source['native_coverage'] and native['training_updates_performed']==0
    assert sha(native['scores_path'])==native['scores_sha256']==binding['scores_sha256']
    artifact=dict(original_result=ROOT/case['job']['output_directory']/'result.json',
        frozen_model_configuration=ROOT/case['job']['model_config'],native_replay=npath,independent_review=rpath)
    for phase,path in [('native',npath),('independent',rpath)]:
        base=ROOT/runtime['state_root']/'attempts'/name/phase;guard=read(base/'guard_receipt.json')
        assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['proof_sha256']==sha(path)
        assert guard['exact_child_handle']['command']==[str(ROOT/case['queue']['python']),'-B','-X','utf8','-u','-m',v.MODULE,'--config',config_path,'--stage',phase,'--job-id',name]
        artifact[phase+'_guard']=base/'guard_receipt.json';artifact[phase+'_launch']=base/'launch.json'
    originals[name]=artifact
    controls=review['independent_confusion_ranking_validation_threshold_PA_and_ten_ratio_controls']
    for ratio,value in source['metrics']['strict'].items():
        c=controls[ratio];ci=value['f1_ci95'];assert c['micro']['f1']==value['micro']['f1'] and c['threshold']==value['calibration']['threshold']
        records.append(dict(model='MOMENT-1-small',dataset=case['job']['dataset'].upper(),context=512,seed=42,
            calibration='validation_only',false_alarm_percent=float(ratio),threshold=c['threshold'],no_PA_F1=c['micro']['f1'],
            same_threshold_PA_F1=c['same_validation_threshold_PA_diagnostic']['f1'],AUROC=c['micro']['auroc'],AP=c['micro']['average_precision'],
            precision=c['micro']['precision'],recall=c['micro']['recall'],F1_CI95_low=ci['low'],F1_CI95_high=ci['high'],
            bootstrap_unit=ci['unit'],bootstrap_resamples=ci['resamples'],test_points=sum(g['native_points'] for g in native['native_coverage'] if g['role']=='test'),
            event_recall=value['event_metrics']['event_recall'],events=value['event_metrics']['events'],detected_events=value['event_metrics']['detected_events']))
    summary.append(dict(id=name,dataset=case['job']['dataset'],parameters=35341512,source_binding=binding,
        complete_arrays=len(native['arrays']),role_group_entries=len(native['native_coverage']),
        native_role_points={role:sum(g['native_points'] for g in native['native_coverage'] if g['role']==role) for role in ['train','validation','test']},
        native_MSE_observation=native['inline_full_window_MSE_check'],all_array_bytes_equal=True,
        full_native_and_independent_review_passed=True,source_execution_guard_history=binding['source_guard_history']))
folder.mkdir(parents=True,exist_ok=False)
for name,artifact in originals.items():
    (folder/name).mkdir()
    for label,path in artifact.items():
        target=folder/name/(label+'.json');shutil.copyfile(path,target);assert sha(path)==sha(target)
shutil.copyfile(validation,folder/'validation_15_checks.xml')
shutil.copyfile(ROOT/config_path,folder/'frozen_complete78_replay_configuration.json')
write(folder/'collector_status_snapshot.json',state)
write(folder/'three_complete_native_model_summaries.json',dict(models=summary,newly_closed_models_in_this_packet=2,
    previously_published_model_ids=[names[0]],source_weights_shared=True,new_training_seed=False,
    bootstrap_intervals_conditional_on_fixed_pretrained_model_not_seed_uncertainty=True,
    bootstrap_recalculation_uses_original_library=True,independent_VUS_affiliation_integrals_recalculated=False,
    preprocessing_values_independently_reconstructed=False,historical_paper_equivalence_certified=False,
    full_TAB_harness_certified=False,required_native_jobs=78,original_scope=runtime['original_scope'],all_original_experiments_complete=False))
write(folder/'all_validation_threshold_metrics.json',dict(records=records))
buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=list(records[0]),lineterminator='\n')
writer.writeheader();writer.writerows(records)
(folder/'all9_validation_threshold_metrics.csv').write_text(buffer.getvalue(),encoding='utf-8',newline='')
lines=['# MOMENT small512：PSM、SMAP、MSL 三个完整原生复核\n',
    '三个均实际严格加载同一完整35,341,512参数官方权重，保留context512、batch1、两线程、seed42、所有原生角色/实体/时间点。保存分数与标签等全部数组和原结果dtype/shape/字节一致，分别通过原生重推理及第二独立进程复核，均有成功退出且无资源中止的回执。PSM已在前一包发布；本包新增SMAP、MSL闭合并并列保留三例，没有增加训练种子或重复计算完成数量。\n',
    '|数据集|1%验证阈值未做PA F1|同阈值PA F1|AUROC|AP|未做PA F1 95%区间|\n|---|---:|---:|---:|---:|---|']
for r in records:
    if r['false_alarm_percent']==1:
        lines.append(f"|{r['dataset']}|{r['no_PA_F1']:.8f}|{r['same_threshold_PA_F1']:.8f}|{r['AUROC']:.8f}|{r['AP']:.8f}|[{r['F1_CI95_low']:.8f},{r['F1_CI95_high']:.8f}]|")
lines += ['\n每个来源原先登记的0.1%、1%、5%验证阈值全部保留。表格的1%为固定主协议，没有用测试集挑最优。PA只作同分数/同验证阈值诊断并使用测试标签，不能混作未做PA排名；原全部10项train+test pooled诊断在各原结果中保留。不能把这些PSM/SMAP/MSL small设置当论文SMD特定指标的直接等价复现。\n',
    'PSM为循环时间块、SMAP/MSL为实体簇bootstrap，均200次、同模型与固定验证阈值的条件区间，不是训练种子区间。核心指标/事件/区间用原库重算；混淆矩阵、排序、阈值及PA/pooled网格另行核对。没有声称独立重写bootstrap估计器、VUS/Affiliation积分或全部预处理值。原始执行的保护历史缺失仍如实保留，不由新回执追溯认证；原窗口有未来上下文，不称因果检测。历史论文流程及完整TAB执行器未认证。\n',
    '15项相关验证通过，源文件固定后未变化，本包复用同次原生重推理验证，不重复执行真实模型。完整78项（72原始+6预处理对照）与至少45篇/681方法记录/273数据任务及新增实验范围不变，其余以实际进程和完整证明继续；全部研究尚未完成。大数组及权重留在原F盘。']
(folder/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),actual_native_models=3,new_model_closures=2,
    tests_passed=15,new_training_seeds=0,files_sha256={f.relative_to(folder).as_posix():sha(f) for f in sorted(folder.rglob('*')) if f.is_file()},all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/MOMENT_small512_first3_full_native_replays.2026-10-15_v1.json'
write(ara,dict(model='MOMENT-1-small',context=512,seed=42,datasets=['PSM','SMAP','MSL'],actual_full_checkpoint_replay_and_independent_review=True,
    new_closures=['SMAP','MSL'],evidence=folder.relative_to(ROOT).as_posix(),required_native_cases=78,new_training_seed=False,
    historical_paper_equivalence_certified=False,full_TAB_harness_certified=False,all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix()];paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/MOMENT_small512_first3_full_replays_publication_manifest_20261015_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(actual_native_cases=3,new_closures=2,tests_passed=15,exact_publication_files=len(paths),primary=[r for r in records if r['false_alarm_percent']==1])))
