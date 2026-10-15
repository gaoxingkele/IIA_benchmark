from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil,subprocess,sys
import xml.etree.ElementTree as ET
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.flow_matching import replay_original_full15_tranad_v1 as s

ROOT=s.ROOT
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def copy(path,dest):
    Path(dest).parent.mkdir(parents=True,exist_ok=True);assert not Path(dest).exists()
    shutil.copyfile(path,dest);assert s.sha(path)==s.sha(dest)
c=s.read(ROOT/s.CONFIG);cases=s.verify(c)
identity='fm_strict_tranad__smap__reference_float64__seed1103__original_full109_v2'
case=next(x for x in cases if x['id']==identity)
upstream=s.ready_scalar(c,case);native_path,native=s.ready_native(c,case)
queue,job,result,dataset,binding=s.scalar.source_binding(c,case)
assert native['source_binding']==binding and native['upstream_scalar_evidence']==upstream
assert native['parameters']==58073 and len(native['complete_restored_state_tensors'])==51
assert len(native['source_arrays'])==216 and len(native['full_native_coverage'])==162
assert native['scores_sha256']==s.sha(native['scores_path'])==result['scores_sha256']
review_attempt,review_output=s.paths(c,case,'review');review_path=review_output/s.REPORTS['review'];review=s.read(review_path)
assert review['passed'] and review['case']==case and review['configuration_sha256']==s.sha(ROOT/s.CONFIG)
assert review['source_binding']==binding and review['native_proof_sha256']==s.sha(native_path)
assert review['all_original_and_native_arrays_independently_compared'] and review['all_original_labels_metrics_events_bootstrap_and_grid_recomputed']
old=s.read(upstream['path']);assert review['all_label_view_results']==old['label_view_results']
assert review['native_source_array_digests']==native['source_arrays']==old['source_array_digests']
assert not review['new_training_or_model_inference_in_review'] and not review['independent_attention_kernel_implementation']
roles={role:sum(r['points'] for r in native['full_native_coverage'] if r['role']==role) for role in ['train','validation','test']}
assert roles['test']==435826 and len(dataset['entities'])==54
state=s.read(ROOT/c['state_root']/'status.json');launch=s.read(ROOT/c['state_root']/'launch.json')
p=psutil.Process(launch['pid']);assert p.create_time()==launch['create_time']==state['create_time'] and p.cmdline()==launch['command']
assert identity in state['completed_cases'] and not state['failed_replays_preserved'] and not state['source_failures_preserved']
validation=ROOT/'tmp/original_full15_tranad_replay_validation_20261016_v1.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'));assert sum(int(x.get('tests','0')) for x in suites)==37
assert all(int(x.get(k,'0'))==0 for x in suites for k in ['failures','errors','skipped'])
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/full15_TranAD_registered_and_SMAP54_seed1103_native_closed_v1'
folder.mkdir(parents=True,exist_ok=False)
for name,path in {'original_native_runner.py':ROOT/(s.MODULE.replace('.','/')+'.py'),'original_native_tests.py':ROOT/s.TEST,
    'frozen_all15_native_configuration.json':ROOT/s.CONFIG,'validation37.xml':validation,
    'actual_native_full_checkpoint_reinference.json':native_path,'actual_full_independent_label_review.json':review_path,
    'actual_upstream_scalar_review.json':Path(upstream['path']),'actual_original_training_and_scoring_result.json':Path(binding['result_path']),
    'exact_native_controller_launch.json':ROOT/c['state_root']/'launch.json'}.items():copy(path,folder/name)
for phase,proof in [('native',native_path),('review',review_path)]:
    attempt,_=s.paths(c,case,phase);guard=s.read(attempt/'guard_receipt.json')
    assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['proof_sha256']==s.sha(proof)
    assert guard['exact_child_handle']['command']==s.worker_command(case,phase)
    copy(attempt/'guard_receipt.json',folder/(phase+'_successful_guard.json'));copy(attempt/'launch.json',folder/(phase+'_exact_worker_launch.json'))
write(folder/'actual_complete_first_model_and_pending_scope.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    actual_case=case,complete_entities=54,full_source_role_points=roles,
    complete_window_count=sum(x['windows'] for x in native['full_native_coverage']),
    model_parameters=58073,restored_parameter_and_buffer_tensors=51,
    all216score_label_arrays_exact_native_dtype_shape_and_bytes=True,
    native_score_archive_SHA256_equals_original_source_archive=True,complete_native_inference_and_independent_review=True,
    actual_label_views=4,all_fixed_validation_quantile_views=12,additional_training_seeds=0,additional_optimizer_updates=0,
    native_runner_required_cases=15,native_runner_required_label_views=35,all_family_fullraw_required_cases=873,
    all_family_label_view_results=2037,other858model_native_verifications_complete=False,
    required_case_matrix=cases,registered_case_counts=dict(Counter(x['dataset'] for x in cases)),
    current_native_controller=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()),state_snapshot=state,
    five_seed_paired_statistics_complete=False,full_TAB_and_historical_author_equivalence=False,all_original_experiments_complete=False))
rows=[]
for label,v in review['all_label_view_results'].items():
    for ratio,metrics in v['metrics']['strict'].items():
        independent=v['independent_confusion_ranking_PA_and_all_grid_controls'][ratio]
        rows.append(dict(policy=label,validation_exceedance_percent=float(ratio),threshold=metrics['calibration']['threshold'],
            no_PA_F1=metrics['micro']['f1'],same_validation_threshold_PA_F1=independent['same_validation_threshold_PA_diagnostic']['f1'],
            AUROC=metrics['micro']['auroc'],AP=metrics['micro']['average_precision'],event_metrics=metrics['event_metrics'],
            conditional_entity_CI95=metrics['f1_ci95']))
assert len(rows)==12
write(folder/'all12_native_backed_label_quantile_metric_rows.json',dict(rows=rows,primary_policy='source_first_exclusive',
    primary_validation_exceedance_percent=1,policy_selected_by_test_score=False,five_training_seed_uncertainty_complete=False))
(folder/'README.md').write_text('''# 全量原始SMAP54实体的首项模型回放关闭；全部15项TranAD回放已接入

本轮真实完成 FP64 TranAD / SMAP54实体 / seed1103 的完整模型重新加载、重新推理及独立指标复核，补上前一保存分数评估的缺口。没有重复训练或新增种子。原源端训练、全量109实体数据准备、4标签评估、此次原生重推理和另一个独立复核进程之间的任务身份、配置、数据/权重/分数/报告哈希均重新核对。

58073参数模型的全部51个参数/缓冲张量按原dtype、shape和完整strict keys恢复，推理前后内容保持完全相等。原window10、batch128、float64评分、2CPU线程、原尾窗覆盖均保留；54实体的全部162个TRAIN/验证/TEST角色被重新推理，TEST覆盖435826点。全部216个分数/标签数组与原文件逐字节一致，完整保存和回读通过；本例连NPZ容器SHA256也与原始分数文件相同。没有仅抽查若干窗口或按误差容差接受不同分数。

另一个新进程比较每个源端/原生数组及原始主标签，再精确重现4种标签视图×全部3个冻结验证分位的12组核心、事件、原200次实体bootstrap区间和全部10项阈值网格指标；独立混淆计数、排名、分段PA及阈值检查一致。原生和复核阶段守卫均成功退出，无资源中止，未初始化CUDA。原生计算使用冻结参考实现和Torch内核，未声称重新独立实现注意力数学。

|预登记主标签与验证1%阈值|无PA F1|同阈值PA F1|AUROC|AP|
|---|---:|---:|---:|---:|
|first / exclusive|0.24797686|0.54916865|0.60916790|0.19644571|

完整其他标注视图和验证0.1/1/5分位结果一并保留，未根据TEST选择标签、方向或阈值。主F1的条件实体bootstrap95%区间为[0.02425172,0.42886127]，不替代五个训练种子的配对统计。原5epoch配置和训练成功守卫保留，但源结果没有提供独立梯度更新计数。当前是验证分位协议，作者padding、预处理、POT和历史环境尚未端到端对齐；完整TAB/VUS/Affiliation仍待完成。

整个新原始数据轨道包含873项实验、2037组标签视图。本次实现并启动的是其中全部15项FP64 TranAD的原生重推理及35组标签视图复核（3数据集×5种子；SMAP4/MSL2/SMD1标签），目前首项完整关闭，其余源任务完成后自动接续。其他858项的方法特定原生验证仍需完成，不由本组件替代。原45论文、681方法记录、273数据任务和702篇内标量指标的总体范围继续保留。

37项验证全部通过，包含SMAP/MSL/SMD特征几何、错误权重/精度/预算/种子/守卫拒绝及禁止拟合的分数恢复；小型神经测试夹具不是benchmark成绩。沿用既有TranAD只读原生审计的16GiB物理、28GiB提交余量、16GiB紧急线、3/6GiB RSS/private上限和原审计锁，未改活跃训练或放松门禁。大数组和权重继续保存在F盘，Git只保存证据与代码。

来源：[TranAD论文](https://doi.org/10.14778/3514061.3514067)、仓库内已冻结的作者代码与配置引用。整体研究目标仍未完成。
''',encoding='utf-8')
copy(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    tests_passed=37,new_complete_model_native_and_independent_replays=1,registered_native_cases=15,
    files_sha256={p.relative_to(folder).as_posix():s.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/full15_TranAD_registered_and_actual_first_native_closure.2026-10-16_v1.json'
write(ara,dict(evidence=folder.relative_to(ROOT).as_posix(),registered_family_native_cases=15,registered_family_label_views=35,
    actual_closed_model_native_and_independent_replays=1,actual_complete_label_views=4,new_training_seeds=0,
    original_scope=c['original_scope'],other858native_cases_complete=False,all_original_experiments_complete=False))
paths=[s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG,ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/full15_TranAD_first_native_closure_publication_manifest_20261016_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),actual_complete_native_models=1,registered_family_native_cases=15,
    array_count=216,state_tensors=51,roles=roles,all_windows=sum(x['windows'] for x in native['full_native_coverage']),
    tests_passed=37,all_original_experiments_complete=False)))
