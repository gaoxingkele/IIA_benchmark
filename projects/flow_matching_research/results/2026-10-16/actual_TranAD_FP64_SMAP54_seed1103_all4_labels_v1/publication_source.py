from datetime import datetime,timezone
import json
from pathlib import Path
import shutil,subprocess,sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.flow_matching import audit_original_full873_label_metrics_v1 as s

ROOT=s.ROOT
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
def copy(path,dest):
    Path(dest).parent.mkdir(parents=True,exist_ok=True);assert not Path(dest).exists()
    shutil.copyfile(path,dest);assert s.sha(path)==s.sha(dest)
c=s.read(ROOT/s.CONFIG);s.verify(c)
identity='fm_strict_tranad__smap__reference_float64__seed1103__original_full109_v2'
case=next(x for x in c['cases'] if x['id']==identity)
queue,job,result,dataset,binding=s.source_binding(c,case)
proof_path=Path(c['artifact_root'])/identity/s.REPORT;proof=s.read(proof_path)
attempt=ROOT/c['state_root']/'attempts'/identity;guard=s.read(attempt/'guard_receipt.json')
assert proof['passed'] and proof['case']==case and proof['source_binding']==binding and proof['configuration_sha256']==s.sha(ROOT/s.CONFIG)
assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['proof_sha256']==s.sha(proof_path)
assert proof['source_metrics_exactly_recomputed'] and proof['all_original_arrays_checked']
assert proof['all_validation_thresholds_and_predictions_unchanged'] and not proof['training_or_model_inference_performed']
assert not proof['independent_model_checkpoint_reload_and_reinference_complete']
assert len(dataset['entities'])==54 and len(proof['full_positional_saved_score_coverage'])==162 and len(proof['source_array_digests'])==216
assert sum(r['points'] for r in proof['full_positional_saved_score_coverage'] if r['role']=='test')==435826
assert len(proof['label_view_results'])==4 and result['parameter_count']==58073
model=s.read(ROOT/job['model_config']);assert model['parameters']['epochs']==5 and case['window']==10 and case['seed']==1103
assert model['precision']=='float64_fit_and_score'
rows=[]
for policy,v in proof['label_view_results'].items():
    labels=proof['label_bindings'][policy];assert s.sha(labels['path'])==labels['sha256']
    assert not v['label_policy_selected_by_TEST_performance'] and v['source_scores_sha256']==result['scores_sha256']
    for ratio,metric in v['metrics']['strict'].items():
        original=result['metrics']['strict'][ratio]
        assert metric['calibration']==original['calibration'] and metric['f1_ci95']['resamples']==200
        assert metric['f1_ci95']['unit']=='entity_cluster'
        independent=v['independent_confusion_ranking_PA_and_all_grid_controls'][ratio]
        if policy==dataset['primary_label_policy']:assert metric==original and v['changed_target_points']==0
        rows.append(dict(label_policy=policy,validation_exceedance_percent=float(ratio),
            unchanged_validation_threshold=metric['calibration']['threshold'],changed_target_points=v['changed_target_points'],
            no_PA_F1=metric['micro']['f1'],same_validation_threshold_PA_F1=independent['same_validation_threshold_PA_diagnostic']['f1'],
            AUROC=metric['micro']['auroc'],AP=metric['micro']['average_precision'],precision=metric['micro']['precision'],recall=metric['micro']['recall'],
            F1_conditional_entity_CI95=metric['f1_ci95'],events=metric['event_metrics']))
assert len(rows)==12
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/actual_TranAD_FP64_SMAP54_seed1103_all4_labels_v1'
folder.mkdir(parents=True,exist_ok=False)
for name,path in {'actual_full4_label_proof.json':proof_path,'successful_label_metric_guard.json':attempt/'guard_receipt.json',
    'exact_label_metric_worker.json':attempt/'launch.json','actual_source_training_and_scoring_result.json':Path(binding['result_path']),
    'actual_original_source_guard.json':Path(case['source_guard']),'frozen_model_configuration.json':ROOT/job['model_config']}.items():copy(path,folder/name)
write(folder/'all12_fixed_validation_quantile_results.json',dict(model='TranAD_FP64_local_reference_control',dataset='SMAP',
    complete_entities=54,complete_TEST_points=435826,seed=1103,parameters=58073,configured_epochs=5,
    persisted_training_loss_or_update_trace=result['training_losses'],source_binding=binding,records=rows,
    configured_budget_and_source_guard_retained=True,observed_gradient_update_count_available=False,
    independent_model_reinference_complete=False,complete_five_seed_pairing=False,historical_author_or_full_TAB_equivalence=False,
    all_original_experiments_complete=False))
validation=ROOT/'tmp/full873_all2037_label_metrics_final_validation_20261016_v1.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'));assert sum(int(x.get('tests','0')) for x in suites)==41
assert all(int(x.get(k,'0'))==0 for x in suites for k in ['failures','errors','skipped'])
copy(validation,folder/'validation41.xml')
(folder/'README.md').write_text('''# 全量SMAP54实体：FP64 TranAD seed1103已完成原训练/评分及全部4种标签视图

这是新增完整原始实体轨道的首项已完成保存分数评估，属于TranAD本地FP64参考控制，不是Flow Matching的成绩。源端训练/评分及原资源守卫均成功，评估器实际重查源端模型、分数、完整数据和标签覆盖哈希，并精确复算原全部指标、事件、200次实体bootstrap区间和10项诊断阈值网格。全部54实体、162个TRAIN/验证/TEST角色、435826个TEST点、216个分数/标签数组已检查；4种标注视图×3个冻结验证分位共12组结果完整保留。

|原始标注视图|验证1%无PA F1|同阈值PA F1|AUROC|AP|变化目标点|
|---|---:|---:|---:|---:|---:|
|first/exclusive（预登记主指标）|0.24797686|0.54916865|0.60916790|0.19644571|0|
|first/inclusive|0.24794195|0.54913803|0.60912043|0.19661960|68|
|last/exclusive|0.24829508|0.54820967|0.60836106|0.19590111|205|
|last/inclusive|0.24825986|0.54817978|0.60831428|0.19607500|271|

同一分数哈希和验证阈值3.4586169685778083用于全部标签视图，没有按TEST选最优标注、倒转分数或修改阈值。此例中端点/重复行差异的F1影响远小于PA与无PA的差异；只代表这个单模型，不能外推为所有模型的结论。主指标实体bootstrap95%区间为[0.02425172,0.42886127]，是条件于固定模型/阈值的区间，不是完整五训练种子区间。

原配置保持58073参数、window10、5epoch和seed1103。原result中的training_losses字段为None；本包没有独立实测的梯度更新计数，不能把配置和成功守卫当作该计数。该参考精度轨道保持作者double()选择，输入仍是预先登记的完整原始实体fit-only归一化控制。作者历史的padding、预处理、POT及环境未由此端到端对齐；当前阈值是验证分位，不是作者POT成绩。

这是源端保存分数的独立指标检查；checkpoint重新加载和重新推理仍未完成，也没有认证完整TAB、VUS/Affiliation底层积分或历史论文指标可复现。全部873项实验/2037组视图还需继续，不能由这一项取代。41项相关单元验证通过，完整实际评估worker及守卫成功，不是合成smoke结果。
''',encoding='utf-8')
copy(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),actual_complete_source_cases=1,
    actual_label_views=4,all_fixed_quantile_records=12,unit_validation_checks=41,
    files_sha256={p.relative_to(folder).as_posix():s.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/actual_TranAD_FP64_SMAP54_seed1103_all4_labels.2026-10-16_v1.json'
write(ara,dict(evidence=folder.relative_to(ROOT).as_posix(),model_dataset_seed_cases=1,full_SMAP_entities=54,label_views=4,
    source_guard_and_all_saved_metrics_passed=True,independent_model_reinference_complete=False,all_original_experiments_complete=False))
paths=[p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]+[ara.relative_to(ROOT).as_posix()]
git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/actual_TranAD54_all4labels_publication_manifest_20261016_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),actual_original_source_and_scalar_review_cases=1,actual_full_label_views=4,
    independent_native_reinference_complete=False)))
