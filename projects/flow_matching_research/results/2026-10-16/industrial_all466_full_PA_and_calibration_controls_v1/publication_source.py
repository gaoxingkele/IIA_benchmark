"""Close the entire frozen466-case PA/calibration obligation after all guards."""
from collections import defaultdict,Counter
from datetime import datetime,timezone
import csv
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import audit_industrial_full466_PA_calibration_controls_v1 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/s.CONFIG);source_bindings,q,manifest=s.verify(c)
base=ROOT/c['state_root'];state=read(base/'status.json');launch=read(base/'launch.json')
assert state['state']=='all466_control_attempts_terminal' and state['active_pid'] is None and not state['failed_preserved']
assert set(state['verified_ids'])==set(source_bindings) and len(state['verified_ids'])==466
records={};bindings={};controls=[];oracles=[]
for cid in sorted(source_bindings):
    path=Path(c['artifact_root'])/'conditions'/(cid+'.json');guard_path=base/'attempts'/cid/'receipt.json'
    r,g=read(path),read(guard_path)
    assert r['passed'] and g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted'] and g['proof_sha256']==sha(path)
    assert r['source_all_ten_diagnostic_threshold_and_PA_formulas_independently_verified']
    assert r['original_primary_threshold_and_score_direction_unchanged'] and r['new_neural_forwards']==r['new_optimizer_updates']==0
    old=source_bindings[cid]
    assert r['source_result_sha256']==old['source_result_sha256'] and r['source_score_archive_sha256']==old['scores_sha256']
    source_path=ROOT/old['source_result_path']
    assert sha(source_path)==old['source_result_sha256'] and sha(source_path.parent/'scores.npz')==old['scores_sha256']
    bindings[cid]=dict(report_path=str(path),report_sha256=sha(path),guard_path=str(guard_path),guard_sha256=sha(guard_path),
        source_result_sha256=r['source_result_sha256'],scores_sha256=r['source_score_archive_sha256'])
    records[cid]=r
    for threshold_source,items in [('validation',list(r['strict_validation_controls'].items())),
        ('pooled_train_and_test_scores',[(str(v['ratio_percent']),v) for v in r['all_ten_test_pool_diagnostic_controls']])]:
        for ratio,item in items:
            for adjustment in ['no_PA','with_PA']:
                controls.append(dict(canonical_id=cid,dataset=r['dataset'],seed=r['seed'],threshold_source=threshold_source,
                    alarm_percent=ratio,point_adjustment=adjustment=='with_PA',threshold=item['threshold'],
                    threshold_uses_test_scores=threshold_source!='validation',threshold_selected_using_test_labels=False,
                    PA_uses_test_ground_truth_events=adjustment=='with_PA',
                    primary_strict_protocol=threshold_source=='validation' and adjustment=='no_PA',
                    **item[adjustment],ground_truth_events=item['ground_truth_events'],detected_events=item['detected_events'],
                    actual_primary_evaluation=threshold_source=='validation' and ratio=='1' and adjustment=='no_PA'))
    oracle=r['test_label_selected_best_diagnostic']
    assert oracle['test_labels_used_for_selection'] and not oracle['primary_performance']
    oracles.append(dict(canonical_id=cid,dataset=r['dataset'],seed=r['seed'],**oracle))
assert len(controls)==12116 and len(oracles)==466
prior=ROOT/'projects/flow_matching_research/results/2026-10-16/industrial_FM25_full5seed_PA_and_calibration_controls_v1'
old_bindings=read(prior/'all25_full_score_control_and_guard_bindings.json')['records']
assert all(bindings[cid]==v for cid,v in old_bindings.items())
with (prior/'all25_fixed_score_cases_650_actual_protocol_controls.csv').open(encoding='utf-8-sig',newline='') as f:old_rows=list(csv.DictReader(f))
index={(r['canonical_id'],r['threshold_source'],r['alarm_percent'],str(r['point_adjustment'])):r for r in controls}
assert len(old_rows)==650
for old in old_rows:
    new=index[old['canonical_id'],old['threshold_source'],old['alarm_percent'],old['point_adjustment']]
    assert {k:str(new[k]) for k in old}==old
groups=defaultdict(list)
for r in controls:groups[records[r['canonical_id']]['model_config'],r['dataset'],r['threshold_source'],r['alarm_percent'],r['point_adjustment']].append(r)
summary=[];partial=[]
for (model,dataset,threshold_source,ratio,PA),values in sorted(groups.items()):
    seeds=sorted(v['seed'] for v in values)
    if seeds!=[1103,1104,1105,1106,1107]:
        partial.append(dict(model_config=model,dataset=dataset,threshold_source=threshold_source,alarm_percent=ratio,point_adjustment=PA,
            completed_training_seeds=seeds,required_training_seeds=[1103,1104,1105,1106,1107],statistics_computed=False));continue
    row=dict(model_config=model,dataset=dataset,threshold_source=threshold_source,alarm_percent=ratio,point_adjustment=PA,complete_training_seeds=5,
        strict_primary_protocol=threshold_source=='validation' and not PA,
        actual_primary_evaluation=threshold_source=='validation' and ratio=='1' and not PA)
    for metric in ['precision','recall','f1']:
        data=[v[metric] for v in values];mean=statistics.mean(data);sd=statistics.stdev(data);half=2.7764451051977987*sd/math.sqrt(5)
        row.update({metric+'_mean':mean,metric+'_sample_std':sd,metric+'_min':min(data),metric+'_max':max(data),metric+'_t95_low':mean-half,metric+'_t95_high':mean+half})
    summary.append(row)
assert len(summary)==2314 and len(partial)==156
primary_map={(r['model_config'],r['dataset'],r['threshold_source'],r['alarm_percent'],r['point_adjustment']):r for r in summary}
with (ROOT/s.PACKET/'all89_complete_five_seed_groups_267_fixed_budget_summaries.csv').open(encoding='utf-8-sig',newline='') as f:previous=list(csv.DictReader(f))
ranks={(r['model_config'],r['dataset']):r for r in previous if r['validation_alarm_percent']=='1'}
comparisons=[]
for (model,dataset),rank in sorted(ranks.items()):
    row=dict(model_config=model,dataset=dataset,complete_training_seeds=5,original_continuous_score_AUROC_mean=float(rank['auroc_mean']),
        original_continuous_score_AP_mean=float(rank['average_precision_mean']))
    for label,threshold_source,PA in [('validation1_no_PA','validation',False),('validation1_with_PA','validation',True),
        ('pooled_train_test1_no_PA','pooled_train_and_test_scores',False),('pooled_train_test1_with_PA','pooled_train_and_test_scores',True)]:
        value=primary_map[model,dataset,threshold_source,'1',PA]
        row[label+'_F1_mean']=value['f1_mean'];row[label+'_F1_sample_std']=value['f1_sample_std']
    values=[r['test_label_selected_best_diagnostic'] for r in records.values() if r['model_config']==model and r['dataset']==dataset]
    assert len(values)==5
    row['test_label_best_grid_no_PA_F1_mean_diagnostic_only']=statistics.mean(v['no_PA_f1'] for v in values)
    row['test_label_best_grid_with_PA_F1_mean_diagnostic_only']=statistics.mean(v['with_PA_f1'] for v in values)
    row['PA_gain_at_same_validation1_threshold']=row['validation1_with_PA_F1_mean']-row['validation1_no_PA_F1_mean']
    row['testpool1_minus_validation1_no_PA_F1']=row['pooled_train_test1_no_PA_F1_mean']-row['validation1_no_PA_F1_mean']
    comparisons.append(row)
assert len(comparisons)==89
low_rank_high_PA=[r for r in comparisons if r['original_continuous_score_AUROC_mean']<.5 and r['validation1_with_PA_F1_mean']>.9]
testpool_improves=sum(r['testpool1_minus_validation1_no_PA_F1']>0 for r in comparisons)
testpool_worsens=sum(r['testpool1_minus_validation1_no_PA_F1']<0 for r in comparisons)
assert sum(r['actual_primary_evaluation'] for r in controls)==466
for count,path in [(20,'tmp/industrial_full466_PA_calibration_controls_validation_20261016_v1.xml'),(53,'tmp/industrial_full466_PA_shared53_validation_20261016_v1.xml')]:
    suites=list(ET.parse(ROOT/path).getroot().iter('testsuite'));assert sum(int(v.get('tests',0)) for v in suites)==count
    assert all(int(v.get(k,0))==0 for v in suites for k in ['failures','errors','skipped'])
out=ROOT/'projects/flow_matching_research/results/2026-10-16/industrial_all466_full_PA_and_calibration_controls_v1';out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8',newline='\n')
for name,rows in [('all466_fixed_score_cases_12116_actual_protocol_controls.csv',controls),('all89_groups_2314_complete_five_seed_protocol_summaries.csv',summary),
    ('all466_test_label_selected_oracles_diagnostic_only.csv',oracles),('all89_complete_groups_same_score_protocol_comparisons.csv',comparisons)]:
    with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write(out/'all466_full_score_control_and_guard_bindings.json',dict(records=bindings,prior25_report_guards_and_all650_values_preserved=True))
write(out/'all156_partial_group_controls_without_statistics.json',dict(records=partial,partial_model_dataset_groups=6,controls_per_group=26))
write(out/'original_continuous_score_below_half_and_PA_F1_above_point9.json',dict(exploratory_description_only=True,records=low_rank_high_PA,
    original_score_orientation_not_changed=True,comparison_does_not_identify_historical_author_protocols=True))
for name,path in dict(frozen_full466_control_registration=ROOT/s.CONFIG,actual_collector_launch=base/'launch.json',
    actual_complete_collector_state=base/'status.json',relevant_fresh20_PA_control_tests=ROOT/'tmp/industrial_full466_PA_calibration_controls_validation_20261016_v1.xml',
    relevant_fresh53_shared_tests=ROOT/'tmp/industrial_full466_PA_shared53_validation_20261016_v1.xml').items():shutil.copyfile(path,out/(name+path.suffix))
write(out/'completion_boundary.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_global_scope=c['original_global_scope'],
    original_CPU_scope=550,original_GPU_scope=850,full_fixed_score_control_scope=466,fully_verified_control_cases=466,
    original_CPU_source_cases_still_not_complete=84,model_dataset_complete_five_seed_groups=89,required_groups=110,
    actual_protocol_control_rows=12116,complete_five_seed_summary_rows=2314,partial_controls_without_statistics=156,
    strictly_primary_evaluation_rows=466,primary_setting='validation_only_1_percent_no_PA',
    nominal_validation_point1_and5_percent_are_fixed_sensitivity_controls=True,
    legacy_primary_strict_protocol_field_is_protocol_family_only_use_actual_primary_evaluation_for_unique_primary=True,
    low_AUROC_high_PA_complete_groups=len(low_rank_high_PA),testpool1_no_PA_improves_F1_groups=testpool_improves,
    testpool1_no_PA_worsens_F1_groups=testpool_worsens,
    new_neural_forwards=0,new_optimizer_updates=0,new_training_seed_slots=0,new_complete_model_results=0,
    full_TAB_equivalence=False,historical_author_protocol_equivalence=False,all_original_experiments_complete=False))
lines=['# 全466个固定分数条件：PA与阈值来源完整对照','',
    '466个已通过原始分数审计的不同CPU条件全部完成三种验证预算、十种原混合训练/测试分数预算及各自PA前后对照；12116组完整点级决策均经独立公式和成功资源守卫验收。',
    '全部原十项网格阈值、无PA与PA F1、测试标签挑选最佳网格值与原程序一致；PA按每个原始实体的真值异常段分别记账，段不跨实体。',
    '唯一实际主设置是验证1%且无PA；0.1%和5%是预先固定敏感性控制。actual_primary_evaluation专列恰有466行；保留旧primary_strict_protocol列为严格协议族标记，不能把它误读为三种主结果。',
    '混合训练/测试分数阈值使用测试分数，最佳网格另使用测试标签，PA使用真值异常段，全部分别标明；不改变原主阈值、模型、分数方向或严格基准成绩。','',
    f'89个完整五种子组提供2314组均值/样本STD/min/max/未截断df4 t区间；6个部分组的156个控制不计算统计，15个无完整结果组仍保留在原550范围。',
    f'探索性描述发现{len(low_rank_high_PA)}个完整五种子组的原AUROC均值低于0.5、但同验证1%阈值加PA后的F1均值高于0.9；详细组全部列出，不只展示有利例子。',
    f'在89组内，改为混合训练/测试分数1%且仍无PA时，{testpool_improves}组F1上升、{testpool_worsens}组下降，其余相同。校准来源影响不能概括为必然涨分。','',
    '固定FM独立目标、完整五种子示例：','',
    '| 数据 | 验证1% 无PA | 同阈值加PA | 测试标签选最佳网格+PA（仅诊断） |','|---|---:|---:|---:|']
for dataset in ['tep_classic','skab','pronto_day4','pronto_day2','pronto_day3']:
    row=next(r for r in comparisons if Path(r['model_config']).stem=='fm_objective_independent' and r['dataset']==dataset)
    lines.append(f"| {dataset} | {row['validation1_no_PA_F1_mean']:.6f} | {row['validation1_with_PA_F1_mean']:.6f} | {row['test_label_best_grid_with_PA_F1_mean_diagnostic_only']:.6f} |")
lines+=['','所有89完整组和466逐种子原值均列出，主结果、PA、混合分数校准和测试标签最优选择互相分开，不把后者冒充无偏性能。',
    'PA只改变真值段内二值预测记账，不改变原连续分数AUROC/AP；本包的AUROC/AP取自先前完整独立排序核验，不是给PA预测虚构新的连续分数。',
    '这些是本地评分与迁移配置的同分数机制对照，不由此认定指定论文作者采用了同样策略、数据泄漏或错误代码。原作者流程与论文表格一致性仍须各自原始轨道证明。',
    '首批25证明/守卫哈希及全部650个旧控制字段保持；新增actual_primary_evaluation明确主设置。20项及53项相关检查通过。',
    '源权重与分数未改写，新训练/神经推理/优化更新均0；原CPU550尚余84个完整源条件，GPU850和45/681/273/702范围保持。完整论文及TAB训练策略等价未认证，整体未完成。','']
(out/'README.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
shutil.copyfile(__file__,out/'publication_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/industrial_all466_PA_calibration_controls.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(packet=out.relative_to(ROOT).as_posix(),fully_verified_fixed_score_conditions=466,
    actual_protocol_control_rows=12116,complete_five_seed_protocol_rows=2314,new_neural_forwards=0,all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8')
text+=f'''
`全部466个固定分数条件完成PA与校准来源对照 <results/2026-10-16/industrial_all466_full_PA_and_calibration_controls_v1/README.md>`_
12116组完整点级控制、89个完整五种子组的2314组统计均验收，6部分组的156控制不计算统计；原CPU550/GPU850范围保持。
唯一实际主设置为验证1%无PA；actual_primary_evaluation专列明确，PA/测试分数校准/测试标签最优选择均独立诊断。
{len(low_rank_high_PA)}个完整组原AUROC均值低于0.5但同阈值PA后F1高于0.9；混合校准有升有降，不能把高F1独自当作算法优质或归因于任一作者。
原首批25证明及650控制值保持，20项及53项相关验证通过；不新增训练或推理，不改变主结果，整体未完成。
'''
page.write_text(text,encoding='utf-8',newline='\n')
paths=[ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()]
gethead=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
write(ROOT/'tmp/industrial_all466_PA_controls_publication_manifest_20261016_v1.json',dict(previous_root=gethead(ROOT),previous_parent=gethead(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(out),publication_files=len(paths),verified_cases=466,actual_control_rows=12116,complete_seed_groups=89,
    low_AUROC_high_PA_groups=len(low_rank_high_PA),testpool1_improves=testpool_improves,testpool1_worsens=testpool_worsens,new_complete_models=0)))
