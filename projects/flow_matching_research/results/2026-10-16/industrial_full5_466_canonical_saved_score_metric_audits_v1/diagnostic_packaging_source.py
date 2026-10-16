"""Read only complete five-seed verified rows; no target tuning or new inference."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import shutil
import statistics
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

out=ROOT/'projects/flow_matching_research/results/2026-10-16/industrial_full5_466_canonical_saved_score_metric_audits_v1'
with (out/'all466_canonical_cases_1398_strict_metric_rows.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
with (out/'all89_complete_five_seed_groups_267_fixed_budget_summaries.csv').open(encoding='utf-8-sig',newline='') as f:summaries=list(csv.DictReader(f))
info=read(out/'all5_dataset_native_scope_and_selection.json')['datasets']
names=['fm_objective_independent','fm_objective_ot','fm_strict_usad__psm__registered','fm_strict_tranad__psm__registered','fm_strict_timesnet__psm__registered']
selected=[];sensitivity=[];paired=[];paired_summary=[]
for dataset,spec in info.items():
    prevalence=spec['positive_test_points']/spec['test_points']
    always=2*spec['positive_test_points']/(spec['test_points']+spec['positive_test_points'])
    for name in names:
        r=next(r for r in summaries if r['dataset']==dataset and Path(r['model_config']).stem==name and r['validation_alarm_percent']=='1')
        assert r['complete_seeds']==r['required_seeds']=='5'
        selected.append(dict(dataset=dataset,model_config=r['model_config'],complete_training_seeds=5,validation_alarm_percent=1,
            **{k:float(r[k]) for k in ['f1_mean','f1_sample_std','auroc_mean','average_precision_mean','actual_test_alarm_fraction_mean','native_normal_false_positive_fraction_mean']},
            test_prevalence=prevalence,constant_score_AP=prevalence,constant_score_AUROC=.5,always_alert_F1=always,
            always_alert_alarm_fraction=1.,prior_controls_are_not_matched_alarm_budget_competitors=True,
            difference_AP_from_constant=float(r['average_precision_mean'])-prevalence))
    for ratio in ['0.1','1','5']:
        r=next(r for r in summaries if r['dataset']==dataset and Path(r['model_config']).stem=='fm_objective_independent' and r['validation_alarm_percent']==ratio)
        sensitivity.append(dict(dataset=dataset,model_config=r['model_config'],complete_training_seeds=5,validation_alarm_percent=ratio,
            **{k:float(r[k]) for k in ['precision_mean','recall_mean','f1_mean','f1_sample_std','auroc_mean','average_precision_mean','actual_test_alarm_fraction_mean','native_normal_false_positive_fraction_mean']}))
    current=[]
    for seed in range(1103,1108):
        a=next(r for r in rows if r['dataset']==dataset and Path(r['model_config']).stem=='fm_objective_independent' and r['validation_alarm_percent']=='1' and int(r['seed'])==seed)
        b=next(r for r in rows if r['dataset']==dataset and Path(r['model_config']).stem=='fm_strict_usad__psm__registered' and r['validation_alarm_percent']=='1' and int(r['seed'])==seed)
        item=dict(dataset=dataset,seed=seed,FM_canonical_id=a['canonical_id'],USAD_canonical_id=b['canonical_id'],validation_alarm_percent=1,
            FM_parameter_count=int(a['parameter_count']),USAD_parameter_count=int(b['parameter_count']),
            FM_configured_epochs=int(a['configured_epochs']),USAD_configured_epochs=int(b['configured_epochs']),
            equal_parameter_or_update_budget_claim=False)
        for key in ['f1','auroc','average_precision','native_normal_false_positive_fraction']:
            av,bv=float(a[key]),float(b[key]);item.update({'FM_'+key:av,'USAD_'+key:bv,'FM_minus_USAD_'+key:av-bv})
        current.append(item);paired.append(item)
    group=dict(dataset=dataset,complete_paired_training_seeds=5,validation_alarm_percent=1)
    for metric in ['f1','auroc','average_precision','native_normal_false_positive_fraction']:
        key='FM_minus_USAD_'+metric;values=[r[key] for r in current]
        mean=statistics.mean(values);sd=statistics.stdev(values);half=2.7764451051977987*sd/math.sqrt(5)
        group.update({key+'_mean':mean,key+'_sample_std':sd,key+'_t95_low':mean-half,key+'_t95_high':mean+half})
    paired_summary.append(group)
assert len(selected)==25 and len(sensitivity)==15 and len(paired)==25 and len(paired_summary)==5
new=[]
for name,records in [('five_fixed_methods_full5seed_prior_and_alarm_controls.csv',selected),
    ('FM_independent_full5seed_all15_frozen_alarm_budget_controls.csv',sensitivity),
    ('FM_USAD_all25_same_dataset_seed_controls.csv',paired),('FM_USAD_all5_complete_paired_seed_summaries.csv',paired_summary)]:
    with (out/name).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    new.append(name)
tep_FM=next(r for r in selected if r['dataset']=='tep_classic' and Path(r['model_config']).stem=='fm_objective_independent')
tep_USAD=next(r for r in selected if r['dataset']=='tep_classic' and Path(r['model_config']).stem=='fm_strict_usad__psm__registered')
lines=['','已核验的性能差异与边界','----------------------','',
    f"TEP验证1%下，FM F1={tep_FM['f1_mean']:.6f}、USAD F1={tep_USAD['f1_mean']:.6f}；FM AUROC={tep_FM['auroc_mean']:.6f}、USAD AUROC={tep_USAD['auroc_mean']:.6f}。正常点误报率分别{100*tep_FM['native_normal_false_positive_fraction_mean']:.4f}%和{100*tep_USAD['native_normal_false_positive_fraction_mean']:.4f}%。F1排序、排序指标与实际误报代价在这里并不一致。",
    'SKAB验证集本身含188个异常点，并且各完整实验运行之间分布不同；原验证1%规则下多种方法的测试正常点误报率约62%，不能把名义验证尾部比例写成测试正常误报率。此处报告分布事实，不据此证明唯一因果机制。',
    'PRONTO day4的FM点级F1约0.0041、AUROC约0.4632，day3的AUROC约0.3778。退化同时出现在多种方法，独立指标和原始预处理核对一致，当前证据不支持把差异解释成一次指标公式计算错误；也不能据此排除其他建模问题。',
    'PRONTO day2异常先验约94.19%，常数排序AP即约0.9419，一直报警F1约0.9701；FM AP约0.9753仍需结合AUROC约0.7171及固定阈值F1约0.1648解释。先验控制仅作诊断，不是相同报警预算的对手。',
    '给出FM在预先固定0.1%、1%、5%验证预算下全部15组完整五种子控制，以及与USAD同划分同种子的25组差值和5个完整配对种子汇总；没有按测试成绩重新选阈值、方向、模型或超参数。',
    '这些是本地评分适配和PSM配置迁移结果。原论文原始评分函数、作者训练流程、PA或测试集校准策略要依据各自单独复现证据核对，不能把本表差异直接归因于所有作者代码错误或指标泄漏。',
    '配对保留原参数量与训练轮数，但结构、窗口和实际更新预算可能不同，配对区间描述这五次初始化条件，不认证容量匹配的组件因果效应。','']
with (out/'README.md').open('a',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines))
write(out/'performance_diagnostic_boundary.json',dict(complete_method_dataset_groups=25,fixed_validation_alarm_sensitivity_rows=15,
    complete_same_seed_FM_USAD_pairs=25,complete_five_seed_paired_groups=5,parameter_or_actual_update_budget_matched=False,
    score_direction_or_threshold_selected_using_test_labels=False,source_paper_error_or_leakage_inferred_from_this_table=False,
    original_source_scores_unchanged=True,new_neural_forwards=0,new_optimizer_updates=0))
new+=['performance_diagnostic_boundary.json','diagnostic_packaging_source.py']
shutil.copyfile(__file__,out/'diagnostic_packaging_source.py')
integrity=out/'packet_integrity.json'
write(integrity,dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p!=integrity]))
manifest=ROOT/'tmp/industrial_full5_466_canonical_metric_publication_manifest_20261016_v1.json';m=read(manifest)
m['paths']=sorted(set(m['paths'])|{(out/name).relative_to(ROOT).as_posix() for name in new});write(manifest,m)
print(json.dumps(dict(complete_method_dataset_controls=len(selected),complete_seed_pairs=len(paired),complete_paired_groups=len(paired_summary))))
