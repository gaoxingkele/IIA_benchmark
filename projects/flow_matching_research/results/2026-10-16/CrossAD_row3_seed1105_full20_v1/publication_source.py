"""Publish only the actual twentieth closure and its fifteen pending controls."""
from pathlib import Path
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import publish_crossad_row3_native_increment_v64 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

packet=ROOT/s.PACKET
for pin in read(packet/'packet_integrity.json')['files']:assert sha(packet/pin['path'])==pin['sha256']
assert read(packet/'iteration_status.json')['CrossAD_full_CUDA_conditions']==20
with (packet/'all9_same_seed_row3_comparisons_with_actual_training_budget.csv').open(encoding='utf-8-sig',newline='') as f:paired=list(csv.DictReader(f))
assert len(paired)==9
primary=sorted([x for x in paired if float(x['validation_alarm_percent'])==1],key=lambda x:int(x['training_seed']))
assert [int(x['training_seed']) for x in primary]==[1103,1104,1105]
for x in paired:
    assert abs(float(x['row3_minus_row2_F1'])-(float(x['row3_F1'])-float(x['nearest_row2_F1'])))<1e-14
    assert abs(float(x['row3_minus_row2_AUROC'])-(float(x['row3_AUROC'])-float(x['nearest_row2_AUROC'])))<1e-14
    assert abs(float(x['row3_minus_row2_AP'])-(float(x['row3_AP'])-float(x['nearest_row2_AP'])))<1e-14
with (packet/'all3_primary_seed_row2_vs_row3_component_controls.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(primary[0]));w.writeheader();w.writerows(primary)
models={}
for i,components in [(2,[True,False,False,False]),(3,[True,True,False,False])]:
    path=f'configs/models/fm_crossad_ablation/fm_crossad_table3_row{i}.json';m=read(ROOT/path)
    assert m['implementation']=='paper_reconstructed_author_harness' and m['parameters']['components']==components
    models['row'+str(i)]=dict(configuration=path,sha256=sha(ROOT/path),components=components)
write(packet/'adjacent_reconstruction_component_comparison_boundary.json',dict(models=models,common_training_seeds=[1103,1104,1105],
    original_primary_training_seeds=[1103,1104,1105,1106,1107],full_five_seed_group_complete=False,
    original_stopping_rule_same_actual_update_counts_separately_reported=True,model_specific_validation_cutoff_values_may_differ=True,
    no_partial_training_seed_aggregate_or_interval=True,author_exact_ablation_implementation_certified=False,new_training_updates=0,new_neural_forwards=0))
binding=read(packet/'all20_proof_bindings.json')['records']['crossad_ablation__row2__gecco__seed1105']
assert sha(binding['independent_metric_report_path'])==binding['independent_metric_report_sha256']
r2=read(binding['independent_metric_report_path'])['independent_strict_metrics']['1']
r3=read(packet/'independent_metric_report.json')['independent_strict_metrics']['1']
confusion_delta={k:r3['micro'][k]-r2['micro'][k] for k in ['true_positives','false_positives','false_negatives']}
write(packet/'seed1105_frozen_validation_rule_confusion_shift.json',dict(seed=1105,validation_alarm_percent=1,
    row2=r2,row3=r3,row3_minus_row2_confusion_counts=confusion_delta,
    thresholds_fitted_separately_on_each_models_validation_scores=True,test_labels_not_used_for_threshold_selection=True,
    point_adjustment=False,partial_five_seed_means_computed=False,pure_component_causality_claim=False))
table='\n'.join(f"| {x['training_seed']} | {float(x['nearest_row2_F1']):.6f} | {float(x['row3_F1']):.6f} | {float(x['nearest_row2_AUROC']):.6f} | {float(x['row3_AUROC']):.6f} | {float(x['nearest_row2_AP']):.6f} | {float(x['row3_AP']):.6f} |" for x in primary)
p=packet/'README.md';p.write_text(p.read_text(encoding='utf-8')+f'''
最近结构对照row2与row3的主要1%验证阈值规则结果，按实际训练种子单列：

| 种子 | row2 F1 | row3 F1 | row2 AUROC | row3 AUROC | row2 AP | row3 AP |
|---|---:|---:|---:|---:|---:|---:|
{table}

两种登记重建配置仅在cross-scale组件开关上不同，完整数据/标签/坐标哈希相同；每个模型使用自身验证分位数阈值，阈值数值不必相等。
排名质量、固定阈值F1和实际早停预算分别报告。参数初始化/结构及实际训练更新可能不同，三种子尚未补齐全部五种子，不能把此表解释成已证实完整论文消融或严格因果结论。

seed1105在该固定验证阈值规则下，row2到row3的TP为{r2['micro']['true_positives']}到{r3['micro']['true_positives']}，FP为{r2['micro']['false_positives']}到{r3['micro']['false_positives']}，FN为{r2['micro']['false_negatives']}到{r3['micro']['false_negatives']}。
增加{confusion_delta['true_positives']}个TP的同时增加{confusion_delta['false_positives']}个FP，Precision由{r2['micro']['precision']:.6f}变为{r3['micro']['precision']:.6f}，Recall由{r2['micro']['recall']:.6f}变为{r3['micro']['recall']:.6f}。
因此本组冻结验证阈值F1的下降可以由完整混淆计数解释，不能据此说排序能力也下降；两模型AUROC/AP和实际阈值均保留。该说明只针对这一实际设置，不把评价结果解释为普遍的组件因果结论。
''',encoding='utf-8',newline='\n')
shutil.copyfile(__file__,packet/'publication_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file() and p.name!='packet_integrity.json']))
assert not (ROOT/s.CONFIG).exists()
with (ROOT/s.CONFIG).open('x',encoding='utf-8') as f:json.dump(s.registration(),f,indent=2)
status=s.publish();out=ROOT/read(ROOT/s.CONFIG)['output_directory']
with (ROOT/s.BASE/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:before=list(csv.DictReader(f))
with (out/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:after=list(csv.DictReader(f))
updates=[b for a,b in zip(before,after) if a!=b]
assert len(updates)==15 and s.merge(before,updates)==after
for name in ['all878_original_TAB_case_scope_and_captured_status.csv','five_shared_original_SMD_comparison_references.csv']:
    assert sha(ROOT/s.BASE/name)==sha(out/name)
xml=ROOT/'tmp/catalog_v64_validation_20261016_v1.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==8 and all(int(x.get('failures',0))==int(x.get('errors',0))==int(x.get('skipped',0))==0 for x in suites)
shutil.copyfile(xml,out/'relevant_fresh8_validation.xml')
shutil.copyfile(ROOT/s.CONFIG,out/'frozen_publication_configuration.json')
shutil.copyfile(__file__,out/'packaging_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8')
for a,b in [
    ('2026-10-16 的 v63 增量快照','2026-10-16 的 v64 增量快照'),
    ('all_algorithm_dataset_metrics_v63/','all_algorithm_dataset_metrics_v64/'),
    ('共11925行指标，3156行有数值','共11925行指标，3171行有数值'),
    ('原v62的11895行全部字段和捕获时间保持；仅填入CrossAD第18与第19条件的30项原待完成指标。','原v63的11910行全部字段和捕获时间保持；仅填入CrossAD第20条件的15项原待完成指标。'),
    ('该快照已核验分别19和8个条件','该快照已核验分别20和8个条件'),
]:
    assert a in text,a;text=text.replace(a,b)
a=read(packet/'independent_metric_report.json')['independent_strict_metrics']['1']['micro']
text+=f'''
`CrossAD row3/seed1105，第20个完整GPU条件 <results/2026-10-16/CrossAD_row3_seed1105_full20_v1/README.md>`_
完整训练、检查点重放和独立全数组指标核验通过；验证1%无PA F1={a['f1']:.12f}，AUROC={a['auroc']:.12f}，AP={a['average_precision']:.12f}。
row3保留多尺度和跨尺度、关闭子序列和全局上下文；row1/row2/row3各3/5训练种子，其他两项各2/5，完整主要组仍1/22。
同种子row2、row3、完整模型的固定阈值与排序指标、全部9模型实际预算和重建边界分别列出，不计算部分五种子均值或区间。
15项相关验证通过，v64另有8项增量保护检查通过；仅填入15项原待完成控制，其他11910行全部字段和捕获时间保持。
全部139条件、原CPU条件及45/681/273/702研究范围不变；该增量不认证历史作者或完整TAB等价，整体未完成。
'''
page.write_text(text,encoding='utf-8',newline='\n')
ara=ROOT/'projects/flow_matching_research/ara/execution/CrossAD_row3_catalog.2026-10-16_v64.json'
assert not ara.exists();write(ara,dict(packet=out.relative_to(ROOT).as_posix(),metric_rows=11925,numeric_rows=3171,
    unchanged_rows_all_fields=11910,new_complete_fixed_case_controls=15,all_original_experiments_complete=False))
manifest=read(ROOT/'tmp/CrossAD_row3_seed1105_full20_publication_manifest_20261016_v1.json')
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
assert head(ROOT)==manifest['previous_root'] and head(ROOT.parent/'flow-matching-research')==manifest['previous_parent']
paths=set(manifest['paths'])|{s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG,page.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix()}
paths|={p.relative_to(ROOT).as_posix() for folder in [packet,out] for p in folder.rglob('*') if p.is_file()}
write(ROOT/'tmp/CrossAD20_and_catalog_v64_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(files=len(paths),full_CrossAD_conditions=20,numeric_rows=3171,primary_component_controls=primary)))
