"""Archive all five datasets/five seeds for one fixed FM score configuration."""
from collections import defaultdict
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
owner=psutil.Process(state['pid']);assert owner.is_running() and owner.create_time()==state['create_time']==launch['create_time']
assert not state['failed_preserved']
ids=sorted(cid for cid in source_bindings if cid.startswith('fm_objective_independent__industrial_v1__'))
assert len(ids)==25 and set(ids).issubset(state['verified_ids'])
records={};bindings={};controls=[];oracles=[]
for cid in ids:
    path=Path(c['artifact_root'])/'conditions'/(cid+'.json');guard_path=base/'attempts'/cid/'receipt.json'
    r,g=read(path),read(guard_path)
    assert r['passed'] and g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted'] and g['proof_sha256']==sha(path)
    assert r['source_all_ten_diagnostic_threshold_and_PA_formulas_independently_verified']
    assert r['original_primary_threshold_and_score_direction_unchanged'] and r['new_neural_forwards']==r['new_optimizer_updates']==0
    old=source_bindings[cid]
    assert r['source_result_sha256']==old['source_result_sha256'] and r['source_score_archive_sha256']==old['scores_sha256']
    assert sha(ROOT/old['source_result_path'])==old['source_result_sha256']
    bindings[cid]=dict(report_path=str(path),report_sha256=sha(path),guard_path=str(guard_path),guard_sha256=sha(guard_path),
        source_result_sha256=r['source_result_sha256'],scores_sha256=r['source_score_archive_sha256'])
    records[cid]=r
    for threshold_source,items in [('validation',[(key,value) for key,value in r['strict_validation_controls'].items()]),
        ('pooled_train_and_test_scores',[(str(value['ratio_percent']),value) for value in r['all_ten_test_pool_diagnostic_controls']])]:
        for ratio,item in items:
            for adjustment in ['no_PA','with_PA']:
                controls.append(dict(canonical_id=cid,dataset=r['dataset'],seed=r['seed'],threshold_source=threshold_source,
                    alarm_percent=ratio,point_adjustment=adjustment=='with_PA',threshold=item['threshold'],
                    threshold_uses_test_scores=threshold_source!='validation',threshold_selected_using_test_labels=False,
                    PA_uses_test_ground_truth_events=adjustment=='with_PA',
                    primary_strict_protocol=threshold_source=='validation' and adjustment=='no_PA',
                    **item[adjustment],ground_truth_events=item['ground_truth_events'],detected_events=item['detected_events']))
    oracle=r['test_label_selected_best_diagnostic']
    assert oracle['test_labels_used_for_selection'] and not oracle['primary_performance']
    oracles.append(dict(canonical_id=cid,dataset=r['dataset'],seed=r['seed'],**oracle))
assert len(controls)==650 and len(oracles)==25
groups=defaultdict(list)
for r in controls:groups[r['dataset'],r['threshold_source'],r['alarm_percent'],r['point_adjustment']].append(r)
summary=[]
for (dataset,threshold_source,ratio,PA),values in sorted(groups.items()):
    assert sorted(v['seed'] for v in values)==[1103,1104,1105,1106,1107]
    row=dict(dataset=dataset,threshold_source=threshold_source,alarm_percent=ratio,point_adjustment=PA,complete_training_seeds=5,
        strict_primary_protocol=threshold_source=='validation' and not PA)
    for metric in ['precision','recall','f1']:
        data=[v[metric] for v in values];mean=statistics.mean(data);sd=statistics.stdev(data);half=2.7764451051977987*sd/math.sqrt(5)
        row.update({metric+'_mean':mean,metric+'_sample_std':sd,metric+'_min':min(data),metric+'_max':max(data),metric+'_t95_low':mean-half,metric+'_t95_high':mean+half})
    summary.append(row)
assert len(summary)==130
comparisons=[]
for dataset in ['tep_classic','skab','pronto_day4','pronto_day2','pronto_day3']:
    row=dict(dataset=dataset,complete_training_seeds=5)
    for label,threshold_source,PA in [('validation1_no_PA','validation',False),('validation1_with_PA','validation',True),
        ('pooled_train_test1_no_PA','pooled_train_and_test_scores',False),('pooled_train_test1_with_PA','pooled_train_and_test_scores',True)]:
        value=next(v for v in summary if v['dataset']==dataset and v['threshold_source']==threshold_source and v['alarm_percent']=='1' and v['point_adjustment']==PA)
        row[label+'_F1_mean']=value['f1_mean'];row[label+'_F1_sample_std']=value['f1_sample_std']
    original=[v for v in oracles if v['dataset']==dataset]
    assert sorted(v['seed'] for v in original)==[1103,1104,1105,1106,1107]
    row['test_label_best_grid_no_PA_F1_mean_diagnostic_only']=statistics.mean(v['no_PA_f1'] for v in original)
    row['test_label_best_grid_with_PA_F1_mean_diagnostic_only']=statistics.mean(v['with_PA_f1'] for v in original)
    comparisons.append(row)
for count,path in [(20,'tmp/industrial_full466_PA_calibration_controls_validation_20261016_v1.xml'),(53,'tmp/industrial_full466_PA_shared53_validation_20261016_v1.xml')]:
    suites=list(ET.parse(ROOT/path).getroot().iter('testsuite'));assert sum(int(v.get('tests',0)) for v in suites)==count
    assert all(int(v.get(k,0))==0 for v in suites for k in ['failures','errors','skipped'])
out=ROOT/'projects/flow_matching_research/results/2026-10-16/industrial_FM25_full5seed_PA_and_calibration_controls_v1';out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8',newline='\n')
for name,rows in [('all25_fixed_score_cases_650_actual_protocol_controls.csv',controls),
    ('all130_complete_five_seed_protocol_summaries.csv',summary),('all25_test_label_selected_oracles_diagnostic_only.csv',oracles),
    ('all5_dataset_same_score_main_protocol_comparisons.csv',comparisons)]:
    with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write(out/'all25_full_score_control_and_guard_bindings.json',dict(records=bindings))
write(out/'actual_live_full466_collector_snapshot.json',state)
for name,path in dict(frozen_full466_control_registration=ROOT/s.CONFIG,actual_collector_launch=base/'launch.json',
    relevant_fresh20_PA_control_tests=ROOT/'tmp/industrial_full466_PA_calibration_controls_validation_20261016_v1.xml',
    relevant_fresh53_shared_tests=ROOT/'tmp/industrial_full466_PA_shared53_validation_20261016_v1.xml').items():shutil.copyfile(path,out/(name+path.suffix))
write(out/'completion_boundary.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_global_scope=c['original_global_scope'],
    required_original_CPU_scope=550,required_original_GPU_scope=850,full_fixed_score_control_scope=466,
    actual_closed_fixed_FM_score_conditions=25,datasets=5,training_seeds_per_dataset=5,protocol_decision_rows=650,
    complete_five_seed_protocol_summary_rows=130,validation_alarm_percent=[.1,1,5],test_pool_diagnostic_alarm_percent=s.RATIOS,
    test_label_selected_best_results_are_not_primary=True,PA_uses_original_ground_truth_anomaly_intervals=True,
    other441_fixed_score_conditions_still_required=True,primary_thresholds_and_score_direction_unchanged=True,
    source_model_or_checkpoint_reinference=False,new_training_seed_slots=0,new_neural_forwards=0,new_optimizer_updates=0,
    exact_historical_author_protocol_certification=False,full_TAB_equivalence=False,all_original_experiments_complete=False))
lines=['# 同一FM分数：五数据集×五种子的PA及阈值来源对照','',
    '固定本地FM独立目标配置的25个原完整模型分数，原五个数据划分和1103..1107五种子全部保留；不重训、不运行新神经推理、不重新选原严格阈值或分数方向。',
    '每项按原始实体分别计算三种验证预算和十种原训练/测试分数混合校准预算，各保留PA前后，650组完整点级计数/指标。原十种混合校准阈值、PA公式和最佳网格值全部经独立公式复算一致。',
    '验证0.1%/1%/5%的阈值不用测试特征或标签；PA在测试异常段内把任意一次命中扩展到整个段，因此使用测试真值。混合校准使用训练与测试分数，最佳网格再使用测试标签挑选，全部标为诊断。','',
    '| 划分 | 验证1% 无PA | 相同验证阈值+PA | 混合分数1% 无PA | 混合分数1%+PA | 测试标签选最佳网格+PA（诊断） |',
    '|---|---:|---:|---:|---:|---:|']
for r in comparisons:
    lines.append(f"| {r['dataset']} | {r['validation1_no_PA_F1_mean']:.6f} | {r['validation1_with_PA_F1_mean']:.6f} | {r['pooled_train_test1_no_PA_F1_mean']:.6f} | {r['pooled_train_test1_with_PA_F1_mean']:.6f} | {r['test_label_best_grid_with_PA_F1_mean_diagnostic_only']:.6f} |")
lines+=['','表中均为完整5种子平均，原始逐种子650行、130组STD/min/max/未截断df4 t区间分别列出；测试标签选最佳网格的均值只是诊断，不是无偏测试性能。',
    'TEP与PRONTO day3在完全相同分数下，PA后可以接近或超过0.97；这说明很高的F1本身不证明排序或点级检测同样好。PA只改二值检测记账，不改变原连续分数的AUROC/AP。',
    '混合测试分数校准不保证总是提高F1：TEP本批的原十项网格+PA最佳值低于原验证阈值+PA，而PRONTO多个划分上可以明显提高。不能把所有高指标单独归因于“用了测试集”。',
    '这项实验解释本地同分数协议差异，不据此断言任何指定论文作者使用了错误代码、泄漏或等同协议；作者原代码与论文声称仍按各自原轨复核。',
    '独立实现按实体扫描异常段；首尾异常段、不同实体边界、严格大于阈值和并列分数、1000点uint8标签都经过测试，避免段间串接或计数溢出。',
    '20项本模块及53项相关检查通过；对照队列继续覆盖全部466个已有完整条件，其他441项仍保留。原CPU550/GPU850及45/681/273/702全局范围保持，整体未完成。','']
(out/'README.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
shutil.copyfile(__file__,out/'publication_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/industrial_FM25_PA_calibration_controls.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(packet=out.relative_to(ROOT).as_posix(),fixed_score_conditions=25,required_condition_scope=466,
    full_five_seed_dataset_groups=5,actual_protocol_control_rows=650,new_neural_forwards=0,all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8')
text+='''
`FM同分数PA与阈值来源控制：五数据集×五种子 <results/2026-10-16/industrial_FM25_full5seed_PA_and_calibration_controls_v1/README.md>`_
25个固定源模型分数、650组点级决策、130组完整五种子统计；原十项混合校准网格与PA值独立复算一致。
验证阈值、PA记账和测试标签选最佳网格分开报告；同一分数可以因PA得到0.97以上，但混合校准并非总提高分数。
20项及53项相关测试通过，全部466条件对照继续执行，不重训、不改原严格阈值、不把诊断冒充主结果；原完整范围保持。
'''
page.write_text(text,encoding='utf-8',newline='\n')
paths=[s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG,ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()]
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
write(ROOT/'tmp/industrial_FM25_PA_controls_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(out),files=len(paths),current_verified=len(state['verified_ids']),comparisons=comparisons)))
