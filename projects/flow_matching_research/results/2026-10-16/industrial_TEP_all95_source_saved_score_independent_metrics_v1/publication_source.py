"""Close every available complete TEP source result; keep missing15/full550."""
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

from scripts.flow_matching import audit_industrial_full550_independent_metrics_v1 as audit
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/audit.CONFIG);q,manifest=audit.verify(c)
base=ROOT/c['state_root'];launch=read(base/'launch.json');state=read(base/'status.json')
owner=psutil.Process(state['pid']);assert owner.is_running() and owner.create_time()==state['create_time']==launch['create_time']
assert not state['failed_preserved']
jobs=[j for j in q['jobs'] if j['dataset']=='tep_classic']
assert len(jobs)==110
available=[j for j in jobs if (ROOT/j['output_directory']/'result.json').exists()]
assert len(available)==95 and {j['id'] for j in available}.issubset(state['verified_ids'])
missing=[dict(id=j['id'],model_config=j['model_config'],seed=j['seed'],source_output_exists=(ROOT/j['output_directory']).exists(),
    status='no_complete_source_result_preserved') for j in jobs if j not in available]
assert len(missing)==15
dataset=next(d for d in manifest['datasets'] if d['id']=='tep_classic')
assert len(dataset['entities'])==21 and sum(g['points'] for g in dataset['entities'])==20160
assert sum(g['anomaly_points'] for g in dataset['entities'])==16800
assert dataset['scaler']['fit_points']==500 and sum(g['points'] for g in dataset['validation_groups'])==960
assert sha(ROOT/dataset['input_npz'])==dataset['input_sha256']
reports,bindings,rows,entity_rows={}, {}, [], []
model_metadata={};source_pins={}
for job in available:
    identity=job['id'];p=Path(c['artifact_root'])/'conditions'/(identity+'.json')
    report=read(p);guard_path=base/'attempts'/identity/'receipt.json';guard=read(guard_path)
    assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['report_sha256']==sha(p)
    assert report['passed'] and report['canonical_id']==identity and report['full_test_points']==20160 and report['full_validation_points']==960
    assert report['positive_test_labels']==16800 and not report['source_checkpoint_neural_replay']
    source_path=ROOT/job['output_directory']/'result.json';source=read(source_path)
    assert sha(source_path)==report['source_result_sha256']
    assert sha(source_path.parent/'scores.npz')==report['scores_sha256'] and sha(source_path.parent/'model.pt')==report['checkpoint_sha256']
    assert report['prepared_input_sha256']==dataset['input_sha256'] and len(report['all_array_bindings'])==92
    for pin in job['frozen_files']:
        assert source_pins.setdefault(pin['path'],pin['sha256'])==pin['sha256']
    configuration=read(ROOT/job['model_config'])
    model_metadata[job['model_config']]=dict(id=configuration['id'],family=configuration.get('family'),
        entrypoint=configuration.get('entrypoint'),implementation=configuration.get('implementation'),
        reproduction_status=configuration['reproduction_status'],parameters=configuration['parameters'],
        citation=configuration.get('citation'),citations=configuration.get('citations'))
    reports[identity]=report
    bindings[identity]=dict(independent_report_path=str(p),independent_report_sha256=sha(p),guard_path=str(guard_path),guard_sha256=sha(guard_path),
        source_result_sha256=report['source_result_sha256'],checkpoint_sha256=report['checkpoint_sha256'],scores_sha256=report['scores_sha256'])
    for ratio,v in report['independent_strict_metrics'].items():
        row=dict(canonical_id=identity,model_config=job['model_config'],dataset='tep_classic',seed=job['seed'],
            validation_alarm_percent=ratio,threshold=v['threshold'],**v['micro'],macro_f1=v['macro_f1'],
            configured_epochs=source['epoch_audit']['declared_epochs'],observed_epochs=source['epoch_audit']['recorded_epochs'],
            parameter_count=source['parameter_count'],test_points=20160,positive_test_points=16800,
            actual_test_alarm_fraction=(v['micro']['true_positives']+v['micro']['false_positives'])/20160,
            native_normal_false_positive_fraction=v['micro']['false_positives']/3360)
        rows.append(row)
        for entity,m in v['per_entity'].items():
            entity_rows.append(dict(canonical_id=identity,model_config=job['model_config'],seed=job['seed'],validation_alarm_percent=ratio,entity=entity,**m))
for path,digest in source_pins.items():assert sha(ROOT/path)==digest,path
cohorts=defaultdict(list)
for r in rows:cohorts[r['model_config'],r['validation_alarm_percent']].append(r)
summary=[]
metrics=['precision','recall','f1','auroc','average_precision','macro_f1','actual_test_alarm_fraction','native_normal_false_positive_fraction']
for (model,ratio),records in sorted(cohorts.items()):
    assert sorted(r['seed'] for r in records)==[1103,1104,1105,1106,1107]
    output=dict(model_config=model,dataset='tep_classic',validation_alarm_percent=ratio,complete_seeds=5,required_seeds=5)
    for metric in metrics:
        values=[r[metric] for r in records];assert all(x is not None and math.isfinite(x) for x in values)
        mean=statistics.mean(values);std=statistics.stdev(values);se=std/math.sqrt(5);half=2.7764451051977987*se
        output.update({metric+'_mean':mean,metric+'_sample_std':std,metric+'_min':min(values),metric+'_max':max(values),
            metric+'_t95_low':mean-half,metric+'_t95_high':mean+half})
    summary.append(output)
assert len(rows)==285 and len(entity_rows)==5985 and len(summary)==57 and len(model_metadata)==19
primary=sorted([r for r in summary if r['validation_alarm_percent']=='1'],key=lambda r:r['f1_mean'],reverse=True)
assert len(primary)==19
for count,path in [(28,'tmp/industrial_full550_independent_metrics_validation_20261016_v1.xml'),
                   (60,'tmp/industrial_full550_shared_independent_metric_validation_20261016_v1.xml')]:
    suites=list(ET.parse(ROOT/path).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==count
    assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
out=ROOT/'projects/flow_matching_research/results/2026-10-16/industrial_TEP_all95_source_saved_score_independent_metrics_v1'
out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8',newline='\n')
for name,records in [('all95_original_conditions_285_strict_metric_rows.csv',rows),
        ('all95_original_conditions_5985_fault_entity_metric_rows.csv',entity_rows),
        ('all19_complete_five_seed_groups_57_fixed_budget_summaries.csv',summary)]:
    with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
write(out/'all95_full_array_metric_and_guard_bindings.json',dict(records=bindings,source_checkpoints_hashed_not_neurally_replayed=True,
    original_historical_resource_guards_not_certified=True,new_complete_models=0))
write(out/'all19_model_status_and_frozen_parameters.json',dict(configurations=model_metadata,all_source_pins=source_pins))
write(out/'all15_missing_original_TEP_conditions.json',dict(required_original_TEP_conditions=110,source_complete_and_independent_metrics_verified=95,remaining=missing))
write(out/'original_TEP_manifest.json',dataset)
for name,path in [('frozen_full550_independent_audit_registration',ROOT/audit.CONFIG),('actual_collector_launch',base/'launch.json')]:
    shutil.copyfile(path,out/(name+'.json'))
write(out/'current_full550_collector_snapshot.json',state)
shutil.copyfile(ROOT/'tmp/industrial_full550_independent_metrics_validation_20261016_v1.xml',out/'fresh28_industrial_audit_tests.xml')
shutil.copyfile(ROOT/'tmp/industrial_full550_shared_independent_metric_validation_20261016_v1.xml',out/'fresh60_shared_metric_and_replay_tests.xml')
write(out/'completion_boundary.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_global_scope=c['original_global_scope'],
    required_industrial_CPU_slots=550,required_industrial_GPU_slots=850,original_source_results_at_audit_launch=463,
    TEP_required_original_slots=110,TEP_verified_source_saved_score_cases=95,TEP_remaining_original_slots=15,
    TEP_complete_five_seed_groups=19,TEP_required_five_seed_groups=22,strict_validation_alarm_budgets_percent=[.1,1,5],
    all_copied_metadata_and_score_arrays_verified=95*92,full_test_points_per_case=20160,
    positive_test_points_per_case=16800,test_prevalence=16800/20160,
    always_alert_F1=2*16800/(20160+16800),always_alert_alarm_budget_percent=100,
    constant_score_AP=16800/20160,constant_score_AUROC=.5,
    constant_controls_are_prevalence_diagnostics_not_matched_alarm_budget_competitors=True,
    uncertainty='Only complete n=5 initialization groups; sample STD/min/max and untruncated Student-t df4 intervals. Not cross-dataset uncertainty or fault-independent confidence.',
    all_remaining_full550_conditions_still_required=True,source_checkpoint_neural_replay=False,
    historical_source_resource_guard_certification=False,source_original_paper_equivalence=False,full_TAB_equivalence=False,
    new_training_seed_slots=0,new_optimizer_updates=0,new_neural_forwards=0,new_complete_model_results=0,
    all_original_experiments_complete=False))
top=primary[0]
lines=['# 工业TEP：95项完整源结果的独立分数与指标核验','',
    '原CPU550项核验器已真实运行，持续跟随全部原条件。此包完成TEP现有95项完整源结果：19种配置×5种子；TEP原范围110项，另外15项没有完整源结果，仍保留。',
    '没有新增训练、神经推理或完整模型成绩。检查点只核对哈希，不能把本包说成95项检查点推理重放或历史资源守卫认证。','',
    '每项核对完整原始文件、冻结代码、配置、模型/分数哈希、训练组预算及可用损失历史；独立解析完整NPY分数与原标签/时间戳/行号，92个评分及元数据数组全部核对。',
    '每项500训练点、960验证点、21个完整故障运行共20160测试点。285组预先固定验证0.1%/1%/5%阈值，5985组逐故障指标独立复算通过。',
    '阈值、严格大于判定、混淆计数、Precision/Recall/F1、并列排名AUROC/AP、逐实体宏平均及实体内事件/延迟不调用源指标函数；不导入NumPy/Torch/sklearn。',
    '标签、时间戳、行号、故障字符串和数组形状/类型/内容逐一对应；统计仍依赖已经发布的独立原始数据预处理证据。','',
    '19个完整五种子组分别提供均值、样本STD、min/max和df=4的95% t区间，不截断区间、不计算部分种子均值；区间只描述这组初始化变异。',
    '各模型原容量、窗口、参数与20/40/60轮预算另列，不能据相同种子声称计算量完全匹配。严格基线使用PSM配置迁移，本包不认证原论文实验或完整TAB训练流程。','',
    'TEP测试异常比例83.3333%；一直报警的F1为0.909091，常数排序AP为0.833333、AUROC为0.5。一直报警消耗100%报警预算，仅作类别先验诊断，不能冒充验证1%预算下的公平竞争算法。',
    '因此，较高F1或AP仍须结合AUROC、实际正常点误报率、逐故障表现与验证报警规则解释；验证1%不是跨分布测试集误报率保证。','',
    '验证1%严格点级F1（全部为完整5种子描述性汇总，不据此重新选择超参数）：','',
    '| 配置 | F1均值 | 样本STD | AUROC均值 | AP均值 |','|---|---:|---:|---:|---:|']
for r in primary:
    lines.append(f"| {Path(r['model_config']).stem} | {r['f1_mean']:.6f} | {r['f1_sample_std']:.6f} | {r['auroc_mean']:.6f} | {r['average_precision_mean']:.6f} |")
lines += ['',
    '本次28项新核验测试和60项共享指标/检查点恢复相关测试通过。数值指标比较沿用独立核验公式rtol=atol=2e-6；坐标元数据和计数精确对应。',
    '未独立重跑bootstrap、VUS、Affiliation；不把部分方法缺失填成0。没有改写旧原始数据、失败、源模型或分数，也没有停止原数值任务。',
    '等待时及守卫日志中可能出现通用旧守卫标签replaying_full_checkpoints；实际子进程是本独立指标模块，报告明确没有神经重放，不凭状态标签扩大证据。',
    '新组件使用原数学锁、22GiB提交准入、16GiB紧急线及1GiB私有/RSS上限；原神经训练的32GiB等门槛保持。',
    '其余工业数据继续核验，原550CPU/850GPU和45/681/273/702研究范围保持，整体未完成。','']
(out/'README.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
shutil.copyfile(__file__,out/'publication_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/industrial_TEP_all95_independent_metrics.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(packet=out.relative_to(ROOT).as_posix(),TEP_verified_complete_source_cases=95,TEP_original_scope=110,
    complete_five_seed_groups=19,strict_metric_rows=285,new_complete_models=0,all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8')
text+='''
`工业TEP的全部95项已有源结果完成独立指标核验 <results/2026-10-16/industrial_TEP_all95_source_saved_score_independent_metrics_v1/README.md>`_
19配置×5种子、每项21故障/20160测试点；285组固定验证阈值、5985组逐故障指标及8740个评分/元数据数组核验通过。
TEP原110项的另外15项仍无完整源结果；完整工业CPU550项持续跟随，GPU850及全局研究范围保持。
只汇总完整5种子组，保留STD/min/max/t区间、原容量/轮次和方法复现状态；高异常先验控制与报警预算区别明确。
28项新验证和60项相关验证通过；检查点仅核对哈希，不计神经重放、历史资源认证、原论文或完整TAB等价，新增完整模型0。
'''
page.write_text(text,encoding='utf-8',newline='\n')
paths=[audit.MODULE.replace('.','/')+'.py',audit.TEST,audit.CONFIG,ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()]
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
write(ROOT/'tmp/industrial_TEP95_independent_metrics_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(out),publication_files=len(paths),TEP_verified=95,remaining_TEP=15,first_primary_summary=top,
    current_total_verified=len(state['verified_ids']),new_complete_models=0)))
