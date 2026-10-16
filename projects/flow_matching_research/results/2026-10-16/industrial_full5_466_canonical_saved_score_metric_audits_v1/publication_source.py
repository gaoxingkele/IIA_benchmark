"""Close all463 direct source scores plus three exact recovered original slots."""
from collections import Counter,defaultdict
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
from scripts.flow_matching import audit_industrial_full550_independent_metrics_v1 as direct
from scripts.flow_matching import audit_industrial_three_recovery_independent_metrics_v1 as recovered
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/direct.CONFIG);q,manifest=direct.verify(c)
rc=read(ROOT/recovered.CONFIG);_,_,_,recovery_cases=recovered.verify(rc)
base=ROOT/c['state_root'];state=read(base/'status.json');owner=psutil.Process(state['pid'])
assert owner.is_running() and owner.create_time()==state['create_time'] and not state['failed_preserved']
direct_jobs=[j for j in q['jobs'] if (ROOT/j['output_directory']/'result.json').exists()]
assert len(direct_jobs)==463 and {j['id'] for j in direct_jobs}.issubset(state['verified_ids'])
rp=Path(rc['artifact_root'])/'report.json';rr=read(rp);rgp=ROOT/rc['state_root']/'receipt.json';rg=read(rgp)
assert rr['passed'] and len(rr['records'])==3
assert rg['passed'] and rg['exit_code']==0 and not rg['resource_guard_aborted'] and rg['proof_sha256']==sha(rp)
jobs={j['id']:j for j in q['jobs']};executions={j['id']:j for j in direct_jobs};reports={};bindings={}
for j in direct_jobs:
    identity=j['id'];path=Path(c['artifact_root'])/'conditions'/(identity+'.json');guard_path=base/'attempts'/identity/'receipt.json'
    report,guard=read(path),read(guard_path)
    assert report['passed'] and report['canonical_id']==identity
    assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['report_sha256']==sha(path)
    reports[identity]=report
    bindings[identity]=dict(actual_execution_id=identity,independent_report_path=str(path),independent_report_sha256=sha(path),
        guard_path=str(guard_path),guard_sha256=sha(guard_path),separate_recovery=False)
for case in recovery_cases:
    identity=case['original_id'];execution=recovered.validate_identity(case,jobs)
    assert identity not in reports and identity in rr['records']
    entry=rr['records'][identity];report=entry['independent_report']
    assert entry['actual_execution_id']==execution['id']==report['canonical_id'] and report['passed']
    reports[identity]=report;executions[identity]=execution
    bindings[identity]=dict(actual_execution_id=execution['id'],independent_report_container=str(rp),independent_report_container_sha256=sha(rp),
        report_json_pointer='/records/'+identity+'/independent_report',guard_path=str(rgp),guard_sha256=sha(rgp),separate_recovery=True)
assert len(reports)==len(executions)==466
assert Counter(jobs[cid]['dataset'] for cid in reports)==dict(tep_classic=95,skab=95,pronto_day4=89,pronto_day2=95,pronto_day3=92)
prior=ROOT/'projects/flow_matching_research/results/2026-10-16/industrial_TEP_all95_source_saved_score_independent_metrics_v1'
previous=read(prior/'all95_full_array_metric_and_guard_bindings.json')['records']
for cid,b in previous.items():
    assert bindings[cid]['independent_report_path']==b['independent_report_path'] and bindings[cid]['independent_report_sha256']==b['independent_report_sha256']
    assert bindings[cid]['guard_path']==b['guard_path'] and bindings[cid]['guard_sha256']==b['guard_sha256']
rows,entity_rows=[],[];models={};pins={};dataset_info={}
for spec in manifest['datasets']:
    assert sha(ROOT/spec['input_npz'])==spec['input_sha256']
    dataset_info[spec['id']]=dict(input_sha256=spec['input_sha256'],fit_points=spec['scaler']['fit_points'],
        validation_points=sum(g['points'] for g in spec['validation_groups']),test_points=sum(g['test_points'] for g in spec['entities']),
        positive_test_points=sum(g['anomaly_points'] for g in spec['entities']),test_entities=len(spec['entities']),
        selection_audit=spec['selection_audit'])
for cid in sorted(reports):
    report=reports[cid];j=jobs[cid];execution=executions[cid];source_path=ROOT/execution['output_directory']/'result.json';source=read(source_path)
    assert sha(source_path)==report['source_result_sha256']
    assert sha(source_path.parent/'model.pt')==report['checkpoint_sha256'] and sha(source_path.parent/'scores.npz')==report['scores_sha256']
    assert not report['source_checkpoint_neural_replay'] and not report['source_historical_resource_guard_certified']
    info=dataset_info[j['dataset']]
    assert report['full_test_points']==info['test_points'] and report['positive_test_labels']==info['positive_test_points']
    bindings[cid].update(source_result_path=source_path.relative_to(ROOT).as_posix(),source_result_sha256=report['source_result_sha256'],
        checkpoint_sha256=report['checkpoint_sha256'],scores_sha256=report['scores_sha256'])
    for pin in j['frozen_files']:assert pins.setdefault(pin['path'],pin['sha256'])==pin['sha256']
    model=read(ROOT/j['model_config']);models[j['model_config']]=dict(id=model['id'],family=model.get('family'),entrypoint=model.get('entrypoint'),
        implementation=model.get('implementation'),reproduction_status=model['reproduction_status'],parameters=model['parameters'],
        citation=model.get('citation'),citations=model.get('citations'))
    for ratio,v in report['independent_strict_metrics'].items():
        micro=v['micro'];points=info['test_points'];positive=info['positive_test_points'];normal=points-positive
        rows.append(dict(canonical_id=cid,actual_execution_id=execution['id'],separate_recovery=cid!=execution['id'],
            model_config=j['model_config'],dataset=j['dataset'],seed=j['seed'],validation_alarm_percent=ratio,threshold=v['threshold'],
            **micro,macro_f1=v['macro_f1'],configured_epochs=source['epoch_audit']['declared_epochs'],observed_epochs=source['epoch_audit']['recorded_epochs'],
            parameter_count=source['parameter_count'],test_points=points,positive_test_points=positive,
            actual_test_alarm_fraction=(micro['true_positives']+micro['false_positives'])/points,
            native_normal_false_positive_fraction=micro['false_positives']/normal if normal else None))
        for entity,m in v['per_entity'].items():
            entity_rows.append(dict(canonical_id=cid,actual_execution_id=execution['id'],model_config=j['model_config'],dataset=j['dataset'],seed=j['seed'],
                validation_alarm_percent=ratio,entity=entity,**m))
for path,digest in pins.items():assert sha(ROOT/path)==digest,path
all_groups=defaultdict(list)
for j in q['jobs']:all_groups[j['model_config'],j['dataset']].append(j)
group_status=[];summary=[];metric_names=['precision','recall','f1','auroc','average_precision','macro_f1','actual_test_alarm_fraction','native_normal_false_positive_fraction']
row_index={(r['canonical_id'],r['validation_alarm_percent']):r for r in rows}
for (model,dataset),group in sorted(all_groups.items()):
    done=[j for j in group if j['id'] in reports]
    assert sorted(j['seed'] for j in group)==[1103,1104,1105,1106,1107]
    group_status.append(dict(model_config=model,dataset=dataset,completed_seeds=sorted(j['seed'] for j in done),
        required_seeds=[1103,1104,1105,1106,1107],complete=len(done)==5,partial_statistics_computed=False))
    if len(done)!=5:continue
    for ratio in ['0.1','1','5']:
        records=[row_index[j['id'],ratio] for j in done]
        item=dict(model_config=model,dataset=dataset,validation_alarm_percent=ratio,complete_seeds=5,required_seeds=5)
        for metric in metric_names:
            values=[r[metric] for r in records];assert all(v is not None and math.isfinite(v) for v in values)
            mean=statistics.mean(values);sd=statistics.stdev(values);half=2.7764451051977987*sd/math.sqrt(5)
            item.update({metric+'_mean':mean,metric+'_sample_std':sd,metric+'_min':min(values),metric+'_max':max(values),metric+'_t95_low':mean-half,metric+'_t95_high':mean+half})
        summary.append(item)
assert len(group_status)==110 and sum(g['complete'] for g in group_status)==89
assert sum(0<len(g['completed_seeds'])<5 for g in group_status)==6 and sum(not g['completed_seeds'] for g in group_status)==15
assert len(rows)==1398 and len(summary)==267 and sum(len(r['all_array_bindings']) for r in reports.values())==31715
missing=[dict(canonical_id=j['id'],model_config=j['model_config'],dataset=j['dataset'],seed=j['seed'],
    category='raw_entropic_original_without_complete_result' if j['model_config'] in ['configs/models/fm_objective_sb.json','configs/models/fm_objective_sf2m.json','configs/models/fm_objective_sf2m_flow_only.json'] else 'stable_entropic_full_original_case_still_required')
    for j in q['jobs'] if j['id'] not in reports]
assert len(missing)==84 and Counter(r['category'] for r in missing)==dict(raw_entropic_original_without_complete_result=75,stable_entropic_full_original_case_still_required=9)
for count,path in [(38,'tmp/industrial_three_recovery_and_full550_validation_20261016_v1.xml'),(60,'tmp/industrial_full550_shared_independent_metric_validation_20261016_v1.xml')]:
    suites=list(ET.parse(ROOT/path).getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==count
    assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
out=ROOT/'projects/flow_matching_research/results/2026-10-16/industrial_full5_466_canonical_saved_score_metric_audits_v1';out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8',newline='\n')
for name,records in [('all466_canonical_cases_1398_strict_metric_rows.csv',rows),('all466_cases_full_fault_entity_metric_rows.csv',entity_rows),
    ('all89_complete_five_seed_groups_267_fixed_budget_summaries.csv',summary)]:
    with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
write(out/'all466_proof_source_and_guard_bindings.json',dict(records=bindings,prior_TEP95_report_and_guard_bindings_preserved=True,
    source_checkpoints_hashed_not_neurally_replayed=True,source_historical_current_resource_gate_certification=False))
write(out/'all110_original_model_dataset_seed_group_completion.json',dict(groups=group_status,complete_groups=89,partial_groups=6,empty_groups=15,
    only_complete_n5_groups_aggregated=True,required_groups=110))
write(out/'all84_remaining_original_CPU_slots.json',dict(original_CPU_scope=550,direct_complete=463,exact_completed_separate_recoveries=3,remaining=missing))
write(out/'all19_available_model_configs_and_frozen_parameters.json',dict(configurations=models,all_frozen_source_pins=pins))
write(out/'all5_dataset_native_scope_and_selection.json',dict(datasets=dataset_info))
for name,path in dict(three_recovery_independent_report=rp,three_recovery_audit_guard=rgp,
    three_recovery_registration=ROOT/recovered.CONFIG,three_recovery_launch=ROOT/rc['state_root']/'launch.json',
    three_recovery_terminal_state=ROOT/rc['state_root']/'status.json',full550_registration=ROOT/direct.CONFIG,
    fresh38_recovery_and_direct_audit_validation=ROOT/'tmp/industrial_three_recovery_and_full550_validation_20261016_v1.xml',
    fresh60_shared_metric_replay_validation=ROOT/'tmp/industrial_full550_shared_independent_metric_validation_20261016_v1.xml').items():
    shutil.copyfile(path,out/(name+path.suffix))
write(out/'actual_full550_collector_waiting_state.json',state)
write(out/'completion_boundary.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_global_scope=c['original_global_scope'],
    required_CPU_scope=550,required_GPU_scope=850,direct_source_saved_score_cases=463,distinct_exact_recovered_original_slots=3,
    canonical_independent_metric_cases=466,remaining_CPU_slots=84,complete_seed_groups=89,required_seed_groups=110,
    partial_seed_groups=6,empty_seed_groups=15,strict_validation_metric_rows=1398,all_score_metadata_arrays_verified=31715,
    full_fault_entity_metric_rows=len(entity_rows),complete_seed_group_budget_summary_rows=267,
    no_original_data_failed_artifact_checkpoint_or_score_overwritten=True,
    source_checkpoint_neural_replay=False,source_historical_current_resource_gate_certification=False,
    independent_bootstrap=False,independent_VUS=False,independent_affiliation=False,
    uncertainty='n5 initialization STD/min/max/untruncated df4 Student t intervals only; not independent fault/day/dataset confidence',
    source_recovery_resource_observations_preserved_without_retroactive_current_32GiB_gate_claim=True,
    new_training_seed_slots=0,new_optimizer_updates=0,new_neural_forwards=0,new_complete_model_results=0,
    full_TAB_equivalence=False,paper_equivalence=False,all_original_experiments_complete=False))
lines=['# 五个工业划分：466个原始条件的完整分数与独立指标核验','',
    '463项原目录完整结果全部通过；另3项SKAB资源恢复结果仅id和输出目录不同，数据、种子、模型、全部参数/窗口和原评估设置逐字段一致，经单独受保护核验后映射回原条件，各计一次。',
    '合计466/550个CPU原条件有完整分数与独立指标证据；剩余84项为75个原始熵运输失败记录和9个稳定熵运输完整条件，原失败保留。GPU850和完整全局范围未缩减。',
    '这不是新增466次训练或检查点神经重放；检查点只核对文件哈希，历史源执行不追认当前32GiB等资源门槛。3个恢复结果的原始resource_usage完整保留。','',
    '| 数据划分 | 已核验原条件 | 原需条件 | 原测试点数/项 | 异常点数/项 |','|---|---:|---:|---:|---:|']
for dataset,info in dataset_info.items():
    count=sum(jobs[cid]['dataset']==dataset for cid in reports)
    lines.append(f"| {dataset} | {count} | 110 | {info['test_points']} | {info['positive_test_points']} |")
lines+=['','全部31715个评分和元数据数组、完整标签/时间戳/行号/故障字符串均核对；1398组固定验证0.1%/1%/5%无PA阈值、微观/逐实体/事件指标通过独立公式复算。',
    '保持完整337数组独立原始预处理证据、冻结代码及模型/分数/原始文件哈希；独立复算不调用源指标函数，不导入NumPy/Torch/sklearn。未独立重跑bootstrap、VUS或Affiliation。','',
    '110个原模型—数据集五种子组中89组完整、6组部分、15组无完整结果；只对89完整组提供267个固定阈值预算下的均值、样本STD、min/max和df4的95% t区间，部分组不统计、不补零。',
    '种子区间只描述初始化变异。PRONTO三个轮换共用原日期，不能当三个独立数据源；正常训练/验证段使用原标签选择，属于标签辅助协议。SKAB验证有异常，验证1%不是正常总体或跨分布测试误报率保证。',
    '各方法容量、原始20/40/60轮预算和复现状态分别保存；PSM配置迁移和本地FM评分适配不等于原论文复现，不能声称相同种子就是计算量完全匹配。','',
    '验证1%下的完整五种子结果（只做已冻结设置的描述性比较，不据测试结果选超参数）：','',
    '| 划分 | 配置 | F1均值 | 样本STD | AUROC均值 | AP均值 |','|---|---|---:|---:|---:|---:|']
for dataset in dataset_info:
    primary=sorted([r for r in summary if r['dataset']==dataset and r['validation_alarm_percent']=='1'],key=lambda r:r['f1_mean'],reverse=True)
    for r in primary:
        lines.append(f"| {dataset} | {Path(r['model_config']).stem} | {r['f1_mean']:.6f} | {r['f1_sample_std']:.6f} | {r['auroc_mean']:.6f} | {r['average_precision_mean']:.6f} |")
lines+=['','TEP95的旧报告/守卫哈希和指标保持；源目录缺口87中3项有完整独立恢复，现在仅计原条件一次，因此为466个不同原条件、84个仍待完整执行。',
    '38项恢复身份/直接核验相关测试及60项共享指标/恢复相关测试通过。小型夹具只验证流程，不算性能；现有数值任务没有停止，所有原数据与失败均保留。',
    '本核验没有新增训练、神经推理或训练种子，不认证历史作者、完整TAB或全部论文效果一致；整体目标未完成。','']
(out/'README.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
shutil.copyfile(__file__,out/'publication_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/industrial_full5_466_canonical_metric_audits.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(packet=out.relative_to(ROOT).as_posix(),canonical_CPU_metric_cases=466,required_CPU_slots=550,
    complete_five_seed_groups=89,required_groups=110,new_complete_models=0,all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8')
text+='''
`五个工业划分的466个原条件通过完整分数独立核验 <results/2026-10-16/industrial_full5_466_canonical_saved_score_metric_audits_v1/README.md>`_
全部463项原目录结果和3项仅身份/输出不同的SKAB完整资源恢复，映射为466个不同原条件；原CPU550尚余84、GPU850保持。
31715个评分/元数据数组、1398组固定验证预算指标核验；89/110完整五种子组提供267组均值/STD/min/max/t区间，6部分组和15空组不补统计。
TEP95、SKAB95、PRONTO day4/day2/day3各89/95/92项；保留全部源模型/分数/原数据/失败和方法复现状态，38项及60项相关测试通过。
本包不含新训练或神经重放，不追认历史执行当前资源门槛；原论文、完整TAB和全部实验一致性仍未完成。
'''
page.write_text(text,encoding='utf-8',newline='\n')
paths=[recovered.MODULE.replace('.','/')+'.py',recovered.TEST,recovered.CONFIG,ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()]
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
write(ROOT/'tmp/industrial_full5_466_canonical_metric_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(out),files=len(paths),canonical_CPU_metric_cases=466,full_groups=89,remaining_CPU_slots=84,
    per_dataset=dict(Counter(jobs[cid]['dataset'] for cid in reports)),entity_metric_rows=len(entity_rows),new_complete_models=0)))
