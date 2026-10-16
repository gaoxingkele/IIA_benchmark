"""Close the full existing466 native range evaluations without training credit."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import audit_industrial_full466_native_ranges_v2 as s
from scripts.flow_matching import audit_industrial_full466_native_ranges_v1 as failed
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

c=read(ROOT/s.CONFIG);q,manifest,jobs,executions,b=s.verify(c)
assert read(ROOT/failed.CONFIG)==failed.registration()
base=ROOT/c['state_root'];state=read(base/'status.json');guard_path=base/'receipt.json';guard=read(guard_path)
target=Path(c['artifact_root']);report_path=target/'report.json';report=read(report_path)
assert state['state']=='complete' and state['active_pid'] is None and state['passed']
assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['proof_sha256']==sha(report_path)
assert report['passed'] and report['cases']==466 and report['complete_groups']==89
assert not report['independent_VUS_reexecution'] and not report['independent_affiliation_integrals']
assert report['new_complete_model_results']==report['new_neural_forwards']==report['new_optimizer_updates']==0
oldbase=ROOT/read(ROOT/failed.CONFIG)['state_root'];oldstate=read(oldbase/'status.json');oldguard=read(oldbase/'receipt.json')
assert oldstate['state']=='terminal_failure_preserved' and oldstate['active_pid'] is None
assert not oldguard['passed'] and oldguard['exit_code']==1 and not oldguard['resource_guard_aborted']
assert 'Previously independently checked calibration changed' in (oldbase/'worker.stderr.log').read_text(encoding='utf-8')
def csvrows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
rows=csvrows(target/'all466_native_range_controls.csv')
summary=csvrows(target/'all89_complete_five_seed_native_range_summaries.csv')
assert len(rows)==6990 and sum(r['value']=='' for r in rows)==164 and len(summary)==1335
assert sum(r['mean']=='' for r in summary)==42
actual=read(target/'all466_native_range_proof_bindings.json')['records']
assert set(actual)==set(b) and len(actual)==466 and sum(r['reused_original_native_calculation'] for r in actual.values())<=466
for cid,entry in actual.items():
    assert entry['actual_execution_id']==b[cid]['actual_execution_id']
    assert entry['independent_point_proof_binding']==b[cid]
    assert sha(ROOT/entry['native_evaluation_path'])==entry['native_evaluation_sha256']
group_rows=read(target/'all110_original_group_completion.json')['groups']
assert len(group_rows)==110 and sum(g['complete'] for g in group_rows)==89
index=defaultdict(list)
for r in rows:index[r['model_config'],r['dataset'],r['validation_alarm_percent'],r['metric']].append(r)
for r in summary:
    values=index[r['model_config'],r['dataset'],r['validation_alarm_percent'],r['metric']]
    assert sorted(int(v['seed']) for v in values)==[1103,1104,1105,1106,1107]
    defined=[float(v['value']) for v in values if v['value']!='']
    if len(defined)<5:
        assert all(r[k]=='' for k in ['mean','sample_std','t95_low','t95_high'])
        assert int(r['defined_seed_count'])==len(defined)
    else:
        m=statistics.mean(defined);sd=statistics.stdev(defined);half=2.7764451051977987*sd/math.sqrt(5)
        for key,value in dict(mean=m,sample_std=sd,t95_low=m-half,t95_high=m+half).items():
            assert math.isclose(float(r[key]),value,rel_tol=1e-13,abs_tol=1e-13)
test=ROOT/'tmp/industrial466_full_native_ranges_validation_20261016_v2.xml'
suites=list(ET.parse(test).getroot().iter('testsuite'))
assert sum(int(v.get('tests',0)) for v in suites)==107
assert all(int(v.get(k,0))==0 for v in suites for k in ['failures','errors','skipped'])
prior=ROOT/s.PRIOR
point_groups=csvrows(prior/'all89_complete_five_seed_groups_267_fixed_budget_summaries.csv')
point_index={(r['model_config'],r['dataset'],r['validation_alarm_percent']):r for r in point_groups}
native_index={(r['model_config'],r['dataset'],r['validation_alarm_percent'],r['metric']):r for r in summary}
comparison=[]
for dataset in ['tep_classic','skab','pronto_day4','pronto_day2','pronto_day3']:
    model='configs/models/fm_objective_independent.json';p=point_index[model,dataset,'1']
    item=dict(model_config=model,dataset=dataset,training_seeds='1103,1104,1105,1106,1107',complete_models=5,
        primary_protocol='validation1percent_noPA',point_F1_mean=float(p['f1_mean']),point_AUROC_mean=float(p['auroc_mean']),
        point_AP_mean=float(p['average_precision_mean']))
    for metric in ['VUS_ROC','VUS_PR',*s.AFFILIATION]:
        item[metric+'_mean']=float(native_index[model,dataset,'1',metric]['mean'])
    comparison.append(item)
roundoff=[]
for cid,binding in b.items():
    if binding['separate_recovery']:
        proof=read(binding['independent_report_container'])['records'][cid]['independent_report']
    else:proof=read(binding['independent_report_path'])
    source=read(ROOT/binding['source_result_path'])
    for ratio,item in proof['independent_strict_metrics'].items():
        a=item['threshold'];v=source['metrics']['strict'][ratio]['calibration']['threshold']
        if a!=v:
            assert math.isclose(a,v,rel_tol=c['independent_threshold_tolerance']['rtol'],abs_tol=c['independent_threshold_tolerance']['atol'])
            roundoff.append(dict(canonical_id=cid,validation_alarm_percent=ratio,independent_threshold=a,
                original_unchanged_native_threshold=v,absolute_difference=abs(a-v)))
assert len(roundoff)==3 and max(r['absolute_difference'] for r in roundoff)==2.842170943040401e-14
out=ROOT/'projects/flow_matching_research/results/2026-10-16/industrial_all466_full_native_range_metrics_v2'
out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n*.log -text\n',encoding='utf-8',newline='\n')
for path in target.iterdir():
    assert path.is_file();shutil.copyfile(path,out/path.name)
for name,data in [('FM_all5_complete_five_seed_point_and_range_comparison.csv',comparison),
                  ('all3_independent_percentile_roundoff_differences.csv',roundoff)]:
    with (out/name).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
copy={
    'actual_guard.json':guard_path,'actual_complete_state.json':base/'status.json','actual_launch.json':base/'launch.json',
    'frozen_v2_registration.json':ROOT/s.CONFIG,'fresh107_validation.xml':test,
    'preserved_v1_registration.json':ROOT/failed.CONFIG,'preserved_v1_failed_guard.json':oldbase/'receipt.json',
    'preserved_v1_failed_state.json':oldbase/'status.json','preserved_v1_worker.stderr.log':oldbase/'worker.stderr.log',
    'preserved_v1_launch.json':oldbase/'launch.json',
    'preserved_first72_test_validation.xml':ROOT/'tmp/industrial466_full_native_ranges_validation_20261016_v1.xml',
    'all84_remaining_original_CPU_slots.json':prior/'all84_remaining_original_CPU_slots.json'}
for name,path in copy.items():shutil.copyfile(path,out/name)
live=[]
for pid in [16360,30032,31444,28460,39444,30800,18292,27336,13636,36108,41992]:
    try:
        p=psutil.Process(pid);live.append(dict(pid=pid,create_time=p.create_time(),command=p.cmdline(),
            cpu_user_seconds=p.cpu_times().user,numerical_descendants=[dict(pid=x.pid,create_time=x.create_time(),command=x.cmdline()) for x in p.children(recursive=True) if x.name().lower()=='python.exe']))
    except psutil.NoSuchProcess:live.append(dict(pid=pid,handle_missing_at_capture=True))
write(out/'actual_other_training_and_review_handles_observed.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=live,
    process_modifications_or_terminations_by_this_audit=False))
write(out/'completion_boundary.json',dict(**report,current_audit_guard_sha256=sha(guard_path),new_metric_integral_executions=0,
    master_metric_catalog_unchanged='v68',source_models_not_neurally_replayed_by_this_audit=True,
    independent_point_metrics_reused_from_verified_packet=True,range_source_metrics_already_executed_before_this_audit=True,
    five_seed_summary_defined=1293,five_seed_summary_metric_undefined=42,native_controls_with_undefined_macro=164,
    prior_failed_check_retained=True,only_correction='Independent percentile reconstruction comparison uses its existing frozen2e-6 tolerance; native thresholds and all source metrics remain unchanged'))
table='\n'.join('| '+r['dataset']+' | '+' | '.join(f"{r[k]:.6f}" for k in ['point_F1_mean','point_AUROC_mean','VUS_ROC_mean','VUS_PR_mean','affiliation_f_mean'])+' |' for r in comparison)
readme='''# 工业数据全部466项完整 TAB 区间指标核验

已核验463项直接结果和3项原任务恢复结果，共6990条固定验证阈值区间指标记录、16218条实体控制记录。110个原模型—数据集组中，89组具有全部五个训练种子，形成1335条统计记录；其中42条因部分种子的指标未定义而保持空白，不以较小种子子集计算均值。6个部分组和15个空组均不生成统计。

五种子均值示例：FM independent，验证集1%阈值、无PA，所有实体和时间点均保留。VUS不使用该校准阈值，而按其完整分数扫描；Affiliation使用固定阈值。

| 数据集 | 点级F1 | 点级AUROC | VUS-ROC | VUS-PR | Affiliation-F |
|---|---:|---:|---:|---:|---:|
'''+table+'''

区间指标与点级指标的定义不同，不能直接比较数值高低。完全沿用冻结TAB定义：250个阈值、从0至两倍截断中位异常段长度的全部整数缓冲区。TEP每条实体序列960点，最大缓冲区1600；PRONTO day2为11698/12420点，day3为7080/21300点。PRONTO day3同一组分数的点级AUROC约0.378而VUS-ROC约0.904，是本地协议差异的实际例子，不能据此归因某篇原论文有错误。

完整清单见 `all466_native_range_controls.csv`；全部实体几何和指标见 `all466_full_entity_geometry_and_native_metrics.csv`；五种子均值、标准差和不截断Student-t区间见 `all89_complete_five_seed_native_range_summaries.csv`。区间只描述冻结数据划分上的五模型种子波动，不代表未见实体的泛化不确定性。VUS在三种阈值行中是同一阈值无关指标的重复展示，不算三次独立执行。验证集0.1%和5%为敏感性控制，唯一主要协议为验证集1%、无PA。另核验全部原十档测试池校准诊断的阈值和宏平均，它们不进入主要结果。

此次核验检查完整原始标签、数据和评分文件哈希、任务身份、原始评估源代码、所有实体、缓冲区、阈值、未定义数量及宏平均，并验证复用记录的每项原始指标和耗时保持不变。它没有独立重算VUS或Affiliation积分，没有重新训练或神经网络前向，也不认证历史资源门槛、原作者完整协议或完整TAB训练框架等价。原CPU550项仍缺84项，GPU850项及45篇/681方法记录/273原数据任务/702论文指标的完整范围不变，总任务未完成。

首版核验将独立分位数与原分位数强制做浮点相等比较而失败；3处差异最大2.8422e-14。失败配置、日志和资源回执均保留。第二版仅改用原独立指标审计已经冻结的rtol=atol=2e-6进行这项交叉比较；实际原校准阈值、预测和原指标均未改变。107项相关测试通过，实际466项核验退出0且无资源中止。其他正在运行的训练进程未修改或停止。主指标总表v68保持原快照，本包单独列出完整工业轨道。
'''
(out/'README.md').write_text(readme,encoding='utf-8',newline='\n')
shutil.copyfile(Path(__file__),out/'publication_source.py')
ara=ROOT/'projects/flow_matching_research/ara/execution/industrial466_full_native_range_metrics.2026-10-16_v2.json'
write(ara,dict(packet=out.relative_to(ROOT).as_posix(),frozen_configuration=s.CONFIG,**report,
    evidence_layer='Existing full native metric execution verification and five-seed aggregation;no new model or integral execution'))
main_readme=ROOT/'projects/flow_matching_research/README.md'
marker='industrial_all466_full_native_range_metrics_v2'
text=main_readme.read_text(encoding='utf-8');assert marker not in text
text+='\n\n工业轨道新增[全部466项完整TAB区间指标核验](results/2026-10-16/'+marker+'/README.md)：6990条区间控制、89个完整五种子组。完整原缓冲区及阈值、未定义指标数量和点级/区间指标差异均已列出；这是既有完整指标的核验，不新增模型训练完成数量。原CPU550项仍缺84项，整体任务未完成。\n'
main_readme.write_text(text,encoding='utf-8',newline='\n')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]))
previous_root=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
previous_parent=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent/'flow-matching-research',text=True).strip()
paths=[p.relative_to(ROOT).as_posix() for p in sorted(out.iterdir()) if p.is_file()]
paths += [p for p in [s.CONFIG,failed.CONFIG,s.MODULE.replace('.','/')+'.py',failed.MODULE.replace('.','/')+'.py',s.TEST,failed.TEST]]
paths += [ara.relative_to(ROOT).as_posix(),'projects/flow_matching_research/README.md']
write(ROOT/'tmp/industrial466_full_native_range_publication_manifest_20261016_v2.json',dict(previous_root=previous_root,previous_parent=previous_parent,paths=paths))
print(json.dumps(dict(packet=str(out),files=len(paths),cases=466,complete_groups=89,metric_rows=6990,
    actual_new_model_results=0,previous_root=previous_root,previous_parent=previous_parent)))
