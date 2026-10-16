"""Publish every fixed orientation control without promoting a selected side."""
from collections import Counter,defaultdict
import csv
from datetime import datetime,timezone
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import diagnose_industrial_full466_score_direction_v1 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/s.CONFIG);q,m,j,e,b=s.verify(c);base=ROOT/c['state_root'];state=read(base/'status.json')
guard=read(base/'receipt.json');target=Path(c['artifact_root']);report=read(target/'report.json')
assert state['state']=='complete' and state['active_pid'] is None and state['passed']
assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['proof_sha256']==sha(target/'report.json')
assert report['passed'] and report['source_cases']==466 and report['orientation_controls']==2796
assert report['primary_scores_thresholds_and_direction_unchanged'] and not report['test_labels_used_for_direction_selection']
def load(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
rows=load(target/'all466_both_orientation_fixed_validation_controls.csv')
summary=load(target/'all89_complete_five_seed_paired_orientation_summaries.csv')
assert len(rows)==2796 and len(summary)==1869 and Counter(r['score_multiplier'] for r in rows)=={'1':1398,'-1':1398}
assert sum(r['actual_primary_evaluation']=='True' for r in rows)==466
assert all(r['score_multiplier']=='1' and r['validation_alarm_percent']=='1' for r in rows if r['actual_primary_evaluation']=='True')
assert all(r['opposite_orientation_primary']=='False' and r['direction_selected_using_test_labels']=='False' for r in rows)
bindings=read(target/'all466_source_and_direction_proof_bindings.json')['records'];assert set(bindings)==set(b)
for cid,item in bindings.items():
    assert sha(item['path'])==item['sha256'];proof=read(item['path'])
    assert proof['passed'] and proof['canonical_id']==cid and len(proof['controls'])==6
    assert proof['complete_source_binding']['source_result_sha256']==b[cid]['source_result_sha256']
    assert proof['complete_source_binding']['scores_sha256']==b[cid]['scores_sha256']
    assert proof['new_neural_forwards']==proof['new_optimizer_updates']==proof['new_training_seed_slots']==0
groups=read(target/'all110_original_group_completion.json')['groups'];assert len(groups)==110 and sum(r['complete'] for r in groups)==89
assert sum(0<len(r['completed_seeds'])<5 for r in groups)==6 and sum(not r['completed_seeds'] for r in groups)==15
index={(r['canonical_id'],r['score_multiplier'],r['validation_alarm_percent']):r for r in rows}
group_jobs=defaultdict(list)
for job in q['jobs']:group_jobs[job['model_config'],job['dataset']].append(job)
for row in summary:
    required=group_jobs[row['model_config'],row['dataset']];assert len(required)==5 and all(job['id'] in bindings for job in required)
    metric=row['metric'];ratio=row['validation_alarm_percent']
    a=[float(index[job['id'],'1',ratio][metric]) if index[job['id'],'1',ratio][metric] else None for job in required]
    z=[float(index[job['id'],'-1',ratio][metric]) if index[job['id'],'-1',ratio][metric] else None for job in required]
    expected=s.paired_summary(a,z)
    for key,value in expected.items():
        if value is None:assert row[key]==''
        elif isinstance(value,str):assert row[key]==value
        else:assert math.isclose(float(row[key]),value,rel_tol=1e-13,abs_tol=1e-13)
tests=ROOT/'tmp/industrial466_score_direction_shared_validation_20261016_v2.xml'
suites=list(ET.parse(tests).getroot().iter('testsuite'));assert sum(int(x.get('tests',0)) for x in suites)==63
assert all(int(x.get(k,0))==0 for x in suites for k in ['failures','errors','skipped'])
primary=[r for r in summary if r['validation_alarm_percent']=='1']
f1=[r for r in primary if r['metric']=='f1'];assert len(f1)==89
counts=Counter('unchanged' if math.isclose(float(r['source_mean']),float(r['opposite_mean']),rel_tol=1e-12,abs_tol=1e-12)
    else 'opposite_higher' if float(r['opposite_mean'])>float(r['source_mean']) else 'source_higher' for r in f1)
low_auc=[r for r in primary if r['metric']=='auroc' and float(r['source_mean'])<.5]
fm=[]
for ds in ['tep_classic','skab','pronto_day4','pronto_day2','pronto_day3']:
    selected={r['metric']:r for r in primary if r['model_config']=='configs/models/fm_objective_independent.json' and r['dataset']==ds}
    assert set(selected)==set(s.METRICS)
    row=dict(dataset=ds,model_config='configs/models/fm_objective_independent.json',complete_training_seeds=5,
        original_primary='source_high_tail_validation1percent_noPA',opposite_is_diagnostic=True)
    for metric in s.METRICS:
        row['source_'+metric]=float(selected[metric]['source_mean']);row['opposite_'+metric]=float(selected[metric]['opposite_mean'])
    fm.append(row)
out=ROOT/'projects/flow_matching_research/results/2026-10-16/industrial_all466_prespecified_score_direction_controls_v1'
out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8',newline='\n')
for path in target.iterdir():
    if path.is_file():shutil.copyfile(path,out/path.name)
with (out/'FM_all5_datasets_complete_five_seed_direction_diagnostics.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(fm[0]));w.writeheader();w.writerows(fm)
for name,path in dict(actual_guard=base/'receipt.json',actual_complete_state=base/'status.json',actual_launch=base/'launch.json',
    frozen_registration=ROOT/s.CONFIG,fresh63_validation=tests).items():shutil.copyfile(path,out/(name+path.suffix))
write(out/'descriptive_diagnostic_boundary.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_scope=c['original_scope'],
    original_CPU_scope=550,original_GPU_scope=850,complete_source_cases=466,all_fixed_orientation_controls=2796,
    original_model_dataset_groups=110,complete_five_seed_groups=89,partial_groups_without_statistics=6,empty_groups=15,
    paired_metric_summaries=1869,validation1percent_fullgroup_F1_comparison_counts=dict(counts),
    complete_groups_original_AUROC_mean_below_half=len(low_auc),
    both_orientations_predeclared=True,no_test_labels_for_direction_or_threshold_selection=True,
    primary_orientation_scores_thresholds_and_model_rankings_unchanged=True,opposite_orientation_is_not_new_primary_method=True,
    paired_CIs_are_descriptive_frozen_split_seed_variation=True,multiple_comparison_confirmatory_claim=False,
    sign_reversal_does_not_identify_physical_cause_of_residual_ranking=True,
    source_weights_not_neurally_replayed_by_this_diagnostic=True,new_training_results=0,new_neural_forwards=0,
    source32GiB_full_native_gate_unchanged=True,master_metric_catalog_unchanged='v70',all_original_experiments_complete=False))
handles=[]
for pid in [31444,42360,36108,41992,39444,29876,30276,32064,16360]:
    try:
        p=psutil.Process(pid);handles.append(dict(pid=pid,create_time=p.create_time(),command=p.cmdline(),user_seconds=p.cpu_times().user,
            numerical_descendants=[dict(pid=x.pid,create_time=x.create_time(),command=x.cmdline()) for x in p.children(recursive=True) if x.name().lower()=='python.exe']))
    except psutil.NoSuchProcess:handles.append(dict(pid=pid,handle_missing_at_capture=True))
write(out/'actual_other_training_and_review_handles_preserved.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=handles,
    numeric_processes_stopped_or_modified_by_this_diagnostic=False))
table='\n'.join('| '+r['dataset']+' | '+' | '.join(f'{r[k]:.6f}' for k in ['source_f1','opposite_f1','source_auroc','opposite_auroc','source_normal_false_positive_fraction','opposite_normal_false_positive_fraction'])+' |' for r in fm)
(out/'README.md').write_text('''# 全部466项冻结分数的预声明方向对照

完整执行原466项已训练条件的正向和反向分数对照，每种方向采用原0.1%、1%、5%验证尾部预算，共2796条完整条件记录、全部原实体记录及89个完整五种子组的1869条成对指标统计。110个原模型—数据集组中，6个部分组和15个空组不生成统计。原550CPU/850GPU及45篇/681方法记录/273数据任务/702论文指标范围保持。

两种方向在执行前固定：原分数s按验证高尾部严格大于阈值，反向分数-s也按其验证高尾部严格大于阈值，等价于原s严格小于验证低尾部阈值。两者均不用测试分数或标签校准，不进行PA。原方向验证1%是唯一主要结果，反向始终是诊断；即使测试成绩更高也不选择或替换原方向。原方向的三个阈值、全部原生点/实体指标和事件计数均再次与既有独立证明一致。

FM independent的五个完整种子均值（1%验证预算）：

| 数据集 | 原F1 | 反向诊断F1 | 原AUROC | 反向AUROC | 原测试正常点FPR | 反向正常点FPR |
|---|---:|---:|---:|---:|---:|---:|
'''+table+f'''

在89个完整五种子组中，原方向1% F1更高的组为{counts['source_higher']}，反向诊断更高的组为{counts['opposite_higher']}，相同为{counts['unchanged']}；这是描述性结果，不是测试集择优方法或经过多重比较校正的优越性结论。{len(low_auc)}组原AUROC均值低于0.5。反向分数的含并列AUROC与原AUROC之和为1；AP不是1减原AP，所有AP均从完整排名重新计算。AUROC/AP不依赖固定阈值，在三条敏感性设置中重复展示不构成三次独立实验。

本对照用于分离“原分数排序方向”和“验证分数尾部分布/报警预算”的影响。反向AUROC改善并不保证固定阈值F1改善；正常点误报率同时列出，避免把高异常占比下的高F1当作充分证据。SKAB的原验证段含原生异常，PRONTO原训练/验证段按标签选择Normal，三种PRONTO日轮换也相关；名义验证尾部预算不是测试正常点误报保证。它不能确定原残差排序为何反转，更不能据此归因某篇论文有误，或宣称简单选反向就复现SOTA。

完整结果见all466_both_orientation_fixed_validation_controls.csv；全部实体及事件见all466_both_orientation_full_entity_controls.csv；完整五种子成对差异、SD、SE及不截断df4t区间见all89_complete_five_seed_paired_orientation_summaries.csv。区间是固定划分上的种子波动，不是未见实体泛化界。原分数/权重/数据/阈值未写回；每个源完整训练结果、独立评分守卫、分数/标签载荷及检查点哈希均绑定。

相关63项测试通过，实际全部466项退出0、无资源中止。此诊断没有新神经网络前向、训练或模型完成信用，未替代尚待执行的完整检查点复核；该复核原32GiB门槛保持。主表v70保持原快照，不用测试结果改变主方法排名。其他正在执行的完整训练进程未停止或修改。整体研究任务尚未完成。
''',encoding='utf-8',newline='\n')
shutil.copyfile(__file__,out/'publication_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/industrial466_prespecified_score_direction_controls.2026-10-16_v1.json'
write(ara,dict(packet=out.relative_to(ROOT).as_posix(),required_complete_sources=466,complete_orientation_controls=2796,
    complete_five_seed_groups=89,paired_metric_summaries=1869,opposite_orientation_primary=False,
    original_primary_unchanged=True,original_scope=c['original_scope'],all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8');assert out.name not in text
text+='\n`工业全部466项完成预声明正反分数方向对照 <results/2026-10-16/'+out.name+'/README.md>`_\n'
text+=f'2796组固定验证校准、无PA控制及89个完整五种子组的1869项成对统计通过；6部分组/15空组不计算统计。验证1% F1原方向更高{counts["source_higher"]}组、反向诊断更高{counts["opposite_higher"]}组、相同{counts["unchanged"]}组；全部两方向均公开，不按测试标签选择或改变主方向。正常点FPR和AUROC/AP解释单列，63项相关验证通过，无新增训练/神经前向。主表v70及原32GiB完整重放门槛、全部研究范围保持，整体未完成。\n'
page.write_text(text,encoding='utf-8',newline='\n')
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
paths=[s.CONFIG,s.MODULE.replace('.','/')+'.py',s.TEST,ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()]
paths.extend(p.relative_to(ROOT).as_posix() for p in out.iterdir() if p.is_file())
write(ROOT/'tmp/industrial466_score_direction_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(dict(packet=str(out),files=len(paths),F1_comparison_counts=dict(counts),original_AUROC_below_half_groups=len(low_auc),FM=fm))
