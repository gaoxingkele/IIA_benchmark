"""Publish all qualified real transport components beside a complete GRASP model."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import diagnose_industrial_remaining8_entropic_couplings_v1 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/s.CONFIG);s.verify(c)
base=ROOT/c['state_root'];state=read(base/'status.json')
assert state['state']=='all_group_attempts_terminal' and state['active_pid'] is None and not state['failed_preserved']
assert set(state['qualified_groups'])==set(c['groups'])
groups={};bindings={};rows=[];case_map=[]
for key,group in c['groups'].items():
    if group['reuse_existing']:
        path=Path(c['prior_qualified_report']);guard_path=ROOT/c['prior_guard'];report=read(path);problems=report['coupling_problems']
        assert report['fit_shape']==[8,100,17] and report['original_case']['dataset']==group['dataset'] and report['original_case']['seed']==group['seed']
        prior_dataset=next(d for d in read(ROOT/group['representative_job']['dataset_manifest_path'])['datasets'] if d['id']==group['dataset'])
        assert report['input_sha256']==sha(ROOT/prior_dataset['input_npz'])
        reused=True
    else:
        path=Path(c['artifact_root'])/key/'report.json';guard_path=base/'attempts'/key/'receipt.json';report=read(path);problems=report['coupling_problems']
        assert report['passed'] and report['group']==group and report['fit_shape']==[group['fit_windows'],100,17]
        reused=False
    guard=read(guard_path)
    assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['proof_sha256']==sha(path)
    assert len(problems)==group['expected_coupling_problems']
    groups[key]=report;bindings[key]=dict(report_path=str(path),report_sha256=sha(path),guard_path=str(guard_path),guard_sha256=sha(guard_path),prior_evidence_reused=reused)
    for r in problems:
        assert r['regularization']==2. and r['tolerance']==1e-6 and r['actual_absolute_marginal_error']<=1e-6 and not r['uniform_or_exact_OT_fallback']
        rows.append(dict(group=key,epoch=r['epoch'],batch_start=r.get('batch_start',0),batch_size=r['batch_size'],
            regularization=r['regularization'],tolerance=r['tolerance'],original_final_cap=r['original_final_cap'],extended_final_cap=r['extended_final_cap'],
            used_extended_cap=r['used_extended_cap'],original_failure=r['original_failure'],actual_absolute_marginal_error=r['actual_absolute_marginal_error'],
            x0_sha256=r['x0_sha256'],x1_sha256=r['x1_sha256'],coupling_sha256=r['coupling_sha256'],prior_evidence_reused=reused))
    for cid in group['covered_original_ids']:
        case_map.append(dict(canonical_original_id=cid,remaining_eight_case=cid in c['remaining_original_cases'],qualified_coupling_group=key,
            original_fit_points=group['fit_points'],original_fit_windows=group['fit_windows'],original_epochs=20,
            original_coupling_problems=group['expected_coupling_problems'],full_neural_training_qualified=False,new_complete_model_results=0))
assert len(rows)==80 and sum(not r['prior_evidence_reused'] for r in rows)==60 and len(case_map)==9
assert sum(r['remaining_eight_case'] for r in case_map)==8 and sum(r['used_extended_cap'] for r in rows)==3
assert len({(r['x0_sha256'],r['x1_sha256']) for r in rows})==80
for count,path,failures in [(28,'tmp/industrial_remaining8_shared28_validation_20261016_v2.xml',0),(6,'tmp/industrial_remaining8_couplings_validation_20261016_v1.xml',3)]:
    suites=list(ET.parse(ROOT/path).getroot().iter('testsuite'));assert sum(int(x.get('tests',0)) for x in suites)==count
    assert sum(int(x.get('failures',0)) for x in suites)==failures and all(int(x.get('errors',0))==0 for x in suites)
out=ROOT/'projects/flow_matching_research/results/2026-10-16/industrial_remaining8_all80_real_couplings_qualified_v1';out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8',newline='\n')
for key,b in bindings.items():
    shutil.copyfile(b['report_path'],out/(key+'_report.json'));shutil.copyfile(b['guard_path'],out/(key+'_guard.json'))
for name,data in [('all80_real_coupling_problem_rows.csv',rows),('all9_original_case_transport_component_coverage.csv',case_map)]:
    with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
write(out/'all3_full_sequence_proof_bindings.json',dict(groups=bindings,prior20_problems_reused=True,new_problems_actually_solved=60))
for name,path in dict(frozen_registration=ROOT/s.CONFIG,actual_launch=base/'launch.json',actual_terminal_state=base/'status.json',
    fresh28_source_RNG_and_solver_validation=ROOT/'tmp/industrial_remaining8_shared28_validation_20261016_v2.xml',
    preserved_first_test_CUDA_isolation_failure=ROOT/'tmp/industrial_remaining8_couplings_validation_20261016_v1.xml').items():shutil.copyfile(path,out/(name+path.suffix))
write(out/'completion_boundary.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_global_scope=c['original_global_scope'],
    original_CPU_scope=550,original_GPU_scope=850,remaining_original_cases_covered=8,total_related_original_cases=9,
    full_coupling_sequences_qualified=3,new_sequences_actually_executed=2,actual_new_coupling_problems=60,prior_qualified_problems_reused=20,
    total_distinct_real_problems=80,original_success_plans_returned_unchanged=77,problems_using_extended_cap=3,
    final_regularization=2.,unchanged_absolute_tolerance=1e-6,original_iteration_cap=20000,extended_iteration_cap=200000,
    iteration_caps_are_not_exact_iteration_counts=True,old_failed_artifacts_and_all_source_jobs_preserved=True,
    original_full_training_commit_admission_bytes=32*2**30,full_model_training_qualified=False,
    new_neural_forwards=0,new_optimizer_updates=0,new_training_seed_slots=0,new_complete_model_results=0,
    all_original_experiments_complete=False))
(out/'README.md').write_text('''# 剩余8项工业熵运输任务：全部相关真实耦合序列通过

原PRONTO day4/seed1103的20个耦合证据复用；新实际运行day4/seed1104的20个和day3/seed1107的40个，共新增60个真实问题。
总计80个不同真实端点问题，覆盖9个相关原条件，其中8个尚未完成完整神经训练。全722/6857训练点、8/70窗口、原20轮与64批量保留；70窗口每轮为64+6的两个完整批次。

全部保持epsilon=2、均匀边际和1e-6绝对容差。原20k上限失败的位置为day4/1103 epoch8（旧证据1.03e-6）、day4/1104 epoch6（1.33e-6）、day3/1107 epoch17尾批6（1.32e-6）。
仅这3个问题使用同一求解函数将最终上限扩展至200k后通过；其余77个直接返回原求解结果，无均匀/精确OT替代、正则变更或容差放宽。
200k为上限，实际迭代次数未单列，不能说每个问题都运行200k次。

SB、SF2M和关闭score分支的SF2M在相同数据/种子/窗口/批量下共享局部Generator的运输问题序列。三种源类小型完整训练夹具均与无神经网络诊断的端点逐字节一致，包括尾批，28项相关检查通过。
网络仅Linear/SiLU，模型初始化使用隔离全局RNG，额外score损失不消费训练局部Generator；共用耦合证据不等于三个完整网络或损失均已训练验收。

首次6项测试中的3项在端点已一致后，于CUDA未初始化断言失败：CPU训练夹具的测试环境当时仍可见GPU。按实际数学worker相同的CUDA_VISIBLE_DEVICES=-1隔离后，28项全通过，首次XML保留；没有放松数值或更改原训练代码。
实际两个新诊断子进程也显式隐藏CUDA，退出0、无资源中止，保持原数学组件22GiB冷启动/20GiB导入后提交余量、16GiB紧急线和2GiB进程上限。

这是运输组件验证，网络前向、优化更新、新训练种子与完整模型成绩新增均为0。源数据、旧失败、原配置和全部已登记模型任务未改写；原完整训练32GiB准入和共享锁保持。
后续仍须完整训练20轮、全部评分点、检查点重放及独立指标核验；不把本包作为原论文或完整TAB性能，整体未完成。
''',encoding='utf-8',newline='\n')
shutil.copyfile(__file__,out/'publication_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/industrial_remaining8_all80_couplings_qualified.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(packet=out.relative_to(ROOT).as_posix(),related_original_cases=9,remaining_original_cases=8,
    new_real_problems=60,prior_real_problems=20,full_neural_training_complete=False,all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8')
text+='''
`工业剩余8条件的相关80个真实运输问题全部通过 <results/2026-10-16/industrial_remaining8_all80_real_couplings_qualified_v1/README.md>`_
复用旧20问题，新执行20+40问题，覆盖全部722/6857训练点与8/70窗口；三个超原20k上限问题在同求解器200k上限内满足原epsilon=2、1e-6容差。
同数据/种子的SB/SF2M/flow-only共用运输序列，28项含源类网络/尾批RNG一致性检查通过；共用证据不等于完整网络训练完成。
新增完整模型成绩0，所有原任务/数据/失败保持；完整训练32GiB及原共享锁未改，仍须完成训练、重放和指标验收。
'''
page.write_text(text,encoding='utf-8',newline='\n')
prior=read(ROOT/'tmp/GRASP_ER1107_full52_publication_manifest_20261016_v1.json')
gethead=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
assert prior['previous_root']==gethead(ROOT) and prior['previous_parent']==gethead(ROOT.parent/'flow-matching-research')
paths=set(prior['paths'])|{s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG,ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()}
paths|={p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()}
write(ROOT/'tmp/GRASP52_and_industrial_remaining8_publication_manifest_20261016_v1.json',dict(previous_root=gethead(ROOT),previous_parent=gethead(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(out),publication_files=len(paths),new_coupling_problems=60,total_coupling_problems=80,GRASP_complete_models=52,new_industrial_complete_models=0)))
