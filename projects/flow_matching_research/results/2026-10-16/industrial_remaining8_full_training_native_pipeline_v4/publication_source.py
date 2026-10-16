"""Archive the actual idle-only handoff and eight full, guarded obligations."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import continue_industrial_remaining8_extended_fullcases_v4 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/s.CONFIG);s.verify(c);s.legacy_layout()
base=ROOT/s.STATE;launch=read(base/'launch.json');state=read(base/'status.json');handoff=read(base/'handoff.json')
owner=psutil.Process(launch['pid']);assert owner.is_running() and owner.create_time()==launch['create_time']==state['create_time'] and owner.cmdline()==launch['command']
assert state['state'] in ['waiting_for_memory','waiting_for_heavy_lock'] and state['active_pid'] is None and not state['jobs']
assert handoff['previous_owner']==c['previous_owner'] and handoff['numerical_workers_stopped'] is False
assert handoff['only_exact_idle_metadata_parent_suspended_rechecked_and_retired']
if psutil.pid_exists(s.EXPECTED['pid']):assert psutil.Process(s.EXPECTED['pid']).create_time()!=s.EXPECTED['create_time']
locks={str(Path(f.path).resolve()).lower() for f in owner.open_files() if f.path.endswith('.lock')}
assert locks=={str((base/'controller.lock').resolve()).lower(),str((ROOT/s.OLD_WORKER_LOCK).resolve()).lower()}
assert all(p.name().lower()=='conhost.exe' for p in owner.children(recursive=True))
assert not (base/'attempts').exists()
rows=[]
for case in c['cases']:
    job=case['job'];assert not (ROOT/job['output_directory']).exists() and not Path(case['native_output']).exists()
    dataset=next(d for d in read(ROOT/job['dataset_manifest_path'])['datasets'] if d['id']==job['dataset'])
    model=read(ROOT/job['model_config'])
    rows.append(dict(canonical_original_id=case['canonical_original_id'],actual_execution_id=job['id'],dataset=job['dataset'],seed=job['seed'],
        model_config=job['model_config'],objective=model['parameters']['objective'],auxiliary_score_weight=model['parameters']['score_weight'],
        original_fit_points=dataset['scaler']['fit_points'],original_fit_windows=case['fit_windows'],original_window=job['window'],
        original_epochs=20,original_batch_size=64,original_optimizer_updates=case['optimizer_updates'],
        full_validation_points=sum(g['points'] for g in dataset['validation_groups']),full_test_points=sum(g['test_points'] for g in dataset['entities']),
        source_output_directory=job['output_directory'],native_output_directory=case['native_output'],source_training_started=False,
        source_training_complete=False,checkpoint_native_replay_complete=False))
assert len(rows)==8 and sorted(r['original_optimizer_updates'] for r in rows)==[20]*5+[40]*3
xml=ROOT/'tmp/industrial_remaining8_fullcase_shared38_validation_20261016_v1.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==38
assert all(int(x.get(k,0))==0 for x in suites for k in ['failures','errors','skipped'])
out=ROOT/'projects/flow_matching_research/results/2026-10-16/industrial_remaining8_full_training_native_pipeline_v4';out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8',newline='\n')
for name,path in dict(frozen_full8_registration=ROOT/s.CONFIG,actual_launch=base/'launch.json',actual_waiting_state=base/'status.json',
    actual_handoff=base/'handoff.json',prelaunch_idle_snapshot=base/'prelaunch_idle_snapshot.json',two_independent_idle_observations=base/'two_pre_handoff_observations.json',
    suspended_metadata_recheck=base/'suspended_metadata_recheck.json',preserved_legacy_idle_status=base/'previous_idle_status.json',
    relevant_fresh38_validation=xml).items():shutil.copyfile(path,out/(name+path.suffix))
with (out/'all8_registered_full_budget_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
handles=[]
for pid,created in [(27336,1792157309.0403843),(13636,1792157846.7278829),(40500,1791915368.3060675),
    (41992,1791948094.8603492),(31444,1792129142.030305),(16360,1792143923.216981)]:
    p=psutil.Process(pid);assert p.is_running() and p.create_time()==created
    handles.append(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline(),cpu_user_seconds=p.cpu_times().user,
        descendants=[dict(pid=x.pid,create_time=x.create_time(),command=x.cmdline()) for x in p.children(recursive=True)]))
write(out/'actual_delegated_first_and_other_scientific_owners_preserved.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=handles))
write(out/'completion_boundary.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_global_scope=c['original_global_scope'],
    original_CPU_scope=550,original_GPU_scope=850,related_original_slots=9,remaining_full_model_slots_registered=8,
    original_first_slot_delegated_to_existing_v3=True,only_old_idle_metadata_controller_retired=True,numerical_workers_stopped=False,
    original_full_training_commit_headroom_bytes=c['settings']['minimum_commit_headroom_bytes'],
    original_emergency_commit_headroom_bytes=c['settings']['emergency_commit_headroom_bytes'],original_full_training_shared_lock=c['resource_lock'],
    new_complete_models=0,actual_new_real_model_training_workers_started=0,
    qualified_transport_components_are_prerequisites_not_model_results=True,
    source_fit_and_train_score_code_objects_unchanged_with_readonly_solver_and_optimizer_observation=True,
    every_actual_plan_endpoint_and_sampled_pair_must_match_qualified_sequence=True,
    native_source_scores_and_metadata_must_match_all_payload_bytes=True,
    native_bootstrap_VUS_affiliation_not_independently_rerun=True,full_TAB_equivalence=False,paper_equivalence=False,
    all_original_experiments_complete=False))
(out/'README.md').write_text('''# 剩余8项工业熵目标：完整训练、检查点重放和独立指标队列已接入

此前80个真实耦合问题已验收，本次把未完成的8个原条件接到完整执行流程；原9条件中首个SB/day4/1103继续由既有v3源训练与原生复核两个控制器负责，不重复登记或停止它们。
五个day4条件保留722训练点/8窗口、20轮/20次更新；三个day3条件保留6857训练点/70窗口、20轮/40次更新。完整4320/722验证点及11700/21300测试点分别保持。

原始稳定实现和所有失败、空目录均保留。仅使用已登记同数学目标的最终迭代上限扩展候选；epsilon=2、1e-6容差、路径、辅助score权重、MLP、batch、种子和全部评分参数不变。
源训练使用原StableCFM.fit代码对象，私有绑定观察求解器输入/输出及真实Adam的只读post-step计数；逐批端点、计划矩阵及抽样配对必须与已验收序列完全一致。
运行原工业train/score代码对象，完整20轮、有限损失、20/40次实际更新以及全部原生分数齐全后，才形成成功源守卫。
随后恢复全部权重，不再训练，重放所有训练/验证/测试点及尾部，要求每个评分/标签/时间戳/原行号/故障字符串的dtype、shape及所有字节一致。
验证0.1%/1%/5%阈值、混淆计数、F1、并列AUROC/AP、逐实体和事件指标由独立公式复算。VUS/Affiliation和bootstrap未在这一步独立重跑，不冒充完整TAB。

38项相关验证通过：三个目标的小型源训练对照权重/损失/分数逐位一致，尾批与真实优化次数正确；小型20轮端到端辅助score分支的源训练→检查点→全部数组→独立指标也通过。
这些夹具验证流程，不是论文数据性能。真实8条件当前没有新训练结果或神经子进程，仍等待原资源准入。

旧PID30860的身份、创建时间、完整命令、状态、两把自身元数据锁、全部旧输出及无数值子进程经过多次实际核验。
只挂起这一已证明空闲的元数据父进程以关闭末次检查后的派生竞争，再复查没有数值子进程后退出它；若派生竞态被检测，测试证明会恢复旧父进程并拒绝替换。
旧首个数值失败保存，8个旧恢复尝试尚未开始；新控制器持有原worker.lock防止旧队列同时派发，原全训练共享锁保持。
原12GiB物理/32GiB提交准入、16GiB紧急线及既有2GiB RSS/4GiB私有上限保持，源训练和原生重放各自重新获取资源；等待时不持有共享训练锁。
没有停止数值进程、重试既有数值失败、降低训练预算或覆盖源数据。任何一次新训练/重放失败均保留，不自动重试。

实际新控制器已启动并接管元数据锁，目前等待内存准入；首个SB的两个既有控制器和其他原CPU/GPU数值任务均仍运行。
新完整模型成绩为0。接入完整队列不等于8项模型已经训练完成或已经达到论文指标；原550CPU/850GPU及45/681/273/702范围保持，整体未完成。
''',encoding='utf-8',newline='\n')
shutil.copyfile(__file__,out/'publication_source.py');write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/industrial_remaining8_fullcase_pipeline.2026-10-16_v4.json';assert not ara.exists()
write(ara,dict(packet=out.relative_to(ROOT).as_posix(),full_original_models_registered=8,new_complete_models=0,original_first_case_delegation_preserved=True,all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8')
text+='''
`工业剩余8项接入完整源训练与原生复核流程 <results/2026-10-16/industrial_remaining8_full_training_native_pipeline_v4/README.md>`_
38项相关验证通过，保留原数据/20轮/64批量/种子/评分；逐批观察真实优化次数和已验收耦合，完整权重重放及独立三阈值指标随后验收。
仅交接经多次及挂起后无数值子进程复核的旧空闲元数据控制器；首个SB既有v3源训练/复核及所有原数值进程保持。
新8条件队列已真实启动，持原worker锁，仍按原32GiB提交及共享训练锁等待，新增完整模型0；原550CPU/850GPU及全局范围保持。
'''
page.write_text(text,encoding='utf-8',newline='\n')
paths=[s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG,'scripts/flow_matching/industrial_extended_fullcase_common_v4.py',
    'tests/test_fm_industrial_remaining8_fullcase_pipeline_v4.py',ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()]
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
write(ROOT/'tmp/industrial_remaining8_fullcase_pipeline_publication_manifest_20261016_v4.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(out),publication_files=len(paths),actual_new_owner=owner.pid,registered_full_cases=8,new_complete_models=0)))
