from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import continue_industrial_SB_math_diagnostic_lock_v4 as m
from scripts.flow_matching import continue_industrial_SB_transport_extension_v3 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/m.CONFIG);assert c==m.registration()
runtime=read(ROOT/s.CONFIG);assert runtime==s.registration()
base=ROOT/s.STATE;handoff=read(ROOT/m.BASE/'handoff.json')
assert handoff['only_idle_metadata_owner_stopped'] and not handoff['numerical_processes_stopped']
assert handoff['previous_owner']==m.PREVIOUS and handoff['full_training_lock_and_all_original_guards_unchanged']
report=read(s.REPORT);guard=read(base/'attempts/coupling_diagnostic/receipt.json')
assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['proof_sha256']==sha(s.REPORT)
assert report['passed'] and len(report['coupling_problems'])==20
assert report['new_neural_forwards']==report['new_optimizer_updates']==report['new_training_seeds']==0
assert report['actual_first_original_nonconvergence_epoch']==8 and sum(x['used_extended_cap'] for x in report['coupling_problems'])==1
assert all(x['regularization']==2. and x['tolerance']==1e-6 and x['actual_absolute_marginal_error']<=1e-6 for x in report['coupling_problems'])
job=read(base/'full_original_case_job.json');assert job==s.full_job_registration(runtime,s.verify(runtime))
assert job['full_original_epochs']==20 and job['full_original_fit_windows']==8 and job['full_original_test_points']==11700
state_bytes=(base/'status.json').read_bytes();state=json.loads(state_bytes)
owner=psutil.Process(state['pid']);assert owner.is_running() and owner.create_time()==state['create_time']
assert owner.cmdline()==read(ROOT/m.BASE/'launch.json')['command']
assert not state['full_training_complete'] and not (base/'full_job/result.json').exists()
xml=ROOT/'tmp/industrial_SB_math_diagnostic_lock_v4_validation_20261016_v4.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==15 and all(int(x.get('failures',0))==int(x.get('errors',0))==int(x.get('skipped',0))==0 for x in suites)
out=ROOT/'projects/flow_matching_research/results/2026-10-16/industrial_SB_PRONTO1103_full20_coupling_qualified_v4';out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
paths=dict(actual_all20_real_coupling_problems=Path(s.REPORT),actual_diagnostic_guard=base/'attempts/coupling_diagnostic/receipt.json',
    actual_full_original_case_job=base/'full_original_case_job.json',original_v3_runtime=ROOT/s.CONFIG,scheduling_v4_runtime=ROOT/m.CONFIG,
    actual_metadata_handoff=ROOT/m.BASE/'handoff.json',actual_two_idle_owner_observations=ROOT/m.BASE/'two_pre_handoff_observations.json',
    preserved_previous_idle_status=ROOT/m.BASE/'previous_idle_status.json',actual_new_scheduler_launch=ROOT/m.BASE/'launch.json')
for label,p in paths.items():shutil.copyfile(p,out/(label+'.json'))
(out/'actual_after_math_diagnostic_status.json').write_bytes(state_bytes)
shutil.copyfile(xml,out/'relevant_fresh15_scheduler_safety_validation.xml')
for version in [1,2]:
    path=ROOT/f'tmp/industrial_SB_math_diagnostic_lock_v4_validation_20261016_v{version}.xml'
    suites=list(ET.parse(path).getroot().iter('testsuite'));assert sum(int(x.get('failures',0)) for x in suites)==1
    shutil.copyfile(path,out/f'preserved_pre_registration_metadata_test_failure{version}.xml')
rows=report['coupling_problems']
with (out/'all20_real_transport_problem_convergence.csv').open('w',encoding='utf-8-sig',newline='') as f:
    fields=['epoch','batch_size','regularization','tolerance','original_final_cap','extended_final_cap','used_extended_cap','original_failure','actual_absolute_marginal_error','x0_sha256','x1_sha256','coupling_sha256']
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
live=[]
for pid,created in [(40500,1791915368.3060675),(41992,1791948094.8603492),(31444,1792129142.030305),(16360,1792143923.216981)]:
    p=psutil.Process(pid);assert p.is_running() and p.create_time()==created
    live.append(dict(pid=pid,create_time=created,command=p.cmdline(),CPU_user_seconds=p.cpu_times().user))
write(out/'ongoing_original_numerical_owners_preserved.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),actual_live_handles=live))
write(out/'qualified_transport_and_unfinished_full_model_boundary.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    canonical_original_case=s.CANONICAL,actual_complete_original_coupling_problems=20,original_failure_reproduced_at_epoch=8,
    extended_cap_used_for_problems=1,other_original_successes_returned_unchanged=19,
    maximum_actual_marginal_error=max(x['actual_absolute_marginal_error'] for x in rows),original_marginal_tolerance=1e-6,
    final_regularization=2.,solver_final_iteration_caps=dict(original=20000,extended=200000),exact_iteration_count_not_separately_logged=True,
    peak_diagnostic_private_bytes=guard['peak_tree_private_bytes'],peak_diagnostic_rss_bytes=guard['peak_tree_rss_bytes'],
    complete_full_training=False,full_checkpoint_native_replay=False,new_model_benchmark_results=0,new_neural_forwards=0,new_optimizer_updates=0,
    original_full_training_commit_admission_bytes=32*2**30,original_emergency_commit_bytes=16*2**30,
    full_training_shared_lock=runtime['resource_lock'],bounded_math_diagnostic_lock=m.DIAGNOSTIC_LOCK,
    scheduling_transformation='Exactly one controller resource-path expression; inverse AST equals original whole controller',
    diagnostic_numerical_command_and_body_unchanged=True,all_original_full_training_guards_unchanged=True,
    original_global_scope=runtime['original_global_scope'],all_original_experiments_complete=False))
(out/'README.md').write_text(f'''# 工业SB／PRONTO day4／seed1103：20个真实耦合问题验收通过

完整722个训练点、两个连续组、8个100×17窗口及seed1103的原随机数序列，20个原训练运输问题均通过原1e-6绝对边际容差。
第8个问题再次得到原v2的1.03e-6未收敛错误；仅此问题使用同一函数、同一12阶段延续、同一最终epsilon=2与均匀边际，将最终迭代上限由20000延长至200000后通过。
其余19个问题直接返回原v2结果；所有原始端点、耦合内容哈希、抽样索引与逐问题误差完整保存。
实际最大边际误差为{max(x['actual_absolute_marginal_error'] for x in rows):.16g}，原阈值保持1e-6。
200000是上限，函数满足条件时即停止；实际迭代次数未单独记录，不能说执行了恰好200000步。

这次是运输组件的完整真实数据诊断，网络前向0、优化器更新0，不是完整20轮训练或模型性能。
原代码只有Linear/SiLU网络，使用独立局部Generator；诊断按原训练顺序消耗randperm、端点噪声、配对抽样、time和噪声调用，省略的网络/损失/Adam不消费该Generator。
源训练代码对象、完整随机流和全部数据哈希绑定保留；这不代替完整模型损失、权重及测试指标验收。

为了让这个至多2GiB私有内存的纯数学诊断不排在长时CPU网络训练后，新调度器只改了一处控制器资源锁路径选择。
诊断使用独立数学锁及原22GiB冷启动、导入依赖库后20GiB提交要求；完整训练继续使用原共享CPU锁、12GiB物理/32GiB提交准入、16GiB紧急线及原RSS/private限制。
两次独立身份/无数值子进程/无尝试输出/仅持有自身元数据锁检查后，只停止先前新建的空闲元数据PID24480。
原CPU训练PID41992、其控制器40500、CUDA CrossAD及Spectral都仍在实际运行，未停止任何数值进程；交接前后证据归档。
15项调度安全检查通过，包括把新锁选择逆替换后整个控制器AST完全等于原件，以及身份、数值子进程、输出已开始或持有训练锁时拒绝交接。

两份初期测试XML保留：其失败来自把嵌套代码对象身份/字节码索引当成锁变换后的等价判据；后续改为整个AST的单点逆替换一致性检查。
这些是生产注册前的元数据测试失败，未启动模型或改变计算参数，不计为实验结果。原求解器与候选模型此前22项检查仍分别保留。

已经生成新的完整原case绑定，原20轮、20次更新、8个fit窗口、4320验证点与11700测试点及所有角色/超参数保持。
当前完整训练尚未开始，等待原32GiB提交余量和原共享CPU锁；该新身份输出独立，旧失败与其他8项工业恢复队列均保持。
源训练完成后还必须通过完整检查点重放、全部评分数组及独立指标；目前新增完整模型成绩为0。
实际验证限这个SB案例的耦合序列，不把另外两个候选SF2M配置说成已经完成真实训练。
原550CPU/850GPU工业范围、45论文/681方法记录/273数据任务/702标量和原论文/TAB/新数据任务保持，整体未完成。
''',encoding='utf-8',newline='\n')
source_files=[m.MODULE.replace('.','/')+'.py',m.TEST,m.CONFIG]
write(out/'new_scheduler_source_bindings.json',dict(files=[dict(path=x,sha256=sha(ROOT/x)) for x in source_files],
    unchanged_original_runtime_sha256=sha(ROOT/s.CONFIG),unchanged_original_controller_sha256=sha(ROOT/(s.MODULE.replace('.','/')+'.py'))))
shutil.copyfile(__file__,out/'publication_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/industrial_SB_PRONTO1103_real20_couplings_qualified.2026-10-16_v4.json'
assert not ara.exists();write(ara,dict(packet=out.relative_to(ROOT).as_posix(),real_coupling_problems_passed=20,
    full_original_model_complete=False,new_model_benchmark_results=0,all_original_experiments_complete=False))
page=ROOT/'projects/flow_matching_research/README.rst';text=page.read_text(encoding='utf-8')
text+='''
`SB／PRONTO1103的20个真实运输问题已验收 <results/2026-10-16/industrial_SB_PRONTO1103_full20_coupling_qualified_v4/README.md>`_
第8个问题重现原1.03e-6未收敛，延长同问题最终迭代上限后通过；其余19个直接返回原v2结果，全部保持原epsilon=2和1e-6容差。
实际最大边际误差约9.999596e-7；诊断未运行网络或优化器，完整模型成绩新增0。
只交接经多次证明无数值子进程的空闲元数据控制器，纯数学诊断使用独立有界锁，完整训练仍按原共享锁和32GiB准入排队。
15项调度安全检查通过，原CPU/GPU数值训练均继续运行；完整20轮、检查点重放及测试指标仍未完成。
'''
page.write_text(text,encoding='utf-8',newline='\n')
source_files += [ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()]
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
write(ROOT/'tmp/industrial_SB_real20_couplings_v4_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(source_files)))
print(json.dumps(dict(packet=str(out),files=len(source_files),complete_real_couplings=20,new_complete_models=0)))
