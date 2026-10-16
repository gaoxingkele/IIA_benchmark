from datetime import datetime,timezone
from pathlib import Path
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import continue_industrial_SB_transport_extension_v3 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

c=read(ROOT/s.CONFIG);assert c==s.registration();case=s.verify(c)
base=ROOT/s.STATE;state_bytes=(base/'status.json').read_bytes();state=json.loads(state_bytes)
p=psutil.Process(state['pid']);assert p.create_time()==state['create_time'] and p.is_running()
assert p.cmdline()==read(base/'launch.json')['command'] and not state['full_training_complete']
assert not (base/'full_job/result.json').exists()
xml=ROOT/'tmp/industrial_SB_extended_solver_full22_validation_20261016_v2.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==22 and all(int(x.get('failures',0))==int(x.get('errors',0))==int(x.get('skipped',0))==0 for x in suites)
out=ROOT/'projects/flow_matching_research/results/2026-10-16/industrial_SB_PRONTO1103_same_objective_solver_extension_candidate_v3';out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.py -text\n*.xml -text -whitespace\n*.log -text -whitespace\n',encoding='utf-8')
prior=ROOT/case['job']['output_directory']
stderr=ROOT/read(ROOT/s.ORIGINAL_QUEUE)['settings']['output_root']/'logs'/case['job']['id']/'stderr.log'
for label,path in dict(preserved_original_failure=prior/'failure.json',preserved_original_resources=prior/'resource_usage.json',
    preserved_original_stderr=stderr,frozen_runtime=ROOT/s.CONFIG,actual_new_controller_launch=base/'launch.json').items():
    suffix='.log' if path.suffix=='.log' else '.json';shutil.copyfile(path,out/(label+suffix))
(out/'actual_controller_status_snapshot.json').write_bytes(state_bytes)
shutil.copyfile(xml,out/'relevant_fresh22_validation.xml')
models=[]
for name in ['sb','sf2m','sf2m_flow_only']:
    path=f'configs/models/fm_objective_{name}_stable_extended_v3.json';m=read(ROOT/path)
    old=read(ROOT/m['numerical_repair']['original_config']);parameters=dict(m['parameters']);assert parameters.pop('sinkhorn_max_final_iterations')==200000 and parameters==old['parameters']
    models.append(dict(path=path,sha256=sha(ROOT/path),original_config=m['numerical_repair']['original_config'],same_all_original_training_scoring_parameters=True))
    shutil.copyfile(ROOT/path,out/(name+'_candidate_model.json'))
write(out/'current_reproduction_boundary.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),canonical_original_case=s.CANONICAL,
    candidate_models=models,actual_live_controller=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()),
    full_real_coupling_diagnostic_passed_and_guarded_receipt_available=(base/'attempts/coupling_diagnostic/receipt.json').exists() and read(base/'attempts/coupling_diagnostic/receipt.json').get('passed',False),
    full_real_original_training_complete=False,full_independent_checkpoint_replay_complete=False,new_benchmark_performance_results=0,
    original_solver_and_failures_preserved=True,original_absolute_marginal_tolerance=1e-6,original_final_epsilon=2.,
    original_final_solver_iteration_cap=20000,candidate_max_final_solver_iterations=200000,changed_full_training_updates=0,
    successful_original_cases_bitwise_same_loss_weight_and_score_in_small_test=True,
    initial_direct_diagnostic_refused_before_coupling_due_to20GiB_post_import_admission=True,
    diagnostic_cold_admission_commit_bytes=22*2**30,diagnostic_only_no_optimizer_or_neural_forward=True,
    original_full_training_admission_commit_bytes=32*2**30,original_emergency_commit_bytes=16*2**30,
    original_industrial_CPU_cases=550,original_industrial_GPU_cases=850,original_global_scope=c['original_global_scope'],
    tests_passed=22,all_original_experiments_complete=False))
files=[s.MODULE.replace('.','/')+'.py',s.DIAGNOSTIC.replace('.','/')+'.py','src/iia_benchmark/models/fm_objective_stable_extended.py',
    'tests/test_fm_objective_stable_extended.py','tests/test_fm_industrial_SB_transport_extension_v3.py',s.CONFIG]+[x['path'] for x in models]
write(out/'candidate_source_bindings.json',dict(files=[dict(path=x,sha256=sha(ROOT/x)) for x in files],
    original_solver_path='src/iia_benchmark/models/fm_objective_stable.py',original_solver_sha256=sha(ROOT/'src/iia_benchmark/models/fm_objective_stable.py')))
(out/'README.md').write_text('''# 工业SB／PRONTO day4／seed1103：同目标求解迭代扩展候选

发现并保留一个实际未完成实验：原stable_v2完整工业恢复在Sinkhorn边际误差1.03e-6时退出，原要求为1e-6。
旧stderr、exit1、未资源中止回执及全部失败输出保持。完整任务仍为722个原始训练点、两个独立连续组、8个100×17窗口、20轮、batch64与20次更新，11700测试点。

新增单独实现和SB/SF2M/flow-only三份候选配置。所有v2已收敛情况直接返回原计划，不改变权重、RNG、训练损失或评分。
只有原来明确的最终迭代未收敛错误才调用相同函数将最终上限20000延长到200000；12阶段温度延续、最终epsilon=2sigma²、均匀边际及1e-6容差均保持。
没有均匀或exact-OT替代，没有降低容差，也没有改变训练轮数、批次、网络、学习率、损失、评分时间网格或蒙特卡洛次数。
通过私有FunctionType绑定原fit代码对象的solver名称，原冻结模块不修改，source工业训练/评分主体也复用原函数代码对象。

22项相关检查通过，包括已收敛小案例的损失/全部权重/分数逐字节相同、只捕获指定未收敛、错误与坏边际不隐藏、原训练与数据字段不变，以及控制器保持冷元数据状态。
这些小型检查不代表原始PRONTO完整训练已经成功。实际首次直接耦合诊断在进口库后未满足20GiB提交余量而拒绝，尚未执行耦合计算。

受保护控制器已经启动，先等待共享CPU锁及22GiB提交余量，逐一诊断完整20个原始训练耦合问题，消耗原相同随机数序列，不运行网络或优化器。
耦合报告必须满足全部原边际误差且进程与守卫成功，才绑定新的完整实验身份/输出。完整训练仍保留原12GiB物理、32GiB提交准入和16GiB紧急线，共享CPU锁不变。
诊断和训练均有进程内存上限，失败保留不自动重试；现有其他源训练和9项旧工业恢复队列不停止、不改写。
只为这个已确证终止的原case安排完整后继，不为其他8个仍由旧队列负责的案例启动重复数值任务。

当前是待真实耦合及完整训练验收的候选实现，新增完整模型成绩为0。后续即使源训练通过，完整检查点重放和独立指标仍需另外验收。
延长运输求解不是增加训练种子，当前官方目标函数的本地TSAD适配也不认证原图像/单细胞论文成绩或完整TAB协议。
原550CPU/850GPU工业任务、全部45篇/681方法记录/273数据任务/702标量和原论文/TAB/新数据范围保持，整体未完成。

引用保留于三份候选配置：[CFM目标函数论文](https://arxiv.org/abs/2302.00482)、原TorchCFM固定提交及PRONTO登记数据来源。
''',encoding='utf-8',newline='\n')
shutil.copyfile(__file__,out/'publication_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/industrial_SB_PRONTO1103_solver_extension_candidate.2026-10-16_v3.json'
assert not ara.exists();write(ara,dict(packet=out.relative_to(ROOT).as_posix(),canonical_original_case=s.CANONICAL,
    real_full_original_training_complete=False,candidate_unit_checks_passed=22,new_benchmark_performance_results=0,all_original_experiments_complete=False))
files += [ara.relative_to(ROOT).as_posix(),'projects/flow_matching_research/README.rst']+[p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()]
head=lambda d:subprocess.check_output(['git','-C',str(d),'rev-parse','HEAD'],text=True).strip()
write(ROOT/'tmp/industrial_SB_extension_candidate_manifest_20261016_v3.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(files)))
print(json.dumps(dict(packet=str(out),files=len(files),new_complete_model_results=0,live_controller_pid=p.pid if hasattr(p,'pid') else state['pid'])))
