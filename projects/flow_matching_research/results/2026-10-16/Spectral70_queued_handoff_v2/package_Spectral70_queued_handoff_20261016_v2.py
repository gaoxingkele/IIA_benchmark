from pathlib import Path
from datetime import datetime,timezone
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import continue_spectral70_queued_handoff_v2 as s
ROOT,read,sha=s.ROOT,s.read,s.sha
write=s.previous.fair.exclusive
config=read(ROOT/s.CONFIG);c,parent,cases=s.verify(config)
base=ROOT/config['state_root'];launch=read(base/'launch.json')
actor=psutil.Process(launch['pid'])
assert actor.create_time()==launch['create_time'] and actor.cmdline()==launch['command']
state=read(base/'status.json')
assert state['state']=='waiting_pending_Windows_GPU_lock_no_process_stopped'
assert not state['kernel_lock_owned'] and not (base/'handoff.json').exists()
assert not (base/'controller.stderr.log').read_bytes()
old=s.exact_process(config['previous_owner']);observer=s.exact_process(config['previous_observer'])
oldstate=read(ROOT/c['state_root']/'status.json')
assert oldstate['active_condition']=='spectral_original__diffusion_ts__energy__released_author_data__seed43'
handle=oldstate['exact_child_handle'];child=psutil.Process(handle['pid'])
assert child.create_time()==handle['create_time'] and child.cmdline()==handle['command']
tests=Path(launch['validation_path']);assert sha(tests)==launch['validation_sha256']
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(v.get('tests',0)) for v in suites)==23
assert not any(int(v.get(k,0)) for v in suites for k in ['errors','failures','skipped'])
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/Spectral70_queued_handoff_v2'
packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.xml -text -whitespace\n*.py -text\n',encoding='utf-8')
for name in ['launch.json','status.json']:
    shutil.copyfile(base/name,packet/('actual_scheduler_'+name))
write(packet/'actual_original_condition_state.json',oldstate)
write(packet/'actual_numeric_and_metadata_handles.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    queued_successor=dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline()),
    old_owner=dict(pid=old.pid,create_time=old.create_time(),command=old.cmdline()),
    old_waiting_observer=dict(pid=observer.pid,create_time=observer.create_time(),command=observer.cmdline()),
    original_numeric_launcher=dict(pid=child.pid,create_time=child.create_time(),command=child.cmdline()),
    original_numeric_descendants=[dict(pid=v.pid,create_time=v.create_time(),command=v.cmdline())
        for v in child.children(recursive=True) if v.name().lower()=='python.exe'],
    no_numerical_process_stopped=True,actual_handoff_completed=False))
write(packet/'unchanged_full70_numerical_binding.json',dict(original_configuration=s.previous.original.CONFIG,
    original_configuration_sha256=sha(ROOT/s.previous.original.CONFIG),worker_module=s.previous.original.MODULE,
    all_conditions=70,all_cohorts=14,settings=c['settings'],source_receipts=c['source_receipts'],
    spectral_flow_minimum_commit_headroom_bytes=c['spectral_flow_minimum_commit_headroom_bytes'],
    original_scientific_state_root=c['state_root'],original_artifact_root=c['artifact_root']))
write(packet/'iteration_status.json',dict(original_scope=c['original_scope'],actual_handoff_completed=False,
    actual_kernel_wait_request_live=True,tests_passed=23,new_complete_benchmark_conditions=0,
    new_model_training_updates=0,numerical_workers_stopped=0,all_original_experiments_complete=False))
shutil.copyfile(tests,packet/'validation_23_cases.xml')
wait=ROOT/'tmp/Spectral70_queued_handoff_numeric_wait_20261016_v2.json'
assert wait.exists()
shutil.copyfile(wait,packet/'verified_numeric_wait_45_seconds.json')
(packet/'README.md').write_text('''# 全70生成条件：等待实际共享GPU锁的独立继任版本

原始完整条件结束后，原父进程立即扫描并重新争锁。v1轮询观察器在Energy42完成并进入Energy43后仍未接管，故不能把其启动描述为实际公平调度已经生效。本v2保留这段历史，不修改正在运行的v1或原科学配置。

新增Windows异步LockFileEx等待请求，锁定同一原始GPU文件的offset0、length1，与现有msvcrt锁互斥。请求在系统中等待，不缩短数值阶段、不暂停数值进程。可用锁、已有msvcrt持锁、释放后立即争锁、取消尚未获得的请求及字节内容保留均实际测试通过。取消请求时处理正常授予与取消的竞争，并等操作完成后才释放OVERLAPPED内存。

参考[LockFileEx官方接口](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-lockfileex)及[CancelIoEx取消语义](https://learn.microsoft.com/en-us/windows/win32/api/ioapiset/nf-ioapiset-cancelioex)。这里没有承诺Windows对所有等待者提供全局FIFO顺序。

取得实际锁仍不足以接管：原父进程与v1等待观察器必须按PID、创建时间、完整命令核验两次；只允许严格限定的console后代，没有数值后代；原active child必须有精确终止守卫；每个已有尝试都在成功或保留失败集合中。暂停也仅限实际共享锁已拥有且已证明空闲的元数据父进程。第二次检查若失败，恢复元数据进程，不终止。所有成功原报告、守卫、全生成/RNG文件、源产物与四类各5次评估先重核，再退役这两个空闲元数据进程并取得它们的控制器锁，保存两份旧状态原始字节。

接续沿用v1经检验的原控制器AST变换，只私有绑定v2元数据配置，不改原模块、原配置或工作命令。原始Popen、守卫与保存过程AST保持；全部70条件/14组、两次完整原始采样、四类各5次评估、原求解器/精度/线程/采样数量/16GiB GPU准入及48GiB Spectral Flow提交内存准入保持。成功与失败槽保留，不重新启动旧槽；每个新完整条件后释放锁并等待35秒。

23项相关检查通过，包括真实Windows文件锁兼容/取消、精确身份漂移后恢复、数值后代/未完成尝试/非终止子进程拒绝、全部原预算与数值启动AST不变。元数据探测未导入NumPy/Torch，也没有执行数值工作。

归档时继任进程已经实际提交等待请求、尚未获得共享锁，也尚未接管或停止任何进程。Energy43、CrossAD下一消融、CFM和原CPU CrossAD在45秒精确身份复查中均有CPU推进。此轮没有新增完整实验或训练更新，不把观察器、测试或部分阶段算作完成。

全部45论文/681方法记录/273数据任务/702既有指标以及原论文、TAB、新数据的未完成项继续保留，整体目标未完成。
''',encoding='utf-8')
for name in ['launch_Spectral70_queued_handoff_20261016_v2.py','package_Spectral70_queued_handoff_20261016_v2.py']:
    shutil.copyfile(ROOT/'tmp'/name,packet/name)
write(packet/'packet_integrity.json',dict(files=[dict(path=v.relative_to(packet).as_posix(),sha256=sha(v))
    for v in sorted(packet.rglob('*')) if v.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/Spectral70_queued_handoff.2026-10-16_v2.json'
write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),actual_handoff_completed=False,
    actual_kernel_wait_request_live=True,tests_passed=23,new_complete_benchmark_conditions=0,
    all_original_experiments_complete=False))
paths=[s.CONFIG,s.MODULE.replace('.','/')+'.py','scripts/flow_matching/win32_queued_byte_lock_v1.py',s.TEST,
       ara.relative_to(ROOT).as_posix()]+[v.relative_to(ROOT).as_posix() for v in packet.rglob('*') if v.is_file()]
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD']).decode().strip()
write(ROOT/'tmp/Spectral70_queued_handoff_manifest_20261016_v2.json',dict(previous_root=head(ROOT),
    previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),actual_waiting_successor=actor.pid,
    actual_handoff_completed=False,new_complete_benchmark_conditions=0)))
