from pathlib import Path
from datetime import datetime,timezone
import json,shutil,subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import bridge_TAB878_queued_boundary_v1 as s

root=Path.cwd();out=root/'projects/flow_matching_research/results/2026-10-16/TAB878_queued_boundary_bridge_v1'
if out.exists():assert {p.name for p in out.iterdir()}<={'actual_bridge_and_existing_fair_successor.json','actual_original_TAB_progress.json'}
else:out.mkdir(parents=True,exist_ok=False)
read,sha=s.read,s.sha
def write(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
c=read(root/s.CONFIG);source=s.verify(c);base=root/c['state_root'];h=read(base/'launch.json');state=read(base/'status.json')
def observe(handle):
    try:
        p=psutil.Process(handle['pid']);return dict(expected=handle,alive=p.is_running(),exact_identity=p.create_time()==handle['create_time'] and ('command' not in handle or p.cmdline()==handle['command']),
            actual_command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2]))
    except psutil.NoSuchProcess:return dict(expected=handle,alive=False)
bridge=observe(h);handed=(root/c['successor_state_root']/'handoff.json').exists()
assert bridge.get('exact_identity') or state['state']=='existing_fair_successor_handoff_observed' and handed
write('actual_bridge_and_existing_fair_successor.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    bridge=bridge,bridge_state=state,stderr_bytes=(base/'controller.stderr.log').stat().st_size,
    fair_successor=observe(c['successor_handle']),fair_state=read(root/c['successor_state_root']/'status.json'),
    original_owner=observe(c['previous_owner']),handoff_completed=handed))
original=read(root/source['state_root']/'status.json');write('actual_original_TAB_progress.json',original)
reviewpath=root/'experiments/runs/fm_saits_completion_replay_queue_v1/children/saits_paper_brits_electricity_author_missing0.5_seed29/review/status.json'
write('actual_other_readonly_review_state.json',dict(process=observe(dict(pid=31092,create_time=1792097303.5664525)),state=read(reviewpath)))
basef=Path('F:/aicoding/IIA_Data/experiments');xml=basef/'TAB878_queue_bridge_tests_v1.xml';tree=ET.parse(xml)
suites=list(tree.getroot().iter('testsuite'));assert sum(int(v.get('tests',0)) for v in suites)==19
assert not sum(int(v.get('failures',0))+int(v.get('errors',0)) for v in suites)
shutil.copy2(xml,out/'validation19.xml')
paths=[s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG,
    'scripts/flow_matching/win32_queued_audit_lock_v1.py','tests/test_fm_win32_queued_audit_lock_v1.py']
mapping=[]
for i,path in enumerate(paths):
    dest='source%02d%s'%(i,Path(path).suffix);shutil.copy2(root/path,out/dest);mapping.append(dict(path=path,sha256=sha(root/path),snapshot=dest))
write('validation_and_original_resource_contract.json',dict(passed_tests=19,xml_sha256=sha(xml),source_mapping=mapping,
    actual_cross_process_msvcrt_and_LockFileEx_interoperation_test_passed=True,
    cancellation_does_not_unlock_other_owner_test_passed=True,existing_file_bytes_not_written_by_queued_lock_test_passed=True,
    source_scope_cases=878,label_views=2057,settings=c['settings'],resource_lock=c['resource_lock'],maximum_hold_seconds=2,
    scientific_worker_or_metric_or_budget_changed=False,any_process_suspended_or_terminated_by_bridge=False,
    kernel_queue_is_not_a_general_FIFO_guarantee=True,handoff_completed_at_capture=handed,all_original_experiments_complete=False,
    documentation=[c['documentation'],'https://learn.microsoft.com/en-us/windows/win32/api/minwinbase/ns-minwinbase-overlapped',
        'https://learn.microsoft.com/en-us/windows/win32/api/ioapiset/nf-ioapiset-cancelioex']))
(out/'README.md').write_text('''# 为原 TAB878 接续控制器提供排队式安全边界窗口

原 TAB 在实体完成后立即重新申请审计锁，续接控制器的一秒轮询已实际错过一次释放边界。新增辅助进程用 Windows 原生异步 LockFileEx 排队等待同一文件的字节零锁，收到释放事件后短暂留出元数据窗口，最多持有两秒；观察原父进程因锁等待而进入原十秒重试周期后释放，既有一秒续接者可以尝试接管。若其他合法复核先拿到锁，不打断它，而在下次释放时再次排队。此处不宣称 Windows 各存储实现有普遍严格 FIFO 保证。

原始指标实体和其他复核运行时均持有同一锁，辅助进程拿不到锁，不会暂停、终止或启动任何计算进程，也不重启等待超时的实验。它只排队自己的文件请求，取消时只取消自身请求并等待完成、释放自身句柄。锁文件必须已经存在，原生排队实现不写文件字节。

保留全部 878 项/2057 组标签、全部指标函数与网格、源工作进程、参数、16/28/16 GiB 保护线、3/6 GiB 上限及同一审计锁。辅助进程不派发科学工作，不改变既有成功、失败或部分结果，不生成生产科学守卫。接管依旧由已验证的原续接控制器完成其两次身份/子进程/终止守卫检查。

19 项测试通过，包括真实子进程中原 msvcrt 锁与新 LockFileEx 锁互操作、排队者在释放时取得锁而旧非阻塞锁立即重抢失败、取消排队不解锁其他所有者、不写原锁文件字节，以及异常/有界窗口结束时始终释放自身锁。元数据模拟只使用独立测试目录，不计数据集实验成绩。

实际是否已接管见 actual_bridge_and_existing_fair_successor.json。若 handoff_completed=false，辅助进程仍在实际系统锁队列中等待，原计算保持运行，不能声称 SAITS 已完成或调度问题已经解决。全部原 45 篇论文、681 项方法/基线/消融、273 项数据任务、702 项论文标量目标及作者队列、原生/标签复核、完整 TAB、五训练种子统计继续保留。

接口依据：[Microsoft LockFileEx](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-lockfileex)、[OVERLAPPED](https://learn.microsoft.com/en-us/windows/win32/api/minwinbase/ns-minwinbase-overlapped)、[CancelIoEx](https://learn.microsoft.com/en-us/windows/win32/api/ioapiset/nf-ioapiset-cancelioex)。
''',encoding='utf-8',newline='\n')
shutil.copy2(__file__,out/'publication_source.py')
git=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
manifest=dict(previous_root=git(root),previous_parent=git(root.parent/'flow-matching-research'),
    paths=paths+[p.relative_to(root).as_posix() for p in out.iterdir() if p.is_file()])
(root/'tmp/TAB878_queued_boundary_exact_publication_v1.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(manifest['paths']),tests=19,bridge_alive=bridge.get('alive'),handoff_completed=handed)))
