from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,time
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import run_crossad_retries_cached_idle_v1 as s

root=Path.cwd();out=root/'projects/flow_matching_research/results/2026-10-16/CrossAD_three_idle_retries_cached_v1'
out.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def write(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
c=read(root/s.CONFIG);base=root/c['handoff_state_root'];before=read(base/'actual_three_old_idle_owners_before.json');after=[]
shutil.copy2(base/'actual_three_old_idle_owners_before.json',out/'actual_three_old_idle_owners_before.json')
for name,role in c['roles'].items():
    s.verify(c,name);h=read(base/('launch_'+name+'.json'));p=psutil.Process(h['pid'])
    assert p.create_time()==h['create_time'] and p.cmdline()==h['command']
    state=read(root/role['previous_state_path']);assert state['pid']==p.pid and state['create_time']==p.create_time()
    assert state['active_pid'] is None and state['state']=='waiting_for_memory'
    try:assert psutil.Process(role['previous_owner']['pid']).create_time()!=role['previous_owner']['create_time']
    except psutil.NoSuchProcess:pass
    handed=read(base/name/'handoff.json');assert handed['waiting_controller_only_stopped'] and not handed['original_native_child_stopped']
    for file in ['handoff.json','handoff_identity.json','previous_status_at_verified_handoff.json','successor_entry.json']:
        shutil.copy2(base/name/file,out/(name+'_'+file))
    after.append(dict(role=name,handle=h,actual_command=p.cmdline(),private_bytes=p.memory_info().private,
        rss_bytes=p.memory_info().rss,state=state,stderr_bytes=(base/(name+'.stderr.log')).stat().st_size))
write('actual_three_successors_after.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),owners=after,
    old_private_bytes=sum(v['private_bytes'] for v in before),new_private_bytes=sum(v['private_bytes'] for v in after),
    released_private_bytes=sum(v['private_bytes'] for v in before)-sum(v['private_bytes'] for v in after)))
basef=Path('F:/aicoding/IIA_Data/experiments');xml=basef/'CrossAD_three_cached_idle_tests_v2.xml'
suite=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(t.get('tests',0)) for t in suite)==28
assert not sum(int(t.get('failures',0))+int(t.get('errors',0)) for t in suite)
shutil.copy2(xml,out/'validation28.xml')
paths=[s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG];sources=[]
for i,path in enumerate(paths):
    dest='source%02d%s'%(i,Path(path).suffix);shutil.copy2(root/path,out/dest)
    sources.append(dict(path=path,sha256=sha(root/path),snapshot=dest))
write('validation_and_preserved_initial_configuration_test_failure.json',dict(passed_tests=28,xml_sha256=sha(xml),sources=sources,
    initial_failure=dict(path=str(basef/'CrossAD_three_cached_idle_tests_v1.xml'),sha256=sha(basef/'CrossAD_three_cached_idle_tests_v1.xml'),
        failed=1,passed=27,reason='Registration initially shared the scope dictionary with the helper constant; a mutation test detected this. Copy scope by value before registration. Fixed before any production handoff or scientific action.'),
    actual_original_guard_probes=3,guard_api_snapshot='cached_snapshot',native_entry_main_called_by_probes=False,
    full_original_slots=139,selected_existing_recoveries=['gecco1103','psm1103','smap1103'],
    exact_original_entry_and_arguments_unchanged=True,original_settings=[v['original_settings'] for v in c['roles'].values()],
    original_budgets_20epochs_batch128_patience3=True,all_original_experiments_complete=False))
observations=[]
for pid,ct in [(36912,1792097263.1796665),(40268,1792097260.102444),(41992,1791948094.8603492)]:
    try:
        p=psutil.Process(pid);assert p.create_time()==ct
        observations.append(dict(pid=pid,create_time=ct,command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2]),alive=p.is_running()))
    except psutil.NoSuchProcess:observations.append(dict(pid=pid,alive=False))
write('actual_original_numeric_processes_not_interrupted.json',observations)
write('actual_TAB_and_common105_pending_state.json',dict(
    TAB=read(root/'experiments/runs/fm_TAB_full878_original_metric_suite_v1/status.json'),
    common105_first_family=read(root/'experiments/runs/fm_common105_matched577_v1/families/objectives/status.json')))
(out/'README.md').write_text('''# 三个 CrossAD 恢复控制器的空闲内存回收和缓存遥测

三个已有 GECCO/PSM/SMAP 种子 1103 恢复控制器均确认等待内存，尚无新尝试的输出目录，没有活动计算 PID，全部后代只有 Windows 控制台。交接前后分别核对实际 PID、创建时间、完整命令、状态哈希、所选规范任务、原 20 轮预算及零中断进度信用，暂停后再次核验；只停止空闲控制器，未停止任何科学计算子进程。

三项继续使用原入口、全部原参数、原 139 项任务清单中的明确选择、原数值工作进程和各自恢复登记，不增加种子或挑选成绩。40 GiB 主机准入、原 8 GiB 紧急线、其余资源上限、家族锁、20 轮上限、batch128 和 patience3 完整保留。科学部分原有失败和权重仍保留，没有从不完整状态冒领训练量。

仅在新进程安装已有、经过数值等价验证的 Windows 内存快照缓存，实际原始资源 API 解析到 cached_snapshot，原资源判断和守卫函数不变。原控制入口以完整相同参数重新进入各自的单所有者锁。新状态中的实际 PID 已全部变为新控制器，日志无错误，仍在等待原资源条件。

实际原控制器合计约 563 MiB 私有内存，交接后约 114 MiB，释放约 450 MiB；准确字节及每项实际进程见前后记录。此数值是本次观察，不能保证所有主机资源条件立即满足。相关测试 28 项通过，包括三个原始 API 的独立冷探针、不加载 NumPy/Torch、不执行模型或入口 main，以及原命令、完整科学参数和严格空闲交接条件。

首次测试发现新登记函数错误共享原 scope 字典引用，篡改范围的测试未被拒绝；在任何生产交接和科学操作之前改为复制范围值，随后全部测试通过。失败 XML 保留在 F 盘并记录 SHA256。原算法、已有配置及计算均未因该单元测试改变。

交接不会算作训练完成或新论文指标。实际 CFM、BRITS、CrossAD 种子 1104 科学进程仍存活，另存其身份与 CPU 观测及 TAB 真实进度。原 45 篇论文、681 项方法/基线/消融、273 项数据任务、702 项论文标量目标，完整作者数据集/消融、原生复核、完整 TAB 和五种子统计均继续保留，整体仍未完成。
''',encoding='utf-8',newline='\n')
shutil.copy2(__file__,out/'publication_source.py')
git=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
manifest=dict(previous_root=git(root),previous_parent=git(root.parent/'flow-matching-research'),
    paths=paths+[p.relative_to(root).as_posix() for p in out.iterdir() if p.is_file()])
(root/'tmp/crossad_cached_idle_exact_publication_v1.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(manifest['paths']),tests=28,live_successors=3,
    released_private_bytes=sum(v['private_bytes'] for v in before)-sum(v['private_bytes'] for v in after))))
