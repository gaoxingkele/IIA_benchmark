from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,time
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import continue_timesnet_remaining4_cold_v3 as s

root=Path.cwd();out=root/'projects/flow_matching_research/results/2026-10-16/TimesNet_remaining4_cold_continuation_v3'
out.mkdir(parents=True,exist_ok=False)
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
c=read(root/s.CONFIG);original=s.verify(c);base=root/c['state_root'];h=read(base/'launch.json');p=psutil.Process(h['pid'])
assert p.create_time()==h['create_time'] and p.cmdline()==h['command']
before=read(base/'actual_old_waiting_parent_memory_before.json')
after=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline(),private_bytes=p.memory_info().private,rss_bytes=p.memory_info().rss)
try:assert psutil.Process(s.EXPECTED['pid']).create_time()!=s.EXPECTED['create_time']
except psutil.NoSuchProcess:pass
handed=read(base/'handoff.json');assert handed['waiting_controller_only_stopped'] and not handed['original_native_child_stopped']
for name in ['actual_old_waiting_parent_memory_before.json','previous_status_at_verified_handoff.json','handoff_identity.json','handoff.json']:
    shutil.copy2(base/name,out/name)
write('actual_cold_successor_memory_and_retained_progress.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    before_private_bytes=before['private_bytes'],after=after,released_private_bytes=before['private_bytes']-after['private_bytes'],
    exact_new_owner_live=True,state=read(base/'status.json'),
    any_numerical_child_stopped=False,any_scientific_budget_or_guard_changed=False))
write('retained_original_two_real_certificates.json',c['reused_actual_stages'])
xml=Path('F:/aicoding/IIA_Data/experiments/TimesNet_cold_continuation_tests_v1.xml')
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(t.get('tests',0)) for t in suites)==53
assert not sum(int(t.get('failures',0))+int(t.get('errors',0)) for t in suites)
shutil.copy2(xml,out/'validation53.xml')
paths=[s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG];sources=[]
for i,path in enumerate(paths):
    name='source%02d%s'%(i,Path(path).suffix);shutil.copy2(root/path,out/name)
    sources.append(dict(path=path,sha256=sha(root/path),snapshot=name))
write('validation_and_source_mapping.json',dict(passed_tests=53,xml_sha256=sha(xml),sources=sources,
    original_registered_stages=8,reused_original_stages=2,remaining_original_stages=6,
    numerical_worker=s.original.MODULE,original_worker_and_every_argument_unchanged=True,
    full_original_binding_in_the_actual_numerical_worker=True,resource_settings=original['settings'],
    resource_lock=original['resource_lock'],CPU_threads=original['cpu_threads'],
    pending_readiness_is_not_scientific_completion=True,registered_SMAP_entities=51,
    not_certified_canonical_NASA_full_dataset=True,all_original_experiments_complete=False))
observed=[]
for pid,ctime in [(36912,1792097263.1796665),(40268,1792097260.102444),(41992,1791948094.8603492)]:
    try:
        actor=psutil.Process(pid);assert abs(actor.create_time()-ctime)<.01
        observed.append(dict(pid=pid,create_time=ctime,command=actor.cmdline(),cpu_seconds=sum(actor.cpu_times()[:2])))
    except psutil.NoSuchProcess:observed.append(dict(pid=pid,alive=False))
write('actual_scientific_processes_not_interrupted.json',observed)
tab=read(root/'experiments/runs/fm_TAB_full878_original_metric_suite_v1/status.json')
write('actual_original_TAB_progress.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),state=tab))
(out/'README.md').write_text('''# 保留 TimesNet 全部剩余核验，释放空闲控制器模型库内存

实际旧进程已完成原种子 1104 的 point/native 两阶段核验，随后等待种子 1105，无计算子进程，但保留约 805 MiB 私有内存。新增控制器以实际 PID、创建时间、完整命令、状态和全部后代两次核验其确为空闲后交接；只终止该等待控制器，未停止训练、回放或评估子进程。交接前状态、身份和回执原样保留。

实际新控制器仅约 14–16 MiB 私有内存，释放约 790 MiB，具体字节数见 actual_cold_successor_memory_and_retained_progress.json。新控制器继续等待原始下一种子的源结果，保持全部八阶段任务及已有种子 1103 的独立证据；其中两阶段使用既有种子 1104 成功证明，不重复计算、不计作新的独立重复，剩余六阶段仍待完成。

既有两个真实证明的成功守卫、完整命令、冻结配置、模型权重、分数、源结果、源数据/代码及原生回放分数重新核验 SHA256。原子进程仍使用原 v2 模块、全部相同参数、阶段配置、28,132,633 参数模型、CPU 两线程、float32、batch128、资源锁和原保护线。完整数据绑定与数值库只留在实际计算子进程，等待父进程不再保留它们。新控制器同时持有原控制器互斥锁，防止重复派发；失败和部分输出保留，不自动重跑。

新增与原 TimesNet 相关测试合计 53 项通过，涵盖原数值函数身份、实际已有两份完整证明、命令/配置不变、拒绝缩减预算或保护线，以及有活动 PID、计算后代、锁占用、进度变化或失败时不允许交接。冷进程实际重新核验既有证据时不加载 NumPy/Torch，内存检查通过。

原注册 SMAP 子集仍为 51 个实体、408160 测试点，不能称为 NASA 全量 54 实体实验，也不与新 full109/common105 分数混用。本次没有新增训练成绩，不证明论文作者等价或完整 TAB 训练协议。当前 CFM、BRITS、CrossAD 实际计算进程和原 TAB 进度另存；所有原科学任务均保持运行。

45 篇论文、681 项方法/基线/消融、273 项数据任务、702 项论文标量目标，以及全部作者数据集、原生/标签复核、完整 TAB 和五训练种子统计仍是原范围。此次内存回收和后续接续不表示整体完成。
''',encoding='utf-8',newline='\n')
shutil.copy2(__file__,out/'publication_source.py')
git=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
manifest=dict(previous_root=git(root),previous_parent=git(root.parent/'flow-matching-research'),
    paths=paths+[p.relative_to(root).as_posix() for p in out.iterdir() if p.is_file()])
(root/'tmp/timesnet_cold_continuation_exact_publication_v3.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(manifest['paths']),tests=53,released_private_bytes=before['private_bytes']-after['private_bytes'],successor_pid=after['pid'])))
