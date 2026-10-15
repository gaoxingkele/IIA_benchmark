from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import run_cfm_two_retries_cached_idle_v1 as s

root=Path.cwd();out=root/'projects/flow_matching_research/results/2026-10-16/CFM_two_idle_retries_cached_v1'
out.mkdir(parents=True,exist_ok=False)
sha=s.sha;read=s.read
def write(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
c=read(root/s.CONFIG);base=root/c['handoff_state_root'];before=read(base/'actual_two_old_idle_owners_before.json');after=[]
shutil.copy2(base/'actual_two_old_idle_owners_before.json',out/'actual_two_old_idle_owners_before.json')
for name,role in c['roles'].items():
    s.verify(c,name);h=read(base/('launch_'+name+'.json'));p=psutil.Process(h['pid'])
    assert p.create_time()==h['create_time'] and p.cmdline()==h['command']
    state=read(root/role['previous_state_path']);assert state['pid']==p.pid and state['active_pid'] is None
    assert state['state']=='waiting_for_memory'
    try:assert psutil.Process(role['previous_owner']['pid']).create_time()!=role['previous_owner']['create_time']
    except psutil.NoSuchProcess:pass
    handed=read(base/name/'handoff.json');assert handed['waiting_controller_only_stopped'] and not handed['original_native_child_stopped']
    for file in ['handoff.json','handoff_identity.json','previous_status_at_verified_handoff.json','successor_entry.json']:
        shutil.copy2(base/name/file,out/(name+'_'+file))
    after.append(dict(role=name,handle=h,actual_command=p.cmdline(),private_bytes=p.memory_info().private,
        state=state,legacy_state_timestamp_not_invented='create_time' not in state,stderr_bytes=(base/(name+'.stderr.log')).stat().st_size))
write('actual_two_successors_after.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),owners=after,
    old_private_bytes=sum(v['private_bytes'] for v in before),new_private_bytes=sum(v['private_bytes'] for v in after),
    released_private_bytes=sum(v['private_bytes'] for v in before)-sum(v['private_bytes'] for v in after)))
xml=Path('F:/aicoding/IIA_Data/experiments/CFM_two_cached_idle_tests_v1.xml')
suite=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(v.get('tests',0)) for v in suite)==26
assert not sum(int(v.get('failures',0))+int(v.get('errors',0)) for v in suite)
shutil.copy2(xml,out/'validation26.xml')
paths=[s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG];mapping=[]
for i,path in enumerate(paths):
    name='source%02d%s'%(i,Path(path).suffix);shutil.copy2(root/path,out/name)
    mapping.append(dict(path=path,sha256=sha(root/path),snapshot=name))
write('validation_and_frozen_mapping.json',dict(passed_tests=26,xml_sha256=sha(xml),source_mapping=mapping,
    original285slots_retained=True,selected_existing_recovery_hosts=2,original_epochs=[400,600],
    original_entry_and_all_arguments_unchanged=True,actual_cached_guard_probes=2,
    original_policies=[v['resource_policy'] for v in c['roles'].values()],
    original40GiB_admission16GiB_emergency12GiB_physical_CPU1_retained=True,
    preserved99_and41epoch_failed_attempts_receive_no_fresh_training_credit=True,
    raw_data_checkpoint_or_source_file_changed=False,all_original_experiments_complete=False))

# A resource-failed original receipt must not cause blind retraining when a
# final model and an existing native record already exist. This is a current
# byte audit of those saved records, not a new solver/model inference.
primary=read(root/'configs/runtime/fm_cfm_completion_replay_queue.v4.json')
primary_records=read(root/primary['state_root']/'verified_full_model_records.json')['records']
canonical='cfm_ts_original__main_text__gp_gaussian_conditional_e600__pendulum__seed1106'
record=next(r for r in primary_records if r['original_canonical_id']==canonical)
checked=[]
for path,key in [(record['checkpoint_path'],'checkpoint_sha256'),(root/record['result_path'],'result_sha256'),
    (record['replay_arrays'],'replay_arrays_sha256')]:
    digest=sha(path);assert digest==record[key];checked.append(dict(path=str(path),sha256=digest,record_field=key))
source=read(root/record['result_path'])
for artifact in source['artifacts']:
    digest=sha(artifact['path']);assert digest==artifact['sha256'];checked.append(dict(path=artifact['path'],sha256=digest,kind='source_artifact'))
guardpath=root/'experiments/runs/fm_cfm_ts_original_v1/receipts'/(canonical+'.json');guard=read(guardpath)
assert guard['exit_code']!=0 and guard['resource_observations']['resource_guard_aborted']
write('GP_saved_model_existing_native_record_and_negative_source_guard_preserved.json',dict(
    canonical_id=canonical,existing_primary_record=record,source_guard=guard,source_guard_path=str(guardpath),source_guard_sha256=sha(guardpath),
    all_referenced_current_saved_artifact_bytes_match=checked,
    new_solver_reinference_performed=False,source_guard_failure_not_overwritten_or_relabelled=True,
    new_training_scheduled=False,reason='Final model/full prediction arrays and existing native record are present. Preserve negative source guard and audit the saved chain rather than repeat600epochs.',
    primary_record_count=len(primary_records),full_research_complete=False))
observations=[]
for pid,ctime in [(36912,1792097263.1796665),(40268,1792097260.102444),(41992,1791948094.8603492)]:
    try:
        p=psutil.Process(pid);assert p.create_time()==ctime
        observations.append(dict(pid=pid,create_time=ctime,command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2]),alive=p.is_running()))
    except psutil.NoSuchProcess:observations.append(dict(pid=pid,alive=False))
write('actual_current_scientific_processes_not_interrupted.json',observations)
write('actual_TAB_progress.json',read(root/'experiments/runs/fm_TAB_full878_original_metric_suite_v1/status.json'))
(out/'README.md').write_text('''# 两个已有 CFM NODE 恢复控制器的空闲内存回收

原 NODE400/pendulum/1104 与 NODE600/pendulum/1107 两个独立恢复队列仍等待资源，没有科学输出、原生子进程或守卫回执。交接两次核对其实际 PID、创建时间、完整命令、规范任务、队列和运行时哈希、空任务进度及全部后代后，只停止等待父进程；旧状态与交接回执全部保留。

新控制器重新进入完整相同入口与每个参数，只把 Windows 资源快照绑定到已有的等价缓存实现。原完整 285 槽位、选择的 400/600 轮、模型/求解器/种子/数据、CPU 单线程、12 GiB 物理内存、40 GiB 准入、16 GiB 紧急线、家族锁和科学进程命令均不改变。原中止尝试的 99/41 轮未保存进度没有获得新完成量。原状态没有创建时间字段，此处没有补造字段；实际 OS 身份仍在交接前后严格核验。

实际两进程的私有内存从合计约 364 MiB 降到约 31 MiB，释放约 333 MiB，准确字节数见 actual_two_successors_after.json。新进程仍等待原资源条件，日志没有报错。相关 26 项验证通过，含两个原始资源 API 的独立冷探针，不加载 NumPy/Torch、不启动计算；所有原保护线、入口参数及完整槽位验证保留。

同时核对一个原守卫失败但已有最终模型的 GP600/pendulum/1106 项：已有原生回放记录及当前权重、结果、源预测和回放数组的字节哈希均相符。负面原守卫仍保留，没有改成成功，也没有据此盲目再训练 600 轮。此次只是已保存证据的字节核验，没有新求解器推理，不构成新的独立实验或整体验收。

原实际 CFM400/1106、BRITS 和 CrossAD1104 计算进程继续存活，另存身份与 CPU 观测。原 TAB 评估继续执行。全部 45 篇论文、681 项方法/基线/消融、273 项数据任务、702 项论文标量目标及原数据集/作者队列/全量复核/完整 TAB/五种子统计继续保留；本次回收内存和引用既有记录不能表示整体完成。
''',encoding='utf-8',newline='\n')
shutil.copy2(__file__,out/'publication_source.py')
git=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
manifest=dict(previous_root=git(root),previous_parent=git(root.parent/'flow-matching-research'),
    paths=paths+[p.relative_to(root).as_posix() for p in out.iterdir() if p.is_file()])
(root/'tmp/cfm_two_cached_idle_exact_publication_v1.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(manifest['paths']),tests=26,live_successors=2,released_private_bytes=sum(v['private_bytes'] for v in before)-sum(v['private_bytes'] for v in after))))
