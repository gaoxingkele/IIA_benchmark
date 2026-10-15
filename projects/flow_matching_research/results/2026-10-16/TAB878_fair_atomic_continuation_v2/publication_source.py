from pathlib import Path
from datetime import datetime,timezone
import json,shutil,subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import continue_TAB878_fair_entities_v2 as runner

s=runner.previous;root=Path.cwd();read=s.read;sha=s.sha
out=root/'projects/flow_matching_research/results/2026-10-16/TAB878_fair_atomic_continuation_v2'
out.mkdir(parents=True,exist_ok=False)
def write(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
config=read(root/runner.CONFIG);c=runner.verify(config);base=root/config['state_root'];h=read(base/'launch.json');p=psutil.Process(h['pid'])
assert p.create_time()==h['create_time'] and p.cmdline()==h['command']
handed=(base/'handoff.json').exists()
def observed(handle):
    try:
        p=psutil.Process(handle['pid']);return dict(expected=handle,alive=p.is_running(),exact_create_time=p.create_time()==handle['create_time'],
            actual_command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2]))
    except psutil.NoSuchProcess:return dict(expected=handle,alive=False)
source_state=read(root/c['state_root']/'status.json')
write('actual_live_successor_original_owner_and_scope.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    successor=observed(h),successor_state=read(base/'status.json'),handoff_completed=handed,
    original_owner=observed(config['previous_owner']),original_scope_status=source_state,
    stderr_bytes=(base/'controller.stderr.log').stat().st_size))
heartbeat=read(base/'actual_atomic_heartbeat_verified.json');assert heartbeat['heartbeat_updated'] and heartbeat['stderr_bytes']==0
shutil.copy2(base/'actual_atomic_heartbeat_verified.json',out/'actual_atomic_heartbeat_verified.json')
if handed:
    for name in ['handoff.json','handoff_identity.json','previous_status_at_handoff.json','previous_status_original_bytes.json']:
        shutil.copy2(base/name,out/name)
    assert source_state['pid']==h['pid']
else:
    old=psutil.Process(config['previous_owner']['pid'])
    assert old.create_time()==config['previous_owner']['create_time'] and old.cmdline()==config['previous_owner']['command']
failed=root/config['prior_metadata_failure']['state_root']
for name in ['launch.json','controller.stderr.log','status.json','original_live_owner_before_passive_wait.json']:
    shutil.copy2(failed/name,out/('preserved_v1_'+name))
assert not (failed/'handoff.json').exists()
write('preserved_v1_failure_classification.json',dict(
    failed_handle=config['prior_metadata_failure']['handle'],exact_failed_handle_now=observed(config['prior_metadata_failure']['handle']),
    cause='Original TAB module exports an exclusive new-file writer. The new v1 scheduler mistakenly reused it for mutable status, so second heartbeat raised FileExistsError.',
    failure_before_any_handoff=True,original_numerical_process_or_metric_changed=False,
    correction='v2 retains entire v1 controller code object in private globals and binds only its status writer to existing atomic replacement. Original metric stage functions remain v1/original.',
    scientific_retry_executed=False))
basef=Path('F:/aicoding/IIA_Data/experiments');xmls=[basef/'TAB878_fair_continuation_tests_v3.xml',basef/'TAB878_fair_atomic_tests_v1.xml']
ids=set();validation=[]
for index,path in enumerate(xmls):
    tree=ET.parse(path);suites=list(tree.getroot().iter('testsuite'))
    tests=sum(int(v.get('tests',0)) for v in suites)
    assert tests==[43,28][index] and not sum(int(v.get('failures',0))+int(v.get('errors',0)) for v in suites)
    ids.update((t.get('classname'),t.get('name')) for t in tree.getroot().iter('testcase'))
    name=['original43_validation.xml','atomic28_validation.xml'][index];shutil.copy2(path,out/name)
    validation.append(dict(path=str(path),sha256=sha(path),tests=tests,snapshot=name))
assert len(ids)==50
paths=[s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG,runner.MODULE.replace('.','/')+'.py',runner.TEST,runner.CONFIG]
sources=[]
for i,path in enumerate(paths):
    name='source%02d%s'%(i,Path(path).suffix);shutil.copy2(root/path,out/name)
    sources.append(dict(path=path,sha256=sha(root/path),snapshot=name))
write('validation_and_frozen_metric_contract.json',dict(distinct_passed_tests=50,validation=validation,sources=sources,
    required_cases=878,required_label_views=2057,original_scientific_configuration=s.original.CONFIG,
    original_scientific_configuration_sha256=sha(root/s.original.CONFIG),original_metric_worker=s.original.MODULE,
    exact_original_worker_cli_and_stage_configuration_unchanged=True,
    settings=c['settings'],resource_lock=c['resource_lock'],CPU_threads=c['CPU_threads'],
    score_metrics=c['score_metrics'],label_metrics=c['label_metrics'],metric_calls_per_entity_label_view=201,
    strict_ratios=c['strict_ratios'],diagnostic_ratios=c['diagnostic_pooled_ratios'],entity_macro_bootstrap_resamples=200,
    full_original_VUS_integer_windows_and250thresholds_unchanged=True,inter_entity_yield_seconds=20,
    all_old_success_partial_and_failed_fragments_preserved=True,automatic_retry=False,
    handoff_completed_at_capture=handed,all_original_experiments_complete=False))
review=root/'experiments/runs/fm_saits_completion_replay_queue_v1/children/saits_paper_brits_electricity_author_missing0.5_seed29/review/status.json'
write('actual_same_lock_waiting_review.json',dict(handle=observed(dict(pid=31092,create_time=1792097303.5664525)),state=read(review),
    same_original_resource_lock=c['resource_lock'],not_proof_of_indefinite_starvation=True))
if source_state.get('active_pid'):
    try:
        child=psutil.Process(source_state['active_pid'])
        write('actual_original_TAB_child_not_interrupted.json',dict(pid=child.pid,create_time=child.create_time(),command=child.cmdline(),cpu_seconds=sum(child.cpu_times()[:2])))
    except psutil.NoSuchProcess:write('actual_original_TAB_child_not_interrupted.json',dict(pid=source_state['active_pid'],already_exited_recheck_owner=True))
(out/'README.md').write_text('''# 原 TAB878 队列逐实体调度接续及原子状态修正

实际 SAITS 种子 29 独立复核进程存活并等待与 TAB 相同的审计锁，内存达到其原门槛。原 TAB 完成实体后立即申请下一实体，没有主动间隔；这不等于证明无限饥饿。新调度在每个新完成实体释放锁后让出 20 秒，给其他原等待者轮询机会。

保留原完整 878 项、2057 组标签、6 个分数指标和 15 个标签指标、每组实体标签 201 次调用、3 个验证阈值及 10 个测试诊断阈值、完整 VUS 整数窗口和 250 阈值、200 次条件实体 bootstrap。原工作进程、解释器、完整命令、阶段配置、科学输出目录及原 16/28/16 GiB 门槛、3/6 GiB 上限、CPU 两线程、审计锁不变。

控制器被动等待当前实体结束，只有取得真实审计锁，检查精确终止守卫和实际子进程身份，且原元数据父进程没有计算后代后，才暂停并再次检查父进程。状态变化会恢复父进程并放弃交接；不停止计算子进程。成功后接管原控制器互斥锁，保存原状态字节快照，并为既有资源中止恢复观察者镜像新的状态。

所有旧成功实体和汇总先核验精确守卫、完整指标覆盖和同一源谱系后复用，不算新独立重复。旧失败和部分输出不自动重跑；原资源中止恢复观察者继续使用自己的独立尝试和保护线。

v1 实际首次启动暴露了状态写入错误：错误复用了原模块只允许新建文件的函数，第二次状态更新报 FileExistsError，续接进程在任何交接之前退出。原父进程及数值子进程保持运行。失败脚本、配置、状态、命令和错误日志保留。v2 完整复用相同控制器字节码，仅在私有环境把动态状态更新绑定到现有原子替换函数；原科学阶段和指标函数不改变，也未重试科学实验。

验证覆盖 50 个不同用例：原指标和调度 43 项通过，另一次调度/状态回归 28 项通过，其中 21 项重复。新增回归在独立测试目录运行完整控制器的实际状态更新器，复现旧写法失败、验证新写法连续更新；不暂停真实进程，不创建生产科学守卫。实际 v2 进程的连续心跳也已核验，错误日志为零。

是否已接管见 actual_live_successor_original_owner_and_scope.json 的 handoff_completed。若为 false，修正版仍等待当前实体的真实安全边界，不能称已改善 SAITS 实际进度或完成新实体。原 TAB 计算继续运行，等待并不是超时失败，不会因此重启它。

原 45 篇论文、681 项方法/基线/消融、273 项数据任务、702 项论文标量目标及全部作者数据集、原生/标签复核、完整 TAB、五训练种子统计继续保留。此处接续和回归验证不表示整体完成或作者代码等价。
''',encoding='utf-8',newline='\n')
shutil.copy2(__file__,out/'publication_source.py')
git=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
manifest=dict(previous_root=git(root),previous_parent=git(root.parent/'flow-matching-research'),
    paths=paths+[p.relative_to(root).as_posix() for p in out.iterdir() if p.is_file()])
(root/'tmp/TAB878_fair_atomic_exact_publication_v2.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(manifest['paths']),unique_tests=50,successor_pid=h['pid'],handoff_completed=handed)))
