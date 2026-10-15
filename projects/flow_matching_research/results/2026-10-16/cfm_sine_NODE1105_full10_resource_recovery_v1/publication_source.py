from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import run_cfm_sine_NODE1105_resource_recovery_v1 as s

root=Path.cwd();out=root/'projects/flow_matching_research/results/2026-10-16/cfm_sine_NODE1105_full10_resource_recovery_v1'
out.mkdir(parents=True,exist_ok=False)
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
c=read(root/s.CONFIG);jobs,q=s.verify(c);replay=s.replay_namespace()['verify'](read(root/s.REPLAY))
base=Path('F:/aicoding/IIA_Data/experiments');xml=base/'CFM_sineN10_recovery_tests_v3.xml'
suite=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(t.get('tests',0)) for t in suite)==13
assert not sum(int(t.get('failures',0))+int(t.get('errors',0)) for t in suite)
paths=[s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG,s.QUEUE,s.REPLAY]
snapshots=[]
for index,path in enumerate(paths):
    name='source%02d%s'%(index,Path(path).suffix);shutil.copy2(root/path,out/name)
    snapshots.append(dict(path=path,sha256=sha(root/path),snapshot=name))
shutil.copy2(xml,out/'validation13.xml')
handles=[]
for label in ['source','replay']:
    h=read(root/c['state_root']/('launch_'+label+'.json'));p=psutil.Process(h['pid'])
    assert abs(p.create_time()-h['create_time'])<.01 and p.cmdline()==h['command']
    statepath=root/(c['state_root'] if label=='source' else c['state_root']+'/replay')/'status.json'
    handles.append(dict(role=label,handle=h,actual_command=p.cmdline(),alive=p.is_running(),state=read(statepath)))
write('actual_two_live_controllers_and_pending_state.json',handles)
write('original_terminal_resource_abort_and_unsaved_progress.json',dict(
    guard=read(root/c['preserved_failed_receipt']),progress=read(root/c['preserved_failed_progress']),
    exact_old_handle=s.FAILED_HANDLE,partial_epochs_credited=0,original_job=c['canonical_job']))
failed=[];source=read(root/s.ORIGINAL)
for job in source['jobs']:
    path=root/source['state_root']/'receipts'/(job['id']+'.json')
    if path.exists():
        g=read(path)
        if g.get('resource_observations',{}).get('resource_guard_aborted'):
            failed.append(dict(canonical_id=job['id'],original_guard=path.relative_to(root).as_posix(),
                original_guard_sha256=sha(path),original_guard_complete_flag=g.get('complete'),exit_code=g['exit_code'],candidates=[]))
byid={row['canonical_id']:row for row in failed}
for path in sorted((root/'configs/experiments').glob('fm_cfm*.json')):
    data=read(path);chosen=data.get('recovery_job_ids',[])
    for job in data.get('jobs',[]):
        canonical=job.get('recovery_original_job_id',job.get('id'))
        if canonical not in byid or job['id']==canonical or chosen and job['id'] not in chosen:continue
        byid[canonical]['candidates'].append(dict(queue=path.relative_to(root).as_posix(),queue_sha256=sha(path),
            execution_id=job['id'],result_exists=(root/job['output_directory']/'result.json').exists(),
            registration_is_not_completed_or_live_proof=True))
write('all19_original_resource_abort_registration_coverage_not_completion.json',dict(
    captured_utc=datetime.now(timezone.utc).isoformat(),rows=failed,
    candidate_presence_is_not_coverage_or_scientific_completion=True,
    result_present_without_distinct_recovery_requires_existing_checkpoint_and_guard_audit=True))
fixture=[]
for p in (base/'CFM_sineN10_recovery_tests_v3').rglob('completed_full_epoch_checkpoints.json'):
    rows=read(p);assert [r['completed_original_epochs'] for r in rows]==list(range(1,11))
    fixture.append(dict(path=str(p),sha256=sha(p),complete_epochs=10,synthetic_only=True,
        records=[dict(epoch=r['completed_original_epochs'],optimizer_updates=r['optimizer_updates'],
            full_state_sha256=r['full_state_sha256']) for r in rows]))
assert len(fixture)==1
write('synthetic_full_capacity10epochs_checkpoint_probe_not_benchmark.json',fixture)
write('validation_and_preserved_environment_failures.json',dict(passed_tests=13,xml_sha256=sha(xml),
    frozen_source_environment=q['python'],CPU_environment=c['CPU_only_environment'],
    prior_attempts=[dict(path=str(base/('CFM_sineN10_recovery_tests_v%d.xml'%i)),sha256=sha(base/('CFM_sineN10_recovery_tests_v%d.xml'%i)),
        failed=1,passed=12,reason=reason) for i,reason in [(1,'System Python lacks torchdiffeq; use the existing frozen CFM environment without installing dependencies.'),
        (2,'Visible CUDA context in CPU fixture. Explicit CPU-only environment retained for new CPU worker; no solver,model or numerical tolerance changed.')]],
    source_snapshots=snapshots,source_science_identical=True,original285_slots=285,selected_new_recovery=1,
    original_epochs=10,original_optimizer_updates=20,test_trajectories=80,test_points_per_trajectory=500,
    commit_admission_bytes=32*2**30,stable_samples=6,poll_seconds=5,emergency_bytes=16*2**30,
    maximum_new_attempts=1,partial_epochs_credited=0,all_original_experiments_complete=False,
    native_replay_routes=len(replay[0]),already_verified_primary_records=len(replay[2])))
(out/'README.md').write_text('''# CFM appendix-d / sine / NODE / 种子 1105 的独立恢复

原始任务在第 1/10 轮、2/20 次更新后，因系统提交余量低于 16 GiB 紧急线中止；其实际子进程已终止，未保存模型或优化器状态。保留原始失败回执及进度，不给未保存训练记完成量。另一个等待中的种子 1105 对应 main-text / pendulum / 400 轮，不能覆盖此缺口。

新登记一次完整原预算尝试：10 轮、320 条训练轨迹、batch 256、20 次更新、80 条测试轨迹和每条 500 个点。原数据、随机种子、258 宽三层模型、dopri5、1e-7/1e-9 容差及其余科学字段不变。队列仍保留全部 285 项，仅该项的执行身份与输出路径改变。原输出及原始数据不覆盖。

复用已验证的完整 epoch 状态保存实现，每轮保存模型、Adam、梯度、所有随机状态、训练顺序身份、损失和计数，保存后逐值核验，不增加优化器更新。新 CPU 工作进程显式屏蔽 CUDA，沿用原 CPU 算法。原家族锁、CPU 单线程、12 GiB 物理内存及 16 GiB 紧急线保留；提交余量准入加强至 32 GiB，并要求连续六次、间隔五秒的稳定观测。只有一次新尝试，失败后保留，不自动重复。

已连接原完整检查点回放与独立逐坐标复核。仍保留全部 285 个路由，其他 284 项不改变；沿用原回放解释器、两线程、资源上限、锁及预测容差。实际新源运行成功守卫闭合后才进入回放，已有主队列证据按原严格检查引用，不重复训练。

13 项测试通过。合成测试采用相同模型容量、10 轮及原求解器，两条很短的轨迹只验证每轮状态保存与未加保存时权重、损失、NFE、更新数和预测完全一致，不能作为原 320/80 条轨迹的数据集结果。首次系统 Python 缺少 torchdiffeq，改用现有冻结环境；随后 CPU 测试检测到可见 CUDA 上下文，补齐新 CPU 进程环境隔离。两次失败 XML 均在 F 盘保留并记录 SHA256，没有修改原模型、容差或已运行任务。

归档时两个新控制器实际存活，源执行仍等待原资源条件，尚未取得新的真实训练完成结果。另附原 285 队列中全部 19 个带资源中止标记的记录及已有恢复登记候选；候选存在不代表进程存活、成功守卫或实验闭合，带 complete 标志但失败的原守卫也不被直接当作成功。

45 篇论文、681 项方法/基线/消融、273 项数据任务、702 项论文标量目标及其他所有作者队列、原生/标签复核、完整 TAB 和五种子统计继续保留。CFM 当前实现仍是声明了差异的论文推导重构，不能声称与作者代码等价。此次登记不表示整体完成。
''',encoding='utf-8',newline='\n')
shutil.copy2(__file__,out/'publication_source.py')
git=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
manifest=dict(previous_root=git(root),previous_parent=git(root.parent/'flow-matching-research'),
    paths=paths+[p.relative_to(root).as_posix() for p in out.iterdir() if p.is_file()])
(root/'tmp/cfm_sineN10_recovery_exact_publication_v1.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(manifest['paths']),tests=13,source_closed=s.source_closed(c),live_controllers=2)))
