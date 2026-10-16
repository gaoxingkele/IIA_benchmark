from pathlib import Path
from datetime import datetime,timezone
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import continue_spectral70_fair_conditions_v1 as s
from scripts.flow_matching.timegan_native_common_v1 import write
ROOT,read,sha=s.ROOT,s.read,s.sha
c=read(ROOT/s.CONFIG);original,parent,cases=s.verify(c);base=ROOT/c['state_root']
h=read(base/'launch.json');p=psutil.Process(h['pid'])
assert p.is_running() and p.create_time()==h['create_time'] and p.cmdline()==h['command']
assert not (base/'controller.stderr.log').read_bytes()
assert not (base/'handoff.json').exists()
tests=Path(h['validation_path']);assert sha(tests)==h['validation_sha256']
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==11
assert not any(int(x.get(k,0)) for x in suites for k in ['failures','errors','skipped'])
old=psutil.Process(c['previous_owner']['pid'])
assert old.create_time()==c['previous_owner']['create_time'] and old.cmdline()==c['previous_owner']['command']
state=read(ROOT/original['state_root']/'status.json')
assert state['active_condition']=='spectral_original__diffusion_ts__energy__released_author_data__seed42'
handle=state['exact_child_handle'];child=psutil.Process(handle['pid'])
assert child.create_time()==handle['create_time'] and child.cmdline()==handle['command']
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/Spectral70_fair_condition_continuation_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.xml -text -whitespace\n*.py -text\n',encoding='utf-8')
for name in ['launch.json','status.json']:shutil.copyfile(base/name,packet/('actual_scheduler_'+name))
write(packet/'actual_original_condition_state.json',state)
write(packet/'actual_running_numeric_and_metadata_handles.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    original_parent=dict(pid=old.pid,create_time=old.create_time(),command=old.cmdline()),
    original_numeric_launcher=dict(pid=child.pid,create_time=child.create_time(),command=child.cmdline()),
    original_numeric_descendants=[dict(pid=v.pid,create_time=v.create_time(),command=v.cmdline()) for v in child.children(recursive=True) if v.name().lower()=='python.exe'],
    waiting_successor=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()),numerical_worker_stopped=False))
shutil.copyfile(tests,packet/'validation_11_cases.xml')
write(packet/'unchanged_full70_numerical_program_binding.json',dict(original_configuration=s.original.CONFIG,
    original_configuration_sha256=sha(ROOT/s.original.CONFIG),worker_module=s.original.MODULE,
    required_conditions=70,required_cohorts=14,original_settings=original['settings'],resource_lock=original['resource_lock'],
    spectral_flow_minimum_commit_headroom_bytes=original['spectral_flow_minimum_commit_headroom_bytes'],
    source_receipts=original['source_receipts'],numeric_worker_source_or_configuration_changed=False))
write(packet/'iteration_status.json',dict(original_scope=c['original_scope'],all70_conditions_retained=True,
    metadata_observer_live=True,actual_handoff_completed=False,tests_passed=11,new_complete_numerical_conditions=0,
    new_model_training_updates=0,numerical_workers_stopped=0,all_original_experiments_complete=False))
(packet/'README.md').write_text('''# Spectral全70条件：完整条件之间让出共享GPU锁

完整生成重跑与原始GRASP/生成训练使用同一家庭GPU锁。原重跑父进程会在一个条件结束后立即争用下一条件，故新增独立元数据调度器，在每个新完整条件结束后释放锁并等待35秒，与既有GRASP公平间隔一致。
保留全部70条件、14个五训练种子组、每个模型两次完整原采样及四类各5次完整原评估。求解器、采样数量/批次、所有字段、RNG与模型/EMA/优化器状态检查、线程、精度、16GiB GPU准入、48GiB Spectral Flow提交内存准入和所有原始守卫保持。
继任观察器已按实际PID/创建时间/完整命令确认存活。当前原Energy seed42数值进程仍在运行，本包归档时尚未接管，没有停止数值子进程。

接管必须先取得实际共享GPU锁，两次检查旧父进程身份，确认没有数值后代、精确子进程有终止守卫，并确认所有已存在尝试都已被原状态记录为成功或保留失败。任何未记录的部分尝试都阻止接管。
接管后，全部成功报告/守卫/生成与RNG文件/原始源产物证明重新校验；成功和失败槽均保留，不重试。旧状态原始字节保存，恢复时仅调整元数据目录存在性、原锁继承、已完成状态恢复和条件间等待；数值子进程仍执行原模块、原配置和原命令，完整科学输出目录与原守卫路径保持。

11项相关检查通过：原始采样/RNG/全部评估预算/完整五种子条件、拒绝活动数值子进程或部分尝试、精确身份、原失败保留及数值启动/守卫AST一致性。本次观察器启动不代表接管完成，也不计新增训练或完整实验。
全部45论文/681方法记录/273数据任务/702指标、GRASP9120条件、原论文/TAB/新数据实验仍保留，整体目标未完成。
''',encoding='utf-8')
shutil.copyfile('tmp/launch_Spectral70_fair_20261016_v1.py',packet/'launch_source.py');shutil.copyfile(__file__,packet/'closure_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=v.relative_to(packet).as_posix(),sha256=sha(v)) for v in sorted(packet.rglob('*')) if v.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/Spectral70_fair_condition_continuation.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),required_conditions=70,
    tests_passed=11,actual_handoff_completed=False,all_original_experiments_complete=False))
paths=[s.CONFIG,s.MODULE.replace('.','/')+'.py',s.TEST,ara.relative_to(ROOT).as_posix()]
paths += [v.relative_to(ROOT).as_posix() for v in packet.rglob('*') if v.is_file()]
def head(p):return subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD']).decode().strip()
write(ROOT/'tmp/Spectral70_fair_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),actual_metadata_observer=p.pid,handoff_completed=False)))
