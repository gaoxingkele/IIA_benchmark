from pathlib import Path
from datetime import datetime,timezone
import json
import shutil
import subprocess
import psutil
from scripts.flow_matching import continue_common105_TAB577_fair_entities_v1 as s
from scripts.flow_matching.timegan_native_common_v1 import write
ROOT,read,sha=s.ROOT,s.read,s.sha
c=read(ROOT/s.CONFIG);original=s.verify(c);base=ROOT/c['state_root']
handoff=read(base/'handoff.json');identity=read(base/'handoff_identity.json')
old=read(base/'previous_status_at_handoff.json');raw=(base/'previous_status_original_bytes.json').read_bytes()
assert json.loads(raw)==old and old['completed_entities']=={'e25abf920638483064c8':11}
assert handoff['previous_owner']==identity['previous_owner']==c['previous_owner']
assert handoff['waiting_metadata_only_stopped'] and not handoff['original_numeric_child_stopped']
assert identity['actual_audit_lock_owned'] and identity['exact_os_identity_checked_twice'] and identity['actual_last_child_terminal']
assert not identity['numerical_child_stopped']
assert all(s.fair.allowed_console_child(x['name'],x['exe']) for x in identity['descendants'])
try:
    prior=psutil.Process(c['previous_owner']['pid'])
    assert prior.create_time()!=c['previous_owner']['create_time']
except psutil.NoSuchProcess:pass
launch=read(base/'launch.json');actor=psutil.Process(launch['pid'])
assert actor.is_running() and actor.create_time()==launch['create_time'] and actor.cmdline()==launch['command']
assert actor.pid==handoff['successor_pid']
assert not (base/'controller.stderr.log').read_bytes()
state=read(base/'status.json');assert not state['failed_cases_preserved']
assert state['required_cases']==577 and state['required_label_views']==1726
entry=next(x for x in original['entries'] if x['key']=='e25abf920638483064c8')
bound=s.matched.ready_case(original,entry);assert bound is not None
ns=s.namespace();records=[]
for entity in bound['dataset']['entities'][:11]:
    prefix=entity['array_prefix'];assert ns['fragment_action'](original,entry,prefix,bound)=='reuse_guarded'
    attempt,output=s.matched.original.paths(original,entry,prefix)
    report,proof=s.matched.checked_fragment(original,entry,prefix)
    guard=attempt/'guard_receipt.json';g=read(guard)
    assert g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted']
    assert g['exact_child_handle']['command']==s.matched.worker_command(original,entry,prefix)
    assert proof['source_lineage']==bound['lineage'] and proof['entity']==entity['entity_id']
    if prefix=='e010':
        h=g['exact_child_handle'];assert h['pid']==8532 and h['create_time']==1792140307.078184
        try:
            child=psutil.Process(h['pid']);assert child.create_time()!=h['create_time']
        except psutil.NoSuchProcess:pass
    records.append(dict(entity=entity['entity_id'],prefix=prefix,report_path=str(report),report_sha256=sha(report),
        guard_path=str(guard),guard_sha256=sha(guard),full_original_fragment_and_cell_journal_revalidated=True))
assert len(records)==11 and state['completed_entities'][entry['key']]>=11
prior_packet=ROOT/'projects/flow_matching_research/results/2026-10-16/common105_TAB577_fair_entities_v1'
prior_status=read(prior_packet/'iteration_status.json');assert not prior_status['handoff_completed_at_capture']
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/common105_TAB577_actual_safe_handoff_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.py -text\n',encoding='utf-8')
for name in ['handoff_identity.json','handoff.json','previous_status_at_handoff.json','previous_status_original_bytes.json','launch.json']:
    shutil.copyfile(base/name,packet/name)
write(packet/'actual_successor_state.json',state)
write(packet/'all11_existing_full_entity_fragment_bindings.json',dict(records=records,case_id=entry['case']['id'],
    source_native_lineage=bound['lineage'],all54_required_entities=len(bound['dataset']['entities']),
    new_metric_computations_by_this_closure=0,original_worker_commands_unchanged=True))
write(packet/'prior_registration_binding.json',dict(packet=prior_packet.relative_to(ROOT).as_posix(),
    iteration_status_sha256=sha(prior_packet/'iteration_status.json'),configuration=s.CONFIG,configuration_sha256=sha(ROOT/s.CONFIG)))
write(packet/'iteration_status.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_scope=c['original_scope'],
    actual_safe_metadata_handoff_verified=True,all11_old_fragments_revalidated=True,
    stopped_numerical_processes=0,new_completed_training_models=0,new_completed_TAB_cases=0,
    required_TAB_cases=577,required_label_views=1726,all_original_experiments_complete=False))
(packet/'README.md').write_text('''# 公共577项TAB队列：实际安全接管完成

2026-10-16 16:57:19（北京时间），原TAB实体e010真实完成后，继任元数据调度器取得实际共享复核锁，完成两次旧父进程PID/创建时间/完整命令校验。后代仅为精确Windows conhost宿主，无数值子进程；仅替换空闲元数据父进程，数值进程停止数为0。
最后数值子进程8532的创建时间、原完整命令、退出0/无资源中止守卫和完整实体报告均已核验。旧元数据状态原始字节及结构化快照保存，两份内容一致。

此前e000–e010共11个完整实体的实际原始指标、全部durable cell journals、报告/守卫哈希和同一源/原生lineage重新核验通过，全部复用，没有重新计算或重复计数。该模型仍需全部54实体及最终汇总；11实体不是一个完整TAB实验。
继任31572的真实OS身份确认存活、无stderr，完整577任务/1726标签视图保持，每个新实体完成后让出原锁20秒的调度已启用。原指标代码/函数调用/阈值/VUS窗口/线程数/批次/内存门槛/科学产物路径未变。其他数值进程继续运行。

此前登记包保留“尚未接管”的历史快照。本包提供实际接管与片段复用证据，不能把观察进程启动时间当成接管完成时间，也不把调度修复或部分实体计入新的完整任务。
当前仍受原定内存条件限制；让锁修复不意味着所有其他复核已进入计算。TimesNet与其他待办由原队列继续，全部45论文/681方法记录/273数据任务/702指标及原论文/TAB/新数据实验目标保持，整体未完成。
''',encoding='utf-8')
shutil.copyfile(__file__,packet/'closure_source.py')
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/common105_TAB577_actual_safe_handoff.2026-10-16_v1.json'
assert not ara.exists();write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),actual_safe_handoff_verified=True,
    old_complete_entity_fragments_revalidated=11,new_completed_TAB_cases=0,all_original_experiments_complete=False))
paths=[ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
def head(p):return subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD']).decode().strip()
write(ROOT/'tmp/common105_TAB577_actual_handoff_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),verified_old_entities=11,actual_successor=actor.pid,numerical_workers_stopped=0)))
