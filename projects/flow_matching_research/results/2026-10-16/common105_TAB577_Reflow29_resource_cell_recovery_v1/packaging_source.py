from datetime import datetime,timezone
from pathlib import Path
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import recover_common105_TAB577_resource_cells_v1 as s

c=s.read(s.ROOT/s.CONFIG);entries=s.verify(c);base=s.ROOT/c['state_root'];launch=s.read(base/'launch.json')
p=psutil.Process(launch['pid']);assert p.is_running() and p.create_time()==launch['create_time'] and p.cmdline()==launch['command']
state=s.read(base/'status.json');assert state['pid']==p.pid and state['create_time']==p.create_time()
assert not state['failed_recoveries_preserved']
entry=next(e for e in entries if e['key']==launch['actual_failed_case_key'])
evidence=s.resource_abort(c,entry);assert evidence==launch['actual_resource_abort']
original=s.read(s.ROOT/s.matched.CONFIG);prefix='e029'
bank=Path(original['artifact_root'])/'cases'/entry['key']/(prefix+'_cell_bank')
cells=[dict(path=str(x),sha256=s.sha(x),preserved_cell_record=s.read(x)) for x in sorted(bank.glob('*.json'))]
assert len(cells)==5
for binding in launch['reused_original_successful_entities']:
    assert s.sha(binding['path'])==binding['sha256']
    report,proof=s.matched.checked_fragment(original,entry,binding['prefix'])
    assert str(report)==binding['path']
oldstate=s.read(s.ROOT/c['original_failed_registry']);oldp=psutil.Process(oldstate['pid'])
assert oldp.is_running() and oldp.create_time()==oldstate['create_time']
xml=Path('F:/aicoding/IIA_Data/experiments/common105_TAB577_resource_cell_recovery_20261016_v1_tests.xml')
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==29
assert all(int(x.get('errors',0))==int(x.get('failures',0))==int(x.get('skipped',0))==0 for x in suites)
packet=s.ROOT/'projects/flow_matching_research/results/2026-10-16/common105_TAB577_Reflow29_resource_cell_recovery_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
for source,name in [(base/'launch.json','actual_launch.json'),(base/'status.json','captured_recovery_controller_state.json'),
    (s.ROOT/s.CONFIG,'frozen_configuration.json'),(s.ROOT/c['original_failed_registry'],'captured_still_live_original_fair_state.json'),
    (xml,'relevant29_resource_and_cell_integrity_checks.xml')]:shutil.copyfile(source,packet/name)
for item in evidence['original_failed_guards']:shutil.copyfile(item['guard_path'],packet/('preserved_failed_'+item['prefix']+'_guard.json'))
s.durable.atomic_new(packet/'all5_preserved_partial_cell_bindings.json',dict(cells=cells,old_bank_unchanged=True,new_complete_guarded_entity_credit=0))
s.durable.atomic_new(packet/'all29_guarded_original_entity_bindings.json',dict(entities=launch['reused_original_successful_entities'],reexecution_performed=False))
s.durable.atomic_new(packet/'iteration_status.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    original_scope=c['original_scope'],required_matched_cases=577,required_actual_label_views=1726,required_entity_label_views=77490,
    actual_resource_aborted_case=entry['case'],required_case_entities=54,preserved_complete_original_entities=29,
    preserved_partial_metric_cells=5,full_case_complete=False,new_complete_recovered_entities=state['completed_recovered_entities'].get(entry['key'],0),
    actual_live_recovery_handle=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()),
    actual_live_original_fair_handle=dict(pid=oldp.pid,create_time=oldp.create_time(),command=oldp.cmdline()),
    old_partial_cells_and_successful_entities_are_not_new_training_repetitions=True,all_original_experiments_complete=False))
(packet/'README.md').write_text('''# matched105 TAB577：Reflow／SMAP／seed1103资源缺口恢复

原公平控制器已完成此案例29/54实体；e029实体实际守卫退出1，提交内存17027133440字节低于原16GiB紧急门槛，旧精确PID/create-time已终止。此为资源中止，不是算法性能失败。失败记录、29个完整成功实体及5个部分单元的原始字节与哈希保留。

新恢复控制器已启动，原公平控制器继续处理其他实际案例，无数值进程被停止。保留全部577案例、1726标签视图、77490实体标签视图。只允许每个缺失实体一次显式恢复；任何新的资源/数值失败继续保留，不自动重复。准入提高到32GiB并要求6次间隔5秒的稳定采样，原物理内存、16GiB紧急门槛、3/6GiB工作进程上限、CPU2及原审计锁保持。实体间释放锁并让出20秒。

原有成功实体逐项重新核验后引用，绝不重复计算成独立实验。原部分指标单元复制到新目录并逐字节核对；实际恢复会核对原模型/分数/标签来源、函数和输入参数哈希、单元输出哈希，再仅执行缺失单元。全部21个TAB函数、每标签201次调用、所有原阈值和完整VUS网格保持；新的完整实体进程成功退出后才产生实体结果，完整54实体和实际汇总成功前不称完整案例。

29项相关检查通过，包含实际资源中止、完整范围、原数值程序绑定、字节复制、损坏来源/参数/结果拒绝及冷启动导入。当前排队和单元准备不是基准性能；未重训或重新推理模型。原873/878轨道与45篇论文、681方法、273数据任务、702指标及原论文/TAB/新数据全部目标仍保留，整体未完成。
''',encoding='utf-8')
shutil.copyfile(__file__,packet/'packaging_source.py')
s.durable.atomic_new(packet/'packet_integrity.json',dict(files=[dict(path=x.relative_to(packet).as_posix(),sha256=s.sha(x)) for x in sorted(packet.rglob('*')) if x.is_file()]))
ara=s.ROOT/'projects/flow_matching_research/ara/execution/common105_TAB577_resource_recovery.2026-10-16_v1.json'
s.durable.atomic_new(ara,dict(packet=packet.relative_to(s.ROOT).as_posix(),configuration=s.CONFIG,full_case_complete=False,
    required_cases=577,required_label_views=1726,required_entities_in_actual_failed_case=54,original_guarded_entities=29,
    old_partial_metric_cells=5,all_original_experiments_complete=False))
paths=[s.CONFIG,s.TEST,s.MODULE.replace('.','/')+'.py',ara.relative_to(s.ROOT).as_posix(),'projects/flow_matching_research/README.rst']
paths+=[x.relative_to(s.ROOT).as_posix() for x in packet.rglob('*') if x.is_file()]
head=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
manifest=s.ROOT/'tmp/matched577_resource_recovery_manifest_20261016_v1.json'
s.durable.atomic_new(manifest,dict(previous_root=head(s.ROOT),previous_parent=head(s.ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),old_entities=29,old_partial_cells=5,full_case_complete=False)))
