"""Publish the registered, tested replay obligation; no completed-model credit."""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import audit_industrial_SB_extended_full_replay_v1 as audit
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

c = read(ROOT/audit.CONFIG); audit.verify(c)
base = ROOT/c['state_root']; launch = read(base/'launch.json'); state = read(base/'status.json')
owner = psutil.Process(launch['pid'])
assert owner.is_running() and owner.create_time() == launch['create_time'] and owner.cmdline() == launch['command']
children = [dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()) for p in owner.children(recursive=True)]
assert all(Path(p['command'][0]).name.lower()=='conhost.exe' and p['command'][1:]==['0x4'] for p in children)
assert state['active_pid'] is None
assert state['state'] == 'waiting_full_source_training_and_guard'
source = read(ROOT/audit.SOURCE/'status.json')
source_owner = psutil.Process(source['pid'])
assert source_owner.is_running() and source_owner.create_time() == source['create_time']
assert not source['full_training_complete'] and source['active_pid'] is None
assert not (ROOT/audit.SOURCE/'attempts/full_training/receipt.json').exists()
assert not Path(c['artifact_root']).exists() and not (base/'worker.stdout.log').exists()
xml = ROOT/'tmp/industrial_SB_complete_solver_scheduler_replay_validation_20261016_v1.xml'
suites = list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites) == 64
assert all(int(s.get(k,0)) == 0 for s in suites for k in ['failures','errors','skipped'])
old_xml = ROOT/'tmp/industrial_SB_full_replay_v1_validation_20261016_v1.xml'
old = list(ET.parse(old_xml).getroot().iter('testsuite'))
assert sum(int(s.get('failures',0)) for s in old) == 9
out = ROOT/'projects/flow_matching_research/results/2026-10-16/industrial_SB_PRONTO1103_registered_full_checkpoint_replay_v1'
out.mkdir(exist_ok=False)
(out/'.gitattributes').write_text('*.json -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8',newline='\n')
files = dict(frozen_replay_configuration=ROOT/audit.CONFIG,
    actual_launch=base/'launch.json', current_replay_waiting_state=base/'status.json',
    current_source_full_training_waiting_state=ROOT/audit.SOURCE/'status.json',
    original_full_case_registration=ROOT/c['source_registration'])
for name, path in files.items(): shutil.copyfile(path,out/(name+'.json'))
shutil.copyfile(xml,out/'relevant_fresh64_solver_scheduler_replay_validation.xml')
shutil.copyfile(old_xml,out/'preserved_pre_registration_fixture_digest_failure.xml')
shutil.copyfile(ROOT/'tmp/industrial_SB_full_replay_v1_validation_20261016_v2.xml',out/'replay_only_fresh27_validation.xml')
sources = [audit.MODULE.replace('.','/')+'.py','tests/test_fm_industrial_SB_extended_full_replay_v1.py',audit.CONFIG]
write(out/'source_bindings.json',dict(files=[dict(path=p,sha256=sha(ROOT/p)) for p in sources],
    exact_all_array_payload_match_required=True,independent_metric_tolerance=c['numerical_metric_tolerance'],
    actual_descendants=children,actual_numerical_descendants=[]))
live = []
for pid, created in [(40500,1791915368.3060675),(41992,1791948094.8603492),(31444,1792129142.030305),(16360,1792143923.216981)]:
    p = psutil.Process(pid); assert p.is_running() and p.create_time() == created
    live.append(dict(pid=pid,create_time=created,command=p.cmdline(),cpu_user_seconds=p.cpu_times().user))
write(out/'original_live_execution_handles_preserved.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=live))
write(out/'completion_boundary.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    original_global_scope=c['original_global_scope'], original_industrial_scope=dict(CPU=550,GPU=850),
    real_coupling_problems_previously_qualified=20,full_source_training_complete=False,
    full_checkpoint_native_replay_complete=False,new_complete_model_results=0,
    full_training_commit_admission_bytes=32*2**30,replay_commit_admission_bytes=c['settings']['minimum_commit_headroom_bytes'],
    emergency_commit_headroom_bytes=c['settings']['emergency_commit_headroom_bytes'],
    exact_full_payload_bytes_required=True,neural_workers_started_by_new_replay_controller=0,
    new_optimizer_updates=0,new_training_seed_slots=0,
    original_source_and_data_unchanged=True,original_numerical_workers_stopped=0,
    bootstrap_independently_recomputed=False,VUS_independently_recomputed=False,affiliation_independently_recomputed=False,
    full_TAB_equivalence=False,paper_equivalence=False,all_original_experiments_complete=False))
(out/'README.md').write_text('''# 工业SB／PRONTO day4／seed1103：完整检查点复核已登记，等待训练

前一证据包只完成20个真实运输问题的数学验收。本次准备并启动后续复核器，源20轮训练仍等待原资源门槛和共享训练锁；新增完整模型成绩为0。

复核器只在完整源训练、20个实际训练耦合、检查点、完整分数及成功资源守卫均存在且哈希一致后启动。
它按原CPU线程数、窗口、batch、随机种子及评分公式恢复全部MLP权重，直接推理，不调用fit、优化器或运输求解器。
分别覆盖训练722点、验证4320点、测试11700点；独立实现分窗和重叠尾部对齐，逐数组要求dtype、shape和全部内容字节完全一致，包括标签、时间戳、原始行号与故障标记。
原完整训练分组和8个窗口预算也必须与检查点、源结果一致。

阈值、混淆计数、Precision/Recall/F1、并列分数AUROC/AP、逐实体和汇总事件检测/延迟以独立Python公式重新计算。
验证0.1%、1%、5%全部保留，事件不跨原实体边界，不使用测试标签选取严格阈值，不做PA。
复用已验证的独立指标公式组件，不调用源训练的指标计算函数；该复核不独立重跑bootstrap、VUS或Affiliation，不声明完整TAB或论文等价。
数值指标比较沿用已有独立核验器的rtol=atol=2e-6；数组和计数要求精确一致，不用指标容差掩盖数组不一致。

保留原共享CPU锁、12GiB物理/32GiB提交准入、16GiB紧急线及RSS/private上限。等待期间不导入Torch/NumPy，不停止或替换任何数值任务。
复核输出位于配置指定F盘新目录，源数据、失败及训练产物不改写；任何一次完整复核失败都保留且不自动重试。

64项相关验证通过，其中27项覆盖本复核器的恢复一致性和拒绝逻辑。小型训练夹具仅用于检查恢复流程，不是新实验性能。
首次测试有9项因夹具使用非SHA256占位字符串被源校准器拒绝；更正夹具来源哈希后27项全部通过，首次失败XML保留。
这不是实际模型训练失败，修正没有改变任何已冻结训练代码、数据或完整实验预算。

当前元数据控制器已真实启动，无数值子进程，等待源训练和成功守卫。全局45篇论文、681项方法/基线/消融、273数据任务、702论文标量及工业550CPU/850GPU范围保持，整体未完成。
''',encoding='utf-8',newline='\n')
shutil.copyfile(__file__,out/'publication_source.py')
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
ara = ROOT/'projects/flow_matching_research/ara/execution/industrial_SB_PRONTO1103_registered_full_replay.2026-10-16_v1.json'
assert not ara.exists()
write(ara,dict(packet=out.relative_to(ROOT).as_posix(),verified_tests=64,replay_only_tests=27,
    full_source_training_complete=False,full_checkpoint_replay_complete=False,new_complete_model_results=0,
    all_original_experiments_complete=False))
page = ROOT/'projects/flow_matching_research/README.rst'
text = page.read_text(encoding='utf-8')
text += '''
`SB／PRONTO1103完整检查点重放与独立指标已登记 <results/2026-10-16/industrial_SB_PRONTO1103_registered_full_checkpoint_replay_v1/README.md>`_
64项相关检查通过，新增27项覆盖恢复、全部数组字节、尾部和泄漏/未完成拒绝逻辑；等待原20轮源训练及成功资源守卫。
后续按原32GiB准入和共享锁恢复原模型，不重训，重放722/4320/11700个原生时间点并独立复算三种验证阈值的微观/逐实体/事件指标。
完整模型训练与重放尚未完成，新增完整模型结果0；保留原数据、失败、预算和全部研究范围。
'''
page.write_text(text,encoding='utf-8',newline='\n')
paths = sources+[ara.relative_to(ROOT).as_posix(),page.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()]
head = lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
write(ROOT/'tmp/industrial_SB_full_replay_publication_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(out),publication_files=len(paths),tests=64,new_complete_models=0,live_replay_owner=owner.pid)))
