from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET

import psutil

from scripts.flow_matching import run_saits_full210_idle_host_cold_v1 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

c = read(ROOT / s.CONFIG)
_, runtime, queue = s.verify(c)
base = ROOT / c['handoff_root']
receipt = read(base / 'actual_handoff.json')
old = read(base / 'previous_status_at_verified_handoff.json')
state_path = ROOT / c['state_root'] / 'status.json'
state = read(state_path)
p = psutil.Process(state['pid'])
assert p.create_time() == state['create_time'] and p.is_running()
assert p.memory_info().private < 64 * 2**20
assert not receipt['original_numeric_workers_suspended_or_stopped'] and receipt['idle_heavy_lock_unowned']
try:
    prior = psutil.Process(s.OWNER['pid'])
except psutil.NoSuchProcess:
    pass
else:
    assert not s.exact_owner(prior, s.OWNER)
assert len(old['jobs']) == 66 and set(old['jobs'].values()) == {'completed'}
expected = set(old['jobs'])
by_id = {read(ROOT / j['config'])['id']:j for j in queue['jobs']}
collection = read(ROOT/'configs/runtime/fm_saits_completion_replay_queue.v1.json')
review_configs = [ROOT/x for x in collection['known_completed_independent_configurations']]
review_configs += list((ROOT/collection['generated_config_root']).glob('*.review.json'))
historical_hashes = {}
for cfg_path in review_configs:
    cfg = read(cfg_path)
    prior_proof = ROOT/cfg['output_directory']/'independent_full_replay_batch_verification.json'
    if not prior_proof.exists():
        continue
    review = read(prior_proof)
    if not review.get('passed'):
        continue
    for row in review['records']:
        if row['id'] in expected and 'result' in row.get('hashes',{}):
            value = dict(result_sha256=row['hashes']['result'], prior_independent_proof=str(prior_proof), prior_independent_proof_sha256=sha(prior_proof))
            if row['id'] in historical_hashes:
                assert historical_hashes[row['id']]['result_sha256'] == value['result_sha256']
            historical_hashes[row['id']] = value
assert set(historical_hashes) == expected
retained = []
for identity in sorted(expected):
    item = by_id[identity]
    model = read(ROOT / item['config'])
    path = ROOT / model['output_root'] / 'result.json'
    result = read(path)
    assert result['status'] == 'completed' and result['config_sha256'] == item['config_sha256']
    assert result['config'] == model
    assert sha(path) == historical_hashes[identity]['result_sha256']
    retained.append(dict(id=identity, result=str(path), config=item['config'], config_sha256=item['config_sha256'], **historical_hashes[identity]))
selected = s.selected_resume_binding(runtime)
assert selected == c['original_saved_epoch_condition'] and selected['state'] == 'pending_exact_saved_epoch271_resume'
tests = Path('F:/aicoding/IIA_Data/experiments/SAITS_full210_cold_idle_handoff_20261016_v3_tests.xml')
suites = list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(x.get('tests', 0)) for x in suites) == 14
assert not any(int(x.get(k, 0)) for x in suites for k in ('failures', 'errors', 'skipped'))
packet = ROOT / 'projects/flow_matching_research/results/2026-10-16/SAITS_full210_idle_metadata_memory_handoff_v1'
packet.mkdir(parents=True, exist_ok=False)
(packet / '.gitattributes').write_text('*.json -text\n*.xml -text -whitespace\n*.log -text -whitespace\n*.py -text\n', encoding='utf-8')
for label, path in dict(actual_handoff=base/'actual_handoff.json',
    previous_status_at_handoff=base/'previous_status_at_verified_handoff.json',
    original_source_binding=base/'actual_source_binding.json', actual_successor_state=state_path,
    original_retirement_state=ROOT/c['previous_state_path']).items():
    shutil.copyfile(path, packet/(label+'.json'))
shutil.copyfile(tests, packet/'validation_14_checks.xml')
shutil.copyfile(Path('F:/aicoding/IIA_Data/experiments/SAITS_full210_cold_idle_handoff_20261016_v1_tests.xml'),
                packet/'preserved_pre_registration_missing_hashlib_test_failure.xml')
shutil.copyfile(__file__, packet/'publication_source.py')
launch = Path('F:/aicoding/IIA_Data/experiments/saits_full210_idle_host_cold_v1_launch')
for name in ['launch.json','stdout.log','stderr.log']:
    shutil.copyfile(launch/name, packet/('actual_launch_'+name))
write(packet/'all66_preexisting_completed_source_results_retained.json', dict(records=retained,
    count=66, native_commands_unchanged=True, actual_source_hash_binding_unchanged=True))
write(packet/'pending_original_saved_epoch271_recovery_binding.json', selected)
handles = []
for pid, ct in [(state['pid'],state['create_time']), (23972,1792133828.538958),
                (39064,1792133880.0474632), (19444,1792131765.9271505),
                (36912,1792097263.1796665), (41992,1791948094.8603492)]:
    proc = psutil.Process(pid)
    assert proc.create_time() == ct and proc.is_running()
    handles.append(dict(pid=pid,create_time=ct,command=proc.cmdline(),cpu_seconds=sum(proc.cpu_times()[:2])))
write(packet/'actual_live_original_numerical_workers_and_successor.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),handles=handles))
after = p.memory_info().private
write(packet/'iteration_status.json', dict(original_scope=c['original_scope'],required_original_jobs=210,
    retained_original_completed_source_models=66, source_models_revalidated_so_far=len(state['jobs']),
    actual_source_revalidation_still_in_progress=state['state']=='starting',
    previous_private_bytes=receipt['previous_private_bytes'], successor_private_bytes=after,
    observed_private_memory_difference_bytes=receipt['previous_private_bytes']-after,
    original_training_resource_gates_lowered=False, selected_epoch271_pending_resume_preserved=True,
    original_native_workers_stopped=False, new_training_or_native_replay_completions=0,
    all_original_experiments_complete=False))
(packet/'README.md').write_text('''# SAITS/BRITS 全210条件空闲控制器内存交接

仅交接原PID2972、创建时间1791918817.7624688及完整命令匹配的空闲元数据控制器。
交接时active_pid为空、共享重资源锁未持有，唯一后代为Windows原生conhost；训练、测试和重放进程未被停止或悬挂。
原私有内存694747136字节（约663MiB），后继约18MiB；每进程观察差约645MiB，不能把整个主机内存波动归因于此交接。

新控制器从冻结源码编译原调度函数，避开未使用的数据处理导入；原210配置、所有原数值命令、环境、数据、随机种子、训练上限与早停设置不变。
原已使用的按准入门槛排序和逐job28GiB/特殊条件40GiB设置完整复用，所有GPU及紧急守卫不变。
原66个已完成源结果及配置哈希仍可核对，后继使用原验证器逐项重校验后继续剩余条件。完成计数不因交接增加。
新状态/恢复输入归档/未来源守卫写入独立F盘元数据目录，模型结果目录保持原配置路径，原源状态记录后继配置；现有自动完整重放收集器继续观察原模型结果。

特殊条件BRITS Electricity nominal0.2 seed27仍待从epoch271、1897步精确恢复，不能误当作已完成。
原检查点、优化器、Python/NumPy/Torch CPU/CUDA RNG恢复证明、现场字节哈希和40GiB准入均保留；新控制器复用原全部前置条件。
注册前测试发现标准库hashlib命名空间缺失并已修正；该失败XML保留。实际saved-epoch预检还纠正了“已完成”的错误假设，在停止旧控制器前核对真实pending状态。
最终14项身份/无数值后代、原调度AST、全210配置、精确saved-epoch恢复条件与低内存检查通过。
此交接不新增训练或原生重放完成数；全部45篇论文、681条方法/基线/消融、273条原始数据任务、702条标量目标及全部未完成实验继续保留。
''',encoding='utf-8')
write(packet/'packet_integrity.json',dict(files=[dict(path=x.relative_to(packet).as_posix(),sha256=sha(x))
    for x in sorted(packet.rglob('*')) if x.is_file()]))
ara = ROOT/'projects/flow_matching_research/ara/execution/SAITS_full210_idle_host_cold.2026-10-16_v1.json'
assert not ara.exists()
write(ara,dict(configuration=s.CONFIG,original_runtime=s.RUNTIME,actual_state_root=c['state_root'],
    packet=packet.relative_to(ROOT).as_posix(),required_original_jobs=210,retained_source_completions=66,
    pending_epoch271_resume_still_required=True,new_training_completions=0,all_original_experiments_complete=False))
def head(path):return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
paths=[s.CONFIG,s.MODULE.replace('.','/')+'.py',s.TEST,ara.relative_to(ROOT).as_posix()]
paths += [x.relative_to(ROOT).as_posix() for x in packet.rglob('*') if x.is_file()]
manifest=ROOT/'tmp/SAITS_full210_cold_handoff_publication_manifest_20261016_v1.json'
write(manifest,dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),source_revalidated_so_far=len(state['jobs']),successor_private_bytes=after)))
