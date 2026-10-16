from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET

import psutil

from scripts.flow_matching import recover_otcfm_completed1000_CPU_test_v1 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

c = read(ROOT / s.CONFIG)
q, job = s.verify(c)
base = ROOT / c['state_root']
state = read(base / 'status.json')
owner = psutil.Process(state['pid'])
assert owner.create_time() == state['create_time'] and owner.is_running()
assert state['completed_phases'] == ['proof'] and state['active_pid'] is None
proof_path = s.phase_report(c, 'proof')
proof = read(proof_path)
guard = read(base / 'proof.guard.json')
assert guard['passed'] and guard['exit_code'] == 0 and not guard['resource_guard_aborted']
assert guard['report_sha256'] == sha(proof_path)
assert proof['original_completed_epochs'] == 1000 and proof['original_optimizer_updates'] == 10000
assert len(proof['actual_checkpoint_observations']) == len(proof['actual_validation_history']) == 100
assert proof['best_epoch'] == 350 and proof['selected_checkpoint_optimizer_updates'] == 3500
assert not proof['full_per_epoch_callback_history_available'] and proof['no_unobserved_epoch_records_invented']
tests = Path('F:/aicoding/IIA_Data/experiments/OTCFM_completed1000_CPU_recovery_metadata_and_integrity_20261016_v1_tests.xml')
suites = list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(v.get('tests', 0)) for v in suites) == 24
assert not any(int(v.get(k, 0)) for v in suites for k in ('failures', 'errors', 'skipped'))
packet = ROOT / 'projects/flow_matching_research/results/2026-10-16/OTCFM_actual_completed1000_CPU_training_and_full_test_recovery_v1'
packet.mkdir(parents=True, exist_ok=False)
(packet / '.gitattributes').write_text('*.json -text\n*.xml -text -whitespace\n*.csv -text\n*.py -text\n*.log -text -whitespace\n', encoding='utf-8')
for label, path in dict(actual_training_proof=proof_path, proof_actual_guard=base / 'proof.guard.json',
    preserved_original_resource_abort=Path(c['original_terminal_guard']),
    preserved_full_training_terminal_progress=Path(c['original_artifact_root']) / 'progress.json',
    current_recovery_state=base / 'status.json').items():
    shutil.copyfile(path, packet / (label + '.json'))
shutil.copyfile(Path(c['original_artifact_root']) / 'csv/version_0/metrics.csv', packet / 'actual_native_training_validation_metrics.csv')
shutil.copyfile(tests, packet / 'validation_24_metadata_and_integrity_checks.xml')
shutil.copyfile(__file__, packet / 'publication_source.py')
launch = Path('F:/aicoding/IIA_Data/experiments/otcfm_completed1000_CPU_test_recovery_v1_launch')
for name in ['launch.json', 'stdout.log', 'stderr.log']:
    shutil.copyfile(launch / name, packet / ('actual_launch_' + name))
write(packet / 'actual_live_recovery_handle.json', dict(pid=owner.pid, create_time=owner.create_time(),
    command=owner.cmdline(), captured_utc=datetime.now(timezone.utc).isoformat(),
    numeric_children=[dict(pid=p.pid, create_time=p.create_time(), command=p.cmdline())
        for p in owner.children(recursive=True) if 'python' in p.name().lower()]))
write(packet / 'validation_environment_notes.json', dict(
    locked_author_environment_modified=False, author_environment_has_pytest=False,
    regression_tests='Main Python3.12 environment,24 metadata/history/AST/source-integrity regression checks',
    actual_full_checkpoint_proof='Separate guarded locked author Python3.10/Torch1.12.1+cu113 process,exit0,all101 original checkpoints bound and100 validation checkpoints/optimizer states read',
    actual_full_test_complete=False, actual_independent_native_full_LP_replay_complete=False))
write(packet / 'iteration_status.json', dict(original_scope=q['original_scope'], original_required_CPU_cases=120,
    original_full_training_epochs=1000, original_full_training_updates=10000,
    original_completed_training_independently_proved=True, actual_new_training_updates=0,
    completed_recovery_phases=['proof'], full_source_test_recovered=False,
    independent_full_distribution_replay_complete=False, new_complete_CPU_conditions=0,
    original_failure_preserved=True, full_numerical_admission_bytes=40*2**30,
    all_original_experiments_complete=False))
(packet / 'README.md').write_text('''# OT-CFM：完整1000轮CPU训练的测试恢复

原条件released_yaml/OT-CFM/作者供给donor32606/leave1/seed43已完成1000轮、10000次优化更新，在后续完整测试阶段因主机提交内存不足触发原守卫中止。
不是训练失败。原训练目录、101个检查点、CSV及失败守卫原样保留；该条件不与已排队的CUDA协议互相抵扣。

已在锁定作者Python3.10/Torch1.12.1+cu113环境中独立读取并核验100个验证检查点与末轮检查点、完整批次/优化器更新计数、所有优化器矩及100条真实验证记录。
末轮epoch999对应完整1000轮和10000步；100次验证的真实最优为epoch349（第350轮），选中检查点为3500步。
原逐轮回调没有落盘，不制造缺失的1000行历史；恢复审计使用真实保留的100个检查点观察和原始CSV。
实际只读证明进程退出0、无资源中止。24项元数据、完整预算和原始审计回归测试通过；回归使用主测试环境，作者锁定环境未安装pytest，也未因此修改其依赖。

隔离恢复控制器已启动，当前已完成训练证明，等待40GiB提交内存准入。
下一阶段直接从原最佳权重执行原完整测试，再在新进程执行原完整ODE测试及全部内点分布W1/W2，并独立用SciPy全精度代价与POT精确LP核验W1最优性和原误差界。
原测试与LP代码不变；仅两处历史元数据索引改为实际保留观察的按epoch查找，AST回归确认所有数值语句相同。
禁止新增训练更新，完整测试/重放沿用原RSS6GiB、private8GiB、emergency16GiB门槛，启动提交内存从原32GiB加强到40GiB，并持有原训练及只读锁以避免并行大LP。
只读检查点证明单独使用24GiB准入，不执行模型/数据实例化、训练、ODE或LP，不降低完整数值任务门槛。

当前完整测试及独立全分布重放均未完成，新增完整CPU条件数仍为0；不能将训练证明称为完整复现或论文成绩。
全120原始CPU条件及全部45篇论文/681条方法记录/273条原始数据任务/702条标量目标仍保留。
''', encoding='utf-8')
write(packet / 'packet_integrity.json', dict(files=[dict(path=p.relative_to(packet).as_posix(), sha256=sha(p))
    for p in sorted(packet.rglob('*')) if p.is_file()]))
ara = ROOT / 'projects/flow_matching_research/ara/execution/OTCFM_completed1000_CPU_test_recovery.2026-10-16_v1.json'
assert not ara.exists()
write(ara, dict(configuration=s.CONFIG, original_queue=s.native.QUEUE, canonical_id=s.SELECTED,
    result_packet=packet.relative_to(ROOT).as_posix(), actual_completed_training_epochs=1000,
    actual_completed_training_updates=10000, actual_new_training_updates=0,
    full_test_and_independent_replay_complete=False, all_original_experiments_complete=False))
paths = [s.MODULE.replace('.', '/') + '.py', s.CONFIG, s.TEST, ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
def head(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD']).decode().strip()
manifest = ROOT / 'tmp/OTCFM_completed1000_CPU_test_recovery_publication_manifest_20261016_v1.json'
write(manifest, dict(previous_root=head(ROOT), previous_parent=head(ROOT.parent / 'flow-matching-research'), paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet), files=len(paths), recovered_full_CPU_conditions=0, actual_proved_training_epochs=1000)))
