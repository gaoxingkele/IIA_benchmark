from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET

import psutil

from scripts.flow_matching import reexecute_spectral_full70_generation_evaluators_v1 as s
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

c = read(ROOT / s.CONFIG)
parent, cases = s.verify(c)
state_path = ROOT / c['state_root'] / 'status.json'
state = read(state_path)
owner = psutil.Process(state['pid'])
assert owner.create_time() == state['create_time'] and owner.is_running()
assert not state['verified_ids'] and not state['failed_preserved'] and state['active_pid'] is None
assert state['state'] == 'waiting_for_heavy_lock'
assert owner.memory_info().private < 64 * 2**20
tests = Path('F:/aicoding/IIA_Data/experiments/Spectral_full70_generation_evaluators_and_source_regression_20261016_v1_tests.xml')
suites = list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(v.get('tests', 0)) for v in suites) == 48
assert not any(int(v.get(k, 0)) for v in suites for k in ('failures', 'errors', 'skipped'))
packet = ROOT / 'projects/flow_matching_research/results/2026-10-16/Spectral_full70_actual_generation_and_evaluator_reexecution_registered_v1'
packet.mkdir(parents=True, exist_ok=False)
(packet / '.gitattributes').write_text('*.json -text\n*.xml -text -whitespace\n*.log -text -whitespace\n*.py -text\n', encoding='utf-8')
shutil.copyfile(tests, packet / 'validation_48_native_environment_checks.xml')
shutil.copyfile(state_path, packet / 'actual_queue_state.json')
launch = Path('F:/aicoding/IIA_Data/experiments/spectral_full70_generation_evaluator_reexecution_v1_launch')
for name in ['launch.json', 'stdout.log', 'stderr.log']:
    shutil.copyfile(launch / name, packet / ('actual_launch_' + name))
handles = []
for pid, ct in [(owner.pid, owner.create_time()), (18588, 1792132261.7147353),
                (4716, 1792131761.7990527), (19444, 1792131765.9271505),
                (36912, 1792097263.1796665), (41992, 1791948094.8603492)]:
    p = psutil.Process(pid)
    assert p.create_time() == ct and p.is_running()
    handles.append(dict(pid=pid, create_time=ct, command=p.cmdline(),
        cpu_seconds=sum(p.cpu_times()[:2]), private_bytes=p.memory_info().private,
        descendants=[dict(pid=x.pid, name=x.name(), command=x.cmdline()) for x in p.children(recursive=True)]))
write(packet / 'actual_live_original_workers_and_new_queue.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), handles=handles))
write(packet / 'all70_condition_bindings.json', dict(canonical_conditions=[dict(
    id=identity, actual_source_queue=case['queue'], method=case['job']['model'],
    dataset=case['job']['dataset'], data_track=case['job']['data_track'],
    original_training_seed=case['job']['seed'], new_sampling_seed=c['new_sampling_seed_offset'] + case['job']['seed'],
    original_model_config=case['job']['model_config'],
    original_full_shape=case['job']['data_shape'], original_batch=case['job']['generation_batch_size'],
    unchanged_resource_settings=s.case_settings(c, case['job'])) for identity, case in cases.items()]))
write(packet / 'unchanged_source_bindings.json', dict(source_receipts=c['source_receipts'],
    original_source_queue_sha256=sha(ROOT / c['original_queue']),
    unmodified_evaluator_section_extraction='complete AST from import tensorflow through all four five-repeat metric statistics; no evaluator statements edited',
    original_sampler='unmodified Trainer.sample with full num, original size_every, all T,D and unchanged generate_mts'))
write(packet / 'iteration_status.json', dict(original_scope=c['original_scope'],
    registered_checkpoint_reexecution_conditions=70, required_full_same_RNG_generations_per_condition=2,
    required_cohorts=14, required_evaluator_families=4, required_repeats_per_family=5,
    new_training_seed_slots=0, new_generator_optimizer_updates=0,
    actual_full_generation_reexecution_conditions_completed=0,
    actual_learned_evaluator_reexecution_conditions_completed=0,
    historical_original_sampling_RNG_recovered=False, historical_generation_equivalence_claim=False,
    original_resource_gates_lowered=False, existing_numerical_workers_stopped=False,
    all_original_experiments_complete=False))
shutil.copyfile(__file__, packet / 'publication_source.py')
(packet / 'README.md').write_text('''# Spectral：全70条件真实生成与评估器重放队列

补齐已知缺口：原来的32个完整源实验与产物审计没有独立执行生成模型，也没有重新训练学习式评估器。
新队列覆盖原70个模型/数据轨道/训练种子条件、14个完整五种子组，并等待每个原始训练、采样、完整评估及产物审计真实通过。

每个条件执行完整原始采样两次：加载并核对完整模型、EMA和优化器状态，使用固定新采样种子1000000+原训练种子，完整保存Python/NumPy/Torch CPU/CUDA随机状态。
第一次生成后从落盘状态恢复，再按相同原始批量、原始网格与求解器生成完整样本；全部样本、额外尾部行、所有特征必须逐字节一致，随机状态消耗也必须相等。
禁止生成器优化器更新，并在采样后再次核对模型、EMA、优化器和原始检查点未变。
随后执行完整原始四类评估器各五次，保留原有学习式评估器训练预算、输入人口和图种子0..4；生成结果和原始结果作逐条件对比。
仅在原五个训练种子42..46齐全后计算配对模型指标差异和描述性t区间；评估器重复及第二次相同RNG采样均不计为新的训练种子。

历史采样起始RNG未保存，本轨道明确使用新随机状态；它验证新快照下的完整重放和质量稳定性，不声称恢复历史生成数组或论文硬件历史等价。
原训练队列、原检查点、原生成数组及原结果均保持不变。新大文件写入F盘独立目录。
沿用原GPU锁、GPU身份、Diffusion-TS原20GiB提交内存/12GiB紧急门槛及全部GPU守卫；Spectral Flow保留48GiB提交内存要求。

48项原生环境测试通过。其中7项新测试覆盖完整随机状态、数组尾部与dtype、全部检查点字段、全量20次评估调度及五种子门禁；模拟评估器只用于调度测试，不能作为真实性能证据。
实际控制器已启动，当前等待原资源锁，真实生成和评估器复跑完成数均为0。不能把注册、测试或等待状态称为已复现实验效果。
全部45篇论文、681条方法/基线/消融、273条原始数据任务、702条标量目标及原论文/TAB/新数据轨道仍保留，整体实验尚未完成。
''', encoding='utf-8')
write(packet / 'packet_integrity.json', dict(files=[dict(path=p.relative_to(packet).as_posix(), sha256=sha(p))
    for p in sorted(packet.rglob('*')) if p.is_file()]))
ara = ROOT / 'projects/flow_matching_research/ara/execution/Spectral_full70_actual_generation_evaluator_reexecution.2026-10-16_v1.json'
assert not ara.exists()
write(ara, dict(configuration=s.CONFIG, original_source_queue=c['original_queue'],
    packet=packet.relative_to(ROOT).as_posix(), required_conditions=70, required_cohorts=14,
    actual_completed_reexecution_conditions=0, original_training_seed_slots_added=0,
    historical_original_generated_bytes_equivalence_claim=False, all_original_experiments_complete=False))
paths = [s.MODULE.replace('.', '/') + '.py', s.CONFIG, s.TEST, ara.relative_to(ROOT).as_posix()]
paths += [p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
def head(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD']).decode().strip()
manifest = ROOT / 'tmp/Spectral_full70_generation_reexecution_publication_manifest_20261016_v1.json'
write(manifest, dict(previous_root=head(ROOT), previous_parent=head(ROOT.parent / 'flow-matching-research'), paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet), files=len(paths), actual_state=state['state'], actual_completed=0)))
