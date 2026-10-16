from datetime import datetime, timezone
from pathlib import Path
import csv
import io
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET

import psutil

from scripts.flow_matching import collect_crossad_CUDA_full139_verified_v1 as original
from scripts.flow_matching import collect_crossad_full139_cold_handoff_v1 as cold
from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1 as metrics
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

identity = 'crossad_author__fresh_train__gecco__seed1104'
c = read(ROOT / cold.CONFIG)
parent, q = cold.verify(c)
mc = read(ROOT / metrics.CONFIG)
mq, upstream = metrics.verify(mc)
assert q == mq
cs = read(ROOT / c['state_root'] / 'status.json')
ms = read(ROOT / mc['state_root'] / 'status.json')
expected = {f'crossad_author__fresh_train__gecco__seed{x}' for x in [1103, 1104, 1105]}
assert expected == set(cs['verified_original_CUDA_ids']) == set(ms['verified_ids'])
assert not cs['errors_preserved'] and not ms['failed_preserved']
for state in [cs, ms]:
    process = psutil.Process(state['pid'])
    assert process.create_time() == state['create_time'] and process.is_running()
job = next(j for j in q['jobs'] if j['canonical_CPU_condition_reference'] == identity)
state_dir = ROOT / q['state_root'] / 'jobs' / identity
artifact = Path(q['artifact_root']) / identity
source = read(ROOT / job['output_directory'] / 'result.json')
trace = read(ROOT / job['output_directory'] / 'training_trace.json')
assert source['status'] == 'completed' and source['details']['epochs_completed'] == 19
assert trace['epochs_finished'] == 19 and trace['declared_max_epochs'] == 20
assert trace['training_forward_calls'] == source['details']['training_forward_calls'] == 19 * 432
losses = trace['validation_losses']
assert len(losses) == 19 and losses.index(min(losses)) == 15
assert all(v > losses[15] for v in losses[16:]) and job['train_parameters']['patience'] == 3
assert source['details']['declared_train_parameters'] == job['train_parameters']
assert source['details']['native_points'] == dict(train=55408, validation=13852, test=69261)
assert source['details']['original_unscored_tail_points'] == 13
tests = Path('F:/aicoding/IIA_Data/experiments/CrossAD_third_full_GPU_result_replay_metric_closure_20261016_v1_tests.xml')
suites = list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 15
assert not any(int(s.get(k, 0)) for s in suites for k in ('failures', 'errors', 'skipped'))

reports = {}
rows = []
for canonical in sorted(expected):
    j = next(j for j in q['jobs'] if j['canonical_CPU_condition_reference'] == canonical)
    proof_path = Path(parent['artifact_root']) / 'conditions' / (canonical + '.json')
    proof = read(proof_path)
    assert original.condition(parent, q, j) == proof
    report_path = Path(mc['artifact_root']) / 'conditions' / (canonical + '.json')
    report = read(report_path)
    assert metrics.audit(mc, q, upstream, j) == report
    guard_path = ROOT / mc['state_root'] / 'guards' / (canonical + '.json')
    guard = read(guard_path)
    assert guard['passed'] and guard['exit_code'] == 0 and not guard['resource_guard_aborted']
    assert guard['report_sha256'] == sha(report_path)
    reports[canonical] = proof
    for ratio, value in report['independent_strict_metrics'].items():
        ci = proof['native_validation_control']['strict'][ratio]['f1_ci95']
        rows.append(dict(canonical_id=canonical, dataset=j['dataset'], seed=j['seed'],
            validation_alarm_percent=ratio, threshold=value['threshold'], **value['micro'],
            **value['event_metrics'], original_conditional_block_f1_CI95_low=ci['low'],
            original_conditional_block_f1_CI95_high=ci['high'], bootstrap_repeats=ci['resamples'],
            bootstrap_independently_recomputed=False, full_test_points=69261, test_positives=730,
            point_adjustment=False, strict_TAB_result=False))
summary = original.cohorts(q['jobs'], reports)
assert summary['complete_primary_cohorts'] == 0
assert summary['primary_cohorts']['GECCO/main']['verified_seed_ids'] == [1103, 1104, 1105]
assert 'metrics' not in summary['primary_cohorts']['GECCO/main']
packet = ROOT / 'projects/flow_matching_research/results/2026-10-16/CrossAD_third_full_GPU_model_native_replay_and_independent_metrics_v1'
packet.mkdir(parents=True, exist_ok=False)
(packet / '.gitattributes').write_text('*.json -text\n*.xml -text -whitespace\n*.csv -text\n*.py -text\n', encoding='utf-8')
new = packet / 'new_complete_model'
new.mkdir()
for name, path in dict(source_result=ROOT / job['output_directory'] / 'result.json',
    training_trace=ROOT / job['output_directory'] / 'training_trace.json',
    source_device_proof=state_dir / 'source_device_proof.json', full_native_replay=artifact / 'native_replay/report.json',
    **{phase + '_guard': state_dir / (phase + '.guard.json') for phase in ['CUDA_probe', 'CPU_reference', 'author', 'replay']}).items():
    shutil.copyfile(path, new / (name + '.json'))
for canonical in sorted(expected):
    out = packet / 'all_complete_conditions' / canonical
    out.mkdir(parents=True)
    for name, path in dict(full_checkpoint_and_arrays_proof=Path(parent['artifact_root']) / 'conditions' / (canonical + '.json'),
        independent_metric_report=Path(mc['artifact_root']) / 'conditions' / (canonical + '.json'),
        independent_metric_guard=ROOT / mc['state_root'] / 'guards' / (canonical + '.json')).items():
        shutil.copyfile(path, out / (name + '.json'))
buffer = io.StringIO(newline='')
writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
writer.writeheader()
writer.writerows(rows)
(packet / 'all_three_models_nine_strict_metric_rows.csv').write_bytes(buffer.getvalue().encode('utf-8-sig'))
write(packet / 'all22_cohort_completion.json', summary)
write(packet / 'early_stopping_completion_evidence.json', dict(declared_maximum_epochs=20,
    author_patience=3, completed_epochs=19, best_validation_epoch=16,
    full_validation_losses=losses, training_forward_calls=8208,
    full_source_guard_passed=True, budget_reduced=False, terminal_reason='original_validation_patience_exhausted',
    checkpoint_sha256=source['details']['algorithm_weights']['checkpoint_sha256']))
shutil.copyfile(tests, packet / 'validation_fifteen_checks.xml')
shutil.copyfile(__file__, packet / 'closure_source.py')
handles = []
for state in [read(ROOT / q['state_root'] / 'status.json'), cs, ms]:
    p = psutil.Process(state['pid'])
    assert p.create_time() == state['create_time'] and p.is_running()
    handles.append(dict(pid=p.pid, create_time=p.create_time(), command=p.cmdline(), state=state,
        descendants=[dict(pid=x.pid, create_time=x.create_time(), command=x.cmdline())
                     for x in p.children(recursive=True) if 'python' in x.name().lower()]))
write(packet / 'actual_live_continuation_handles.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), handles=handles))
write(packet / 'iteration_status.json', dict(original_scope=q['original_scope'],
    verified_original_CUDA_conditions=3, required_original_CUDA_conditions=139,
    primary_GECCO_training_seeds_verified=[1103, 1104, 1105], missing_GECCO_primary_seeds=[1106, 1107],
    new_complete_original_CUDA_models=1, original_CPU_completion_credit=0,
    required_primary_cohorts=22, complete_primary_cohorts=0,
    full_original_score_points_per_model=138521, positive_test_points_per_model=730,
    all_original_experiments_complete=False))
(packet / 'README.md').write_text('''# CrossAD：第三个完整GPU模型

GECCO seed1104按原始20轮上限、patience=3、batch=128、完整样本与原模型训练，在第19轮正常早停；最佳验证轮为16，最后三轮均未改善，共8208个完整训练前向批次，没有缩减预算。
原作者测试/校准评估完成，随后新GPU进程重载检查点，完整55408个训练点、13852个验证点、69261个测试点的评分全部逐字节一致，含原前缀未覆盖的13个尾部点。
CPU/GPU初始评分一致性、完整源执行、全量GPU检查点重放四项实际进程守卫全部通过。
独立标准库实现重新核算三个验证集告警阈值、混淆矩阵、F1、ROC-AUC、AP及事件统计，全部通过。

| GECCO seed1104，验证集1%阈值，无PA | 数值 |
| --- | --- |
| F1 | 0.4155844155844156 |
| Precision | 0.6436781609195402 |
| Recall | 0.30684931506849317 |
| ROC-AUC | 0.986339288908949 |
| AP | 0.545477959231486 |

全139条件中现有3个完整GPU模型通过验证；本目录包含三模型、三阈值的全部9行指标，以及原始条件块bootstrap区间。bootstrap区间沿用完整源评估，未由标准库审计独立重算。
GECCO原定五个主要种子仅完成1103/1104/1105；1106/1107仍缺，禁止用3个种子均值或区间冒充完整五种子结论。
源队列已经继续处理seed1106。GPU协议和原CPU协议保留独立身份，本次不抵扣CPU实验，也不声称历史论文等价或完整TAB成绩。
15项配置/设备/全量数组/指标/完整五种子统计/交接回归测试通过。所有45篇论文、681条方法/基线/消融、273条原始数据任务和702条标量目标及全部未完成实验保持在范围内。
''', encoding='utf-8')
write(packet / 'packet_integrity.json', dict(files=[dict(path=p.relative_to(packet).as_posix(), sha256=sha(p))
    for p in sorted(packet.rglob('*')) if p.is_file()]))
ara = ROOT / 'projects/flow_matching_research/ara/execution/CrossAD_third_full_GPU_model.2026-10-16_v1.json'
assert not ara.exists()
write(ara, dict(configuration=original.QUEUE, checkpoint_collector_configuration=cold.CONFIG,
    independent_metric_configuration=metrics.CONFIG, new_complete_condition=identity,
    result_packet=packet.relative_to(ROOT).as_posix(), actual_epochs=19, early_stopped_under_original_patience=True,
    full_original_CUDA_conditions_verified=3, required_original_CUDA_conditions=139,
    original_CPU_completion_credit=0, all_original_experiments_complete=False))
def head(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD']).decode().strip()
manifest = ROOT / 'tmp/CrossAD_third_full_GPU_publication_manifest_20261016_v1.json'
paths = [p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()] + [ara.relative_to(ROOT).as_posix()]
write(manifest, dict(previous_root=head(ROOT), previous_parent=head(ROOT.parent / 'flow-matching-research'), paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet), full_CUDA_models=3, required_CUDA_models=139, metric_rows=len(rows), files=len(paths))))
