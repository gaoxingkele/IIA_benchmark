from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import csv
import io
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET

import psutil

from scripts.flow_matching import collect_crossad_full139_cold_handoff_v1 as cold
from scripts.flow_matching import collect_crossad_CUDA_full139_verified_v1 as cross
from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1 as metrics
from scripts.flow_matching import run_spectral_full70_artifact_reviews_v1 as spectral
from scripts.flow_matching.summarize_spectral_full_cohorts_v1 import summarize_cohort
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

packet = ROOT / 'projects/flow_matching_research/results/2026-10-16/CrossAD_fourth_complete_GPU_and_Spectral33_full_source_artifacts_v1'
assert not packet.exists()
tests = Path('F:/aicoding/IIA_Data/experiments/CrossAD4_Spectral33_full_result_closure_20261016_v1_tests.xml')
suites = list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(v.get('tests', 0)) for v in suites) == 56
assert not any(int(v.get(k, 0)) for v in suites for k in ('failures', 'errors', 'skipped'))

cc = read(ROOT / cold.CONFIG)
parent, cq = cold.verify(cc)
mc = read(ROOT / metrics.CONFIG)
_, upstream = metrics.verify(mc)
cs = read(ROOT / cc['state_root'] / 'status.json')
ms = read(ROOT / mc['state_root'] / 'status.json')
expected = {f'crossad_author__fresh_train__gecco__seed{s}' for s in [1103, 1104, 1105, 1106]}
assert set(cs['verified_original_CUDA_ids']) == set(ms['verified_ids']) == expected
assert not cs['errors_preserved'] and not ms['failed_preserved']
proofs, metric_reports, cross_rows = {}, {}, []
for identity in sorted(expected):
    job = next(j for j in cq['jobs'] if j['canonical_CPU_condition_reference'] == identity)
    p = Path(parent['artifact_root']) / 'conditions' / (identity + '.json')
    proof = read(p)
    assert cross.condition(parent, cq, job) == proof
    m = Path(mc['artifact_root']) / 'conditions' / (identity + '.json')
    report = read(m)
    assert metrics.audit(mc, cq, upstream, job) == report
    guard = read(ROOT / mc['state_root'] / 'guards' / (identity + '.json'))
    assert guard['passed'] and guard['exit_code'] == 0 and not guard['resource_guard_aborted']
    assert guard['report_sha256'] == sha(m)
    proofs[identity], metric_reports[identity] = proof, report
    for ratio, values in report['independent_strict_metrics'].items():
        ci = proof['native_validation_control']['strict'][ratio]['f1_ci95']
        cross_rows.append(dict(canonical_id=identity, training_seed=job['seed'], validation_alarm_percent=ratio,
            threshold=values['threshold'], **values['micro'], **values['event_metrics'],
            source_conditional_block_f1_CI95_low=ci['low'], source_conditional_block_f1_CI95_high=ci['high'],
            source_bootstrap_repeats=ci['resamples'], bootstrap_independently_recomputed=False,
            point_adjustment=False, strict_TAB_result=False))
cross_cohorts = cross.cohorts(cq['jobs'], proofs)
assert cross_cohorts['complete_primary_cohorts'] == 0
assert cross_cohorts['primary_cohorts']['GECCO/main']['verified_seed_ids'] == [1103, 1104, 1105, 1106]
assert 'metrics' not in cross_cohorts['primary_cohorts']['GECCO/main']
new_cross = 'crossad_author__fresh_train__gecco__seed1106'
new_job = next(j for j in cq['jobs'] if j['canonical_CPU_condition_reference'] == new_cross)
new_state = ROOT / cq['state_root'] / 'jobs' / new_cross
result = read(ROOT / new_job['output_directory'] / 'result.json')
assert result['details']['epochs_completed'] == 20 and result['details']['training_forward_calls'] == 8640
assert result['details']['native_points'] == dict(train=55408, validation=13852, test=69261)

sc = read(ROOT / spectral.CONFIG)
cases, _ = spectral.verify(sc)
ss = read(ROOT / sc['state_root'] / 'status.json')
sid = 'spectral_original__diffusion_ts__sines__paper_sines_distribution__seed45'
item = ss['verified'][sid]
sp = ROOT / item['proof']
sv = read(sp)
assert sha(sp) == item['proof_sha256'] and sv['passed'] and sv['canonical_id'] == sid
assert sv['original_optimizer_updates'] == 12000 and sv['original_training_minibatches'] == 24000
assert sv['actual_model_parameters'] == 235249
assert sv['complete_reference_shape'] == [10000, 24, 5] and sv['complete_generated_shape'] == [10005, 24, 5]
assert sv['actual_full_original_checkpoint_tensors_restored_bitwise'] and sv['all_original_generated_arrays_finite']
assert sv['complete_original_reference_array_bitwise_verified']
assert sv['all_four_five_repeat_evaluator_statistics_independently_recomputed']
assert not sv['generative_model_reinferred_by_this_audit'] and not sv['learned_evaluators_independently_retrained']
sgp = ROOT / sc['state_root'] / 'attempts' / sid / 'guard_receipt.json'
sg = read(sgp)
assert sg['passed'] and sg['exit_code'] == 0 and not sg['resource_guard_aborted'] and sg['proof_sha256'] == sha(sp)
case = cases[sid]
sj = case['job']
sq = read(ROOT / case['queue'])
srp = ROOT / sj['output_directory'] / 'result.json'
stp = ROOT / sq['state_root'] / 'receipts' / (sid + '.json')
sr, st = read(srp), read(stp)
assert sr['job'] == sj and sha(srp) == sv['original_result_sha256'] and sha(stp) == sv['actual_terminal_receipt_sha256']
assert st['exitcode'] == 0 and st['full_result_validated'] and not st['resource_guard_aborted']
validation = read(sp.parent / 'validation.json')
assert validation['passed'] and validation['outputs']['independent_full_native_artifact_audit.json'] == sha(sp)
for pin in validation['source_receipts']:
    assert sha(ROOT / pin['path']) == pin['sha256']
prior_path = ROOT / 'projects/flow_matching_research/results/2026-10-16/Spectral_PaperSines44_full32_source_artifact_metrics_v1/unique32_full_source_and_artifact_model_catalog.json'
prior = read(prior_path)
assert len(prior['records']) == 32
packet.mkdir(parents=True)
(packet / '.gitattributes').write_text('*.json -text\n*.xml -text -whitespace\n*.csv -text\n*.py -text\n', encoding='utf-8')
spectral_dir = packet / 'Spectral_new_complete_seed45'
spectral_dir.mkdir()
for label, path in dict(source_result=srp, source_terminal_guard=stp, artifact_proof=sp,
    artifact_review_guard=sgp, review_validation=sp.parent / 'validation.json').items():
    shutil.copyfile(path, spectral_dir / (label + '.json'))
new = dict(canonical_id=sid, model=sj['model'], dataset=sj['dataset'], seed=sj['seed'],
    optimizer_updates=12000, training_minibatches=24000, complete_reference_shape=sv['complete_reference_shape'],
    complete_generated_shape=sv['complete_generated_shape'], model_parameters=235249, metrics=sv['metrics'],
    original_result_sha256=sha(srp), artifacts=sv['artifacts'],
    full_native_proof=(spectral_dir / 'artifact_proof.json').relative_to(ROOT).as_posix(),
    full_native_proof_sha256=sha(sp), original_runtime_proof_path=str(sp),
    independently_retrained_learned_evaluators=False, generative_output_reinferred=False)
records = deepcopy(prior['records']) + [new]
assert records[:32] == prior['records'] and len({r['canonical_id'] for r in records}) == 33
groups = defaultdict(list)
for row in records:
    actual = read(ROOT / row['full_native_proof'])
    assert sha(ROOT / row['full_native_proof']) == row['full_native_proof_sha256'] and actual['passed']
    assert actual['canonical_id'] == row['canonical_id'] and actual['metrics'] == row['metrics']
    source_job = cases[row['canonical_id']]['job']
    assert sha(ROOT / source_job['output_directory'] / 'result.json') == row['original_result_sha256']
    for artifact in row['artifacts']:
        assert sha(ROOT / artifact['path']) == artifact['sha256']
    groups[(row['model'], row['dataset'], source_job['data_track'])].append(
        dict(actual, model=row['model'], dataset=row['dataset'], data_track=source_job['data_track']))
summaries = [summarize_cohort(values) for _, values in sorted(groups.items())]
sines = next(v for v in summaries if v['dataset'] == 'sines' and v['data_track'] == 'paper_sines_distribution')
assert sines['independently_reviewed_training_seeds'] == [42, 43, 44, 45]
assert sines['missing_original_training_seeds'] == [46] and not sines['full_five_seed_cohort_complete']
assert all(v['full_cohort_mean'] is None and v['between_training_model_CI95'] is None for v in sines['metrics'].values())
write(packet / 'Spectral_unique33_full_source_artifact_model_catalog.json', dict(records=records,
    full_source_and_artifact_reviewed_models=33, required_original_models=70, previous32_records_unchanged=True,
    previous_catalog_sha256=sha(prior_path), independent_generative_reexecution_certified=False,
    learned_evaluator_retraining_certified=False, all_original_experiments_complete=False))
write(packet / 'Spectral_all33_cohort_completion_and_metrics.json', dict(cohorts=summaries, all_original_experiments_complete=False))
with (packet / 'Spectral_all33_models_132_metric_rows.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['canonical_id', 'method', 'dataset', 'data_track', 'training_seed', 'metric',
                     'five_evaluator_mean', 'within_model_evaluator_SD', 'within_model_evaluator_t95_low', 'within_model_evaluator_t95_high'])
    for row in records:
        for key, v in row['metrics'].items():
            writer.writerow([row['canonical_id'], row['model'], row['dataset'], cases[row['canonical_id']]['job']['data_track'],
                row['seed'], key, v['mean'], v['std'], v['ci95_low'], v['ci95_high']])
buf = io.StringIO(newline='')
writer = csv.DictWriter(buf, fieldnames=list(cross_rows[0]))
writer.writeheader()
writer.writerows(cross_rows)
(packet / 'CrossAD_all4_models_12_strict_metric_rows.csv').write_bytes(buf.getvalue().encode('utf-8-sig'))
write(packet / 'CrossAD_all22_primary_cohort_completion.json', cross_cohorts)
for identity in sorted(expected):
    out = packet / 'CrossAD_all_complete_conditions' / identity
    out.mkdir(parents=True)
    shutil.copyfile(Path(parent['artifact_root']) / 'conditions' / (identity + '.json'), out / 'full_checkpoint_array_proof.json')
    shutil.copyfile(Path(mc['artifact_root']) / 'conditions' / (identity + '.json'), out / 'independent_metric_report.json')
out = packet / 'CrossAD_new_complete_seed1106'
out.mkdir()
for label, path in dict(source_result=ROOT / new_job['output_directory'] / 'result.json',
    training_trace=ROOT / new_job['output_directory'] / 'training_trace.json',
    native_replay=Path(cq['artifact_root']) / new_cross / 'native_replay/report.json',
    **{phase + '_guard': new_state / (phase + '.guard.json') for phase in ['CUDA_probe', 'CPU_reference', 'author', 'replay']}).items():
    shutil.copyfile(path, out / (label + '.json'))
shutil.copyfile(tests, packet / 'validation_56_checks.xml')
shutil.copyfile(__file__, packet / 'closure_source.py')
handles = []
for state in [cs, ms, ss]:
    p = psutil.Process(state['pid'])
    assert p.create_time() == state['create_time'] and p.is_running()
    handles.append(dict(pid=p.pid, create_time=p.create_time(), command=p.cmdline()))
write(packet / 'actual_live_result_collector_handles.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), handles=handles))
write(packet / 'iteration_status.json', dict(original_scope=cq['original_scope'],
    CrossAD_full_CUDA_verified=4, CrossAD_required_conditions=139, CrossAD_GECCO_missing_primary_training_seeds=[1107],
    Spectral_full_source_artifact_models=33, Spectral_required_conditions=70,
    Spectral_paper_Sines_missing_training_seeds=[46], new_CrossAD_full_GPU_models=1,
    new_Spectral_full_source_artifact_models=1, new_independent_generative_reexecution_conditions=0,
    original_CPU_CrossAD_completion_credit=0, all_original_experiments_complete=False))
(packet / 'README.md').write_text('''# 第四个CrossAD完整GPU模型与第33个Spectral完整源实验

CrossAD GECCO seed1106完整20轮、8640个训练批次及原始评估完成，四项CPU/GPU初始一致性、源执行、全量检查点重放守卫通过。
55408个训练点、13852个验证点、69261个测试点全部评分重放逐字节一致，独立标准库实现重算所有三个验证阈值及严格逐点指标通过。
验证集1%阈值、无PA：F1=0.39925373134328357，Precision=0.6257309941520468，Recall=0.29315068493150687，ROC-AUC=0.986797295814333，AP=0.5473016910200021。
当前4/139个完整GPU条件通过，GECCO主要训练种子1103..1106齐备，仅缺1107；不计算4种子均值或完整五种子区间。原CPU协议不抵扣。
CSV汇总全部4模型、3阈值的12行指标，区间沿用原完整源条件块bootstrap，未由独立标准库核算重做。

Diffusion-TS论文分布Sines seed45完成12000次优化更新、24000训练小批次、10005×24×5完整生成及四类各五次原始评估；原始执行和独立产物审计均退出0且无资源中止。
seed45五次评估均值：Context-FID=0.10500461065756453，Correlational=0.0061803575388997695，Discriminative=0.17855，Predictive=0.31833012758503054。
全部33个已闭合源实验的结果、证明、检查点/参考/生成数组哈希重新检查，前32条记录保持不变，CSV含132行指标。
论文分布Sines种子42..45齐备，仅缺46，不汇报4种子均值。五次评估不是五个训练模型。
该Spectral阶段为原完整源执行加产物审计，独立生成/学习式评估器重跑另由已启动全70条件队列执行，当前仍未计入新的重跑完成数。

56项相关回归检查通过。全部45篇论文、681条方法/基线/消融、273条原始数据任务、702条标量目标及原论文/TAB/新数据轨道仍在范围内；整体实验尚未完成。
''', encoding='utf-8')
write(packet / 'packet_integrity.json', dict(files=[dict(path=p.relative_to(packet).as_posix(), sha256=sha(p))
    for p in sorted(packet.rglob('*')) if p.is_file()]))
ara = ROOT / 'projects/flow_matching_research/ara/execution/CrossAD4_Spectral33_full_results.2026-10-16_v1.json'
assert not ara.exists()
write(ara, dict(result_packet=packet.relative_to(ROOT).as_posix(), CrossAD_configuration=cold.CONFIG,
    Spectral_configuration=spectral.CONFIG, new_CrossAD_complete_condition=new_cross,
    new_Spectral_complete_source_condition=sid, original_scope=cq['original_scope'],
    CrossAD_completed_full_GPU_conditions=4, Spectral_complete_source_artifact_conditions=33,
    all_original_experiments_complete=False))
def head(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD']).decode().strip()
paths = [p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()] + [ara.relative_to(ROOT).as_posix()]
manifest = ROOT / 'tmp/CrossAD4_Spectral33_publication_manifest_20261016_v1.json'
write(manifest, dict(previous_root=head(ROOT), previous_parent=head(ROOT.parent / 'flow-matching-research'), paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet), files=len(paths), CrossAD_conditions=4, Spectral_conditions=33)))
