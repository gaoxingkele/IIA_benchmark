from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET

import psutil

from scripts.flow_matching import run_spectral_full70_artifact_reviews_v1 as collector
from scripts.flow_matching.summarize_spectral_full_cohorts_v1 import summarize_cohort
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

identity = 'spectral_original__diffusion_ts__sines__paper_sines_distribution__seed44'
prior_path = ROOT / 'projects/flow_matching_research/results/2026-10-16/spectral_PaperSines43_full31_and_CITE_admission_v2/unique31_full_source_and_artifact_model_catalog.json'
prior = read(prior_path)
assert len(prior['records']) == prior['full_source_and_artifact_reviewed_models'] == 31
c = read(ROOT / collector.CONFIG)
cases, _ = collector.verify(c)
state_path = ROOT / c['state_root'] / 'status.json'
state = read(state_path)
owner = psutil.Process(state['pid'])
assert owner.create_time() == state['create_time'] and owner.is_running()
assert identity in state['verified'] and not state['failed_preserved']
entry = state['verified'][identity]
proof_path = ROOT / entry['proof']
proof = read(proof_path)
assert proof['passed'] and sha(proof_path) == entry['proof_sha256']
assert proof['canonical_id'] == identity and proof['seed'] == 44
assert proof['original_optimizer_updates'] == 12000 and proof['original_training_minibatches'] == 24000
assert proof['actual_model_parameters'] == 235249 and proof['paper_reported_parameters'] == 232177
assert proof['complete_reference_shape'] == [10000, 24, 5] and proof['complete_generated_shape'] == [10005, 24, 5]
for key in ['actual_full_original_checkpoint_tensors_restored_bitwise',
            'complete_original_reference_array_bitwise_verified',
            'all_original_generated_arrays_finite',
            'all_four_five_repeat_evaluator_statistics_independently_recomputed']:
    assert proof[key]
assert not proof['cuda_context_initialized'] and not proof['generative_model_reinferred_by_this_audit']
assert not proof['learned_evaluators_independently_retrained']
case = cases[identity]
job = case['job']
queue = read(ROOT / case['queue'])
result_path = ROOT / job['output_directory'] / 'result.json'
terminal_path = ROOT / queue['state_root'] / 'receipts' / (identity + '.json')
result, terminal = read(result_path), read(terminal_path)
assert result['job'] == job and result['optimizer_updates'] == 12000 and result['training_minibatches'] == 24000
assert sha(result_path) == proof['original_result_sha256'] and sha(terminal_path) == proof['actual_terminal_receipt_sha256']
assert terminal['exitcode'] == 0 and terminal['full_result_validated'] and not terminal['resource_guard_aborted']
guard_path = ROOT / c['state_root'] / 'attempts' / identity / 'guard_receipt.json'
guard = read(guard_path)
assert guard['passed'] and guard['exit_code'] == 0 and not guard['resource_guard_aborted']
assert guard['proof_sha256'] == sha(proof_path)
validation_path = proof_path.parent / 'validation.json'
validation = read(validation_path)
assert validation['passed'] and validation['outputs']['independent_full_native_artifact_audit.json'] == sha(proof_path)
for pin in validation['source_receipts']:
    assert sha(ROOT / pin['path']) == pin['sha256'], pin['path']
test_path = Path('F:/aicoding/IIA_Data/experiments/Spectral_PaperSines44_full32_artifact_closure_20261016_v1_tests.xml')
suites = list(ET.parse(test_path).getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 41
assert not any(int(s.get(k, 0)) for s in suites for k in ('failures', 'errors', 'skipped'))

packet = ROOT / 'projects/flow_matching_research/results/2026-10-16/Spectral_PaperSines44_full32_source_artifact_metrics_v1'
assert not packet.exists()
packet.mkdir(parents=True)
(packet / '.gitattributes').write_text('*.json -text\n*.xml -text -whitespace\n*.csv -text\n*.py -text\n', encoding='utf-8')
sources = dict(new_original_result=result_path, new_source_terminal_guard=terminal_path,
    new_independent_artifact_proof=proof_path, new_artifact_review_guard=guard_path,
    new_review_validation=validation_path,
    new_review_configuration=ROOT / c['configuration_root'] / (identity + '.json'))
for name, path in sources.items():
    shutil.copyfile(path, packet / (name + '.json'))
new = dict(canonical_id=identity, model=job['model'], dataset=job['dataset'], seed=job['seed'],
    optimizer_updates=12000, training_minibatches=24000,
    complete_reference_shape=proof['complete_reference_shape'], complete_generated_shape=proof['complete_generated_shape'],
    model_parameters=235249, metrics=proof['metrics'], original_result_sha256=sha(result_path),
    artifacts=proof['artifacts'], full_native_proof=(packet / 'new_independent_artifact_proof.json').relative_to(ROOT).as_posix(),
    full_native_proof_sha256=sha(proof_path), original_runtime_proof_path=str(proof_path),
    independently_retrained_learned_evaluators=False, generative_output_reinferred=False)
records = deepcopy(prior['records']) + [new]
assert len(records) == len({r['canonical_id'] for r in records}) == 32 and records[:31] == prior['records']
groups = defaultdict(list)
artifact_checks = []
for row in records:
    full = ROOT / row['full_native_proof']
    actual = read(full)
    assert sha(full) == row['full_native_proof_sha256'] and actual['passed']
    assert actual['canonical_id'] == row['canonical_id'] and actual['seed'] == row['seed']
    assert actual['metrics'] == row['metrics']
    source_job = cases[row['canonical_id']]['job']
    original_result = ROOT / source_job['output_directory'] / 'result.json'
    assert sha(original_result) == row['original_result_sha256']
    for artifact in row['artifacts']:
        assert sha(ROOT / artifact['path']) == artifact['sha256']
        artifact_checks.append(artifact)
    groups[(row['model'], row['dataset'], source_job['data_track'])].append(
        dict(actual, model=row['model'], dataset=row['dataset'], data_track=source_job['data_track']))
summaries = [summarize_cohort(v) for _, v in sorted(groups.items())]
sines = next(v for v in summaries if v['dataset'] == 'sines' and v['data_track'] == 'paper_sines_distribution')
assert sines['independently_reviewed_training_seeds'] == [42, 43, 44]
assert sines['missing_original_training_seeds'] == [45, 46]
assert not sines['full_five_seed_cohort_complete']
assert all(v['full_cohort_mean'] is None and v['between_training_model_CI95'] is None for v in sines['metrics'].values())
write(packet / 'unique32_full_source_and_artifact_model_catalog.json', dict(
    captured_utc=datetime.now(timezone.utc).isoformat(), records=records,
    full_source_and_artifact_reviewed_models=32, original_registered_generation_jobs=70,
    previous31_records_unchanged=True, previous_catalog=str(prior_path), previous_catalog_sha256=sha(prior_path),
    original_total_optimizer_updates=sum(v['optimizer_updates'] for v in records),
    original_total_training_minibatches=sum(v['training_minibatches'] for v in records),
    original_total_full_evaluator_runs=32 * 4 * 5,
    generative_reinference_certified=False, learned_evaluator_retraining_certified=False,
    historical_author_equivalence_certified=False, strict_TAB_result=False,
    all_original_experiments_complete=False))
write(packet / 'all32_cohort_completion_and_metrics.json', dict(cohorts=summaries,
    required_original_training_seeds=[42, 43, 44, 45, 46], all_original_experiments_complete=False))
with (packet / 'all32_model_evaluator_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['canonical_id', 'model', 'dataset', 'data_track', 'training_seed', 'metric',
        'five_evaluator_mean', 'within_model_evaluator_sample_STD', 'within_model_evaluator_t95_low', 'within_model_evaluator_t95_high'])
    for row in records:
        for metric, value in row['metrics'].items():
            writer.writerow([row['canonical_id'], row['model'], row['dataset'], cases[row['canonical_id']]['job']['data_track'],
                row['seed'], metric, value['mean'], value['std'], value['ci95_low'], value['ci95_high']])
write(packet / 'all32_actual_artifact_hash_checks.json', dict(all_original_artifacts_checked=artifact_checks,
    source_results_checked=32, independent_proofs_checked=32))
shutil.copyfile(test_path, packet / 'validation_41_checks.xml')
shutil.copyfile(state_path, packet / 'actual_full70_artifact_collector_state.json')
shutil.copyfile(__file__, packet / 'closure_source.py')
write(packet / 'iteration_status.json', dict(original_scope=dict(papers=45, method_baseline_ablation_records=681,
    original_data_task_records=273, paper_scalar_targets=702), full_source_and_artifact_models=32,
    required_source_models=70, paper_distribution_Sines_completed_training_seeds=[42, 43, 44],
    paper_distribution_Sines_missing_training_seeds=[45, 46], new_complete_source_artifact_records=1,
    independent_generation_and_evaluator_reexecution_complete=False, all_original_experiments_complete=False))
(packet / 'README.md').write_text('''# Spectral：第32个完整源实验与产物核验

新增Diffusion-TS论文分布Sines seed44：完整12000次优化更新、24000个训练小批次，235249个参数；原论文列出的参数量为232177，历史等价性未认证。
原始源任务退出码0、无资源中止，完成10000×24×5参考样本、10005×24×5生成样本，以及四类指标各5次完整评估。
独立产物审计核对完整检查点模型/EMA/优化器、全部参考/生成数组与评估统计，审计退出码0。

| seed44指标 | 五次评估均值 | 五次评估样本标准差 |
| --- | --- | --- |
| Context-FID | 0.066381950363761 | 0.0033224587118370003 |
| Correlational | 0.005297215796915046 | 0.001816009530038946 |
| Discriminative | 0.1144 | 0.04445243244188108 |
| Predictive | 0.3183208145619315 | 0.000011288842264181163 |

五次评估属于一个训练模型。论文分布Sines仅完成训练种子42/43/44，45/46仍需完成；不计算部分种子组的完整均值或种子置信区间。
发布32个完整源任务的128行指标CSV及各数据轨道的种子覆盖，前31条记录保持不变，并重新核对全部32个源结果、独立证明与原始产物哈希。
41项审计及完整种子统计回归检查通过。
此阶段没有重新执行生成模型或重新训练评估器；原始采样起始RNG缺失，检查点与数组核验不能代替独立生成重放。
整体45篇论文、681条方法/基线/消融、273条原始数据任务、702条标量目标与全部原论文/TAB/新数据轨道仍保留，所有实验尚未完成。
''', encoding='utf-8')
write(packet / 'packet_integrity.json', dict(files=[dict(path=p.relative_to(packet).as_posix(), sha256=sha(p))
    for p in sorted(packet.rglob('*')) if p.is_file()]))
ara = ROOT / 'projects/flow_matching_research/ara/execution/Spectral_PaperSines44_full32.2026-10-16_v1.json'
assert not ara.exists()
write(ara, dict(source_queue=case['queue'], source_model=job['model_config'],
    result_packet=packet.relative_to(ROOT).as_posix(), completed_source_artifact_models=32,
    required_source_models=70, independently_reexecuted_generation=False,
    new_original_training_seed_slots=0, all_original_experiments_complete=False))
paths = [p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()] + [ara.relative_to(ROOT).as_posix()]
def head(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD']).decode().strip()
manifest = ROOT / 'tmp/Spectral_PaperSines44_full32_publication_manifest_20261016_v1.json'
write(manifest, dict(previous_root=head(ROOT), previous_parent=head(ROOT.parent / 'flow-matching-research'), paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet), catalog_records=len(records), metrics_rows=32*4,
    full_five_seed_cohorts=sum(v['full_five_seed_cohort_complete'] for v in summaries), files=len(paths))))
