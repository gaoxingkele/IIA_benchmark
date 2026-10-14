from collections import defaultdict
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import shutil

import psutil

from scripts.flow_matching.prepare_spectral_original_data import ROOT, read, sha
from scripts.flow_matching.audit_spectral_completed_cohorts_v1 import registration
from scripts.flow_matching.summarize_spectral_full_cohorts_v1 import summarize_cohort

BASE = 'projects/flow_matching_research/results/2026-10-14/spectral_after20_MuJoCo43_full21_native_model_increment_v1'
CONFIG = 'configs/reproducibility/fm_spectral_after20_MuJoCo43_full21_native_increment.2026-10-14_v1.json'
PLAN = 'configs/reproducibility/fm_spectral_mujoco_seed43_full_native_review.2026-10-14_v1.json'
PREVIOUS = 'projects/flow_matching_research/results/2026-10-14/spectral_full20_native_model_catalog_and_paper_comparison_v1/unique20_full_native_model_catalog.json'
PREVIOUS_CONFIG = 'configs/reproducibility/fm_spectral_full20_native_catalog_and_comparison.2026-10-14_v1.json'
previous = read(ROOT / PREVIOUS)
assert previous['full_native_generators'] == len(previous['records']) == 20
for pin in read(ROOT / PREVIOUS_CONFIG)['source_receipts']:
    assert sha(ROOT / pin['path']) == pin['sha256'], pin['path']
plan = read(ROOT / PLAN)
registration(plan)
receipt_path = plan['state_root'] + '/receipt.json'
receipt = read(ROOT / receipt_path)
assert receipt['passed'] and receipt['full_native_artifact_audits_completed'] == len(receipt['records']) == 1
worker = receipt['records'][0]
assert worker['native_exit_code'] == 0 and worker['full_native_artifact_audit_passed'] and not worker['resource_guard_aborted']
proof_path = Path(worker['proof_path']).relative_to(ROOT).as_posix()
assert sha(ROOT / proof_path) == worker['proof_sha256']
proof = read(ROOT / proof_path)
assert proof['canonical_id'] == 'spectral_original__diffusion_ts__mujoco__released_author_data__seed43'
assert proof['passed'] and not proof['cuda_context_initialized']
assert proof['original_optimizer_updates'] == 14000 and proof['original_training_minibatches'] == 28000
assert proof['complete_reference_shape'] == [10000, 24, 14] and proof['complete_generated_shape'] == [10005, 24, 14]
assert proof['actual_full_original_checkpoint_tensors_restored_bitwise']
assert proof['complete_original_reference_array_bitwise_verified'] and proof['all_original_generated_arrays_finite']
assert proof['all_four_five_repeat_evaluator_statistics_independently_recomputed']
assert not proof['learned_evaluators_independently_retrained'] and not proof['generative_model_reinferred_by_this_audit']
queue = read(ROOT / plan['original_queue'])
jobs = {job['id']: job for job in queue['jobs']}
job = jobs[proof['canonical_id']]
result_path = job['output_directory'] + '/result.json'
terminal_path = queue['state_root'] + '/receipts/' + job['id'] + '.json'
assert sha(ROOT / result_path) == proof['original_result_sha256']
assert sha(ROOT / terminal_path) == proof['actual_terminal_receipt_sha256']
terminal = read(ROOT / terminal_path)
assert terminal['exitcode'] == 0 and terminal['full_result_validated'] and not terminal['resource_guard_aborted']
new = dict(canonical_id=job['id'], model=job['model'], dataset=job['dataset'], seed=job['seed'],
    optimizer_updates=proof['original_optimizer_updates'], training_minibatches=proof['original_training_minibatches'],
    complete_reference_shape=proof['complete_reference_shape'], complete_generated_shape=proof['complete_generated_shape'],
    model_parameters=proof['actual_model_parameters'], metrics=proof['metrics'],
    original_result_sha256=proof['original_result_sha256'], artifacts=proof['artifacts'],
    full_native_proof=proof_path, full_native_proof_sha256=sha(ROOT / proof_path),
    independently_retrained_learned_evaluators=False, generative_output_reinferred=False)
records = copy.deepcopy(previous['records']) + [new]
assert len(records) == len({row['canonical_id'] for row in records}) == 21
assert records[:20] == previous['records']
pins = [PLAN, PREVIOUS, PREVIOUS_CONFIG, receipt_path, worker['configuration'], proof_path,
    str(Path(proof_path).parent / 'validation.json').replace('\\', '/'), result_path, terminal_path,
    'scripts/flow_matching/summarize_spectral_full_cohorts_v1.py', 'tests/test_fm_spectral_full_cohort_summary_v1.py']
grouped = defaultdict(list)
for row in records:
    native = read(ROOT / row['full_native_proof'])
    assert sha(ROOT / row['full_native_proof']) == row['full_native_proof_sha256']
    assert native['passed'] and native['canonical_id'] == row['canonical_id'] and native['seed'] == row['seed']
    assert native['metrics'] == row['metrics']
    for artifact in row['artifacts']:
        assert sha(ROOT / artifact['path']) == artifact['sha256']
    original_job = jobs[row['canonical_id']]
    original_path = original_job['output_directory'] + '/result.json'
    assert sha(ROOT / original_path) == row['original_result_sha256']
    grouped[(row['model'], row['dataset'], original_job['data_track'])].append(
        dict(native, model=row['model'], dataset=row['dataset'], data_track=original_job['data_track']))
    pins.extend([row['full_native_proof'], original_path, *[artifact['path'] for artifact in row['artifacts']]])
summaries = [summarize_cohort(proofs) for key, proofs in sorted(grouped.items())]
mujoco = next(row for row in summaries if row['dataset'] == 'mujoco')
assert mujoco['independently_reviewed_training_seeds'] == [42, 43]
assert not mujoco['full_five_seed_cohort_complete']
assert all(values['full_cohort_mean'] is None and values['between_training_model_CI95'] is None for values in mujoco['metrics'].values())
folder = ROOT / BASE
folder.mkdir(parents=True, exist_ok=False)
shutil.copyfile(__file__, folder / 'closure_source.py')
sources = dict(new_original_result=ROOT / result_path, new_terminal_guard=ROOT / terminal_path,
    new_independent_native_artifact_proof=ROOT / proof_path, new_native_review_guard=ROOT / receipt_path)
for name, path in sources.items():
    shutil.copyfile(path, folder / (name + '.json'))
pins.append(BASE + '/closure_source.py')


def save(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


save(CONFIG, dict(original_queue=plan['original_queue'], previous_closed_catalog=PREVIOUS,
    new_original_job=job['id'], source_receipts=[dict(path=path, sha256=sha(ROOT / path)) for path in dict.fromkeys(pins)],
    original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)))
save(BASE + '/unique21_full_native_model_catalog.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    full_native_generators=21, original_registered_generation_jobs=70,
    newly_independently_artifact_reviewed_models=1, reused_previous_full_native_reviews=20,
    previous20_closed_records_unchanged=True, previous_closed_catalog=PREVIOUS, previous_closed_catalog_sha256=sha(ROOT / PREVIOUS),
    original_total_optimizer_updates=sum(row['optimizer_updates'] for row in records),
    original_total_training_minibatches=sum(row['training_minibatches'] for row in records),
    original_total_full_evaluator_runs=21 * 4 * 5, records=records,
    full_generative_reinference_certified=False, learned_evaluator_retraining_certified=False,
    author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False))
save(BASE + '/five_cohort_completion_and_per_model_metrics.json', dict(cohorts=summaries,
    required_training_seeds=[42, 43, 44, 45, 46], all_original_experiments_complete=False,
    boundary='MuJoCo42/43 per-model evaluator measurements are retained; full-five-seed mean/STD/CI remain null. fMRI43 and MuJoCo44..46 remain required. Stocks/Energy/ETTh full-five-seed summaries remain unchanged. Artifact/statistic reproduction does not independently retrain evaluators or reinfer generative samples, and does not certify author-equivalent or strictTAB detection performance.'))
for pin in read(ROOT / CONFIG)['source_receipts']:
    assert sha(ROOT / pin['path']) == pin['sha256']
save(BASE + '/closure_validation.json', dict(passed=True, all21_canonical_ids_unique=True,
    every_current_native_proof_result_checkpoint_generated_reference_hash_rechecked=True,
    previous20_closed_records_unchanged=True, new_original_budget_and_terminal_guard_verified=True,
    MuJoCo_partial_two_model_full_cohort_statistics_remain_null=True,
    configuration=CONFIG, configuration_sha256=sha(ROOT / CONFIG), all_original_experiments_complete=False))
handles = []
for process in psutil.process_iter(['pid', 'name']):
    try:
        command = process.cmdline()
        if any('flow_matching' in argument for argument in command):
            cpu = process.cpu_times()
            handles.append(dict(pid=process.pid, create_time=process.create_time(), command=command,
                cpu_seconds=cpu.user + cpu.system))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
save(BASE + '/actual_remaining_full_scope_live_handles.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), actual_handles=handles))
lines = ['Full original generation models independently artifact-reviewed:21/70.', '',
    'Added one full original Diffusion-TS MuJoCo seed43 model; the previous20 closed canonical records remain unchanged. All14,000 optimizer updates/28,000 minibatches, full10,000x24x14 reference and10,005x24x14 generated arrays, complete native checkpoint tensors and4x5 evaluator statistics passed the frozen v1 CPU checks. Every current proof/result/checkpoint/generated/reference SHA256 was rechecked.', '',
    '| MuJoCo seed43 metric | Five-evaluator mean | SampleSTD within this model |', '|---|---:|---:|']
for metric, values in proof['metrics'].items():
    lines.append(f"|{metric}|{values['mean']:.9g}|{values['std']:.9g}|")
lines += ['', 'Evaluator repetitions describe one trained model. MuJoCo now has2/5 independently reviewed original models (42,43); seeds44..46 remain required and full-cohort statistics stay null. No new fitting, generative reinference or independent learned-evaluator retraining was performed for this closure. Original all-source scaling and reconstruction boundaries remain; author equivalence and strictTAB anomaly-detection performance are not certified.', '',
    'unique21_full_native_model_catalog.json contains every current model/dataset/seed metric and artifact/proof identity. five_cohort_completion_and_per_model_metrics.json preserves every evaluator repetition and the missing original seed obligations. The full45-paper/681-method-ablation/273-data-task objective remains active.']
with (folder / 'README.md').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write('\n'.join(lines) + '\n')
changed = [CONFIG, PLAN, worker['configuration']]
for base in [BASE, 'projects/flow_matching_research/results/2026-10-14/spectral_mujoco_seed43_full_native_review_v1']:
    changed.extend(path.relative_to(ROOT).as_posix() for path in (ROOT / base).rglob('*') if path.is_file())
save('tmp/spectral_MuJoCo43_full21_increment_publication_manifest_20261014_v1.json', dict(
    previous_root='756abb5fb777a22c2d108ee28ed95012c8c6f859', previous_parent='a166828f720f8dbc0424d956cccef525ad6d3b25', paths=sorted(set(changed))))
print(json.dumps(dict(full_native_models=21, new_models=1, optimizer_updates=sum(row['optimizer_updates'] for row in records),
    new_metrics={key: value['mean'] for key, value in proof['metrics'].items()}, exact_publication_files=len(set(changed)))))
