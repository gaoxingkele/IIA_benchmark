from collections import defaultdict
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import shutil

import psutil

from scripts.flow_matching.prepare_spectral_original_data import ROOT, read, sha
from scripts.flow_matching import audit_spectral_completed_cohorts_v1 as original_cohort
from scripts.flow_matching.audit_spectral_completed_cohorts_v2 import validate_selection
original_cohort.validate_selection = validate_selection
registration = original_cohort.registration
from scripts.flow_matching.summarize_spectral_full_cohorts_v1 import summarize_cohort

BASE = 'projects/flow_matching_research/results/2026-10-16/spectral_after29_PaperSines42_full30_native_model_increment_v1'
CONFIG = 'configs/reproducibility/fm_spectral_after29_PaperSines42_full30_native_increment.2026-10-16_v1.json'
PLAN = 'configs/reproducibility/fm_spectral_sines_paper_seed42_full_native_review.2026-10-16_v1.json'
PREVIOUS = 'projects/flow_matching_research/results/2026-10-15/spectral_after28_Sines46_full29_native_model_increment_v1/unique29_full_native_model_catalog.json'
PREVIOUS_CONFIG = 'configs/reproducibility/fm_spectral_after28_Sines46_full29_native_increment.2026-10-15_v1.json'
PREV_COUNT, NEW_COUNT = 29, 30
previous = read(ROOT / PREVIOUS)
assert previous['full_native_generators'] == len(previous['records']) == PREV_COUNT
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
assert proof['canonical_id'] == 'spectral_original__diffusion_ts__sines__paper_sines_distribution__seed42'
assert proof['passed'] and not proof['cuda_context_initialized']
assert proof['original_optimizer_updates'] == 12000 and proof['original_training_minibatches'] == 24000
assert proof['complete_reference_shape'] == [10000, 24, 5] and proof['complete_generated_shape'] == [10005, 24, 5]
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
assert len(records) == len({row['canonical_id'] for row in records}) == NEW_COUNT
assert records[:PREV_COUNT] == previous['records']
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
assert mujoco['independently_reviewed_training_seeds'] == [42, 43, 44, 45, 46]
assert mujoco['full_five_seed_cohort_complete']
assert all(values['full_cohort_mean'] is not None and values['between_training_model_CI95'] is not None and values['degrees_of_freedom'] == 4 and values['independent_training_models'] == 5 and values['evaluator_runs'] == 25 for values in mujoco['metrics'].values())
# Independent arithmetic on all25 evaluator records per metric; do not pool25 as25model seeds.
import math
from scipy.stats import t as student_t
for metric, row in mujoco['metrics'].items():
    means=[math.fsum(item['five_evaluator_repetitions'])/5 for item in row['per_training_model']]
    assert len(means)==5
    mean=math.fsum(means)/5
    std=math.sqrt(math.fsum((value-mean)**2 for value in means)/4)
    ci=student_t.interval(.95,4,loc=mean,scale=std/math.sqrt(5))
    for actual, expected in [(mean,row['full_cohort_mean']),(std,row['between_training_model_sample_STD']),*zip(ci,row['between_training_model_CI95'])]:
        assert math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12)

sines = next(row for row in summaries if row['dataset'] == 'sines' and row['data_track'] == 'released_author_data')
assert sines['independently_reviewed_training_seeds'] == [42, 43, 44, 45, 46]
assert sines['full_five_seed_cohort_complete']
assert all(v['full_cohort_mean'] is not None and v['between_training_model_CI95'] is not None
    and v['independent_training_models']==5 and v['evaluator_runs']==25 and v['degrees_of_freedom']==4
    for v in sines['metrics'].values())
# Independent scalar sample-variance and analytic df4 critical value; five model
# means are the units. The 25 evaluator runs never become 25 training seeds.
for metric,row in sines['metrics'].items():
    means=[math.fsum(item['five_evaluator_repetitions'])/5 for item in row['per_training_model']]
    assert len(means)==5
    mean=math.fsum(means)/5
    std=math.sqrt(math.fsum((x-mean)**2 for x in means)/4)
    halfwidth=2.7764451051977987*std/math.sqrt(5)
    assert math.isclose(mean,row['full_cohort_mean'],rel_tol=1e-12,abs_tol=1e-12)
    assert math.isclose(std,row['between_training_model_sample_STD'],rel_tol=1e-12,abs_tol=1e-12)
    assert all(math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12) for x,y in
        zip([mean-halfwidth,mean+halfwidth],row['between_training_model_CI95']))

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
save(BASE + '/unique30_full_native_model_catalog.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    full_native_generators=NEW_COUNT, original_registered_generation_jobs=70,
    newly_independently_artifact_reviewed_models=1, reused_previous_full_native_reviews=PREV_COUNT,
    previous29_closed_records_unchanged=True, previous_closed_catalog=PREVIOUS, previous_closed_catalog_sha256=sha(ROOT / PREVIOUS),
    original_total_optimizer_updates=sum(row['optimizer_updates'] for row in records),
    original_total_training_minibatches=sum(row['training_minibatches'] for row in records),
    original_total_full_evaluator_runs=NEW_COUNT * 4 * 5, records=records,
    full_generative_reinference_certified=False, learned_evaluator_retraining_certified=False,
    author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False))
save(BASE + '/seven_cohort_completion_and_per_model_metrics.json', dict(cohorts=summaries,
    required_training_seeds=[42, 43, 44, 45, 46], all_original_experiments_complete=False,
    boundary='Paper-distribution Sines42 is a separate1/5 cohort;43..46 remain required. Sines released-track42,43,44,45,46 are5/5 original generator models; all full-five-model means, between-model sample STD and df4 t CI95 independently verified. The distinct paper_sines_distribution track remains in the original70 registry. MuJoCo42/43/44/45/46 full-five-model evaluator means, between-model STD and df4 t CI95 are complete. Evaluator repetitions remain nested within each trained model. fMRI43 remains required; Stocks/Energy/ETTh full-five-seed summaries remain unchanged. Artifact/statistic reproduction does not independently retrain evaluators or reinfer generative samples, and does not certify author-equivalent or strictTAB detection performance.'))
for pin in read(ROOT / CONFIG)['source_receipts']:
    assert sha(ROOT / pin['path']) == pin['sha256']
save(BASE + '/closure_validation.json', dict(passed=True, all30_canonical_ids_unique=True,
    every_current_native_proof_result_checkpoint_generated_reference_hash_rechecked=True,
    previous29_closed_records_unchanged=True, new_original_budget_and_terminal_guard_verified=True,
    MuJoCo_full_five_model_mean_STD_CI95_verified=True, Sines_full_five_model_mean_STD_CI95_independently_verified=True,
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
lines = ['Full original generation models independently artifact-reviewed:30/70.', '',
    'Added Diffusion-TS paper-distribution Sines seed42. The previous29 source/result/weight/generated/reference/proof hashes were rechecked and their catalog records are unchanged.', '',
    'The new original model completed12,000 optimizer updates,24,000 training minibatches,10,000x24x5 reference rows and10,005x24x5 generated rows, with all four five-repeat evaluators. Native Py310 independently restored every model tensor and checked all EMA/optimizer tensors, full arrays and evaluator statistics. This audit does not reinfer generation or retrain learned evaluators.', '',
    '| New paper-distribution seed42 metric | Five-evaluator mean | Within-model sample STD |', '|---|---:|---:|']
for metric, values in proof['metrics'].items():
    lines.append(f"|{metric}|{values['mean']:.9g}|{values['std']:.9g}|")
paper = next(x for x in summaries if x['dataset']=='sines' and x['data_track']=='paper_sines_distribution')
assert paper['independently_reviewed_training_seeds']==[42]
assert paper['missing_original_training_seeds']==[43,44,45,46]
assert not paper['full_five_seed_cohort_complete']
assert all(x['full_cohort_mean'] is None and x['between_training_model_CI95'] is None for x in paper['metrics'].values())
lines += ['', 'Paper-distribution Sines has1/5 training seeds;43..46 remain required. The released-Sines5/5 cohort remains separate. These are different frozen frequency/phase distributions; neither can replace the other or be pooled for a five-seed interval. Evaluator repetitions never count as training seeds.', '',
    'Full30-model total:467,000 optimizer updates,934,000 training minibatches,600 evaluator runs. fMRI43 and all other uncompleted original70 slots remain required. Source parameter count235,249 differs from paper232,177; this is recorded, not silently altered. These are generative distribution-reconstruction metrics, not anomaly-detection F1 or strictTAB results. Original45 papers,681 method/baseline/ablation records,273 data tasks and702 paper scalar targets remain in scope.']
with (folder/'README.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(dict(full_native_models=NEW_COUNT,new_models=1,required_original_models=70,paper_Sines_training_seeds=[42],all_original_experiments_complete=False)))
