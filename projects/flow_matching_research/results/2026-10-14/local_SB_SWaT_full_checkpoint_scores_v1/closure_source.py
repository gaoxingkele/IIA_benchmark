import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import sha, read
from scripts.flow_matching.run_tsad_queue import complete_result

ROOT = Path(__file__).resolve().parents[1]
BASE = 'projects/flow_matching_research/results/2026-10-14/local_SB_SWaT_full_checkpoint_scores_v1'
CONFIG = 'configs/reproducibility/fm_local_SB_SWaT_full_checkpoint_audit.2026-10-14_v1.json'
JOB = 'fm_objective_sb__swat__seed1103__stable_numerics_v2'

def save(name, value):
    with (ROOT / BASE / name).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')

config = read(ROOT / CONFIG)
for pin in config['source_receipts']:
    assert sha(ROOT / pin['path']) == pin['sha256'], pin['path']
proof_path = config['output_directory'] + '/independent_full_model_score_verification.json'
proof = read(ROOT / proof_path)
guard_path = config['state_root'] + '/receipt.json'
guard = read(ROOT / guard_path)
assert guard['exit_code'] == 0 and guard['full_cold_model_replay_passed'] and not guard['resource_guard_aborted']
assert proof['passed'] and proof['canonical_id'] == JOB and proof['configuration_sha256'] == sha(ROOT / CONFIG)
assert proof['original_train_validation_test_native_scores_reinferred_bitwise']
assert proof['all_original_metrics_events_and_bootstrap_intervals_recomputed_exact']
assert proof['additional_fit_calls'] == proof['additional_optimizer_updates'] == 0
queue = read(ROOT / config['queue'])
assert len(queue['jobs']) == 105
job = next(item for item in queue['jobs'] if item['id'] == JOB)
assert complete_result(ROOT, job)
result_path = job['output_directory'] + '/result.json'
result = read(ROOT / result_path)
assert sha(ROOT / result_path) == proof['source_result_sha256']
assert sha(ROOT / job['output_directory'] / 'scores.npz') == result['scores_sha256'] == proof['source_scores_sha256']
manifest = read(ROOT / job['dataset_manifest_path'])
dataset = next(item for item in manifest['datasets'] if item['id'] == job['dataset'])
assert sha(ROOT / dataset['input_npz']) == dataset['input_sha256']
for pin in dataset['source_files'] + job['frozen_files']:
    assert sha(ROOT / pin['path']) == pin['sha256'], pin['path']
range_result = read(ROOT / config['range_evaluation'])
range_guard_path = 'experiments/runs/fm_tsad_tab_range_sparse_v2/logs/' + JOB + '/resource_usage.json'
range_guard = read(ROOT / range_guard_path)
assert not range_guard['resource_guard_aborted']
range_state = read(ROOT / 'experiments/runs/fm_tsad_tab_range_sparse_v2/status.json')
assert range_state['completed'][JOB]['evaluation_identity'] == range_result['evaluation_identity']
assert sha(ROOT / config['range_evaluation']) == proof['range_evaluation_sha256']
for name, path in dict(original_result=result_path, original_source_guard=config['terminal_receipt'],
        original_range_evaluation=config['range_evaluation'], original_range_guard=range_guard_path,
        independent_audit_guard=guard_path).items():
    shutil.copyfile(ROOT / path, ROOT / BASE / (name + '.json'))
shutil.copyfile(__file__, ROOT / BASE / 'closure_source.py')
save('frozen_dataset_manifest_entry.json', dataset)
primary = proof['primary_metrics']
save('complete_metrics_and_score_shift_diagnostics.json', dict(canonical_id=JOB,
    primary_validation_only_no_PA=primary, original_full_core_diagnostic=proof['tab_core_diagnostic'],
    VUS_entity_macro=proof['VUS_entity_macro'], strict_affiliation_entity_macro=proof['strict_affiliation_entity_macro'],
    score_distribution_diagnostics=proof['score_distribution_diagnostics'],
    measured_test_normal_false_alarm_fraction=proof['measured_test_normal_false_alarm_fraction'],
    full_native_split_coverage=proof['full_native_split_coverage'],
    frozen_validation_threshold=proof['frozen_validation_threshold'],
    parameter_count=result['parameter_count'], training_seconds=result['training_seconds'], score_seconds=result['score_seconds'],
    native_test_points=proof['full_native_test_points'], native_entities=proof['full_native_entities'], original_epochs=20,
    all_complete_checkpoint_tensors_and_scores_bitwise_replayed=True,
    primary_test_labels_used_for_calibration=False, additional_fit_or_optimizer_updates=0,
    original_author_TSAD_experiment_equivalence=False, strict_TAB_training_harness_equivalence=False,
    same_score_grid_diagnostic_is_not_detector_improvement=True,
    single_model_seed_not_complete_five_seed_cohort=True, all_original_experiments_complete=False))
save('closure_validation.json', dict(passed=True, captured_utc=datetime.now(timezone.utc).isoformat(),
    all_original_raw_processed_data_and_model_source_hashes_rechecked=True,
    full_native_checkpoint_and_timestamp_scores_bitwise_reinferred=True,
    full_original_core_event_bootstrap_metrics_recomputed_exact=True,
    complete_original_range_source_scope_macros_and_guard_success_verified=True,
    VUS_and_affiliation_integrals_independently_rerun=False,
    proof_sha256=sha(ROOT / proof_path), audit_guard_sha256=sha(ROOT / guard_path),
    all_original_experiments_complete=False))
model = read(ROOT / job['model_config'])
metrics = primary['micro']
lines = ['SB objective adapter / SWaT / full frozen seed1103 experiment.', '',
    'The original20-epoch CPU fit completed successfully. The complete saved checkpoint was restored without fitting or coupling solves. Every train, validation and test timestamp score was independently reinferred bitwise equal; all original core/event/bootstrap metrics recomputed exactly in the original environment.', '',
    '| Metric | Value |', '|---|---:|']
for label, key in [('Precision','precision'), ('Recall','recall'), ('F1','f1'), ('AUROC','auroc'), ('Average precision','average_precision')]:
    lines.append(f"| {label} | {metrics[key]:.9g} |")
for key in ('VUS_ROC', 'VUS_PR'):
    lines.append(f"| {key} (entity macro) | {proof['VUS_entity_macro'][key]['mean_over_defined_entities']:.9g} |")
lines.extend(['', 'Primary metrics use the original frozen validation-only1% threshold without point adjustment. The conditional one-model temporal-block F1 interval is ' + json.dumps(primary['f1_ci95']) + '.', '',
    f"Test normal timestamp false-alarm fraction at this unchanged threshold: {proof['measured_test_normal_false_alarm_fraction']:.9g}. Full score distributions are retained in the diagnostic JSON. These test-label diagnostics were calculated after freezing calibration and did not select the threshold.", '',
    'Test-score-calibrated and test-label-selected PA grid metrics are retained separately in the complete metrics JSON. They must not be substituted for the primary metric or interpreted as detector improvement.', '',
    'Original numerical repair: ' + model['numerical_repair']['repair'] + '. Original failure and source remain preserved. This audit did not run a new solver or alter training settings.', '',
    'This is a local MLP/backbone and velocity-discrepancy TSAD score adapter of the cited SB objective, not a reproduction of an original-author TSAD experiment or the complete TAB training harness. Full original range buffers, source identity and macros were audited; VUS/Affiliation integrals were not independently rerun. This is one full seed, not a completed five-seed comparison. The full45paper/681method-ablation/273data-task scope and all105 entropic repair jobs remain retained.', '',
    'Paper: ' + model['citation']])
with (ROOT / BASE / 'README.md').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write('\n'.join(lines) + '\n')
paths = ['scripts/flow_matching/audit_local_SB_full_checkpoint_scores_v1.py',
    'tests/test_fm_local_SB_full_budget_identity_v1.py', CONFIG]
paths.extend(path.relative_to(ROOT).as_posix() for path in (ROOT / BASE).rglob('*') if path.is_file() and not path.name.endswith('.log'))
with (ROOT / 'tmp/local_SB_SWaT_publication_manifest_20261014_v1.json').open('x', encoding='utf-8') as stream:
    json.dump(dict(previous_root='93061782d9debd16bdc374eb33d176f04e6fddb4',
        previous_parent='eccae5995bc98c07e0ac4165fd3103172e851851', paths=sorted(paths)), stream, indent=2)
print(json.dumps(dict(passed=True, primary_f1=metrics['f1'], test_normal_false_alarm_fraction=proof['measured_test_normal_false_alarm_fraction'],
    full_test_points=proof['full_native_test_points'], exact_publication_files=len(paths))))
