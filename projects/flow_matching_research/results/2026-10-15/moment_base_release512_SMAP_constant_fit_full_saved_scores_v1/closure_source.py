from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import psutil

from scripts.flow_matching.prepare_tsad_execution import sha
from scripts.flow_matching.run_tsad_queue import complete_result

ROOT = Path(__file__).resolve().parents[1]
BASE = 'projects/flow_matching_research/results/2026-10-15/moment_base_release512_SMAP_constant_fit_full_saved_scores_v1'
CONFIG = 'configs/reproducibility/fm_moment_base_release512_SMAP_constant_fit_full_saved_score_audit.2026-10-15_v1.json'
JOB = 'fm_moment_pretrained_base__release512__smap__seed42__constant_fit_v2'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def save(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


config = read(CONFIG)
for pin in config['source_receipts']:
    assert sha(ROOT / pin['path']) == pin['sha256'], pin['path']
proof_path = config['output_directory'] + '/independent_full_saved_score_verification.json'
proof = read(proof_path)
audit_guard_path = config['state_root'] + '/receipt.json'
audit_guard = read(audit_guard_path)
assert proof['passed'] and proof['canonical_id'] == JOB
assert audit_guard['exit_code'] == 0 and audit_guard['full_saved_score_review_passed'] and not audit_guard['resource_guard_aborted']
assert proof['audit_configuration_sha256'] == sha(ROOT / CONFIG)
assert proof['full_original_entities'] == 51 and proof['full_original_test_points'] == 408160
assert proof['all_three_strict_calibrations_event_metrics_bootstraps_and_ten_TAB_core_grid_metrics_recomputed_exact']
assert not proof['model_reinference_performed'] and not proof['numerical_VUS_and_affiliation_integrals_independently_rerun']
queue = read(config['queue'])
assert len(queue['jobs']) == 6
job = next(item for item in queue['jobs'] if item['id'] == JOB)
assert complete_result(ROOT, job)
result_path = job['output_directory'] + '/result.json'
result = read(result_path)
assert sha(ROOT / result_path) == proof['original_result_sha256']
assert sha(ROOT / job['output_directory'] / 'scores.npz') == proof['original_scores_sha256'] == result['scores_sha256']
for pin in job['frozen_files']:
    assert sha(ROOT / pin['path']) == pin['sha256'], pin['path']
manifest = read(job['dataset_manifest_path'])
dataset = next(item for item in manifest['datasets'] if item['id'] == 'smap')
for pin in dataset['source_files']:
    assert sha(ROOT / pin['path']) == pin['sha256'], pin['path']
range_result = read(config['range_evaluation'])
assert sha(ROOT / config['range_evaluation']) == proof['range_evaluation_sha256']
range_state = read(config['range_controller_state'])
assert range_state['completed'][JOB]['evaluation_identity'] == range_result['evaluation_identity']
source_guard = read(config['terminal_receipt'])
assert source_guard['exit_code'] == 0 and source_guard['full_registered_budget_retained'] and not source_guard['resource_guard_aborted']
sources = dict(original_result=result_path, original_source_guard=config['terminal_receipt'],
    original_range_evaluation=config['range_evaluation'], original_range_resource_receipt=config['range_resource_receipt'],
    independent_audit_guard=audit_guard_path)
for name, path in sources.items():
    shutil.copyfile(ROOT / path, ROOT / BASE / (name + '.json'))
shutil.copyfile(__file__, ROOT / BASE / 'closure_source.py')
save(BASE + '/frozen_dataset_manifest_entry.json', dataset)
controls = proof['full_saved_score_protocol_controls']
primary = proof['primary_metrics']
ordinary = controls['validation_only_no_PA']
pa = controls['validation_only_same_threshold_with_PA']
assert ordinary == {key: primary['micro'][key] for key in ordinary}
assert ordinary['false_positives'] == pa['false_positives']
save(BASE + '/same_scores_protocol_effect_and_complete_metrics.json', dict(
    canonical_id=JOB, model='MOMENT-1-base official fixed release;512context;zero-shot', dataset='SMAP_constant_fit mTSBench published entities',
    full_original_entities=51, native_test_points=408160, native_validation_points=primary['calibration']['calibration_points'],
    model_parameters=result['parameter_count'], scoring_seconds=result['score_seconds'], training_seconds=0,
    fixed_pretrained_weight_evaluation_not_independent_training_seed_repeat=True,
    primary_validation_only_no_PA=primary,
    same_scores_protocol_controls=controls,
    same_validation_threshold_PA_f1_increase=pa['f1'] - ordinary['f1'],
    same_validation_threshold_PA_true_positive_increase=pa['true_positives'] - ordinary['true_positives'],
    raw_detector_scores_and_threshold_unchanged_in_PA_control=True,
    VUS_entity_macro=range_result['VUS_entity_macro'], strict_affiliation_entity_macro=range_result['strict_affiliation_entity_macro'],
    every_VUS_entity_buffer_and_threshold_budget=range_result['VUS_per_entity'],
    source_result_static_pending_range_fields_resolved_by_completed_external_range_artifact=['VUS_ROC', 'VUS_PR', 'affiliation_f'],
    data_grouping=dataset['grouping'], data_generalization_boundary=dataset['generalization'], data_source_track=dataset['source_track'],
    native_score_context_and_offline_delay_boundaries=primary['event_metrics']['delay_boundary'],
    raw_scores_sha256=result['scores_sha256'], pretrained_load_receipt=result['checkpoint_load_audit'],
    model_reinference_in_this_review=False, VUS_and_affiliation_integrals_numerically_rerun_in_this_review=False,
    official_historical_paper_checkpoint_equivalence=False, complete_TAB_training_harness_equivalence=False,
    original_full_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
    all_original_experiments_complete=False))
save(BASE + '/closure_validation.json', dict(passed=True, all_original_raw_CSV_and_processed_data_hashes_rechecked=True,
    all_current_full_weight_asset_code_result_checkpoint_and_score_hashes_rechecked=True,
    all_original_split_scores_and_labels_and_full_native_coverage_independently_checked=True,
    primary_and_all_core_diagnostics_events_and_bootstrap_intervals_exact_in_actual_producer_environment=True,
    full_original_range_integral_source_scope_macros_and_success_receipt_verified=True,
    native_audit_guard_sha256=sha(ROOT / audit_guard_path), native_proof_sha256=sha(ROOT / proof_path),
    native_configuration_sha256=sha(ROOT / CONFIG), all_original_experiments_complete=False))
handles = []
for state_path in ['experiments/runs/fm_tsad_constant_fit_cpu_controls_v2/moment_pretrained/status.json', config['range_controller_state']]:
    state = read(state_path)
    process = psutil.Process(state['pid'])
    if state.get('create_time') is not None:
        assert process.create_time() == state['create_time']
    handles.append(dict(state_path=state_path, actual_pid=process.pid, actual_create_time=process.create_time(),
        actual_command=process.cmdline(), state=state))
save(BASE + '/actual_live_full6_corrected_score_and_full291_range_queue_handles.json', handles)
model_config = read(job['model_config'])
lines = ['MOMENT-1-base / release512 context / SMAP corrected constant-fit / full zero-shot evaluation.', '',
    'All51 published entities and408160test points. All109641608 fixed parameters were strictly loaded; no fitting, head reinitialization or new training seed. Original producer environment:' + json.dumps(proof['actual_producer_environment']), '',
    'Primary validation-only1% threshold, noPA:' + json.dumps(primary['micro']),
    'Conditional entity-cluster uncertainty:' + json.dumps(primary['f1_ci95']),
    'These200 resamples describe one fixed model and threshold; they are not independent training seeds.', '',
    'Same frozen score protocol controls:' + json.dumps(controls),
    'At identical scores and the same validation threshold, ground-truth PA adds' + str(int(pa['true_positives']-ordinary['true_positives'])) + 'counted true-positive timestamps; false positives stay' + str(int(ordinary['false_positives'])) + '. This is a metric-protocol effect, not an improved detector or a claim about how a paper evaluated.', '',
    'Full entity-macro range metrics:' + json.dumps(range_result['VUS_entity_macro']),
    'Strict validation-only affiliation macros, retaining every undefined entity:' + json.dumps(range_result['strict_affiliation_entity_macro']['1']), '',
    'Original full VUS/Affiliation source hashes, array identity, every original integer buffer,250threshold budget and macro arithmetic were verified. No range-integral numerical rerun and no model score re-inference in this saved-score review. All3calibrations,10TAB-core grid controls,event metrics and grouped bootstrap values were independently recomputed exactly in the producer environment.', '',
    'Original constant-fit preprocessing, entity-isolated native data and offline512-point context retained. This is the current official checkpoint zero-shot evaluation, not historical-paper weight, fine-tuning, author-pipeline or fullTAB-training-harness equivalence. All6corrected MOMENT slots,72original MOMENT jobs,291corrected data-control range jobs and45papers/681methods/273data obligations remain required. 15relevant checks passed.', '',
    'Paper:' + model_config['citation'], 'Checkpoint:' + model_config['checkpoint_citation'],
    'Evaluation reference:https://github.com/decisionintelligence/TAB/tree/' + model_config['source_commit']]
with (ROOT / BASE / 'README.md').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write('\n'.join(lines) + '\n')
changed = [CONFIG]
changed.extend(path.relative_to(ROOT).as_posix() for path in (ROOT / BASE).rglob('*') if path.is_file() and not path.name.endswith('.log'))
save('tmp/moment_SMAP_constant_fit_saved_score_publication_manifest_20261015_v1.json', dict(
    previous_root='ba4bf33554470b869278948a257276f2a1c43b39', previous_parent='452e4519116788d1894769fc48762a0786c9e0c4', paths=sorted(changed)))
print(json.dumps(dict(full_MOMENT_SMAP_constant_fit_saved_score_audit_passed=True, full_test_points=408160,
    primary_F1=ordinary['f1'], same_threshold_PA_F1=pa['f1'], best_test_label_selected_PA_F1=controls['best_test_label_selected_PA_grid_F1'],
    VUS_ROC=range_result['VUS_entity_macro']['VUS_ROC']['mean_over_defined_entities'], exact_publication_files=len(changed))))
