"""Publish the full industrial SKAB score audit with unchanged protocol boundaries."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import psutil

from scripts.flow_matching.prepare_tsad_execution import sha
from scripts.flow_matching.run_tsad_queue import complete_result

ROOT = Path(__file__).resolve().parents[1]
BASE = 'projects/flow_matching_research/results/2026-10-14/moment_base_release512_SKAB_full_saved_scores_v1'
CONFIG = 'configs/reproducibility/fm_moment_base_release512_SKAB_full_saved_score_audit.2026-10-14_v1.json'
JOB = 'fm_moment_pretrained_base__release512__skab__seed42'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def save(path, value):
    destination = ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


config = read(CONFIG)
for pin in config['source_receipts']:
    assert sha(ROOT / pin['path']) == pin['sha256'], pin['path']
proof_path = config['output_directory'] + '/independent_full_saved_score_verification.json'
proof = read(proof_path)
audit_guard_path = config['state_root'] + '/receipt.json'
guard = read(audit_guard_path)
assert proof['passed'] and proof['canonical_id'] == JOB
assert proof['audit_configuration_sha256'] == sha(ROOT / CONFIG)
assert guard['exit_code'] == 0 and guard['full_saved_score_review_passed']
assert guard['resource_guard_aborted'] is False
assert proof['all_three_strict_calibrations_event_metrics_bootstraps_and_ten_TAB_core_grid_metrics_recomputed_exact']
assert not proof['model_reinference_performed']
assert not proof['numerical_VUS_and_affiliation_integrals_independently_rerun']
queue = read(config['queue'])
assert len(queue['jobs']) == 72 and len({j['id'] for j in queue['jobs']}) == 72
job = next(j for j in queue['jobs'] if j['id'] == JOB)
assert job['input_kind'] == 'industrial' and job['window'] == 512
assert complete_result(ROOT, job)
result_path = job['output_directory'] + '/result.json'
result = read(result_path)
assert sha(ROOT / result_path) == proof['original_result_sha256']
assert sha(ROOT / job['output_directory'] / 'scores.npz') == proof['original_scores_sha256'] == result['scores_sha256']
for pin in job['frozen_files']:
    assert sha(ROOT / pin['path']) == pin['sha256'], pin['path']
dataset = next(d for d in read(job['dataset_manifest_path'])['datasets'] if d['id'] == 'skab')
for pin in dataset['source_files']:
    assert sha(ROOT / pin['path']) == pin['sha256'], pin['path']
entities = len(dataset['entities'])
test_points = sum(entry['test_points'] for entry in dataset['entities'])
assert entities == proof['full_original_entities'] == 33
assert test_points == proof['full_original_test_points'] == 36656
assert len(dataset['training_groups']) == len(dataset['validation_groups']) == 1
assert len(result['native_coverage']) == 35
source_guard = read(config['terminal_receipt'])
assert source_guard['exit_code'] == 0 and source_guard['full_registered_budget_retained']
assert source_guard['resource_guard_aborted'] is False
assert sha(ROOT / config['terminal_receipt']) == proof['source_guard_sha256']
ranged = read(config['range_evaluation'])
assert sha(ROOT / config['range_evaluation']) == proof['range_evaluation_sha256']
range_state = read(config['range_controller_state'])
assert range_state['completed'][JOB]['evaluation_identity'] == ranged['evaluation_identity']
assert len(ranged['VUS_per_entity']) == entities
assert all(row['threshold_samples'] == 250 for row in ranged['VUS_per_entity'].values())
controls = proof['full_saved_score_protocol_controls']
primary = proof['primary_metrics']
plain, adjusted = controls['validation_only_no_PA'], controls['validation_only_same_threshold_with_PA']
assert plain == {key: primary['micro'][key] for key in plain}
assert plain['false_positives'] == adjusted['false_positives']
for name, path in dict(original_result=result_path, original_source_guard=config['terminal_receipt'],
    original_range_evaluation=config['range_evaluation'], original_range_resource_receipt=config['range_resource_receipt'],
    independent_audit_guard=audit_guard_path).items():
    shutil.copyfile(ROOT / path, ROOT / BASE / (name + '.json'))
shutil.copyfile(__file__, ROOT / BASE / 'closure_source.py')
(ROOT / BASE / '.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
save(BASE + '/frozen_dataset_manifest_entry.json', dataset)
save(BASE + '/complete_metrics_and_protocol_effect.json', dict(canonical_id=JOB,
    model='MOMENT-1-base official fixed release;512context;zero-shot', dataset='SKAB grouped industrial runs',
    full_test_groups=entities, native_test_points=test_points,
    native_train_points=sum(g['points'] for g in dataset['training_groups']),
    native_validation_points=primary['calibration']['calibration_points'],
    model_parameters=result['parameter_count'], source_score_seconds=result['score_seconds'], training_seconds=0,
    primary_validation_only_no_PA=primary, same_scores_protocol_controls=controls,
    same_validation_threshold_PA_f1_increase=adjusted['f1']-plain['f1'],
    same_validation_threshold_PA_true_positive_increase=adjusted['true_positives']-plain['true_positives'],
    raw_scores_and_threshold_unchanged=True, VUS_entity_macro=ranged['VUS_entity_macro'],
    strict_affiliation_entity_macro=ranged['strict_affiliation_entity_macro'],
    every_VUS_entity_buffer_and_threshold_budget=ranged['VUS_per_entity'],
    source_track=dataset['source_track'], generalization_boundary=dataset['generalization'],
    native_score_context_and_offline_delay_boundaries=primary['event_metrics']['delay_boundary'],
    pretrained_load_receipt=result['checkpoint_load_audit'], scores_sha256=result['scores_sha256'],
    model_reinference_performed=False, numeric_range_integrals_rerun=False,
    historical_paper_checkpoint_equivalence=False, complete_TAB_training_harness_equivalence=False,
    independent_training_seed_repetitions=False, full_registered_MOMENT_jobs=72,
    original_scope=config['original_scope'],all_original_experiments_complete=False))
save(BASE + '/closure_validation.json',dict(passed=True,
    all_current_raw_processed_data_weight_asset_code_checkpoint_reference_result_and_score_hashes_verified=True,
    full_grouped_train_validation_test_score_and_metadata_coverage_verified=True,
    primary_core_events_and_grouped_uncertainty_exact_in_actual_producer_environment=True,
    original_full_range_scope_identity_buffer_grid_macros_and_success_receipt_verified=True,
    independent_guard_sha256=sha(ROOT / audit_guard_path), independent_proof_sha256=sha(ROOT / proof_path),
    configuration_sha256=sha(ROOT / CONFIG), all_original_experiments_complete=False))
handles=[]
for path in ['experiments/runs/fm_bounded_remaining_cpu_v3/moment_cpu/status.json',config['range_controller_state']]:
    state=read(path);p=psutil.Process(state['pid']);command=p.cmdline()
    assert any('scripts.flow_matching.' in argument for argument in command)
    if state.get('create_time') is not None:assert state['create_time']==p.create_time()
    handles.append(dict(state_path=path,pid=p.pid,create_time=p.create_time(),command=command))
save(BASE + '/actual_live_full72_original_score_and_range_queue_handles.json',handles)
model=read(job['model_config'])
aff=ranged['strict_affiliation_entity_macro']['1']['affiliation_f']
lines=['MOMENT-1-base / released512 context / SKAB / full frozen zero-shot evaluation.','',
    f'All {entities} registered test runs and {test_points:,} test timestamps were scored. One complete anomaly-free run is used for training/reference and other/1.csv is reserved for validation before outcomes. '
    'The validation run retains its native anomalies; the nominal1% quantile is not a normal-only false-alarm guarantee. '
    'Original 109,641,608 parameters were strictly loaded; no random head, fine-tuning or parameter fitting. '
    f'Full CPU score time: {result["score_seconds"]:.3f}s.','',
    '| Metric | Value | Protocol / aggregation |','|---|---:|---|',
    f'| F1 | {plain["f1"]:.9g} | Validation-only1% threshold, noPA, micro |',
    f'| Precision | {plain["precision"]:.9g} | Same fixed threshold |',
    f'| Recall | {plain["recall"]:.9g} | Same fixed threshold |',
    f'| AUROC | {primary["micro"]["auroc"]:.9g} | Pooled timestamp ranking |',
    f'| AP | {primary["micro"]["average_precision"]:.9g} | Pooled timestamp ranking |',
    f'| VUS_ROC | {ranged["VUS_entity_macro"]["VUS_ROC"]["mean_over_defined_entities"]:.9g} | Macro,33defined/0undefined |',
    f'| VUS_PR | {ranged["VUS_entity_macro"]["VUS_PR"]["mean_over_defined_entities"]:.9g} | Macro,33defined/0undefined |',
    f'| Affiliation F | {aff["mean_over_defined_entities"]:.9g} | Macro,{aff["defined_entities"]}defined/{aff["undefined_entities"]}undefined |','',
    'Primary F1 entity-cluster95% interval: '+json.dumps(primary['f1_ci95'])+'. '
    'Conditional on this fixed released checkpoint and threshold, not independent training seeds.','',
    '| Same frozen scores | F1 |','|---|---:|',
    f'| Validation threshold,noPA | {plain["f1"]:.9g} |',
    f'| Same validation threshold,ground-truth PA | {adjusted["f1"]:.9g} |',
    f'| Test-pool1% threshold,noPA | {controls["test_pool_calibration_ratio1"]["f_score"]:.9g} |',
    f'| Same test-pool threshold,ground-truth PA | {controls["test_pool_calibration_ratio1"]["adjust_f_score"]:.9g} |',
    f'| Best test-label-selected plain grid | {controls["best_test_label_selected_plain_grid_F1"]:.9g} |',
    f'| Best test-label-selected PA grid | {controls["best_test_label_selected_PA_grid_F1"]:.9g} |','',
    f'PA at the same validation threshold adds {int(adjusted["true_positives"]-plain["true_positives"]):,} counted true-positive timestamps; '
    f'false positives stay {int(plain["false_positives"]):,}. This is a metric-protocol effect on identical scores, not an improved detector. '
    'Test calibration and test-label oracle controls remain separate from the primary result.','',
    'The separate industrial layout was independently checked, including every native train/validation/test score, label, timestamp, source row and fault-label array present. '
    'All primary/core/event/bootstrap values recomputed exactly in Python3.10.21/NumPy1.25.2/Torch2.2.1+cu121 CPU. '
    '38 relevant validation tests passed. All original VUS integer buffers and250thresholds per entity, source identities and macro/undefined counts were verified. '
    'Range integrals and model inference were not numerically rerun in this saved-score review. '
    'Offline512-point context can use future points; delay is not causal deployment latency. '
    'Historical-paper weights and complete TAB training-harness equivalence are not certified. '
    'All72 MOMENT jobs and the full45/681/273 obligations remain active.','',
    'Paper: '+model['citation'],'Checkpoint: '+model['checkpoint_citation'],
    'Evaluation reference: https://github.com/decisionintelligence/TAB/tree/'+model['source_commit']]
with (ROOT / BASE / 'README.md').open('x',encoding='utf-8',newline='\n') as stream:
    stream.write('\n'.join(lines)+'\n')
paths=['scripts/flow_matching/audit_moment_industrial_saved_scores_v1.py',
    'tests/test_fm_moment_industrial_saved_score_audit_v1.py',CONFIG]
paths.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/BASE).rglob('*') if p.is_file())
save('tmp/moment_SKAB_full_saved_score_publication_manifest_20261014_v1.json',dict(
    previous_root='b6bfba3bd05a2776fb8389f14ba70ac2a495586f',
    previous_parent='b895310e7117fffa76ce5c04c8d0ff48900e0c5e',paths=sorted(paths)))
print(json.dumps(dict(full_saved_score_audit_passed=True,entities=entities,test_points=test_points,
    primary_F1=plain['f1'],same_threshold_PA_F1=adjusted['f1'],
    best_test_label_PA_F1=controls['best_test_label_selected_PA_grid_F1'],exact_files=len(paths))))
