"""Freeze and verify a newly complete full-budget model; no training or inference."""
from datetime import datetime, timezone
import json
from pathlib import Path
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching.audit_cfm_gp_target_pair_v1 import audit

root = cold.ROOT
relative = 'projects/flow_matching_research/results/2026-10-14/cfm_gp400_second_original_target_pair_v1'
folder = root / relative
folder.mkdir(exist_ok=False)
read, sha = cold.original.read, cold.streaming_sha
runtime_path = root / 'configs/runtime/fm_cfm_completion_replay_queue.v4.json'
runtime = read(runtime_path)
snapshot_bytes = (root / 'experiments/runs/fm_cfm_completion_replay_queue_v4/verified_full_model_records.json').read_bytes()
snapshot = json.loads(snapshot_bytes)
assert snapshot['required_original_models'] == 285
ids = ['cfm_ts_original__appendix_d__gp_' + mode + '_e400__sine__seed1104'
       for mode in ['paper_literal', 'gaussian_conditional']]
records = [next(r for r in snapshot['records'] if r['original_canonical_id'] == i) for i in ids]
for record in records:
    cold.validate_closed_primary_record(runtime, record)
snapshot_path = folder / 'frozen_actual_closed_models.json'
snapshot_path.write_bytes(snapshot_bytes)
original_result = read(root / records[1]['result_path'])
job = original_result['job']
assert original_result['epochs_completed'] == 400 and original_result['optimizer_updates'] == 250000
assert job['test_trajectories'] == 80
certificate = folder / 'new_closed_GP400_gaussian_seed1104'
certificate.mkdir()
native_path = root / runtime['child_config_root'] / (ids[1] + '.native.json')
review_path = root / runtime['child_config_root'] / (ids[1] + '.review.json')
native, review = read(native_path), read(review_path)
sources = {
    'original_result.json': root / records[1]['result_path'],
    'original_guarded_receipt.json': root / 'experiments/runs/fm_cfm_ts_original_v1/receipts' / (ids[1] + '.json'),
    'native_config.json': native_path,
    'native_report.json': root / native['output_directory'] / 'checkpoint_replay_audit.json',
    'native_guarded_receipt.json': root / native['state_root'] / 'receipt.json',
    'independent_config.json': review_path,
    'independent_proof.json': root / review['output_proof'],
    'independent_guarded_receipt.json': root / review['state_root'] / 'receipt.json',
}
receipt = read(sources['original_guarded_receipt.json'])
assert receipt['complete'] and receipt['exit_code'] == 0 and not receipt['resource_observations']['resource_guard_aborted']
for name, source in sources.items():
    (certificate / name).write_bytes(source.read_bytes())
config_path = root / 'configs/reproducibility/fm_cfm_GP400_original_target_pair_seed1104.2026-10-14_v1.json'
pins = [runtime_path, snapshot_path, root / 'scripts/flow_matching/audit_cfm_gp_target_pair_v1.py',
        root / 'tests/test_cfm_gp_target_pair_v1.py']
config = {
    'replay_runtime': runtime_path.relative_to(root).as_posix(),
    'closed_record_snapshot': snapshot_path.relative_to(root).as_posix(),
    'paired_canonical_ids': ids,
    'bootstrap_seed': 20261014,
    'bootstrap_replicates': 5000,
    'output_report': relative + '/complete_seed1104_paired_original_target_variants.json',
    'citation': 'https://openreview.net/forum?id=Hqn4Aj7xrQ',
    'source_receipts': [{'path': p.relative_to(root).as_posix(), 'sha256': sha(p)} for p in pins],
}
with config_path.open('x', encoding='utf-8') as stream:
    json.dump(config, stream, indent=2)
report = audit(config)
with (root / config['output_report']).open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2)
previous_path = root / 'projects/flow_matching_research/results/2026-10-14/cfm_gp400_target_variant_pair_v1/complete_seed1103_paired_original_target_variants.json'
previous = read(previous_path)
assert previous['training_seed'] == 1103 and report['training_seed'] == 1104
comparison = {
    'captured_utc': datetime.now(timezone.utc).isoformat(),
    'required_original_training_seeds': [1103, 1104, 1105, 1106, 1107],
    'completed_seed_pairs': [1103, 1104],
    'pending_seed_pairs': [1105, 1106, 1107],
    'source_pair_reports': [{'path': p.relative_to(root).as_posix(), 'sha256': sha(p)}
                            for p in [previous_path, root / config['output_report']]],
    'per_trained_seed_pairs': [{key: pair[key] for key in [
        'training_seed', 'literal_MSE', 'conditional_MSE',
        'paired_MSE_difference_conditional_minus_literal', 'paired_trajectory_bootstrap_ci95',
        'conditional_lower_error_trajectory_fraction', 'bootstrap_unit',
        'training_seed_uncertainty_estimated']} for pair in [previous, report]],
    'ranking_direction_changes_across_observed_training_seeds': (
        previous['paired_MSE_difference_conditional_minus_literal'] *
        report['paired_MSE_difference_conditional_minus_literal'] < 0),
    'training_seed_confidence_interval_estimated': False,
    'full_five_seed_comparison_complete': False,
    'pooled_trajectory_groups_are_not_independent_training_seeds': True,
    'test_based_recipe_selection': False,
    'additional_model_inference': 0,
    'additional_optimizer_updates': 0,
    'author_equivalence_certified': False,
    'strict_TAB_result': False,
    'all_original_experiments_complete': False,
}
(folder / 'two_complete_seed_pairs_with_uncertainty_boundary.json').write_text(json.dumps(comparison, indent=2) + '\n', encoding='utf-8')
closure = {
    'captured_utc': datetime.now(timezone.utc).isoformat(),
    'new_record': records[1],
    'frozen_primary_record_count': len(snapshot['records']),
    'required_original_models': 285,
    'every_current_model_and_data_artifact_SHA256_rechecked': True,
    'native_and_independent_full_grid_guarded_proofs_verified': True,
    'closed_certificate_sources': [{'destination': name, 'source': str(path), 'sha256': sha(path)}
                                   for name, path in sources.items()],
    'additional_model_inference': 0,
    'additional_optimizer_updates': 0,
    'full_scope': {'papers': 45, 'method_baseline_ablation_records': 681, 'original_data_task_records': 273},
    'all_original_experiments_complete': False,
}
(folder / 'new_complete_model_certificate_closure.json').write_text(json.dumps(closure, indent=2) + '\n', encoding='utf-8')
(folder / 'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n', encoding='utf-8')
print(json.dumps({'closed_original_models': len(snapshot['records']), 'paired_analysis': comparison['per_trained_seed_pairs']}, indent=2))
