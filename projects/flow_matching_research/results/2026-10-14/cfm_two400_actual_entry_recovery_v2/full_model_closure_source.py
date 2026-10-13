from datetime import datetime, timezone
from pathlib import Path
import json
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching.audit_cfm_gp_target_pair_v1 import audit

root = cold.ROOT
relative = 'projects/flow_matching_research/results/2026-10-14/cfm_two400_actual_entry_recovery_v2'
folder = root / relative
read, sha = cold.original.read, cold.streaming_sha
runtimepath = root / 'configs/runtime/fm_cfm_completion_replay_queue.v4.json'
runtime = read(runtimepath)
snapshotpath = folder / 'frozen_actual_138_closed_models.json'
with snapshotpath.open('xb') as stream:
    stream.write((root / runtime['state_root'] / 'verified_full_model_records.json').read_bytes())
snapshot = read(snapshotpath)
ids = ['cfm_ts_original__appendix_d__gp_' + mode + '_e400__sine__seed1105'
       for mode in ['paper_literal', 'gaussian_conditional']]
records = [next(r for r in snapshot['records'] if r['original_canonical_id'] == i) for i in ids]
for record in records: cold.validate_closed_primary_record(runtime, record)
assert len(snapshot['records']) >= 138 and snapshot['required_original_models'] == 285
certificate = folder / 'new_closed_GP400_gaussian_seed1105'
certificate.mkdir(exist_ok=False)
nativepath = root / runtime['child_config_root'] / (ids[1] + '.native.json')
reviewpath = root / runtime['child_config_root'] / (ids[1] + '.review.json')
native, review = read(nativepath), read(reviewpath)
sources = {
    'original_result.json': root / records[1]['result_path'],
    'original_guarded_receipt.json': root / 'experiments/runs/fm_cfm_ts_original_v1/receipts' / (ids[1] + '.json'),
    'native_config.json': nativepath,
    'native_report.json': root / native['output_directory'] / 'checkpoint_replay_audit.json',
    'native_guarded_receipt.json': root / native['state_root'] / 'receipt.json',
    'independent_config.json': reviewpath,
    'independent_proof.json': root / review['output_proof'],
    'independent_guarded_receipt.json': root / review['state_root'] / 'receipt.json',
}
receipt = read(sources['original_guarded_receipt.json'])
assert receipt['exit_code'] == 0 and receipt['complete'] and not receipt['resource_observations']['resource_guard_aborted']
for name, source in sources.items(): (certificate / name).write_bytes(source.read_bytes())
config = dict(replay_runtime=runtimepath.relative_to(root).as_posix(), closed_record_snapshot=snapshotpath.relative_to(root).as_posix(),
    paired_canonical_ids=ids, bootstrap_seed=20261014, bootstrap_replicates=5000,
    output_report=relative + '/complete_seed1105_paired_original_target_variants.json',
    citation='https://openreview.net/forum?id=Hqn4Aj7xrQ',
    source_receipts=[dict(path=p.relative_to(root).as_posix(), sha256=sha(p)) for p in [runtimepath, snapshotpath,
                    root / 'scripts/flow_matching/audit_cfm_gp_target_pair_v1.py', root / 'tests/test_cfm_gp_target_pair_v1.py']])
with (root / 'configs/reproducibility/fm_cfm_GP400_original_target_pair_seed1105.2026-10-14_v1.json').open('x', encoding='utf-8') as stream:
    json.dump(config, stream, indent=2)
report = audit(config)
with (root / config['output_report']).open('x', encoding='utf-8') as stream: json.dump(report, stream, indent=2)
paths = [root / 'projects/flow_matching_research/results/2026-10-14/cfm_gp400_target_variant_pair_v1/complete_seed1103_paired_original_target_variants.json',
         root / 'projects/flow_matching_research/results/2026-10-14/cfm_gp400_second_original_target_pair_v1/complete_seed1104_paired_original_target_variants.json',
         root / config['output_report']]
keys = ['training_seed', 'literal_MSE', 'conditional_MSE', 'paired_MSE_difference_conditional_minus_literal',
        'paired_trajectory_bootstrap_ci95', 'conditional_lower_error_trajectory_fraction']
pair_reports = [read(p) for p in paths]
assert [r['training_seed'] for r in pair_reports] == [1103, 1104, 1105]
closure = dict(captured_utc=datetime.now(timezone.utc).isoformat(), new_record=records[1],
    frozen_primary_record_count=len(snapshot['records']), required_original_models=285,
    native_and_independent_full_grid_guarded_proofs_verified=True,
    every_current_checkpoint_data_result_and_replay_SHA256_rechecked=True,
    certificate_sources=[dict(destination=name, source=str(p), sha256=sha(p)) for name, p in sources.items()],
    required_original_training_seeds=[1103, 1104, 1105, 1106, 1107], completed_seed_pairs=[1103, 1104, 1105],
    pending_seed_pairs=[1106, 1107], pair_reports=[dict(path=p.relative_to(root).as_posix(), sha256=sha(p)) for p in paths],
    per_training_seed_pairs=[{k: r[k] for k in keys} for r in pair_reports],
    training_seed_confidence_interval_estimated=False, full_five_seed_comparison_complete=False,
    bootstrap_boundary='Each interval conditions on its trained pair and resamples whole trajectories, never cells or pooled training seeds.',
    additional_model_inference=0, additional_optimizer_updates=0, test_based_recipe_selection=False,
    strict_TAB_result=False, author_equivalence_certified=False, all_original_experiments_complete=False)
(folder / 'new_complete_138_model_and_three_seed_pair_closure.json').write_text(json.dumps(closure, indent=2) + '\n', encoding='utf-8')
(folder / 'full_model_closure_source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'verified_models': len(snapshot['records']), 'paired_seed1105': {k: report[k] for k in keys}}, indent=2))
