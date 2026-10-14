"""Archive only an actually closed full original seed1106/1107 model pair."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching.audit_cfm_gp_target_pair_v1 import audit
from scripts.flow_matching.audit_cfm_gp_five_seed_targets_v1 import identities

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, choices=[1106, 1107], required=True)
args = parser.parse_args()
root, read, sha = cold.ROOT, cold.original.read, cold.streaming_sha
runtimepath = root / 'configs/runtime/fm_cfm_completion_replay_queue.v4.json'
runtime = read(runtimepath)
snapshot_bytes = (root / runtime['state_root'] / 'verified_full_model_records.json').read_bytes()
snapshot = json.loads(snapshot_bytes)
byid = {r['original_canonical_id']: r for r in snapshot['records']}
pair = next(pair for pair in identities() if pair[0].endswith(str(args.seed)))
records = [byid[i] for i in pair]
for record in records:
    cold.validate_closed_primary_record(runtime, record)
assert snapshot['required_original_models'] == 285 and len(byid) == len(snapshot['records'])
relative = 'projects/flow_matching_research/results/2026-10-14/cfm_GP400_conditional_seed' + str(args.seed) + '_full_pair_closure_v1'
folder = root / relative
folder.mkdir(exist_ok=False)
snapshotpath = folder / 'frozen_actual_closed_primary_models.json'
snapshotpath.write_bytes(snapshot_bytes)
certificate = folder / 'new_complete_native_and_independent_certificate'
certificate.mkdir()
nativepath = root / runtime['child_config_root'] / (pair[1] + '.native.json')
reviewpath = root / runtime['child_config_root'] / (pair[1] + '.review.json')
native, review = read(nativepath), read(reviewpath)
sources = {
    'original_result.json': root / records[1]['result_path'],
    'original_guarded_receipt.json': root / 'experiments/runs/fm_cfm_ts_original_v1/receipts' / (pair[1] + '.json'),
    'native_config.json': nativepath,
    'native_report.json': root / native['output_directory'] / 'checkpoint_replay_audit.json',
    'native_guarded_receipt.json': root / native['state_root'] / 'receipt.json',
    'independent_config.json': reviewpath,
    'independent_proof.json': root / review['output_proof'],
    'independent_guarded_receipt.json': root / review['state_root'] / 'receipt.json',
}
receipt = read(sources['original_guarded_receipt.json'])
assert receipt['exit_code'] == 0 and receipt['complete'] and not receipt['resource_observations']['resource_guard_aborted']
for name, source in sources.items():
    (certificate / name).write_bytes(source.read_bytes())
config = dict(replay_runtime=runtimepath.relative_to(root).as_posix(),
    closed_record_snapshot=snapshotpath.relative_to(root).as_posix(), paired_canonical_ids=pair,
    bootstrap_seed=20261014, bootstrap_replicates=5000,
    output_report=relative + '/complete_original_target_seed_pair.json',
    citation='https://openreview.net/forum?id=Hqn4Aj7xrQ',
    source_receipts=[dict(path=p.relative_to(root).as_posix(), sha256=sha(p)) for p in [runtimepath, snapshotpath,
        root / 'scripts/flow_matching/audit_cfm_gp_target_pair_v1.py', root / 'tests/test_cfm_gp_target_pair_v1.py']])
configpath = root / ('configs/reproducibility/fm_cfm_GP400_original_target_pair_seed' + str(args.seed) + '.2026-10-14_v1.json')
with configpath.open('x', encoding='utf-8') as stream: json.dump(config, stream, indent=2)
report = audit(config)
with (root / config['output_report']).open('x', encoding='utf-8') as stream: json.dump(report, stream, indent=2)
complete_seed_pairs = [int(pair[0].rsplit('seed', 1)[1]) for pair in identities() if all(i in byid for i in pair)]
closure = dict(captured_utc=datetime.now(timezone.utc).isoformat(), new_record=records[1],
    frozen_closed_primary_model_count=len(snapshot['records']), required_original_CFM_models=285,
    every_current_model_data_result_and_replay_SHA256_rechecked=True,
    original_native_and_independent_all_grid_proofs_rechecked=True,
    complete_original_target_seed_pairs=complete_seed_pairs,
    required_original_target_seed_pairs=[1103, 1104, 1105, 1106, 1107],
    closed_certificate_sources=[dict(destination=name, source=str(path), sha256=sha(path)) for name, path in sources.items()],
    additional_optimizer_updates=0, additional_model_inference=0,
    test_based_recipe_selection=False, author_equivalence_certified=False, strict_TAB_result=False,
    original_scope={'papers': 45, 'method_baseline_ablation_records': 681, 'original_data_task_records': 273},
    all_original_experiments_complete=False)
(folder / 'new_full_model_and_pair_certificate_closure.json').write_text(json.dumps(closure, indent=2) + '\n')
(folder / 'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n')
print(json.dumps({'primary_full_models': len(snapshot['records']), 'complete_seed_pairs': complete_seed_pairs,
    'metrics': {k: report[k] for k in ['literal_MSE', 'conditional_MSE', 'paired_MSE_difference_conditional_minus_literal', 'paired_trajectory_bootstrap_ci95']}}, indent=2))
