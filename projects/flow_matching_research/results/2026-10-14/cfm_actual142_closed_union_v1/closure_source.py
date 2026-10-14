from datetime import datetime, timezone
from pathlib import Path
import json
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching import audit_cfm_gp_five_seed_targets_v1 as cohort

root, read, sha = cold.ROOT, cold.original.read, cold.streaming_sha
relative = 'projects/flow_matching_research/results/2026-10-14/cfm_actual142_closed_union_v1'
folder = root / relative
folder.mkdir(exist_ok=False)
primary_path = root / 'configs/runtime/fm_cfm_completion_replay_queue.v4.json'
supplement_path = root / 'configs/runtime/fm_cfm_metadata_entry_full_replay.v2.json'
primary, supplement = read(primary_path), read(supplement_path)
snapshots = [root / runtime['state_root'] / 'verified_full_model_records.json' for runtime in [primary, supplement]]
records = {}
for path in snapshots:
    for record in read(path)['records']:
        identity = record['original_canonical_id']
        if identity in records: assert records[identity] == record
        records[identity] = record
assert len(records) >= 142 and len(primary['routes']) == len(supplement['routes']) == 285
new_ids = ['cfm_ts_original__main_text__bb_paper_literal_e400__sine__seed' + str(seed) for seed in [1104, 1106]]
for identity in new_ids:
    record = records[identity]
    cold.validate_closed_primary_record(supplement, record)
    assert record['epochs'] == 400 and record['optimizer_updates'] == 6400 and record['full_replay_shape'] == [20, 500, 1]
    case = folder / ('new_closed_BB400_seed' + str(record['seed']))
    case.mkdir()
    nativepath = root / supplement['child_config_root'] / (identity + '.native.json')
    reviewpath = root / supplement['child_config_root'] / (identity + '.review.json')
    native, review = read(nativepath), read(reviewpath)
    job = next(r['job'] for r in supplement['routes'] if r['original_canonical_id'] == identity)
    sources = {
        'original_result.json': root / record['result_path'],
        'original_guarded_receipt.json': root / Path(job['output_directory']).parent.parent / 'receipts' / (job['id'] + '.json'),
        'native_config.json': nativepath,
        'native_report.json': root / native['output_directory'] / 'checkpoint_replay_audit.json',
        'native_guarded_receipt.json': root / native['state_root'] / 'receipt.json',
        'independent_config.json': reviewpath,
        'independent_proof.json': root / review['output_proof'],
        'independent_guarded_receipt.json': root / review['state_root'] / 'receipt.json',
    }
    for name, path in sources.items(): (case / name).write_bytes(path.read_bytes())
    (case / 'closed_full_record.json').write_text(json.dumps(record, indent=2) + '\n')
windowbase = root / 'experiments/runs/fm_cfm_complete_review_priority_window_v1'
window = read(windowbase / 'status.json')
assert window['state'] == 'release_all_target_proofs_complete' and not window['shared_lock_owned']
assert window['models_stopped_or_restarted'] == 0 and not window['original_guards_changed']
for name in ['status.json', 'complete_target_proof_union.json']:
    (folder / ('completed_review_window_' + name)).write_bytes((windowbase / name).read_bytes())
configpath = root / 'configs/reproducibility/fm_cfm_GP400_all_five_original_target_pairs.2026-10-14_v1.json'
config = read(configpath)
snapshot = read(root / config['frozen_complete_snapshot'])
reportpath = root / config['output_report']
reported = read(reportpath)
assert reported['frozen_complete_snapshot_sha256'] == sha(root / config['frozen_complete_snapshot'])
# Freshly re-read every coordinate and source proof. No model inference/update.
checked = cohort.audit_complete(config, cohort.verify_config(config), snapshot)
for key in ['paper_literal', 'gaussian_conditional', 'paired_difference_mean', 'paired_difference_joint_seed_std',
            'hierarchical_paired_bootstrap_ci95', 'paired_difference_per_joint_data_and_training_seed']:
    assert checked[key] == reported[key], key
closure = dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    full_unique_canonical_union=len(records), required_original_CFM_models=285,
    primary_closed_models=len(read(snapshots[0])['records']),
    supplement_total_closed_models=len(read(snapshots[1])['records']),
    deduplicated_canonical_records=list(records.values()),
    new_closed_full_BB_records=[records[i] for i in new_ids],
    original_review_window_completed_with_unchanged_guards=True,
    all_ten_full_original_GP400_target_models_freshly_reverified=True,
    all_complete_target_prediction_coordinates_recomputed=400000,
    five_seed_target_report_path=reportpath.relative_to(root).as_posix(), five_seed_target_report_sha256=sha(reportpath),
    source_snapshot_receipts=[dict(path=p.relative_to(root).as_posix(), sha256=sha(p)) for p in [primary_path, supplement_path,
        reportpath, configpath, root / config['frozen_complete_snapshot']]],
    extra_optimizer_updates=0, extra_model_inference=0, author_equivalence_certified=False, strict_TAB_result=False,
    original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
    all_original_experiments_complete=False)
(folder / 'actual_deduplicated_full142_and_complete_target_cohort_closure.json').write_text(json.dumps(closure, indent=2) + '\n')
(folder / 'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n')
print(json.dumps(dict(unique_closed_models=len(records), new_BB_full_records=[dict(seed=records[i]['seed'],
    max_difference=records[i]['max_absolute_difference'], bitwise=records[i]['bitwise_identical'],
    metrics=records[i]['replayed_metrics']) for i in new_ids],
    literal=reported['paper_literal'], conditional=reported['gaussian_conditional'],
    paired_ci95=reported['hierarchical_paired_bootstrap_ci95']), indent=2))
