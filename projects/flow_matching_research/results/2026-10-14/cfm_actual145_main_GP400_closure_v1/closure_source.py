from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching.audit_cfm_full_arrays_v1 import audit_arrays

root, read, sha = cold.ROOT, cold.original.read, cold.streaming_sha
runtime_path = root / 'configs/runtime/fm_cfm_completion_replay_queue.v4.json'
supp_path = root / 'configs/runtime/fm_cfm_metadata_entry_full_replay.v2.json'
runtime, supplement = read(runtime_path), read(supp_path)
primary_bytes = (root / runtime['state_root'] / 'verified_full_model_records.json').read_bytes()
primary = json.loads(primary_bytes)
records = {r['original_canonical_id']: r for r in primary['records']}
for record in read(root / supplement['state_root'] / 'verified_full_model_records.json')['records']:
    identity = record['original_canonical_id']
    if identity in records: assert records[identity] == record
    records[identity] = record
assert len(records) >= 145 and len(runtime['routes']) == len(supplement['routes']) == 285
relative = 'projects/flow_matching_research/results/2026-10-14/cfm_actual145_main_GP400_closure_v1'
folder = root / relative
folder.mkdir(exist_ok=False)
(folder / 'frozen_actual_primary_full_models.json').write_bytes(primary_bytes)
new = []
for seed in [1103, 1104, 1105]:
    identity = 'cfm_ts_original__main_text__gp_gaussian_conditional_e400__sine__seed' + str(seed)
    record = records[identity];cold.validate_closed_primary_record(runtime, record)
    assert record['epochs'] == 400 and record['optimizer_updates'] == 62400 and record['full_replay_shape'] == [20, 500, 1]
    case = folder / ('new_closed_main_GP400_seed' + str(seed));case.mkdir()
    nativepath = root / runtime['child_config_root'] / (identity + '.native.json')
    reviewpath = root / runtime['child_config_root'] / (identity + '.review.json')
    native, review = read(nativepath), read(reviewpath)
    resultpath = root / record['result_path'];result = read(resultpath);job = result['job']
    sources = {
        'original_result.json': resultpath,
        'original_guarded_receipt.json': root / Path(job['output_directory']).parent.parent / 'receipts' / (job['id'] + '.json'),
        'native_config.json': nativepath,
        'native_report.json': root / native['output_directory'] / 'checkpoint_replay_audit.json',
        'native_guarded_receipt.json': root / native['state_root'] / 'receipt.json',
        'independent_config.json': reviewpath,
        'independent_proof.json': root / review['output_proof'],
        'independent_guarded_receipt.json': root / review['state_root'] / 'receipt.json',
    }
    for name, path in sources.items(): (case / name).write_bytes(path.read_bytes())
    predictionpath = next(a['path'] for a in result['artifacts'] if Path(a['path']).name == 'predictions.npz')
    with np.load(predictionpath, allow_pickle=False) as arrays, np.load(root / job['data_path'], allow_pickle=False) as data:
        metrics = audit_arrays(job, result, arrays, data)
        assert not set(data['train_ids']) & set(data['test_ids'])
    new.append(dict(record=record, full_original_array_metrics_recomputed=metrics,
                    original_guarded_source_receipt_SHA256=sha(sources['original_guarded_receipt.json'])))
window = root / 'experiments/runs/fm_cfm_recurring_ready_review_windows_v3/window_1_receipt.json'
receipt = read(window)
assert receipt['state'] == 'release_all_target_proofs_complete' and not receipt['shared_lock_owned']
assert not receipt['original_guards_changed'] and receipt['numeric_workers_stopped_or_restarted'] == 0
(folder / 'completed_first_original_guard_recurring_window_receipt.json').write_bytes(window.read_bytes())
closure = dict(captured_utc=datetime.now(timezone.utc).isoformat(), full_unique_canonical_models=len(records),
    required_original_models=285, primary_closed_models=len(primary['records']), deduplicated_full_records=list(records.values()),
    new_fully_closed_original_main_models=new, main_text_epochs=400, main_text_updates_per_model=62400,
    new_original_grid_coordinates_recomputed=30000, full_model_replays_bitwise_identical=True,
    full_actual_guarded_native_and_independent_proofs_rechecked=True,
    original_data_and_model_and_source_and_proof_SHA256_rechecked=True,
    source_group_boundary='Main20 trajectories/62400 updates differs from Appendix80 trajectories/250000; do not merge these protocols.',
    additional_optimizer_updates=0, additional_model_inference=0,
    original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
    author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False)
(folder / 'actual145_unique_full_models_and_three_original_main_GP400_closure.json').write_text(json.dumps(closure, indent=2) + '\n')
(folder / 'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n')
print(json.dumps({'unique_canonical_models':len(records), 'new_main_metrics':[dict(seed=r['record']['seed'],metrics=r['full_original_array_metrics_recomputed']) for r in new]},indent=2))
