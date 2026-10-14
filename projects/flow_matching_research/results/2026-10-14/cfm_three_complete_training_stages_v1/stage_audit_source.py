"""Verify complete stored grids; preserve pending native/independent gates."""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching.audit_cfm_full_arrays_v1 import audit_arrays
from scripts.flow_matching.run_cfm_completion_replay_queue_v1 import check_binding

root, read, sha = cold.ROOT, cold.original.read, cold.streaming_sha
relative = 'projects/flow_matching_research/results/2026-10-14/cfm_three_complete_training_stages_v1'
folder = root / relative
folder.mkdir(exist_ok=False)
original_path = root / 'configs/experiments/fm_cfm_ts_original_queue.v1.json'
recovery_path = root / 'configs/experiments/fm_cfm_metadata_entry_full_retry_queue.v2.json'
original, recovery = read(original_path), read(recovery_path)
old = {j['id']: j for j in original['jobs']}
jobs = [j for j in recovery['jobs'] if j['id'] in recovery['recovery_job_ids']]
gp_id = 'cfm_ts_original__appendix_d__gp_gaussian_conditional_e400__sine__seed1106'
jobs.append(old[gp_id])
records, pins = [], [original_path, recovery_path, root / 'scripts/flow_matching/audit_cfm_full_arrays_v1.py',
                    root / 'scripts/flow_matching/run_cfm_streaming_idle_handoff_v3.py',
                    root / 'scripts/flow_matching/run_cfm_completion_replay_queue_v1.py']
for job in jobs:
    identity = job.get('recovery_original_job_id', job['id'])
    check_binding(old[identity], job)
    queue = recovery if job['id'] != identity else original
    resultpath = root / job['output_directory'] / 'result.json'
    receiptpath = root / queue['state_root'] / 'receipts' / (job['id'] + '.json')
    result, receipt = read(resultpath), read(receiptpath)
    assert receipt['exit_code'] == 0 and receipt['complete'] and not receipt['resource_observations']['resource_guard_aborted']
    assert sha(root / job['data_path']) == job['data_sha256']
    assert sha(root / job['model_config']) == job['model_config_sha256']
    for artifact in result['artifacts']:
        assert sha(artifact['path']) == artifact['sha256']
    predictions = next(a['path'] for a in result['artifacts'] if Path(a['path']).name == 'predictions.npz')
    with np.load(predictions, allow_pickle=False) as arrays, np.load(root / job['data_path'], allow_pickle=False) as data:
        metrics = audit_arrays(job, result, arrays, data)
    prefix = None
    if job['id'] != identity:
        prefixpath = root / old[identity]['output_directory'] / 'progress.json'
        previous = read(prefixpath)
        epoch = previous['epoch']
        prefix = dict(original_preserved_last_epoch=epoch,
                      original_last_epoch_loss=previous['loss'],
                      new_full_training_same_epoch_loss=result['training_losses'][epoch - 1],
                      last_preserved_loss_bitwise_identical=(np.float64(previous['loss']).tobytes() ==
                                                           np.float64(result['training_losses'][epoch - 1]).tobytes()),
                      original_unsaved_partial_model_available=False,
                      full_optimizer_or_all_weight_trajectory_equivalence_certified=False,
                      unsaved_partial_epoch_credit=0)
        pins.append(prefixpath)
    case = folder / ('BB400_seed' + str(job['seed']) if job['method'] == 'bb' else 'GP400_seed' + str(job['seed']))
    case.mkdir()
    (case / 'original_complete_result.json').write_bytes(resultpath.read_bytes())
    (case / 'original_complete_guarded_receipt.json').write_bytes(receiptpath.read_bytes())
    record = dict(canonical_id=identity, execution_id=job['id'], seed=job['seed'],
        track=job['track'], variant=job['variant'], epochs_completed=result['epochs_completed'],
        optimizer_updates=result['optimizer_updates'], full_prediction_shape=result['test_prediction_shape'],
        full_original_data_arrays_and_training_history_verified=True, all_original_metrics_recomputed=metrics,
        original_source_exit_code=receipt['exit_code'], resource_guard_aborted=False,
        result_path=resultpath.relative_to(root).as_posix(), result_sha256=sha(resultpath),
        receipt_path=receiptpath.relative_to(root).as_posix(), receipt_sha256=sha(receiptpath),
        checkpoint_and_prediction_artifact_SHA256_verified=True, recovery_prefix_observation=prefix,
        additional_model_inference=0, additional_optimizer_updates=0,
        new_closed_native_plus_independent_certificate_awarded_by_this_audit=False,
        native_complete_model_replay_and_guarded_independent_review_remain_separate=True,
        strict_TAB_result=False, author_equivalence_certified=False)
    records.append(record)
    pins += [resultpath, receiptpath, root / job['data_path'], root / job['model_config']]
    pins += [Path(a['path']) for a in result['artifacts']]
pins = list(dict.fromkeys(pins))
config = dict(complete_source_results=[dict(canonical_id=r['canonical_id'], result_path=r['result_path']) for r in records],
    source_receipts=[dict(path=p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p), sha256=sha(p)) for p in pins],
    citation='https://openreview.net/forum?id=Hqn4Aj7xrQ',
    original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
    boundary='All three full trained/scored source models and every stored grid independently recomputed, but do not award guarded model-replay-plus-independent completion. Both raw metadata retries retain zero partial credit. No solver/model inference, optimizer, threshold or guard change.')
with (root / 'configs/reproducibility/fm_cfm_three_new_full_training_stage_audit.2026-10-14_v1.json').open('x', encoding='utf-8') as stream:
    json.dump(config, stream, indent=2)
report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), new_full_training_and_scoring_models=3,
    audited_prediction_coordinates=60000, source_guards_successful=True, records=records,
    new_guarded_native_plus_independent_certificates_awarded_by_this_audit=0,
    completed_first_and_second_metadata_entry_recovery=True,
    actual_source_and_original_failures_preserved=True, original_scope=config['original_scope'],
    all_original_experiments_complete=False)
(folder / 'three_full_source_training_scoring_and_stored_grid_audit.json').write_text(json.dumps(report, indent=2) + '\n')
(folder / 'stage_audit_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n')
print(json.dumps({'full_source_models': 3, 'verified_complete_coordinates': 60000,
    'new_guarded_certificates_awarded': 0,
    'metrics_and_prefix': [dict(canonical_id=r['canonical_id'], metrics=r['all_original_metrics_recomputed'],
                              prefix=r['recovery_prefix_observation']) for r in records]}, indent=2))
