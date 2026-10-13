"""Audit every complete prediction in the frozen five-slot GP/Sine400 cohort.

This is a post-training result/data audit, not a new training recipe or model
reinference. All trajectories, including large errors, stay in the reported MSE.
CFM-TS paper: https://openreview.net/forum?id=Hqn4Aj7xrQ
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.flow_matching import run_cfm_ts_original_job as source
from scripts.flow_matching.audit_cfm_full_arrays_v1 import audit_arrays, prediction_metrics, independent_stats


def dispersion(errors, trajectory_ids, initial_states, times, predictions, truth):
    import numpy as np
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(trajectory_ids) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('All finite nonnegative trajectory errors must remain')
    order = np.argsort(-values, kind='stable')
    total = float(values.sum())
    records = []
    for position in order[:5]:
        coordinate_error = np.abs(predictions[position].astype(np.float64) - truth[position])
        index = np.unravel_index(int(np.argmax(coordinate_error)), coordinate_error.shape)
        records.append(dict(trajectory_id=int(trajectory_ids[position]), MSE=float(values[position]),
            fraction_of_total_squared_error=float(values[position] / total) if total else 0.,
            initial_state=initial_states[position].tolist(), worst_point_index=int(index[0]),
            worst_time=float(times[position, index[0]]), worst_dimension=int(index[1]),
            worst_absolute_error=float(coordinate_error[index]),
            prediction_at_worst_point=float(predictions[(position,) + index]),
            truth_at_worst_point=float(truth[(position,) + index])))
    return dict(full_trajectory_count=len(values), all_trajectory_MSE=values.tolist(),
        all_trajectory_ids=[int(i) for i in trajectory_ids], full_MSE=float(values.mean()),
        trajectory_MSE_median=float(np.median(values)), trajectory_MSE_p95=float(np.quantile(values, .95)),
        top1_squared_error_fraction=float(values[order[:1]].sum() / total) if total else 0.,
        top5_squared_error_fraction=float(values[order[:5]].sum() / total) if total else 0.,
        largest_error_trajectories=records, trajectories_removed=0,
        ranking_boundary='Median and concentration diagnose the full metric; they never replace full MSE.')


def run(config):
    import numpy as np
    queue = source.read(source.ROOT / config['original_queue'])
    source.verify(queue)
    for pin in config['source_receipts']:
        if source.sha(source.ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen audit input changed: ' + pin['path'])
    jobs = {j['id']: j for j in queue['jobs']}
    required = ['cfm_ts_original__appendix_d__gp_paper_literal_e400__sine__seed' + str(seed) for seed in range(1103, 1108)]
    if config['required_canonical_ids'] != required or config['required_training_seed_count'] != 5:
        raise ValueError('Every original five seed slot remains required')
    completed = config['completed_snapshot_ids']
    if len(completed) != len(set(completed)) or not set(completed) <= set(required):
        raise ValueError('Distinct original completed snapshot slots required')
    pins = {p['path']: p['sha256'] for p in config['source_receipts']}
    records = []
    for identity in completed:
        job = jobs[identity]
        if job['epochs'] != 400 or job['expected_optimizer_updates'] != 250000 or job['training_items'] != 160000:
            raise ValueError('Full original GP training recipe required')
        resultpath = (Path(job['output_directory']) / 'result.json').as_posix()
        if pins.get(resultpath) != source.sha(source.ROOT / resultpath):
            raise ValueError('Completed result must be frozen before the audit')
        result = source.read(source.ROOT / resultpath)
        if not source.complete(job):
            raise ValueError('Actual full native source verification required')
        receiptpath = 'experiments/runs/fm_cfm_ts_original_v1/receipts/' + identity + '.json'
        receipt = source.read(source.ROOT / receiptpath)
        if (pins.get(receiptpath) != source.sha(source.ROOT / receiptpath) or receipt['exit_code'] != 0
                or not receipt['complete'] or receipt['resource_observations']['resource_guard_aborted']):
            raise ValueError('Original successful guarded training receipt required')
        artifact = next(a['path'] for a in result['artifacts'] if Path(a['path']).name == 'predictions.npz')
        with np.load(artifact, allow_pickle=False) as arrays, np.load(source.ROOT / job['data_path'], allow_pickle=False) as data:
            metrics = audit_arrays(job, result, arrays, data)
            if set(data['train_ids']) & set(data['test_ids']):
                raise ValueError('Held-out trajectory groups may not overlap training')
            _, errors = prediction_metrics(arrays['predictions'], arrays['truth'])
            detail = dispersion(errors, arrays['test_ids'], arrays['initial_states'], arrays['times'],
                                arrays['predictions'], arrays['truth'])
            train_initial = data['initial_states'][data['train_ids']]
            test_initial = data['initial_states'][data['test_ids']]
            split = dict(training_groups=len(data['train_ids']), held_out_groups=len(data['test_ids']), overlap=0,
                training_initial_state_min=float(train_initial.min()), training_initial_state_max=float(train_initial.max()),
                held_out_initial_state_min=float(test_initial.min()), held_out_initial_state_max=float(test_initial.max()))
        records.append(dict(canonical_id=identity, seed=job['seed'], epochs=400, optimizer_updates=250000,
            metrics=metrics, dispersion=detail, split_audit=split, result_sha256=source.sha(source.ROOT / resultpath),
            native_guarded_receipt_sha256=source.sha(source.ROOT / receiptpath),
            complete_native_source_and_full_arrays_verified=True,
            maximum_training_epoch_loss=max(result['training_losses']), final_training_epoch_loss=result['training_losses'][-1],
            losses_over1_epoch_count=sum(v > 1. for v in result['training_losses']),
            training_seconds=result['training_seconds'], independent_model_reinference_certified=False))
    values = [r['metrics']['MSE'] for r in records]
    stats = independent_stats(values)
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), required_canonical_ids=required,
        required_training_seed_count=5, completed_snapshot_count=len(records),
        pending_canonical_ids=[i for i in required if i not in completed], records=records,
        completed_snapshot_MSE_summary=stats,
        full_five_seed_training_cohort_complete=len(records) == 5,
        full_five_seed_model_reinference_verified=False,
        performance_boundary='This frozen completed subset is descriptive; no five-seed conclusion, author equivalence or method ranking follows.',
        diagnosis_boundary='Observed seed dispersion, full-error concentration and finite loss spikes are established; their causal mechanism is not certified.',
        test_based_parameter_selection=False, trajectories_removed=0, additional_training_seed_slots=0,
        original_scope=config['original_scope'], all_original_experiments_complete=False,
        author_equivalence_certified=False, strict_TAB_result=False)
    path = source.ROOT / config['output_report']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(completed_snapshot_count=len(records), full_cohort_complete=len(records) == 5,
        MSE_by_seed={r['seed']:r['metrics']['MSE'] for r in records},
        top1_squared_error_fraction_by_seed={r['seed']:r['dispersion']['top1_squared_error_fraction'] for r in records})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(source.read(source.ROOT / args.config))
