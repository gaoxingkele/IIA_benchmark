"""Independently audit all original grids and the preserved failing RHS tail.

Reads the already completed single native observation, without model inference,
optimizer updates or solver changes. Failure remains an explicit failure;79
finite trajectories cannot be promoted to complete80-trajectory performance.
CFM-TS: https://openreview.net/forum?id=Hqn4Aj7xrQ.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from scripts.flow_matching import audit_cfm_sine600_full80_failure_observation_v1 as native

ROOT = native.ROOT


def tail_evidence(record, float32_max):
    observations = record['last_RHS_observations']
    first_velocity = next((r for r in observations if not r['velocity_finite']), None)
    first_state = next((r for r in observations if not r['state_finite']), None)
    finite_velocity_then_nonfinite = bool(first_velocity and any(
        r['call'] < first_velocity['call'] and r['velocity_finite'] for r in observations))
    return dict(trajectory_id=record['trajectory_id'], error_type=record.get('error_type'),
                error_message=record.get('error_message'), RHS_calls=record['RHS_calls'],
                last_finite_state_magnitude=record['maximum_finite_state_abs'],
                last_finite_velocity_magnitude=record['maximum_finite_velocity_abs'],
                float32_max_finite=float32_max,
                finite_velocity_near_float32_limit=record['maximum_finite_velocity_abs'] > .9*float32_max,
                nonfinite_velocity_observed=first_velocity is not None,
                first_nonfinite_velocity_tail_observation=first_velocity,
                first_nonfinite_state_tail_observation=first_state,
                finite_state_at_first_nonfinite_velocity=bool(first_velocity and first_velocity['state_finite']),
                finite_velocity_then_nonfinite=finite_velocity_then_nonfinite,
                learned_training_failure_root_cause_certified=False)


def audit(config):
    import numpy as np
    for pin in config['source_receipts']:
        if native.sha(ROOT/pin['path']) != pin['sha256']:
            raise ValueError('Frozen original observation or audit artifact changed: '+pin['path'])
    runtime = native.read(ROOT/config['native_configuration'])
    job, model = native.verify(runtime)
    output = Path(runtime['output_root'])
    report = native.read(output/'full80_native_failure_observation.json')
    receipt = native.read(ROOT/runtime['state_root']/'receipt.json')
    status = native.read(ROOT/runtime['state_root']/'status.json')
    if (receipt['exit_code'] != 0 or receipt['resource_guard_aborted']
            or not receipt['full80_failure_observation_captured']
            or receipt['original_benchmark_scoring_complete']
            or status['config_sha256'] != native.sha(ROOT/config['native_configuration'])
            or report['canonical_job'] != job or report['checkpoint_sha256'] != runtime['checkpoint_sha256']
            or report['solver_parameters'] != runtime['solver_parameters']
            or not report['original_native_simulate_code_unchanged']
            or not report['observer_does_not_replace_outputs']
            or report['numeric_optimizer_updates_executed'] != 0
            or report['original_scoring_certified_complete']
            or report['full_original_MSE'] is not None
            or report['censored_finite_subset_promoted_as_original_result']):
        raise ValueError('One actual unchanged full80 diagnostic, never a benchmark result, required')
    summary = native.validity_summary(report['records'],80)
    if any(report[k] != v for k,v in summary.items()):
        raise ValueError('Original complete coverage and failure accounting differ')
    for row in report['records']:
        if native.read(output/('trajectory_%02d.json'%row['position'])) != row:
            raise ValueError('Independent individual trajectory receipt differs')
    with np.load(ROOT/job['data_path'],allow_pickle=False) as source, \
            np.load(output/'all_original_grid_attempts.npz',allow_pickle=False) as arrays:
        ids = source['test_ids']
        train_ids = source['train_ids']
        expected_shape = (80,500,1)
        if (arrays['predictions'].shape != expected_shape
                or not np.array_equal(arrays['test_ids'],ids)
                or set(ids.tolist()) & set(train_ids.tolist())
                or not np.array_equal(arrays['truth'],source['test_states'][ids])
                or not np.array_equal(arrays['times'],source['test_times'][ids])):
            raise ValueError('Every original held-out trajectory, time, truth and coordinate required')
        mask = np.array([row['finite_full_grid_completed'] for row in report['records']],dtype=bool)
        if not np.array_equal(mask,arrays['finite_trajectory_mask']):
            raise ValueError('Every failed trajectory remains in full arrays')
        checked_points = 0
        for position,row in enumerate(report['records']):
            if row['position'] != position or row['trajectory_id'] != int(ids[position]):
                raise ValueError('Original full ordering and every trajectory required')
            if row['initial_state'] != source['initial_states'][ids[position]].tolist():
                raise ValueError('Original initial state differs')
            prediction = arrays['predictions'][position]
            if mask[position]:
                if not np.isfinite(prediction).all():
                    raise ValueError('Finite trajectory receipt has nonfinite grid coordinates')
                mse = float(np.square(prediction.astype(np.float64)-arrays['truth'][position]).mean())
                if mse != row['MSE']:
                    raise ValueError('Independent all-coordinate trajectory MSE differs')
            elif not np.isnan(prediction).all() or 'MSE' in row:
                raise ValueError('Failed grid must remain allNaN without invented metric')
            checked_points += prediction.size
        training = source['train_states'][train_ids]
        limits = dict(min=float(training.min()),max=float(training.max()))
        failed = [row for row in report['records'] if not row['finite_full_grid_completed']]
        failure_evidence = []
        for row in failed:
            detail = tail_evidence(row,float(np.finfo(np.float32).max))
            detail.update(original_initial_state=row['initial_state'],actual_training_state_range=limits,
                          initial_state_inside_observed_training_state_range=bool(
                              all(limits['min'] <= x <= limits['max'] for x in row['initial_state'])))
            failure_evidence.append(detail)
    return dict(captured_utc=datetime.now(timezone.utc).isoformat(), canonical_job_id=job['id'],
                independent_full_grid_shape=[80,500,1], independent_checked_grid_coordinates=checked_points,
                original_train_test_trajectory_split_disjoint=True, every_original_truth_time_identity_checked=True,
                every_finite_trajectory_MSE_recomputed=True, **summary,
                array_sha256=native.sha(output/'all_original_grid_attempts.npz'),
                native_report_sha256=native.sha(output/'full80_native_failure_observation.json'),
                native_guard_receipt_sha256=native.sha(ROOT/runtime['state_root']/'receipt.json'),
                checkpoint_sha256=runtime['checkpoint_sha256'],data_sha256=job['data_sha256'],
                solver_parameters=runtime['solver_parameters'], failure_evidence=failure_evidence,
                state_and_velocity_growth_and_nonfinite_output_observed=True,
                training_root_cause_or_author_code_bug_certified=False,
                additional_ODE_solver_calls=0, additional_optimizer_updates=0,
                original_author_equivalence_certified=False, strict_TAB_result=False,
                all_original_experiments_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    config = native.read(ROOT/args.config)
    result = audit(config)
    destination = ROOT/config['output_report']
    if destination.exists():
        raise FileExistsError('Preserve the one independent post-observation audit')
    native.write(destination,result)
    print(dict(attempted=result['attempted_original_trajectories'],finite=result['finite_full_grid_trajectories'],
               failed=result['failed_trajectory_ids'],checked_coordinates=result['independent_checked_grid_coordinates']))


if __name__ == '__main__':
    main()
