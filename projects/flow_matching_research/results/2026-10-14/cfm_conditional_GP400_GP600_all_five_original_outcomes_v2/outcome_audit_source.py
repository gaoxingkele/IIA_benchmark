"""Audit all five frozen GP400/600 pairs, retaining independently observed failures.

This reads complete original checkpoints/proofs/grids without fitting or ODE calls.
A failed full grid keeps its metric missing and prevents a finite five-seed mean.
CFM-TS: https://openreview.net/forum?id=Hqn4Aj7xrQ.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from scripts.flow_matching import audit_cfm_gp_conditional_five_seed_budgets_v1 as planned
from scripts.flow_matching import audit_cfm_conditional_GP6001104_full80_failure_v1 as failed_native

ROOT, read, sha = planned.ROOT, planned.read, planned.sha


def summarize_outcomes(rows):
    if len(rows) != 5 or {r['seed'] for r in rows} != set(planned.SEEDS):
        raise ValueError('Every unique original seed1103..1107 must have both outcomes')
    summaries = {}
    for epoch in (400, 600):
        values = [r[str(epoch)] for r in rows]
        for value in values:
            if value['status'] == 'complete_finite_full_grid':
                if not isinstance(value['MSE'], (float, int)) or not math.isfinite(value['MSE']) or value['MSE'] < 0:
                    raise ValueError('A successful original full grid needs a finite nonnegative MSE')
            elif value['status'] == 'full_once_observed_numerical_failure':
                if value['MSE'] is not None or not value['failed_trajectory_ids']:
                    raise ValueError('A failed full grid has no finite-subset aggregate')
            else:
                raise ValueError('Pending, partial and missing outcomes cannot close a five-seed comparison')
        failures = sum(v['status'] != 'complete_finite_full_grid' for v in values)
        mse = [v['MSE'] for v in values]
        summaries[str(epoch)] = dict(required_seeds=5, finite_full_grid_models=5 - failures,
            full_once_observed_failures=failures, observed_failure_fraction=failures / 5,
            full_five_seed_MSE_mean=statistics.mean(mse) if not failures else None,
            full_five_seed_MSE_sample_std=statistics.stdev(mse) if not failures else None,
            successful_subset_mean_published=False)
    return summaries


def verify_config(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen original outcome audit evidence changed: ' + pin['path'])
    comparison = read(ROOT / config['comparison_configuration'])
    runtime = planned.verify_config(comparison)
    failures = config['failed_observations']
    expected = 'cfm_ts_original__appendix_d__gp_gaussian_conditional_e600__sine__seed1104'
    if (config['original_scope'] != comparison['original_scope'] or len(failures) != 1
            or failures[0]['canonical_id'] != expected):
        raise ValueError('Retain all five original pairs and the actual full seed1104 failure')
    return comparison, runtime


def audit(config):
    import numpy as np
    comparison, runtime = verify_config(config)
    snapshot = read(ROOT / runtime['state_root'] / 'verified_full_model_records.json')
    records = {r['original_canonical_id']: r for r in snapshot['records']}
    if len(records) != len(snapshot['records']):
        raise ValueError('Native certificate records must be unique')
    failure_binding = config['failed_observations'][0]
    missing = [identity for pair in planned.identities() for identity in pair
        if identity not in records and identity != failure_binding['canonical_id']]
    if missing:
        raise ValueError('Original full outcome certificates remain missing: ' + ', '.join(missing))
    if failure_binding['canonical_id'] in records:
        raise ValueError('Never silently replace a preserved failed seed with another successful execution')
    failure_config = read(ROOT / failure_binding['independent_configuration'])
    stored_failure = read(ROOT / failure_binding['independent_proof'])
    fresh_failure = failed_native.post_audit(failure_config)
    if any(fresh_failure[key] != value for key, value in stored_failure.items() if key != 'captured_utc'):
        raise ValueError('Independent original failed-grid proof changed')
    failed_runtime = read(ROOT / failure_config['native_configuration'])
    failed_job, _ = failed_native.verify(failed_runtime)
    import torch
    checkpoint = torch.load(failed_runtime['checkpoint_path'], map_location='cpu', weights_only=False)
    if (checkpoint['job'] != failed_job or checkpoint['optimizer_updates'] != 375000
            or len(checkpoint['training_losses']) != 600 or not np.isfinite(checkpoint['training_losses']).all()
            or not all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values())):
        raise ValueError('Actual complete finite trained seed1104 weights and full history required')
    rows = []
    baseline_errors = []
    for seed, pair in zip(planned.SEEDS, planned.identities()):
        row = dict(seed=seed)
        reference = None
        histories = []
        errors = []
        for epoch, identity in zip((400, 600), pair):
            if identity == failed_job['id']:
                arrays_path = Path(failed_runtime['output_root']) / 'all_original_grid_attempts.npz'
                histories.append(checkpoint['training_losses'])
                row[str(epoch)] = dict(canonical_id=identity, status='full_once_observed_numerical_failure',
                    MSE=None, failed_trajectory_ids=stored_failure['failed_trajectory_ids'],
                    finite_trajectories=stored_failure['finite_full_grid_trajectories'], attempted_trajectories=80,
                    original_epochs=600, optimizer_updates=375000,
                    independent_proof=failure_binding['independent_proof'],
                    independent_proof_sha256=sha(ROOT / failure_binding['independent_proof']))
                errors.append(None)
            else:
                record = records[identity]
                planned.previous.cold.validate_closed_primary_record(runtime, record)
                if (record['seed'] != seed or record['epochs'] != epoch
                        or record['optimizer_updates'] != {400: 250000, 600: 375000}[epoch]
                        or record['full_replay_shape'] != [80, 500, 1]):
                    raise ValueError('Full original paired budget and complete grid required')
                result = read(ROOT / record['result_path'])
                histories.append(result['training_losses'])
                arrays_path = Path(record['replay_arrays'])
                row[str(epoch)] = dict(canonical_id=identity, status='complete_finite_full_grid',
                    MSE=record['replayed_metrics']['MSE'], failed_trajectory_ids=[], record=record)
            with np.load(arrays_path, allow_pickle=False) as arrays:
                identity_arrays = {key: arrays[key].copy() for key in ('test_ids', 'truth', 'times')}
                if reference is None:
                    reference = identity_arrays
                elif not all(np.array_equal(reference[key], value) for key, value in identity_arrays.items()):
                    raise ValueError('Paired budgets must use identical full trajectory identities, truth and times')
                if arrays['predictions'].shape != (80, 500, 1) or len(set(arrays['test_ids'].tolist())) != 80:
                    raise ValueError('Every original full grid required')
                if identity != failed_job['id']:
                    if not np.isfinite(arrays['predictions']).all():
                        raise ValueError('Complete successful grids must be finite')
                    value = np.square(arrays['predictions'].astype(np.float64) - arrays['truth']).mean(axis=(1, 2))
                    if not np.isclose(value.mean(), row[str(epoch)]['MSE'], rtol=1e-12, atol=1e-12):
                        raise ValueError('Every-coordinate MSE differs')
                    errors.append(value)
        row['actual_first400_loss_prefix'] = planned.prefix_evidence(*histories)
        row['paired_full_grid_MSE_difference_600_minus_400'] = (
            float((errors[1] - errors[0]).mean()) if errors[1] is not None else None)
        row['original_whole_trajectory_ids'] = reference['test_ids'].tolist()
        baseline_errors.append(errors[0])
        rows.append(row)
    summary = summarize_outcomes(rows)
    # The successful 400-epoch arm retains all five groups. No bootstrap of four surviving 600 seeds.
    baseline = np.asarray(baseline_errors)
    rng = np.random.default_rng(comparison['bootstrap_seed'])
    samples = []
    for start in range(0, comparison['bootstrap_replicates'], 256):
        count = min(256, comparison['bootstrap_replicates'] - start)
        outer = rng.integers(0, 5, size=(count, 5))
        inner = rng.integers(0, 80, size=(count, 5, 80))
        samples.extend(baseline[outer[:, :, None], inner].mean(axis=(1, 2)).tolist())
    summary['400']['hierarchical_seed_whole_trajectory_bootstrap_mean_CI95'] = np.quantile(samples, [.025, .975]).tolist()
    summary['600']['hierarchical_seed_whole_trajectory_bootstrap_mean_CI95'] = None
    return dict(captured_utc=datetime.now(timezone.utc).isoformat(), original_scope=config['original_scope'],
        original_registered_CFM_slots=285, original_required_seed_pairs=5, original_required_trained_models=10,
        all_five_original_full_training_and_scoring_outcomes_audited=True,
        complete_finite_grid_models=9, full_once_observed_failed_grid_models=1,
        all_original_grid_coordinates_checked=400000, rows=rows, budget_outcome_summaries=summary,
        bootstrap_seed=comparison['bootstrap_seed'], bootstrap_replicates=comparison['bootstrap_replicates'],
        full_five_seed_finite_paired_performance_difference=None,
        all_ten_original_finite_native_certificates_passed=False, failed_seed_exclusion=False,
        additional_ODE_solver_calls=0, additional_optimizer_updates=0,
        intermediate_weight_or_optimizer_equality_claimed=False, training_root_cause_certified=False,
        author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False,
        boundary='All five frozen joint data/training seed pairs retained. Seed1104 completed600 training but failed a full original grid; full600 five-seed MSE and paired mean remain undefined. Per-successful-pair differences are individual outcomes, not a surviving-subset performance mean. Baseline400 CI is descriptive at five outer groups. Loss-prefix equality does not certify intermediate weights/optimizer state or isolate training RNG.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    result = audit(config)
    target = ROOT / config['output_report']
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['budget_outcome_summaries']))


if __name__ == '__main__':
    main()
