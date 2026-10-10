"""Independently recompute all full CFM-TS prediction metrics and seed aggregates."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prediction_metrics(predictions, truth):
    import numpy as np
    error = predictions.astype(np.float64) - truth
    squared = np.square(error)
    per_trajectory = squared.mean(axis=(1, 2))
    return dict(MSE=float(squared.mean()), MAE=float(np.abs(error).mean()),
        RMSE=float(np.sqrt(squared.mean())), trajectory_MSE_mean=float(per_trajectory.mean()),
        trajectory_MSE_std=float(per_trajectory.std(ddof=1)) if len(per_trajectory)>1 else None,
        MSE_by_dimension=squared.mean(axis=(0, 1)).tolist()), per_trajectory


def audit_arrays(job, result, arrays, data):
    import numpy as np
    if (result['epochs_completed'] != job['epochs'] or result['optimizer_updates'] != job['expected_optimizer_updates']
            or result['training_items'] != job['training_items']
            or len(result['training_losses']) != job['epochs']
            or not np.isfinite(result['training_losses']).all()):
        raise ValueError('Full frozen training budget/history differs')
    ids = data['test_ids']
    if (not np.array_equal(ids, arrays['test_ids']) or not np.array_equal(ids, result['test_trajectory_ids'])
            or not np.array_equal(data['train_ids'], result['training_trajectory_ids'])):
        raise ValueError('Full original training/test trajectory identities differ')
    shape = (job['test_trajectories'], job['test_points_per_trajectory'], job['observed_dimensions'])
    if (arrays['predictions'].shape != shape or arrays['truth'].shape != shape
            or not np.isfinite(arrays['predictions']).all()
            or not np.array_equal(arrays['truth'], data['test_states'][ids])
            or not np.array_equal(arrays['times'], data['test_times'][ids])
            or not np.array_equal(arrays['initial_states'], data['initial_states'][ids])):
        raise ValueError('Full original evaluation arrays/grid/initial states differ')
    metrics, per_trajectory = prediction_metrics(arrays['predictions'], arrays['truth'])
    if not np.allclose(per_trajectory, arrays['per_trajectory_MSE'], rtol=1e-12, atol=1e-12):
        raise ValueError('Full per-trajectory squared errors differ')
    for key, expected in metrics.items():
        actual = result['metrics'][key]
        if expected is None:
            if actual is not None:
                raise ValueError('Single-trajectory uncertainty must be missing')
        elif not np.allclose(expected, actual, rtol=1e-12, atol=1e-12):
            raise ValueError('Independent full metric differs: ' + key)
    return metrics


def independent_stats(values):
    import numpy as np
    from scipy.stats import t
    if not values:
        return dict(n=0, mean=None, std=None, se=None, ci95_low=None, ci95_high=None)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values)>1 else None
    se = std / len(values)**.5 if std is not None else None
    margin = float(t.ppf(.975, len(values)-1))*se if se is not None else None
    return dict(n=len(values), mean=mean, std=std, se=se,
        ci95_low=mean-margin if margin is not None else None,
        ci95_high=mean+margin if margin is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    for receipt in config['source_receipts']:
        if sha(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen full independent audit source differs')
    capture = ROOT / config['capture']
    snapshot = read(capture / 'execution_audit.json')
    target = capture / 'independent_full_metrics_audit.json'
    if target.exists():
        raise FileExistsError('Preserve independent audit capture')
    import numpy as np
    records = []
    groups = defaultdict(list)
    for row in snapshot['jobs']:
        if row['status'] != 'completed':
            continue
        path = ROOT / row['result_path']
        if sha(path) != row['result_sha256']:
            raise ValueError('Frozen completed result differs')
        result = read(path)
        job = result['job']
        if sha(ROOT / job['data_path']) != job['data_sha256']:
            raise ValueError('Registered original full dataset differs')
        if sha(ROOT / job['model_config']) != job['model_config_sha256']:
            raise ValueError('Registered original full model config differs')
        for artifact in result['artifacts']:
            if sha(artifact['path']) != artifact['sha256']:
                raise ValueError('Full training/scoring artifact differs')
        arrays_path = next(a['path'] for a in result['artifacts'] if Path(a['path']).name == 'predictions.npz')
        with np.load(arrays_path, allow_pickle=False) as arrays, np.load(ROOT / job['data_path'], allow_pickle=False) as data:
            metrics = audit_arrays(job, result, arrays, data)
        record = dict(original_canonical_id=row['id'], execution_id=row['execution_id'],
            seed=row['seed'], track=row['track'], dataset=row['dataset'], variant=row['variant'], epochs=row['epochs'],
            result_path=row['result_path'], result_sha256=sha(path), scores_path=arrays_path,
            scores_sha256=sha(arrays_path), original_data_sha256=job['data_sha256'],
            metrics=metrics, full_training_budget_and_history_verified=True,
            full_array_and_original_data_identity_verified=True, all_metrics_independently_recomputed=True,
            checkpoint_sha256_verified=True, checkpoint_contents_audit_pending=True)
        records.append(record)
        groups[row['track'], row['dataset'], row['variant'], row['method'], row['epochs']].append(record)
    if len(records) != snapshot['job_counts']['completed'] or len({r['original_canonical_id'] for r in records}) != len(records):
        raise ValueError('Canonical full completed-slot coverage differs')
    for group in snapshot['seed_aggregates']:
        key = (group['track'], group['dataset'], group['variant'], group['method'], group['epochs'])
        for metric in ('MSE', 'MAE', 'RMSE'):
            independent = independent_stats([r['metrics'][metric] for r in groups[key]])
            for statistic, value in independent.items():
                actual = group[metric + '_' + statistic]
                if (value is None and actual is not None) or (value is not None and not math.isclose(value, actual, rel_tol=1e-12, abs_tol=1e-12)):
                    raise ValueError('Independent original seed aggregate differs')
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        source_capture_sha256=sha(capture / 'execution_audit.json'), configuration=args.config,
        configuration_sha256=sha(ROOT / args.config), audited_full_canonical_slots=len(records),
        canonical_group_count=len(snapshot['seed_aggregates']),
        full_five_seed_groups=sum(r['completed_seeds']==r['required_seeds'] for r in snapshot['seed_aggregates']),
        all_completed_full_arrays_and_metrics_verified=True, all_seed_aggregates_independently_verified=True,
        original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
        additional_training_seed_slots=0, all_original_experiments_complete=False,
        checkpoint_contents_audit_pending=True, records=records,
        boundary='Full trained-model prediction arrays and frozen original data; not smoke. Checkpoint hashes verified; content inspection and trained-model re-inference remain separate obligations. Author code and original random trajectories were unavailable; exact author equivalence is not certified.')
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(full_canonical_slots=len(records), full_five_seed_groups=report['full_five_seed_groups'], all_metric_types_recomputed=True)))


if __name__ == '__main__':
    main()
