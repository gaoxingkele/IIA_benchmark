"""Wait for every full original GP400 target pair, then audit all seed groups.

Only saved, guarded native-plus-independent full-grid models qualify. Original
seeds jointly change data generation/split and training RNG: report that joint
variation, not an isolated training-only effect. No fitting/inference/tuning.
CFM-TS: https://openreview.net/forum?id=Hqn4Aj7xrQ.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold

ROOT, read, sha = cold.ROOT, cold.original.read, cold.streaming_sha
SEEDS = [1103, 1104, 1105, 1106, 1107]


def identities():
    return [['cfm_ts_original__appendix_d__gp_' + mode + '_e400__sine__seed' + str(seed)
             for mode in ['paper_literal', 'gaussian_conditional']] for seed in SEEDS]


def select_complete_pairs(records, pairs):
    if pairs != identities():
        raise ValueError('All five preregistered original seed pairs are required')
    byid = {r['original_canonical_id']: r for r in records}
    if len(byid) != len(records):
        raise ValueError('Duplicate canonical slots cannot become more seed repetitions')
    missing = [identity for pair in pairs for identity in pair if identity not in byid]
    return None if missing else [[byid[identity] for identity in pair] for pair in pairs], missing


def grouped_summary(literal, conditional, seeds, bootstrap_seed, replicates):
    import numpy as np
    a, b = np.asarray(literal, dtype=np.float64), np.asarray(conditional, dtype=np.float64)
    if (seeds != SEEDS or a.shape != (5, 80) or b.shape != a.shape or replicates < 1000
            or not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any()):
        raise ValueError('Every five original paired models and all80 full trajectory errors required')
    delta = b - a
    rng = np.random.default_rng(bootstrap_seed)
    samples = []
    # Bounded batches keep this read-only analysis below100 MiB of work arrays.
    for start in range(0, replicates, 256):
        count = min(256, replicates - start)
        outer = rng.integers(0, 5, size=(count, 5))
        inner = rng.integers(0, 80, size=(count, 5, 80))
        samples.extend(delta[outer[:, :, None], inner].mean(axis=(1, 2)).tolist())
    result = dict(required_original_seed_pairs=SEEDS, complete_seed_pairs=5,
                  whole_test_trajectory_groups_per_model=80,
                  paired_difference_per_joint_data_and_training_seed=delta.mean(axis=1).tolist(),
                  paired_difference_mean=float(delta.mean()),
                  paired_difference_joint_seed_std=float(delta.mean(axis=1).std(ddof=1)),
                  paired_difference_joint_seed_se=float(delta.mean(axis=1).std(ddof=1) / np.sqrt(5)),
                  hierarchical_paired_bootstrap_ci95=[float(v) for v in np.quantile(samples, [.025, .975])],
                  bootstrap_outer_unit='joint data-generation/split and training seed pair',
                  bootstrap_inner_unit='whole paired held-out trajectory within that seed pair',
                  bootstrap_seed=bootstrap_seed, bootstrap_replicates=replicates,
                  isolated_training_rng_uncertainty_estimated=False,
                  only_five_outer_groups_uncertainty_is_descriptive=True)
    for mode, values in [('paper_literal', a), ('gaussian_conditional', b)]:
        means = values.mean(axis=1)
        result[mode] = dict(per_joint_seed_MSE=means.tolist(), mean_MSE=float(means.mean()),
                            joint_seed_std=float(means.std(ddof=1)), joint_seed_se=float(means.std(ddof=1) / np.sqrt(5)))
    return result


def verify_config(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen complete-cohort analysis changed: ' + pin['path'])
    runtime = read(ROOT / config['primary_runtime'])
    if len(runtime['routes']) != 285 or config['paired_canonical_ids'] != identities() or runtime['original_scope'] != config['original_scope']:
        raise ValueError('All original285 slots and full45-paper scope must remain')
    byid = {r['original_canonical_id']: r['job'] for r in runtime['routes']}
    for seed, pair in zip(SEEDS, config['paired_canonical_ids']):
        jobs = [byid[i] for i in pair]
        if (any(j['epochs'] != 400 or j['expected_optimizer_updates'] != 250000 or j['seed'] != seed
                or j['test_trajectories'] != 80 or j['test_points_per_trajectory'] != 500 for j in jobs)
                or jobs[0]['data_sha256'] != jobs[1]['data_sha256']
                or jobs[0]['model_overrides'] != jobs[1]['model_overrides']):
            raise ValueError('Every paired full budget, data identity and seed must match')
        parameters = [read(ROOT / j['model_config'])['parameters'] for j in jobs]
        if ({k for k in set(parameters[0]) | set(parameters[1]) if parameters[0].get(k) != parameters[1].get(k)} != {'target_mode'}
                or [p['target_mode'] for p in parameters] != ['paper_literal', 'gaussian_conditional']):
            raise ValueError('Only the already registered target interpretation may differ')
    return runtime


def audit_complete(config, runtime, snapshot):
    import numpy as np
    selected, missing = select_complete_pairs(snapshot['records'], config['paired_canonical_ids'])
    if missing:
        raise ValueError('No incomplete or successful-subset five-seed result may be published')
    errors = [[], []]
    exact_pairs = []
    for seed, pair in zip(SEEDS, selected):
        previous_ids = previous_truth = previous_times = None
        for position, record in enumerate(pair):
            cold.validate_closed_primary_record(runtime, record)
            if record['epochs'] != 400 or record['optimizer_updates'] != 250000 or record['full_replay_shape'] != [80, 500, 1]:
                raise ValueError('Full original training and replay grid required')
            with np.load(record['replay_arrays'], allow_pickle=False) as arrays:
                prediction, truth = arrays['predictions'], arrays['truth']
                if prediction.shape != (80, 500, 1) or truth.shape != prediction.shape or not np.isfinite(prediction).all() or not np.isfinite(truth).all():
                    raise ValueError('Every paired full coordinate must remain finite')
                if position == 0:
                    previous_ids, previous_truth, previous_times = arrays['test_ids'].copy(), truth.copy(), arrays['times'].copy()
                elif (not np.array_equal(previous_ids, arrays['test_ids']) or not np.array_equal(previous_truth, truth)
                      or not np.array_equal(previous_times, arrays['times'])):
                    raise ValueError('Whole paired trajectory identity, times and truth must match')
                if len(set(arrays['test_ids'].tolist())) != 80:
                    raise ValueError('Distinct complete held-out trajectory groups required')
                error = np.square(prediction.astype(np.float64) - truth).mean(axis=(1, 2))
                if not np.isclose(error.mean(), record['replayed_metrics']['MSE'], rtol=1e-12, atol=1e-12):
                    raise ValueError('Complete saved metrics must match all original coordinates')
                errors[position].append(error)
        exact_pairs.append(dict(seed=seed, records=pair, trajectory_ids=previous_ids.tolist()))
    return dict(captured_utc=datetime.now(timezone.utc).isoformat(), original_scope=config['original_scope'],
                complete_model_count=10, original_registered_CFM_models=285,
                epochs_per_model=400, optimizer_updates_per_model=250000,
                all_complete_prediction_coordinates_recomputed=400000,
                actual_native_and_independent_guarded_proofs_verified=True,
                pairs=exact_pairs,
                **grouped_summary(*errors, SEEDS, config['bootstrap_seed'], config['bootstrap_replicates']),
                full_original_five_seed_target_comparison_complete=True,
                additional_optimizer_updates=0, additional_model_inference=0,
                test_based_recipe_selection=False, author_equivalence_certified=False,
                strict_TAB_result=False, all_original_experiments_complete=False)


def main():
    import os
    import psutil
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / args.config)
    runtime = verify_config(config)
    snapshot_path = ROOT / runtime['state_root'] / 'verified_full_model_records.json'
    base = ROOT / config['state_root']
    if args.probe_only:
        _, missing = select_complete_pairs(read(snapshot_path)['records'], config['paired_canonical_ids'])
        print(dict(required_complete_models=10, original_CFM_scope=285, missing_complete_model_ids=missing, model_training_executed=False))
        return
    base.mkdir(exist_ok=True, parents=True)
    lock = cold.cached_api()['exclusive_lock'](base / 'controller.lock')
    if lock is None:
        raise RuntimeError('Full five-seed comparison already owned')
    try:
        while True:
            snapshot_bytes = snapshot_path.read_bytes()
            snapshot = json.loads(snapshot_bytes)
            selected, missing = select_complete_pairs(snapshot['records'], config['paired_canonical_ids'])
            state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
                         configuration_sha256=sha(ROOT / args.config), required_complete_models=10,
                         missing_complete_model_ids=missing, original_registered_models=285,
                         state='waiting_every_original_full_seed_pair' if missing else 'auditing_all_ten_full_models',
                         all_original_experiments_complete=False)
            cold.original.write(base / 'status.json', state)
            if selected is not None:
                verify_config(config)
                report = audit_complete(config, runtime, snapshot)
                with (ROOT / config['frozen_complete_snapshot']).open('xb') as stream:
                    stream.write(snapshot_bytes)
                report['frozen_complete_snapshot_sha256'] = sha(ROOT / config['frozen_complete_snapshot'])
                with (ROOT / config['output_report']).open('x', encoding='utf-8') as stream:
                    json.dump(report, stream, indent=2)
                cold.original.write(base / 'status.json', dict(state, state='completed_full_five_seed_pair_analysis',
                    output_report=config['output_report'], output_report_sha256=sha(ROOT / config['output_report'])))
                break
            time.sleep(config['poll_seconds'])
    finally:
        lock.close()


if __name__ == '__main__':
    main()
