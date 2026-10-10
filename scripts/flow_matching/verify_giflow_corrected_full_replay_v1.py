"""Verify every saved original/replayed GiFlow batch, preserving native reductions."""
import argparse
import math
from pathlib import Path

import numpy as np

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write
from scripts.flow_matching.audit_giflow_full_arrays_v2 import batch_errors


def verify_batch(actual, original, windows, targets, tolerance):
    required = {'prediction', 'target', 'masked_flat_positions', 'batch_shape'}
    if set(actual) != required or set(original) != required:
        raise ValueError('All native saved batch arrays required')
    shape = actual['batch_shape']
    if shape.tolist() != [windows, 36, 24, 1]:
        raise ValueError('Complete original BNLC geometry required')
    positions = actual['masked_flat_positions']
    if (positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer)
            or len(positions) != targets or len(np.unique(positions)) != targets
            or np.any(positions < 0) or np.any(positions >= np.prod(shape))):
        raise ValueError('Every unique native masked position required')
    for key in required - {'prediction'}:
        if not np.array_equal(actual[key], original[key]):
            raise ValueError('Original target identity/geometry changed: ' + key)
    if actual['prediction'].shape != (targets,) or original['prediction'].shape != (targets,):
        raise ValueError('Complete masked prediction vector required')
    if not np.isfinite(original['prediction']).all():
        raise ValueError('Finite original predictions required')
    errors = batch_errors(actual['prediction'], actual['target'])
    if not np.allclose(actual['prediction'], original['prediction'], **tolerance):
        raise ValueError('Full CPU/GPU predictions exceed frozen tolerance')
    delta = np.abs(actual['prediction'].astype(np.float64) - original['prediction'].astype(np.float64))
    return dict(windows=windows, targets=targets, independently_recomputed_metrics=errors,
                max_absolute_prediction_difference=float(delta.max()))


def verify(config):
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen full replay verification input changed')
    replay_config = read(ROOT / config['replay_config'])
    for item in replay_config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Actual replay source or original artifact changed')
    receipt = read(ROOT / config['worker_receipt'])
    if receipt['exit_code'] != 0 or receipt['full_model_replay_passed'] is not True or receipt['resource_guard_aborted']:
        raise ValueError('Successful terminal complete replay required')
    proof = read(ROOT / config['reviewed_proof'])
    raw = read(ROOT / config['raw_proof'])
    if sha(ROOT / config['raw_proof']) != proof['original_raw_replay_proof_sha256']:
        raise ValueError('Original raw replay proof changed')
    for key, value in raw.items():
        if key != 'boundary' and proof[key] != value:
            raise ValueError('Reviewed clarification changed actual replay evidence')
    required = dict(full_test_windows=1747, full_test_batches=14, full_masked_test_targets=268613,
                    original_Euler_steps=20, observed_forward_calls=280, observed_window_evaluations=34940,
                    original_batch_size=128, actual_full_training_epochs=68, actual_optimizer_updates=3264,
                    additional_training_seed_slots=0)
    for key, value in required.items():
        if proof.get(key) != value:
            raise ValueError('Complete native budget/coverage required: ' + key)
    for key in ['full_checkpoint_contents_and_reinference_passed', 'full_data_masks_graph_scaler_and_splits_bitwise',
                'actual_tested_weights_restored_bitwise', 'tested_checkpoint_selected_on_validation',
                'source_test_block_unchanged']:
        if proof.get(key) is not True:
            raise ValueError('Actual full model/data proof required: ' + key)
    if proof['test_used_in_validation'] or proof['cuda_context_initialized'] or proof['author_equivalence_certified']:
        raise ValueError('Reviewed CPU replay boundary differs')
    queue = read(ROOT / replay_config['queue'])
    job = next(j for j in queue['jobs'] if j['id'] == replay_config['job_id'])
    result = read(ROOT / job['output_directory'] / 'result.json')
    original = [r for r in result['artifacts'] if 'targets' in r]
    replayed = proof['full_batch_receipts']
    if len(original) != 14 or len(replayed) != 14:
        raise ValueError('All fourteen original and replayed batches required')
    batches = []
    for index, (old, new) in enumerate(zip(original, replayed)):
        if new['batch'] != index or new['original_sha256'] != old['sha256']:
            raise ValueError('Original batch ordering/identity changed')
        if sha(old['path']) != old['sha256'] or sha(new['path']) != new['sha256']:
            raise ValueError('Actual original or replayed batch bytes changed')
        with np.load(old['path'], allow_pickle=False) as a, np.load(new['path'], allow_pickle=False) as b:
            row = verify_batch(dict(b), dict(a), 128 if index < 13 else 83, old['targets'],
                               replay_config['prediction_tolerance'])
        batches.append(dict(batch=index, original_path=old['path'], original_sha256=old['sha256'],
                            replay_path=new['path'], replay_sha256=new['sha256'], **row))
    if sum(r['targets'] for r in batches) != 268613 or sum(r['windows'] for r in batches) != 1747:
        raise ValueError('Every native target and window required')
    values = {k: float(np.mean([r['independently_recomputed_metrics'][k] for r in batches]))
              for k in ('native_mean_batch_mae', 'native_mean_batch_mse', 'native_mean_batch_mape_percent')}
    values['sqrt_native_mean_batch_mse'] = math.sqrt(values['native_mean_batch_mse'])
    for key, value in values.items():
        for reference in [proof['replayed_metrics'][key], proof['original_metrics'][key], result['metrics'][key]]:
            if not math.isclose(value, reference, **replay_config['metric_tolerance']):
                raise ValueError('Independent full native metric differs: ' + key)
    original_binding = ROOT / job['output_directory'] / 'split_mask_graph_scaler.npz'
    replay_binding = Path(replay_config['artifact_root']) / 'split_mask_graph_scaler.npz'
    with np.load(original_binding, allow_pickle=False) as a, np.load(replay_binding, allow_pickle=False) as b:
        if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError('Full original data masks graph scaler or splits changed')
        binding_array_count = len(a.files)
    return dict(job_id=job['id'], full_model_replay_and_independent_saved_arrays_verified=True,
                coverage=required, full_batch_receipts=batches, independently_recomputed_metrics=values,
                original_metrics=result['metrics'], replayed_metrics=proof['replayed_metrics'],
                full_binding_array_count=binding_array_count,
                original_binding_sha256=sha(original_binding), replay_binding_sha256=sha(replay_binding),
                prediction_tolerance=replay_config['prediction_tolerance'], metric_tolerance=replay_config['metric_tolerance'],
                additional_training_seed_slots=0, training_seeds_completed=1, training_seeds_required=5,
                author_equivalence_certified=False, all_original_experiments_complete=False,
                boundary='Complete selected native checkpoint replay and independent all-array verification. '
                         'All 1747 overlapping windows/268613 masked targets/14 batches/20 Euler steps; '
                         'original unweighted batch means and MAPE cutoff retained. Frozen CPU/GPU tolerances, '
                         'no bitwise prediction claim, no new training seed, no original-paper or TAB certification.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Preserve immutable verification attempts')
    proof = verify(config)
    target.mkdir(parents=True)
    write(target / 'independent_full_replay_verification.json', proof)
    write(target / 'validation.json', dict(passed=True, source_receipts=config['source_receipts'] +
          [dict(path=args.config, sha256=sha(ROOT / args.config))],
          outputs={'independent_full_replay_verification.json': sha(target / 'independent_full_replay_verification.json')}))
    print(dict(passed=True, windows=1747, targets=268613, batches=14, new_training_seeds=0))


if __name__ == '__main__':
    main()
