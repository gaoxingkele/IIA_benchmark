"""Independently check every native BRITS replay coordinate against frozen HDF5."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_arrays(arrays, native_truth, native_mask, unit_ids, groups, target_count,
                 tolerance=None):
    import numpy as np
    tolerance = tolerance or dict(rtol=2e-6, atol=2e-6)
    prediction, truth, mask = (arrays[k] for k in ('predictions', 'truth', 'mask'))
    indices = arrays['test_indices']
    shape = native_truth.shape
    if (prediction.shape != shape or truth.shape != shape or mask.shape != shape
            or indices.shape != (shape[0],) or indices.dtype.kind not in 'iu'
            or not np.array_equal(np.sort(indices), np.arange(shape[0]))):
        raise ValueError('Every original test window and coordinate must occur exactly once')
    if not all(np.isfinite(a).all() for a in (prediction, truth, mask)):
        raise ValueError('All replay coordinates must be finite')
    expected_truth = np.nan_to_num(native_truth).astype(np.float32)[indices]
    if (not np.array_equal(truth, expected_truth)
            or not np.array_equal(mask, native_mask.astype(np.float32)[indices])
            or not np.isin(mask, [0, 1]).all()
            or int(mask.sum(dtype=np.float64)) != target_count):
        raise ValueError('Original full truth, mask and held-out identities must agree')
    if not np.array_equal(arrays['unit_ids'], unit_ids[indices]):
        raise ValueError('Original full test unit identities must agree')
    error = (prediction.astype(np.float64) - truth.astype(np.float64)) * mask
    statistics = np.column_stack((np.abs(error).sum(axis=(1, 2)),
        np.square(error).sum(axis=(1, 2)), mask.sum(axis=(1, 2), dtype=np.float64),
        np.abs(truth.astype(np.float64) * mask).sum(axis=(1, 2))))
    identifiers = np.unique(groups[indices])
    grouped = np.stack([statistics[groups[indices] == group].sum(axis=0) for group in identifiers])
    if (not np.array_equal(identifiers, arrays['group_ids'])
            or grouped.shape != arrays['group_statistics'].shape
            or not np.allclose(grouped, arrays['group_statistics'], **tolerance)):
        raise ValueError('Every original group statistic must match independent full arrays')
    total = grouped.sum(axis=0)
    return dict(mae=float(total[0] / total[2]), rmse=float(np.sqrt(total[1] / total[2])),
                mre=float(total[0] / total[3])), grouped


def verify(config_path):
    import h5py
    import numpy as np
    config = read(ROOT / config_path)
    for item in config['source_receipts']:
        if digest(ROOT / item['path']) != item['sha256']:
            raise ValueError('Registered independent replay verification source differs')
    replay_config = read(ROOT / config['replay_configuration'])
    for item in replay_config['source_receipts']:
        if digest(ROOT / item['path']) != item['sha256']:
            raise ValueError('Original frozen replay source or data differs')
    replay_path = ROOT / replay_config['artifact_directory'] / 'checkpoint_replay_audit.json'
    replay = read(replay_path)
    if (replay['configuration_sha256'] != digest(ROOT / config['replay_configuration'])
            or replay['full_models_replayed'] != 1 or len(replay['records']) != 1
            or not replay['full_checkpoint_contents_and_re_inference_passed']
            or replay['cuda_context_initialized'] or replay['additional_training_seed_slots'] != 0):
        raise ValueError('Exactly the completed native full CPU replay is required')
    record = replay['records'][0]
    native_path = ROOT / record['config']
    native = read(native_path)
    result_path = ROOT / native['output_root'] / 'result.json'
    result = read(result_path)
    arrays_path = Path(record['replay_arrays'])
    hashes = dict(config=digest(native_path), result=digest(result_path),
        checkpoint=digest(ROOT / native['output_root'] / 'best.pt'), arrays=digest(arrays_path),
        dataset=digest(ROOT / native['dataset_root'] / 'datasets.h5'))
    for field, actual in hashes.items():
        key = 'original_full_dataset_sha256' if field == 'dataset' else (
            'replay_arrays_sha256' if field == 'arrays' else field + '_sha256')
        if actual != record[key]:
            raise ValueError('Full native replay content hash differs: ' + field)
    if record['id'] != native['id'] or record['seed'] != native['seed'] or native['bootstrap_group'] != 'month':
        raise ValueError('Original registered seed and month grouping required')
    receipt = read(ROOT / replay_config['state_root'] / 'receipt.json')
    if receipt['exit_code'] != 0 or receipt['resource_guard_aborted'] or not receipt['full_native_model_replay_passed']:
        raise ValueError('Native replay process must finish successfully')
    with h5py.File(ROOT / native['dataset_root'] / 'datasets.h5', 'r') as hf:
        truth, mask = hf['test']['X'][:], hf['test']['indicating_mask'][:]
    with np.load(ROOT / native['dataset_root'] / 'audit_arrays.npz', allow_pickle=False) as identities:
        ids, groups = identities['test_ids'], identities['test_groups']
    tolerance = replay_config['metric_tolerance']
    with np.load(arrays_path, allow_pickle=False) as arrays:
        actual, grouped = check_arrays(arrays, truth, mask, ids, groups, result['target_count'], tolerance)
    for key, value in actual.items():
        if (not np.isclose(value, result['metrics'][key], **tolerance)
                or not np.isclose(value, record['replayed_metrics'][key], **tolerance)
                or not np.isclose(value, record['independently_recomputed_metrics'][key], **tolerance)):
            raise ValueError('Independently recomputed full replay metric differs: ' + key)
    with np.load(ROOT / native['output_root'] / 'group_statistics.npz', allow_pickle=False) as original:
        if not np.allclose(grouped, original['statistics'], **tolerance):
            raise ValueError('Full original native grouped errors differ')
    completion = record['full_training_completion']
    if (not completion['full_training_finished'] or completion['benchmark_smoke']
            or completion['full_test_units'] != truth.shape[0]
            or completion['held_out_coordinates'] != result['target_count']):
        raise ValueError('Full original training and held-out completion required')
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), id=native['id'],
        replay_configuration=config['replay_configuration'], replay_audit_sha256=digest(replay_path),
        hashes=hashes, full_test_shape=list(truth.shape), full_test_grid_cells=int(truth.size),
        held_out_coordinates=result['target_count'], full_original_truth_masks_units_groups_equal=True,
        full_finite_predictions=True, full_group_statistics_independently_recomputed=True,
        months=len(grouped), independently_recomputed_metrics=actual, original_metrics=result['metrics'],
        metric_tolerance=tolerance, full_training_completion=completion, native_replay_receipt=receipt,
        actual_full_checkpoint_reinference_completed=True, additional_training_seed_slots=0,
        author_equivalence_certified=False, all_original_experiments_complete=False)
    target = ROOT / config['output_directory']
    for name, value in [('independent_full_replay_verification.json', report),
                        ('result.json', result), ('checkpoint_replay_audit.json', replay),
                        ('replay_resource_receipt.json', receipt)]:
        with (target / name).open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(id=native['id'], full_test_windows=truth.shape[0],
        held_out_coordinates=result['target_count'], independent_metrics=actual)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    verify(args.config)


if __name__ == '__main__':
    main()
