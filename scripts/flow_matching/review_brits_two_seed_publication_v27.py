"""Independently review all table differences and both original full HDF5 replays."""
import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics

from scripts.flow_matching.verify_brits_electricity_full_replay_v3 import ROOT, digest, read, check_arrays


def check_statistics(row, values):
    from scipy.stats import t
    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    se = std / math.sqrt(n)
    margin = float(t.ppf(.975, n - 1)) * se
    expected = dict(mean=mean, std=std, se=se, ci95_low=mean-margin, ci95_high=mean+margin)
    if int(row['n']) != n or int(row['required_repeats']) != 5:
        raise ValueError('Original independent training repeats or full five-seed obligation differs')
    for key, value in expected.items():
        if not math.isclose(float(row[key]), value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Independent full original seed statistics differ: ' + key)
    return expected


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def review(config_path):
    import h5py
    import numpy as np
    config = read(ROOT / config_path)
    for source in config['source_receipts']:
        if digest(ROOT / source['path']) != source['sha256']:
            raise ValueError('Registered original publication input changed')
    base, target = (ROOT / config[key] for key in ('base_publication', 'output_directory'))
    original, published = (csv_rows(directory / 'all_algorithm_dataset_metrics.csv') for directory in (base, target))
    if len(original) != 7283 or len(published) != len(original):
        raise ValueError('Every registered method/dataset/metric obligation must remain')
    receipt = read(target / 'publication_validation.json')
    for name, expected in receipt['outputs'].items():
        if digest(target / name) != expected:
            raise ValueError('Published full output differs')
    differences = [i for i, (a, b) in enumerate(zip(original, published)) if a != b]
    wanted = ['mae', 'mre', 'rmse']
    if (len(differences) != 3 or sorted(published[i]['metric'] for i in differences) != wanted
            or any(published[i]['algorithm'] != 'brits' or published[i]['dataset'] != 'electricity'
                or published[i]['task'] != 'imputation' or published[i]['protocol'] != 'author'
                or published[i]['pattern'] != 'paper_mask' or published[i]['missing_ratio'] != '0.1'
                for i in differences)):
        raise ValueError('Only the three frozen original BRITS slots may differ')
    for name in ('overview.md', 'strict_algorithm_dataset.md', 'original_generation_jobs.csv', 'original_energy_jobs.csv'):
        if (base / name).read_bytes() != (target / name).read_bytes():
            raise ValueError('Unchanged track export differs')
    records = [record for path in config['full_model_proofs'] for record in read(ROOT / path)['records']
               if record['id'].startswith('saits_paper_brits_')]
    if sorted(record['seed'] for record in records) != [26, 27] or len(records) != 2:
        raise ValueError('Only two unique complete original training seeds required')
    independently_verified = []
    for record in records:
        native = read(ROOT / record['config'])
        dataset_root = ROOT / native['dataset_root']
        if digest(dataset_root / 'datasets.h5') != record['original_full_dataset_sha256']:
            raise ValueError('Full original frozen HDF5 differs')
        if digest(Path(record['replay_arrays'])) != record['replay_arrays_sha256']:
            raise ValueError('Full actual replay array differs')
        with h5py.File(dataset_root / 'datasets.h5', 'r') as hf:
            truth, mask = hf['test']['X'][:], hf['test']['indicating_mask'][:]
        with np.load(dataset_root / 'audit_arrays.npz', allow_pickle=False) as identities:
            ids, groups = identities['test_ids'], identities['test_groups']
        with np.load(record['replay_arrays'], allow_pickle=False) as arrays:
            actual, grouped = check_arrays(arrays, truth, mask, ids, groups, 1020954,
                dict(rtol=2e-6, atol=2e-6))
        for key, value in actual.items():
            if not math.isclose(value, record['original_metrics'][key], rel_tol=2e-6, abs_tol=2e-6):
                raise ValueError('Independently recomputed original full-array metric differs')
        independently_verified.append(dict(id=record['id'], seed=record['seed'],
            full_test_shape=list(truth.shape), full_truth_mask_window_month_identity_checked=True,
            held_out_coordinates=1020954, months=len(grouped), metrics=actual,
            replay_arrays_sha256=record['replay_arrays_sha256']))
    statistics_review = {}
    for index in differences:
        row = published[index]
        values = [record['original_metrics'][row['metric']] for record in records]
        statistics_review[row['metric']] = check_statistics(row, values)
        if (row['paper_mean'] != original[index]['paper_mean']
                or row['author_equivalence_certified'].lower() != 'false'
                or row['strict_TAB_result'].lower() != 'false'):
            raise ValueError('Paper comparison or unsupported equivalence promotion differs')
    if (receipt['performance_rows_changed'] is not True
            or receipt['all_numeric_tables_byte_identical_to_base'] is not False
            or receipt['base_numeric_csv_prefix_byte_identical'] is not False
            or receipt['changed_numeric_metric_rows'] != 3
            or receipt['newly_measured_metric_rows'] != 0
            or receipt['all_original_experiments_complete']):
        raise ValueError('Current publication numerical equality and completion declarations differ')
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), publication=config['output_directory'],
        publication_receipt_sha256=digest(target / 'publication_validation.json'),
        all_7283_registered_rows_retained=True, changed_rows=3,
        all_other_7280_metric_rows_and_capture_times_exact=True,
        four_other_track_exports_byte_identical=True, full_native_original_models_independently_checked=2,
        complete_original_HDF5_replay_checks=independently_verified,
        original_independent_training_seeds=[26, 27], required_original_training_seeds=5,
        mean_std_se_and_Student_t95_CI_independently_recomputed=statistics_review,
        current_numerical_equality_flags_verified=True, additional_training_seed_slots=0,
        author_equivalence_certified=False, all_original_experiments_complete=False)
    path = ROOT / config['independent_review_path']
    import json
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(dict(full_models_checked=2, unchanged_rows=7280, changed_rows=3,
        independent_two_seed_statistics_verified=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    review(args.config)


if __name__ == '__main__':
    main()
