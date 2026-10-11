"""Independently review the complete four-seed native BRITS publication."""
import argparse
import math
from datetime import datetime, timezone

from scripts.flow_matching.review_brits_two_seed_publication_v27 import (
    ROOT, digest, read, check_arrays, csv_rows, check_statistics)
from scripts.flow_matching.publish_brits_four_seed_listing_v39 import complete_delta, SEEDS, METRICS


def review(config_path):
    import h5py
    import numpy as np
    config = read(ROOT / config_path)
    for source in config['source_receipts']:
        if digest(ROOT / source['path']) != source['sha256']:
            raise ValueError('Frozen independent review source changed')
    base, target = (ROOT / config[k] for k in ('base_publication', 'output_directory'))
    before, after = (csv_rows(p / 'all_algorithm_dataset_metrics.csv') for p in (base, target))
    if len(before) != 8098 or len(after) != 8098 or any(len(row) != 31 for row in after):
        raise ValueError('Every original metric slot and31CSVfields required')
    positions = complete_delta(before, after)
    receipt = read(target / 'publication_validation.json')
    for filename, expected in receipt['outputs'].items():
        if digest(target / filename) != expected:
            raise ValueError('Actual published output hash differs')
    unchanged_exports = []
    for path in base.iterdir():
        if path.is_file() and path.name not in ('all_algorithm_dataset_metrics.csv', 'all_metrics.md',
                'results.html', 'README.rst', 'publication_validation.json'):
            if path.read_bytes() != (target / path.name).read_bytes():
                raise ValueError('Every other original track export must remain byte-identical')
            unchanged_exports.append(path.name)
    records = [r for p in config['full_model_proofs'] for r in read(ROOT / p)['records']
        if r['id'].startswith('saits_paper_brits_')]
    if sorted(r['seed'] for r in records) != SEEDS or len(records) != 4:
        raise ValueError('Exactly the four unique completed original training seeds required')
    checked = []
    for record in records:
        native = read(ROOT / record['config'])
        result = read(ROOT / native['output_root'] / 'result.json')
        completion = record['full_training_completion']
        if (result['status'] != 'completed' or result['parameters'] != 6993328
                or result['config'] != native or native['seed'] != record['seed']
                or not completion['full_training_finished'] or completion['benchmark_smoke']
                or completion['original_maximum_epochs'] != 10000
                or (completion['next_epoch'] < 10000 and completion['author_patience_remaining'] != 0)):
            raise ValueError('Complete original training and full architecture required')
        for field in ('config', 'result', 'checkpoint', 'replay_arrays'):
            path = record[field] if field in ('config', 'replay_arrays') else record[field + '_path']
            if digest(ROOT / path) != record[field + '_sha256']:
                raise ValueError('Original full model/data/result/replay binding changed')
        data = ROOT / native['dataset_root']
        if digest(data / 'datasets.h5') != record['original_full_dataset_sha256']:
            raise ValueError('Full original HDF5 data changed')
        with h5py.File(data / 'datasets.h5', 'r') as hf:
            truth, mask = hf['test']['X'][:], hf['test']['indicating_mask'][:]
        with np.load(data / 'audit_arrays.npz', allow_pickle=False) as identity:
            ids, groups = identity['test_ids'], identity['test_groups']
        with np.load(record['replay_arrays'], allow_pickle=False) as arrays:
            metrics, grouped = check_arrays(arrays, truth, mask, ids, groups, 1020954,
                dict(rtol=2e-6, atol=2e-6))
        for metric, value in metrics.items():
            if not math.isclose(value, record['original_metrics'][metric], rel_tol=2e-6, abs_tol=2e-6):
                raise ValueError('Independent complete native array metric differs')
        checked.append(dict(id=record['id'], seed=record['seed'], full_test_shape=list(truth.shape),
            held_out_coordinates=1020954, original_maximum_epochs=10000,
            actual_stopping_epoch=completion['next_epoch'], months=len(grouped),
            independent_metrics=metrics, original_metrics=record['original_metrics'],
            full_truth_mask_unit_month_identity_checked=True))
    statistics = {}
    for index in positions:
        row = after[index]
        statistics[row['metric']] = check_statistics(row, [r['original_metrics'][row['metric']] for r in records])
        if (row['paper_mean'] != before[index]['paper_mean']
                or row['author_equivalence_certified'].lower() != 'false'
                or row['strict_TAB_result'].lower() != 'false'):
            raise ValueError('Preserve paper values and uncertified evaluation boundaries')
    if (receipt['changed_numeric_metric_rows'] != 3 or receipt['newly_measured_metric_rows'] != 0
            or receipt['brits_electricity_completed_training_seeds'] != 4
            or receipt['all_original_experiments_complete']
            or receipt['all_numeric_tables_byte_identical_to_base']
            or not receipt['performance_rows_changed']):
        raise ValueError('Completion, seed and numerical-change declarations must be correct')
    lines = (target / 'all_metrics.md').read_text(encoding='utf-8').splitlines()
    header = next(line for line in lines if line.startswith('| '))
    if len(header[2:-2].split(' | ')) != 31:
        raise ValueError('Display every original CSV field')
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        publication=config['output_directory'], original_independent_training_seeds=SEEDS,
        required_training_seeds=5, full_native_original_models_independently_checked=4,
        complete_original_HDF5_replay_checks=checked, independent_training_seed_statistics=statistics,
        all8098_original_metric_slots_and31fields_retained=True, changed_metric_rows=3,
        unchanged_metric_rows=8095, unchanged_track_exports=unchanged_exports,
        publication_receipt_sha256=digest(target / 'publication_validation.json'),
        measured_metric_rows=sum(bool(row['mean']) for row in after),
        additional_training_seed_slots=0, author_equivalence_certified=False,
        strict_TAB_result=False, all_original_experiments_complete=False)
    import json
    with (ROOT / config['independent_review_path']).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(full_native_models_checked=4, unchanged_rows=8095, changed_rows=3, passed=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    review(parser.parse_args().config)
