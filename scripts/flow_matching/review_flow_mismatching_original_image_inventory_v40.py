"""Independently check every appended original target and untouched scientific row."""
import argparse
import csv
from datetime import datetime, timezone
import json

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha, verify_receipts


def load(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); return reader.fieldnames, list(reader)


def review(config_path):
    config = read(ROOT / config_path); verify_receipts(config['source_receipts'])
    base, target = (ROOT / config[k] for k in ('base_publication', 'output_directory'))
    old_header, before = load(base / 'all_algorithm_dataset_metrics.csv')
    header, after = load(target / 'all_algorithm_dataset_metrics.csv')
    if header != old_header or len(header) != 31 or len(before) != 8098 or len(after) != 8618:
        raise ValueError('All original/new metric slots and31fields required')
    if before != after[:8098] or not (target / 'all_algorithm_dataset_metrics.csv').read_bytes().startswith(
            (base / 'all_algorithm_dataset_metrics.csv').read_bytes()):
        raise ValueError('Every existing scientific row and exact byte prefix must be unchanged')
    queue = read(ROOT / config['image_queue']); protocol = read(ROOT / queue['original_protocol'])
    original = []
    for key in ('Table6_compute', 'Table7_path_aggregation', 'Table8_conditional_mean', 'Table9_epoch_checkpoints', 'Table2_same_backbone_FM_comparison'):
        for index, case in enumerate(protocol[key]):
            name = {'MVTec-AD': 'mvtec', 'VisA': 'visa'}[case['dataset']]
            identity = 'flow_mismatching_original_image/' + key.lower() + '__row%02d__' % index + name
            for metric, printed in case['paper_reported'].items():
                original.append((identity, case['dataset'], metric, printed,
                    50 if case.get('method') == 'TCCM' else case.get('checkpoint_epoch', 2000)))
    added = after[8098:]
    if len(original) != 520:
        raise ValueError('Every original520 transcribed paper metric target required')
    for row, (identity, dataset, metric, printed, epochs) in zip(added, original):
        if ((row['algorithm'], row['dataset'], row['metric']) != (identity, dataset, metric)
                or float(row['paper_mean']) != printed or int(row['epochs']) != epochs
                or row['n'] != '0' or row['required_repeats'] != '5'
                or any(row[key] for key in ('mean', 'std', 'se', 'ci95_low', 'ci95_high'))
                or row['author_equivalence_certified'].lower() != 'false' or row['strict_TAB_result'].lower() != 'false'
                or row['task'] != 'image_anomaly_detection'):
            raise ValueError('Original target identity/value/budget or unmeasured status differs')
    _, native_only = load(target / 'flow_mismatching_original_image_metric_inventory.csv')
    if native_only != added:
        raise ValueError('Native-only and complete scientific inventories must agree')
    receipt = read(target / 'publication_validation.json')
    for name, digest in receipt['outputs'].items():
        if sha(target / name) != digest:
            raise ValueError('Actual publication output differs')
    unchanged_exports = []
    for file in base.iterdir():
        if file.is_file() and file.name not in {'all_algorithm_dataset_metrics.csv', 'all_metrics.md', 'results.html', 'README.rst', 'publication_validation.json'}:
            if file.read_bytes() != (target / file.name).read_bytes():
                raise ValueError('Every unrelated native/strict track export must remain byte-identical')
            unchanged_exports.append(file.name)
    if (sum(bool(row['mean']) for row in after) != 2120 or receipt['measured_metric_rows'] != 2120
            or receipt['missing_metric_rows'] != 6498 or receipt['changed_numeric_metric_rows'] != 0
            or receipt['newly_measured_metric_rows'] != 0 or receipt['all_original_experiments_complete']
            or receipt['original_image_performance_results'] != 0
            or not receipt['base_numeric_csv_prefix_byte_identical']):
        raise ValueError('No registered or data-readiness row may become invented performance')
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        original8098_rows_and_capture_times_byte_identical=True, retained_csv_fields=31,
        independently_checked_original_paper_target_rows=520, image_pixel_score_targets=455,
        author_A100_hardware_dependent_FPS_targets=30, other_original_comparator_targets=35,
        total_metric_slots=8618, measured_metric_slots=2120, missing_metric_slots=6498,
        newly_measured_metric_slots=0, other_track_exports_unchanged=unchanged_exports,
        all_original_experiments_complete=False, strict_TAB_result=False,
        publication_receipt_sha256=sha(target / 'publication_validation.json'))
    output = ROOT / config['independent_review_path']; output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(dict(original_rows_unchanged=8098, original_image_targets_checked=520, measured_slots=2120, passed=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    review(parser.parse_args().config)
