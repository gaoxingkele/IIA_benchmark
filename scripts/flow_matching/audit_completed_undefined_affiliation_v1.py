"""Audit full saved-score TAB undefined metrics without inventing measurements."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.flow_matching import publish_full_range_increment_v25 as full
from scripts.flow_matching import publish_result_listing_audited_v9 as previous
from scripts.flow_matching.tab_range_metrics import load_reference

ROOT = full.ROOT
METADATA_FIELDS = {'comparison', 'uncertainty_scope', 'source', 'captured_utc'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def review_empty_prediction(labels, scores, threshold, stored, reference):
    """Execute untouched TAB affiliation functions on every test timestamp."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if (labels.ndim != 1 or labels.shape != scores.shape or not labels.size
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all()
            or not math.isfinite(threshold)):
        raise ValueError('Require full aligned binary labels and finite scores/threshold')
    prediction = (scores > threshold).astype(int)
    label_events = reference['events'](labels)
    prediction_events = reference['events'](prediction)
    if prediction.any() or prediction_events or not label_events:
        raise ValueError('This classification requires no predictions and actual reference events')
    output = {}
    for field in ('affiliation_f', 'affiliation_precision', 'affiliation_recall'):
        value = float(reference[field](labels, prediction))
        converted = value if math.isfinite(value) else None
        if converted != stored[field]:
            raise ValueError('Untouched full TAB reference differs from saved affiliation output')
        output[field] = converted
    if (stored['status'] != 'undefined_reference_nan' or output['affiliation_f'] is not None
            or output['affiliation_precision'] is not None or output['affiliation_recall'] != 0.0):
        raise ValueError('Preserve undefined precision/F and defined zero recall')
    return dict(full_test_points=int(labels.size), actual_anomaly_points=int(labels.sum()),
                actual_reference_events=len(label_events), predicted_positive_points=0,
                predicted_events=0, score_maximum=float(scores.max()), frozen_threshold=float(threshold),
                strict_comparison='score > frozen_validation_threshold',
                original_TAB_reference_recomputed=output,
                classification='completed_evaluation_undefined_reference_no_predicted_events')


def review_validation(values, calibration):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError('Complete finite validation scores required')
    digest = hashlib.sha256(values.astype('<f8').tobytes()).hexdigest()
    threshold = float(np.percentile(values, 99, method='linear'))
    if (calibration['calibration_split'] != 'validation' or calibration['false_alarm_percent'] != 1
            or calibration['rule'] != 'validation_score_percentile_linear_strict_greater'
            or calibration['calibration_points'] != values.size
            or calibration['calibration_sha256'] != digest or calibration['threshold'] != threshold):
        raise ValueError('Frozen full validation99percentile identity differs')
    return dict(full_validation_points=int(values.size), validation_sha256=digest,
                exact_validation_threshold_recomputed=threshold, test_labels_used_for_threshold=False)


def annotate_complete_undefined(rows, records, source, captured):
    """Only an entirely completed, entirely undefined original seed cohort qualifies."""
    groups = {}
    for record in records:
        identity = (record['algorithm'], record['dataset'])
        seed = record['seed']
        group = groups.setdefault(identity, {})
        if seed in group:
            raise ValueError('Duplicate original training seed')
        if record['classification'] != 'completed_evaluation_undefined_reference_no_predicted_events':
            raise ValueError('Actual full reference review required')
        group[seed] = record
    changed = []
    for position, row in enumerate(rows):
        if (row['task'], row['protocol'], row['metric']) != ('anomaly_detection', full.PROTOCOL, 'affiliation_f'):
            continue
        group = groups.get((row['algorithm'], row['dataset']), {})
        if len(group) != int(row['required_repeats']):
            continue
        if set(group) != set(range(1103, 1108)):
            raise ValueError('Exact original five seed identities required')
        if int(row['n']) != 0 or any(row[field] not in ('', None) for field in ('mean', 'std', 'se', 'ci95_low', 'ci95_high')):
            raise ValueError('Entirely undefined cohort must have zero defined metrics and no statistics')
        row.update(comparison='five_of_five_evaluations_completed_affiliation_F_undefined_no_predicted_events',
                   uncertainty_scope='All5originalseeds1103..1107executed;0definedaffiliation_Fvalues. Mean/STD/SE/CI remain undefined; no predicted events at frozen validation-only1percent threshold. This is not an unrun cohort.',
                   source=source, captured_utc=captured)
        changed.append(position)
    return changed


def verify_metadata_delta(before, after, positions):
    if len(before) != len(after) or len(set(positions)) != len(positions):
        raise ValueError('Canonical row scope changed')
    actual = []
    for index, (old, new) in enumerate(zip(before, after)):
        delta = {key for key in set(old) | set(new) if old.get(key) != new.get(key)}
        if delta:
            if index not in positions or not delta.issubset(METADATA_FIELDS):
                raise ValueError('Numeric values, repeats, identities or unrelated metadata changed')
            actual.append(index)
    if actual != positions:
        raise ValueError('Exact reviewed metadata positions required')


def review_current_numeric_coverage(rows, evaluations):
    """Sparse queue captures need not contain historical superseded outputs."""
    groups = defaultdict(list)
    for e in evaluations:
        for metric in full.METRICS:
            macro = e['VUS_entity_macro'] if metric != 'affiliation_f' else e['strict_affiliation_entity_macro']['1']
            value = macro[metric]['mean_over_defined_entities']
            if value is not None:
                if not math.isfinite(value):
                    raise ValueError('Defined native metrics must be finite')
                groups[(Path(e['model_config']).stem, e['dataset'].upper(), metric)].append(value)
    partial, matching = [], 0
    for identity, values in groups.items():
        matches = [r for r in rows if (r['task'], r['protocol']) == ('anomaly_detection', full.PROTOCOL)
                   and (r['algorithm'], r['dataset'], r['metric']) == identity]
        if len(matches) != 1:
            raise ValueError('Unique canonical numeric slot required')
        row = matches[0]
        if len(values) > int(row['n']):
            raise ValueError('New actual numeric evaluations require separate scientific publication')
        if len(values) < int(row['n']):
            partial.append(dict(algorithm=identity[0], dataset=identity[1], metric=identity[2],
                                current_sparse_artifact_values=len(values), published_defined_values=int(row['n']),
                                original_published_source=row['source'],
                                current_sparse_outputs_do_not_replace_historical_audited_outputs=True))
            continue
        stats = full.independent_stats(values)
        for field in ('mean', 'std', 'se', 'ci95_low', 'ci95_high'):
            old, new = row[field], stats[field]
            if old in ('', None) and new is None:
                continue
            if old in ('', None) or new is None or not math.isclose(float(old), new, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Same-count current range numeric statistics differ')
        matching += 1
    return dict(current_numeric_group_metrics=len(groups),
                complete_same_count_aggregates_matching_published_statistics=matching,
                historical_aggregates_with_partial_current_sparse_coverage=partial,
                new_numeric_aggregates=0)


def execute(config_path):
    config = read(ROOT / config_path)
    if config['original_scope'] != full.SCOPE:
        raise ValueError('Original full research scope required')
    for source in config['source_receipts']:
        if full.stream_sha(ROOT / source['path']) != source['sha256']:
            raise ValueError('Frozen diagnostic/publication source changed')
    previous.verified_capture(ROOT / config['base_publication'], 'publication_validation.json')
    base = ROOT / config['base_publication'] / 'all_algorithm_dataset_metrics.csv'
    with base.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        headers, before = list(reader.fieldnames), list(reader)
    rows = [dict(row) for row in before]
    evaluations, selected, identities = [], [], set()
    for entry in config['completed_evaluations']:
        path = ROOT / entry['path']
        if full.stream_sha(path) != entry['sha256'] or path.with_name('evaluation.sha256').read_text(encoding='ascii').strip() != entry['sha256']:
            raise ValueError('Frozen completed range output changed')
        e = read(path)
        identity = (Path(e['model_config']).stem, e['dataset'].upper(), e['seed'])
        if e['status'] != 'completed_range_evaluation' or identity in identities:
            raise ValueError('Unique completed original model evaluation required')
        identities.add(identity)
        evaluations.append(e)
        if e['strict_affiliation_entity_macro']['1']['affiliation_f']['mean_over_defined_entities'] is None:
            selected.append((entry, e))
    if len(evaluations) != config['expected_completed_evaluations'] or len(selected) != config['expected_undefined_evaluations']:
        raise ValueError('Complete frozen scan and every undefined evaluation required')
    reviewed, hash_cache = [], {}
    for entry, e in selected:
        result = read(ROOT / e['model_result_path'])
        science_path = entry['scientific_configuration']
        science = read(ROOT / science_path)
        candidates = [job for queue_path in science['queue_paths']
                      for job in read(ROOT / queue_path)['jobs'] if job['id'] == e['id']]
        if len(candidates) != 1 or candidates[0] != result['job']:
            raise ValueError('Exact originally registered full training job required')
        run = dict(source_execution_id=e['id'], seed=e['seed'], algorithm_config=Path(e['model_config']).stem,
                   dataset=e['dataset'].upper(), result_sha256=e['model_result_sha256'],
                   scores_sha256=result['scores_sha256'], checkpoint_sha256=result['checkpoint_sha256'],
                   threshold=result['metrics']['strict']['1']['calibration']['threshold'])
        audited, proof = full.audit_evaluation(dict(evaluation_path=entry['path'], evaluation_sha256=entry['sha256']),
                                             run, science_path, candidates[0], hash_cache)
        dataset = next(d for d in read(ROOT / result['job']['dataset_manifest_path'])['datasets'] if d['id'] == e['dataset'])
        reference = load_reference(ROOT / science['reference_metric_root'])
        with np.load((ROOT / e['model_result_path']).parent / 'scores.npz', allow_pickle=False) as scores:
            validation_keys = [key for key in scores.files if key.endswith('_validation')]
            validation = review_validation(np.concatenate([scores[key] for key in validation_keys]),
                                           result['metrics']['strict']['1']['calibration'])
            entities = []
            for entity in dataset['entities']:
                prefix = entity['array_prefix']
                entities.append(dict(entity_id=entity['entity_id'], **review_empty_prediction(
                    scores[prefix+'_labels'], scores[prefix+'_test'], run['threshold'],
                    audited['strict_affiliation_per_entity'][entity['entity_id']]['1'], reference)))
        reviewed.append(dict(algorithm=run['algorithm_config'], dataset=run['dataset'], seed=run['seed'],
                             classification='completed_evaluation_undefined_reference_no_predicted_events',
                             full_original_artifact_audit=proof, validation=validation, full_entities=entities))
    captured = datetime.now(timezone.utc).isoformat()
    coverage = review_current_numeric_coverage(before, evaluations)
    positions = annotate_complete_undefined(rows, reviewed, config['diagnostic_output']+'/full_reference_review.json', captured)
    if len(positions) != config['expected_metadata_only_rows']:
        raise ValueError('Exact independently reviewed entirely undefined cohorts required')
    verify_metadata_delta(before, rows, positions)
    output = ROOT / config['output_directory']
    output.mkdir(parents=True, exist_ok=False)
    report = dict(captured_utc=captured, configuration=config_path, configuration_sha256=full.stream_sha(ROOT/config_path),
                  original_scope=config['original_scope'], completed_range_artifacts_scanned=len(evaluations),
                  current_sparse_numeric_coverage_review=coverage,
                  newly_available_numeric_aggregates=0, full_original_saved_score_reference_reviews=reviewed,
                  full_reviewed_original_models=len(reviewed), metadata_only_changed_positions=positions,
                  undefined_values_promoted_to_zero=False, checkpoint_reinference_performed=False,
                  new_benchmark_results=False, original_jobs_interrupted_or_restarted=False,
                  all_original_experiments_complete=False, author_equivalence_certified=False, strict_TAB_result=False,
                  boundary='Full original saved score reference replay for every frozen undefined affiliation-F case. '
                  'Training/checkpoint/data/science identities verified; no new model inference or VUS numerical replay. '
                  'No evidence that the score-distribution discrepancy is caused by model architecture or a code bug.')
    json_write(ROOT/config['diagnostic_output']/'full_reference_review.json', report)
    with (output/'all_algorithm_dataset_metrics.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    measured = sum(bool(row['mean']) for row in rows)
    boundary = ('Only two existing USAD/PRONTO_DAY4 affiliation-F rows receive completed-but-undefined metadata. '
                'All15undefined original evaluations were replayed on full saved test scores with untouched TAB '
                'affiliation code; all original five validation99percentile thresholds and score/data/checkpoint '
                'identities were verified for each applicable cohort. Every numerical field, repeat budget and '
                'other row/capture time remains unchanged. No predicted events means TAB precision/F are undefined, '
                'while recall is0. No value is replaced by0, no finite-value uncertainty is invented, and no experiment '
                'is rerun merely to obtain a defined metric. Original full research scope remains incomplete.')
    (output/'README.rst').write_text('Algorithm-dataset metrics v54\n============================\n\n'+boundary+
        f'\n\n{len(rows)}metric rows;{measured}numeric;{len(rows)-measured}without numeric means. '
        'Blank numeric rows include completed-but-undefined metrics; they are not a count of unrun experiments.\n', encoding='utf-8')
    json_write(output/'publication_validation.json', dict(published_utc=captured, base_publication=config['base_publication'],
        aggregate_metric_rows=len(rows), fields=len(headers), measured_metric_rows=measured,
        missing_metric_rows=len(rows)-measured, changed_metric_rows=len(positions),
        changed_positions=positions, numeric_fields_and_repeat_counts_unchanged=True,
        other_rows_and_capture_times_unchanged=True, new_numeric_results=0,
        original_scope=config['original_scope'], all_original_experiments_complete=False,
        author_equivalence_certified=False, strict_TAB_result=False, source_receipts=config['source_receipts'],
        outputs={path.name:full.stream_sha(path) for path in output.iterdir() if path.is_file()}))
    print(json.dumps(dict(full_original_reference_reviews=len(reviewed), metadata_only_rows=len(positions),
                         new_numeric_results=0, measured_rows=measured, all_original_experiments_complete=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    execute(parser.parse_args().config)
