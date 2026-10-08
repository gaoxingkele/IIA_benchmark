"""Add pinned TAB VUS/Affiliation to full saved scores, without retraining."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.run_tsad_queue import complete_result
from scripts.flow_matching.tab_range_metrics import accelerated_vus, load_reference


def finite(value):
    return float(value) if np.isfinite(value) else None


def affiliation_record(reference, labels, prediction):
    if not labels.any():
        return {'status': 'undefined_no_ground_truth_event', 'affiliation_f': None, 'affiliation_precision': None, 'affiliation_recall': None}
    # Same original underlying calculation as all three pinned wrapper bodies;
    # compute once rather than triplicating expensive event integrals.
    values = reference['pr_from_events'](reference['events'](prediction), reference['events'](labels), (0, len(labels)))
    precision, recall = values['precision'], values['recall']
    f_score = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
    result = {'affiliation_f': finite(f_score), 'affiliation_precision': finite(precision), 'affiliation_recall': finite(recall)}
    return {'status': 'defined' if all(v is not None for v in result.values()) else 'undefined_reference_nan', **result}


def macro(records, keys):
    result = {}
    for key in keys:
        values = [r[key] for r in records.values() if r.get(key) is not None]
        result[key] = {'defined_entities': len(values), 'undefined_entities': len(records) - len(values),
                       'mean_over_defined_entities': float(np.mean(values)) if values else None}
    return result


def evaluate(root, result_path, config, reference):
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if not complete_result(root, result['job']):
        raise ValueError('Full model result is not complete')
    result_sha = sha(result_path)
    settings_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    identity = hashlib.sha256((result_sha + settings_sha).encode()).hexdigest()
    output = root / config['output_root'] / 'jobs' / result['id']
    final_path = output / 'evaluation.json'
    if final_path.exists():
        previous = json.loads(final_path.read_text(encoding='utf-8'))
        if previous['evaluation_identity'] != identity:
            raise ValueError('Saved TAB evaluation belongs to different inputs or rules')
        receipt = output / 'evaluation.sha256'
        if not receipt.exists() or receipt.read_text(encoding='ascii').strip() != sha(final_path):
            raise ValueError('Saved TAB evaluation content changed or has no checksum receipt')
        return previous
    manifest = json.loads((root / result['job']['dataset_manifest_path']).read_text(encoding='utf-8'))
    dataset = next(d for d in manifest['datasets'] if d['id'] == result['dataset'])
    with np.load(result_path.parent / 'scores.npz', allow_pickle=False) as source:
        data = {name: source[name].copy() for name in source.files}
    vus_records, strict, ratio_records = {}, {}, {}
    started = time.monotonic()
    for entity in dataset['entities']:
        prefix, entity_id = entity['array_prefix'], entity['entity_id']
        labels, scores = data[prefix + '_labels'], data[prefix + '_test']
        if len(labels) != entity['test_points'] or labels.shape != scores.shape or not np.isfinite(scores).all():
            raise ValueError('TAB input lacks complete original entity timestamps')
        if np.unique(labels).size == 2:
            max_buffer = 2 * int(np.median(reference['get_list_anomaly'](labels)))
            values = accelerated_vus(labels, scores, max_buffer)
            vus_records[entity_id] = {'status': 'defined', **values}
        else:
            vus_records[entity_id] = {'status': 'undefined_single_class_reference', 'VUS_ROC': None, 'VUS_PR': None, 'max_buffer': None}
        strict[entity_id] = {}
        for ratio, record in result['metrics']['strict'].items():
            threshold = record['calibration']['threshold']
            strict[entity_id][ratio] = affiliation_record(reference, labels, scores > threshold)
        ratio_records[entity_id] = []
        for row in result['metrics']['tab_core_diagnostic']['ratio_grid']:
            ratio_records[entity_id].append({'ratio_percent': row['ratio_percent'], 'threshold': row['threshold'],
                                            **affiliation_record(reference, labels, scores > row['threshold'])})
        print(json.dumps({'id': result['id'], 'entity': entity_id, 'stage': 'TAB_ranges', 'max_buffer': vus_records[entity_id]['max_buffer']}), flush=True)
    affiliation_keys = ['affiliation_f', 'affiliation_precision', 'affiliation_recall']
    strict_macro = {ratio: macro({entity: values[ratio] for entity, values in strict.items()}, affiliation_keys)
                    for ratio in result['metrics']['strict']}
    tab_macro = []
    for index, row in enumerate(result['metrics']['tab_core_diagnostic']['ratio_grid']):
        tab_macro.append({'ratio_percent': row['ratio_percent'], 'threshold': row['threshold'],
                          'metrics': macro({entity: values[index] for entity, values in ratio_records.items()}, affiliation_keys)})
    record = {'status': 'completed_range_evaluation', 'id': result['id'], 'dataset': result['dataset'], 'model_config': result['model_config'], 'seed': result['seed'],
              'evaluation_identity': identity, 'model_result_path': result_path.relative_to(root).as_posix(),
              'model_result_sha256': result_sha, 'scores_sha256': result['scores_sha256'],
              'evaluation_config_sha256': settings_sha, 'source_receipts': config['source_receipts'],
              'tab_commit': config['tab_commit'], 'evaluation_seconds': time.monotonic() - started,
              'VUS_per_entity': vus_records, 'VUS_entity_macro': macro(vus_records, ['VUS_ROC', 'VUS_PR']),
              'strict_affiliation_per_entity': strict, 'strict_affiliation_entity_macro': strict_macro,
              'tab_ratio_affiliation_per_entity': ratio_records, 'tab_ratio_affiliation_entity_macro': tab_macro,
              'VUS_buffer_rule': 'Every integer from zero to twice the truncated median TEST anomaly event length, matching pinned TAB wrapper. Label-dependent evaluation, not model calibration.',
              'aggregation': 'Entity macro over defined metrics; counts of undefined/no-event/no-alarm cases reported, never replaced by zero. No cross-entity event merging.',
              'boundary': 'Pinned TAB metric semantics on local saved scores; global local-model fitting, split, feature and score layout are independent. This is not full TAB training-harness or author-paper equivalence. Model artifacts and primary validation thresholds unchanged.'}
    write_json(final_path, record)
    (output / 'evaluation.sha256').write_text(sha(final_path) + '\n', encoding='ascii')
    return record
