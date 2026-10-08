"""Artifact/undefined-metric checks on fixtures; no benchmark performance claim."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.flow_matching.evaluate_tab_ranges import affiliation_record, evaluate
from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.tab_range_metrics import load_reference

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/runs/mtsad_protocol_audit/sources/tab/original/ts_benchmark/evaluation/metrics'


def test_affiliation_matches_original_partial_and_empty_prediction_semantics():
    reference = load_reference(SOURCE)
    labels = np.array([0, 1, 1, 0, 0, 1, 1, 1])
    for prediction in (np.array([1, 0, 1, 0, 1, 0, 0, 1]), labels, np.zeros(8)):
        actual = affiliation_record(reference, labels, prediction)
        for key in ('affiliation_f', 'affiliation_precision', 'affiliation_recall'):
            expected = reference[key](labels, prediction)
            if np.isfinite(expected):
                assert actual[key] == pytest.approx(expected)
            else:
                assert actual[key] is None
    empty_truth = affiliation_record(reference, np.zeros(8), labels)
    assert empty_truth['status'] == 'undefined_no_ground_truth_event'
    assert empty_truth['affiliation_f'] is None


def test_range_evaluation_preserves_inputs_thresholds_and_rejects_modified_cache(tmp_path):
    output = tmp_path / 'model_run'
    output.mkdir()
    labels = np.array([0, 1, 1, 0, 0, 1, 0, 0])
    np.savez_compressed(output / 'scores.npz', e000_labels=labels, e000_test=np.arange(8.), e000_train=np.arange(8.))
    (output / 'model.pt').write_bytes(b'test-only checkpoint hash fixture')
    manifest = {'datasets': [{'id': 'fixture', 'entities': [{'array_prefix': 'e000', 'entity_id': 'e', 'test_points': 8}]}]}
    write_json(tmp_path / 'manifest.json', manifest)
    job = {'output_directory': 'model_run', 'dataset_manifest_path': 'manifest.json'}
    identity = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    result = {'id': 'fixture', 'dataset': 'fixture', 'model_config': 'fixture_config', 'seed': 1, 'status': 'completed',
              'experiment_sha256': identity, 'job': job, 'scores_sha256': sha(output / 'scores.npz'), 'checkpoint_sha256': sha(output / 'model.pt'),
              'metrics': {'strict': {'1': {'calibration': {'threshold': 5.}}},
                          'tab_core_diagnostic': {'ratio_grid': [{'ratio_percent': 1., 'threshold': 4.}]}}}
    result_path = output / 'result.json'
    write_json(result_path, result)
    original_bytes = result_path.read_bytes()
    config = {'output_root': 'evaluations', 'source_receipts': [], 'tab_commit': 'fixture'}
    reference = load_reference(SOURCE)
    record = evaluate(tmp_path, result_path, config, reference)
    assert result_path.read_bytes() == original_bytes
    assert record['tab_ratio_affiliation_entity_macro'][0]['threshold'] == 4.
    assert record['VUS_per_entity']['e']['max_buffer'] == 2
    assert evaluate(tmp_path, result_path, config, reference) == record
    cached = tmp_path / 'evaluations/jobs/fixture/evaluation.json'
    damaged = json.loads(cached.read_text(encoding='utf-8'))
    damaged['VUS_per_entity']['e']['VUS_ROC'] = 100.
    write_json(cached, damaged)
    with pytest.raises(ValueError, match='content changed'):
        evaluate(tmp_path, result_path, config, reference)
