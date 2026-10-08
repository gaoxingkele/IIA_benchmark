import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.flow_matching.mtsbench_stat_protocol import ROOT, complete, environment_versions, fingerprint, run, sha, verify, write_json


def test_configuration_discloses_fit_and_oracle_roles():
    c = json.loads((ROOT / 'configs/experiments/fm_grasp_stat_registration.v1.json').read_text(encoding='utf-8'))
    assert c['protocol'] == 'native_transductive_test_feature_fit_oracle_f1'
    assert len(c['seeds']) == 10 and len(set(c['seeds'])) == 10
    assert [(d['id'], d['entities']) for d in c['datasets']] == [('swan', 1), ('smd', 18), ('smap', 51), ('cicids', 6)]
    assert c['algorithms'][0]['parameters']['slidingWindow'] == 1


def test_receipts_and_protocol_cannot_be_silently_changed(tmp_path):
    p = tmp_path / 'source.py'
    p.write_text('source')
    q = {'source_receipts': [{'path': 'source.py', 'sha256': sha(p)}], 'jobs': []}
    verify(q, tmp_path)
    p.write_text('changed')
    with pytest.raises(ValueError, match='Frozen source'):
        verify(q, tmp_path)
    q['source_receipts'] = []
    q['jobs'] = [{'id': 'x', 'protocol': 'strict_train_only'}]
    with pytest.raises(ValueError, match='disclose'):
        verify(q, tmp_path)


@pytest.mark.parametrize('field,value', [('diagnostic', True), ('strict_TAB_result', True), ('job_sha256', 'wrong')])
def test_diagnostic_or_wrong_binding_cannot_complete(tmp_path, field, value):
    j = {'id': 'x', 'output_directory': 'job'}
    r = {'status': 'completed', 'diagnostic': False, 'strict_TAB_result': False, 'job_sha256': fingerprint(j), 'artifacts': []}
    r[field] = value
    write_json(tmp_path / 'job/result.json', r)
    with pytest.raises(ValueError):
        complete(j, tmp_path)


@pytest.mark.parametrize('algorithm,parameters', [('HBOS', {'slidingWindow': 1, 'n_bins': 10, 'tol': .5, 'n_jobs': 1}), ('COPOD', {'n_jobs': 1})])
def test_native_replay_preserves_full_coverage_and_labels_do_not_fit(tmp_path, algorithm, parameters):
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(rng.normal(size=(50, 4)), columns=list('abcd'))
    frame.insert(0, 'timestamp', np.arange(50))
    frame['is_anomaly'] = [0] * 40 + [1] * 10
    test = tmp_path / 'test.csv'
    frame.to_csv(test, index=False)
    expected = environment_versions()
    write_json(tmp_path / 'environment.json', expected)
    q = {'source_receipts': [], 'jobs': [], 'environment_lock': 'environment.json', 'settings': {'cpu_threads': 1},
         'author_source': str(ROOT / 'experiments/runs/fm_grasp_paper_protocol_v1/sources/mtsbench/original'),
         'artifact_root': str(tmp_path / 'artifacts')}
    scores = []
    for attempt in range(2):
        if attempt:
            frame['is_anomaly'] = 1 - frame['is_anomaly']
            frame.to_csv(test, index=False)
        j = {'id': str(attempt), 'output_directory': 'job' + str(attempt), 'algorithm': algorithm, 'dataset': 'plumbing_only',
             'seed': 1103, 'parameters': parameters, 'protocol': 'native_transductive_test_feature_fit_oracle_f1',
             'entities': [{'id': 'e', 'test_path': 'test.csv', 'test_sha256': sha(test), 'feature_names': list('abcd')}]}
        q['jobs'] = [j]
        run(q, j, tmp_path)
        assert complete(j, tmp_path)
        result = json.loads((tmp_path / j['output_directory'] / 'result.json').read_text(encoding='utf-8'))
        assert result['entities'][0]['points'] == 50
        assert result['strict_TAB_result'] is False
        score_path = next(Path(a['path']) for a in result['artifacts'] if a['path'].endswith('.npz'))
        with np.load(score_path, allow_pickle=False) as values:
            scores.append(values['scores'].copy())
        if attempt:
            score_path.write_bytes(b'tampered')
            with pytest.raises(ValueError, match='artifact changed'):
                complete(j, tmp_path)
    np.testing.assert_array_equal(*scores)
