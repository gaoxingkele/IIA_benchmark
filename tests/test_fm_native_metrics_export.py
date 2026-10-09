import copy
import numpy as np
import pytest

from scripts.flow_matching.collect_native_metrics import audit_stat_result
from scripts.flow_matching.mtsbench_stat_protocol import sha


def fixture(tmp_path):
    raw = tmp_path / 'test.csv'
    raw.write_text('timestamp,x,is_anomaly\n0,1,0\n1,2,1\n')
    np.savez(tmp_path / 'entity_scores.npz', scores=[1., 2.], labels=[0, 1], timestamps=['0', '1'])
    job = {'dataset': 'dataset', 'entities': [{'id': 'entity', 'test_path': 'test.csv', 'test_sha256': sha(raw)}]}
    metrics = {'PRC': 1., 'ROC': 1., 'Best_F1': .999}
    result = {'entities': [{'entity': 'entity', 'features': 1, 'points': 2, 'anomaly_points': 1, 'metrics': metrics}],
              'artifacts': [{'path': 'entity_scores.npz'}], 'metrics': metrics}
    manifest = {'datasets': [{'id': 'dataset', 'entities': [{'entity_id': 'entity', 'features': 1,
                   'test_points': 2, 'test_anomaly_points': 1}]}]}
    return job, result, manifest


def test_full_native_scope_accepts_complete_scores(tmp_path):
    job, result, manifest = fixture(tmp_path)
    audit_stat_result(job, result, manifest, tmp_path, set())


@pytest.mark.parametrize('field,value', [('points', 1), ('features', 2), ('anomaly_points', 0), ('entity', 'other')])
def test_partial_or_wrong_native_coverage_rejected(tmp_path, field, value):
    job, result, manifest = fixture(tmp_path)
    result['entities'][0][field] = value
    with pytest.raises(ValueError):
        audit_stat_result(job, result, manifest, tmp_path, set())


def test_native_macro_and_arrays_are_independently_audited(tmp_path):
    job, result, manifest = fixture(tmp_path)
    result['metrics'] = copy.deepcopy(result['metrics'])
    result['metrics']['PRC'] = .2
    with pytest.raises(ValueError, match='macro'):
        audit_stat_result(job, result, manifest, tmp_path, set())
    result['metrics']['PRC'] = 1.
    np.savez(tmp_path / 'entity_scores.npz', scores=[1.], labels=[1], timestamps=['0'])
    with pytest.raises(ValueError, match='coverage'):
        audit_stat_result(job, result, manifest, tmp_path, set())


def test_pending_native_group_has_no_fabricated_zero_or_interval(tmp_path):
    from scripts.flow_matching.collect_native_metrics import native_tables
    jobs = [{'algorithm': 'grasp_mtsbench_statistical_replay', 'recipe': 'HBOS', 'dataset': 'x',
             'track': 'native', 'status': 'pending', 'metrics': None} for _ in range(10)]
    rows, leaves, entities = native_tables(tmp_path, jobs, {})
    assert len(rows) == 3 and not leaves and not entities
    assert all(r['mean'] is None and r['std'] is None and r['ci95_low'] is None and r['n'] == 0 for r in rows)
    assert all(r['required_seeds'] == 10 for r in rows)


def test_test_feature_native_refits_never_gain_stochastic_ci(tmp_path):
    import json
    from scripts.flow_matching.collect_native_metrics import native_tables
    job, result, manifest = fixture(tmp_path)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    (tmp_path / 'queue.json').write_text(json.dumps({'jobs': [dict(job, id='slot')], 'source_receipts': [
        {'path': 'manifest.json', 'sha256': sha(tmp_path / 'manifest.json')}]}), encoding='utf-8')
    (tmp_path / 'result.json').write_text(json.dumps(result), encoding='utf-8')
    registered = dict(algorithm='grasp_mtsbench_statistical_replay', recipe='HBOS', dataset='dataset',
                      track='native', status='completed', metrics=result['metrics'], output_directory='.',
                      queue='queue.json', seed=1103, id='slot', result_sha256=sha(tmp_path / 'result.json'))
    rows, _, _ = native_tables(tmp_path, [registered] * 10, {'statistical_coverage_manifest': 'manifest.json'})
    assert all(r['n'] == 10 and r['mean'] is not None for r in rows)
    assert all(r['se'] is None and r['ci95_low'] is None and r['ci95_high'] is None for r in rows)
