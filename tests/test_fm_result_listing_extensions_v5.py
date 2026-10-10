import json
from pathlib import Path

import pytest

from scripts.flow_matching import result_listing_extensions_v5 as listing
from scripts.flow_matching.publish_result_listing_v4 import validate_rows


def author_report(tmp_path, monkeypatch, repeats=1):
    monkeypatch.setattr(listing, 'ROOT', tmp_path)
    result = tmp_path / 'result.json'
    result.write_text('{}', encoding='utf-8')
    completed = dict(algorithm='pi_transformer_historical_mirror', dataset='PSM',
        author_recipe='full', status='completed', seed=1103, result_path='result.json',
        result_sha256=listing.sha(result),
        metrics=dict(precision=.97, recall=.98, f1=.9789,
            pointwise_no_PA_same_author_threshold=dict(precision=.27, recall=.01, f1=.0191)))
    return dict(tsad_capture_utc='fixed', author_pipeline_jobs=[completed] +
        [dict(completed, status='pending', seed=1104 + i) for i in range(repeats-1)])


def test_author_pa_and_same_threshold_pointwise_are_distinct(tmp_path, monkeypatch):
    report = author_report(tmp_path, monkeypatch, repeats=5)
    rows = listing.author_rows(report, 'capture')
    validate_rows(rows)
    f1 = {r['protocol']: r for r in rows if r['metric'] == 'f1'}
    assert f1['author_threshold_PA']['mean'] == .9789
    assert f1['same_author_threshold_no_PA']['mean'] == .0191
    assert all(r['required_repeats'] == 5 and r['n'] == 1 and r['std'] is None for r in rows)


def test_changed_author_result_is_rejected(tmp_path, monkeypatch):
    report = author_report(tmp_path, monkeypatch)
    (tmp_path / 'result.json').write_text('{"changed":true}', encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        listing.author_rows(report, 'capture')


def table45_fixture(tmp_path, monkeypatch, completed=False):
    from scripts.flow_matching import run_spectral_table45_original_v1 as worker
    monkeypatch.setattr(listing, 'ROOT', tmp_path)
    job = dict(model_config='model.json', output_directory='run', id='generator')
    queue = dict(jobs=[job])
    model = dict(task='generation', method='spectral_flow', table=4, dataset='stock',
        data_audit_key='regular', metric='discriminative', evaluator_repeats=10,
        author_arguments=dict(epochs=600, missing_value=0, w_eig=0),
        paper_claim=dict(mean=.1, reported_plus_minus=.02))
    (tmp_path / 'queue.json').write_text(json.dumps(queue), encoding='utf-8')
    (tmp_path / 'model.json').write_text(json.dumps(model), encoding='utf-8')
    monkeypatch.setattr(worker, 'verify', lambda q: None)
    if completed:
        (tmp_path / 'run').mkdir()
        (tmp_path / 'run/result.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(worker, 'verify_result', lambda q, j:
        dict(mean=.2, std=.04, se=.01, ci95=[.18,.22]))
    return worker


def test_evaluator_repeats_do_not_become_training_seed_uncertainty(tmp_path, monkeypatch):
    table45_fixture(tmp_path, monkeypatch, completed=True)
    rows, _ = listing.table45_rows(dict(table45_queue='queue.json'))
    validate_rows(rows)
    row = rows[0]
    assert row['n'] == row['required_repeats'] == 1
    assert row['evaluator_repeats'] == 10 and row['evaluator_std'] == .04
    assert row['std'] is None and row['ci95_low'] is None


def test_pending_result_keeps_missing_and_bad_receipt_fails(tmp_path, monkeypatch):
    worker = table45_fixture(tmp_path, monkeypatch)
    rows, jobs = listing.table45_rows(dict(table45_queue='queue.json'))
    validate_rows(rows)
    assert rows[0]['n'] == 0 and rows[0]['mean'] is None
    assert jobs[0]['status'] == 'no_complete_result'
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run/result.json').write_text('{}', encoding='utf-8')
    def reject(*args):
        raise ValueError('checkpoint differs')
    monkeypatch.setattr(worker, 'verify_result', reject)
    with pytest.raises(ValueError, match='checkpoint differs'):
        listing.table45_rows(dict(table45_queue='queue.json'))


def test_overview_distinguishes_missing_zero_and_unregistered():
    report = dict(tsad_capture_utc='fixed', strict_summary=[
        dict(algorithm_config='method', dataset='PSM', completed_seeds=1, required_seeds=5, f1_mean=0.),
        dict(algorithm_config='method', dataset='MSL', completed_seeds=0, required_seeds=5, f1_mean=None)])
    page = listing.overview(report, None)
    assert '| method | 0.00 [1/5] | 未登记 | — | 未登记 | 未登记 |' in page
    report['strict_summary'].append(dict(report['strict_summary'][0], dataset='unaccounted'))
    with pytest.raises(ValueError, match='lost a registered'):
        listing.overview(report, None)


def test_table4_without_physics_weight_does_not_invent_one(tmp_path, monkeypatch):
    table45_fixture(tmp_path, monkeypatch)
    path = tmp_path / 'model.json'
    model = json.loads(path.read_text())
    del model['author_arguments']['w_eig']
    path.write_text(json.dumps(model), encoding='utf-8')
    rows, _ = listing.table45_rows(dict(table45_queue='queue.json'))
    assert rows[0]['score_recipe'] == ''
