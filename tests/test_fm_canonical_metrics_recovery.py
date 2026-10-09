"""Result exports retain seed slots and prefer provenance over a better score."""
import json
from types import SimpleNamespace

import pytest

from scripts.flow_matching import audit_native_recovery_progress as audit


@pytest.mark.parametrize('original_ready', [False, True])
def test_recovery_resolves_one_slot_without_metric_selection(tmp_path, monkeypatch, original_ready):
    original = {'jobs': [{'id': 'original', 'algorithm': 'HBOS', 'dataset': 'x',
                         'protocol': 'native', 'seed': 1103, 'output_directory': 'original'}]}
    recovery = {'recovery_track': 'statistical', 'recovery_original_queue': 'queue.json',
                'jobs': [dict(original['jobs'][0], id=name, output_directory=name,
                              recovery_original_job_id='original') for name in ['retry1', 'retry2']]}
    (tmp_path / 'queue.json').write_text(json.dumps(original))
    (tmp_path / 'recovery.json').write_text(json.dumps(recovery))
    for name, score in [('original', .1), ('retry1', .2), ('retry2', .9)]:
        if name == 'original' and not original_ready:
            continue
        (tmp_path / name).mkdir()
        (tmp_path / name / 'result.json').write_text(json.dumps({'metrics': {'Best_F1': score}}))
    worker = SimpleNamespace(verify=lambda queue: None,
                             complete=lambda job, root=None: (tmp_path / job['output_directory'] / 'result.json').exists())
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    monkeypatch.setattr(audit, 'components', lambda track: (worker, None))
    monkeypatch.setattr(audit, 'verify', lambda queue: None)
    rows, snapshots = audit.canonical_statistical_jobs('queue.json', ['recovery.json'])
    assert len(rows) == len(snapshots) == 1
    assert rows[0]['id'] == 'original' and rows[0]['seed'] == 1103
    assert rows[0]['execution_id'] == ('original' if original_ready else 'retry1')
    assert rows[0]['metrics']['Best_F1'] == (.1 if original_ready else .2)
    assert rows[0]['original_failed_output_preserved'] == 'original'


def test_no_complete_attempt_remains_missing(tmp_path, monkeypatch):
    queue = {'jobs': [{'id': 'original', 'algorithm': 'HBOS', 'dataset': 'x',
                      'protocol': 'native', 'seed': 1103, 'output_directory': 'original'}]}
    (tmp_path / 'queue.json').write_text(json.dumps(queue))
    worker = SimpleNamespace(verify=lambda queue: None, complete=lambda job: False)
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    monkeypatch.setattr(audit, 'components', lambda track: (worker, None))
    rows, snapshots = audit.canonical_statistical_jobs('queue.json', [])
    assert len(rows) == 1 and not snapshots
    assert rows[0]['metrics'] is None and rows[0]['result_sha256'] is None
    assert rows[0]['status'] == 'incomplete'
