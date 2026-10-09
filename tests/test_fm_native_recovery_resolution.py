import json
from pathlib import Path
from types import SimpleNamespace


def test_original_first_then_first_verified_recovery_without_selecting_best_score(tmp_path, monkeypatch):
    import scripts.flow_matching.audit_native_recovery_progress as audit
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    valid = {'original': True, 'retry1': True, 'retry2': True}
    worker = SimpleNamespace(verify=lambda queue: None, complete=lambda job: valid[job['id']])
    monkeypatch.setattr(audit, 'components', lambda track: (worker, None))
    monkeypatch.setattr(audit, 'verify', lambda queue: None)
    job = {'id': 'original', 'algorithm': 'HBOS', 'dataset': 'swan', 'seed': 1103,
           'protocol': 'native_transductive_test_feature_fit_oracle_f1', 'output_directory': 'original'}
    (tmp_path / 'original.json').write_text(json.dumps({'jobs': [job]}))
    for i, metric in enumerate([.2, .1, .9]):
        name = ['original', 'retry1', 'retry2'][i]
        output = tmp_path / name; output.mkdir()
        (output / 'result.json').write_text(json.dumps({'metrics': {'Best_F1': metric}}))
        if i:
            candidate = dict(job, id=name, output_directory=name, recovery_original_job_id='original')
            (tmp_path / (name + '.json')).write_text(json.dumps({
                'recovery_track': 'statistical', 'recovery_original_queue': 'original.json', 'jobs': [candidate]}))
    captures, _ = audit.canonical_statistical_jobs('original.json', ['retry1.json', 'retry2.json'])
    assert len(captures) == 1 and captures[0]['execution_id'] == 'original'
    valid['original'] = False
    captures, _ = audit.canonical_statistical_jobs('original.json', ['retry1.json', 'retry2.json'])
    assert len(captures) == 1 and captures[0]['id'] == 'original'
    assert captures[0]['execution_id'] == 'retry1' and captures[0]['metrics']['Best_F1'] == .1
    valid['retry1'] = False
    captures, _ = audit.canonical_statistical_jobs('original.json', ['retry1.json', 'retry2.json'])
    assert len(captures) == 1 and captures[0]['execution_id'] == 'retry2'
