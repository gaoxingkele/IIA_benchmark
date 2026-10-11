import copy

import pytest

from scripts.flow_matching import publish_ready_strict_metrics_v33 as publication


def fixture(monkeypatch):
    before = dict(id='original', output_directory='first', seed=1105, model_config='model',
        dataset='SMAP', window=100, parameter_overrides={'device': 'cpu'}, frozen_files=[{'sha256': 'abc'}])
    after = dict(before, id='recovered', output_directory='preserved_recovery')
    files = {'recovery': {'jobs': [dict(original_id='original', original_queue='canonical',
        original_queue_sha256='hash', original_job=before, job=after)]},
        'canonical': {'jobs': [before]}, 'mirror': {'jobs': [after]},
        'result': dict(job=after, status='completed')}
    monkeypatch.setattr(publication.original, 'read', lambda p: files[str(p.relative_to(publication.ROOT))])
    monkeypatch.setattr(publication.original, 'stream_sha', lambda p: 'hash')
    binding = dict(original_id='original', recovery_id='recovered', recovery_queue='recovery', mirror_queue='mirror')
    cohorts = [dict(runs=[dict(id='recovered', seed=1105, result='result')])]
    return before, after, files, binding, cohorts


def test_original_budget_recovery_is_one_original_seed(monkeypatch):
    _, _, _, binding, cohorts = fixture(monkeypatch)
    proof = publication.verify_recovery_binding(binding, cohorts)
    assert proof['seed'] == 1105 and proof['new_training_seed_slots'] == 0


@pytest.mark.parametrize('field,value', [('seed', 1106), ('window', 50), ('model_config', 'smaller'),
    ('parameter_overrides', {'device': 'cuda'}), ('frozen_files', [])])
def test_recovery_cannot_change_original_science(monkeypatch, field, value):
    _, after, _, binding, cohorts = fixture(monkeypatch)
    after[field] = value
    with pytest.raises(ValueError, match='only change'):
        publication.verify_recovery_binding(binding, cohorts)


def test_duplicate_original_and_recovery_seed_rejected(monkeypatch):
    _, _, _, binding, cohorts = fixture(monkeypatch)
    cohorts[0]['runs'].append(dict(id='original', seed=1105))
    with pytest.raises(ValueError, match='both count'):
        publication.verify_recovery_binding(binding, cohorts)


def test_recovery_result_must_match_frozen_mirror_and_complete(monkeypatch):
    _, _, files, binding, cohorts = fixture(monkeypatch)
    files['result']['job'] = copy.deepcopy(files['result']['job'])
    files['result']['job']['seed'] = 1106
    with pytest.raises(ValueError, match='result'):
        publication.verify_recovery_binding(binding, cohorts)
