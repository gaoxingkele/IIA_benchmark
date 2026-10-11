from copy import deepcopy
from pathlib import Path

import pytest

from scripts.flow_matching import audit_spectral_stocks_last_seed_v3 as runner


def fixture(monkeypatch):
    guards = dict(minimum_available_memory_bytes=12, minimum_commit_headroom_bytes=20,
                  emergency_commit_headroom_bytes=12)
    job = dict(id='final_seed46', seed=46, model='diffusion_ts', dataset='stocks',
        data_track='released_author_data', training_updates=10000, gradient_accumulate_every=2,
        generation_batch_size=2001, generation_field_chunk_size=None)
    queue = dict(jobs=[job], settings=deepcopy(guards))
    audit = dict(job_id=job['id'], queue='queue')
    config = dict(source_receipts=[dict(path='frozen', sha256='registered')],
        original_queue='queue', original_required_seeds=[42,43,44,45,46], newly_completed_seeds=[46],
        audits=[dict(id=job['id'], configuration='audit')], settings=guards)
    monkeypatch.setattr(runner, 'ROOT', Path('/fixture'))
    monkeypatch.setattr(runner, 'sha', lambda path: 'registered')
    monkeypatch.setattr(runner, 'read', lambda path: {'queue':queue, 'audit':audit}[path.name])
    return config, queue, job


def test_last_original_seed_requires_all_five_seed_obligation(monkeypatch):
    config, _, _ = fixture(monkeypatch)
    runner.registration(config)
    config['original_required_seeds'] = [46]
    with pytest.raises(ValueError, match='five-seed obligation'):
        runner.registration(config)


@pytest.mark.parametrize('field,value', [('newly_completed_seeds',[42]), ('audits',[]),
    ('audits',[dict(id='final_seed46',configuration='audit')]*2)])
def test_missing_duplicate_or_substituted_seed_refused(monkeypatch, field, value):
    config, _, _ = fixture(monkeypatch)
    config[field] = value
    with pytest.raises(ValueError, match='five-seed obligation'):
        runner.registration(config)


@pytest.mark.parametrize('field,value', [('seed',45), ('model','spectral_flow'), ('dataset','energy'),
    ('data_track','other'), ('training_updates',100), ('gradient_accumulate_every',1),
    ('generation_batch_size',10), ('generation_field_chunk_size',100)])
def test_full_original_model_data_sampling_and_update_budget_preserved(monkeypatch, field, value):
    config, _, job = fixture(monkeypatch)
    job[field] = value
    with pytest.raises(ValueError, match='full budget'):
        runner.registration(config)


@pytest.mark.parametrize('field', ['minimum_available_memory_bytes','minimum_commit_headroom_bytes',
                                 'emergency_commit_headroom_bytes'])
def test_original_native_host_guards_unchanged(monkeypatch, field):
    config, _, _ = fixture(monkeypatch)
    config['settings'][field] = 1
    with pytest.raises(ValueError, match='memory safeguards'):
        runner.registration(config)
