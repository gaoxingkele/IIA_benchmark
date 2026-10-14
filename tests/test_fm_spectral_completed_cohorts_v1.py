import copy

import pytest

from scripts.flow_matching.audit_spectral_completed_cohorts_v1 import validate_selection


def fixture():
    jobs = [dict(id=f'job{i}', model='spectral_flow', data_track='released_author_data',
        dataset='energy', seed=42, generation_batch_size=2001, generation_field_chunk_size=None) for i in range(70)]
    for i, seed in enumerate([42, 43, 44, 45, 46]):
        jobs[i].update(model='diffusion_ts', seed=seed)
    settings = dict(minimum_available_memory_bytes=12, minimum_commit_headroom_bytes=20, emergency_commit_headroom_bytes=12)
    queue = dict(jobs=jobs, settings=settings, state_root='original')
    config = dict(original_queue='frozen.json', original_required_seeds=[42, 43, 44, 45, 46],
        completed_cohort_seeds={'energy': [42, 43, 44, 45, 46]}, settings=copy.deepcopy(settings),
        audits=[dict(id=f'job{i}', configuration=f'audit{i}') for i in range(5)])
    audits = {f'job{i}': dict(queue='frozen.json', job_id=f'job{i}', terminal_receipt=f'original/receipts/job{i}.json') for i in range(5)}
    return config, queue, audits


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'other_model', 'queue', 'id', 'receipt', 'chunk', 'batch', 'guard', 'scope', 'seeds'])
def test_reject_incomplete_or_changed_full_model_obligations(fault):
    config, queue, audits = fixture()
    validate_selection(config, queue, audits)
    if fault == 'missing': config['audits'].pop()
    elif fault == 'duplicate': config['audits'][1] = config['audits'][0]
    elif fault == 'other_model': config['audits'][0]['id'] = 'job69'
    elif fault in ['queue', 'id', 'receipt']:
        audits['job0'][{'queue': 'queue', 'id': 'job_id', 'receipt': 'terminal_receipt'}[fault]] = 'changed'
    elif fault == 'chunk': queue['jobs'][0]['generation_field_chunk_size'] = 1
    elif fault == 'batch': queue['jobs'][0]['generation_batch_size'] = 2
    elif fault == 'guard': config['settings']['minimum_commit_headroom_bytes'] = 1
    elif fault == 'scope': queue['jobs'].pop()
    elif fault == 'seeds': config['original_required_seeds'].pop()
    with pytest.raises(ValueError):
        validate_selection(config, queue, audits)
