import copy

import pytest

from scripts.flow_matching.audit_spectral_completed_cohorts_v2 import validate_selection


def fixture():
    jobs = [dict(id=f'j{i}', model='spectral_flow', dataset='stocks',
        data_track='released_author_data', seed=42, generation_batch_size=2001,
        generation_field_chunk_size=None) for i in range(70)]
    for offset, track in [(0, 'released_author_data'), (5, 'paper_sines_distribution')]:
        for i, seed in enumerate([42, 43, 44, 45, 46], offset):
            jobs[i].update(model='diffusion_ts', dataset='sines', data_track=track, seed=seed)
    settings = dict(minimum_available_memory_bytes=12, minimum_commit_headroom_bytes=20,
                    emergency_commit_headroom_bytes=12)
    queue = dict(jobs=jobs, settings=settings, state_root='original')
    config = dict(original_queue='frozen.json', completed_data_track='paper_sines_distribution',
        original_required_seeds=[42, 43, 44, 45, 46], completed_cohort_seeds={'sines': [42]},
        settings=copy.deepcopy(settings), audits=[dict(id='j5', configuration='a.json')])
    audits = {j['id']: dict(queue='frozen.json', job_id=j['id'],
        terminal_receipt='original/receipts/' + j['id'] + '.json') for j in jobs[:10]}
    return config, queue, audits


def test_exact_same_seed_other_distribution_is_not_reused():
    c, q, a = fixture()
    validate_selection(c, q, a)
    c['audits'][0]['id'] = 'j0'
    with pytest.raises(ValueError):
        validate_selection(c, q, a)
    c['completed_data_track'] = 'released_author_data'
    validate_selection(c, q, a)


@pytest.mark.parametrize('fault', ['missing_job', 'duplicate_job', 'missing_audit',
    'duplicate_audit', 'wrong_receipt', 'wrong_queue', 'wrong_id', 'chunk', 'batch',
    'guard', 'seed_scope', 'unknown_track', 'wrong_dataset', 'missing_seed',
    'duplicate_seed', 'empty_seeds'])
def test_frozen_scope_and_original_complete_budget_are_required(fault):
    c, q, a = fixture()
    validate_selection(c, q, a)
    if fault == 'missing_job': q['jobs'].pop()
    elif fault == 'duplicate_job': q['jobs'][-1] = q['jobs'][-2]
    elif fault == 'missing_audit': c['audits'].clear()
    elif fault == 'duplicate_audit': c['audits'] *= 2
    elif fault == 'wrong_receipt': a['j5']['terminal_receipt'] = 'other.json'
    elif fault == 'wrong_queue': a['j5']['queue'] = 'other.json'
    elif fault == 'wrong_id': a['j5']['job_id'] = 'j0'
    elif fault == 'chunk': q['jobs'][5]['generation_field_chunk_size'] = 1
    elif fault == 'batch': q['jobs'][5]['generation_batch_size'] = 2
    elif fault == 'guard': c['settings']['minimum_commit_headroom_bytes'] = 1
    elif fault == 'seed_scope': c['original_required_seeds'].pop()
    elif fault == 'unknown_track': c['completed_data_track'] = 'synthetic'
    elif fault == 'wrong_dataset': c['completed_cohort_seeds'] = {'stocks': [42]}
    elif fault == 'missing_seed': c['completed_cohort_seeds']['sines'] = [99]
    elif fault == 'duplicate_seed': c['completed_cohort_seeds']['sines'] = [42, 42]
    elif fault == 'empty_seeds': c['completed_cohort_seeds']['sines'] = []
    with pytest.raises(ValueError):
        validate_selection(c, q, a)
