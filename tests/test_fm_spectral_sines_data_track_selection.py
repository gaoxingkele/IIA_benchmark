import pytest

from scripts.flow_matching.audit_spectral_completed_cohorts_v1 import validate_selection
from tests.test_fm_spectral_completed_cohorts_v1 import fixture as original_fixture


def fixture():
    config, queue, audits = original_fixture()
    queue['jobs'][0].update(dataset='sines', seed=42, data_track='released_author_data')
    queue['jobs'][1].update(dataset='sines', seed=42, data_track='paper_sines_distribution')
    config['completed_cohort_seeds'] = {'sines': [42]}
    config['audits'] = config['audits'][:1]
    return config, queue, audits


def test_same_dataset_and_seed_data_tracks_remain_separate_without_shrinking_scope():
    config, queue, audits = fixture()
    validate_selection(config, queue, audits)
    assert len(queue['jobs']) == 70
    assert {j['data_track'] for j in queue['jobs'] if j['dataset'] == 'sines'} == {
        'released_author_data', 'paper_sines_distribution'}
    assert config['original_required_seeds'] == [42,43,44,45,46]


@pytest.mark.parametrize('fault', ['wrong_track', 'both_tracks', 'drop_other_track', 'second_required_job'])
def test_ambiguous_track_or_reduced_original_obligations_cannot_be_registered(fault):
    config, queue, audits = fixture()
    if fault == 'wrong_track':
        config['audits'][0]['id'] = 'job1'
    elif fault == 'both_tracks':
        config['audits'].append(dict(id='job1', configuration='audit1'))
    elif fault == 'drop_other_track':
        queue['jobs'].pop(1)
    else:
        queue['jobs'][1]['data_track'] = 'released_author_data'
    with pytest.raises(ValueError):
        validate_selection(config, queue, audits)
