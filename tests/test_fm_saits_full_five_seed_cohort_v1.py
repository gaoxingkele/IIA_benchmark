import copy

import pytest

from scripts.flow_matching.summarize_saits_full_five_seed_cohort_v1 import summarize


def rows():
    return [dict(seed=seed, method='saits', dataset='electricity', nominal_missing_ratio=.2,
        dataset_sha256='frozen', original_metrics={key: float(seed) for key in ('mae', 'rmse', 'mre')},
        independent_metrics={key: float(seed) for key in ('mae', 'rmse', 'mre')}) for seed in range(26, 31)]


def test_five_training_models_have_df4_and_unclipped_descriptive_intervals():
    summary = summarize(rows())
    assert summary['full_five_seed_cohort_complete'] and summary['independently_trained_models'] == 5
    for values in summary['summaries'].values():
        assert values['mean'] == 28 and values['n'] == 5 and values['degrees_of_freedom'] == 4
        assert values['std'] == 2.5 ** .5
        assert values['independent_scalar_reduction_and_df4_interval_verified']


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'split', 'method', 'rate', 'nonfinite', 'negative', 'tolerance'])
def test_incomplete_mixed_or_unverified_metrics_cannot_enter_full_cohort(fault):
    values = copy.deepcopy(rows())
    if fault == 'missing': values.pop()
    elif fault == 'duplicate': values[1]['seed'] = 26
    elif fault == 'split': values[0]['dataset_sha256'] = 'changed'
    elif fault == 'method': values[0]['method'] = 'brits'
    elif fault == 'rate': values[0]['nominal_missing_ratio'] = .3
    elif fault == 'nonfinite': values[0]['independent_metrics']['mae'] = float('nan')
    elif fault == 'negative': values[0]['original_metrics']['mae'] = -1
    elif fault == 'tolerance': values[0]['independent_metrics']['mae'] += .01
    with pytest.raises(ValueError): summarize(values)
