import copy

import pytest

from scripts.flow_matching.publish_brits_three_seed_listing_v36 import complete_delta, validated_cohort, SEEDS
from tests.test_fm_brits_two_seed_publication_v27 import proof


def test_three_original_seeds_require_complete_original_model_proofs():
    value = proof()
    third = copy.deepcopy(value['records'][1])
    third.update(seed=28, id='saits_paper_brits_electricity_author_missing0.1_seed28')
    value['records'].append(third)
    value['full_models_replayed'] = 3
    assert [r['seed'] for r in validated_cohort([value], SEEDS)] == SEEDS
    for key, replacement in [('seed', 29), ('model_parameters', 100), ('held_out_coordinates', 1020953)]:
        wrong = copy.deepcopy(value)
        wrong['records'][2][key] = replacement
        with pytest.raises(ValueError):
            validated_cohort([wrong], SEEDS)


def test_only_full_three_metric_original_2_to_3_of_5_update_is_accepted():
    before = [dict(algorithm='brits', dataset='electricity', task='imputation', protocol='author',
        pattern='paper_mask', missing_ratio='0.1', metric=m, n='2', required_repeats='5', mean='.8')
        for m in ('mae', 'rmse', 'mre')]
    before.append(dict(algorithm='other', metric='f1', mean='.3'))
    after = copy.deepcopy(before)
    for row in after[:3]:
        row.update(n='3', mean='.85')
    assert complete_delta(before, after) == [0, 1, 2]
    for index, key, value in [(0, 'n', '5'), (0, 'required_repeats', '3'),
            (0, 'missing_ratio', '0.2'), (3, 'mean', '.4')]:
        wrong = copy.deepcopy(after)
        wrong[index][key] = value
        with pytest.raises(ValueError):
            complete_delta(before, wrong)
    with pytest.raises(ValueError):
        complete_delta(before, after[:-1])
