import copy
import pytest

from scripts.flow_matching.publish_brits_five_seed_listing_v41 import complete_delta, validated_cohort, SEEDS
from tests.test_fm_brits_two_seed_publication_v27 import proof


def test_five_unique_full_model_proofs_complete_only_registered_brits_cohort():
    value = proof()
    for seed in (28, 29, 30):
        record = copy.deepcopy(value['records'][1])
        record.update(seed=seed, id='saits_paper_brits_electricity_author_missing0.1_seed%d' % seed)
        value['records'].append(record)
    value['full_models_replayed'] = 5
    assert [r['seed'] for r in validated_cohort([value], SEEDS)] == SEEDS
    for field, replacement in [('seed', 28), ('model_parameters', 100), ('held_out_coordinates', 1020953)]:
        wrong = copy.deepcopy(value); wrong['records'][3][field] = replacement
        with pytest.raises(ValueError): validated_cohort([wrong], SEEDS)


def test_only_three_original_metrics_can_change_without_paper_value_or_other_track_changes():
    before = [dict(algorithm='brits', dataset='electricity', task='imputation', protocol='author',
        pattern='paper_mask', missing_ratio='0.1', metric=m, n='4', required_repeats='5',
        mean='.8', paper_mean='.847') for m in ('mae', 'rmse', 'mre')]
    before.append(dict(algorithm='other', metric='f1', mean='.3'))
    after = copy.deepcopy(before)
    for row in after[:3]: row.update(n='5', mean='.85')
    assert complete_delta(before, after) == [0, 1, 2]
    for index, key, replacement in [(0, 'n', '4'), (0, 'required_repeats', '4'),
            (0, 'paper_mean', '.85'), (0, 'missing_ratio', '.2'), (3, 'mean', '.4')]:
        wrong = copy.deepcopy(after); wrong[index][key] = replacement
        with pytest.raises(ValueError): complete_delta(before, wrong)
    with pytest.raises(ValueError): complete_delta(before, after[:-1])
