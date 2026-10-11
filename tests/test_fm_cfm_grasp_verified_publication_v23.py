import copy
import csv
import json
from pathlib import Path

import pytest

from scripts.flow_matching.publish_cfm_grasp_verified_listing_v23 import validate_er_metadata

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'projects/flow_matching_research/results/2026-10-11/cfm_node1104_and_grasp_er1104_native_verification_v1'


def original_metadata():
    return [json.loads((REPORT / name).read_text(encoding='utf-8')) for name in
            ['grasp_preserved_complete_training_result.json', 'grasp_original_complete_training_result.json']]


def test_original_complete_metadata_accepts_two_distinct_training_seeds():
    records = original_metadata()
    before = copy.deepcopy(records)
    validate_er_metadata(records)
    assert records == before


def test_reinference_or_duplicate_attempt_is_not_another_training_seed():
    records = original_metadata()
    with pytest.raises(ValueError, match='two original ER training seeds'):
        validate_er_metadata([records[0], records[0]])
    with pytest.raises(ValueError, match='two original ER training seeds'):
        validate_er_metadata(records[:1])


@pytest.mark.parametrize('field,value', [('epochs_completed', 1499), ('optimizer_updates', 92999),
    ('parameter_count', 69243), ('scored_points', 71999), ('validation_points', 3934), ('diagnostic', True)])
def test_partial_budget_grid_architecture_or_smoke_is_not_performance(field, value):
    records = original_metadata()
    records[1][field] = value
    with pytest.raises(ValueError, match='full architecture, training and scoring budgets'):
        validate_er_metadata(records)


@pytest.mark.parametrize('field,value', [('point_adjustment', True), ('test_labels_used_for_calibration', True),
    ('test_features_used_for_calibration', True), ('quantile', .98)])
def test_validation_controls_cannot_silently_become_oracle_or_pa(field, value):
    records = original_metadata()
    records[1]['metrics'][0]['metrics']['validation_only_control'][field] = value
    with pytest.raises(ValueError, match='protocols must stay separate'):
        validate_er_metadata(records)


def test_published_increment_preserves_every_other_row_and_training_repeat_budget():
    directory = ROOT / 'projects/flow_matching_research/results/2026-10-11'
    def rows(version):
        with (directory / version / 'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    before, after = rows('all_algorithm_dataset_metrics_v22'), rows('all_algorithm_dataset_metrics_v23')
    assert len(before) == len(after) == 7283
    changed = [(old, new) for old, new in zip(before, after) if old != new]
    assert len(changed) == 9
    for old, new in changed:
        assert old['required_repeats'] == new['required_repeats']
        assert new['author_equivalence_certified'] == 'False'
        if new['algorithm'] == 'CFM-TS/node':
            assert (new['dataset'], new['protocol'], new['epochs']) == ('pendulum', 'appendix_d', '10')
            assert (old['n'], new['n'], new['required_repeats']) == ('4', '5', '5')
        else:
            assert new['algorithm'] == 'GRASP/er' and new['dataset'] == 'swan'
            assert (old['n'], new['n'], new['required_repeats']) == ('1', '2', '10')
        for field in ['std', 'se', 'ci95_low', 'ci95_high']:
            assert new[field] != ''
    assert sum(bool(row['mean']) for row in after) == 1793
