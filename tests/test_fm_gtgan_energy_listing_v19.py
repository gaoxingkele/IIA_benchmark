import copy
import csv
from pathlib import Path

import pytest

from scripts.flow_matching.gtgan_energy_native_common_v1 import recipes, release_recipes
from scripts.flow_matching.register_gtgan_native_commands_v1 import native_arguments
from scripts.flow_matching.publish_gtgan_energy_listing_v19 import pending_energy


def inputs():
    source = Path('data/public_datasets/flow_matching/gtgan_native_source_v1/original')
    cases, released = [], []
    for release, collection, commands in [(False, cases, recipes()), (True, released, release_recipes())]:
        for rate, arguments in commands:
            collection.append(dict(id=str(release) + '_' + str(rate), missing_rate=rate, original_seed=7777,
                execution_mode='native_release_checkpoint_evaluation' if release else 'native_from_scratch_training',
                parsed_original_arguments=native_arguments(source, 'GTGAN_energy.py', arguments)))
    model = dict(cases=cases, release_evaluation_cases=released)
    queue = dict(jobs=[dict(case_id=c['id']) for c in cases + released])
    p = Path('projects/flow_matching_research/results/2026-10-11/all_algorithm_dataset_metrics_v18/all_algorithm_dataset_metrics.csv')
    with p.open(encoding='utf-8-sig') as stream:
        headers = csv.DictReader(stream).fieldnames
    return model, queue, headers


def test_all_eight_metrics_remain_pending_with_distinct_generator_training_and_release_modes():
    model, queue, headers = inputs()
    rows = pending_energy(model, queue, '2026-10-11T00:00:00+00:00', headers, 'queue.json')
    assert len(rows) == 8
    assert all(r['n'] == 0 and r['mean'] == '' and r['std'] == '' and r['required_repeats'] == 1 for r in rows)
    assert all(r['author_equivalence_certified'] is False and r['strict_TAB_result'] is False for r in rows)
    assert sum('released-checkpoint' in r['algorithm'] for r in rows) == 4
    assert all(r['evaluator_repeats'] == 10 for r in rows)
    predictive = [r for r in rows if r['metric'] == 'predictive']
    assert {r['metric_unit'] for r in predictive} == {'native_normalized_last_feature_MAE', 'native_normalized_all28_features_MAE'}


def test_missing_original_command_is_rejected_instead_of_silently_shrinking_scope():
    model, queue, headers = inputs()
    broken = copy.deepcopy(queue)
    broken['jobs'].pop()
    with pytest.raises(ValueError, match='four'):
        pending_energy(model, broken, 'timestamp', headers, 'queue.json')
