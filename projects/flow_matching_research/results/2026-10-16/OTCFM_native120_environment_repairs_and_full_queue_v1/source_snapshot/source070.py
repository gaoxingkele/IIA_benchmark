from copy import deepcopy
import numpy as np
import pytest
import torch
import hashlib
import json
from pathlib import Path

from scripts.flow_matching import otcfm_singlecell_native_contract_v1 as s


@pytest.mark.parametrize('n', [4, 7, 9546, 8965, 9407, 7466, 7264, 8544, 10933, 6960])
def test_fractional_compatibility_matches_modern_native_split_on_every_full_cohort_size(n):
    values = list(range(n)); lengths = [.8, .1, .1]
    actual = s.compatible_random_split(values, lengths, torch.Generator().manual_seed(42), torch.utils.data.random_split)
    expected = torch.utils.data.random_split(values, lengths, generator=torch.Generator().manual_seed(42))
    assert [x.indices for x in actual] == [x.indices for x in expected]
    flattened = [i for part in actual for i in part.indices]
    assert len(flattened) == len(set(flattened)) == n and sorted(flattened) == values


@pytest.mark.parametrize('size,fractions', [(0, [.8, .1, .1]), (5, [.8, .1]),
    (5, [-.1, .5, .6]), (5, [np.nan, .5, .5])])
def test_invalid_splits_cannot_hide_missing_cells(size, fractions):
    with pytest.raises(ValueError): s.fraction_lengths(size, fractions)


def test_original_integer_split_is_unchanged():
    values = list(range(10)); lengths = [8, 1, 1]
    actual = s.compatible_random_split(values, lengths, torch.Generator().manual_seed(42), torch.utils.data.random_split)
    expected = torch.utils.data.random_split(values, lengths, generator=torch.Generator().manual_seed(42))
    assert [a.indices for a in actual] == [a.indices for a in expected]


@pytest.mark.parametrize('sizes', [[9546, 8965, 9407, 7466], [7264, 8544, 10933, 6960]])
def test_one_original_generator_is_shared_across_all_four_real_timepoint_splits(sizes):
    actual_rng, expected_rng = torch.Generator().manual_seed(42), torch.Generator().manual_seed(42)
    for n in sizes:
        values = list(range(n))
        actual = s.compatible_random_split(values, [.8, .1, .1], actual_rng, torch.utils.data.random_split)
        expected = torch.utils.data.random_split(values, [.8, .1, .1], generator=expected_rng)
        assert [a.indices for a in actual] == [a.indices for a in expected]
    assert torch.equal(actual_rng.get_state(), expected_rng.get_state())


def test_both_full_scientific_recipes_remain_explicit_and_unpooled():
    released, paper = s.protocol('released_yaml'), s.protocol('paper_declared')
    assert (released['batch_size'], paper['batch_size']) == (512, 128)
    assert (released['early_stopping_patience_validation_checks'], paper['early_stopping_patience_validation_checks']) == (100, 3)
    assert (released['maximum_training_wall_seconds'], paper['maximum_training_wall_seconds']) == (None, 6000)
    assert (released['ode_solver'], paper['ode_solver']) == ('euler', 'rk4')
    assert released['max_epochs'] == paper['max_epochs'] == 1000
    assert released['ode_grid_points_per_interval'] == paper['ode_grid_points_per_interval'] == 101


@pytest.mark.parametrize('fault', ['missing_model', 'missing_seed', 'changed_budget', 'changed_solver',
    'changed_sigma', 'changed_split', 'changed_dataset', 'changed_leaveout'])
def test_all120_conditions_and_original_science_are_required(fault):
    cases = deepcopy(s.scientific_cases()); s.validate_cases(cases)
    if fault == 'missing_model': cases.pop()
    elif fault == 'missing_seed': cases[0]['seed'] = 999
    elif fault == 'changed_budget': cases[0]['scientific_protocol']['max_epochs'] = 1
    elif fault == 'changed_solver': cases[0]['scientific_protocol']['ode_solver'] = 'midpoint'
    elif fault == 'changed_sigma': cases[0]['sigma_min'] = 0.
    elif fault == 'changed_split': cases[0]['scientific_protocol']['data_split_fractions'] = [1., 0., 0.]
    elif fault == 'changed_dataset': cases[0]['dataset'] = 'smaller_proxy'
    else: cases[0]['leaveout_timepoint'] = 3
    with pytest.raises(ValueError): s.validate_cases(cases)


@pytest.mark.parametrize('recipe,epochs,stop,elapsed,expected', [
    ('released_yaml', 1000, 0, 80., 'maximum_epochs'),
    ('released_yaml', 900, 899, 100., 'native_early_stopping'),
    ('paper_declared', 40, 39, 100., 'native_early_stopping'),
    ('paper_declared', 80, 0, 6000., 'native_paper_wall_time_cap')])
def test_complete_original_stopping_rules_are_explicit(recipe, epochs, stop, elapsed, expected):
    assert s.terminal_reason(s.protocol(recipe), epochs, stop, elapsed, False, True) == expected


@pytest.mark.parametrize('recipe,epochs,stop,elapsed,interrupted,finite', [
    ('released_yaml', 999, 0, 10000., False, True),
    ('paper_declared', 30, 0, 5999., False, True),
    ('released_yaml', 1000, 0, 100., True, True),
    ('released_yaml', 1000, 0, 100., False, False),
    ('paper_declared', 0, 0, 6000., False, True)])
def test_partial_interrupted_or_unqualified_fit_never_counts_as_complete(recipe, epochs, stop, elapsed, interrupted, finite):
    with pytest.raises(ValueError): s.terminal_reason(s.protocol(recipe), epochs, stop, elapsed, interrupted, finite)


def test_each_native_method_has_a_callable_config_and_original_literature_data_sources():
    root = Path(__file__).resolve().parents[1]
    for method in s.METHODS:
        c = json.loads((root / f'configs/models/fm_otcfm_singlecell_native__{method}.v1.json').read_text())
        case = next(j for j in s.scientific_cases() if j['method'] == method)
        assert c['callable_implementation'].endswith(':instantiate_case')
        assert c['parameters']['net'] == case['net']
        assert c['parameters']['sigma_min'] == case['sigma_min'] and c['parameters']['ot_sampler'] == case['ot_sampler']
        assert hashlib.sha256((root / c['native_author_yaml']).read_bytes()).hexdigest() == c['native_author_yaml_sha256']
        assert c['citation'] == 'https://arxiv.org/abs/2302.00482' and len(c['data_citations']) == 2
        assert not c['historical_paper_equivalence_certified'] and not c['strict_TAB_anomaly_detection_method']
