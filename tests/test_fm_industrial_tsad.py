"""Industrial leakage, group pooling and native coverage contracts."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.flow_matching.prepare_industrial_tsad import (
    contiguous_normal_intervals, normalize_groups, validate_group_roles,
)
from scripts.flow_matching.register_industrial_tsad import model_window, model_roster
from scripts.flow_matching.run_industrial_tsad_experiment import pooled_training_windows, verify_training_and_checkpoint
from scripts.flow_matching.run_tsad_experiment import covering_windows
from scripts.flow_matching.model_registry import load_model_config


def fixture_roles():
    return {role: [{'group_id': role, 'source_path': role+'.csv', 'source_interval': [0, 6],
                    'values': np.arange(12., dtype=float).reshape(6, 2)}]
            for role in ['train', 'validation', 'test']}


def test_scaler_and_fit_arrays_ignore_validation_and_test_distribution():
    roles = fixture_roles()
    baseline, first = normalize_groups(roles)
    roles['validation'][0]['values'] += 10000
    roles['test'][0]['values'] *= -1000
    changed, second = normalize_groups(roles)
    for key in ['median', 'mean', 'std', 'fit_points']:
        assert first[key] == second[key]
    np.testing.assert_array_equal(baseline['train'][0], changed['train'][0])
    assert not np.array_equal(changed['test'][0], baseline['test'][0])


@pytest.mark.parametrize('key', ['group_id', 'source_path'])
def test_run_day_and_raw_source_reuse_across_roles_is_rejected(key):
    roles = fixture_roles()
    roles['test'][0][key] = roles['train'][0][key]
    with pytest.raises(ValueError, match='shared across roles'):
        validate_group_roles(roles)


def test_repeated_training_source_interval_is_rejected():
    roles = fixture_roles()
    roles['train'].append(copy.deepcopy(roles['train'][0]))
    with pytest.raises(ValueError, match='overlapping source'):
        normalize_groups(roles)


def test_normal_intervals_keep_fault_gaps_out_of_windows():
    labels = np.array(['Normal']*5+['Fault']*2+['Normal']*4)
    intervals = contiguous_normal_intervals(labels)
    assert intervals == [(0, 5), (7, 11)]
    values = np.arange(len(labels))[:, None]
    windows = np.concatenate([covering_windows(values[a:b], 3)[0] for a, b in intervals])
    assert not np.isin(windows, [5, 6]).any()
    assert np.all(np.diff(windows, axis=1) == 1)


def test_shared_fit_run_not_duplicated_for_many_fault_test_entities():
    dataset = {'training_groups': [{'array_prefix': 't000', 'group_id': 'normal', 'source_interval': [0, 10], 'points': 10}],
               'scaler': {'fit_points': 10}, 'entities': [{'entity_id': str(i)} for i in range(21)]}
    data = {'t000_values': np.arange(20.).reshape(10, 2)}
    windows, budget = pooled_training_windows(data, dataset, 4)
    assert len(windows) == 3
    assert sum(g['points'] for g in budget) == 10
    dataset['entities'] *= 2
    second, _ = pooled_training_windows(data, dataset, 4)
    np.testing.assert_array_equal(second, windows)


def test_full_reflow_budget_and_finite_checkpoint_required():
    class Model:
        training_losses_ = [1.]*60
    cfg = {'parameters': {'epochs': 20, 'reflow_stages': 3, 'reflow_epochs': 20}}
    _, audit = verify_training_and_checkpoint(Model(), cfg, {'model': {'w': torch.ones(2)}})
    assert audit['declared_epochs'] == audit['recorded_epochs'] == 60
    model = Model()
    model.training_losses_ = [1.]*20
    with pytest.raises(ValueError, match='full training epoch'):
        verify_training_and_checkpoint(model, cfg, {'model': {'w': torch.ones(2)}})
    with pytest.raises(ValueError, match='nonfinite industrial checkpoint'):
        verify_training_and_checkpoint(Model(), cfg, {'model': {'w': torch.tensor([float('inf')])}})


def test_roster_preserves_all_56_configs_and_nested_baseline_windows():
    root = Path(__file__).resolve().parents[1]
    settings = load_model_config(root/'configs/experiments/fm_industrial_tsad_execution.v1.json')
    roster = model_roster(root, settings)
    assert len(set(roster)) == 56
    assert all(load_model_config(root/p)['parameters']['epochs'] > 0 for p in roster)
    assert model_window({'parameters': {'parameters': {'window': 192}}}, 100) == 192
    assert model_window({'parameters': {'window': 8}}, 100) == 8
