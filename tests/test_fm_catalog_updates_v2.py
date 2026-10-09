import copy
import pytest
from scripts.flow_matching.update_algorithm_dataset_catalog_v2 import replace_audited_rows


def fixture_rows():
    cfm = {'captured_utc': 'new', 'seed_aggregates': [{'variant': 'bb', 'dataset': 'pendulum',
        'track': 'appendix', 'epochs': 400, 'comparison': 'partial',
        **{metric + '_' + field: value for metric in ('MSE', 'MAE', 'RMSE')
           for field, value in [('n', 2), ('mean', 0.5), ('std', 0.1), ('se', 0.07),
                                ('ci95_low', 0.0), ('ci95_high', 1.0)]}}]}
    rows = [{'task': 'continuous_time_dynamics', 'algorithm': 'CFM-TS/bb', 'dataset': 'pendulum',
        'protocol': 'appendix', 'epochs': '400', 'metric': metric, 'n': '1', 'source': 'old',
        'required_repeats': '5'} for metric in ('MSE', 'MAE', 'RMSE')]
    rows += [{'task': 'anomaly_detection', 'algorithm': 'main', 'dataset': 'swan',
        'protocol': 'oracle', 'metric': 'AP', 'n': '0', 'required_repeats': '10',
        'source': 'old/grasp_algorithm_dataset_metrics.csv',
        'score_recipe': "{'source_samples': 5, 'flow_evaluations': 10}"}]
    native = [{'algorithm': 'GRASP/main', 'dataset': 'SWAN', 'protocol': 'oracle', 'metric': 'AP',
        'source_samples': '5', 'flow_evaluations': '10', 'completed_seeds': '1', 'required_seeds': '10',
        'value': '0.0', 'paper_mean': '0.7', 'original_canonical_id': 'original', 'execution_id': 'recovery'}]
    return rows, cfm, native


def test_update_retains_real_zero_and_same_seed_slot():
    rows, cfm, native = fixture_rows()
    updated = replace_audited_rows(rows, cfm, native, 'cfm', 'native', 'now')
    assert len(updated) == 4 and len(rows) == 4
    assert rows[-1]['n'] == 1 and rows[-1]['mean'] == 0.0
    assert rows[-1]['std'] is None and rows[-1]['paper_mean'] == '0.7'
    assert rows[-1]['original_canonical_id'] == 'original'


def test_reject_protocol_mixing_duplicate_slot_or_seed_replacement():
    rows, cfm, native = fixture_rows()
    changed = copy.deepcopy(native); changed[0]['protocol'] = 'strict'
    with pytest.raises(KeyError):
        replace_audited_rows(copy.deepcopy(rows), cfm, changed, 'c', 'n', 'now')
    with pytest.raises(ValueError):
        replace_audited_rows(copy.deepcopy(rows) + [copy.deepcopy(rows[-1])], cfm, native, 'c', 'n', 'now')
    rows[-1]['n'] = '1'
    with pytest.raises(ValueError):
        replace_audited_rows(rows, cfm, native, 'c', 'n', 'now')
