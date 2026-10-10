import pytest

from scripts.flow_matching.audit_saits_full_checkpoint_replay_v1 import compare_metrics, verify_registration


def test_registered_repeat_must_bind_exactly_one_full_model_per_id():
    config = dict(source_receipts=[], full_model_count=2, completed_jobs=[dict(id='seed26'), dict(id='seed27')])
    verify_registration(config)
    config['completed_jobs'][1]['id'] = 'seed26'
    with pytest.raises(ValueError, match='Exactly the registered'):
        verify_registration(config)


@pytest.mark.parametrize('key', ['mae', 'rmse', 'mre'])
def test_all_metrics_must_match_and_nonfinite_values_never_pass(key):
    expected = dict(mae=.8, rmse=1.2, mre=.4)
    bad = dict(expected)
    bad[key] = float('nan')
    with pytest.raises(ValueError, match='metric differs'):
        compare_metrics(bad, expected, dict(rtol=2e-6, atol=2e-6))
    bad[key] = expected[key] + .01
    with pytest.raises(ValueError, match='metric differs'):
        compare_metrics(bad, expected, dict(rtol=2e-6, atol=2e-6))


def test_existing_metric_tolerance_accepts_cpu_rounding_only():
    expected = dict(mae=.8, rmse=1.2, mre=.4)
    compare_metrics(dict(mae=.8000001, rmse=1.2000001, mre=.4000001), expected,
                    dict(rtol=2e-6, atol=2e-6))
