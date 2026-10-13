import copy
import pytest
from scripts.flow_matching import run_cfm_supplement_streaming_handoff_v2 as current

CONFIG = 'configs/runtime/fm_cfm_all8_uncovered_recovery_full285_completion.v1.json'


def test_all285_routes_and_all8_retry_bindings_are_exact():
    runtime = current.supplement.original.read(current.ROOT / CONFIG)
    routes, _, records = current.namespace()['verify'](runtime)
    assert routes == runtime['routes'] and len(routes) == 285
    assert len(runtime['selected_canonical_ids']) == 8 and len(records) >= 128


def test_original_numeric_worker_code_objects_are_unchanged():
    values = current.namespace()
    for key in ['review_worker', 'run_review', 'verify_record', 'check_native_terminal']:
        assert values[key].__code__ is getattr(current.supplement.original, key).__code__
    reference = current.supplement.namespace()['controller'].__code__
    assert values['controller'].__code__.co_code == reference.co_code
    assert values['controller'].__code__.co_names == reference.co_names


@pytest.mark.parametrize('field,value', [('cpu_threads', 1), ('prediction_tolerance', {'rtol': .001, 'atol': .001})])
def test_original_numerical_guards_cannot_be_relaxed(field, value):
    runtime = copy.deepcopy(current.supplement.original.read(current.ROOT / CONFIG))
    runtime[field] = value
    with pytest.raises(ValueError, match='worker or guard changed'):
        current.namespace()['verify'](runtime)


def test_reused_closed_certificate_conflict_is_fatal():
    runtime = current.supplement.original.read(current.ROOT / CONFIG)
    primary = current.supplement.original.read(current.ROOT / current.supplement.PREVIOUS)
    records = current.supplement.original.read(current.ROOT / primary['state_root'] / 'verified_full_model_records.json')['records']
    original = records[0]
    altered = dict(original, epochs=-1)
    with pytest.raises(ValueError, match='Conflicting immutable'):
        current.adopt_primary(runtime, {original['original_canonical_id']: altered})


def test_closed_proof_digest_corruption_is_fatal(monkeypatch):
    runtime = current.supplement.original.read(current.ROOT / CONFIG)
    primary = current.supplement.original.read(current.ROOT / current.supplement.PREVIOUS)
    snapshot = current.supplement.original.read(current.ROOT / primary['state_root'] / 'verified_full_model_records.json')
    original85 = current.supplement.original.read(current.ROOT / runtime['preserved_verified_union'])['records']
    baseline = {r['original_canonical_id']: r for r in original85}
    monkeypatch.setattr(current.cold, 'streaming_sha', lambda path: '0' * 64)
    with pytest.raises(ValueError, match='proof chain required'):
        current.adopt_primary(runtime, baseline)


def test_idle_handoff_refuses_active_native_child_or_unclosed_state():
    state = dict(state='waiting_actual_original_training_completion', active_pid=None,
                 verified_original_ids=list(range(128)), failed_preserved={})
    assert current.idle_allowed('original', state, False, [])
    assert not current.idle_allowed('original', dict(state, active_pid=7), False, [])
    assert not current.idle_allowed('original', state, False, [{'name': 'python.exe', 'exe': 'worker.exe'}])
