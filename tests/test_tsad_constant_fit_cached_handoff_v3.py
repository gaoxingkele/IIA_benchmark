import pytest
from scripts.flow_matching import run_tsad_constant_fit_cached_handoff_v3 as current

RUNTIME = 'configs/runtime/fm_tsad_constant_fit_cpu_controls.v2.json'


def test_monitor_binding_retains_original_main_and_cached_API_globals():
    main = current.rebound_main()
    assert main.__code__ is current.original.main.__code__
    assert main.__globals__['bind'] is current.original.bind
    assert main.__globals__['resource_api'] is current.cached_api
    api = current.cached_api()
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    assert api['resource_snapshot'] is resource_snapshot
    assert api['resource_slot'].__wrapped__.__globals__['resource_snapshot'] is resource_snapshot


@pytest.mark.parametrize('field,value', [('active_pid', 5), ('state','running'), ('shared_lock_owned',True)])
def test_idle_handoff_refuses_native_children_and_resource_owner(field,value):
    state = dict(active_pid=None, state='waiting_for_memory', shared_lock_owned=False)
    assert current.idle_allowed('original', state, False, [])
    assert not current.idle_allowed('original', dict(state, **{field:value}), False, [])
    assert not current.idle_allowed('original', state, False, [{'name':'python.exe','exe':'model.exe'}])


def test_original_all121_jobs_have_distinct_original_training_or_eval_identity():
    runtime = current.cold.original.read(current.ROOT / RUNTIME)
    assert len(runtime['controllers']) == 6
    assert sum(row['jobs'] for row in runtime['controllers']) == 121
    assert runtime['full_registered_corrected_GPU_slots_separate_original_gate_required'] == 170
    ids = []
    for row in runtime['controllers']:
        binding, queue = current.original.validate_queue(current.ROOT, runtime, row['original_lane'])
        assert binding == row and len(queue['jobs']) == row['jobs']
        ids += [j['id'] for j in queue['jobs']]
    assert len(ids) == len(set(ids)) == 121


def test_original_worker_logs_are_append_only_and_capped_class_remains_original():
    # Cold handoff re-enters the same controller without overwriting raw logs.
    for path in ['run_tsad_queue.py','run_moment_pretrained_queue.py']:
        source = (current.ROOT / 'scripts/flow_matching' / path).read_text(encoding='utf-8')
        assert ".open('a', encoding='utf-8')" in source
    assert current.rebound_main().__globals__['bind'].__code__ is current.original.bind.__code__
