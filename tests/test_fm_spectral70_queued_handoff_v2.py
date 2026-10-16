import ast
import copy
import json
from pathlib import Path
import sys
import time

import pytest
from scripts.flow_matching import continue_spectral70_queued_handoff_v2 as s
from scripts.flow_matching.win32_queued_byte_lock_v1 import QueuedByteLock
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api


@pytest.mark.skipif(sys.platform != 'win32', reason='Actual Windows lock interoperability required')
def test_real_pending_request_blocks_on_legacy_owner_then_excludes_reacquisition(tmp_path):
    path = tmp_path/'actual_byte0.lock'
    legacy = resource_api()['exclusive_lock']
    owner = legacy(path)
    assert owner is not None
    request = QueuedByteLock(path)
    try:
        assert not request.poll()
        owner.close()
        # Windows has a waiting exclusive request, not a new polling attempt.
        contender = legacy(path)
        if contender is not None:
            contender.close()
        assert contender is None
        until = time.monotonic()+2
        while not request.poll() and time.monotonic() < until:
            time.sleep(.01)
        assert request.poll()
        assert legacy(path) is None
    finally:
        owner.close()
        request.close()
    after = legacy(path)
    assert after is not None
    after.close()


@pytest.mark.skipif(sys.platform != 'win32', reason='Actual Windows lock cancellation required')
def test_cancelling_pending_request_does_not_release_the_original_owner(tmp_path):
    path = tmp_path/'cancel.lock'
    legacy = resource_api()['exclusive_lock']
    owner = legacy(path)
    request = QueuedByteLock(path)
    try:
        assert not request.poll()
        request.close()
        request.close()
        assert legacy(path) is None
        with pytest.raises(RuntimeError, match='closed'):
            request.poll()
    finally:
        owner.close()
        request.close()
    after = legacy(path)
    assert after is not None
    after.close()


@pytest.mark.skipif(sys.platform != 'win32', reason='Actual Windows byte lock required')
def test_immediate_free_lock_preserves_file_bytes_and_excludes_legacy(tmp_path):
    path = tmp_path/'existing.lock'
    payload = b'preserve all existing metadata bytes'
    path.write_bytes(payload)
    with QueuedByteLock(path) as request:
        assert request.poll()
        legacy = resource_api()['exclusive_lock']
        assert legacy(path) is None
    # The old helper appends one byte on each attempted acquisition. The new
    # request itself must preserve the existing content exactly.
    assert path.read_bytes() == payload+b'1'


def test_all70_original_workers_resources_and_exact_observer_are_bound():
    config = s.registration()
    original, parent, cases = s.verify(config)
    assert len(cases) == config['required_conditions'] == 70
    assert config['required_cohorts'] == 14
    assert config['settings'] == original['settings']
    assert config['resource_lock'] == original['resource_lock']
    assert config['spectral_flow_minimum_commit_headroom_bytes'] == 48*2**30
    assert config['inter_condition_yield_seconds'] == 35
    assert config['lock_offset'] == 0 and config['lock_bytes'] == 1
    assert config['previous_owner'] == s.previous.EXPECTED
    assert config['previous_observer'] == s.EXPECTED_OBSERVER
    changed = copy.deepcopy(config); changed['settings']['minimum_commit_headroom_bytes'] = 0
    with pytest.raises(ValueError, match='All70'):
        s.verify(changed)


def test_observer_gate_rejects_any_numeric_descendant_state_or_identity_change():
    h = s.EXPECTED_OBSERVER
    state = dict(pid=h['pid'], create_time=h['create_time'], active_pid=None,
        required_conditions=70, required_cohorts=14,
        state='waiting_actual_metric_lock_release_no_process_stopped')
    assert s.observer_allowed(state, h, [])
    assert not s.observer_allowed(state, h, [dict(name='python.exe', exe='python.exe')])
    for key, value in [('pid', 1), ('create_time', 0), ('active_pid', 42),
                       ('required_conditions', 5), ('required_cohorts', 1), ('state', 'running')]:
        assert not s.observer_allowed(dict(state, **{key:value}), h, [])


def test_private_continuation_has_same_numerical_AST_and_v2_metadata_binding(monkeypatch, tmp_path):
    config = dict(state_root='unchanged')
    old = dict(verified_ids=['done'], failed_preserved={'failed':'preserved'})
    reports = {'done':{'complete':True}}
    before, tree1 = s.previous.controller_program(config, old, reports, None, tmp_path)
    after, tree2 = s.continuation_program(config, old, reports, None, tmp_path)
    assert ast.dump(tree1) == ast.dump(tree2)
    assert s.previous.CONFIG != s.CONFIG
    initial = after.__globals__['_initial'](dict(new_generator_optimizer_updates=0,new_training_seed_slots=0))
    assert initial['scheduler_configuration'] == s.CONFIG
    assert initial['verified_ids'] == ['done'] and initial['failed_preserved'] == old['failed_preserved']
    assert initial['new_generator_optimizer_updates'] == initial['new_training_seed_slots'] == 0
    source = Path(s.previous.original.__file__).read_text(encoding='utf-8')
    original = next(x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef) and x.name=='controller')
    def calls(tree):
        return [ast.dump(x) for x in ast.walk(tree) if isinstance(x, ast.Call) and (
            isinstance(x.func, ast.Attribute) and x.func.attr in ['Popen','exclusive'] or
            isinstance(x.func, ast.Subscript) and isinstance(x.func.slice, ast.Constant)
                and x.func.slice.value == 'guarded_gpu_wait')]
    assert calls(original) == calls(tree2)


def test_queue_does_not_request_a_different_range_or_fail_immediately():
    source = Path(sys.modules[QueuedByteLock.__module__].__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    lock_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr=='LockFileEx']
    assert len(lock_calls) == 1
    args = lock_calls[0].args
    assert [ast.literal_eval(x) for x in args[1:5]] == [2, 0, 1, 0]


def fake_handoff(monkeypatch, tmp_path, bad=None):
    """Exercise ordering with fake metadata actors; never touch actual jobs."""
    events = []
    original = tmp_path/'original'; original.mkdir()
    observer = tmp_path/'observer'; observer.mkdir()
    base = tmp_path/'new'; base.mkdir()
    (original/'attempts').mkdir()
    old = dict(pid=1, create_time=1, required_conditions=70, required_cohorts=14,
               active_pid=None, state='scanning_all70_full_source_proofs',
               verified_ids=[], failed_preserved={})
    obs = dict(pid=2, create_time=2, required_conditions=70, required_cohorts=14,
               active_pid=None, state='waiting_actual_metric_lock_release_no_process_stopped')
    def save(path, data):
        path.write_text(json.dumps(data), encoding='utf-8')
    save(original/'status.json', old); save(observer/'status.json', obs)
    (tmp_path/s.previous.CONFIG).parent.mkdir(parents=True)
    save(tmp_path/s.previous.CONFIG, dict(state_root='observer'))
    config = dict(previous_owner=dict(pid=1,create_time=1,command=['metadata_owner']),
        previous_observer=dict(pid=2,create_time=2,command=['idle_observer']),
        resource_lock='shared.lock', pending_lock_poll_seconds=0)
    class Actor:
        def __init__(self, pid): self.pid=pid; self.running=True
        def suspend(self): events.append(('suspend',self.pid))
        def resume(self): events.append(('resume',self.pid))
        def terminate(self): events.append(('terminate',self.pid)); self.running=False
        def wait(self, timeout): events.append(('wait',self.pid))
        def is_running(self): return self.running
    actors={p:Actor(p) for p in [1,2]}
    calls={1:0,2:0}
    def exact(expected):
        pid=expected['pid'];calls[pid]+=1;events.append(('identity',pid))
        if bad=='second_identity' and calls[pid]==2:
            raise ValueError('Identity changed after suspension')
        return actors[pid]
    class Kernel:
        def __init__(self, path): self.acquired=True;events.append(('kernel_owned',0))
        def poll(self): return True
        def close(self): events.append(('kernel_release',0))
    class Lane:
        def close(self): events.append(('lane_close',0))
    def restored(c,parent,cases,state):
        assert not any(e[0]=='terminate' for e in events)
        assert ('suspend',1) in events and ('suspend',2) in events
        events.append(('all_old_reports_verified',0))
        return {},{}
    def update(**kw):
        if kw.get('state')=='boundary_not_proven_original_processes_preserved':
            raise ValueError('Unproven boundary safely preserved')
    monkeypatch.setattr(s,'ROOT',tmp_path)
    monkeypatch.setattr(s,'QueuedByteLock',Kernel)
    monkeypatch.setattr(s,'exact_process',exact)
    monkeypatch.setattr(s,'descendants',lambda p: ([dict(pid=3,name='python.exe',exe='python.exe')]
        if bad=='numeric_child' and p.pid==1 else []))
    monkeypatch.setattr(s.previous,'restored_reports',restored)
    if bad=='partial_attempt': (original/'attempts/unknown_incomplete').mkdir()
    if bad=='nonterminal_child': monkeypatch.setattr(s.previous,'terminal_child',lambda c,state:False)
    return lambda:s.handoff(config,dict(state_root='original'),{}, {},
        dict(exclusive_lock=lambda path:Lane()),base,update), events, base


def test_handoff_requires_lock_twice_checked_idle_actors_and_rehashes_before_retirement(monkeypatch,tmp_path):
    run,events,base=fake_handoff(monkeypatch,tmp_path)
    result=run()
    assert result is not None
    assert events.count(('identity',1))==events.count(('identity',2))==2
    verified=events.index(('all_old_reports_verified',0))
    assert events.index(('kernel_owned',0)) < events.index(('suspend',1)) < verified
    assert verified < events.index(('terminate',2)) < events.index(('terminate',1))
    assert events.index(('terminate',1)) < events.index(('kernel_release',0))
    assert not any(e[0]=='resume' for e in events)
    assert (base/'handoff.json').exists()
    assert json.loads((base/'handoff_identity.json').read_text('utf-8'))['numerical_child_stopped'] is False


@pytest.mark.parametrize('bad',['numeric_child','partial_attempt','nonterminal_child'])
def test_unsafe_boundary_never_suspends_or_terminates_any_actor(monkeypatch,tmp_path,bad):
    run,events,base=fake_handoff(monkeypatch,tmp_path,bad)
    with pytest.raises(ValueError,match='safely preserved'):run()
    assert not any(e[0] in ['suspend','terminate'] for e in events)
    assert ('kernel_release',0) in events and not (base/'handoff.json').exists()


def test_identity_drift_after_suspension_resumes_both_actors_without_retirement(monkeypatch,tmp_path):
    run,events,base=fake_handoff(monkeypatch,tmp_path,'second_identity')
    with pytest.raises(ValueError,match='Identity changed'):run()
    assert ('resume',1) in events and ('resume',2) in events
    assert not any(e[0]=='terminate' for e in events)
    assert ('kernel_release',0) in events and not (base/'handoff.json').exists()
