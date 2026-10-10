import copy
from types import SimpleNamespace

import pytest

from scripts.flow_matching.transition_giflow_at_boundary_v1 import candidate, compare_full_jobs


class Parent:
    pid = 21
    def create_time(self): return 123.5
    def cmdline(self): return ['python', '-m', 'native_queue', '--queue', 'frozen.json']
    def children(self, recursive=False): return self.child_list
    child_list = []


def test_boundary_never_stops_a_live_model_or_unreported_child():
    parent = Parent()
    expected = dict(pid=21, create_time=123.5, command=parent.cmdline())
    state = dict(pid=21, active_pid=22)
    assert not candidate(parent, state, expected, pid_exists=lambda p:True)
    parent.child_list = [SimpleNamespace(pid=23)]
    assert not candidate(parent, dict(pid=21, active_pid=None), expected, helper_check=lambda c:False)
    parent.child_list = []
    assert candidate(parent, dict(pid=21, active_pid=None), expected)
    assert not candidate(parent, dict(pid=21, active_pid=None), dict(expected, create_time=124))


def test_handoff_preserves_all_full_seeds_algorithms_and_budgets():
    job = dict(id='seed393', dataset='air36', track='released_mirror', recipe='main', seed=393,
               arguments=dict(batch_size=128, training_epoch=300), author_source='original',
               model_config='old', output_directory='old')
    old = dict(settings=dict(emergency=4), resource_lock='gpu', jobs=[job])
    new = copy.deepcopy(old)
    new['jobs'][0].update(model_config='new', output_directory='new_attempt')
    compare_full_jobs(old, new)
    new['jobs'][0]['arguments']['training_epoch'] = 1
    with pytest.raises(ValueError): compare_full_jobs(old, new)
    new = copy.deepcopy(old); new['jobs'] = []
    with pytest.raises(ValueError): compare_full_jobs(old, new)


def test_successor_binding_keeps_original_gpu_guard_and_corrected_worker():
    from scripts.flow_matching.run_giflow_serial_native_queue_v4 import bound_main
    function = bound_main()
    assert 'scripts.flow_matching.run_giflow_native_job_v2' in function.__code__.co_consts
    assert 'resource_slot' in function.__code__.co_consts
    assert 'guarded_gpu_wait' in function.__code__.co_names
    assert 'sleep' in function.__code__.co_names
