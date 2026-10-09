from types import SimpleNamespace
from scripts.flow_matching.transition_cpu_controllers_at_boundary_v1 import boundary_observation


def parent(children=(), command=None, ctime=1):
    return SimpleNamespace(pid=12, create_time=lambda: ctime, cmdline=lambda: command or ['old'],
        children=lambda recursive=True: children)


EXPECTED = {'pid': 12, 'create_time': 1, 'command': ['old'], 'job_ids': {'full'}}


def test_only_completed_exited_model_is_a_boundary():
    state = {'pid': 12, 'active_pid': 99, 'active_job': 'full'}
    assert boundary_observation(parent(), state, EXPECTED, lambda _: True, pid_exists=lambda _: True) is None
    assert boundary_observation(parent(), state, EXPECTED, lambda _: False, pid_exists=lambda _: False) is None
    assert boundary_observation(parent(), state, EXPECTED, lambda _: True, pid_exists=lambda _: False) == 'full'


def test_reused_pid_new_child_or_new_command_defers():
    state = {'pid': 12, 'active_job': 'full'}
    for process in [parent(ctime=2), parent(command=['new']), parent(children=['model'])]:
        assert boundary_observation(process, state, EXPECTED, lambda _: True,
            pid_exists=lambda _: False, helper_check=lambda _: False) is None
    assert boundary_observation(parent(), {'pid': 13}, EXPECTED, lambda _: True) is None
    assert boundary_observation(parent(), {'pid': 12}, EXPECTED, lambda _: True) == 'verified_idle_boundary'
