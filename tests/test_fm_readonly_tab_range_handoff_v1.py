import copy

import pytest

from scripts.flow_matching.handoff_readonly_tab_range_queues_v1 import verify_idle_identity


def fixture():
    selected = dict(old_controller_pid=123, old_controller_create_time=456., queue_sha256='original-science-sha')
    state = dict(pid=123, config_sha256='original-science-sha', state='waiting_for_heavy_lock', active_pid=None, active_job=None)
    observed = dict(pid=123, create_time=456., command=['python', '-m', 'scripts.flow_matching.run_light_controller',
        '--runtime-config', 'original-runtime.json', '--controller', 'range_main'], children=[])
    return selected, state, observed


def test_only_exact_original_idle_handle_can_be_retired():
    selected, state, observed = fixture()
    before = copy.deepcopy((selected, state, observed))
    assert verify_idle_identity('range_main', selected, state, observed, 'original-runtime.json')['verified_idle']
    assert (selected, state, observed) == before


@pytest.mark.parametrize('case', ['reused_pid', 'different_command', 'active_pid', 'active_job', 'evaluator_child', 'not_waiting', 'changed_science'])
def test_active_worker_changed_identity_or_wrong_science_blocks_handoff(case):
    selected, state, observed = fixture()
    if case == 'reused_pid':
        observed['create_time'] += 1
    elif case == 'different_command':
        observed['command'][-1] = 'different_controller'
    elif case == 'active_pid':
        state['active_pid'] = 999
    elif case == 'active_job':
        state['active_job'] = 'full-experiment'
    elif case == 'evaluator_child':
        observed['children'] = [dict(name='python.exe', pid=999)]
    elif case == 'not_waiting':
        state['state'] = 'evaluating'
    else:
        state['config_sha256'] = 'changed-science'
    with pytest.raises(ValueError, match='exact owned idle'):
        verify_idle_identity('range_main', selected, state, observed, 'original-runtime.json')
