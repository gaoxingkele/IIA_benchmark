"""Exact full GRASP registry and idle-only cached telemetry dispatch."""
import copy

import pytest

from scripts.flow_matching import run_cached_grasp_monitor_v3 as monitor


def config():
    return monitor.base.read(monitor.base.ROOT/'configs/runtime/fm_grasp_cached_resource_monitor.v3.json')


def test_all9120_original_jobs_and480_cohorts_with_identical_native_settings_are_retained():
    module,queue=monitor.verify(config())
    assert len(queue['jobs'])==9120 and queue['cohort_count']==480
    assert queue['settings']==config()['settings']
    assert {j['seed'] for j in queue['jobs']}==set(range(1103,1113))
    assert all(j['expected_optimizer_updates']>0 for j in queue['jobs'])


def test_same_atomic_idle_handoff_and_cached_host_dispatch_code_are_used():
    namespace=monitor.runtime_namespace()
    assert namespace['safe_handoff'].__code__ is monitor.base.safe_handoff.__code__
    assert namespace['main'].__code__ is monitor.base.main.__code__
    assert namespace['safe_handoff'].__globals__['safe_phase'] is monitor.safe_phase


@pytest.mark.parametrize('mutation',['native_pid','native_child','wrong_clock','wrong_queue','wrong_phase'])
def test_native_training_or_changed_source_owner_blocks_handoff(mutation):
    value=config();expected=value['handoff']['previous_owner']
    state=dict(pid=expected['pid'],create_time=expected['create_time'],active_pid=None,
        state='waiting_full_gpu_resources',queue_sha256=value['expected_state_identity']['queue_sha256'])
    children=[]
    assert monitor.safe_phase(value,state,children)
    if mutation=='native_pid':state['active_pid']=123
    elif mutation=='native_child':children=[dict(name='python.exe',exe='F:/native/python.exe')]
    elif mutation=='wrong_clock':state['create_time']+=1
    elif mutation=='wrong_queue':state['queue_sha256']='0'*64
    else:state['state']='running'
    assert not monitor.safe_phase(value,state,children)


@pytest.mark.parametrize('mutation',['guard','scope','command','history'])
def test_caps_full_scope_and_independent_exact_history_cannot_change(mutation):
    value=copy.deepcopy(config())
    if mutation=='guard':value['settings']['minimum_commit_headroom_bytes']-=1
    elif mutation=='scope':value['original_scope']['original_data_task_records']=272
    elif mutation=='command':value['handoff']['previous_owner']['command'][-1]='other.json'
    else:value['historical_identity']['sha256']='0'*64
    with pytest.raises(ValueError):monitor.verify(value)
