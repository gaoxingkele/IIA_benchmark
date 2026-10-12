import copy

import pytest

from scripts.flow_matching import run_cached_idle_spectral_gpu_monitors_v1 as monitor


def config(role):
    return monitor.base.read(monitor.base.ROOT/('configs/runtime/fm_'+role+'_cached_idle_gpu_monitor.v1.json'))


@pytest.mark.parametrize('role,count',[('ls4_fred_nn5',20),('ls4_solar_temperature',20),('ls4_ushcn',100),('sashimi',40)])
def test_complete_original_jobs_all_CPU_GPU_memory_gates_and_prior_diagnostics_preserved(role,count):
    value=config(role); module,queue=monitor.verify(value)
    assert len(queue['jobs'])==count
    assert value['settings']==queue['settings'] and value['resource_lock']==queue['resource_lock']
    assert value['settings']['controller_yield_seconds']==35
    assert value['settings']['minimum_gpu_free_bytes']==14*2**30
    assert value['settings']['emergency_gpu_free_bytes']==4*2**30
    assert monitor.pending_unstarted(value)


@pytest.mark.parametrize('role',list(monitor.ROLES))
@pytest.mark.parametrize('mutation',['native_pid','native_child','clock','queue','stage'])
def test_exact_idle_owner_required_never_stop_native_child_or_changed_stage(role,mutation):
    value=config(role); old=value['handoff']['previous_owner']
    state=dict(pid=old['pid'],create_time=old['create_time'],active_pid=None,active_job=None,
        state='waiting_full_gpu_resources',queue_sha256=value['expected_state_identity']['queue_sha256'])
    children=[]; assert monitor.safe_phase(value,state,children)
    if mutation=='native_pid':state['active_pid']=123
    elif mutation=='native_child':children=[dict(name='python.exe',exe='F:/native/python.exe')]
    elif mutation=='clock':state['create_time']+=1
    elif mutation=='queue':state['queue_sha256']='0'*64
    else:state['active_job']='actual_training'
    assert not monitor.safe_phase(value,state,children)


@pytest.mark.parametrize('mutation',['guard','selection','scope','GPU'])
def test_full_scientific_selection_and_safeguards_cannot_change(mutation):
    value=copy.deepcopy(config('ls4_fred_nn5'))
    if mutation=='guard':value['settings']['minimum_commit_headroom_bytes']-=1
    elif mutation=='selection':value['original_arguments']+=['--job-id','subset']
    elif mutation=='scope':value['original_scope']['original_data_task_records']-=1
    else:value['settings']['minimum_gpu_free_bytes']-=1
    with pytest.raises(ValueError):monitor.verify(value)


def test_same_atomic_handoff_and_original_full_monitor_main_used():
    namespace=monitor.runtime_namespace()
    assert namespace['safe_handoff'].__code__ is monitor.base.safe_handoff.__code__
    assert namespace['main'].__code__ is monitor.base.main.__code__


def test_new_pending_preflight_artifact_or_nonempty_log_never_permits_handoff(tmp_path):
    value=copy.deepcopy(config('ls4_fred_nn5'))
    logs=tmp_path/'logs';logs.mkdir(); pending=tmp_path/'gpu_preflight'
    value['original_log_directory']=str(logs);value['completed_log_snapshot']={}
    value['pending_paths_that_must_remain_absent']=[str(pending)]
    assert monitor.pending_unstarted(value)
    pending.mkdir();assert not monitor.pending_unstarted(value)
    pending.rmdir();(logs/'gpu_preflight.stdout.log').write_text('actual source attempt')
    assert not monitor.pending_unstarted(value)
