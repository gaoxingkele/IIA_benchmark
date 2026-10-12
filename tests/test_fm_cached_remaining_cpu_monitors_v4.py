"""Full CPU job/cap preservation and prepared-but-unstarted handoff boundaries."""
import copy

import pytest

from scripts.flow_matching import run_cached_remaining_cpu_monitors_v4 as monitor


def config(role):
    return monitor.base.read(monitor.base.ROOT/('configs/runtime/fm_'+role+'_cached_resource_monitor.v4.json'))


@pytest.mark.parametrize('role,count',[('entropic_cpu',105),('pi_cpu',1140),('moment_cpu',72)])
def test_all_original_jobs_and32GiB_admission_emergency_and_worker_caps_are_retained(role,count):
    value=config(role);module,queue=monitor.verify(value)
    assert len(queue['jobs'])==count
    assert value['settings']['minimum_commit_headroom_bytes']==32*1024**3
    assert value['settings']['emergency_commit_headroom_bytes']==16*1024**3
    assert value['settings']['maximum_worker_rss_bytes']==10*1024**3
    assert value['settings']['maximum_worker_private_bytes']==12*1024**3


@pytest.mark.parametrize('role',['entropic_cpu','pi_cpu','moment_cpu'])
@pytest.mark.parametrize('mutation',['native_pid','native_child','clock','wrong_job','queue'])
def test_only_exact_unstarted_idle_model_request_can_be_handed_off(role,mutation):
    value=config(role);expected=value['handoff']['previous_owner']
    state=dict(value['expected_state_identity'],pid=expected['pid'],create_time=expected['create_time'],
        active_pid=None,state='waiting_for_memory',requested_job=value['pending_job']['id'])
    children=[]
    assert monitor.safe_phase(value,state,children)
    if mutation=='native_pid':state['active_pid']=123
    elif mutation=='native_child':children=[dict(name='python.exe',exe='F:/native/python.exe')]
    elif mutation=='clock':state['create_time']+=1
    elif mutation=='wrong_job':state['requested_job']='other'
    else:state['queue_sha256']='0'*64
    assert not monitor.safe_phase(value,state,children)


@pytest.mark.parametrize('role',['entropic_cpu','pi_cpu','moment_cpu'])
@pytest.mark.parametrize('mutation',['guard','cap','selection','job_budget','scope'])
def test_every_scientific_selection_and_existing_resource_contract_is_immutable(role,mutation):
    value=copy.deepcopy(config(role))
    if mutation=='guard':value['settings']['minimum_commit_headroom_bytes']-=1
    elif mutation=='cap':value['settings']['maximum_worker_private_bytes']+=1
    elif mutation=='selection':value['original_arguments'].append('--probe-only')
    elif mutation=='job_budget':value['pending_job']['seed']+=1
    else:value['original_scope']['method_baseline_ablation_records']-=1
    with pytest.raises(ValueError):monitor.verify(value)


@pytest.mark.parametrize('artifact',['checkpoints','failure.json','nonempty_log'])
def test_Pi_prepared_log_exception_never_admits_model_artifacts_or_prior_nonempty_attempt(tmp_path,artifact):
    value=copy.deepcopy(config('pi_cpu'))
    value['pending_job']['output_directory']=str(tmp_path)
    (tmp_path/'stdout.log').write_bytes(b'');(tmp_path/'stderr.log').write_bytes(b'')
    assert monitor.pending_unstarted(value)
    if artifact=='checkpoints':(tmp_path/'checkpoints').mkdir()
    elif artifact=='failure.json':(tmp_path/'failure.json').write_text('{}')
    else:(tmp_path/'stderr.log').write_text('previous attempt')
    assert not monitor.pending_unstarted(value)


def test_same_atomic_suspend_recheck_no_native_child_handoff_and_full_dispatch_main_used():
    namespace=monitor.runtime_namespace()
    assert namespace['safe_handoff'].__code__ is monitor.base.safe_handoff.__code__
    assert namespace['main'].__code__ is monitor.base.main.__code__
    assert namespace['safe_handoff'].__globals__['safe_phase'] is monitor.safe_phase
