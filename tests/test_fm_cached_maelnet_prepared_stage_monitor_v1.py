import copy

import pytest

from scripts.flow_matching import run_cached_maelnet_prepared_stage_monitor_v1 as monitor


def config(role):
    return monitor.base.read(monitor.base.ROOT/('configs/runtime/fm_maelnet_'+role+'_cached_prepared_stage_monitor.v1.json'))


@pytest.mark.parametrize('role',['original150','seed1105','seed1106'])
def test_full150_job_registry_and_each_original_stage_guard_and_selection_unchanged(role):
    value=config(role); module,queue=monitor.verify(value)
    original=monitor.base.read(monitor.base.ROOT/value['original_runtime_configuration'])
    assert len(queue['jobs'])==150
    assert value['settings']==original['settings']
    assert value['settings']['minimum_commit_headroom_bytes']==32*2**30
    assert value['settings']['emergency_commit_headroom_bytes']==16*2**30
    assert value['resource_lock']==original['resource_lock']
    assert monitor.pending_unstarted(value)


@pytest.mark.parametrize('role',['original150','seed1105','seed1106'])
@pytest.mark.parametrize('mutation',['native_pid','native_child','clock','command','queue'])
def test_exact_idle_identity_required_no_native_child_or_changed_request_allowed(role,mutation):
    value=config(role); old=value['handoff']['previous_owner']
    state=dict(value['expected_state_identity'],pid=old['pid'],create_time=old['create_time'],active_pid=None,
        state='waiting_for_memory',requested_job=value['pending_job']['id'],requested_command=value['prepared_stage']['requested_command'])
    children=[]; assert monitor.safe_phase(value,state,children)
    if mutation=='native_pid': state['active_pid']=777
    elif mutation=='native_child': children=[dict(name='python.exe',exe='F:/native/python.exe')]
    elif mutation=='clock': state['create_time']+=1
    elif mutation=='command': state['requested_command']=state['requested_command']+['--train_epochs','1']
    else: state['queue_sha256']='0'*64
    assert not monitor.safe_phase(value,state,children)


@pytest.mark.parametrize('mutation',['guard','selection','job_seed','scope'])
def test_no_scientific_or_guard_change_permitted(mutation):
    value=copy.deepcopy(config('original150'))
    if mutation=='guard': value['settings']['minimum_commit_headroom_bytes']-=1
    elif mutation=='selection': value['original_arguments']+=['--job-id','other']
    elif mutation=='job_seed': value['pending_job']['seed']+=1
    else: value['original_scope']['papers']-=1
    with pytest.raises(ValueError): monitor.verify(value)


@pytest.mark.parametrize('artifact',['nonempty_log','stage_receipt','workspace_checkpoint','extra_workspace_file'])
def test_only_exact_empty_log_stage_and_unchanged_copied_workspace_can_be_carried(tmp_path,artifact):
    value=copy.deepcopy(config('original150')); output=tmp_path/'output'; stage=output/'stages'/'prepared'
    stage.mkdir(parents=True); (stage/'stdout.log').write_bytes(b''); (stage/'stderr.log').write_bytes(b'')
    workspace=output/'author_workspace'; workspace.mkdir(); source=workspace/'source.py'; source.write_bytes(b'original')
    value['pending_job']['output_directory']=str(output); value['prepared_stage']['path']=str(stage)
    value['workspace_snapshot']=[dict(relative_path='source.py',sha256=monitor.base.sha(source))]
    assert monitor.pending_unstarted(value)
    if artifact=='nonempty_log': (stage/'stderr.log').write_bytes(b'actual failed execution')
    elif artifact=='stage_receipt': (stage/'receipt.json').write_text('{}')
    elif artifact=='workspace_checkpoint': (workspace/'checkpoint.pth').write_bytes(b'actual model')
    else: (workspace/'new.txt').write_text('new content')
    assert not monitor.pending_unstarted(value)


def test_same_suspend_recheck_handoff_and_original_full_monitor_main_code_objects():
    namespace=monitor.runtime_namespace()
    assert namespace['main'].__code__ is monitor.base.main.__code__
    assert namespace['_original_safe_handoff'].__code__ is monitor.base.safe_handoff.__code__
    assert namespace['safe_phase'] is monitor.safe_phase
