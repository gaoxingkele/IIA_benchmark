"""Idle handoff cannot stop native children or accept unproven legacy identities."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.flow_matching import run_cached_resource_monitor_controller_v2 as monitor


def config(role='cfm'):
    return monitor.read(monitor.ROOT / ('configs/runtime/fm_' + role + '_cached_resource_monitor.v2.json'))


@pytest.mark.parametrize('role,count', [('cfm',285),('saits',210)])
def test_entire_original_registry_and_unchanged_native_policy_are_bound(role,count):
    module,queue=monitor.verify(config(role))
    assert len(queue['jobs'])==count
    assert module.__name__==config(role)['entry_module']


@pytest.mark.parametrize('role', ['cfm','saits'])
@pytest.mark.parametrize('mutation', ['native_pid','native_child','wrong_pid','wrong_queue','wrong_clock','wrong_phase'])
def test_any_native_child_or_changed_owner_binding_blocks_handoff(role,mutation):
    value=config(role)
    expected=value['handoff']['previous_owner']
    state=dict(value['expected_state_identity'],pid=expected['pid'],create_time=expected['create_time'],
        active_pid=None,state='waiting_for_memory')
    children=[]
    assert monitor.safe_phase(value,state,children)
    if mutation=='native_pid': state['active_pid']=123
    elif mutation=='native_child': children=[dict(name='python.exe',exe='F:/native/python.exe')]
    elif mutation=='wrong_pid': state['pid']+=1
    elif mutation=='wrong_queue': state['queue_sha256']='0'*64
    elif mutation=='wrong_clock': state['create_time']+=1
    else: state['state']='running'
    assert not monitor.safe_phase(value,state,children)


def test_legacy_state_requires_exact_independent_historical_ctime_module_and_hashes():
    value=config()
    state=dict(value['expected_state_identity'],pid=value['handoff']['previous_owner']['pid'],
        active_pid=None,state='waiting_for_memory')
    assert monitor.safe_phase(value,state,[])
    broken=copy.deepcopy(value);broken['historical_identity']=None
    assert not monitor.safe_phase(broken,state,[])
    broken=copy.deepcopy(value);broken['handoff']['previous_owner']['create_time']+=1
    with pytest.raises(ValueError): monitor.safe_phase(broken,state,[])
    broken=copy.deepcopy(value);broken['historical_identity']['sha256']='0'*64
    with pytest.raises(ValueError): monitor.safe_phase(broken,state,[])


def test_status_identity_injection_never_changes_nonstatus_artifacts(tmp_path):
    calls=[]
    def write(path,data): calls.append((path,data))
    module=SimpleNamespace(write=write)
    target=tmp_path/'status.json'
    value=dict(handoff=dict(previous_state_path=str(target)),runtime_configuration='monitor.json')
    function=monitor.install_status_identity(module,value,42.25)
    original=dict(pid=os.getpid(),jobs={'full':'completed'},active_pid=None)
    function(target,original)
    assert calls[-1][1]==dict(original,create_time=42.25,resource_monitor_configuration='monitor.json')
    function(tmp_path/'result.json',original)
    assert calls[-1][1] is original
    function(target,dict(pid=123,optimizer_updates=50000))
    assert calls[-1][1]==dict(pid=123,optimizer_updates=50000)


@pytest.mark.parametrize('role', ['cfm','saits'])
@pytest.mark.parametrize('mutation', ['guard','scope','selection'])
def test_changed_full_scope_selection_or_resource_limits_are_rejected(role,mutation):
    value=copy.deepcopy(config(role))
    if mutation=='guard': value['settings']['minimum_commit_headroom_bytes']-=1
    elif mutation=='scope': value['original_scope']['method_baseline_ablation_records']=680
    else: value['original_arguments'].append('--probe-only')
    with pytest.raises(ValueError): monitor.verify(value)
