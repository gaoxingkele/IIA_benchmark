from copy import deepcopy
from types import FunctionType
import subprocess
import sys
import pytest
from scripts.flow_matching import continue_TAB878_fair_entities_v2 as s


def test_full_controller_and_all_numerical_stages_keep_original_code_objects():
    c=s.registration();old=s.previous.registration();ns=s.namespace()
    for key in ['required_cases','required_label_views','settings','resource_lock','cpu_threads','inter_entity_yield_seconds',
        'maximum_new_attempts_per_fragment','scientific_artifact_root','original_scope']:
        assert c[key]==old[key]
    assert ns['write'] is s.atomic and ns['stage'] is s.previous.stage
    assert ns['handoff'] is s.previous.handoff and ns['fragment_action'] is s.previous.fragment_action
    assert FunctionType(s.previous.controller.__code__,ns).__code__ is s.previous.controller.__code__
    assert not c['prior_metadata_failure']['handoff_performed']
    assert not c['prior_metadata_failure']['numerical_child_stopped']


@pytest.mark.parametrize('fault',['writer','scope','yield'])
def test_only_the_exact_atomic_metadata_correction_is_allowed(fault):
    c=s.registration()
    if fault=='writer':c['atomic_mutable_status_writer']='other'
    elif fault=='scope':c['required_cases']=1
    else:c['inter_entity_yield_seconds']=0
    with pytest.raises(ValueError):s.verify(c)


class ControlledWaitingComplete(RuntimeError):pass


@pytest.mark.parametrize('writer',['old_exclusive','atomic'])
def test_real_full_controller_updates_waiting_status_repeatedly_before_any_handoff(tmp_path,monkeypatch,writer):
    from scripts.flow_matching import run_anomaly_transformer_cold_resource_entry_v1 as facade
    ns=s.namespace();events=[]
    class Lock:
        def close(self):events.append('metadata_lock_closed')
    monkeypatch.setattr(facade,'install_facades',lambda:dict(exclusive_lock=lambda path:Lock()))
    config=s.registration();config['state_root']=str(tmp_path/'metadata_only_no_scientific_guards')
    source=s.read(s.ROOT/s.previous.original.CONFIG)
    # Exercise the actual entire controller and its actual nested status updater.
    # Only handoff is replaced: no OS process is suspended/stopped in this test.
    def wait(config,c,api,base,update):
        update(state='waiting_actual_metric_lock_release_no_process_stopped',unit_heartbeat=1)
        update(state='waiting_actual_metric_lock_release_no_process_stopped',unit_heartbeat=2)
        raise ControlledWaitingComplete()
    ns['handoff']=wait
    ns['sha']=lambda path:'0'*64 if path==s.ROOT/s.CONFIG else s.sha(path)
    if writer=='old_exclusive':ns['write']=s.previous.write
    body=FunctionType(s.previous.controller.__code__,ns)
    if writer=='old_exclusive':
        with pytest.raises(FileExistsError):body(config,source)
    else:
        with pytest.raises(ControlledWaitingComplete):body(config,source)
        state=s.read(tmp_path/'metadata_only_no_scientific_guards/status.json')
        assert state['unit_heartbeat']==2 and state['active_pid'] is None
        assert state['required_cases']==878 and state['required_label_views']==2057
    assert events==['metadata_lock_closed']
    assert not list(tmp_path.rglob('guard_receipt.json'))


def test_atomic_corrected_full_registration_is_cold():
    code='from scripts.flow_matching import continue_TAB878_fair_entities_v2 as s;import sys;s.registration();assert "numpy" not in sys.modules and "torch" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
