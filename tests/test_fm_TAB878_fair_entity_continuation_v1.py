"""Scheduling and actual saved-proof checks, never benchmark results."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import pytest
from scripts.flow_matching import continue_TAB878_fair_entities_v1 as s


def test_complete_original878_2057_numerics_resources_and_yield_are_retained():
    c=s.registration();old=s.verify(c)
    assert c['required_cases']==len(old['entries'])==878 and c['required_label_views']==2057
    assert c['settings']==old['settings'] and c['resource_lock']==old['resource_lock']
    assert c['scientific_artifact_root']==old['artifact_root'] and c['cpu_threads']==old['CPU_threads']==2
    assert c['inter_entity_yield_seconds']==20 and c['maximum_new_attempts_per_fragment']==1
    assert old['metric_calls_per_entity_label_view']==201 and len(old['score_metrics'])==6 and len(old['label_metrics'])==15


@pytest.mark.parametrize('fault',['scope','labels','yield','emergency','cap','lock','retry'])
def test_any_scope_numerical_policy_or_guard_weakening_rejected(fault):
    c=s.registration()
    if fault=='scope':c['required_cases']=1
    elif fault=='labels':c['required_label_views']=1
    elif fault=='yield':c['inter_entity_yield_seconds']=0
    elif fault=='emergency':c['settings']['emergency_commit_headroom_bytes']=1
    elif fault=='cap':c['settings']['maximum_worker_private_bytes']*=2
    elif fault=='lock':c['resource_lock']='other.lock'
    else:c['maximum_new_attempts_per_fragment']=2
    with pytest.raises(ValueError):s.verify(c)


def state():
    return dict(pid=s.EXPECTED['pid'],create_time=s.EXPECTED['create_time'],required_cases=878,required_label_views=2057,
        active_pid=None,state='between_original_entities')


@pytest.mark.parametrize('fault',['pid','time','scope','labels','native_child','not_terminal','phase'])
def test_boundary_rejects_any_live_child_or_changed_original_owner(fault):
    value=state();children=[];terminal=True
    if fault=='pid':value['pid']+=1
    elif fault=='time':value['create_time']+=1
    elif fault=='scope':value['required_cases']=1
    elif fault=='labels':value['required_label_views']=1
    elif fault=='native_child':children=[dict(name='python.exe',exe='C:/Python/python.exe')]
    elif fault=='not_terminal':terminal=False
    else:value['state']='unrecognized'
    assert not s.boundary_allowed(value,s.EXPECTED,children,terminal)


def test_actual_successful_fragments_are_reused_and_failed_partial_never_retried(tmp_path,monkeypatch):
    c=s.read(s.ROOT/s.original.CONFIG)
    good=next(e for e in c['entries'] if e['key']=='e8a311b6a8a4968fb22a')
    bound=s.original.ready_case(c,good);assert bound is not None
    assert s.fragment_action(c,good,'e000',bound)=='reuse_guarded'
    assert s.fragment_action(c,good,'e001',bound)=='reuse_guarded'
    failed=next(e for e in c['entries'] if e['key']=='5eedb87987df7781d41c')
    with pytest.raises(ValueError,match='failed fragment'):s.fragment_action(c,failed,'e006',{})
    monkeypatch.setattr(s.original,'paths',lambda config,e,p:(tmp_path/'attempt',tmp_path/'output'))
    assert s.fragment_action(c,good,'e000',bound)=='new'
    (tmp_path/'attempt').mkdir()
    with pytest.raises(ValueError,match='Partial'):s.fragment_action(c,good,'e000',bound)


def test_actual_old_guard_must_match_exact_command_and_dead_child(monkeypatch):
    import psutil
    c=s.read(s.ROOT/s.original.CONFIG);entry=next(e for e in c['entries'] if e['key']=='e8a311b6a8a4968fb22a')
    attempt,_=s.original.paths(c,entry,'e000');path=attempt/'guard_receipt.json';g=s.read(path)
    value=dict(state(),active_pid=g['exact_child_handle']['pid'],active_case=entry['case']['id'],active_entity_prefix='e000')
    assert s.terminal_child(c,value)
    class StillAlive:
        def create_time(self):return g['exact_child_handle']['create_time']
    monkeypatch.setattr(psutil,'Process',lambda pid:StillAlive())
    assert not s.terminal_child(c,value)
    read=s.read
    monkeypatch.setattr(s,'read',lambda p:dict(g,exact_child_handle=dict(g['exact_child_handle'],command=['wrong'])) if p==path else read(p))
    assert not s.terminal_child(c,value)


class StopPolling(RuntimeError):pass


@pytest.mark.parametrize('race',['new_child','changed_command'])
def test_handoff_resumes_metadata_if_second_check_changes_and_never_terminates_it(tmp_path,monkeypatch,race):
    import psutil
    expected=dict(s.EXPECTED);events=[]
    class Child:
        pid=123
        def name(self):return 'python.exe'
        def exe(self):return 'C:/Python/python.exe'
    class Actor:
        paused=False
        def create_time(self):return expected['create_time']
        def cmdline(self):return ['changed'] if self.paused and race=='changed_command' else expected['command']
        def children(self,recursive=True):return [Child()] if self.paused and race=='new_child' else []
        def suspend(self):self.paused=True;events.append('suspend')
        def resume(self):self.paused=False;events.append('resume')
        def is_running(self):return True
        def terminate(self):raise AssertionError('Never stop an owner when its second check changes')
    class Slot:
        def close(self):events.append('release_audit_lock')
    actor=Actor();monkeypatch.setattr(psutil,'Process',lambda pid:actor)
    monkeypatch.setattr(s,'terminal_child',lambda c,v:True)
    monkeypatch.setattr(s,'read',lambda p:state())
    monkeypatch.setattr(s.time,'sleep',lambda n:(_ for _ in ()).throw(StopPolling()))
    c=dict(state_root='old',resource_lock='audit.lock');config=dict(previous_owner=expected,handoff_poll_seconds=1)
    api=dict(exclusive_lock=lambda p:Slot())
    with pytest.raises((StopPolling,ValueError)):s.handoff(config,c,api,tmp_path,lambda **kw:None)
    assert events==['suspend','resume','release_audit_lock']
    assert not list(tmp_path.iterdir())


def test_original_worker_cli_is_identical_and_cold_registration_loads_no_numeric_libraries():
    code='from scripts.flow_matching import continue_TAB878_fair_entities_v1 as s;import sys;c=s.registration();s.verify(c);assert "numpy" not in sys.modules and "torch" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=90)
    assert p.returncode==0,p.stdout+p.stderr
    c=s.read(s.ROOT/s.original.CONFIG);command=s.original.worker_command(c,c['entries'][0],'e000')
    assert command[command.index('-m')+1]==s.original.MODULE


def test_successful_handoff_holds_real_slot_contract_until_old_metadata_lock_transfers(tmp_path,monkeypatch):
    import psutil
    events=[];expected=deepcopy(s.EXPECTED)
    class Actor:
        def create_time(self):return expected['create_time']
        def cmdline(self):return expected['command']
        def children(self,recursive=True):return []
        def suspend(self):events.append('suspend')
        def terminate(self):events.append('terminate_metadata')
        def wait(self,timeout):events.append('wait_terminal')
    class Slot:
        def __init__(self,name):self.name=name
        def close(self):events.append('release_'+self.name)
    audit=Slot('audit');owner=Slot('old_controller')
    def lock(path):
        if path.name=='audit.lock':events.append('acquire_audit');return audit
        events.append('acquire_old_controller');return owner
    monkeypatch.setattr(s,'ROOT',tmp_path);monkeypatch.setattr(psutil,'Process',lambda pid:Actor())
    monkeypatch.setattr(s,'terminal_child',lambda config,value:True)
    old=tmp_path/'old';old.mkdir();(old/'status.json').write_text(json.dumps(state()))
    base=tmp_path/'unit_metadata_only';base.mkdir()
    previous,transferred=s.handoff(dict(previous_owner=expected,handoff_poll_seconds=1),
        dict(state_root='old',resource_lock='audit.lock'),dict(exclusive_lock=lock),base,lambda **kw:None)
    assert transferred is owner and previous==state()
    assert events==['acquire_audit','suspend','terminate_metadata','wait_terminal','acquire_old_controller','release_audit']
    assert not s.read(base/'handoff.json')['original_numeric_child_stopped']
    assert (base/'previous_status_original_bytes.json').read_bytes()==(old/'status.json').read_bytes()
