from datetime import datetime,timezone
import ast
import inspect
import pytest
from scripts.flow_matching import bridge_TAB878_queued_boundary_v1 as s


def test_full_scope_exact_same_lock_and_bounded_window_preserved():
    c=s.registration();old=s.verify(c)
    assert c['required_cases']==878 and c['required_label_views']==2057
    assert c['settings']==old['settings'] and c['resource_lock']==old['resource_lock']
    assert c['maximum_hold_seconds']==2 and c['successor_observation_seconds']==3


@pytest.mark.parametrize('fault',['scope','lock','hold','settings'])
def test_source_or_resource_or_unbounded_hold_shortcuts_rejected(fault):
    c=s.registration()
    if fault=='scope':c['required_cases']=1
    elif fault=='lock':c['resource_lock']='other.lock'
    elif fault=='hold':c['maximum_hold_seconds']=10000
    else:c['settings']['emergency_commit_headroom_bytes']=1
    with pytest.raises(ValueError):s.verify(c)


@pytest.mark.parametrize('fault',['active','pid','time','phase','lock','stale','child'])
def test_window_must_be_a_fresh_observation_of_exact_idle_original_owner(fault):
    c=s.registration();h=c['previous_owner'];now=datetime.now(timezone.utc)
    state=dict(pid=h['pid'],create_time=h['create_time'],active_pid=None,state='waiting_for_heavy_lock',shared_lock_owned=False,captured_utc=now.isoformat())
    children=[]
    assert s.window_ready(state,h,children,now.timestamp()-1)
    if fault=='active':state['active_pid']=123
    elif fault=='pid':state['pid']+=1
    elif fault=='time':state['create_time']+=1
    elif fault=='phase':state['state']='running'
    elif fault=='lock':state['shared_lock_owned']=True
    elif fault=='stale':state['captured_utc']='2020-01-01T00:00:00+00:00'
    else:children=[dict(name='python.exe',exe='C:/Python/python.exe')]
    assert not s.window_ready(state,h,children,now.timestamp()-1)


def test_bridge_never_suspends_stops_or_launches_any_numerical_process():
    tree=ast.parse(inspect.getsource(s.controller))
    assert not any(isinstance(n,ast.Attribute) and n.attr in {'suspend','terminate','kill','Popen','run'} for n in ast.walk(tree))


@pytest.mark.parametrize('interruption',[False,True])
def test_bounded_lock_is_released_on_wait_or_exception_before_handoff_observation(tmp_path,monkeypatch,interruption):
    from scripts.flow_matching import run_cfm_ts_low_memory_queue_v2 as api_module
    events=[];held=[False];clock=[0.]
    class Owner:
        def close(self):events.append('owner_closed')
    class Lock:
        def close(self):held[0]=False;events.append('audit_released')
    class Process:
        def children(self,recursive=True):return []
    monkeypatch.setattr(api_module,'resource_api',lambda:dict(exclusive_lock=lambda path:Owner()))
    monkeypatch.setattr(s,'exact_process',lambda h:Process())
    def acquire(path,update):held[0]=True;events.append('audit_acquired');return Lock()
    monkeypatch.setattr(s,'acquire',acquire)
    def monotonic():clock[0]+=.5;return clock[0]
    monkeypatch.setattr(s.time,'monotonic',monotonic)
    target=tmp_path/'target';target.mkdir();old=tmp_path/'old';old.mkdir()
    (old/'status.json').write_text('{}')
    def sleep(n):
        if n==.1:
            assert not held[0];(target/'handoff.json').write_text('{}')
    monkeypatch.setattr(s.time,'sleep',sleep)
    def ready(*args):
        if interruption:raise RuntimeError('controlled metadata-only interruption')
        return False
    monkeypatch.setattr(s,'window_ready',ready)
    c=s.registration();c.update(state_root=str(tmp_path/'bridge'),successor_state_root=str(target))
    if interruption:
        with pytest.raises(RuntimeError,match='metadata-only'):s.controller(c,dict(state_root=str(old)))
    else:
        s.controller(c,dict(state_root=str(old)))
        assert s.read(tmp_path/'bridge/status.json')['handoff_completed']
    assert events==['audit_acquired','audit_released','owner_closed']
    assert not held[0] and not list(tmp_path.rglob('guard_receipt.json'))
