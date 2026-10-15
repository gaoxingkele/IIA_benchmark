from copy import deepcopy
import json
import subprocess
import sys
from types import FunctionType
import pytest
from scripts.flow_matching import run_crossad_retries_cached_idle_v1 as s


def test_every_original139_slot_and_all_three_existing_selected_routes_preserved():
    c=s.registration()
    assert set(c['roles'])=={'gecco','psm','smap'}
    selected=[]
    for name in s.ENTRIES:
        role=s.verify(c,name);production=s.read(s.ROOT/role['original_runtime'])
        q=s.read(s.ROOT/role['original_queue']);selected.append(role['canonical_id'])
        assert len(q['jobs'])==len({j['id'] for j in q['jobs']})==139
        assert role['original_settings']==production['settings']
        assert role['original_settings']['minimum_commit_headroom_bytes']==40*2**30
        assert role['original_settings']['emergency_commit_headroom_bytes']==8*2**30
        assert role['original_shared_lock']==production['shared_lock']
        assert role['original_arguments']==['--config',role['original_runtime']]
        index=role['previous_owner']['command'].index('-m')
        assert role['previous_owner']['command'][index+1:]==[role['entry_module'],*role['original_arguments']]
        job=next(j for j in q['jobs'] if j['id']==role['canonical_id'])
        assert job['train_parameters']['train_epochs']==20 and job['train_parameters']['batch_size']==128
        assert job['train_parameters']['patience']==3
    assert len(set(selected))==3 and all(x.endswith('seed1103') for x in selected)
    assert not c['all_original_experiments_complete']


@pytest.mark.parametrize('fault',['role','scope','slots','memory','lock','entry','arguments','owner'])
def test_narrowed_scope_or_changed_policy_or_dispatch_rejected(fault):
    c=s.registration();r=c['roles']['gecco']
    if fault=='role':c['roles'].pop('psm')
    elif fault=='scope':c['original_scope']['papers']=1
    elif fault=='slots':r['required_original_slots']=1
    elif fault=='memory':r['original_settings']['minimum_commit_headroom_bytes']=1
    elif fault=='lock':r['original_shared_lock']='other.lock'
    elif fault=='entry':r['entry_module']='different'
    elif fault=='arguments':r['original_arguments'].append('--small')
    else:r['previous_owner']['create_time']+=1
    with pytest.raises(ValueError):s.verify(c,'gecco')


def state(role):
    return dict(pid=role['previous_owner']['pid'],create_time=role['previous_owner']['create_time'],
        active_pid=None,state='waiting_for_memory',shared_lock_owned=False,
        selected_original_id=role['canonical_id'],configuration_sha256=role['original_runtime_sha256'],
        original_max_epochs=20,credited_previous_epochs_for_fresh_retry=0)


@pytest.mark.parametrize('fault',['active','child','shared_lock','phase','canonical','hash','time','budget','credit','output'])
def test_any_science_activity_progress_or_output_prevents_handoff(monkeypatch,fault):
    role=s.registration()['roles']['gecco'];actual=state(role);children=[]
    monkeypatch.setattr(s,'pending_unstarted',lambda role:True)
    if fault=='active':actual['active_pid']=1
    elif fault=='child':children=[dict(name='python.exe',exe='C:/Python/python.exe')]
    elif fault=='shared_lock':actual['shared_lock_owned']=True
    elif fault=='phase':actual['state']='running_full_original_pipeline'
    elif fault=='canonical':actual['selected_original_id']='different'
    elif fault=='hash':actual['configuration_sha256']='different'
    elif fault=='time':actual['create_time']+=1
    elif fault=='budget':actual['original_max_epochs']=1
    elif fault=='credit':actual['credited_previous_epochs_for_fresh_retry']=12
    else:monkeypatch.setattr(s,'pending_unstarted',lambda role:False)
    assert not s.safe_phase(role,actual,role['previous_owner'],children)


def test_exact_unchanged_main_and_only_idle_console_is_allowed(monkeypatch):
    role=s.registration()['roles']['gecco']
    monkeypatch.setattr(s,'pending_unstarted',lambda role:True)
    assert s.safe_phase(role,state(role),role['previous_owner'],[dict(name='conhost.exe',exe='C:/Windows/System32/conhost.exe')])
    ns=s.namespace();main=FunctionType(s.original.main.__code__,ns)
    assert main.__code__ is s.original.main.__code__ and main.__globals__['handoff'] is s.handoff


@pytest.mark.parametrize('role',list(s.ENTRIES))
def test_actual_original_guard_api_uses_cached_snapshot_without_numeric_imports_or_execution(role):
    code='from scripts.flow_matching import run_crossad_retries_cached_idle_v1 as s;import json,sys;print(json.dumps(s.actual_guard_api_probe(s.registration(),sys.argv[1])))'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code,role],cwd=s.ROOT,capture_output=True,text=True,timeout=180)
    assert p.returncode==0,p.stdout+p.stderr
    proof=json.loads(p.stdout)
    assert proof['original_jobs']==139 and proof['actual_guard_snapshot']=='cached_snapshot'
    assert not proof['numpy_imported'] and not proof['torch_imported']
    assert not proof['entry_main_executed'] and not proof['training_executed']
    assert proof['private_bytes']<128*2**20
    assert proof['settings']['minimum_commit_headroom_bytes']==40*2**30
    assert proof['snapshot']['physical_available_bytes']>0 and proof['snapshot']['commit_available_bytes']>0
